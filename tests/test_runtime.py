# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for aevyra_witness.runtime — the live tracer."""

from __future__ import annotations

import asyncio

import pytest

from aevyra_witness import (
    AgentTrace,
    KIND_REASON,
    KIND_TOOL,
    Tracer,
    current_tracer,
    span,
    trace,
)


# ---------------------------------------------------------------------------
# Basic decorator form
# ---------------------------------------------------------------------------


class TestDecorator:
    def test_single_decorated_call_produces_one_span(self):
        @span("classify")
        def classify(text):
            return "billing"

        with trace() as t:
            out = classify("ticket")
        at = t.finish()

        assert out == "billing"
        assert len(at.nodes) == 1
        n = at.nodes[0]
        assert n.name == "classify"
        assert n.input == "ticket"
        assert n.output == "billing"
        assert n.parent_id is None
        assert n.started_at is not None
        assert n.ended_at is not None
        assert n.ended_at >= n.started_at
        assert n.error is None

    def test_sibling_decorated_calls_share_parent_none(self):
        @span("a")
        def a(x):
            return x + 1

        @span("b")
        def b(x):
            return x + 2

        with trace() as t:
            a(1)
            b(2)
        at = t.finish()

        assert [n.name for n in at.nodes] == ["a", "b"]
        assert all(n.parent_id is None for n in at.nodes)

    def test_nested_decorated_calls_auto_parent(self):
        @span("inner")
        def inner(x):
            return x * 2

        @span("outer")
        def outer(x):
            return inner(x) + 1

        with trace() as t:
            outer(3)
        at = t.finish()

        assert len(at.nodes) == 2
        by_name = {n.name: n for n in at.nodes}
        assert by_name["outer"].parent_id is None
        assert by_name["inner"].parent_id == by_name["outer"].id

    def test_decorator_forwards_metadata_fields(self):
        @span("answer", optimize=True, kind=KIND_REASON, prompt_id="answer_v1", tokens=42)
        def answer(q, docs):
            return "ok"

        with trace() as t:
            answer("q", ["d1"])
        n = t.finish().nodes[0]

        assert n.optimize is True
        assert n.kind == KIND_REASON
        assert n.prompt_id == "answer_v1"
        assert n.tokens == 42

    def test_decorator_preserves_function_identity(self):
        @span("named")
        def named_fn(x):
            """a docstring"""
            return x

        assert named_fn.__name__ == "named_fn"
        assert named_fn.__doc__ == "a docstring"

    def test_exception_records_span_and_propagates(self):
        @span("boom")
        def boom():
            raise ValueError("kaboom")

        with trace() as t:
            with pytest.raises(ValueError, match="kaboom"):
                boom()
        at = t.finish()

        assert len(at.nodes) == 1
        assert at.nodes[0].error is not None
        assert "kaboom" in at.nodes[0].error

    def test_exception_still_records_parent_chain(self):
        @span("inner")
        def inner():
            raise RuntimeError("inner failed")

        @span("outer")
        def outer():
            inner()

        with trace() as t:
            with pytest.raises(RuntimeError):
                outer()
        at = t.finish()

        # Both spans recorded, inner parented to outer.
        assert len(at.nodes) == 2
        by_name = {n.name: n for n in at.nodes}
        assert by_name["inner"].parent_id == by_name["outer"].id
        assert by_name["inner"].error is not None
        assert by_name["outer"].error is not None

    def test_no_op_outside_trace(self):
        @span("standalone")
        def f(x):
            return x * 10

        # Outside any trace() scope — function still runs, no crash.
        assert f(5) == 50

    def test_input_capture_single_positional_arg(self):
        @span("f")
        def f(text):
            return text

        with trace() as t:
            f("hello")
        assert t.finish().nodes[0].input == "hello"

    def test_input_capture_multiple_args_uses_param_names(self):
        @span("f")
        def f(a, b, c=3):
            return (a, b, c)

        with trace() as t:
            f(1, 2)
        n = t.finish().nodes[0]
        assert n.input == {"a": 1, "b": 2, "c": 3}

    def test_input_capture_keyword_only(self):
        @span("f")
        def f(*, name, value):
            return (name, value)

        with trace() as t:
            f(name="x", value=10)
        assert t.finish().nodes[0].input == {"name": "x", "value": 10}


# ---------------------------------------------------------------------------
# Context manager form
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_inline_span_captures_input_output_metadata(self):
        with trace() as t:
            with span("manual", kind=KIND_TOOL) as s:
                s.input = {"query": "refund"}
                s.output = ["doc1"]
                s.metadata["latency_ms"] = 17
                s.tokens = 99

        n = t.finish().nodes[0]
        assert n.name == "manual"
        assert n.kind == KIND_TOOL
        assert n.input == {"query": "refund"}
        assert n.output == ["doc1"]
        assert n.metadata == {"latency_ms": 17}
        assert n.tokens == 99

    def test_span_is_reusable_across_invocations(self):
        s = span("reused")

        with trace() as t:
            with s:
                s.input = 1
                s.output = 2
            with s:
                s.input = 10
                s.output = 20

        at = t.finish()
        assert len(at.nodes) == 2
        assert at.nodes[0].input == 1
        assert at.nodes[0].output == 2
        assert at.nodes[1].input == 10
        assert at.nodes[1].output == 20

    def test_context_manager_nesting_inside_decorator(self):
        @span("outer")
        def outer():
            with span("inline_child") as s:
                s.output = "child done"
            return "outer done"

        with trace() as t:
            outer()
        at = t.finish()

        by_name = {n.name: n for n in at.nodes}
        assert by_name["inline_child"].parent_id == by_name["outer"].id

    def test_manual_error_assignment_not_overwritten(self):
        # If user pre-sets s.error, an exception shouldn't clobber it.
        with trace() as t:
            try:
                with span("x") as s:
                    s.error = "user-recorded"
                    raise RuntimeError("boom")
            except RuntimeError:
                pass
        assert t.finish().nodes[0].error == "user-recorded"

    def test_context_manager_outside_trace_is_noop(self):
        # Should not raise and should not record anything.
        with span("orphan") as s:
            s.input = "x"
            s.output = "y"
        # No trace scope, so nothing to inspect — just verify no exception.


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


class TestAsync:
    def test_async_decorator_records_span(self):
        @span("astep")
        async def astep(x):
            await asyncio.sleep(0)
            return x + 1

        async def main():
            with trace() as t:
                result = await astep(5)
            return t.finish(), result

        at, r = asyncio.run(main())
        assert r == 6
        assert len(at.nodes) == 1
        assert at.nodes[0].name == "astep"
        assert at.nodes[0].input == 5
        assert at.nodes[0].output == 6

    def test_async_parent_propagates_across_await(self):
        @span("child")
        async def child(x):
            await asyncio.sleep(0)
            return x * 2

        @span("parent")
        async def parent(x):
            a = await child(x)
            b = await child(a)
            return b

        async def main():
            with trace() as t:
                return await parent(3), t.finish()

        async def runner():
            out, at = await main()
            return out, at

        out, at = asyncio.run(runner())
        assert out == 12
        by_name = [n.name for n in at.nodes]
        assert by_name.count("child") == 2
        parent_node = next(n for n in at.nodes if n.name == "parent")
        for n in at.nodes:
            if n.name == "child":
                assert n.parent_id == parent_node.id

    def test_async_exception_records_span(self):
        @span("aboom")
        async def aboom():
            raise ValueError("async fail")

        async def main():
            with trace() as t:
                with pytest.raises(ValueError):
                    await aboom()
            return t.finish()

        at = asyncio.run(main())
        assert at.nodes[0].error is not None
        assert "async fail" in at.nodes[0].error


# ---------------------------------------------------------------------------
# Tracer + trace() mechanics
# ---------------------------------------------------------------------------


class TestTracerMechanics:
    def test_finish_is_idempotent(self):
        with trace() as t:
            pass
        at1 = t.finish()
        at2 = t.finish()
        assert at1 is at2

    def test_ideal_and_metadata_propagate_to_trace(self):
        with trace(ideal="correct answer", metadata={"run": "abc"}) as t:
            with span("x"):
                pass
        at = t.finish()
        assert at.ideal == "correct answer"
        assert at.metadata == {"run": "abc"}

    def test_current_tracer_returns_active_tracer(self):
        assert current_tracer() is None
        with trace() as t:
            assert current_tracer() is t
        assert current_tracer() is None

    def test_nested_trace_scopes_isolated(self):
        @span("x")
        def x():
            return "x"

        with trace() as outer_t:
            x()
            with trace() as inner_t:
                x()
            x()
        outer = outer_t.finish()
        inner = inner_t.finish()

        # Outer sees 2 spans (before + after inner block)
        assert len(outer.nodes) == 2
        # Inner sees 1 span
        assert len(inner.nodes) == 1

    def test_sealed_tracer_rejects_late_additions(self):
        t = Tracer()
        t.finish()  # seal it
        with pytest.raises(RuntimeError, match="sealed"):
            t._add_node(object())  # type: ignore[arg-type]

    def test_nodes_returned_in_completion_order(self):
        # When inner finishes before outer, inner comes first in the list.
        @span("inner")
        def inner():
            return 1

        @span("outer")
        def outer():
            return inner()

        with trace() as t:
            outer()
        at = t.finish()

        # Inner closes first, outer closes second — execution order.
        assert [n.name for n in at.nodes] == ["inner", "outer"]


# ---------------------------------------------------------------------------
# Integration with AgentTrace consumers
# ---------------------------------------------------------------------------


class TestAgentTraceIntegration:
    def test_produced_trace_round_trips(self):
        @span("step", optimize=True, prompt_id="s1")
        def step(x):
            return x.upper()

        with trace(ideal="HELLO") as t:
            step("hello")
        at = t.finish()

        d = at.to_dict()
        rebuilt = AgentTrace.from_dict(d)
        assert rebuilt.ideal == "HELLO"
        assert len(rebuilt.nodes) == 1
        assert rebuilt.nodes[0].optimize is True
        assert rebuilt.nodes[0].prompt_id == "s1"

    def test_produced_trace_has_valid_topology(self):
        @span("a")
        def a():
            return b() + c()

        @span("b")
        def b():
            return 1

        @span("c")
        def c():
            return 2

        with trace() as t:
            a()
        at = t.finish()

        # Every non-root node points at an existing id.
        ids = {n.id for n in at.nodes}
        for n in at.nodes:
            if n.parent_id is not None:
                assert n.parent_id in ids
        # Exactly one root: "a"
        roots = at.roots
        assert len(roots) == 1
        assert roots[0].name == "a"
