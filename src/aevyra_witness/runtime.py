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

"""Witness runtime — turn a live pipeline into an ``AgentTrace``.

The ``trace()`` context manager installs a tracer for the duration of
its block; the ``span`` decorator / context manager emits one
``TraceNode`` per call and wires ``parent_id`` automatically via
``contextvars`` so nested functions attach to their caller.

Example::

    from aevyra_witness.runtime import trace, span

    @span(name="classify")
    def classify(text): ...

    @span(name="answer", optimize=True, prompt_id="answer_v1")
    def answer(q, docs): ...

    def my_agent(q):
        topic = classify(q)
        docs = retrieve(topic)
        return answer(q, docs)

    with trace() as t:
        result = my_agent("how do I refund?")
    agent_trace = t.finish()

Scoping rules:

* ``span`` outside a ``trace()`` scope is a no-op — the wrapped function
  runs, but no node is emitted. This means decorated library code can
  run in untraced contexts without issue.
* Exceptions propagate. The partial span is still recorded, with
  ``error`` populated from the exception ``repr``.
* Both sync and async functions are supported; the decorator detects
  coroutine functions and preserves their shape.
* ``contextvars`` handles propagation across ``await`` boundaries; no
  threadlocals.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from contextvars import ContextVar, Token
from functools import wraps
from typing import Any, Callable

from aevyra_witness.trace import (
    KIND_OTHER,
    AgentTrace,
    TraceNode,
)

__all__ = ["Tracer", "trace", "span", "current_tracer"]


# ---------------------------------------------------------------------------
# Context variables
# ---------------------------------------------------------------------------

_current_tracer: ContextVar["Tracer | None"] = ContextVar(
    "aevyra_witness_current_tracer", default=None
)
_current_parent_id: ContextVar["str | None"] = ContextVar(
    "aevyra_witness_current_parent_id", default=None
)


def current_tracer() -> "Tracer | None":
    """Return the ``Tracer`` installed by the nearest enclosing ``trace()``.

    Returns ``None`` if no tracer is active. Useful for library code
    that wants to attach metadata to the current run without assuming a
    scope.
    """
    return _current_tracer.get()


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------


class Tracer:
    """Accumulator for spans emitted during a ``trace()`` block.

    One tracer per ``trace()`` scope. Spans append themselves via
    ``_add_node`` as they close. ``finish()`` produces the final
    ``AgentTrace`` once and caches it for repeated access.
    """

    def __init__(
        self,
        *,
        ideal: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._nodes: list[TraceNode] = []
        self._ideal = ideal
        self._metadata: dict[str, Any] = dict(metadata or {})
        self._finished: AgentTrace | None = None
        self._index: int = 0  # monotonic span counter, used for auto ids

    # -- Internal span plumbing -------------------------------------------

    def _next_id(self) -> str:
        i = self._index
        self._index += 1
        return f"n{i}"

    def _add_node(self, node: TraceNode) -> None:
        if self._finished is not None:
            raise RuntimeError("cannot add a span after Tracer.finish() — trace is sealed")
        self._nodes.append(node)

    # -- Public API --------------------------------------------------------

    @property
    def nodes(self) -> list[TraceNode]:
        """Live view of the spans captured so far (mutation not supported)."""
        return list(self._nodes)

    def finish(self) -> AgentTrace:
        """Return the captured ``AgentTrace``. Idempotent."""
        if self._finished is None:
            self._finished = AgentTrace(
                nodes=list(self._nodes),
                ideal=self._ideal,
                metadata=dict(self._metadata),
            )
        return self._finished


# ---------------------------------------------------------------------------
# trace() context manager
# ---------------------------------------------------------------------------


class trace:
    """Install a ``Tracer`` for the duration of a ``with`` block.

    Args:
        ideal:    Optional reference output for the whole run; stored
                  on the resulting ``AgentTrace.ideal``.
        metadata: Optional trace-level metadata (model name, run id, ...);
                  stored on ``AgentTrace.metadata``.

    The context manager yields the underlying ``Tracer``; call
    ``.finish()`` (inside or outside the block) to materialize the
    ``AgentTrace``.

    Nested ``trace()`` blocks are allowed; the inner block installs its
    own tracer and the outer tracer is restored on exit.
    """

    def __init__(
        self,
        *,
        ideal: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._tracer = Tracer(ideal=ideal, metadata=metadata)
        self._tracer_token: Token | None = None
        self._parent_token: Token | None = None

    def __enter__(self) -> Tracer:
        self._tracer_token = _current_tracer.set(self._tracer)
        # Reset parent to None so root-level spans have parent_id=None
        # even when nested inside an outer trace block.
        self._parent_token = _current_parent_id.set(None)
        return self._tracer

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        assert self._parent_token is not None
        assert self._tracer_token is not None
        _current_parent_id.reset(self._parent_token)
        _current_tracer.reset(self._tracer_token)
        # Seal the trace — subsequent span additions would be surprising.
        self._tracer.finish()
        return False

    def finish(self) -> AgentTrace:
        """Convenience: finish and return the AgentTrace without a with block."""
        return self._tracer.finish()


# ---------------------------------------------------------------------------
# span() — decorator + context manager
# ---------------------------------------------------------------------------


class span:
    """Emit one ``TraceNode`` per invocation.

    Usage as a decorator::

        @span(name="classify", optimize=True)
        def classify(text): ...

    Usage as a context manager, when you want manual control over
    input/output or metadata::

        with span("retrieve") as s:
            s.input = query
            s.output = do_search(query)
            s.metadata["latency_ms"] = 42

    When ``span`` is used outside a ``trace()`` scope the wrapped
    function still runs but no node is emitted. This means decorated
    library code won't break its callers when tracing isn't requested.

    Args:
        name:       Span label. Appears as ``TraceNode.name``.
        optimize:   Mark this span's prompt as a Reflex optimization target.
        kind:       One of ``KIND_REASON``, ``KIND_TOOL``, ``KIND_RETRIEVE``,
                    ``KIND_AGENT``, ``KIND_OTHER``.
        prompt_id:  Prompt identity for prompt-level rollup. Multiple
                    call sites firing the same prompt should share this.
        tokens:     Static token count for this span. Most LLM call sites
                    will set this at runtime via the context-manager form
                    (``s.tokens = ...``) instead.
    """

    __slots__ = (
        "name",
        "optimize",
        "kind",
        "prompt_id",
        "_static_tokens",
        # Per-invocation state, reset in __enter__:
        "input",
        "output",
        "tokens",
        "metadata",
        "error",
        "_id",
        "_parent_id",
        "_parent_token",
        "_tracer",
        "_started_at",
        "_entered",
    )

    def __init__(
        self,
        name: str,
        *,
        optimize: bool = False,
        kind: str = KIND_OTHER,
        prompt_id: str | None = None,
        tokens: int = 0,
    ) -> None:
        self.name = name
        self.optimize = optimize
        self.kind = kind
        self.prompt_id = prompt_id
        self._static_tokens = tokens
        # Per-invocation state defaults; populated in __enter__.
        self.input: Any = None
        self.output: Any = None
        self.tokens: int = tokens
        self.metadata: dict[str, Any] = {}
        self.error: str | None = None
        self._id: str | None = None
        self._parent_id: str | None = None
        self._parent_token: Token | None = None
        self._tracer: Tracer | None = None
        self._started_at: float | None = None
        self._entered: bool = False

    # -- Context manager form ---------------------------------------------

    def __enter__(self) -> "span":
        tracer = _current_tracer.get()
        if tracer is None:
            # No-op: outside any trace() scope.
            self._entered = False
            return self
        # Fresh per-invocation state.
        self.input = None
        self.output = None
        self.tokens = self._static_tokens
        self.metadata = {}
        self.error = None
        self._tracer = tracer
        self._id = tracer._next_id()
        self._parent_id = _current_parent_id.get()
        self._parent_token = _current_parent_id.set(self._id)
        self._started_at = time.time()
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if not self._entered:
            return False
        assert self._parent_token is not None
        assert self._tracer is not None
        assert self._id is not None
        _current_parent_id.reset(self._parent_token)
        ended_at = time.time()
        if exc_type is not None and self.error is None:
            self.error = repr(exc_val)
        node = TraceNode(
            name=self.name,
            input=self.input,
            output=self.output,
            id=self._id,
            parent_id=self._parent_id,
            kind=self.kind,
            prompt_id=self.prompt_id,
            optimize=self.optimize,
            tokens=self.tokens,
            started_at=self._started_at,
            ended_at=ended_at,
            error=self.error,
            metadata=dict(self.metadata),
        )
        self._tracer._add_node(node)
        # Clear transient refs so the span object is reusable.
        self._parent_token = None
        self._tracer = None
        self._entered = False
        return False  # don't suppress

    # -- Decorator form ----------------------------------------------------

    def __call__(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        # Snapshot config — decorated functions must get a fresh span per call.
        name = self.name
        optimize = self.optimize
        kind = self.kind
        prompt_id = self.prompt_id
        static_tokens = self._static_tokens

        def _make_span() -> "span":
            return span(
                name,
                optimize=optimize,
                kind=kind,
                prompt_id=prompt_id,
                tokens=static_tokens,
            )

        if asyncio.iscoroutinefunction(fn):

            @wraps(fn)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                s = _make_span()
                with s:
                    if s._entered:
                        s.input = _capture_input(fn, args, kwargs)
                    result = await fn(*args, **kwargs)
                    if s._entered:
                        s.output = result
                    return result

            return awrapper

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            s = _make_span()
            with s:
                if s._entered:
                    s.input = _capture_input(fn, args, kwargs)
                result = fn(*args, **kwargs)
                if s._entered:
                    s.output = result
                return result

        return wrapper


# ---------------------------------------------------------------------------
# Input capture
# ---------------------------------------------------------------------------


def _capture_input(fn: Callable[..., Any], args: tuple, kwargs: dict) -> Any:
    """Render the call's (args, kwargs) as the span's ``input``.

    Heuristic:
      - Single positional arg, no kwargs → that arg (the common case
        for ``classify(text)`` / ``retrieve(query)``).
      - Otherwise → ``{name_or_pos: value}`` using the function's
        parameter names where possible; falls back to ``args``/``kwargs``
        if signature inspection fails or the callable is a builtin.

    Rendered values are stored as-is — ``AgentTrace.to_trace_text()``
    and JSON serialization handle the rest.
    """
    if len(args) == 1 and not kwargs:
        return args[0]
    try:
        sig = inspect.signature(fn)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except (TypeError, ValueError):
        return {"args": list(args), "kwargs": dict(kwargs)}
