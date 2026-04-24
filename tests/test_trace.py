# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for aevyra_witness.trace."""

from __future__ import annotations

import json

import pytest

from aevyra_witness import AgentTrace, TraceNode


# ---------------------------------------------------------------------------
# TraceNode
# ---------------------------------------------------------------------------


class TestTraceNode:
    def test_defaults(self):
        n = TraceNode("classify", input="hi", output="greeting")
        assert n.name == "classify"
        assert n.input == "hi"
        assert n.output == "greeting"
        assert n.optimize is False

    def test_optimize_flag(self):
        n = TraceNode("answer", input="q", output="a", optimize=True)
        assert n.optimize is True

    def test_to_dict(self):
        n = TraceNode("answer", input="q", output="a", optimize=True)
        assert n.to_dict() == {
            "name": "answer",
            "input": "q",
            "output": "a",
            "id": "",
            "parent_id": None,
            "kind": "other",
            "prompt_id": None,
            "step": None,
            "optimize": True,
            "tokens": 0,
            "started_at": None,
            "ended_at": None,
            "error": None,
            "metadata": {},
        }

    def test_from_dict_roundtrip(self):
        original = TraceNode("answer", input={"a": 1}, output=["b", "c"], optimize=True)
        restored = TraceNode.from_dict(original.to_dict())
        assert restored == original

    def test_from_dict_missing_optional_fields(self):
        # optimize defaults to False, input/output default to None
        n = TraceNode.from_dict({"name": "x"})
        assert n.name == "x"
        assert n.input is None
        assert n.output is None
        assert n.optimize is False

    def test_from_dict_ignores_unknown_keys(self):
        n = TraceNode.from_dict({"name": "x", "input": 1, "output": 2, "bogus": "ignored"})
        assert n.name == "x"


# ---------------------------------------------------------------------------
# AgentTrace
# ---------------------------------------------------------------------------


class TestAgentTrace:
    def _sample(self) -> AgentTrace:
        return AgentTrace(
            nodes=[
                TraceNode("classify", input="ticket text", output="billing"),
                TraceNode("retrieve", input="billing", output="policy text"),
                TraceNode("answer", input="ticket text", output="reply", optimize=True),
            ],
            ideal="expected reply",
            metadata={"source": "test"},
            tokens=123,
        )

    def test_construct(self):
        t = self._sample()
        assert len(t.nodes) == 3
        assert t.ideal == "expected reply"
        assert t.metadata == {"source": "test"}
        assert t.tokens == 123

    def test_defaults(self):
        t = AgentTrace(nodes=[TraceNode("n", input=None, output=None)])
        assert t.ideal is None
        assert t.metadata == {}
        assert t.tokens == 0

    # optimize_node --------------------------------------------------------

    def test_optimize_node_explicit(self):
        t = self._sample()
        assert t.optimize_node is not None
        assert t.optimize_node.name == "answer"

    def test_optimize_node_fallback_to_last(self):
        t = AgentTrace(
            nodes=[
                TraceNode("a", input=None, output=None),
                TraceNode("b", input=None, output=None),
            ]
        )
        assert t.optimize_node is not None
        assert t.optimize_node.name == "b"

    def test_optimize_node_empty_trace(self):
        t = AgentTrace(nodes=[])
        assert t.optimize_node is None

    # to_trace_text --------------------------------------------------------

    def test_to_trace_text_contains_all_nodes(self):
        text = self._sample().to_trace_text()
        assert "=== AGENT TRACE ===" in text
        assert "Node 1 — classify" in text
        assert "Node 2 — retrieve" in text
        assert "Node 3 — answer" in text

    def test_to_trace_text_marks_optimize_node(self):
        text = self._sample().to_trace_text()
        # Only the node with optimize=True is marked
        assert "Node 3 — answer  [optimize]" in text
        assert "Node 1 — classify  [optimize]" not in text

    def test_to_trace_text_renders_dicts_as_json(self):
        t = AgentTrace(nodes=[TraceNode("n", input={"a": 1, "b": 2}, output=None)])
        text = t.to_trace_text()
        assert '{"a":1,"b":2}' in text

    def test_to_trace_text_non_serializable_fallback(self):
        class Weird:
            def __str__(self) -> str:
                return "<weird-obj>"

        t = AgentTrace(nodes=[TraceNode("n", input=Weird(), output=None)])
        text = t.to_trace_text()
        assert "<weird-obj>" in text

    # to_dataset_record ----------------------------------------------------

    def test_to_dataset_record(self):
        rec = self._sample().to_dataset_record()
        assert rec["ideal"] == "expected reply"
        assert rec["messages"][0]["role"] == "user"
        assert "=== AGENT TRACE ===" in rec["messages"][0]["content"]

    # Dict / JSON round-trip ----------------------------------------------

    def test_to_dict_roundtrip(self):
        original = self._sample()
        d = original.to_dict()
        assert d["ideal"] == "expected reply"
        assert d["tokens"] == 123
        assert len(d["nodes"]) == 3

        restored = AgentTrace.from_dict(d)
        assert restored == original

    def test_to_json_roundtrip(self):
        original = self._sample()
        s = original.to_json()
        # Must be valid JSON
        parsed = json.loads(s)
        assert parsed["ideal"] == "expected reply"

        restored = AgentTrace.from_json(s)
        assert restored == original

    def test_from_dict_missing_optional_fields(self):
        t = AgentTrace.from_dict({"nodes": [{"name": "only"}]})
        assert len(t.nodes) == 1
        assert t.nodes[0].name == "only"
        assert t.ideal is None
        assert t.metadata == {}
        assert t.tokens == 0

    def test_from_dict_empty_nodes(self):
        t = AgentTrace.from_dict({})
        assert t.nodes == []

    def test_metadata_is_copied_not_shared(self):
        meta = {"k": "v"}
        t = AgentTrace.from_dict({"nodes": [], "metadata": meta})
        t.metadata["k"] = "mutated"
        assert meta["k"] == "v"  # original untouched


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------


def test_public_imports():
    import aevyra_witness as w

    assert hasattr(w, "AgentTrace")
    assert hasattr(w, "TraceNode")
    assert hasattr(w, "__version__")
    assert w.AgentTrace is AgentTrace
    assert w.TraceNode is TraceNode


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
