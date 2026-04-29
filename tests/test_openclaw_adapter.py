# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for aevyra_witness.adapters.openclaw."""

from __future__ import annotations

import json

import pytest

from aevyra_witness.adapters.openclaw import from_openclaw_jsonl
from aevyra_witness.trace import KIND_AGENT, KIND_REASON, KIND_TOOL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _line(ev: dict) -> str:
    return json.dumps(ev)


# ---------------------------------------------------------------------------
# Classification and basic node construction
# ---------------------------------------------------------------------------


class TestClassify:
    def test_llm_event_is_reason(self):
        trace = from_openclaw_jsonl([
            _line({"type": "llm.response", "span_id": "s1", "name": "plan",
                   "input": "Q", "output": "A"})
        ])
        assert len(trace.nodes) == 1
        assert trace.nodes[0].kind == KIND_REASON

    def test_tool_event_is_tool(self):
        trace = from_openclaw_jsonl([
            _line({"type": "tool.call", "span_id": "t1", "name": "search",
                   "arguments": {"q": "foo"}, "result": ["bar"]})
        ])
        assert len(trace.nodes) == 1
        assert trace.nodes[0].kind == KIND_TOOL

    def test_mcp_call_is_tool(self):
        trace = from_openclaw_jsonl([
            _line({"type": "mcp.call", "span_id": "m1", "name": "read_file",
                   "mcp_server": "fs", "arguments": {"path": "/tmp/x"},
                   "result": "content"})
        ])
        assert len(trace.nodes) == 1
        assert trace.nodes[0].kind == KIND_TOOL

    def test_agent_start_finish_is_agent(self):
        trace = from_openclaw_jsonl([
            _line({"type": "agent.start", "span_id": "a1", "name": "orchestrator",
                   "input": "task"}),
            _line({"type": "agent.finish", "span_id": "a1", "output": "done"}),
        ])
        # paired into one span
        assert len(trace.nodes) == 1
        assert trace.nodes[0].kind == KIND_AGENT

    def test_sub_agent_is_agent(self):
        """sub_agent events must be classified as KIND_AGENT, not dropped."""
        trace = from_openclaw_jsonl([
            _line({"type": "sub_agent", "span_id": "sa1", "name": "researcher",
                   "input": "find X", "output": "found X"})
        ])
        assert len(trace.nodes) == 1
        assert trace.nodes[0].kind == KIND_AGENT
        assert trace.nodes[0].name == "researcher"

    def test_unknown_event_dropped(self):
        trace = from_openclaw_jsonl([
            _line({"type": "heartbeat", "ts": 123}),
        ])
        assert len(trace.nodes) == 0

    def test_blank_lines_skipped(self):
        trace = from_openclaw_jsonl(["", "   ", "# comment"])
        assert len(trace.nodes) == 0

    def test_invalid_json_skipped(self):
        trace = from_openclaw_jsonl(["{bad json"])
        assert len(trace.nodes) == 0


# ---------------------------------------------------------------------------
# Start/end pairing
# ---------------------------------------------------------------------------


class TestPairing:
    def test_paired_start_end_merged(self):
        trace = from_openclaw_jsonl([
            _line({"type": "llm.request", "span_id": "r1", "name": "plan",
                   "input": "Q"}),
            _line({"type": "llm.response", "span_id": "r1",
                   "output": "A", "tokens": 42}),
        ])
        assert len(trace.nodes) == 1
        n = trace.nodes[0]
        assert n.input == "Q"
        assert n.output == "A"
        assert n.tokens == 42

    def test_stray_start_kept_with_error(self):
        trace = from_openclaw_jsonl([
            _line({"type": "llm.request", "span_id": "orphan", "name": "plan",
                   "input": "Q"}),
        ])
        assert len(trace.nodes) == 1
        assert "unfinished" in (trace.nodes[0].error or "")

    def test_stray_end_kept_as_is(self):
        trace = from_openclaw_jsonl([
            _line({"type": "llm.response", "span_id": "ghost", "output": "A"}),
        ])
        assert len(trace.nodes) == 1
        assert trace.nodes[0].output == "A"


# ---------------------------------------------------------------------------
# Tool-call auto-parenting
# ---------------------------------------------------------------------------


class TestToolParenting:
    def test_tool_auto_parented_via_tool_call_id(self):
        trace = from_openclaw_jsonl([
            _line({
                "type": "llm.response", "span_id": "llm1", "name": "plan",
                "input": "Q", "output": "dispatching",
                "tool_calls": [{"id": "tc1", "name": "search", "arguments": {"q": "x"}}],
            }),
            _line({
                "type": "tool.call", "span_id": "tool1", "tool_name": "search",
                "tool_call_id": "tc1", "result": ["doc"],
            }),
        ])
        assert len(trace.nodes) == 2
        llm_id = trace.nodes[0].id
        tool_node = trace.nodes[1]
        assert tool_node.parent_id == llm_id

    def test_explicit_parent_id_takes_precedence(self):
        trace = from_openclaw_jsonl([
            _line({
                "type": "llm.response", "span_id": "llm1", "name": "plan",
                "input": "Q", "output": "dispatching",
                "tool_calls": [{"id": "tc1", "name": "search", "arguments": {}}],
            }),
            _line({
                "type": "tool.call", "span_id": "tool1", "tool_name": "search",
                "tool_call_id": "tc1", "parent_span_id": "other_parent",
                "result": "r",
            }),
        ])
        assert trace.nodes[1].parent_id == "other_parent"


# ---------------------------------------------------------------------------
# MCP tool spans
# ---------------------------------------------------------------------------


class TestMcpTool:
    def test_mcp_tool_uses_factory(self):
        trace = from_openclaw_jsonl([
            _line({
                "type": "mcp.call", "span_id": "m1", "name": "list_files",
                "mcp_server": "filesystem", "tool_call_id": "tc99",
                "arguments": {"dir": "/tmp"}, "result": ["a.txt"],
                "latency_ms": 12.5,
            }),
        ])
        n = trace.nodes[0]
        assert n.kind == KIND_TOOL
        assert n.name == "list_files"
        # mcp_server and tool_call_id should be in metadata
        assert n.metadata.get("mcp_server") == "filesystem"


# ---------------------------------------------------------------------------
# optimize / prompt_id sweep
# ---------------------------------------------------------------------------


class TestOptimize:
    def test_optimize_flag_on_event(self):
        trace = from_openclaw_jsonl([
            _line({"type": "llm.response", "span_id": "s1", "name": "answer",
                   "prompt_id": "answerer", "optimize": True,
                   "input": "Q", "output": "A"}),
        ])
        assert trace.nodes[0].optimize is True

    def test_optimize_via_arg(self):
        trace = from_openclaw_jsonl(
            [_line({"type": "llm.response", "span_id": "s1", "name": "plan",
                    "prompt_id": "planner", "input": "Q", "output": "A"})],
            optimize_prompt_ids=["planner"],
        )
        assert trace.nodes[0].optimize is True

    def test_no_optimize_by_default(self):
        trace = from_openclaw_jsonl([
            _line({"type": "llm.response", "span_id": "s1", "name": "plan",
                   "prompt_id": "planner", "input": "Q", "output": "A"}),
        ])
        assert trace.nodes[0].optimize is False


# ---------------------------------------------------------------------------
# Task Brain event families (OpenClaw ≥ 2026.3.31)
# ---------------------------------------------------------------------------


class TestTaskBrain:
    def test_task_event_is_agent(self):
        trace = from_openclaw_jsonl([
            _line({"type": "task.start", "span_id": "tk1", "task_type": "scrape",
                   "trigger": "cron", "input": "url"}),
            _line({"type": "task.end", "span_id": "tk1", "output": "scraped"}),
        ])
        assert len(trace.nodes) == 1
        assert trace.nodes[0].kind == KIND_AGENT

    def test_cron_event_is_agent(self):
        trace = from_openclaw_jsonl([
            _line({"type": "cron.run", "span_id": "cr1", "name": "nightly",
                   "input": None, "output": None}),
        ])
        assert trace.nodes[0].kind == KIND_AGENT

    def test_acp_event_is_agent(self):
        trace = from_openclaw_jsonl([
            _line({"type": "acp.dispatch", "span_id": "ac1",
                   "name": "dispatch", "input": {}, "output": None}),
        ])
        assert trace.nodes[0].kind == KIND_AGENT


# ---------------------------------------------------------------------------
# End-to-end: multi-span trace
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_rag_pipeline(self):
        lines = [
            _line({"type": "llm.response", "span_id": "qr", "name": "query_rewrite",
                   "prompt_id": "rewriter", "input": "Q", "output": "rewritten Q"}),
            _line({"type": "tool.call", "span_id": "vs", "name": "vector_search",
                   "tool_call_id": "tc1", "arguments": {"q": "rewritten Q"},
                   "result": [{"doc": "kb_returns"}]}),
            _line({"type": "llm.response", "span_id": "sy", "name": "synthesize",
                   "prompt_id": "answer_writer", "input": "docs", "output": "answer",
                   "tokens": 150}),
        ]
        trace = from_openclaw_jsonl(lines, ideal="correct answer")
        assert len(trace.nodes) == 3
        assert trace.ideal == "correct answer"
        kinds = [n.kind for n in trace.nodes]
        assert kinds == [KIND_REASON, KIND_TOOL, KIND_REASON]
        assert trace.nodes[2].tokens == 150

    def test_pre_parsed_dicts_accepted(self):
        events = [
            {"type": "llm.response", "span_id": "s1", "name": "plan",
             "input": "Q", "output": "A"},
        ]
        trace = from_openclaw_jsonl(events)
        assert len(trace.nodes) == 1
