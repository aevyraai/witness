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

"""Import OpenClaw JSONL telemetry into an :class:`AgentTrace`.

`OpenClaw`_ is an open-source MCP-native agent framework. It streams
telemetry as newline-delimited JSON, one event per line: LLM requests
and responses, tool calls (native and MCP), and agent lifecycle events.
This adapter converts those events into a Witness :class:`AgentTrace`
that the rest of the Aevyra stack (Verdict, Origin, Reflex) can consume
directly.

.. _OpenClaw: https://github.com/openclaw/openclaw

Event shape assumed
-------------------

OpenClaw's JSONL emits events keyed by a ``type`` (or ``event``) field.
The adapter recognizes three families:

- **LLM turns**: ``llm.request`` / ``llm.response``, ``llm_turn``,
  ``llm_call``, or a ``reason`` / ``plan`` event. Emitted as a
  ``KIND_REASON`` span. The prompt name (if present on the event)
  becomes the span's ``prompt_id``.

- **Tool calls**: ``tool.call``, ``tool_call``, ``mcp.call``, or any
  event whose type contains ``tool``. Emitted as a ``KIND_TOOL`` span
  via :meth:`TraceNode.mcp_tool` when the event carries an
  ``mcp_server`` field, and as a plain tool span otherwise.

- **Agent lifecycle**: ``agent.start`` / ``agent.finish``,
  ``sub_agent``. Emitted as a ``KIND_AGENT`` span. Nested agents become
  their own span with children attached via ``parent_id``.

Events may be emitted in either of two forms:

1. **Single completed events** — one event per span, containing input,
   output, timestamps, and (if any) error. This is the common case
   after a run finishes and telemetry is flushed.

2. **Paired start/end events** — a ``*.start`` event opens a span and a
   matching ``*.end`` event (sharing the same ``span_id`` or ``id``)
   closes it with the output / error / end timestamp.

The adapter auto-detects which form is being used per event and merges
start/end pairs automatically.

DAG wiring
----------

OpenClaw emits either ``parent_span_id`` (OTel-style) or
``parent_id``. The adapter honors whichever is present and writes it
through to the :class:`TraceNode`. For tool calls that lack an explicit
parent but carry a ``tool_call_id`` that was seen in a prior LLM
response's ``tool_calls`` array, the adapter auto-links the tool span
to the reasoning turn that emitted the call.

Optimization targets
--------------------

A prompt the user wants Reflex to optimize can be marked in the
OpenClaw event stream with ``"optimize": true`` on the LLM event (or
globally via the :param:`optimize_prompt_ids` argument to this
adapter). Either path sets ``TraceNode.optimize = True`` on the
matching span(s). Because multiple reasoning turns may share a
``prompt_id``, all of them are marked — Reflex sees them as a single
optimization target.

Usage::

    from pathlib import Path
    from aevyra_witness.adapters import from_openclaw_jsonl

    trace = from_openclaw_jsonl(Path("run_2026_04_21.jsonl").read_text().splitlines())
    # or pass pre-parsed dicts:
    trace = from_openclaw_jsonl(events)

The adapter is intentionally tolerant. Unknown event types are skipped
with a warning (via the standard :mod:`logging` module). Missing
optional fields fall back to the schema defaults. The adapter never
raises on a malformed event; it drops it and continues, so a single bad
line in a long JSONL stream cannot poison the whole import.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable

from aevyra_witness.trace import (
    KIND_AGENT,
    KIND_REASON,
    KIND_TOOL,
    META_LATENCY_MS,
    META_TOOL_CALL_ID,
    AgentTrace,
    TraceNode,
)

_log = logging.getLogger("aevyra_witness.adapters.openclaw")

# Event-type keys OpenClaw has used across versions. Adapters should
# check these in order.
_TYPE_KEYS = ("type", "event", "event_type", "name")

# Event-type fragments that classify the event into a kind. Substring
# matching (case-insensitive) — this is more tolerant than exact
# matching and survives OpenClaw's event-name churn.
_LLM_FRAGMENTS = ("llm", "reason", "plan", "respond")
_TOOL_FRAGMENTS = ("tool", "mcp.call", "mcp_call")
_AGENT_FRAGMENTS = ("agent.", "sub_agent")  # "agent.start", "agent.finish",
# "sub_agent"; the dot in "agent." avoids matching a user-named "research_agent"
# tool, and "sub_agent" is listed explicitly for the same reason

# Task Brain (OpenClaw ≥ 2026.3.31) unified ACP, subagents, cron tasks,
# and background CLI processes onto a SQLite-backed task ledger. The new
# event families are treated as agent-level spans so Origin can attribute
# failures to a specific task or cron job rather than losing them.
_TASK_FRAGMENTS = ("task.",)  # "task.start", "task.end", "task.error",
# "task.scheduled", "task.timeout"
_CRON_FRAGMENTS = ("cron.",)  # "cron.run", "cron.finish", "cron.error"
_ACP_FRAGMENTS = ("acp.",)  # "acp.dispatch", "acp.result" — Agent Control Protocol

# Event-type suffixes for start/end pairing.
_START_SUFFIXES = (".start", "_start", ".begin", "_begin", ".request", "_request")
_END_SUFFIXES = (".end", "_end", ".finish", "_finish", ".response", "_response")


def from_openclaw_jsonl(
    lines: Iterable[str | dict[str, Any]],
    *,
    optimize_prompt_ids: Iterable[str] | None = None,
    ideal: str | None = None,
    trace_metadata: dict[str, Any] | None = None,
) -> AgentTrace:
    """Convert an OpenClaw JSONL event stream into an :class:`AgentTrace`.

    Args:
        lines: Iterable of JSONL strings (one event per line) or
               pre-parsed event dicts. Blank lines and comments
               (lines starting with ``#``) are skipped. Invalid JSON
               lines are logged and dropped — a bad line does not
               abort the import.
        optimize_prompt_ids: Prompt ids whose reasoning spans should be
               marked ``optimize=True``. Useful when the OpenClaw run
               didn't annotate optimization targets in the event
               stream. Applied in addition to any spans that carry
               ``"optimize": true`` directly in the event.
        ideal: Expected/reference output for the run, passed through
               to ``AgentTrace.ideal``. Optional but recommended for
               judges.
        trace_metadata: Extra metadata to attach to the trace (model
               name, session id, etc.).

    Returns:
        An :class:`AgentTrace` whose nodes reflect the OpenClaw run in
        execution order. Per-span tokens, timestamps, errors, MCP
        server, and ``tool_call_id`` are preserved.

    The returned trace's topology uses :meth:`AgentTrace.__post_init__`
    id auto-assignment for any span the OpenClaw stream left without
    an explicit id — so every span has a stable id suitable for
    Origin's attribution output.
    """
    targets = set(optimize_prompt_ids or ())

    # Parse events up front so we can walk the stream twice (pairing,
    # then building). JSONL is small relative to an agent run's context;
    # no need to stream.
    events: list[dict[str, Any]] = []
    for raw in lines:
        ev = _parse_line(raw)
        if ev is not None:
            events.append(ev)

    # Merge paired start/end events into single completed events.
    completed = _pair_events(events)

    # Tool-call id → reasoning-span id, populated as we walk LLM events
    # that declare their pending tool calls. Used to auto-wire tool
    # spans back to the reasoning turn that emitted them when the
    # OpenClaw event stream didn't carry parent_span_id.
    tool_call_parent: dict[str, str] = {}

    nodes: list[TraceNode] = []
    for ev in completed:
        kind = _classify(ev)
        if kind is None:
            etype = _type_of(ev) or "<no-type>"
            _log.debug("openclaw adapter: skipping unknown event type %r", etype)
            continue

        if kind == KIND_REASON:
            node = _build_reason_node(ev, targets)
            nodes.append(node)
            # Record any tool_call_ids this turn dispatched so downstream
            # tool spans can be auto-parented to this reasoning span.
            for tcid in _pending_tool_call_ids(ev):
                tool_call_parent[tcid] = node.id or f"n{len(nodes) - 1}"

        elif kind == KIND_TOOL:
            node = _build_tool_node(ev, tool_call_parent)
            nodes.append(node)

        elif kind == KIND_AGENT:
            node = _build_agent_node(ev)
            nodes.append(node)

    trace = AgentTrace(
        nodes=nodes,
        ideal=ideal,
        metadata=dict(trace_metadata or {}),
    )

    # After ids are auto-assigned, apply target-prompt-id sweep so
    # late-added targets find their spans.
    if targets:
        for n in trace.nodes:
            if n.prompt_id in targets:
                n.optimize = True

    return trace


# ---------------------------------------------------------------------------
# Event parsing and pairing
# ---------------------------------------------------------------------------


def _parse_line(raw: str | dict[str, Any]) -> dict[str, Any] | None:
    """Coerce a JSONL line (or dict) into an event dict.

    Returns ``None`` for blank lines, comment lines, and parse
    failures. Parse failures are logged but never raise.
    """
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        _log.debug("openclaw adapter: non-str, non-dict entry %r skipped", type(raw))
        return None
    s = raw.strip()
    if not s or s.startswith("#"):
        return None
    try:
        ev = json.loads(s)
    except json.JSONDecodeError as e:
        _log.warning("openclaw adapter: failed to parse JSONL line: %s", e)
        return None
    if not isinstance(ev, dict):
        _log.warning("openclaw adapter: JSONL entry is not an object: %r", ev)
        return None
    return ev


def _type_of(ev: dict[str, Any]) -> str | None:
    for k in _TYPE_KEYS:
        v = ev.get(k)
        if isinstance(v, str) and v:
            return v
    return None


def _phase_of(ev: dict[str, Any]) -> str | None:
    """Return ``"start"``, ``"end"``, or ``None`` for a single-shot event.

    Checked first from an explicit ``phase`` field, then inferred from
    the event-type suffix.
    """
    phase = ev.get("phase")
    if isinstance(phase, str):
        p = phase.lower()
        if p in ("start", "begin", "request"):
            return "start"
        if p in ("end", "finish", "response", "complete", "completed"):
            return "end"

    etype = _type_of(ev) or ""
    lowered = etype.lower()
    for suffix in _START_SUFFIXES:
        if lowered.endswith(suffix):
            return "start"
    for suffix in _END_SUFFIXES:
        if lowered.endswith(suffix):
            return "end"
    return None


def _pair_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge ``*.start`` / ``*.end`` pairs into single completed events.

    Pairing is by ``span_id`` (preferred) or ``id``. A stray ``end``
    event with no matching start is kept as-is (treated as a
    single-shot). A stray ``start`` with no matching end is kept but
    flagged with ``error="<span unfinished — no end event>"`` so
    Origin can reason about truncated runs.
    """
    open_starts: dict[str, dict[str, Any]] = {}
    out: list[dict[str, Any]] = []

    for ev in events:
        phase = _phase_of(ev)
        span_id = ev.get("span_id") or ev.get("id")

        if phase == "start" and isinstance(span_id, str) and span_id:
            open_starts[span_id] = ev
            continue

        if phase == "end" and isinstance(span_id, str) and span_id in open_starts:
            merged = _merge_start_end(open_starts.pop(span_id), ev)
            out.append(merged)
            continue

        out.append(ev)

    # Any leftover starts had no matching end — surface as unfinished
    # spans so downstream tools don't silently lose them.
    for span_id, ev in open_starts.items():
        ev = dict(ev)
        ev.setdefault("error", "span unfinished — no end event")
        out.append(ev)

    return out


def _merge_start_end(start: dict[str, Any], end: dict[str, Any]) -> dict[str, Any]:
    """Combine a start event's metadata with the end event's outcome.

    The start event usually carries input, prompt name, and start
    timestamp; the end event carries output, error, tokens, and end
    timestamp. We prefer end-event values for overlapping keys (the
    end event is the more recent, authoritative record) but fall back
    to the start event for anything end doesn't have.
    """
    merged: dict[str, Any] = {}
    merged.update(start)
    for k, v in end.items():
        if v is None:
            continue
        # Don't let the end event's type ("llm.response") overwrite the
        # start event's type ("llm.request") — the phase suffix would
        # confuse _classify downstream. Drop phase; rely on merged fields.
        if k in ("phase",):
            continue
        if k in _TYPE_KEYS:
            # Keep the start event's type; it's the canonical record.
            continue
        merged[k] = v
    # Drop the phase so downstream consumers treat this as a completed span.
    merged.pop("phase", None)
    return merged


# ---------------------------------------------------------------------------
# Event → TraceNode
# ---------------------------------------------------------------------------


def _classify(ev: dict[str, Any]) -> str | None:
    """Return the :mod:`aevyra_witness.trace` kind constant for the event.

    ``None`` means the event doesn't map to a span (lifecycle noise,
    unknown type, etc.) and should be skipped.
    """
    etype = _type_of(ev)
    if not etype:
        return None
    lowered = etype.lower()
    # Order matters: tool before llm so "llm.tool_call.end" (hypothetical)
    # classifies as a tool, not a reason.
    if any(frag in lowered for frag in _TOOL_FRAGMENTS):
        return KIND_TOOL
    if any(frag in lowered for frag in _LLM_FRAGMENTS):
        return KIND_REASON
    # Task Brain families (OpenClaw ≥ 2026.3.31) — tasks, cron jobs, and
    # ACP dispatch events are all agent-level spans.
    if any(frag in lowered for frag in _AGENT_FRAGMENTS):
        return KIND_AGENT
    if any(frag in lowered for frag in _TASK_FRAGMENTS):
        return KIND_AGENT
    if any(frag in lowered for frag in _CRON_FRAGMENTS):
        return KIND_AGENT
    if any(frag in lowered for frag in _ACP_FRAGMENTS):
        return KIND_AGENT
    return None


def _build_reason_node(ev: dict[str, Any], targets: set[str]) -> TraceNode:
    prompt_id = _first_str(ev, "prompt_id", "prompt", "prompt_name")
    optimize = bool(ev.get("optimize", False)) or (prompt_id is not None and prompt_id in targets)
    return TraceNode(
        name=_first_str(ev, "name", "label") or prompt_id or "reason",
        input=ev.get("input") or ev.get("prompt_input") or ev.get("messages"),
        output=(
            ev.get("output")
            or ev.get("response")
            or ev.get("completion")
            or _tool_calls_summary(ev)
        ),
        id=_first_str(ev, "span_id", "id") or "",
        parent_id=_first_str(ev, "parent_span_id", "parent_id"),
        kind=KIND_REASON,
        prompt_id=prompt_id,
        step=_first_int(ev, "step", "step_index"),
        optimize=optimize,
        tokens=_first_int(ev, "tokens", "total_tokens") or 0,
        started_at=_first_float(ev, "started_at", "start_time", "timestamp_start"),
        ended_at=_first_float(ev, "ended_at", "end_time", "timestamp_end"),
        error=_first_str(ev, "error", "error_message"),
        metadata=_residual_event_metadata(ev),
    )


def _build_tool_node(ev: dict[str, Any], tool_call_parent: dict[str, str]) -> TraceNode:
    tool_call_id = _first_str(ev, "tool_call_id", "tool_use_id", "call_id")
    explicit_parent = _first_str(ev, "parent_span_id", "parent_id")
    # Auto-wire to the reasoning turn that emitted this tool call if the
    # OpenClaw stream didn't carry an explicit parent.
    parent_id = explicit_parent
    if parent_id is None and tool_call_id is not None:
        parent_id = tool_call_parent.get(tool_call_id)

    started_at = _first_float(ev, "started_at", "start_time", "timestamp_start")
    ended_at = _first_float(ev, "ended_at", "end_time", "timestamp_end")
    latency_ms = _first_float(ev, "latency_ms", "duration_ms")
    if latency_ms is None and started_at is not None and ended_at is not None:
        latency_ms = (ended_at - started_at) * 1000.0

    server = _first_str(ev, "mcp_server", "server", "server_name")
    tool_name = _first_str(ev, "tool_name", "name", "tool") or "tool"

    # mcp_tool() is the preferred factory when an MCP server is
    # present — it normalizes the metadata keys. For non-MCP tools we
    # build the TraceNode directly but still surface tool_call_id /
    # latency_ms in metadata.
    if server is not None:
        return TraceNode.mcp_tool(
            tool_name,
            arguments=_first_any(ev, "arguments", "input", "args"),
            result=_first_any(ev, "result", "output", "response"),
            error=_first_str(ev, "error", "error_message"),
            error_code=_first_str(ev, "error_code", "code"),
            server=server,
            tool_call_id=tool_call_id,
            id=_first_str(ev, "span_id", "id") or "",
            parent_id=parent_id,
            step=_first_int(ev, "step", "step_index"),
            started_at=started_at,
            ended_at=ended_at,
            latency_ms=latency_ms,
            extra_metadata=_residual_event_metadata(ev),
        )

    meta = _residual_event_metadata(ev)
    if tool_call_id is not None:
        meta[META_TOOL_CALL_ID] = tool_call_id
    if latency_ms is not None:
        meta[META_LATENCY_MS] = latency_ms
    return TraceNode(
        name=tool_name,
        input=_first_any(ev, "arguments", "input", "args"),
        output=_first_any(ev, "result", "output", "response"),
        id=_first_str(ev, "span_id", "id") or "",
        parent_id=parent_id,
        kind=KIND_TOOL,
        step=_first_int(ev, "step", "step_index"),
        started_at=started_at,
        ended_at=ended_at,
        error=_first_str(ev, "error", "error_message"),
        metadata=meta,
    )


def _build_agent_node(ev: dict[str, Any]) -> TraceNode:
    # Task Brain (≥ 2026.3.31) uses "task_id", "task_type", "cron_expr",
    # and "trigger" fields not present in earlier agent events. We surface
    # them through the residual metadata so Origin can reason about them.
    return TraceNode(
        name=_first_str(ev, "name", "agent_name", "task_type", "label") or "agent",
        input=ev.get("input") or ev.get("payload") or ev.get("trigger"),
        output=ev.get("output") or ev.get("result") or ev.get("return_value"),
        id=_first_str(ev, "span_id", "task_id", "id") or "",
        parent_id=_first_str(ev, "parent_span_id", "parent_task_id", "parent_id"),
        kind=KIND_AGENT,
        step=_first_int(ev, "step", "step_index"),
        tokens=_first_int(ev, "tokens", "total_tokens") or 0,
        started_at=_first_float(ev, "started_at", "start_time", "timestamp_start"),
        ended_at=_first_float(ev, "ended_at", "end_time", "timestamp_end"),
        error=_first_str(ev, "error", "error_message"),
        metadata=_residual_event_metadata(ev),
    )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _first_str(ev: dict[str, Any], *keys: str) -> str | None:
    for k in keys:
        v = ev.get(k)
        if isinstance(v, str) and v:
            return v
    return None


def _first_int(ev: dict[str, Any], *keys: str) -> int | None:
    for k in keys:
        v = ev.get(k)
        if isinstance(v, bool):
            continue  # bool is an int subclass; exclude explicitly
        if isinstance(v, int):
            return v
        if isinstance(v, float) and v.is_integer():
            return int(v)
    return None


def _first_float(ev: dict[str, Any], *keys: str) -> float | None:
    for k in keys:
        v = ev.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _first_any(ev: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in ev and ev[k] is not None:
            return ev[k]
    return None


def _tool_calls_summary(ev: dict[str, Any]) -> Any:
    """Summarize a reasoning turn's dispatched tool calls as its output.

    If the event carries a ``tool_calls`` list (OpenAI-style
    ``[{"id": ..., "name": ..., "arguments": ...}, ...]``), this
    returns a compact summary suitable for display inside the
    rendered trace — so a reasoning turn whose job was to fan out
    three tools doesn't appear to have no output.
    """
    calls = ev.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return None
    summary: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        summary.append(
            {
                "tool_call_id": call.get("id") or call.get("tool_call_id"),
                "tool": call.get("name") or call.get("tool") or call.get("tool_name"),
                "arguments": call.get("arguments") or call.get("args"),
            }
        )
    return summary or None


def _pending_tool_call_ids(ev: dict[str, Any]) -> list[str]:
    """Extract tool_call_ids a reasoning turn declared it would invoke.

    Used to auto-wire tool-call spans back to their reasoning parent
    when the OpenClaw stream doesn't carry an explicit
    ``parent_span_id`` on the tool event.
    """
    ids: list[str] = []
    calls = ev.get("tool_calls")
    if isinstance(calls, list):
        for call in calls:
            if not isinstance(call, dict):
                continue
            tcid = call.get("id") or call.get("tool_call_id")
            if isinstance(tcid, str) and tcid:
                ids.append(tcid)
    return ids


# Keys already mapped onto TraceNode fields — we strip these out of the
# residual metadata dict so adapters don't double-encode them.
_CONSUMED_KEYS = frozenset(
    {
        # Identity / topology
        "id",
        "span_id",
        "parent_id",
        "parent_span_id",
        # Type / phase
        "type",
        "event",
        "event_type",
        "phase",
        # Content
        "input",
        "output",
        "prompt_input",
        "messages",
        "response",
        "completion",
        "tool_calls",
        "result",
        "arguments",
        "args",
        # Prompt identity
        "prompt_id",
        "prompt",
        "prompt_name",
        # Execution
        "step",
        "step_index",
        "tokens",
        "total_tokens",
        "started_at",
        "ended_at",
        "start_time",
        "end_time",
        "timestamp_start",
        "timestamp_end",
        "error",
        "error_message",
        "error_code",
        "code",
        # Tool / MCP
        "tool_name",
        "tool",
        "name",
        "label",
        "mcp_server",
        "server",
        "server_name",
        "tool_call_id",
        "tool_use_id",
        "call_id",
        "latency_ms",
        "duration_ms",
        # Control flags
        "optimize",
        # Task Brain (OpenClaw ≥ 2026.3.31)
        "task_id",
        "task_type",
        "parent_task_id",
        "trigger",
        "payload",
        "return_value",
        "cron_expr",
        "scheduled_at",
        "attempt",
        "max_attempts",
    }
)


def _residual_event_metadata(ev: dict[str, Any]) -> dict[str, Any]:
    """Return event keys not already mapped onto a :class:`TraceNode` field.

    Keeps custom OpenClaw keys (retry counts, model names, session
    ids, cost, ...) visible on the span without double-encoding the
    canonical fields.
    """
    return {k: v for k, v in ev.items() if k not in _CONSUMED_KEYS}


__all__ = ["from_openclaw_jsonl"]
