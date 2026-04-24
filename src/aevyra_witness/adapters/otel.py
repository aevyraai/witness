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

"""OpenTelemetry span adapter — convert OTel traces into :class:`AgentTrace`.

OpenTelemetry is the de facto standard for distributed tracing. In 2025,
OTel published official *GenAI semantic conventions* (``gen_ai.*``
attributes) that every major agentic framework now emits. This adapter
converts those spans — regardless of which framework produced them —
into a Witness :class:`AgentTrace` that Origin and Reflex can consume.

Supported frameworks (any that emit OTel spans):

- **LangGraph** / LangChain (via ``opentelemetry-instrumentation-langchain``)
- **CrewAI** (built-in OTel export)
- **AutoGen** / Microsoft Agent Framework
- **Vercel AI SDK** (TypeScript, via OTLP HTTP export)
- **OpenClaw** (OTel path via ``diagnostics.otel.logs``)
- Any other framework using the `GenAI semantic conventions`_

.. _GenAI semantic conventions: https://opentelemetry.io/docs/specs/semconv/gen-ai/

Accepted input formats
----------------------

Spans can be provided as:

1. **Python SDK ``ReadableSpan`` objects** — from
   ``opentelemetry.sdk.trace``. Pass them directly; the adapter reads
   their attributes via the SDK API.

2. **Plain dicts** — the format produced by OTel's ``JSONSpanExporter``
   or any OTLP JSON export. Each dict must have at least a ``"name"``
   key; the rest are optional and will be read defensively.

Both formats can be mixed in the same list.

Usage::

    # From the Python OTel SDK:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from aevyra_witness.adapters.otel import from_otel_spans

    exporter = InMemorySpanExporter()
    # ... run your agent ...
    spans = exporter.get_finished_spans()
    trace = from_otel_spans(spans)

    # From OTLP JSON export (e.g. a TypeScript agent posting to a collector):
    import json
    spans = json.loads(Path("spans.json").read_text())
    trace = from_otel_spans(spans)

    # Then feed to Origin:
    from aevyra_origin import Origin
    from aevyra_origin.llm import anthropic_llm
    origin = Origin(llm=anthropic_llm())
    result = origin.diagnose(trace=trace, score=0.4, rubric=rubric)

GenAI attribute mapping
-----------------------

=============================================  ===============================
OTel attribute                                 TraceNode field
=============================================  ===============================
``gen_ai.system``                              ``metadata["gen_ai_system"]``
``gen_ai.request.model``                       ``metadata["model"]``
``gen_ai.usage.input_tokens``                  ``tokens`` (partial)
``gen_ai.usage.output_tokens``                 ``tokens`` (partial)
``gen_ai.usage.prompt_tokens`` (deprecated)    ``tokens`` (fallback)
``gen_ai.usage.completion_tokens`` (deprecated)``tokens`` (fallback)
``gen_ai.prompt`` (event)                      ``input``
``gen_ai.completion`` (event)                  ``output``
``gen_ai.tool.name``                           span ``name`` (KIND_TOOL)
``error.type`` / ``exception.message``         ``error``
=============================================  ===============================

Span classification:

- Spans with ``gen_ai.operation.name`` in (``"chat"``, ``"text_completion"``,
  ``"embeddings"``) → ``KIND_REASON``
- Spans whose name starts with ``"mcp."`` or has attribute
  ``mcp.server.name`` → ``KIND_TOOL`` (MCP tool call)
- Spans with ``gen_ai.tool.name`` → ``KIND_TOOL``
- All other spans → ``KIND_OTHER``
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from aevyra_witness.trace import (
    KIND_AGENT,
    KIND_OTHER,
    KIND_REASON,
    KIND_TOOL,
    META_MCP_SERVER,
    META_TOOL_CALL_ID,
    AgentTrace,
    TraceNode,
)

_log = logging.getLogger("aevyra_witness.adapters.otel")

# OTel GenAI semantic convention: operation names that indicate an LLM call.
_GENAI_LLM_OPERATIONS = frozenset(
    {"chat", "text_completion", "completions", "generate", "embeddings"}
)

# Nanoseconds → seconds conversion for OTel timestamps.
_NS_TO_S = 1e-9


def from_otel_spans(
    spans: Iterable[Any],
    *,
    optimize_prompt_ids: Iterable[str] | None = None,
    ideal: str | None = None,
    trace_metadata: dict[str, Any] | None = None,
) -> AgentTrace:
    """Convert OpenTelemetry spans into an :class:`AgentTrace`.

    Args:
        spans: Iterable of OTel spans. Accepts either Python SDK
               ``ReadableSpan`` objects or plain dicts (from OTLP JSON
               export). Spans from different trace IDs are merged into a
               single flat trace — this is intentional for multi-service
               agent pipelines where each service emits its own trace ID.
        optimize_prompt_ids: Prompt ids whose reasoning spans should be
               marked ``optimize=True`` for Reflex. Applied in addition
               to any spans that carry ``gen_ai.prompt.id`` or
               ``prompt_id`` directly.
        ideal: Expected/reference output for the run.
        trace_metadata: Extra metadata to attach to the trace.

    Returns:
        An :class:`AgentTrace` in execution order (sorted by start time).
        DAG structure is wired via OTel's ``parent_span_id``.
    """
    targets = set(optimize_prompt_ids or ())
    raw_spans: list[dict[str, Any]] = [_normalise(s) for s in spans]

    # Sort by start time so nodes appear in execution order.
    raw_spans.sort(key=lambda s: s.get("start_time_unix_nano", 0))

    nodes: list[TraceNode] = []
    for s in raw_spans:
        node = _span_to_node(s, targets)
        if node is not None:
            nodes.append(node)

    trace = AgentTrace(
        nodes=nodes,
        ideal=ideal,
        metadata=dict(trace_metadata or {}),
    )

    if targets:
        for n in trace.nodes:
            if n.prompt_id in targets:
                n.optimize = True

    return trace


# ---------------------------------------------------------------------------
# Normalisation — SDK objects → plain dicts
# ---------------------------------------------------------------------------


def _normalise(span: Any) -> dict[str, Any]:
    """Coerce an OTel span (SDK object or plain dict) into a plain dict."""
    if isinstance(span, dict):
        return span

    # Python OTel SDK ReadableSpan has these attributes.
    d: dict[str, Any] = {}

    # Identity
    ctx = getattr(span, "context", None)
    if ctx is not None:
        d["span_id"] = _hex_id(getattr(ctx, "span_id", None))
        d["trace_id"] = _hex_id(getattr(ctx, "trace_id", None))
    parent_ctx = getattr(span, "parent", None)
    if parent_ctx is not None:
        d["parent_span_id"] = _hex_id(getattr(parent_ctx, "span_id", None))

    d["name"] = getattr(span, "name", "") or ""
    d["start_time_unix_nano"] = getattr(span, "start_time", 0) or 0
    d["end_time_unix_nano"] = getattr(span, "end_time", 0) or 0

    # Attributes (SDK uses a dict-like object)
    attrs = getattr(span, "attributes", {}) or {}
    d["attributes"] = dict(attrs)

    # Status
    status = getattr(span, "status", None)
    if status is not None:
        d["status"] = {
            "status_code": str(getattr(status, "status_code", "UNSET")),
            "description": getattr(status, "description", None),
        }

    # Events (gen_ai.content.prompt / gen_ai.content.completion)
    events = getattr(span, "events", []) or []
    d["events"] = [
        {
            "name": getattr(e, "name", ""),
            "attributes": dict(getattr(e, "attributes", {}) or {}),
        }
        for e in events
    ]

    # Exceptions
    for event in events:
        if getattr(event, "name", "") == "exception":
            exc_attrs = dict(getattr(event, "attributes", {}) or {})
            d.setdefault("_exception_message", exc_attrs.get("exception.message"))

    return d


def _hex_id(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return format(value, "032x") if value > 0xFFFFFFFF else format(value, "016x")
    return str(value)


# ---------------------------------------------------------------------------
# Span → TraceNode
# ---------------------------------------------------------------------------


def _span_to_node(s: dict[str, Any], targets: set[str]) -> TraceNode | None:
    attrs = s.get("attributes") or {}
    events = s.get("events") or []

    name = s.get("name") or "span"
    span_id = s.get("span_id") or ""
    parent_span_id = s.get("parent_span_id")

    # Timestamps — OTel uses Unix nanoseconds; TraceNode uses seconds.
    start_ns = s.get("start_time_unix_nano") or 0
    end_ns = s.get("end_time_unix_nano") or 0
    started_at = start_ns * _NS_TO_S if start_ns else None
    ended_at = end_ns * _NS_TO_S if end_ns else None

    # Error — check status and exception events.
    error = _extract_error(s, attrs)

    # Tokens
    tokens = _extract_tokens(attrs)

    # Input / output from GenAI events or attributes
    input_val = _extract_input(attrs, events)
    output_val = _extract_output(attrs, events)

    # Prompt identity
    prompt_id = (
        attrs.get("gen_ai.prompt.id")
        or attrs.get("prompt_id")
        or attrs.get("llm.prompt_template.template")
        or None
    )
    if isinstance(prompt_id, str) and not prompt_id:
        prompt_id = None

    optimize = bool(attrs.get("optimize", False)) or (
        prompt_id is not None and prompt_id in targets
    )

    # Kind classification
    kind = _classify_kind(name, attrs)

    # Residual metadata — keep useful non-standard keys
    metadata = _extract_metadata(attrs)
    if kind == KIND_TOOL:
        server = (
            attrs.get("mcp.server.name")
            or attrs.get("server.address")
            or None
        )
        if server:
            metadata[META_MCP_SERVER] = server
        tcid = attrs.get("gen_ai.tool.call.id") or attrs.get("tool_call_id")
        if tcid:
            metadata[META_TOOL_CALL_ID] = str(tcid)

    return TraceNode(
        name=name,
        input=input_val,
        output=output_val,
        id=span_id,
        parent_id=parent_span_id,
        kind=kind,
        prompt_id=prompt_id,
        optimize=optimize,
        tokens=tokens,
        started_at=started_at,
        ended_at=ended_at,
        error=error,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _classify_kind(name: str, attrs: dict[str, Any]) -> str:
    lowered = name.lower()

    # MCP tool calls: name starts with "mcp." or carries mcp.server.name
    if lowered.startswith("mcp.") or attrs.get("mcp.server.name"):
        return KIND_TOOL

    # GenAI tool call: has gen_ai.tool.name
    if attrs.get("gen_ai.tool.name"):
        return KIND_TOOL

    # GenAI LLM call: operation name is a known LLM operation
    op = str(attrs.get("gen_ai.operation.name", "")).lower()
    if op in _GENAI_LLM_OPERATIONS:
        return KIND_REASON

    # GenAI system present without specific operation → likely LLM
    if attrs.get("gen_ai.system"):
        return KIND_REASON

    # Agent spans from multi-agent frameworks
    if "agent" in lowered and ("start" in lowered or "run" in lowered or "finish" in lowered):
        return KIND_AGENT

    return KIND_OTHER


# ---------------------------------------------------------------------------
# Field extraction helpers
# ---------------------------------------------------------------------------


def _extract_tokens(attrs: dict[str, Any]) -> int:
    """Sum input + output tokens from GenAI attributes."""
    # Newer convention
    inp = attrs.get("gen_ai.usage.input_tokens") or 0
    out = attrs.get("gen_ai.usage.output_tokens") or 0
    # Deprecated fallbacks
    if not inp:
        inp = attrs.get("gen_ai.usage.prompt_tokens") or 0
    if not out:
        out = attrs.get("gen_ai.usage.completion_tokens") or 0
    try:
        return int(inp) + int(out)
    except (TypeError, ValueError):
        return 0


def _extract_input(attrs: dict[str, Any], events: list[dict[str, Any]]) -> Any:
    """Extract the LLM input prompt from attributes or events."""
    # Check events first (newer convention uses gen_ai.content.prompt event)
    for event in events:
        if event.get("name") == "gen_ai.content.prompt":
            val = (event.get("attributes") or {}).get("gen_ai.prompt")
            if val is not None:
                return val

    # Older convention: attribute directly
    val = attrs.get("gen_ai.prompt") or attrs.get("llm.prompts")
    if val is not None:
        return val

    # Tool call input
    val = attrs.get("gen_ai.tool.parameters") or attrs.get("tool.parameters")
    if val is not None:
        return val

    return None


def _extract_output(attrs: dict[str, Any], events: list[dict[str, Any]]) -> Any:
    """Extract the LLM output completion from attributes or events."""
    for event in events:
        if event.get("name") == "gen_ai.content.completion":
            val = (event.get("attributes") or {}).get("gen_ai.completion")
            if val is not None:
                return val

    val = attrs.get("gen_ai.completion") or attrs.get("llm.completions")
    if val is not None:
        return val

    val = attrs.get("gen_ai.tool.result") or attrs.get("tool.result")
    if val is not None:
        return val

    return None


def _extract_error(s: dict[str, Any], attrs: dict[str, Any]) -> str | None:
    """Extract error from status description, error.type, or exception events."""
    status = s.get("status") or {}
    code = str(status.get("status_code", "")).upper()
    if code == "ERROR":
        desc = status.get("description")
        if desc:
            return str(desc)

    err_type = attrs.get("error.type")
    if err_type:
        return str(err_type)

    exc_msg = s.get("_exception_message")
    if exc_msg:
        return str(exc_msg)

    return None


# Well-known OTel keys that map to TraceNode fields — strip from metadata.
_CONSUMED_ATTRS = frozenset({
    "gen_ai.system", "gen_ai.request.model", "gen_ai.operation.name",
    "gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens",
    "gen_ai.usage.prompt_tokens", "gen_ai.usage.completion_tokens",
    "gen_ai.prompt", "gen_ai.completion", "gen_ai.tool.name",
    "gen_ai.tool.parameters", "gen_ai.tool.result", "gen_ai.tool.call.id",
    "gen_ai.prompt.id", "llm.prompts", "llm.completions",
    "llm.prompt_template.template", "mcp.server.name",
    "error.type", "tool.parameters", "tool.result", "tool_call_id",
    "prompt_id", "optimize",
})


def _extract_metadata(attrs: dict[str, Any]) -> dict[str, Any]:
    """Keep useful non-standard attributes as span metadata."""
    meta: dict[str, Any] = {}
    model = attrs.get("gen_ai.request.model") or attrs.get("llm.model_name")
    if model:
        meta["model"] = str(model)
    system = attrs.get("gen_ai.system")
    if system:
        meta["gen_ai_system"] = str(system)
    for k, v in attrs.items():
        if k not in _CONSUMED_ATTRS and v is not None:
            meta[k] = v
    return meta


__all__ = ["from_otel_spans"]
