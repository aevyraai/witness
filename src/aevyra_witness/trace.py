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

"""Agent trace types for the Aevyra stack.

Witness is the shared, dependency-free record of what happened during an
agent run. The schema targets the full complexity of modern agent
systems — N-step reasoning chains where each step dispatches M tools in
parallel — while remaining ergonomic for simple linear pipelines.

The topology is a DAG
---------------------

A trace is a flat ``list[TraceNode]`` in execution order. DAG structure
is expressed via ``parent_id``:

- A reasoning step's tool calls list that step as their parent.
- Parallel tool calls are siblings (same parent_id, no ordering between
  them other than list position).
- A nested sub-agent is a single span whose children are its own
  internal spans.
- Simple linear pipelines leave ``parent_id=None`` everywhere — every
  span is a root, rendered as a flat list.

Identity vs. execution
----------------------

One prompt fires at many call sites. A planner prompt runs once per
reasoning step; a tool's system prompt runs once per call. The trace
distinguishes:

- ``name`` — human label for display. Not unique.
- ``id`` — unique within this trace. Auto-assigned if left blank.
- ``prompt_id`` — identity of the underlying prompt. Multiple spans
  share a prompt_id when they share a prompt. Reflex optimizes at the
  prompt_id level; when the prompt is updated, every span with that id
  benefits.

This split is what makes ``optimize=True`` meaningful on repeated
reasoning steps: mark every step-N planner span, they all share
``prompt_id="planner"``, Reflex updates one prompt.

Simple usage (linear pipeline)::

    trace = AgentTrace(nodes=[
        TraceNode("classify", input=ticket,    output="billing"),
        TraceNode("retrieve", input="billing", output=policy),
        TraceNode("answer",   input=ticket,    output=reply, optimize=True),
    ])

Complex usage (N-step plan-act with M-parallel tools)::

    trace = AgentTrace(nodes=[
        TraceNode("plan", id="p1", kind=KIND_REASON, prompt_id="planner",
                  step=1, input=user_query, output=plan1, optimize=True),
        TraceNode("search_docs", id="t1a", kind=KIND_TOOL, parent_id="p1",
                  input={"query": "..."}, output=[...]),
        TraceNode("check_db", id="t1b", kind=KIND_TOOL, parent_id="p1",
                  input={"id": 42}, output={...}),
        TraceNode("get_weather", id="t1c", kind=KIND_TOOL, parent_id="p1",
                  input={"city": "Tokyo"}, output={...}),

        TraceNode("plan", id="p2", kind=KIND_REASON, prompt_id="planner",
                  step=2, input=step1_context, output=plan2, optimize=True),
        TraceNode("summarize", id="t2a", kind=KIND_TOOL, parent_id="p2",
                  input={...}, output="..."),

        TraceNode("respond", id="r", kind=KIND_REASON, prompt_id="responder",
                  step=3, input=final_context, output=final_reply),
    ])

Both ``p1`` and ``p2`` carry ``prompt_id="planner"`` and ``optimize=True``.
Reflex will optimize the single planner prompt and the update applies to
every step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# Values whose compact JSON representation fits on one line with no nested
# containers render inline; anything longer or nested renders as an
# indented block on its own line. 60 chars is the crossover point.
_INLINE_MAX_LEN = 60

# ---------------------------------------------------------------------------
# Span kinds
# ---------------------------------------------------------------------------

# Recommended values for ``TraceNode.kind``. Not enforced — adapters may
# emit custom kinds — but downstream tools (Origin, Verdict) render and
# reason about these specifically.
KIND_REASON = "reason"  # An LLM reasoning / planning step.
KIND_TOOL = "tool"  # A tool / function call (native or MCP).
KIND_RETRIEVE = "retrieve"  # A retrieval or memory lookup.
KIND_AGENT = "agent"  # A nested sub-agent invocation.
KIND_OTHER = "other"  # Anything else / unspecified.

VALID_KINDS = (KIND_REASON, KIND_TOOL, KIND_RETRIEVE, KIND_AGENT, KIND_OTHER)

# Metadata key conventions. These are strings adapters and users are
# encouraged to use so downstream tools can render them specially. All
# are optional — missing keys fall back to generic rendering.
META_MCP_SERVER = "mcp_server"  # Name of the MCP server that exposed
# this tool (e.g. "github", "slack").
# Presence signals "this is an MCP tool
# call" to downstream tools and is
# rendered prominently in to_trace_text.
META_TOOL_CALL_ID = "tool_call_id"  # The LLM's tool_use id, linking this
# tool span back to the reasoning turn
# that emitted it.
META_ERROR_CODE = "error_code"  # Machine-readable error code from a
# failed tool call. Rendered next to
# the Error: line when present.
META_LATENCY_MS = "latency_ms"  # Wall-clock duration in milliseconds,
# when started_at/ended_at aren't
# available.


@dataclass
class TraceNode:
    """One span in an agent execution trace.

    A span is one execution of something — an LLM call, a tool call, a
    retrieval, or a nested sub-agent. Many spans per trace. Spans form
    a DAG via ``parent_id``.

    Args:
        name:       Human-readable node name (e.g. "classify_ticket",
                    "search_flights"). Not required to be unique.
        input:      The span's input. Any JSON-serializable value.
        output:     The span's output. Any JSON-serializable value.
        id:         Unique identifier within this trace. If left empty,
                    ``AgentTrace`` auto-assigns ``n{index}`` at
                    construction time.
        parent_id:  ``id`` of the parent span, or ``None`` if this is a
                    root (top-level) span. Parallel siblings share a
                    parent_id.
        kind:       What kind of work this span did. Recommended values
                    are ``KIND_REASON``, ``KIND_TOOL``, ``KIND_RETRIEVE``,
                    ``KIND_AGENT``, ``KIND_OTHER``. Custom strings are
                    allowed but downstream tooling won't render them
                    specially.
        prompt_id:  Identity of the underlying prompt. Multiple spans may
                    share a ``prompt_id`` (e.g. the planner prompt fired
                    at each reasoning step). ``None`` for deterministic
                    spans like pure tools. Reflex uses this to aggregate
                    "the same prompt at all its call sites".
        step:       Optional logical step index in a plan-act loop.
                    ``None`` for simple linear traces.
        optimize:   Mark this span's prompt as the optimization target.
                    When multiple spans share a ``prompt_id``, mark all
                    of them ``True`` — Reflex updates the prompt once
                    and every span benefits.
        tokens:     LLM tokens spent in this span (prompt + completion).
                    ``0`` for non-LLM spans.
        started_at: Unix timestamp (float seconds) when the span began.
        ended_at:   Unix timestamp when the span ended.
        error:      Short error message if the span raised. ``None`` on
                    success.
        metadata:   Arbitrary key/value metadata. Adapters that import
                    traces from LangGraph, OpenAI tool use, LangSmith,
                    etc. carry tool_call_id, retry counts, latencies,
                    and other per-span context here without polluting
                    ``input`` / ``output``. Rendered in
                    ``to_trace_text()`` when non-empty.
    """

    name: str
    input: Any = None
    output: Any = None
    id: str = ""
    parent_id: str | None = None
    kind: str = KIND_OTHER
    prompt_id: str | None = None
    step: int | None = None
    optimize: bool = False
    tokens: int = 0
    started_at: float | None = None
    ended_at: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (JSON-ready).

        Always emits the full schema so consumers can rely on every key
        being present. Missing values are ``None`` / ``0`` / ``""`` /
        ``{}`` as appropriate.
        """
        return {
            "name": self.name,
            "input": self.input,
            "output": self.output,
            "id": self.id,
            "parent_id": self.parent_id,
            "kind": self.kind,
            "prompt_id": self.prompt_id,
            "step": self.step,
            "optimize": self.optimize,
            "tokens": self.tokens,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "error": self.error,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TraceNode":
        """Construct from a plain dict.

        Unknown keys are ignored. Missing optional fields fall back to
        their defaults, so dicts produced by earlier schema versions
        still load.
        """
        return cls(
            name=d["name"],
            input=d.get("input"),
            output=d.get("output"),
            id=d.get("id", ""),
            parent_id=d.get("parent_id"),
            kind=d.get("kind", KIND_OTHER),
            prompt_id=d.get("prompt_id"),
            step=d.get("step"),
            optimize=bool(d.get("optimize", False)),
            tokens=int(d.get("tokens", 0)),
            started_at=d.get("started_at"),
            ended_at=d.get("ended_at"),
            error=d.get("error"),
            metadata=dict(d.get("metadata", {})),
        )

    @classmethod
    def mcp_tool(
        cls,
        tool_name: str,
        *,
        arguments: Any = None,
        result: Any = None,
        error: str | None = None,
        error_code: str | None = None,
        server: str | None = None,
        tool_call_id: str | None = None,
        id: str = "",
        parent_id: str | None = None,
        step: int | None = None,
        started_at: float | None = None,
        ended_at: float | None = None,
        latency_ms: float | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> "TraceNode":
        """Build a ``TraceNode`` for an MCP tool call.

        MCP tool calls are the most common failure surface in modern
        agent systems — wrong arguments, malformed JSON responses,
        server errors, timeouts. This factory pins the conventions so
        downstream tools (Origin's critic, dashboards, Reflex) can
        reason about MCP failures consistently.

        The span's ``name`` is the tool name (e.g. ``"GMAIL_SEND_EMAIL"``,
        ``"GITHUB_CREATE_ISSUE"``). Its ``kind`` is ``KIND_TOOL``, and
        ``metadata`` is populated with:

        - ``mcp_server``   — which MCP server exposed the tool
        - ``tool_call_id`` — the LLM tool_use id (for correlation with
                             the reasoning turn)
        - ``error_code``   — machine-readable error code when the call
                             failed
        - ``latency_ms``   — wall-clock duration when
                             ``started_at``/``ended_at`` aren't set

        Set ``parent_id`` to the reasoning span that dispatched this
        tool call so the DAG is wired up correctly. In a plan-act loop,
        several MCP calls typically share a single reasoning parent and
        render as parallel siblings.

        Example::

            TraceNode.mcp_tool(
                "GMAIL_SEND_EMAIL",
                arguments={"to": "alice@example.com", "subject": "hi"},
                result={"message_id": "abc"},
                server="gmail",
                tool_call_id="toolu_01ABC",
                parent_id="plan_step_1",
                latency_ms=420,
            )
        """
        meta: dict[str, Any] = dict(extra_metadata or {})
        if server is not None:
            meta[META_MCP_SERVER] = server
        if tool_call_id is not None:
            meta[META_TOOL_CALL_ID] = tool_call_id
        if error_code is not None:
            meta[META_ERROR_CODE] = error_code
        if latency_ms is not None:
            meta[META_LATENCY_MS] = latency_ms

        return cls(
            name=tool_name,
            input=arguments,
            output=result,
            kind=KIND_TOOL,
            id=id,
            parent_id=parent_id,
            step=step,
            error=error,
            started_at=started_at,
            ended_at=ended_at,
            metadata=meta,
        )


@dataclass
class AgentTrace:
    """Full execution trace of one pipeline run.

    A trace is a flat list of spans in execution order. DAG structure is
    expressed via ``TraceNode.parent_id``. Roots have ``parent_id=None``.

    After construction, any span with an empty ``id`` is auto-assigned
    ``n{index}`` so every span is uniquely referenceable by downstream
    tools (attribution results cite span ids).

    The trace is consumed by:
      - **Verdict** — reads ``to_trace_text()`` as the "response" to score.
      - **Origin** — reads the trace to attribute a failure to specific
        spans (and roll those up to prompt_ids for Reflex).
      - **Reflex** — reads ``optimize_nodes`` / ``optimize_prompt_ids`` to
        know which prompt(s) to optimize.

    Args:
        nodes:    Ordered list of ``TraceNode`` spans.
        ideal:    Expected/reference output. Optional but recommended:
                  used by judges and displayed in failure reports.
        metadata: Arbitrary key/value metadata attached to the trace
                  (model name, user id, session id, ...).
        tokens:   Total tokens across the run. If left as ``0`` and any
                  span has ``tokens > 0``, the sum is filled in at
                  ``__post_init__``. Set explicitly when you want the
                  whole-pipeline total including work done outside any
                  individual span.
    """

    nodes: list[TraceNode]
    ideal: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tokens: int = 0

    def __post_init__(self) -> None:
        # Auto-assign ids to any span that lacks one.
        for i, n in enumerate(self.nodes):
            if not n.id:
                n.id = f"n{i}"
        # Fill tokens from per-span sum if not explicitly set.
        if self.tokens == 0 and self.nodes:
            s = sum(n.tokens for n in self.nodes)
            if s > 0:
                self.tokens = s

    # ---- Topology queries -----------------------------------------------

    @property
    def roots(self) -> list[TraceNode]:
        """Spans with no parent — the top level of the DAG."""
        return [n for n in self.nodes if n.parent_id is None]

    def children_of(self, node_id: str) -> list[TraceNode]:
        """Direct children of the given span, in execution order."""
        return [n for n in self.nodes if n.parent_id == node_id]

    def by_id(self, node_id: str) -> TraceNode | None:
        """Lookup a span by id. Returns ``None`` if not found."""
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def depth_of(self, node: TraceNode) -> int:
        """Distance from root. Root spans return 0."""
        d = 0
        cur = node.parent_id
        seen: set[str] = set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            d += 1
            parent = self.by_id(cur)
            if parent is None:
                break
            cur = parent.parent_id
        return d

    # ---- Optimization targets -------------------------------------------

    @property
    def optimize_nodes(self) -> list[TraceNode]:
        """All spans marked ``optimize=True``.

        When multiple spans are marked, they should share a
        ``prompt_id`` — they're the same prompt at different call
        sites. Reflex optimizes the prompt once.
        """
        return [n for n in self.nodes if n.optimize]

    @property
    def optimize_node(self) -> TraceNode | None:
        """First span marked ``optimize=True``, else the last span.

        Backward-compatible accessor for linear-pipeline callers. For
        DAG traces prefer :py:attr:`optimize_nodes` (plural) or
        :py:attr:`optimize_prompt_ids`.
        """
        for n in self.nodes:
            if n.optimize:
                return n
        return self.nodes[-1] if self.nodes else None

    @property
    def optimize_prompt_ids(self) -> list[str]:
        """Distinct ``prompt_id``s across all ``optimize=True`` spans.

        This is what Reflex ultimately optimizes: one prompt per id. In
        a well-formed trace this list has length ≤ 1.
        """
        seen: list[str] = []
        for n in self.optimize_nodes:
            if n.prompt_id and n.prompt_id not in seen:
                seen.append(n.prompt_id)
        return seen

    # ---- Rendering ------------------------------------------------------

    def to_trace_text(self) -> str:
        """Format the full trace as structured text for LLM consumption.

        Renders the DAG as an indented, hierarchically-numbered tree.
        Root spans are numbered ``Node 1``, ``Node 2``, ...; their
        children are ``Node 1.1``, ``Node 1.2``, and so on. Span ids
        are printed in parentheses alongside the kind so attribution
        output can cite them unambiguously — multiple spans may share
        the same ``name``.

        Rendering rules:

        - Strings render inline after the field label.
        - Non-string values render as compact JSON when short and flat
          (≤60 chars, no nested container), otherwise as a pretty-printed
          indented block on their own line.
        - ``step``, ``prompt_id``, ``error``, and ``metadata`` are shown
          on their own lines when present/non-empty.

        Example output (plan-act with parallel tools)::

            === AGENT TRACE ===

            Node 1 — plan (reason, id=p1)  [optimize]
              Prompt: planner
              Step: 1
              Input:  Find me a flight to Tokyo next week
              Output: call [search_flights, check_calendar]

              Node 1.1 — search_flights (tool, id=t1a)
                Input:  {"destination":"Tokyo"}
                Output:
                    [
                      {"airline": "JAL", ...},
                      ...
                    ]

              Node 1.2 — check_calendar (tool, id=t1b)
                Input:  {"dates":"next week"}
                Output: {"busy":["2026-04-22"]}

            Node 2 — plan (reason, id=p2)  [optimize]
              Prompt: planner
              Step: 2
              ...
        """
        lines = ["=== AGENT TRACE ==="]
        # Only annotate id/kind when they add signal. Simple linear traces
        # (no DAG structure, unique names, default kinds) stay clean.
        show_ids = self._ids_are_useful()
        show_kinds = any(n.kind and n.kind != KIND_OTHER for n in self.nodes)
        for i, root in enumerate(self.roots, 1):
            self._render_node(
                root,
                prefix=str(i),
                depth=0,
                lines=lines,
                show_ids=show_ids,
                show_kinds=show_kinds,
            )
        return "\n".join(lines)

    def _ids_are_useful(self) -> bool:
        """True when span ids add information worth printing in rendered text.

        Ids are useful when:
        - the trace has DAG structure (any non-None ``parent_id``), or
        - node names repeat (ambiguous without ids), or
        - any node has a user-supplied id (not the auto-assigned ``n{index}``
          pattern) — such ids carry semantic meaning and help the LLM cite
          culprits unambiguously.

        For flat linear traces with unique names and only auto-assigned ids,
        printing ``n{index}`` adds noise without helping attribution.
        """
        if any(n.parent_id is not None for n in self.nodes):
            return True
        names = [n.name for n in self.nodes]
        if len(names) != len(set(names)):
            return True
        import re

        _auto_id = re.compile(r"^n\d+$")
        return any(not _auto_id.match(n.id) for n in self.nodes)

    def _render_node(
        self,
        node: TraceNode,
        *,
        prefix: str,
        depth: int,
        lines: list[str],
        show_ids: bool,
        show_kinds: bool,
    ) -> None:
        indent = "  " * depth
        marker = "  [optimize]" if node.optimize else ""

        # Header: "Node {prefix} — {name} (kind, id=...)"
        annot_parts: list[str] = []
        if show_kinds and node.kind and node.kind != KIND_OTHER:
            annot_parts.append(node.kind)
        if show_ids and node.id:
            annot_parts.append(f"id={node.id}")
        annotation = f" ({', '.join(annot_parts)})" if annot_parts else ""
        lines.append(f"\n{indent}Node {prefix} — {node.name}{annotation}{marker}")

        if node.prompt_id:
            lines.append(f"{indent}  Prompt: {node.prompt_id}")
        if node.step is not None:
            lines.append(f"{indent}  Step: {node.step}")
        # Surface MCP server prominently — it's load-bearing for attribution
        # (a failing MCP tool often points to a server issue vs. wrong args).
        mcp_server = node.metadata.get(META_MCP_SERVER) if node.metadata else None
        if mcp_server:
            lines.append(f"{indent}  MCP server: {mcp_server}")
        lines.append(_fmt_field("Input", node.input, indent=indent))
        lines.append(_fmt_field("Output", node.output, indent=indent))
        if node.error:
            err_code = node.metadata.get(META_ERROR_CODE) if node.metadata else None
            suffix = f"  [code={err_code}]" if err_code else ""
            lines.append(f"{indent}  Error: {node.error}{suffix}")
        # Render residual metadata (anything not already surfaced above).
        residual = _residual_metadata(node.metadata)
        if residual:
            lines.append(_fmt_metadata(residual, indent=indent))

        for j, child in enumerate(self.children_of(node.id), 1):
            self._render_node(
                child,
                prefix=f"{prefix}.{j}",
                depth=depth + 1,
                lines=lines,
                show_ids=show_ids,
                show_kinds=show_kinds,
            )

    def to_dataset_record(self) -> dict[str, Any]:
        """Convert to a Verdict Dataset-compatible record.

        Useful when pre-building traces and using a traditional dataset
        flow rather than a live pipeline callback.
        """
        return {
            "messages": [{"role": "user", "content": self.to_trace_text()}],
            "ideal": self.ideal,
        }

    # ---- Serialization --------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the trace to a plain dict (JSON-ready)."""
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "ideal": self.ideal,
            "metadata": dict(self.metadata),
            "tokens": self.tokens,
        }

    def to_json(self, **kwargs: Any) -> str:
        """Serialize to a JSON string. Extra kwargs pass to ``json.dumps``."""
        kwargs.setdefault("ensure_ascii", False)
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AgentTrace":
        """Construct a trace from a plain dict (inverse of ``to_dict``).

        Unknown keys are ignored. Missing optional fields fall back to
        their defaults.
        """
        return cls(
            nodes=[TraceNode.from_dict(n) for n in d.get("nodes", [])],
            ideal=d.get("ideal"),
            metadata=dict(d.get("metadata", {})),
            tokens=int(d.get("tokens", 0)),
        )

    @classmethod
    def from_json(cls, s: str) -> "AgentTrace":
        """Construct a trace from a JSON string (inverse of ``to_json``)."""
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

# Column where "  Input: " / "  Output: " / "  Metadata: " align.
_FIELD_INDENT = "    "


def _fmt_field(label: str, value: Any, *, indent: str = "") -> str:
    """Render one ``Input:`` or ``Output:`` line, with optional tree indent.

    Short/flat values render inline on the same line as the label. Long
    or nested structured values render on the next line as an indented,
    pretty-printed JSON block.
    """
    inline, block = _fmt_value(value)
    if block is None:
        pad = " " * max(1, 8 - len(label))
        return f"{indent}  {label}:{pad}{inline}"
    return f"{indent}  {label}:\n{block}"


def _fmt_metadata(metadata: dict[str, Any], *, indent: str = "") -> str:
    """Render the ``Metadata:`` line for a node."""
    inline, block = _fmt_value(metadata)
    if block is None:
        return f"{indent}  Metadata: {inline}"
    return f"{indent}  Metadata:\n{block}"


def _fmt_value(v: Any) -> tuple[str, str | None]:
    """Return ``(inline, block)`` where exactly one side is non-None-equivalent.

    - For strings: inline is the string, block is None.
    - For JSON-serializable containers: if the compact form is short and
      flat, inline is the compact JSON and block is None; otherwise
      inline is empty and block is the indented pretty-printed JSON
      (every line prefixed by ``_FIELD_INDENT``).
    - For non-serializable values: inline is ``str(v)``, block is None.
    """
    if isinstance(v, str):
        return v, None
    try:
        compact = json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(v), None

    if len(compact) <= _INLINE_MAX_LEN and not _has_nested_container(v):
        return compact, None

    pretty = json.dumps(v, ensure_ascii=False, indent=2)
    block = "\n".join(_FIELD_INDENT + line for line in pretty.split("\n"))
    return "", block


def _residual_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Return metadata minus keys already surfaced in the render header.

    ``mcp_server`` and ``error_code`` are shown as their own lines
    (``MCP server:`` / ``Error: ... [code=...]``), so re-rendering them
    in the ``Metadata:`` block would be noise. Everything else passes
    through so adapters' custom keys (tool_call_id, retry_count,
    latency_ms, ...) stay visible to the LLM.
    """
    if not metadata:
        return {}
    skip = {META_MCP_SERVER, META_ERROR_CODE}
    return {k: v for k, v in metadata.items() if k not in skip}


def _has_nested_container(v: Any) -> bool:
    """True iff ``v`` is a container holding another container.

    Used to decide whether to render inline. Flat containers (``{"a": 1}``,
    ``[1, 2, 3]``) can render on one line; nested ones (``{"a": [1, 2]}``)
    get the block treatment.
    """
    if isinstance(v, dict):
        return any(isinstance(x, (dict, list)) for x in v.values())
    if isinstance(v, list):
        return any(isinstance(x, (dict, list)) for x in v)
    return False


__all__ = [
    "AgentTrace",
    "TraceNode",
    "KIND_REASON",
    "KIND_TOOL",
    "KIND_RETRIEVE",
    "KIND_AGENT",
    "KIND_OTHER",
    "VALID_KINDS",
    "META_MCP_SERVER",
    "META_TOOL_CALL_ID",
    "META_ERROR_CODE",
    "META_LATENCY_MS",
]
