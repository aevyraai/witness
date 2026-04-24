# Witness — schema design notes

Witness is the agent-trace primitive that the rest of the Aevyra stack
reads from: Verdict scores it, Origin attributes failures against it,
Reflex optimizes prompts inside it. Its job is to capture **what
happened** with enough fidelity that every downstream tool can reason
about the run without needing to re-run the pipeline.

This document explains why the schema looks the way it does. For API
reference see the docstrings in `aevyra_witness/trace.py`.

## Scope

Witness is designed to capture the full complexity of modern agent
systems from day one:

- **N-step reasoning chains.** A planner model decides what to do,
  acts, reflects on the result, decides again, acts again — any number
  of iterations before producing a final output.
- **M-parallel tool dispatch.** A single reasoning step may call
  several tools concurrently (search, calendar, weather) and then fuse
  the results.
- **Nested sub-agents.** A span can itself be an agent whose children
  are its own internal spans. The DAG is recursive.
- **Repeated prompts.** The same prompt fires at many call sites (the
  planner at step 1, step 2, step 3, ...). Optimizing that prompt
  should affect every call site.

The schema also has to stay ergonomic for the common simple case — a
three-step classify/retrieve/answer pipeline — because most users
start there and any friction kills adoption.

## Core shape

A trace is a flat `list[TraceNode]` in execution order. DAG structure
is expressed through `parent_id` rather than nesting the list. The
flat list is the source of truth; tree views are derived.

Why flat-plus-parent-id rather than nested children?

- **Ergonomic construction.** You can append to a list as the pipeline
  runs, without hunting for the right parent to attach to.
- **Serialization-friendly.** Round-trips through JSON without
  special-casing references. LangSmith/OpenAI tool-use traces
  flatten identically.
- **Query-friendly.** `children_of`, `roots`, `depth_of` are cheap
  scans. For short traces (N < a few hundred) indexing is overkill.
- **Composable.** Adapters from other trace formats produce this
  shape without a second normalization pass.

## Identity vs. execution

A single prompt fires at many call sites. A planner prompt runs once
per reasoning step; a routing prompt runs once per sub-agent
invocation. Flattening that into a single "node" conflates two very
different things, and each downstream tool wants a different one:

- **Reflex** wants to know: which *prompt* to optimize. It updates the
  prompt once; every call site benefits.
- **Origin** wants to know: which *call site* (span) was at fault.
  A span is a concrete, inspectable piece of evidence — it has
  specific input and output.
- **Verdict** wants to read the whole execution as structured text.

So the schema separates:

| Field        | What it identifies                                    | Unique? |
| ------------ | ----------------------------------------------------- | ------- |
| `name`       | Human label, for display                              | No      |
| `id`         | This specific span within this trace                  | Yes     |
| `prompt_id`  | The prompt behind this span (many spans may share)    | No      |

`optimize=True` can appear on multiple spans — typically every span
that shares the target `prompt_id`. `AgentTrace.optimize_prompt_ids`
returns the distinct prompt ids that need updating (one prompt,
possibly fired many times).

## Kinds

Spans have a `kind` field with recommended values: `reason`, `tool`,
`retrieve`, `agent`, `other`. The string is not validated — adapters
may emit custom kinds. The downstream tools render `reason` and `tool`
specially (they're the dominant case), but they don't refuse to work on
unknown kinds.

Kind is descriptive, not semantic. It tells the LLM reading the trace
"this was a tool call, don't treat its output as a conclusion". It does
not drive any validation or behavior.

## Observability fields

`started_at`, `ended_at`, `tokens`, `error` are optional. They're
present in the schema so adapters that have the data can preserve it,
and so downstream tools that want to surface latency, cost, or
failure locations don't need a parallel data path.

`AgentTrace.tokens` auto-fills from the per-span sum when left at the
default `0`. Set it explicitly when you want to include pipeline-level
overhead tokens that aren't attributable to any single span.

## Rendering for LLM consumption

`to_trace_text()` is read by Verdict's judges and Origin's critics.
The format has to be unambiguous enough for the LLM to reason about
DAG structure, while still compact enough to fit in a context window
alongside a rubric.

The current design:

- Spans are numbered hierarchically: `Node 1`, `Node 1.1`, `Node
  1.1.1`, then `Node 2`, etc. The prefix encodes position in the DAG.
- Children are indented under their parent.
- `id=...` and kind are shown in the span header **only when they add
  signal** — a flat linear trace with unique names doesn't need them
  and keeps the simple-case output clean. The `_ids_are_useful`
  heuristic decides per-trace.
- Scalars render inline. Short flat containers (≤60 chars, no nested
  container) render as compact JSON. Everything else renders as a
  pretty-printed indented block on its own line — so the LLM isn't
  parsing a 400-character one-liner.
- `Prompt: ...`, `Step: N`, `Error: ...`, `Metadata: ...` render on
  their own lines when present.

This format is load-bearing. Origin's prompts instruct the LLM to cite
culprits by `node_id` (shown in the header) when names repeat. If the
renderer changes how ids appear, update the prompts in lockstep.

## Tool calls and MCP

Tool spans are the dominant failure surface in modern agent systems —
an agent plans well but hits the wrong tool, passes malformed
arguments, or misreads what came back. The schema treats tool calls as
a first-class concern rather than just another `kind`:

- `TraceNode.mcp_tool(...)` is a factory for tool spans that fills in
  the conventional metadata (`mcp_server`, `tool_call_id`,
  `error_code`, `latency_ms`). Callers don't need to remember key names.
- `to_trace_text()` surfaces the MCP server on its own line
  (`MCP server: gmail`) because "which server" is usually the first
  signal the critic needs when diagnosing a tool failure.
- Errors render as `Error: <message>  [code=<error_code>]` when an
  error code is present — the code is load-bearing for distinguishing
  infra failures (auth, quota, timeout) from logic failures.

The metadata key conventions are in `aevyra_witness.trace` as
`META_*` constants. They're strings, not enums, so adapters and tools
that don't know about Witness can still produce spans the schema
recognizes by convention.

## Runtime

The runtime (`aevyra_witness.runtime`) is a live tracer — a decorator
(`@span`) and a context manager (`trace`) that produce an `AgentTrace`
from an executing pipeline. The schema is the source of truth; the
runtime is one way of populating it, on equal footing with the adapters.

Design decisions:

- **`contextvars` for current tracer and current parent id.** Two
  `ContextVar`s (`_current_tracer`, `_current_parent_id`) carry the
  live state. This propagates across `await` boundaries (for async
  agents) and across threads that inherit the context. Using globals
  would fall over in either case.
- **Decorator and context manager are the same class.** `span` is a
  class with `__call__` (decorator form) and `__enter__` / `__exit__`
  (context-manager form). One object, one set of semantics. Shared
  state uses `__slots__` so metadata assignment on `s.input = ...` /
  `s.output = ...` doesn't accidentally typo into silent no-ops.
- **Input capture heuristic.** Single positional argument → pass the
  value through as-is. Multiple arguments → bind via `inspect.signature`
  and store a `{param_name: value}` dict. This produces intuitive
  traces for both shapes without forcing users to tag arguments.
- **No-op outside `trace()`.** `@span`-decorated functions run normally
  when no tracer is active. This lets libraries ship instrumentation
  that is free to callers who don't opt in. The context-manager form
  behaves the same — sets accept, nothing records.
- **Sealed tracers.** Calling `Tracer.finish()` returns the completed
  `AgentTrace` and rejects subsequent span additions. Subsequent
  `finish()` calls return the cached result. This catches ordering
  bugs where a span escapes the `trace()` block instead of silently
  recording into a finished trace.
- **IDs generated proactively.** `Tracer._next_id()` mints `n0`, `n1`,
  ... as spans open, so `parent_id` references are valid during
  execution (not just after `to_dict()` assigns them). Completion
  order is preserved in `AgentTrace.nodes`: a child that finishes
  before its parent appears before the parent in the list — the list
  is execution-ordered, which is useful for downstream tools that
  render in execution time.
- **Errors are recorded, not swallowed.** Exceptions are captured on
  the span (`error = repr(exc_val)`) and re-raised. Origin relies on
  this to attribute failures that propagated through the DAG.

The runtime intentionally does not attempt to auto-detect prompt ids,
infer kinds, or rewrite your functions. Users who want those fields
pass them to `@span(...)`. Staying explicit keeps the runtime small
and predictable; anything more opinionated belongs in a higher layer
(Origin's `diagnose_pipeline`, for example, is purely composition on
top of this).

## Adapters

External trace formats change. We absorb the change in `adapters/`
rather than in the schema. Each adapter converts one external format
into `AgentTrace`:

- `adapters/openclaw.from_openclaw_jsonl(lines)` — OpenClaw's JSONL
  telemetry (LLM turns, tool/MCP calls, agent lifecycle). Tolerant of
  version drift: unknown event types are skipped with a warning; bad
  JSON lines never abort the import. Supports both paired
  start/end events and single-event completed records. Auto-wires tool
  calls back to their reasoning parent via `tool_call_id` when the
  stream didn't carry an explicit `parent_span_id`.

Adapters are additive. Adding support for LangSmith, OpenAI tool-use,
LangGraph, or OTel spans is a new file in `adapters/` with the same
signature: `from_<format>(...) -> AgentTrace`.

## What's intentionally *not* here

- **Streaming.** A trace is a completed record, not a live event
  stream. Adapters may buffer events and emit the trace at the end.
  If live observation is needed later, it goes in a separate channel
  (think: OTel spans) and gets converted to an `AgentTrace` once
  finalized.
- **Graph edges beyond parent-child.** Some workflow engines have
  richer edge types (conditional, retry, compensation). Witness only
  models the execution parent. If a retry happened, it shows up as
  two sibling spans with identical input, not as a dedicated edge.
- **Schema validation at construction time.** Witness is
  dependency-free and tolerant — it doesn't check that `parent_id`
  references an existing span, or that `kind` is one of
  `VALID_KINDS`. Downstream tools validate the subset they care about.
- **Metrics, cost.** `tokens` is carried because every downstream
  tool asks for it. Dollar cost, per-model tokens, and latency
  percentiles live in `metadata` or in a higher-level analytics layer
  that reads many traces.

## Extension points

Adding a field should be **additive and default-preserving**: existing
traces (both Python and JSON) continue to load. The checklist:

1. Add the field to `TraceNode` or `AgentTrace` with a safe default.
2. Add it to `to_dict()` and read it in `from_dict()` with a default.
3. If it's informative for LLMs, surface it in `to_trace_text()`
   behind an "only when non-default" check so simple cases stay clean.
4. Document it in the dataclass docstring.
5. Never bump the schema version — the dict form is self-describing.

## Versioning stance

Witness is pre-1.0. The dict/JSON shape may add fields freely and is
expected to round-trip across versions as long as old dicts load (they
do — every field is optional with a default). The dataclass
constructor may gain new keyword-only arguments but positional signature
stability is not guaranteed until 1.0.
