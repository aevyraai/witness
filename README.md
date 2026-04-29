# aevyra-witness

[![CI](https://github.com/aevyraai/witness/actions/workflows/ci.yml/badge.svg)](https://github.com/aevyraai/witness/actions/workflows/ci.yml)
[![Security](https://github.com/aevyraai/witness/actions/workflows/security.yml/badge.svg)](https://github.com/aevyraai/witness/actions/workflows/security.yml)

When an AI agent gives a wrong answer, which step caused it? The LLM call that
misread the context? The tool that returned stale data? The retrieval that pulled
the wrong docs? Without a structured record of what actually ran, you're guessing.

Witness records every step of an agent pipeline — its inputs, outputs, timing,
and how the steps relate to each other — in a single structured object called
an `AgentTrace`. Add `@span` to each function you want to instrument, wrap the
run in `trace()`, and Witness does the rest. That trace is then ready to hand
to [Origin](https://github.com/aevyraai/origin) for failure attribution,
[Verdict](https://github.com/aevyraai/verdict) for scoring, or
[Reflex](https://github.com/aevyraai/reflex) for prompt optimization.

## Use cases

- **Debugging a failing agent** — see exactly which step produced the bad output, what it received as input, and what it returned, without adding print statements everywhere.
- **Attributing failures across a multi-step pipeline** — when a plan-act-respond loop fails, know whether the planner, a tool call, or the final response step was responsible.
- **Feeding evaluation and optimization tools** — pass the same trace to a judge that scores it, a diagnoser that finds the root cause, and an optimizer that fixes the prompt — all without reformatting your data.

```
Witness  →  captures what happened
Verdict  →  judges it
Origin   →  finds where it went wrong
Reflex   →  fixes it
```

Zero runtime dependencies. Works with any LLM framework. Non-Python users can emit
traces as JSON directly — see the [schema spec](schema/README.md).

## Install

```bash
pip install aevyra-witness
```

Python 3.10+.

## Quick start (manual AgentTrace)

The simplest way to build a trace is to construct it directly. Run your
pipeline, collect the inputs and outputs, and wrap them in an `AgentTrace`:

```python
from aevyra_witness import AgentTrace, TraceNode

def run_pipeline(prompt: str, ticket: str) -> AgentTrace:
    ticket_type = classify_ticket(ticket)
    policy      = retrieve_policy(ticket_type)
    response    = generate_response(ticket, ticket_type, policy, prompt)

    return AgentTrace(
        nodes=[
            TraceNode("classify_ticket",   input=ticket,      output=ticket_type),
            TraceNode("retrieve_policy",   input=ticket_type, output=policy),
            TraceNode("generate_response", input=ticket,      output=response,
                      optimize=True),
        ],
        ideal=expected_response,
    )
```

That trace is a plain Python object — no I/O, no side effects. Call
`at.to_trace_text()` to see what a judge or critic will read:

```
=== AGENT TRACE ===

Node 1 — classify_ticket
  Input:  billing dispute on invoice #4821
  Output: billing

Node 2 — retrieve_policy
  Input:  billing
  Output: Refund requests must be submitted within 30 days of the
          invoice date. Disputes after that window require manager approval.

Node 3 — generate_response  [optimize]
  Input:  billing dispute on invoice #4821
  Output: I can help with your billing dispute. Our policy requires
          disputes to be submitted within 30 days of the invoice date...
```

`[optimize]` marks the span whose prompt Reflex will rewrite if the trace
scores poorly. Pass the trace to Origin to find out *which* span caused the
failure before asking Reflex to fix anything.

## Quick start (live capture)

The runtime instruments your existing functions without changing their
signatures. Add `@span` to each node you want to track, wrap the run in
`trace()`, and call `t.finish()` for the completed `AgentTrace`:

```python
from aevyra_witness.runtime import span, trace

@span("classify")
def classify(text):
    return "billing"

@span("retrieve")
def retrieve(topic):
    return ["doc1", "doc2"]

@span("answer", optimize=True, prompt_id="answer_v1")
def answer(q, docs):
    return "your refund will post in 3–5 business days"

def my_agent(q):
    topic = classify(q)
    return answer(q, retrieve(topic))

with trace(ideal="your refund will post in 3–5 business days") as t:
    my_agent("how do I refund?")
at = t.finish()

# at is an AgentTrace with three spans, input/output/timing captured,
# ready to hand to Verdict or Origin.
```

`span` doubles as a context manager for places where a decorator doesn't
fit (tool calls, inline LLM calls):

```python
from aevyra_witness.runtime import span, KIND_TOOL

with span("gmail_search", kind=KIND_TOOL) as s:
    s.input = {"query": "from:billing"}
    s.output = gmail.search(s.input["query"])
```

Outside a `trace()` scope, `@span` is a silent no-op — your code still
runs, nothing is recorded. This lets you instrument a library without
forcing every caller to adopt the tracer.

`@span` supports async functions, propagates the current parent id across
`await` via `contextvars`, and records exceptions with `error = repr(exc)`
without swallowing them.

## Quick start (non-Python)

The trace format is plain JSON — any language can produce it without installing
this library. Write a conforming object and save it to a file; the
[Origin CLI](https://github.com/aevyraai/origin) will take it from there.

```typescript
// TypeScript / JavaScript
import { writeFileSync } from "fs";

const trace = {
  nodes: [
    { name: "classify", kind: "reason", input: userMessage, output: category },
    { name: "lookup",   kind: "tool",   input: { id },      output: result,
      metadata: { mcp_server: "stripe" } },
    { name: "answer",   kind: "reason", input: prompt,      output: reply },
  ],
  ideal: expectedAnswer,
  metadata: { session_id: sessionId },
};

writeFileSync("trace.json", JSON.stringify(trace, null, 2));
```

```bash
# Then run attribution with the Origin CLI (Python, one-time install)
pip install aevyra-origin[anthropic]
aevyra-origin diagnose trace.json --score 0.2 --rubric rubric.txt
```

For Go, Java, and full field reference see the [schema spec](schema/README.md).

## Complex usage (N-step plan-act with M-parallel tools)

Witness is designed for real agent systems from day one — a reasoning
model that dispatches several tools in parallel, reflects on the
results, and iterates. The DAG is expressed through `parent_id`; one
prompt fired at many call sites is tracked via `prompt_id`.

```python
from aevyra_witness import AgentTrace, TraceNode, KIND_REASON, KIND_TOOL

trace = AgentTrace(nodes=[
    TraceNode("plan", id="p1", kind=KIND_REASON, prompt_id="planner",
              step=1, input=user_query, output=plan1, optimize=True),

    # Three parallel tool calls, all spawned by the step-1 plan.
    TraceNode("search_flights", id="t1a", kind=KIND_TOOL, parent_id="p1",
              input={"destination": "Tokyo"}, output=[...]),
    TraceNode("check_calendar", id="t1b", kind=KIND_TOOL, parent_id="p1",
              input={"dates": "next week"}, output={...}),
    TraceNode("get_weather", id="t1c", kind=KIND_TOOL, parent_id="p1",
              input={"city": "Tokyo"}, output={...}),

    TraceNode("plan", id="p2", kind=KIND_REASON, prompt_id="planner",
              step=2, input=context, output=plan2, optimize=True),
    TraceNode("book_flight", id="t2a", kind=KIND_TOOL, parent_id="p2",
              input={...}, output={"confirmation": "JL123"}),

    TraceNode("respond", id="r", kind=KIND_REASON, prompt_id="responder",
              step=3, input=final_context, output=reply),
])
```

Both `p1` and `p2` carry `prompt_id="planner"` and `optimize=True` — they're
the same prompt fired at two steps. Reflex updates the planner prompt **once**
and both call sites benefit. `trace.optimize_prompt_ids` returns `["planner"]`.

## Integrations

Already have traces from another system? Witness ships adapters that convert
external formats into `AgentTrace` in one call — no re-instrumentation needed.

### OpenTelemetry (LangGraph, CrewAI, AutoGen, Vercel AI SDK)

Any framework that emits [OpenTelemetry](https://github.com/open-telemetry/opentelemetry-python)
spans with the [GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
works out of the box. Pass the finished spans to `from_otel_spans`:

```python
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from aevyra_witness.adapters import from_otel_spans

exporter = InMemorySpanExporter()
# ... configure your OTel TracerProvider with this exporter, run your agent ...
spans = exporter.get_finished_spans()
trace = from_otel_spans(spans)
```

Plain dicts from an OTLP JSON export are also accepted.

### OpenClaw

[OpenClaw](https://github.com/openclaw/openclaw) streams telemetry as JSONL —
one event per line. Pass the lines (strings or pre-parsed dicts) directly:

```python
from pathlib import Path
from aevyra_witness.adapters import from_openclaw_jsonl

lines = Path("run_2026_04_21.jsonl").read_text().splitlines()
trace = from_openclaw_jsonl(lines, ideal="expected output")
```

The adapter handles paired start/end events, auto-wires tool calls back to the
reasoning turn that dispatched them via `tool_call_id`, and recognises all
OpenClaw event families including Task Brain (`task.*`, `cron.*`, `acp.*`).

To mark specific prompts as Reflex optimization targets without annotating the
event stream:

```python
trace = from_openclaw_jsonl(lines, optimize_prompt_ids=["planner", "responder"])
```

### MCP sessions

The MCP interceptor wraps any `ClientSession` and records every `call_tool`
invocation as a `TraceNode` automatically — no `@span` decorators needed in
agent code:

```python
from mcp import ClientSession
from aevyra_witness.interceptors.mcp import wrap_mcp_session

async with ClientSession(read, write) as session:
    await session.initialize()
    mcp = wrap_mcp_session(session, server_name="github")

    result = await mcp.call_tool("create_issue", {"title": "Bug"})
    result = await mcp.call_tool("list_repos", {})

    trace = mcp.to_trace()  # AgentTrace with all captured spans
```

To record MCP calls alongside `@span`-instrumented functions, pass the active
tracer so the spans land in the same trace:

```python
from aevyra_witness.runtime import trace as witness_trace

with witness_trace() as t:
    mcp = wrap_mcp_session(session, server_name="slack", tracer=t)
    await mcp.call_tool("post_message", {...})

at = t.finish()  # includes both @span spans and MCP calls
```

### Bring your own format

If you control the producer (TypeScript, Go, Rust), the simplest path is to
emit a JSON file that matches the `AgentTrace` schema and run the Origin CLI
against it — see [Quick start (non-Python)](#quick-start-non-python) above.
For structured logs from Langfuse, LangSmith, or a home-grown JSONL store, the
[Origin BYO trace tutorial](https://github.com/aevyraai/origin/tree/main/examples/byo_trace)
shows a 30-line adapter pattern that works for any source format.

## What's in the box

Schema:

- `AgentTrace` / `TraceNode` — the dataclasses
- Recommended `kind` constants: `KIND_REASON`, `KIND_TOOL`, `KIND_RETRIEVE`,
  `KIND_AGENT`, `KIND_OTHER` (custom kinds allowed)

Runtime (`aevyra_witness.runtime`):

- `@span(name, ...)` — decorator that captures a function's call as a
  `TraceNode`. Forwards `optimize`, `kind`, `prompt_id`, `tokens`. Async
  and sync both work; exceptions are recorded and re-raised.
- `span(...)` — the same object, used as a context manager for inline
  blocks (tool calls, LLM calls not wrapped in a function).
- `trace(*, ideal=None, metadata=None)` — context manager that installs
  a `Tracer` via `contextvars`. `t.finish()` returns the completed
  `AgentTrace` and is idempotent.
- `current_tracer()` — access the active tracer (for writing custom
  instrumentation).
- `Tracer` — the underlying accumulator, exposed for advanced users who
  want to drive the runtime by hand.

Rendering:

- `to_trace_text()` — hierarchical indented tree for LLM consumption
  (judges and critics read this)
- `to_dataset_record()` — Verdict-compatible dataset record

Topology queries:

- `roots`, `children_of(id)`, `by_id(id)`, `depth_of(node)`

Optimization targets:

- `optimize_nodes` — every span marked `optimize=True`
- `optimize_prompt_ids` — distinct prompt ids Reflex will update
- `optimize_node` — first marked span (linear-trace convenience)

Serialization:

- `to_dict()` / `from_dict()` / `to_json()` / `from_json()`

Tool calls (including MCP):

- `TraceNode.mcp_tool(...)` — factory for MCP tool-call spans that
  normalizes `mcp_server`, `tool_call_id`, `error_code`, `latency_ms`
  metadata so downstream tools render them consistently.

Adapters:

- `from_openclaw_jsonl(lines)` — convert an
  [OpenClaw](https://github.com/openclaw/openclaw) JSONL event stream into an
  `AgentTrace`. Handles start/end pairing, auto-parents tool calls via
  `tool_call_id`, and covers Task Brain event families.
- `from_otel_spans(spans)` — convert OpenTelemetry `ReadableSpan` objects or
  OTLP JSON dicts into an `AgentTrace`. Works with LangGraph, CrewAI, AutoGen,
  Vercel AI SDK, and any framework emitting the GenAI semantic conventions.

Interceptors:

- `wrap_mcp_session(session, server_name=...)` — wrap any MCP `ClientSession`
  to record every `call_tool` invocation as a `TraceNode`, with no decorators
  needed in agent code.

No LLM calls, no HTTP, no optimizer. Just the schema.

## Why it's its own package

A trace type is a contract. If Reflex and Origin each defined their own copy,
the contract would drift — a field added here, a rename there, and suddenly
a trace that works with the optimizer doesn't work with the diagnoser. Witness
is the single source of truth that every Aevyra tool imports. Adding a new
tool (a trace viewer, an OTel importer, a failure clusterer) is as simple as
`pip install aevyra-witness` and reading the same type.

## Design notes

See [`DESIGN.md`](DESIGN.md) for the rationale behind the schema:
identity vs. execution, flat-list-plus-parent-id, render-for-LLM rules, and
what's intentionally out of scope.

## License

Apache-2.0.
