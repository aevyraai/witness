# aevyra-witness

[![CI](https://github.com/aevyraai/witness/actions/workflows/ci.yml/badge.svg)](https://github.com/aevyraai/witness/actions/workflows/ci.yml)
[![Security](https://github.com/aevyraai/witness/actions/workflows/security.yml/badge.svg)](https://github.com/aevyraai/witness/actions/workflows/security.yml)

**The record of what happened during an agent run.**

Witness is the shared trace primitive for the [Aevyra](https://aevyra.ai) stack.
It is a tiny, dependency-free package that defines one thing well: `AgentTrace`,
the structured record of an agent pipeline's execution — every span, its input,
its output, its place in the DAG, and which prompt(s) are the optimization
target. A small **runtime** layer (`@span` + `trace()`) captures an `AgentTrace`
automatically from a live pipeline when you don't want to build one by hand.

Reflex, Origin, Verdict, and any future Aevyra tool all consume the same
`AgentTrace`. That's the whole point: one canonical shape, so the optimizer,
the diagnoser, and the viewer never disagree about what a run looked like.

```
Witness  →  captures what happened
Verdict  →  judges it
Origin   →  finds where it went wrong
Reflex   →  fixes it
```

## Install

```bash
pip install aevyra-witness
```

Zero runtime dependencies. Python 3.10+.

## Quick start (live capture)

The runtime turns any instrumented pipeline into an `AgentTrace` with no
manual wiring. Decorate your nodes with `@span`, wrap the run in
`trace()`, and ask the tracer for the finished trace:

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

# at is an AgentTrace with three spans, `answer` parented under the root
# call stack, input/output/timing captured, ready to hand to Verdict or
# Origin.
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

## Quick start (manual AgentTrace)

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

- `aevyra_witness.adapters.from_openclaw_jsonl(lines)` — import an
  [OpenClaw](https://github.com/openclaw-ai) JSONL telemetry stream
  (LLM turns, tool calls, MCP calls, agent lifecycle) into an
  `AgentTrace`. Auto-wires tool calls back to their reasoning parent
  via `tool_call_id`.

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
