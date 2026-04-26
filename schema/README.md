# AgentTrace JSON Schema

**Schema ID:** `https://aevyra.ai/schemas/agent-trace/v1`  
**Spec file:** [`agent-trace.schema.json`](./agent-trace.schema.json)  
**JSON Schema draft:** 2020-12

This is the language-agnostic spec for an `AgentTrace`. Any language can produce a conforming JSON object and feed it directly to the [Origin CLI](https://github.com/aevyra-ai/origin) for failure attribution — no Python dependency required.

---

## Minimal valid trace

```json
{
  "nodes": [
    { "name": "classify", "kind": "reason", "input": "user message", "output": "category" },
    { "name": "lookup",   "kind": "tool",   "input": { "id": 42 },   "output": { "status": "active" } },
    { "name": "answer",   "kind": "reason", "input": "...",          "output": "final reply" }
  ]
}
```

`nodes` is the only required field. Every other field has a sensible default.

---

## Node fields

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | string | — | **Required.** Human-readable span label (e.g. `"classify_ticket"`). |
| `id` | string | `"n{index}"` | Unique within the trace. Required when using `parent_id` to wire the DAG. |
| `parent_id` | string \| null | `null` | `id` of the parent span. `null` = root span. |
| `kind` | enum | `"other"` | `"reason"` · `"tool"` · `"retrieve"` · `"agent"` · `"other"` |
| `input` | any | `null` | Span input — string, object, array, or null. |
| `output` | any | `null` | Span output — string, object, array, or null. |
| `prompt_id` | string \| null | `null` | Stable identity for the underlying prompt. Spans sharing a `prompt_id` are treated as the same prompt at different steps. Used by Reflex for optimization. |
| `optimize` | boolean | `false` | Mark this span's prompt as a Reflex optimization target. |
| `tokens` | integer | `0` | Total LLM tokens (prompt + completion). |
| `started_at` | number \| null | `null` | Unix timestamp (float seconds) when the span started. |
| `ended_at` | number \| null | `null` | Unix timestamp (float seconds) when the span ended. |
| `error` | string \| null | `null` | Short error message if the span failed; `null` on success. |
| `metadata` | object | `{}` | Arbitrary key/value bag. Well-known keys below. |

### Well-known `metadata` keys

| Key | Type | Meaning |
|---|---|---|
| `mcp_server` | string | Name of the MCP server for `tool` spans (e.g. `"stripe"`, `"knowledge_base"`). |
| `tool_call_id` | string | LLM tool-use ID linking this tool span to the reasoning turn that dispatched it. |
| `error_code` | string | Machine-readable error code (e.g. `"rate_limit"`, `"context_overflow"`). |
| `latency_ms` | number | Wall-clock duration when `started_at`/`ended_at` aren't available. |

---

## Trace-level fields

| Field | Type | Default | Description |
|---|---|---|---|
| `nodes` | array | — | **Required.** Ordered list of spans. |
| `ideal` | string \| null | `null` | Expected / reference output for the run. Used by ablation and judges. |
| `metadata` | object | `{}` | Arbitrary trace-level key/value bag (e.g. `session_id`, `model`, `pipeline_version`). |

---

## Emitting a trace from other languages

### TypeScript / JavaScript

```typescript
interface TraceNode {
  name: string;
  id?: string;
  parent_id?: string | null;
  kind?: "reason" | "tool" | "retrieve" | "agent" | "other";
  input?: unknown;
  output?: unknown;
  prompt_id?: string | null;
  optimize?: boolean;
  tokens?: number;
  started_at?: number | null;
  ended_at?: number | null;
  error?: string | null;
  metadata?: Record<string, unknown>;
}

interface AgentTrace {
  nodes: TraceNode[];
  ideal?: string | null;
  metadata?: Record<string, unknown>;
}

// Emit a trace
const trace: AgentTrace = {
  nodes: [
    { name: "classify", kind: "reason", input: userMessage, output: category },
    { name: "lookup",   kind: "tool",   input: { id }, output: result, metadata: { mcp_server: "stripe" } },
    { name: "answer",   kind: "reason", input: prompt, output: reply },
  ],
  metadata: { session_id: sessionId },
};

// Write to file for Origin CLI
import { writeFileSync } from "fs";
writeFileSync("trace.json", JSON.stringify(trace, null, 2));
```

### Go

```go
package main

import (
    "encoding/json"
    "os"
    "time"
)

type TraceNode struct {
    Name      string         `json:"name"`
    ID        string         `json:"id,omitempty"`
    ParentID  *string        `json:"parent_id,omitempty"`
    Kind      string         `json:"kind,omitempty"`
    Input     any            `json:"input,omitempty"`
    Output    any            `json:"output,omitempty"`
    PromptID  *string        `json:"prompt_id,omitempty"`
    Optimize  bool           `json:"optimize,omitempty"`
    Tokens    int            `json:"tokens,omitempty"`
    StartedAt *float64       `json:"started_at,omitempty"`
    EndedAt   *float64       `json:"ended_at,omitempty"`
    Error     *string        `json:"error,omitempty"`
    Metadata  map[string]any `json:"metadata,omitempty"`
}

type AgentTrace struct {
    Nodes    []TraceNode    `json:"nodes"`
    Ideal    *string        `json:"ideal,omitempty"`
    Metadata map[string]any `json:"metadata,omitempty"`
}

func nowUnix() *float64 {
    t := float64(time.Now().UnixNano()) / 1e9
    return &t
}

func main() {
    trace := AgentTrace{
        Nodes: []TraceNode{
            {Name: "classify", Kind: "reason", Input: "user message", Output: "category"},
            {Name: "lookup",   Kind: "tool",   Input: map[string]any{"id": 42}, Output: map[string]any{"status": "active"}},
            {Name: "answer",   Kind: "reason", Input: "...", Output: "final reply"},
        },
    }

    f, _ := os.Create("trace.json")
    defer f.Close()
    json.NewEncoder(f).Encode(trace)
}
```

### Java

```java
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.File;
import java.util.*;

@JsonInclude(JsonInclude.Include.NON_NULL)
record TraceNode(
    String name,
    String id,
    @JsonProperty("parent_id") String parentId,
    String kind,
    Object input,
    Object output,
    @JsonProperty("prompt_id") String promptId,
    Boolean optimize,
    Integer tokens,
    @JsonProperty("started_at") Double startedAt,
    @JsonProperty("ended_at") Double endedAt,
    String error,
    Map<String, Object> metadata
) {}

@JsonInclude(JsonInclude.Include.NON_NULL)
record AgentTrace(
    List<TraceNode> nodes,
    String ideal,
    Map<String, Object> metadata
) {}

// Emit
var trace = new AgentTrace(
    List.of(
        new TraceNode("classify", null, null, "reason", "user message", "category", null, null, null, null, null, null, null),
        new TraceNode("answer",   null, null, "reason", "...", "reply", null, null, null, null, null, null, null)
    ),
    null,
    Map.of("session_id", "sess_abc123")
);
new ObjectMapper().writeValue(new File("trace.json"), trace);
```

---

## Feeding a trace to Origin

Once you have a `trace.json`, run attribution with the Origin CLI:

```bash
# Install Origin (Python only requirement — for running attribution, not for emitting traces)
pip install aevyra-origin[anthropic]

# Run attribution — outputs which node(s) caused the failure and why
aevyra-origin diagnose trace.json --score 0.2 --ideal "expected answer"
```

---

## Validating against the schema

```bash
# Python
pip install jsonschema
python -c "
import json, jsonschema
schema = json.load(open('agent-trace.schema.json'))
trace  = json.load(open('trace.json'))
jsonschema.validate(trace, schema)
print('valid')
"

# Node.js
npm install -g ajv-cli
ajv validate -s agent-trace.schema.json -d trace.json
```

---

## DAG wiring example

For traces with branching or hierarchy, use `id` + `parent_id`:

```json
{
  "nodes": [
    { "name": "plan",     "id": "n0",                    "kind": "reason" },
    { "name": "tool_a",   "id": "n1", "parent_id": "n0", "kind": "tool"   },
    { "name": "tool_b",   "id": "n2", "parent_id": "n0", "kind": "tool"   },
    { "name": "synthesize","id": "n3", "parent_id": "n0", "kind": "reason" }
  ]
}
```

`tool_a` and `tool_b` are parallel siblings (same `parent_id`). `synthesize` runs after both.
