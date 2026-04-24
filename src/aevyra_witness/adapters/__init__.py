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

"""Adapters that import external trace formats into :class:`AgentTrace`.

Each adapter lives in its own submodule and exposes a small, stable
function surface. Adapters are intentionally tolerant: external trace
formats evolve, and we prefer to absorb shape drift here rather than
push it into the core schema.

Currently provided:

- :mod:`aevyra_witness.adapters.openclaw` — convert OpenClaw JSONL
  telemetry (LLM turns, tool / MCP tool calls, agent lifecycle) into an
  ``AgentTrace``.
- :mod:`aevyra_witness.adapters.otel` — convert OpenTelemetry spans
  (Python SDK ``ReadableSpan`` objects or OTLP JSON dicts) into an
  ``AgentTrace``. Covers LangGraph, CrewAI, AutoGen, Vercel AI SDK, and
  any framework emitting the GenAI semantic conventions.

Future adapters will live alongside (LangSmith, A2A). They all emit the
same ``AgentTrace``.
"""

from aevyra_witness.adapters.openclaw import from_openclaw_jsonl
from aevyra_witness.adapters.otel import from_otel_spans

__all__ = ["from_openclaw_jsonl", "from_otel_spans"]
