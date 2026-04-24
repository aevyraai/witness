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

"""aevyra-witness — the shared agent trace primitive for the Aevyra stack.

Supports simple linear pipelines and full DAG topologies (N-step reasoning
chains with M-parallel tool calls). See :mod:`aevyra_witness.trace` for the
schema.
"""

from aevyra_witness.adapters import from_openclaw_jsonl, from_otel_spans
from aevyra_witness.interceptors import MCPInterceptor, wrap_mcp_session
from aevyra_witness.runtime import Tracer, current_tracer, span, trace
from aevyra_witness.trace import (
    KIND_AGENT,
    KIND_OTHER,
    KIND_REASON,
    KIND_RETRIEVE,
    KIND_TOOL,
    META_ERROR_CODE,
    META_LATENCY_MS,
    META_MCP_SERVER,
    META_TOOL_CALL_ID,
    VALID_KINDS,
    AgentTrace,
    TraceNode,
)

__version__ = "0.1.0"

__all__ = [
    "AgentTrace",
    "TraceNode",
    "Tracer",
    "trace",
    "span",
    "current_tracer",
    "from_openclaw_jsonl",
    "from_otel_spans",
    "MCPInterceptor",
    "wrap_mcp_session",
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
