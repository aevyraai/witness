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

"""Live interceptors that capture agent traffic into :class:`AgentTrace` spans.

Unlike *adapters* (which parse existing log files), *interceptors* wrap
live objects — MCP sessions, HTTP clients, LLM SDKs — and record each
call as it happens.

Currently provided:

- :mod:`aevyra_witness.interceptors.mcp` — wraps any MCP ``ClientSession``
  and captures every ``call_tool`` invocation as a ``KIND_TOOL`` span.
"""

from aevyra_witness.interceptors.mcp import MCPInterceptor, wrap_mcp_session

__all__ = ["MCPInterceptor", "wrap_mcp_session"]
