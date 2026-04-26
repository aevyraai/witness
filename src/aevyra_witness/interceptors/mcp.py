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

"""MCP session interceptor — capture every tool call as a Witness span.

The Model Context Protocol (MCP) is the de facto standard for agent-tool
connectivity. Every ``call_tool`` invocation has a clean input/output
boundary, making it a natural :class:`~aevyra_witness.trace.TraceNode`.
This interceptor wraps any MCP ``ClientSession`` and records each call
automatically — no ``@span`` decorators needed in agent code.

Usage (async, typical)::

    from mcp import ClientSession
    from aevyra_witness.interceptors.mcp import wrap_mcp_session

    async with ClientSession(read, write) as session:
        await session.initialize()
        mcp = wrap_mcp_session(session, server_name="github")

        # Call tools as normal — every call is recorded
        result = await mcp.call_tool("create_issue", {"title": "Bug"})
        result = await mcp.call_tool("list_repos", {})

        trace = mcp.to_trace()   # AgentTrace with all captured spans

Attach to an existing Witness tracer::

    from aevyra_witness import trace as witness_trace

    with witness_trace() as tracer:
        mcp = wrap_mcp_session(session, server_name="slack", tracer=tracer)
        result = await mcp.call_tool("post_message", {...})

    captured = tracer.finish()   # spans include the MCP calls

Multiple servers::

    github = wrap_mcp_session(gh_session, server_name="github")
    slack  = wrap_mcp_session(sl_session, server_name="slack")

    # ... run agent ...

    from aevyra_witness import AgentTrace
    trace = AgentTrace(nodes=github.nodes + slack.nodes)

Design notes
------------

The interceptor uses duck typing — it wraps any object that exposes an
async ``call_tool(name, arguments, **kwargs)`` method. This means it
works with the official ``mcp`` Python SDK, any compatible shim, and
mocks in tests. The ``mcp`` package is **not** a hard dependency of
``aevyra-witness``.

The interceptor does not touch MCP's transport, session lifecycle, or
resource/prompt primitives — only ``call_tool``. Non-tool methods
(``list_tools``, ``read_resource``, etc.) are forwarded transparently
via ``__getattr__``.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from aevyra_witness.trace import AgentTrace, TraceNode


class MCPInterceptor:
    """Wraps an MCP ``ClientSession`` and records every ``call_tool`` call.

    Args:
        session:     Any object with an async ``call_tool(name, arguments,
                     **kwargs)`` method — typically ``mcp.ClientSession``.
        server_name: Human-readable name for this MCP server (e.g.
                     ``"github"``, ``"slack"``). Stored on each span's
                     ``mcp_server`` metadata key so Origin can attribute
                     failures to a specific server.
        parent_id:   Optional span id to set as ``parent_id`` on every
                     tool span. Useful when the MCP calls are children of
                     a reasoning span in a larger trace.
        tracer:      Optional Witness :class:`~aevyra_witness.runtime.Tracer`.
                     When provided, captured spans are appended to the
                     tracer's node list so they appear in
                     ``tracer.finish()`` automatically.

    The captured spans are also always available on :attr:`nodes` and via
    :meth:`to_trace`, regardless of whether a tracer is attached.
    """

    def __init__(
        self,
        session: Any,
        *,
        server_name: str = "",
        parent_id: str | None = None,
        tracer: Any | None = None,
    ) -> None:
        self._session = session
        self._server_name = server_name
        self._parent_id = parent_id
        self._tracer = tracer
        self.nodes: list[TraceNode] = []

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Call a tool on the underlying MCP session, recording the span.

        Passes all arguments through to the real ``call_tool`` unchanged.
        The span is appended to :attr:`nodes` (and the attached tracer's
        nodes, if any) whether the call succeeds or raises.
        """
        span_id = str(uuid.uuid4())[:8]
        started_at = time.time()
        error: str | None = None
        result: Any = None

        try:
            result = await self._session.call_tool(name, arguments, **kwargs)
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            ended_at = time.time()
            latency_ms = (ended_at - started_at) * 1000.0

            # Normalise the MCP result to a plain Python value.
            # mcp.types.CallToolResult has a .content list; extract text
            # from it so the span's output is human-readable in renders.
            output = _extract_output(result)

            node = TraceNode.mcp_tool(
                name,
                arguments=arguments,
                result=output,
                error=error,
                server=self._server_name or None,
                id=span_id,
                parent_id=self._parent_id,
                started_at=started_at,
                ended_at=ended_at,
                latency_ms=latency_ms,
            )
            self.nodes.append(node)

            # If a Witness tracer is attached, inject the node so it
            # appears in tracer.finish() without manual wiring.
            if self._tracer is not None:
                try:
                    self._tracer._nodes.append(node)
                except AttributeError:
                    pass  # tracer interface changed; degrade gracefully

        return result

    def to_trace(
        self,
        *,
        ideal: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentTrace:
        """Return an :class:`AgentTrace` containing all captured spans."""
        return AgentTrace(
            nodes=list(self.nodes),
            ideal=ideal,
            metadata=dict(metadata or {}),
        )

    # ------------------------------------------------------------------
    # Transparent proxy — forward every non-intercepted attribute to the
    # underlying session so callers can use the interceptor as a drop-in.
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)

    def __repr__(self) -> str:
        return f"MCPInterceptor(server={self._server_name!r}, calls={len(self.nodes)})"


def wrap_mcp_session(
    session: Any,
    *,
    server_name: str = "",
    parent_id: str | None = None,
    tracer: Any | None = None,
) -> MCPInterceptor:
    """Wrap an MCP ``ClientSession`` in an :class:`MCPInterceptor`.

    Convenience factory — equivalent to
    ``MCPInterceptor(session, server_name=..., ...)``.

    Args:
        session:     MCP ``ClientSession`` or any compatible object.
        server_name: Name of this MCP server (e.g. ``"github"``).
        parent_id:   Optional parent span id for all captured tool spans.
        tracer:      Optional Witness tracer to attach spans to.

    Returns:
        An :class:`MCPInterceptor` that proxies the session and records
        every ``call_tool`` invocation.

    Example::

        mcp = wrap_mcp_session(session, server_name="stripe")
        charge = await mcp.call_tool("get_charge", {"id": "ch_123"})
        trace = mcp.to_trace()
    """
    return MCPInterceptor(
        session,
        server_name=server_name,
        parent_id=parent_id,
        tracer=tracer,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_output(result: Any) -> Any:
    """Normalise an MCP ``CallToolResult`` to a plain Python value.

    The MCP SDK returns a ``CallToolResult`` with a ``content`` list of
    ``TextContent``, ``ImageContent``, or ``EmbeddedResource`` objects.
    We extract the text content so spans are readable in CLI renders and
    LLM prompts. Non-SDK objects (mocks, test stubs) are returned as-is.
    """
    if result is None:
        return None

    # mcp.types.CallToolResult exposes .content and .isError
    content = getattr(result, "content", None)
    if not isinstance(content, list):
        # Not an MCP SDK result — return as-is (handles mocks/stubs).
        return result

    parts: list[Any] = []
    for item in content:
        # TextContent has .text; ImageContent has .data; fall back to repr.
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(text)
        else:
            parts.append(repr(item))

    if not parts:
        return None
    return parts[0] if len(parts) == 1 else parts


__all__ = ["MCPInterceptor", "wrap_mcp_session"]
