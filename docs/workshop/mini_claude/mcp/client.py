"""
MCP (Model Context Protocol) 클라이언트 — 외부 도구 서버와의 통신

MCP는 Claude Code가 외부 도구 서버(예: 파일시스템, 데이터베이스, API 등)와
통신하기 위한 프로토콜입니다. 이 모듈은 stdio 기반 MCP 클라이언트를 구현합니다.

핵심 흐름:
  1. 서버 프로세스를 subprocess로 시작 (stdio transport)
  2. JSON-RPC로 initialize 핸드셰이크
  3. tools/list로 사용 가능한 도구 목록 받기
  4. 각 MCP 도구를 우리 Tool 인터페이스로 래핑 (MCPTool)
  5. tools/call로 실제 도구 호출

참조:
- src/services/mcp/mcpClient.ts  — MCP 클라이언트 구현
- MCP 스펙: https://modelcontextprotocol.io/
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any

from mini_claude.tool_base import Tool, ToolResult


# ─── MCP 서버 설정 ──────────────────────────────────────────────

@dataclass
class MCPServerConfig:
    """
    MCP 서버 연결 설정 — Claude Code의 McpServerConfig에 대응

    name:    서버 이름 (도구 이름 접두사로 사용)
    command: 실행할 명령어 (예: "npx", "python")
    args:    명령어 인자 리스트
    env:     추가 환경 변수 (현재 환경에 병합됨)
    """
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


# ─── JSON-RPC 메시지 헬퍼 ───────────────────────────────────────

def _make_request(method: str, params: dict | None = None, req_id: int = 1) -> bytes:
    """JSON-RPC 2.0 요청 메시지를 생성합니다."""
    msg = {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": method,
    }
    if params is not None:
        msg["params"] = params
    # MCP uses newline-delimited JSON over stdio
    return (json.dumps(msg) + "\n").encode()


async def _read_response(stdout: asyncio.StreamReader) -> dict:
    """
    stdout에서 JSON-RPC 응답 한 줄을 읽어 파싱합니다.

    MCP stdio transport는 줄 단위(newline-delimited) JSON을 사용합니다.
    서버가 notification을 먼저 보낼 수 있으므로, id가 있는
    응답(response)이 올 때까지 건너뜁니다.
    """
    while True:
        line = await stdout.readline()
        if not line:
            raise ConnectionError("MCP server closed stdout unexpectedly")
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        # Skip JSON-RPC notifications (no "id" field)
        if "id" in data:
            return data


# ─── MCPTool: MCP 도구를 Tool 인터페이스로 래핑 ────────────────

class MCPTool(Tool):
    """
    MCP 서버의 도구를 우리 Tool 인터페이스로 래핑합니다.

    Claude Code에서도 MCP 도구는 내장 도구와 동일한 Tool 인터페이스를
    구현합니다. 이 덕분에 에이전트 루프는 도구가 내장인지 MCP인지 구분할
    필요가 없습니다.

    도구 이름은 "서버이름__도구이름" 형식으로 지어져 충돌을 방지합니다.
    """

    def __init__(
        self,
        server_name: str,
        tool_name: str,
        tool_description: str,
        tool_input_schema: dict,
        client: MCPClient,
    ):
        self._server_name = server_name
        self._tool_name = tool_name
        self._description = tool_description
        self._input_schema = tool_input_schema
        self._client = client

    @property
    def name(self) -> str:
        # Namespaced: "server__tool" to avoid collisions
        return f"{self._server_name}__{self._tool_name}"

    @property
    def description(self) -> str:
        return self._description

    @property
    def input_schema(self) -> dict:
        return self._input_schema

    async def call(self, args: dict[str, Any]) -> ToolResult:
        """MCP tools/call을 호출하여 결과를 반환합니다."""
        return await self._client.call_tool(self._tool_name, args)

    def is_read_only(self, args=None) -> bool:
        # MCP tools are external — assume not read-only (fail-closed)
        return False


# ─── MCPClient: 서버 연결 및 통신 ──────────────────────────────

class MCPClient:
    """
    MCP 서버와 stdio로 통신하는 클라이언트.

    사용법:
        config = MCPServerConfig(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        )
        client = MCPClient(config)
        tools = await client.connect()   # Tool 인터페이스 리스트 반환
        # ... 에이전트 루프에서 tools 사용 ...
        await client.disconnect()

    라이프사이클:
      connect()    → 프로세스 시작 + 핸드셰이크 + 도구 목록 획득
      call_tool()  → tools/call JSON-RPC 호출
      disconnect() → 프로세스 종료
    """

    PROTOCOL_VERSION = "2024-11-05"

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._process: asyncio.subprocess.Process | None = None
        self._request_id = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    # ── 연결 ──

    async def connect(self) -> list[Tool]:
        """
        MCP 서버에 연결하고 사용 가능한 도구 목록을 반환합니다.

        단계:
          1) subprocess로 서버 프로세스 시작
          2) initialize 핸드셰이크 (프로토콜 버전 협상)
          3) initialized 알림 전송
          4) tools/list로 도구 목록 획득
          5) 각 도구를 MCPTool로 래핑하여 반환
        """
        # Merge environment variables
        env = {**os.environ, **self.config.env}

        # Start server process with stdio transport
        self._process = await asyncio.create_subprocess_exec(
            self.config.command,
            *self.config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        # Step 1: Send initialize request
        init_params = {
            "protocolVersion": self.PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {
                "name": "mini-claude",
                "version": "0.1.0",
            },
        }
        await self._send("initialize", init_params)
        init_response = await _read_response(self._process.stdout)

        if "error" in init_response:
            raise RuntimeError(
                f"MCP initialize failed: {init_response['error']}"
            )

        # Step 2: Send initialized notification (no id = notification)
        notification = json.dumps({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }) + "\n"
        self._process.stdin.write(notification.encode())
        await self._process.stdin.drain()

        # Step 3: List available tools
        await self._send("tools/list", {})
        tools_response = await _read_response(self._process.stdout)

        if "error" in tools_response:
            raise RuntimeError(
                f"MCP tools/list failed: {tools_response['error']}"
            )

        # Step 4: Wrap each MCP tool as MCPTool
        raw_tools = tools_response.get("result", {}).get("tools", [])
        wrapped: list[Tool] = []
        for t in raw_tools:
            wrapped.append(MCPTool(
                server_name=self.config.name,
                tool_name=t["name"],
                tool_description=t.get("description", ""),
                tool_input_schema=t.get("inputSchema", {"type": "object"}),
                client=self,
            ))

        return wrapped

    # ── 도구 호출 ──

    async def call_tool(self, tool_name: str, arguments: dict) -> ToolResult:
        """
        MCP tools/call을 호출합니다.

        JSON-RPC 요청을 보내고 응답을 ToolResult로 변환합니다.
        MCP의 content 배열에서 text 항목들을 결합하여 반환합니다.
        """
        if not self._process or self._process.returncode is not None:
            return ToolResult(error="MCP server is not running")

        await self._send("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        response = await _read_response(self._process.stdout)

        if "error" in response:
            return ToolResult(error=str(response["error"]))

        result = response.get("result", {})

        # MCP returns content as an array of {type, text} objects
        content_parts = result.get("content", [])
        text_parts = [
            part.get("text", "")
            for part in content_parts
            if part.get("type") == "text"
        ]
        combined_text = "\n".join(text_parts)

        if result.get("isError"):
            return ToolResult(error=combined_text)

        return ToolResult(data=combined_text)

    # ── 내부 헬퍼 ──

    async def _send(self, method: str, params: dict | None = None) -> None:
        """JSON-RPC 요청을 서버 stdin에 씁니다."""
        if not self._process or not self._process.stdin:
            raise RuntimeError("MCP server process not started")
        req_id = self._next_id()
        data = _make_request(method, params, req_id)
        self._process.stdin.write(data)
        await self._process.stdin.drain()

    # ── 종료 ──

    async def disconnect(self) -> None:
        """MCP 서버 프로세스를 종료합니다."""
        if self._process and self._process.returncode is None:
            self._process.stdin.close()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
        self._process = None
