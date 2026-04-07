"""
Bash 도구 — 셸 명령 실행

Claude Code에서 가장 많이 사용되는 도구 중 하나입니다.
원본은 src/tools/BashTool/BashTool.tsx (~1000줄)로,
보안 파싱, 샌드박스 실행, 프로그레스 스트리밍 등이 포함되어 있습니다.

여기서는 핵심 실행 로직만 구현합니다.
"""

from __future__ import annotations

import asyncio

from ..tool_base import Tool, ToolResult, build_tool


async def _execute_bash(args: dict) -> ToolResult:
    command = args.get("command", "")
    if not command:
        return ToolResult(error="command가 비어 있습니다")

    timeout = args.get("timeout", 30000) / 1000  # ms → sec

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

        output_parts = []
        if stdout:
            output_parts.append(stdout.decode("utf-8", errors="replace"))
        if stderr:
            output_parts.append(f"STDERR:\n{stderr.decode('utf-8', errors='replace')}")
        if proc.returncode != 0:
            output_parts.append(f"Exit code: {proc.returncode}")

        return ToolResult(data="\n".join(output_parts) if output_parts else "(빈 출력)")

    except asyncio.TimeoutError:
        return ToolResult(error=f"타임아웃 ({timeout}초)")
    except Exception as e:
        return ToolResult(error=str(e))


BashTool = build_tool(
    name="Bash",
    description=(
        "셸 명령을 실행합니다. ls, cat, grep, git 등 모든 셸 명령을 사용할 수 있습니다. "
        "장시간 실행되는 명령은 timeout 파라미터로 제한 시간을 지정하세요."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "실행할 셸 명령",
            },
            "timeout": {
                "type": "number",
                "description": "타임아웃 (밀리초). 기본값: 30000 (30초)",
            },
        },
        "required": ["command"],
    },
    call=_execute_bash,
    read_only=False,        # 셸은 파일을 수정할 수 있으므로
    concurrency_safe=False, # 셸 명령은 상호 간섭 가능
    destructive=False,      # 명령에 따라 다르지만 기본은 False
)
