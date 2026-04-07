"""
파일 읽기 도구

Claude Code의 FileReadTool (src/tools/FileReadTool/)에 대응합니다.
원본은 이미지, PDF, 노트북 등 다양한 포맷을 지원하지만,
여기서는 텍스트 파일 읽기만 구현합니다.
"""

from __future__ import annotations

import os

from ..tool_base import Tool, ToolResult, build_tool


async def _read_file(args: dict) -> ToolResult:
    file_path = args.get("file_path", "")
    if not file_path:
        return ToolResult(error="file_path가 비어 있습니다")

    if not os.path.isabs(file_path):
        return ToolResult(error=f"절대 경로를 사용하세요: {file_path}")

    if not os.path.exists(file_path):
        return ToolResult(error=f"파일이 존재하지 않습니다: {file_path}")

    offset = args.get("offset", 0)
    limit = args.get("limit", 2000)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        selected = lines[offset:offset + limit]
        # cat -n 형식: 줄 번호 + 탭 + 내용
        numbered = [f"{i + offset + 1}\t{line}" for i, line in enumerate(selected)]
        result = "".join(numbered)

        if offset + limit < len(lines):
            result += f"\n... [{len(lines) - offset - limit}줄 더 있음]"

        return ToolResult(data=result)
    except UnicodeDecodeError:
        return ToolResult(error=f"텍스트 파일이 아닙니다: {file_path}")
    except Exception as e:
        return ToolResult(error=str(e))


FileReadTool = build_tool(
    name="Read",
    description=(
        "파일을 읽습니다. 줄 번호와 함께 내용을 반환합니다. "
        "offset과 limit으로 읽을 범위를 지정할 수 있습니다."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "읽을 파일의 절대 경로",
            },
            "offset": {
                "type": "integer",
                "description": "읽기 시작할 줄 번호 (0-indexed). 기본값: 0",
            },
            "limit": {
                "type": "integer",
                "description": "읽을 최대 줄 수. 기본값: 2000",
            },
        },
        "required": ["file_path"],
    },
    call=_read_file,
    read_only=True,          # 파일을 수정하지 않음
    concurrency_safe=True,   # 여러 파일을 동시에 읽어도 안전
)
