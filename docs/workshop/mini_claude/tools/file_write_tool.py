"""
파일 쓰기 도구

Claude Code의 FileWriteTool (src/tools/FileWriteTool/)에 대응합니다.
"""

from __future__ import annotations

import os

from ..tool_base import Tool, ToolResult, build_tool


async def _write_file(args: dict) -> ToolResult:
    file_path = args.get("file_path", "")
    content = args.get("content", "")

    if not file_path:
        return ToolResult(error="file_path가 비어 있습니다")
    if not os.path.isabs(file_path):
        return ToolResult(error=f"절대 경로를 사용하세요: {file_path}")

    try:
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        return ToolResult(data=f"파일 작성 완료: {file_path} ({line_count}줄)")
    except Exception as e:
        return ToolResult(error=str(e))


FileWriteTool = build_tool(
    name="Write",
    description=(
        "파일을 생성하거나 덮어씁니다. "
        "기존 파일이 있으면 내용이 완전히 교체됩니다."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "작성할 파일의 절대 경로",
            },
            "content": {
                "type": "string",
                "description": "파일에 쓸 내용",
            },
        },
        "required": ["file_path", "content"],
    },
    call=_write_file,
    read_only=False,
    concurrency_safe=False,  # 같은 파일에 동시 쓰기 불안전
    destructive=True,        # 기존 파일을 덮어씀
)
