"""
도구 레지스트리 — Claude Code의 도구 관리 시스템

Claude Code는 src/tools.ts에서 모든 도구를 등록하고 관리합니다.
- getAllBaseTools(): 모든 빌트인 도구를 반환
- getTools(): 권한 필터링된 도구 반환
- assembleToolPool(): 빌트인 + MCP 도구를 합쳐 최종 도구 풀 생성

참조:
- src/tools.ts:193-249    — getAllBaseTools()
- src/tools.ts:345+        — assembleToolPool()
- src/Tool.ts:358           — findToolByName()
"""

from __future__ import annotations

from typing import Any

from .tool_base import Tool
from .llm_client import tool_to_anthropic_schema, tool_to_openai_schema


class ToolRegistry:
    """
    도구 레지스트리 — Claude Code의 tools.ts에 대응

    도구를 등록하고, 이름으로 찾고, API 스키마를 생성합니다.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._aliases: dict[str, str] = {}  # alias → primary name

    def register(self, tool: Tool) -> None:
        """도구를 레지스트리에 등록"""
        if not tool.is_enabled():
            return  # 비활성 도구는 등록하지 않음
        self._tools[tool.name] = tool
        for alias in tool.aliases:
            self._aliases[alias] = tool.name

    def find_by_name(self, name: str) -> Tool | None:
        """
        이름으로 도구를 찾습니다 — Claude Code의 findToolByName() (Tool.ts:358)
        별칭(alias)으로도 찾을 수 있습니다.
        """
        if name in self._tools:
            return self._tools[name]
        primary = self._aliases.get(name)
        if primary:
            return self._tools.get(primary)
        return None

    def get_all(self) -> list[Tool]:
        """등록된 모든 활성 도구 반환"""
        return list(self._tools.values())

    def get_names(self) -> list[str]:
        """등록된 모든 도구 이름"""
        return list(self._tools.keys())

    def to_anthropic_schemas(self) -> list[dict]:
        """모든 도구를 Anthropic API 스키마로 변환"""
        return [
            tool_to_anthropic_schema(t.name, t.description, t.input_schema)
            for t in self._tools.values()
        ]

    def to_openai_schemas(self) -> list[dict]:
        """모든 도구를 OpenAI API 스키마로 변환"""
        return [
            tool_to_openai_schema(t.name, t.description, t.input_schema)
            for t in self._tools.values()
        ]

    def to_api_schemas(self, provider: str) -> list[dict]:
        """provider에 맞는 API 스키마 반환"""
        if provider == "anthropic":
            return self.to_anthropic_schemas()
        elif provider == "openai":
            return self.to_openai_schemas()
        else:
            raise ValueError(f"지원하지 않는 provider: {provider}")

    async def execute(self, name: str, args: dict[str, Any]) -> str:
        """
        도구를 실행하고 결과 문자열을 반환합니다.
        에이전트 루프의 tool_executor로 사용할 수 있습니다.

        이 메서드는 에이전트 루프와 도구 시스템을 연결하는 다리 역할을 합니다.
        """
        tool = self.find_by_name(name)
        if not tool:
            return f"Error: 알 수 없는 도구 '{name}'"

        # 입력 검증
        error = await tool.validate_input(args)
        if error:
            return f"Error: {error}"

        # 실행
        try:
            result = await tool.call(args)
            # 결과 크기 제한 (maxResultSizeChars, Tool.ts)
            content = result.content
            if len(content) > tool.max_result_size:
                content = content[:tool.max_result_size] + "\n... [결과가 잘렸습니다]"
            return content
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools or name in self._aliases
