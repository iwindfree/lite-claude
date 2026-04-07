"""
도구(Tool) 기반 클래스 — Claude Code의 Tool 인터페이스를 Python으로 구현

Claude Code에서 모든 도구(Bash, FileRead, FileEdit 등 ~40개)는 동일한
Tool 인터페이스를 구현합니다. 이 통일된 인터페이스 덕분에:
- 에이전트 루프가 도구의 구체적 구현을 몰라도 됩니다
- 새 도구를 추가할 때 루프를 수정할 필요가 없습니다
- 권한 체크, 병렬 실행 등을 도구에 무관하게 처리할 수 있습니다

참조:
- src/Tool.ts:362-410   — Tool 타입 정의
- src/Tool.ts:757-792   — buildTool() 팩토리와 TOOL_DEFAULTS
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


# ─── 도구 실행 결과 ──────────────────────────────────────────────

@dataclass
class ToolResult:
    """
    도구 실행 결과 — Claude Code의 ToolResult<T>에 대응 (Tool.ts:329)

    data: 도구가 반환하는 실제 결과 (문자열이 일반적)
    error: 에러 발생 시 메시지
    """
    data: str = ""
    error: str = ""

    @property
    def is_error(self) -> bool:
        return bool(self.error)

    @property
    def content(self) -> str:
        """API에 전달할 콘텐츠 — 에러가 있으면 에러 메시지, 없으면 데이터"""
        return self.error if self.is_error else self.data


# ─── Tool 추상 클래스 ────────────────────────────────────────────

class Tool(ABC):
    """
    모든 도구의 기반 클래스 — Claude Code의 Tool 타입 (Tool.ts:362)

    Claude Code에서 각 도구가 반드시 구현해야 하는 것:
    - name: 도구 이름 (LLM이 호출할 때 사용)
    - input_schema: JSON Schema로 정의된 입력 스키마
    - call(): 실제 실행 로직
    - description(): 도구 설명 (LLM 프롬프트에 포함)

    그리고 선택적으로 오버라이드할 수 있는 것 (기본값은 TOOL_DEFAULTS):
    - is_read_only(): 읽기 전용인가? (기본: False — 안전한 쪽으로)
    - is_concurrency_safe(): 병렬 실행 가능한가? (기본: False)
    - is_enabled(): 활성화 상태인가? (기본: True)
    - check_permissions(): 권한 확인 (기본: 항상 허용 — Step 7에서 확장)
    - validate_input(): 입력 검증 (기본: 항상 통과)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """도구 이름. LLM이 tool_use에서 이 이름으로 호출합니다."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """도구 설명. LLM 프롬프트에 포함되어 도구 선택에 영향을 줍니다."""
        ...

    @property
    @abstractmethod
    def input_schema(self) -> dict:
        """JSON Schema 형식의 입력 스키마"""
        ...

    @abstractmethod
    async def call(self, args: dict[str, Any]) -> ToolResult:
        """
        도구 실행 — Claude Code의 Tool.call() (Tool.ts:379)

        args: input_schema에 맞는 인자 딕셔너리
        returns: ToolResult
        """
        ...

    # ── 기본값을 가진 메서드들 (TOOL_DEFAULTS, Tool.ts:757) ──

    def is_read_only(self, args: dict | None = None) -> bool:
        """읽기 전용이면 True. 기본: False (쓰기 가능으로 가정 — fail-closed)"""
        return False

    def is_concurrency_safe(self, args: dict | None = None) -> bool:
        """병렬 실행 안전하면 True. 기본: False (안전하지 않다고 가정)"""
        return False

    def is_enabled(self) -> bool:
        """도구 활성화 여부. 기본: True"""
        return True

    def is_destructive(self, args: dict | None = None) -> bool:
        """파괴적 작업(삭제, 덮어쓰기 등)이면 True. 기본: False"""
        return False

    async def check_permissions(self, args: dict[str, Any]) -> bool:
        """권한 확인. True=허용, False=거부. 기본: 항상 허용 (Step 7에서 확장)"""
        return True

    async def validate_input(self, args: dict[str, Any]) -> str | None:
        """입력 검증. None=통과, 문자열=에러 메시지. 기본: 항상 통과"""
        return None

    @property
    def max_result_size(self) -> int:
        """결과 최대 크기 (문자 수). 이를 초과하면 잘립니다."""
        return 30_000

    @property
    def aliases(self) -> list[str]:
        """도구의 대체 이름 (이전 이름 호환용)"""
        return []


# ─── build_tool: 간편 도구 생성 팩토리 ──────────────────────────
#
# Claude Code의 buildTool() (Tool.ts:783)은 TOOL_DEFAULTS를 기본값으로
# 제공하여, 도구 정의 시 필수 필드만 지정하면 되게 합니다:
#
#   export const MyTool = buildTool({
#     name: "MyTool",
#     inputSchema: z.object({ ... }),
#     call(args, context) { ... },
#     description(input) { return "..." },
#   })
#
# Python에서는 함수형 도구를 간편하게 만드는 build_tool()을 제공합니다.

class _FunctionalTool(Tool):
    """build_tool()로 생성되는 내부 Tool 구현"""

    def __init__(
        self,
        *,
        tool_name: str,
        tool_description: str,
        tool_input_schema: dict,
        tool_call: Callable[[dict[str, Any]], Awaitable[ToolResult]],
        read_only: bool = False,
        concurrency_safe: bool = False,
        destructive: bool = False,
        tool_aliases: list[str] | None = None,
    ):
        self._name = tool_name
        self._description = tool_description
        self._input_schema = tool_input_schema
        self._call = tool_call
        self._read_only = read_only
        self._concurrency_safe = concurrency_safe
        self._destructive = destructive
        self._aliases = tool_aliases or []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def input_schema(self) -> dict:
        return self._input_schema

    async def call(self, args: dict[str, Any]) -> ToolResult:
        return await self._call(args)

    def is_read_only(self, args=None) -> bool:
        return self._read_only

    def is_concurrency_safe(self, args=None) -> bool:
        return self._concurrency_safe

    def is_destructive(self, args=None) -> bool:
        return self._destructive

    @property
    def aliases(self) -> list[str]:
        return self._aliases


def build_tool(
    *,
    name: str,
    description: str,
    input_schema: dict,
    call: Callable[[dict[str, Any]], Awaitable[ToolResult]],
    read_only: bool = False,
    concurrency_safe: bool = False,
    destructive: bool = False,
    aliases: list[str] | None = None,
) -> Tool:
    """
    도구를 간편하게 생성하는 팩토리 — Claude Code의 buildTool() (Tool.ts:783)

    사용법:
        my_tool = build_tool(
            name="get_time",
            description="현재 시간을 반환합니다",
            input_schema={"type": "object", "properties": {...}},
            call=async_func,
            read_only=True,          # 기본값 False (fail-closed)
            concurrency_safe=True,   # 기본값 False (fail-closed)
        )

    기본값 철학 (TOOL_DEFAULTS):
    - read_only=False      → 쓰기 가능으로 가정 (더 제한적)
    - concurrency_safe=False → 병렬 불가로 가정 (더 안전)
    - destructive=False     → 비파괴적으로 가정
    """
    return _FunctionalTool(
        tool_name=name,
        tool_description=description,
        tool_input_schema=input_schema,
        tool_call=call,
        read_only=read_only,
        concurrency_safe=concurrency_safe,
        destructive=destructive,
        tool_aliases=aliases,
    )
