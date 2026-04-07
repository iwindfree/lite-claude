"""
서브 에이전트 생성 도구 — 독립 에이전트를 포크하여 작업 위임

Claude Code의 AgentTool은 복잡한 작업을 서브 에이전트에게 위임합니다.
서브 에이전트는 부모의 컨텍스트를 포크하여 독립적으로 실행되며,
자체 도구 세트와 모델을 사용할 수 있습니다.

핵심 개념:
- 부모 컨텍스트 포크: 서브 에이전트는 부모의 시스템 프롬프트를 상속하되 독립 실행
- 도구 제한: 서브 에이전트는 제한된 도구만 사용 가능 (보안/효율)
- 결과 수집: 서브 에이전트의 최종 응답을 부모에게 반환

참조:
- src/tools/AgentTool/runAgent.ts     — 서브 에이전트 실행
- src/tools/AgentTool/forkSubagent.ts — 컨텍스트 포크
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from ..tool_base import Tool, ToolResult


# ─── 에이전트 정의 ──────────────────────────────────────────────

@dataclass
class AgentDefinition:
    """
    서브 에이전트를 정의하는 데이터.

    name: 에이전트 식별자
    description: 에이전트 설명 (LLM이 에이전트 선택 시 참고)
    system_prompt: 에이전트의 시스템 프롬프트
    tools: 사용 가능한 도구 이름 목록 (None이면 부모와 동일)
    model: 사용할 LLM 모델 (None이면 부모와 동일)
    """
    name: str
    description: str
    system_prompt: str = ""
    tools: list[str] | None = None
    model: str | None = None


# ─── 내장 에이전트 정의 ─────────────────────────────────────────
#
# Claude Code에는 다양한 서브 에이전트 패턴이 있습니다.
# 여기서는 대표적인 3가지를 정의합니다.

BUILT_IN_AGENTS: dict[str, AgentDefinition] = {
    "general_purpose": AgentDefinition(
        name="general_purpose",
        description="범용 서브 에이전트. 복잡한 작업을 분할하여 처리합니다.",
        system_prompt=(
            "당신은 범용 서브 에이전트입니다. "
            "부모 에이전트로부터 위임받은 작업을 수행합니다. "
            "작업을 완료하면 결과를 명확하게 요약하세요."
        ),
    ),
    "explorer": AgentDefinition(
        name="explorer",
        description="코드베이스 탐색 전문 에이전트. 파일 검색, 구조 분석 등을 수행합니다.",
        system_prompt=(
            "당신은 코드베이스 탐색 전문 에이전트입니다. "
            "파일을 검색하고, 코드 구조를 분석하고, 관련 정보를 수집합니다. "
            "파괴적인 변경은 하지 마세요. 읽기 전용 도구만 사용합니다."
        ),
        tools=["Bash", "Read", "Grep", "Glob"],
    ),
    "planner": AgentDefinition(
        name="planner",
        description="작업 계획 수립 에이전트. 복잡한 작업을 단계별로 분해합니다.",
        system_prompt=(
            "당신은 작업 계획 수립 전문 에이전트입니다. "
            "주어진 목표를 분석하고 실행 가능한 단계별 계획을 만듭니다. "
            "코드를 직접 수정하지 말고, 계획만 수립하세요."
        ),
        tools=["Bash", "Read", "Grep", "Glob"],
    ),
}


# ─── 서브 에이전트 실행 ─────────────────────────────────────────

# 에이전트 루프 실행 함수 타입
# (prompt, system, tools, model) -> final_text
AgentRunner = Callable[
    [str, str, list[str] | None, str | None],
    Awaitable[str],
]


async def spawn_agent(
    *,
    definition: AgentDefinition,
    prompt: str,
    runner: AgentRunner | None = None,
    parent_system: str = "",
) -> str:
    """
    서브 에이전트를 생성하고 실행 — Claude Code의 forkSubagent() + runAgent()에 대응

    1. 부모 컨텍스트를 포크 (시스템 프롬프트 상속)
    2. 에이전트 정의에 따라 도구/모델 설정
    3. 독립적으로 에이전트 루프 실행
    4. 최종 결과 반환

    Args:
        definition: 에이전트 정의
        prompt: 서브 에이전트에게 전달할 작업 프롬프트
        runner: 실제 에이전트 루프를 실행하는 함수 (None이면 시뮬레이션)
        parent_system: 부모의 시스템 프롬프트 (상속용)
    """
    # 시스템 프롬프트 구성: 부모 + 에이전트 고유 프롬프트
    system_parts = []
    if parent_system:
        system_parts.append(f"[상위 에이전트 컨텍스트]\n{parent_system}")
    if definition.system_prompt:
        system_parts.append(f"[서브 에이전트 역할]\n{definition.system_prompt}")
    merged_system = "\n\n".join(system_parts)

    if runner:
        # 실제 에이전트 루프 실행
        result = await runner(
            prompt,
            merged_system,
            definition.tools,
            definition.model,
        )
        return result

    # runner가 없으면 시뮬레이션 결과 반환
    return (
        f"[서브 에이전트 '{definition.name}' 시뮬레이션]\n"
        f"프롬프트: {prompt[:200]}...\n"
        f"시스템: {merged_system[:200]}...\n"
        f"(실제 실행하려면 runner를 제공하세요)"
    )


# ─── AgentTool ───────────────────────────────────────────────────

class AgentTool(Tool):
    """
    서브 에이전트 생성 도구 — Claude Code의 AgentTool에 대응

    LLM이 이 도구를 호출하면 서브 에이전트가 생성되어 독립적으로 작업을 수행합니다.
    서브 에이전트는 자체 에이전트 루프를 돌며, 완료 후 결과를 반환합니다.
    """

    def __init__(
        self,
        *,
        runner: AgentRunner | None = None,
        parent_system: str = "",
        custom_agents: dict[str, AgentDefinition] | None = None,
    ):
        self._runner = runner
        self._parent_system = parent_system
        # 내장 에이전트 + 커스텀 에이전트
        self._agents = dict(BUILT_IN_AGENTS)
        if custom_agents:
            self._agents.update(custom_agents)

    @property
    def name(self) -> str:
        return "Agent"

    @property
    def description(self) -> str:
        agent_list = ", ".join(
            f"{name}({defn.description})"
            for name, defn in self._agents.items()
        )
        return (
            "서브 에이전트를 생성하여 복잡한 작업을 위임합니다. "
            f"사용 가능한 에이전트: {agent_list}"
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "description": "사용할 에이전트 이름 (비워두면 general_purpose 사용)",
                    "default": "general_purpose",
                },
                "prompt": {
                    "type": "string",
                    "description": "서브 에이전트에게 전달할 작업 설명",
                },
            },
            "required": ["prompt"],
        }

    def is_read_only(self, args=None) -> bool:
        return False

    async def call(self, args: dict[str, Any]) -> ToolResult:
        """서브 에이전트를 생성하고 결과를 반환합니다."""
        agent_name = args.get("agent", "general_purpose")
        prompt = args.get("prompt", "")

        if not prompt:
            return ToolResult(error="prompt가 비어 있습니다")

        # 에이전트 정의 조회
        definition = self._agents.get(agent_name)
        if not definition:
            available = ", ".join(self._agents.keys())
            return ToolResult(
                error=f"에이전트 '{agent_name}'을 찾을 수 없습니다. 사용 가능: {available}"
            )

        try:
            result = await spawn_agent(
                definition=definition,
                prompt=prompt,
                runner=self._runner,
                parent_system=self._parent_system,
            )
            return ToolResult(data=result)
        except Exception as e:
            return ToolResult(error=f"서브 에이전트 실행 실패: {e}")
