"""
스킬 도구 — 스킬을 호출하는 도구

Claude Code에서 SkillTool은 사용자가 스킬을 호출할 때 사용하는 도구입니다.
LLM이 스킬 이름과 인자를 전달하면, 해당 스킬의 프롬프트를 확장하여
현재 대화에 주입(inline)하거나 서브 에이전트를 생성(fork)합니다.

핵심 실행 경로:
1. 스킬 이름으로 레지스트리에서 스킬 정의를 조회
2. 프롬프트 템플릿에 인자를 치환
3. inline 모드: 확장된 프롬프트를 새 메시지로 반환
4. fork 모드: 서브 에이전트에 프롬프트를 전달하여 독립 실행

참조:
- src/tools/SkillTool/SkillTool.ts — 스킬 도구 구현
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..tool_base import Tool, ToolResult
from ..skills.loader import SkillRegistry


# ─── 스킬 실행 결과 ─────────────────────────────────────────────

@dataclass
class SkillExecutionResult:
    """
    스킬 실행 결과.

    inline 모드에서는 expanded_prompt에 확장된 프롬프트가 들어가고,
    이를 대화 히스토리에 주입하여 LLM이 이어서 처리하게 합니다.

    fork 모드에서는 sub_agent_result에 서브 에이전트의 최종 응답이 들어갑니다.
    """
    skill_name: str
    mode: str  # "inline" | "fork"
    expanded_prompt: str = ""
    sub_agent_result: str = ""
    context_messages: list[dict] | None = None


# ─── SkillTool ───────────────────────────────────────────────────

class SkillTool(Tool):
    """
    스킬 호출 도구 — Claude Code의 SkillTool에 대응

    LLM이 이 도구를 호출하면:
    - inline 스킬: 프롬프트가 현재 대화에 주입됨
    - fork 스킬: 서브 에이전트가 생성되어 독립 실행됨

    이 구현에서는 inline 경로에 집중합니다.
    fork 경로는 AgentTool과 연동하여 구현할 수 있습니다.
    """

    def __init__(self, skill_registry: SkillRegistry):
        self._registry = skill_registry

    @property
    def name(self) -> str:
        return "Skill"

    @property
    def description(self) -> str:
        skill_list = ", ".join(self._registry.list_names()) if self._registry else "(없음)"
        return (
            f"등록된 스킬을 실행합니다. 사용 가능한 스킬: {skill_list}. "
            "스킬 이름과 선택적 인자를 전달하세요."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "skill": {
                    "type": "string",
                    "description": "실행할 스킬 이름",
                },
                "args": {
                    "type": "string",
                    "description": "스킬에 전달할 인자 (선택)",
                    "default": "",
                },
            },
            "required": ["skill"],
        }

    def is_read_only(self, args=None) -> bool:
        # 스킬 자체는 읽기 전용이 아님 (스킬이 쓰기 도구를 사용할 수 있음)
        return False

    async def call(self, args: dict[str, Any]) -> ToolResult:
        """
        스킬을 실행합니다.

        inline 모드: 확장된 프롬프트를 반환합니다.
            호출자(에이전트 루프)는 이 프롬프트를 대화에 주입하여
            LLM이 스킬의 지시를 따르게 합니다.

        fork 모드: 서브 에이전트 생성이 필요함을 알립니다.
            실제 구현에서는 AgentTool을 통해 서브 에이전트를 생성합니다.
        """
        skill_name = args.get("skill", "")
        skill_args = args.get("args", "")

        if not skill_name:
            return ToolResult(error="skill 이름이 비어 있습니다")

        # 스킬 조회
        skill = self._registry.find_by_name(skill_name)
        if not skill:
            available = ", ".join(self._registry.list_names())
            return ToolResult(
                error=f"스킬 '{skill_name}'을 찾을 수 없습니다. 사용 가능: {available}"
            )

        # 프롬프트 렌더링
        expanded = skill.render(args=skill_args)

        # ── inline 모드: 프롬프트를 현재 대화에 주입 ──
        if skill.context == "inline":
            # Claude Code에서는 이 프롬프트가 새 user 메시지로 주입됩니다.
            # 여기서는 tool_result로 프롬프트를 반환하여,
            # LLM이 이 지시를 읽고 따르게 합니다.
            result_text = (
                f"[스킬 '{skill_name}' 실행 — inline 모드]\n\n"
                f"{expanded}\n\n"
            )
            if skill.allowed_tools:
                result_text += f"사용 가능한 도구: {', '.join(skill.allowed_tools)}\n"

            return ToolResult(data=result_text)

        # ── fork 모드: 서브 에이전트 실행 필요 ──
        # 실제로는 AgentTool.spawn_agent()를 호출해야 합니다.
        # 여기서는 프롬프트를 반환하고, 서브 에이전트 실행은 호출자에게 위임합니다.
        result_text = (
            f"[스킬 '{skill_name}' 실행 — fork 모드]\n\n"
            f"서브 에이전트에게 전달할 프롬프트:\n{expanded}\n\n"
            "이 스킬은 fork 모드이므로 서브 에이전트가 독립적으로 실행합니다. "
            "결과를 기다려주세요."
        )
        return ToolResult(data=result_text)

    async def execute_inline(self, skill_name: str, skill_args: str = "") -> SkillExecutionResult | None:
        """
        inline 스킬을 실행하고, 대화에 주입할 메시지를 반환하는 편의 메서드.

        에이전트 루프에서 스킬 결과를 대화에 직접 주입할 때 사용합니다.
        """
        skill = self._registry.find_by_name(skill_name)
        if not skill or skill.context != "inline":
            return None

        expanded = skill.render(args=skill_args)

        # 대화에 주입할 메시지 생성
        context_messages = [
            {
                "role": "user",
                "content": (
                    f"[스킬 '{skill_name}' 활성화]\n\n"
                    f"{expanded}"
                ),
            }
        ]

        return SkillExecutionResult(
            skill_name=skill_name,
            mode="inline",
            expanded_prompt=expanded,
            context_messages=context_messages,
        )
