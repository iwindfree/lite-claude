"""
다단계 권한 시스템 — Claude Code의 Tool Permission 파이프라인을 Python으로 구현

Claude Code는 도구 실행 전에 다단계 권한 검사를 수행합니다:

    1. 정적 규칙 검사 (allow/deny 패턴 매칭)
    2. Hooks 검사 (외부 프로세스가 허용/거부 결정)
    3. 사용자 프롬프트 (interactive 모드일 때 사용자에게 확인)

이 파이프라인은 "fail-closed" 원칙을 따릅니다:
- 명시적으로 허용하지 않은 것은 확인이 필요합니다
- deny 규칙은 allow 규칙보다 우선합니다

참조:
- src/Tool.ts:123-148              — ToolPermissionContext 타입
- src/hooks/toolPermission/        — Hook 기반 권한 검사
- src/permissions/permissionRules.ts — 규칙 매칭 로직
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable


# ─── 권한 모드 ──────────────────────────────────────────────────
#
# Claude Code에는 여러 권한 모드가 있습니다:
# - AUTO (--dangerously-skip-permissions): 모든 도구 자동 허용
# - ASK: 위험한 도구는 사용자에게 확인 (기본 모드)
# - DENY: 확인 없이 거부 (CI/CD 환경 등)

class PermissionMode(Enum):
    """권한 모드 — Claude Code의 PermissionMode에 대응"""
    AUTO = "auto"       # 모든 도구 자동 허용 (주의: 위험할 수 있음)
    ASK = "ask"         # 사용자에게 확인 (기본값)
    DENY = "deny"       # 자동 거부 (비대화형 환경)


# ─── 권한 규칙 ──────────────────────────────────────────────────
#
# Claude Code의 settings.json에서 permissions 배열로 규칙을 정의합니다:
#
#   "permissions": [
#     {"tool": "Bash", "input": "git *", "action": "allow"},
#     {"tool": "Bash", "input": "rm -rf *", "action": "deny"},
#     {"tool": "FileRead", "action": "allow"}
#   ]
#
# 규칙은 순서대로 평가되며, 첫 번째 매칭 규칙이 적용됩니다.

class PermissionAction(Enum):
    """규칙의 액션"""
    ALLOW = "allow"     # 도구 실행 허용
    DENY = "deny"       # 도구 실행 거부
    ASK = "ask"         # 사용자에게 확인


@dataclass
class PermissionRule:
    """
    권한 규칙 — Claude Code의 PermissionRule에 대응

    tool_pattern: 도구 이름 매칭 패턴 (glob, 예: "Bash", "File*")
    input_pattern: 입력 매칭 패턴 (glob, 선택적)
    action: 매칭 시 수행할 액션
    """
    tool_pattern: str
    action: PermissionAction
    input_pattern: str | None = None

    def matches(self, tool_name: str, tool_input: str = "") -> bool:
        """
        이 규칙이 주어진 도구/입력에 매칭되는지 확인.

        tool_pattern은 glob 패턴으로 도구 이름을 매칭합니다.
        input_pattern이 있으면 입력 문자열도 매칭해야 합니다.
        """
        if not fnmatch.fnmatch(tool_name, self.tool_pattern):
            return False
        if self.input_pattern is not None:
            return fnmatch.fnmatch(tool_input, self.input_pattern)
        return True


# ─── 권한 검사 결과 ─────────────────────────────────────────────

class PermissionResult(Enum):
    """권한 검사 결과"""
    ALLOWED = "allowed"         # 실행 허용
    DENIED = "denied"           # 실행 거부
    ASK_USER = "ask_user"       # 사용자 확인 필요


@dataclass
class PermissionCheckResult:
    """권한 검사의 상세 결과"""
    result: PermissionResult
    reason: str = ""
    matched_rule: PermissionRule | None = None


# ─── Hook 레지스트리 (간소화) ───────────────────────────────────
#
# Claude Code의 hooks 시스템은 외부 프로세스를 실행하여
# 도구 권한을 확인할 수 있습니다. 여기서는 Python 콜백으로 간소화합니다.
#
# 실제 Claude Code에서는:
# - settings.json의 hooks.tool_permission 배열
# - 각 hook은 외부 명령어 (예: "python check_safety.py")
# - 종료 코드 0=허용, 비0=거부, stderr=거부 메시지

# Hook 콜백 타입: (tool_name, tool_input) -> (허용여부, 거부이유)
HookCallback = Callable[[str, dict[str, Any]], Awaitable[tuple[bool, str]]]


@dataclass
class HookRegistry:
    """
    Hook 레지스트리 — Claude Code의 HookRegistry 간소화 버전

    tool_permission 이벤트에 대한 콜백을 관리합니다.
    """
    _hooks: list[HookCallback] = field(default_factory=list)

    def register(self, hook: HookCallback) -> None:
        """hook 콜백 등록"""
        self._hooks.append(hook)

    def clear(self) -> None:
        """모든 hook 제거"""
        self._hooks.clear()

    async def check_tool_permission(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> tuple[bool, str]:
        """
        등록된 모든 hook을 실행하여 도구 권한 확인.

        모든 hook이 허용해야 최종 허용됩니다 (AND 로직).
        하나라도 거부하면 즉시 거부 반환.
        """
        for hook in self._hooks:
            try:
                allowed, reason = await hook(tool_name, tool_input)
                if not allowed:
                    return False, reason or f"Hook이 {tool_name} 실행을 거부했습니다"
            except Exception as e:
                # Hook 실행 실패 시 거부 (fail-closed)
                return False, f"Hook 실행 오류: {e}"
        return True, ""


# ─── 사용자 프롬프트 콜백 ───────────────────────────────────────

# 사용자에게 확인을 요청하는 콜백 타입
# (tool_name, tool_input, description) -> 허용여부
UserPromptCallback = Callable[[str, dict[str, Any], str], Awaitable[bool]]


async def default_user_prompt(
    tool_name: str, tool_input: dict[str, Any], description: str
) -> bool:
    """기본 사용자 프롬프트 — 항상 거부 (비대화형 환경)"""
    return False


# ─── 권한 시스템 ────────────────────────────────────────────────

class PermissionSystem:
    """
    다단계 권한 시스템 — Claude Code의 도구 권한 파이프라인

    검사 순서 (파이프라인):
    1. 모드 확인 — AUTO면 즉시 허용, DENY면 즉시 거부
    2. 정적 규칙 검사 — 규칙 순서대로 매칭, 첫 매칭 적용
    3. Hooks 검사 — 등록된 hook 콜백 실행
    4. 사용자 프롬프트 — ASK 모드이고 읽기전용이 아니면 사용자에게 확인

    중요한 원칙:
    - deny 규칙은 다른 모든 것보다 우선
    - 읽기전용 도구는 기본적으로 허용 (명시적 deny가 없는 한)
    - 규칙에 매칭되지 않으면 사용자 확인으로 fall-through
    """

    def __init__(
        self,
        *,
        mode: PermissionMode = PermissionMode.ASK,
        rules: list[PermissionRule] | None = None,
        hooks: HookRegistry | None = None,
        user_prompt: UserPromptCallback | None = None,
    ):
        self.mode = mode
        self.rules: list[PermissionRule] = rules or []
        self.hooks = hooks or HookRegistry()
        self.user_prompt = user_prompt or default_user_prompt

    def add_rule(self, rule: PermissionRule) -> None:
        """규칙 추가 (뒤에 추가 — 나중에 추가한 규칙이 후순위)"""
        self.rules.append(rule)

    def _input_to_string(self, tool_input: dict[str, Any]) -> str:
        """도구 입력을 패턴 매칭용 문자열로 변환"""
        # 주요 필드를 공백으로 연결 (command, file_path 등)
        parts = []
        for key in ("command", "file_path", "path", "content", "query"):
            if key in tool_input:
                parts.append(str(tool_input[key]))
        return " ".join(parts) if parts else str(tool_input)

    def _check_rules(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> PermissionCheckResult | None:
        """
        정적 규칙 검사 — 순서대로 매칭, 첫 매칭 적용

        반환: 매칭 결과 또는 None (매칭 규칙 없음)
        """
        input_str = self._input_to_string(tool_input)

        for rule in self.rules:
            if rule.matches(tool_name, input_str):
                if rule.action == PermissionAction.ALLOW:
                    return PermissionCheckResult(
                        result=PermissionResult.ALLOWED,
                        reason=f"규칙 허용: {rule.tool_pattern}",
                        matched_rule=rule,
                    )
                elif rule.action == PermissionAction.DENY:
                    return PermissionCheckResult(
                        result=PermissionResult.DENIED,
                        reason=f"규칙 거부: {rule.tool_pattern}",
                        matched_rule=rule,
                    )
                elif rule.action == PermissionAction.ASK:
                    return PermissionCheckResult(
                        result=PermissionResult.ASK_USER,
                        reason=f"규칙에 의해 사용자 확인 필요: {rule.tool_pattern}",
                        matched_rule=rule,
                    )
        return None  # 매칭 규칙 없음

    async def check_permission(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        *,
        is_read_only: bool = False,
    ) -> PermissionCheckResult:
        """
        도구 실행 권한을 다단계로 검사 — Claude Code의 ToolPermissionContext 처리 흐름

        파이프라인:
        1. 모드 확인 (AUTO → 허용, DENY → 거부)
        2. 정적 규칙 검사
        3. Hooks 검사
        4. 읽기전용 도구는 기본 허용
        5. 사용자 프롬프트 (ASK 모드)

        Args:
            tool_name: 도구 이름
            tool_input: 도구 입력 딕셔너리
            is_read_only: 읽기전용 도구 여부

        Returns:
            PermissionCheckResult
        """
        # Stage 1: 모드 확인
        if self.mode == PermissionMode.AUTO:
            return PermissionCheckResult(
                result=PermissionResult.ALLOWED,
                reason="AUTO 모드: 모든 도구 허용",
            )

        if self.mode == PermissionMode.DENY:
            return PermissionCheckResult(
                result=PermissionResult.DENIED,
                reason="DENY 모드: 모든 도구 거부",
            )

        # Stage 2: 정적 규칙 검사
        rule_result = self._check_rules(tool_name, tool_input)
        if rule_result is not None:
            # deny 규칙은 최종 결정
            if rule_result.result == PermissionResult.DENIED:
                return rule_result
            # allow 규칙은 허용 (hooks 건너뜀)
            if rule_result.result == PermissionResult.ALLOWED:
                return rule_result
            # ask 규칙은 아래 사용자 프롬프트로 fall-through

        # Stage 3: Hooks 검사
        hook_allowed, hook_reason = await self.hooks.check_tool_permission(
            tool_name, tool_input
        )
        if not hook_allowed:
            return PermissionCheckResult(
                result=PermissionResult.DENIED,
                reason=hook_reason,
            )

        # Stage 4: 읽기전용 도구는 기본 허용
        if is_read_only:
            return PermissionCheckResult(
                result=PermissionResult.ALLOWED,
                reason="읽기전용 도구: 기본 허용",
            )

        # Stage 5: 사용자 프롬프트
        description = f"{tool_name}을(를) 실행하시겠습니까?"
        try:
            user_allowed = await self.user_prompt(
                tool_name, tool_input, description
            )
        except Exception:
            user_allowed = False

        if user_allowed:
            return PermissionCheckResult(
                result=PermissionResult.ALLOWED,
                reason="사용자가 허용",
            )
        else:
            return PermissionCheckResult(
                result=PermissionResult.DENIED,
                reason="사용자가 거부",
            )


# ─── 편의 함수: 일반적인 규칙 프리셋 ──────────────────────────

def create_default_rules() -> list[PermissionRule]:
    """
    기본 권한 규칙 — Claude Code의 기본 설정과 유사

    읽기 도구는 허용, 위험한 명령은 거부/확인.
    """
    return [
        # 읽기 도구는 항상 허용
        PermissionRule("FileRead", PermissionAction.ALLOW),
        PermissionRule("Glob", PermissionAction.ALLOW),
        PermissionRule("Grep", PermissionAction.ALLOW),

        # 위험한 bash 명령은 거부
        PermissionRule("Bash", PermissionAction.DENY, input_pattern="*rm -rf /*"),
        PermissionRule("Bash", PermissionAction.DENY, input_pattern="*sudo *"),

        # 안전한 git 명령은 허용
        PermissionRule("Bash", PermissionAction.ALLOW, input_pattern="git log*"),
        PermissionRule("Bash", PermissionAction.ALLOW, input_pattern="git status*"),
        PermissionRule("Bash", PermissionAction.ALLOW, input_pattern="git diff*"),

        # 나머지 Bash 명령은 사용자 확인
        PermissionRule("Bash", PermissionAction.ASK),

        # 파일 쓰기는 사용자 확인
        PermissionRule("FileWrite", PermissionAction.ASK),
        PermissionRule("FileEdit", PermissionAction.ASK),
    ]
