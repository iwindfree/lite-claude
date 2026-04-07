"""
라이프사이클 훅(Hooks) 시스템 — 도구 실행 전후에 커스텀 로직 삽입

Claude Code의 훅 시스템은 도구 사용 전/후, 세션 시작/종료 등 주요 시점에
사용자 정의 로직을 실행할 수 있게 합니다.

활용 예:
  - pre_tool_use:   특정 도구 실행을 차단하거나 인자를 수정
  - post_tool_use:  도구 실행 결과를 로깅하거나 알림 전송
  - session_start:  환경 검증, 초기화 작업
  - session_end:    정리 작업, 통계 보고

훅은 settings.json에서 선언하거나 프로그래밍 방식으로 등록합니다.
Claude Code에서 훅은 셸 명령어(command)로 실행되지만, 이 구현에서는
Python callable도 지원합니다.

참조:
- src/types/hooks.ts   — HookType, Hook 타입 정의
- src/utils/hooks.ts   — matchHook(), executeHooks() 구현
"""

from __future__ import annotations

import asyncio
import fnmatch
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable


# ─── HookType: 훅이 실행되는 시점 ──────────────────────────────

class HookType(str, Enum):
    """
    훅이 실행되는 라이프사이클 시점.

    Claude Code의 HookType에 대응합니다 (hooks.ts).
    str을 상속하여 직렬화/비교가 편리합니다.
    """
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    NOTIFICATION = "notification"


# ─── HookResult: 훅 실행 결과 ──────────────────────────────────

class HookAction(str, Enum):
    """훅이 반환할 수 있는 액션"""
    ALLOW = "allow"      # Continue normally
    DENY = "deny"        # Block the operation
    MODIFY = "modify"    # Modify the input and continue


@dataclass
class HookResult:
    """
    훅 실행 결과.

    action:  allow(계속), deny(차단), modify(수정 후 계속)
    message: 사용자에게 표시할 메시지 (deny 사유 등)
    modified_input: action이 modify일 때 수정된 도구 입력
    """
    action: HookAction = HookAction.ALLOW
    message: str = ""
    modified_input: dict[str, Any] | None = None


# ─── Hook: 훅 정의 ─────────────────────────────────────────────

# Type alias for Python callable hooks
HookCallable = Callable[[dict[str, Any]], Awaitable[HookResult]]


@dataclass
class Hook:
    """
    하나의 훅 정의.

    type:    이 훅이 실행되는 시점 (pre_tool_use, post_tool_use 등)
    matcher: 매칭할 도구 이름 패턴 (glob: "Bash", "File*", "*" 등)
             None이면 해당 타입의 모든 이벤트에 매칭
    command: 실행할 셸 명령어 (Claude Code 방식)
    handler: Python callable (command 대신 사용 가능)

    command와 handler 중 하나만 지정합니다.
    command는 셸에서 실행되며, 종료 코드 0=allow, 2=deny 입니다.
    """
    type: HookType
    matcher: str | None = None
    command: str | None = None
    handler: HookCallable | None = None

    def __post_init__(self):
        if not self.command and not self.handler:
            raise ValueError("Hook must have either 'command' or 'handler'")

    def matches(self, tool_name: str | None = None) -> bool:
        """
        이 훅이 주어진 도구 이름에 매칭되는지 확인합니다.

        matcher가 None이면 모든 도구에 매칭됩니다.
        glob 패턴을 지원합니다 (예: "File*"는 FileRead, FileEdit 등에 매칭).

        참조: src/utils/hooks.ts의 matchHook()
        """
        if self.matcher is None:
            return True
        if tool_name is None:
            return True
        return fnmatch.fnmatch(tool_name, self.matcher)


# ─── HookRegistry: 훅 등록 및 실행 ─────────────────────────────

class HookRegistry:
    """
    훅을 등록하고 실행하는 레지스트리.

    Claude Code에서 훅은 settings.json의 hooks 섹션에서 선언됩니다:
      {
        "hooks": {
          "pre_tool_use": [
            { "matcher": "Bash", "command": "check-safety.sh" }
          ]
        }
      }

    이 클래스는 프로그래밍 방식으로 훅을 등록하고 실행합니다.

    사용법:
        registry = HookRegistry()

        # 셸 명령어 훅 등록
        registry.register(Hook(
            type=HookType.PRE_TOOL_USE,
            matcher="Bash",
            command="echo 'checking bash safety'",
        ))

        # Python callable 훅 등록
        async def log_tool_use(context):
            print(f"Tool used: {context.get('tool_name')}")
            return HookResult(action=HookAction.ALLOW)

        registry.register(Hook(
            type=HookType.POST_TOOL_USE,
            handler=log_tool_use,
        ))

        # 훅 실행
        result = await registry.execute_hooks(
            HookType.PRE_TOOL_USE,
            tool_name="Bash",
            context={"command": "rm -rf /"},
        )
        if result.action == HookAction.DENY:
            print(f"Blocked: {result.message}")
    """

    def __init__(self):
        self._hooks: dict[HookType, list[Hook]] = {
            hook_type: [] for hook_type in HookType
        }

    def register(self, hook: Hook) -> None:
        """훅을 등록합니다."""
        self._hooks[hook.type].append(hook)

    def unregister_all(self, hook_type: HookType | None = None) -> None:
        """
        훅을 일괄 해제합니다.

        hook_type을 지정하면 해당 타입만, None이면 전체를 해제합니다.
        """
        if hook_type is None:
            for ht in HookType:
                self._hooks[ht].clear()
        else:
            self._hooks[hook_type].clear()

    async def execute_hooks(
        self,
        hook_type: HookType,
        tool_name: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> HookResult:
        """
        등록된 훅들을 순서대로 실행합니다.

        실행 규칙 (Claude Code의 executeHooks()를 따름):
          1) 해당 hook_type에 등록된 훅 중 tool_name에 매칭되는 것만 실행
          2) 훅은 등록 순서대로 실행됨 (순차)
          3) 어떤 훅이 DENY를 반환하면 즉시 중단하고 DENY 반환
          4) MODIFY를 반환하면 modified_input을 context에 반영하고 계속
          5) 모든 훅이 ALLOW이면 최종 ALLOW 반환

        Args:
            hook_type: 실행할 훅 타입
            tool_name: 매칭에 사용할 도구 이름
            context: 훅에 전달할 컨텍스트 정보

        Returns:
            최종 HookResult
        """
        if context is None:
            context = {}

        # Add tool_name to context for convenience
        if tool_name:
            context.setdefault("tool_name", tool_name)

        matching_hooks = [
            h for h in self._hooks[hook_type]
            if h.matches(tool_name)
        ]

        final_result = HookResult(action=HookAction.ALLOW)

        for hook in matching_hooks:
            if hook.handler:
                result = await hook.handler(context)
            elif hook.command:
                result = await self._execute_command_hook(hook.command, context)
            else:
                continue

            if result.action == HookAction.DENY:
                # Short-circuit: deny immediately
                return result

            if result.action == HookAction.MODIFY and result.modified_input:
                # Apply modification and continue
                context["tool_input"] = result.modified_input
                final_result = result

        return final_result

    @staticmethod
    async def _execute_command_hook(
        command: str,
        context: dict[str, Any],
    ) -> HookResult:
        """
        셸 명령어 훅을 실행합니다.

        Claude Code 규칙:
          - 종료 코드 0 → ALLOW
          - 종료 코드 2 → DENY
          - 그 외       → ALLOW (에러지만 차단하지는 않음)

        환경변수로 컨텍스트를 전달합니다:
          - HOOK_TOOL_NAME: 도구 이름
          - HOOK_TOOL_INPUT: 도구 입력 (JSON)
        """
        import json

        env = dict(__import__("os").environ)
        env["HOOK_TOOL_NAME"] = str(context.get("tool_name", ""))
        tool_input = context.get("tool_input")
        if tool_input is not None:
            env["HOOK_TOOL_INPUT"] = json.dumps(tool_input)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=30
            )

            if proc.returncode == 0:
                return HookResult(action=HookAction.ALLOW)
            elif proc.returncode == 2:
                msg = stderr.decode().strip() or stdout.decode().strip()
                return HookResult(
                    action=HookAction.DENY,
                    message=msg or f"Hook denied: {command}",
                )
            else:
                # Non-zero, non-2 exit: warn but allow
                return HookResult(
                    action=HookAction.ALLOW,
                    message=f"Hook '{command}' exited with code {proc.returncode}",
                )

        except asyncio.TimeoutError:
            return HookResult(
                action=HookAction.ALLOW,
                message=f"Hook '{command}' timed out (30s)",
            )
        except Exception as e:
            return HookResult(
                action=HookAction.ALLOW,
                message=f"Hook '{command}' failed: {e}",
            )
