"""
불변 상태 관리 — agent_loop.py의 State를 확장한 전체 상태 모델

Claude Code의 query.ts에서 State는 에이전트 루프의 모든 상태를 담는 불변 객체입니다.
agent_loop.py에서는 간소화된 State를 사용했지만, 실제 Claude Code에는 더 많은 필드가 있습니다:

    - 전이 상태 (Transition): 왜 이 턴이 끝났는가?
    - 복구 카운터: max output 복구, 에러 복구 횟수
    - 컴팩션 추적: 마지막 컴팩션 시점, 횟수
    - 스트리밍 상태: 현재 스트리밍 중인지

이 모듈은 agent_loop.py의 State를 확장하여 Claude Code의 전체 State 타입을 구현합니다.

참조:
- src/query.ts:204-217          — State 타입
- src/query/transitions.ts      — Transition enum, 상태 전이 로직
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any


# ─── Transition: 상태 전이 이유 ─────────────────────────────────
#
# Claude Code의 query/transitions.ts에서 정의된 전이 유형.
# 에이전트 루프가 매 턴 끝에 "왜 이 턴이 끝났는가"를 기록합니다.
# 이 정보는 다음 턴의 동작을 결정하는 데 사용됩니다.

class Transition(Enum):
    """
    상태 전이 유형 — Claude Code의 Transition enum (transitions.ts)

    각 값은 에이전트 루프의 한 반복이 끝난 이유를 나타냅니다.
    """
    COMPLETED = "completed"
    """LLM이 도구 호출 없이 텍스트만 반환 — 작업 완료"""

    TOOL_USE = "tool_use"
    """LLM이 도구를 호출함 — 다음 턴에서 도구 결과를 전달해야 함"""

    MAX_OUTPUT_RECOVERY = "max_output_recovery"
    """LLM 응답이 max_tokens에 도달하여 잘림 — 자동으로 이어서 생성"""

    REACTIVE_COMPACT = "reactive_compact"
    """컨텍스트 윈도우 초과로 자동 압축 수행됨 — 압축 후 재시도"""

    ABORTED = "aborted"
    """사용자가 작업을 중단함 (Ctrl+C 등)"""

    MAX_TURNS = "max_turns"
    """최대 턴 수에 도달 — 무한 루프 방지"""

    ERROR = "error"
    """API 오류 또는 도구 실행 오류 — 복구 시도 또는 종료"""


# ─── AgentState: 전체 에이전트 상태 ─────────────────────────────
#
# agent_loop.py의 State를 확장한 전체 버전.
# frozen=True로 불변성을 보장하며, 상태 변경은 항상 새 객체를 생성합니다.

@dataclass(frozen=True)
class AgentState:
    """
    에이전트 루프의 전체 불변 상태 — Claude Code의 State 타입 (query.ts:204-217)

    agent_loop.py의 간소화된 State와 달리, Claude Code의 모든 상태 필드를 포함합니다.
    frozen dataclass이므로 모든 '변경'은 replace()로 새 객체를 만듭니다.
    """

    # ── 대화 히스토리 ──
    messages: tuple[dict, ...] = ()
    """대화 메시지 (불변을 위해 tuple 사용)"""

    # ── 턴 관리 ──
    turn_count: int = 0
    """현재 턴 번호 (0부터 시작)"""

    max_turns: int = 20
    """최대 허용 턴 수"""

    # ── 전이 상태 ──
    transition: Transition | None = None
    """마지막 전이 이유 — 이전 턴이 왜 끝났는가"""

    # ── 복구 카운터 ──
    max_output_recovery_count: int = 0
    """max_tokens 도달로 인한 복구 횟수"""

    error_recovery_count: int = 0
    """API 에러로 인한 복구 횟수"""

    MAX_RECOVERY_ATTEMPTS: int = 3
    """최대 복구 시도 횟수 (이를 넘으면 종료)"""

    # ── 컴팩션 추적 ──
    compact_count: int = 0
    """자동 컴팩션 수행 횟수"""

    last_compact_turn: int = -1
    """마지막 컴팩션이 수행된 턴 번호"""

    # ── 스트리밍 ──
    is_streaming: bool = False
    """현재 LLM 응답을 스트리밍 중인지"""

    # ── 메타데이터 ──
    session_id: str = ""
    """세션 식별자"""

    abort_requested: bool = False
    """사용자가 중단을 요청했는지"""

    # ── 상태 변경 메서드 (새 객체 반환) ──

    def with_messages(self, new_messages: list[dict]) -> AgentState:
        """메시지를 추가한 새 상태 반환"""
        return replace(
            self,
            messages=tuple(list(self.messages) + new_messages),
        )

    def next_turn(
        self,
        new_messages: list[dict],
        transition: Transition = Transition.TOOL_USE,
    ) -> AgentState:
        """
        다음 턴으로 전이 — query.ts:1715-1728

        메시지를 추가하고, 턴 카운터를 증가시키고, 전이 이유를 기록합니다.
        """
        return replace(
            self,
            messages=tuple(list(self.messages) + new_messages),
            turn_count=self.turn_count + 1,
            transition=transition,
        )

    def with_compact(self, compacted_messages: list[dict]) -> AgentState:
        """컴팩션 수행 후 새 상태 반환"""
        return replace(
            self,
            messages=tuple(compacted_messages),
            compact_count=self.compact_count + 1,
            last_compact_turn=self.turn_count,
            transition=Transition.REACTIVE_COMPACT,
        )

    def with_recovery(self, recovery_type: Transition) -> AgentState:
        """복구 시도 후 새 상태 반환"""
        if recovery_type == Transition.MAX_OUTPUT_RECOVERY:
            return replace(
                self,
                max_output_recovery_count=self.max_output_recovery_count + 1,
                transition=recovery_type,
            )
        elif recovery_type == Transition.ERROR:
            return replace(
                self,
                error_recovery_count=self.error_recovery_count + 1,
                transition=recovery_type,
            )
        return replace(self, transition=recovery_type)

    def with_abort(self) -> AgentState:
        """중단 요청 후 새 상태 반환"""
        return replace(
            self,
            abort_requested=True,
            transition=Transition.ABORTED,
        )

    # ── 상태 질의 메서드 ──

    @property
    def is_completed(self) -> bool:
        """작업이 완료되었는가"""
        return self.transition == Transition.COMPLETED

    @property
    def is_aborted(self) -> bool:
        """작업이 중단되었는가"""
        return self.transition == Transition.ABORTED or self.abort_requested

    @property
    def has_reached_max_turns(self) -> bool:
        """최대 턴 수에 도달했는가"""
        return self.turn_count >= self.max_turns

    @property
    def can_recover_max_output(self) -> bool:
        """max_output 복구가 가능한가 (최대 횟수 미만)"""
        return self.max_output_recovery_count < self.MAX_RECOVERY_ATTEMPTS

    @property
    def can_recover_error(self) -> bool:
        """에러 복구가 가능한가 (최대 횟수 미만)"""
        return self.error_recovery_count < self.MAX_RECOVERY_ATTEMPTS

    @property
    def should_continue(self) -> bool:
        """에이전트 루프가 계속되어야 하는가"""
        if self.is_completed or self.is_aborted:
            return False
        if self.has_reached_max_turns:
            return False
        return True


# ─── 상태 전이 로직 ─────────────────────────────────────────────
#
# Claude Code의 transitions.ts에서 정의된 전이 결정 로직.
# "이 상황에서 다음에 무엇을 해야 하는가?"를 결정합니다.

def determine_transition(
    *,
    has_tool_calls: bool,
    was_truncated: bool = False,
    had_error: bool = False,
    abort_requested: bool = False,
    state: AgentState,
) -> Transition:
    """
    다음 전이를 결정 — Claude Code의 transitions.ts 로직

    Args:
        has_tool_calls: LLM이 도구를 호출했는가
        was_truncated: 응답이 max_tokens에서 잘렸는가
        had_error: 에러가 발생했는가
        abort_requested: 사용자가 중단을 요청했는가
        state: 현재 상태

    Returns:
        다음 Transition
    """
    # 중단 요청 확인
    if abort_requested or state.abort_requested:
        return Transition.ABORTED

    # 최대 턴 확인
    if state.has_reached_max_turns:
        return Transition.MAX_TURNS

    # 에러 처리
    if had_error:
        if state.can_recover_error:
            return Transition.ERROR  # 복구 시도
        return Transition.ABORTED  # 복구 불가 → 중단

    # max_output 잘림 처리
    if was_truncated:
        if state.can_recover_max_output:
            return Transition.MAX_OUTPUT_RECOVERY
        return Transition.COMPLETED  # 복구 불가 → 현재까지의 결과로 완료

    # 도구 호출 여부
    if has_tool_calls:
        return Transition.TOOL_USE

    # 도구 호출 없음 → 완료
    return Transition.COMPLETED


# ─── 초기 상태 생성 ─────────────────────────────────────────────

def create_initial_state(
    messages: list[dict],
    *,
    max_turns: int = 20,
    session_id: str = "",
) -> AgentState:
    """
    초기 AgentState 생성 — query.ts:268-279의 초기화 로직

    Args:
        messages: 초기 대화 메시지
        max_turns: 최대 턴 수
        session_id: 세션 ID

    Returns:
        초기 AgentState
    """
    return AgentState(
        messages=tuple(messages),
        max_turns=max_turns,
        session_id=session_id,
    )
