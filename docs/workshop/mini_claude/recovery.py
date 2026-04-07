"""
에러 복구 전략 — 장애에 우아하게 대응하는 패턴들

Claude Code는 다양한 에러 상황에 대해 자동 복구를 시도합니다:
- API 에러 → 지수 백오프로 재시도
- max_output_tokens 초과 → 토큰 한도 증가 또는 "continue" 주입
- prompt_too_long → 컨텍스트 컴팩션 트리거
- Ctrl+C → 현재 작업만 중단하고 REPL은 유지

이 모듈은 이러한 복구 전략들을 Python으로 구현합니다.

참조:
- src/query.ts:1085-1252 — 에러 핸들링 및 복구 로직
"""

from __future__ import annotations

import asyncio
import functools
import random
import signal
import time
from dataclasses import dataclass
from typing import Any, Callable, Awaitable, TypeVar

T = TypeVar("T")


# ─── 재시도 설정 ─────────────────────────────────────────────────

@dataclass
class RetryConfig:
    """
    재시도 설정 — 지수 백오프 파라미터

    max_retries: 최대 재시도 횟수
    base_delay: 첫 번째 재시도 대기 시간 (초)
    max_delay: 최대 대기 시간 (초)
    jitter: True이면 대기 시간에 랜덤 지터 추가 (thundering herd 방지)
    """
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: bool = True

    def delay_for_attempt(self, attempt: int) -> float:
        """attempt번째 재시도의 대기 시간을 계산합니다 (지수 백오프)."""
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        if self.jitter:
            delay = delay * (0.5 + random.random() * 0.5)
        return delay


# ─── 지수 백오프 재시도 ─────────────────────────────────────────

async def retry_with_backoff(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    config: RetryConfig | None = None,
    retryable_errors: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[int, Exception, float], Awaitable[None] | None] | None = None,
    **kwargs: Any,
) -> T:
    """
    지수 백오프로 비동기 함수를 재시도합니다.

    Claude Code의 API 호출 재시도 로직(query.ts:1085+)에 대응합니다.
    네트워크 에러, 429 Rate Limit, 서버 에러(5xx) 등에 사용됩니다.

    Args:
        func: 재시도할 비동기 함수
        config: 재시도 설정 (None이면 기본값 사용)
        retryable_errors: 재시도할 예외 타입들
        on_retry: 재시도 시 호출되는 콜백 (attempt, error, delay)

    Returns:
        func의 반환값

    Raises:
        마지막 시도에서 발생한 예외
    """
    if config is None:
        config = RetryConfig()

    last_error: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except retryable_errors as e:
            last_error = e

            if attempt >= config.max_retries:
                break

            delay = config.delay_for_attempt(attempt)

            if on_retry:
                result = on_retry(attempt + 1, e, delay)
                if asyncio.iscoroutine(result):
                    await result

            await asyncio.sleep(delay)

    raise last_error  # type: ignore[misc]


def with_retry(
    config: RetryConfig | None = None,
    retryable_errors: tuple[type[Exception], ...] = (Exception,),
):
    """
    재시도를 적용하는 데코레이터.

    사용법:
        @with_retry(config=RetryConfig(max_retries=5))
        async def call_api():
            ...
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await retry_with_backoff(
                func, *args,
                config=config,
                retryable_errors=retryable_errors,
                **kwargs,
            )
        return wrapper
    return decorator


# ─── max_output_tokens 복구 ─────────────────────────────────────

@dataclass
class MaxTokensRecovery:
    """
    max_output_tokens 초과 시 복구 전략.

    Claude Code는 응답이 max_tokens에서 잘리면:
    1. 먼저 토큰 한도를 증가시켜 재시도
    2. 그래도 잘리면 "Please continue" 메시지를 주입하여 이어서 응답하게 함

    참조: query.ts:1147-1195
    """
    initial_max_tokens: int = 4096
    max_tokens_limit: int = 16384
    increase_factor: float = 2.0
    _current_max_tokens: int = 0

    def __post_init__(self):
        self._current_max_tokens = self.initial_max_tokens

    @property
    def current_max_tokens(self) -> int:
        return self._current_max_tokens


def handle_max_output_tokens(
    stop_reason: str,
    current_max_tokens: int,
    max_limit: int = 16384,
) -> dict[str, Any]:
    """
    max_output_tokens 초과 시 복구 액션을 결정합니다.

    Returns:
        {
            "action": "increase_limit" | "inject_continue" | "none",
            "new_max_tokens": int,  # increase_limit일 때
            "continue_message": dict,  # inject_continue일 때
        }
    """
    if stop_reason != "max_tokens":
        return {"action": "none"}

    # 한도를 늘릴 수 있으면 늘림
    new_limit = min(current_max_tokens * 2, max_limit)
    if new_limit > current_max_tokens:
        return {
            "action": "increase_limit",
            "new_max_tokens": new_limit,
        }

    # 더 이상 늘릴 수 없으면 "continue" 메시지 주입
    return {
        "action": "inject_continue",
        "continue_message": {
            "role": "user",
            "content": "계속 이어서 작성해주세요. (이전 응답이 길이 제한으로 잘렸습니다)",
        },
    }


# ─── prompt_too_long 복구 ───────────────────────────────────────

def handle_prompt_too_long(
    messages: list[dict],
    max_context_tokens: int = 100_000,
    estimated_tokens: int = 0,
) -> dict[str, Any]:
    """
    프롬프트가 너무 길 때 복구 액션을 결정합니다.

    Claude Code에서는 이 상황에서 컨텍스트 컴팩션을 트리거합니다.
    컴팩션은 대화 히스토리를 요약하여 토큰 수를 줄이는 과정입니다.

    Returns:
        {
            "action": "compact" | "none",
            "message_count": int,
            "estimated_tokens": int,
        }
    """
    if estimated_tokens <= max_context_tokens:
        return {"action": "none"}

    return {
        "action": "compact",
        "message_count": len(messages),
        "estimated_tokens": estimated_tokens,
    }


# ─── Graceful Shutdown ──────────────────────────────────────────

class GracefulShutdown:
    """
    Ctrl+C를 우아하게 처리하는 클래스.

    Claude Code에서 Ctrl+C는:
    - 1번 누름: 현재 도구 실행만 중단 (에이전트 루프 유지)
    - 2번 누름: 에이전트 루프 중단 (REPL은 유지)
    - 3번 누름: 프로세스 종료

    이 구현에서는 asyncio.Event를 사용하여 비동기적으로 처리합니다.

    사용법:
        shutdown = GracefulShutdown()
        shutdown.install()

        # 에이전트 루프에서
        if shutdown.is_requested:
            break  # 루프 종료

        # REPL에서
        if shutdown.is_forced:
            sys.exit(0)
    """

    def __init__(self):
        self._interrupt_count = 0
        self._event = asyncio.Event()
        self._force_event = asyncio.Event()
        self._last_interrupt_time: float = 0
        self._installed = False

    def install(self) -> None:
        """시그널 핸들러를 설치합니다."""
        if self._installed:
            return

        loop = asyncio.get_event_loop()

        def _handler():
            now = time.monotonic()
            # 2초 이내의 연속 Ctrl+C를 카운트
            if now - self._last_interrupt_time > 2.0:
                self._interrupt_count = 0
            self._last_interrupt_time = now
            self._interrupt_count += 1

            if self._interrupt_count == 1:
                print("\n[인터럽트] 현재 작업을 중단합니다... (한번 더 누르면 루프 종료)")
                self._event.set()
            elif self._interrupt_count == 2:
                print("\n[인터럽트] 에이전트 루프를 종료합니다... (한번 더 누르면 프로세스 종료)")
                self._force_event.set()
            else:
                print("\n[인터럽트] 프로세스를 종료합니다.")
                raise SystemExit(130)

        try:
            loop.add_signal_handler(signal.SIGINT, _handler)
            self._installed = True
        except NotImplementedError:
            # Windows에서는 add_signal_handler가 지원되지 않을 수 있음
            signal.signal(signal.SIGINT, lambda s, f: _handler())
            self._installed = True

    def reset(self) -> None:
        """인터럽트 상태를 초기화합니다. 새 턴을 시작할 때 호출합니다."""
        self._interrupt_count = 0
        self._event.clear()
        self._force_event.clear()

    @property
    def is_requested(self) -> bool:
        """인터럽트가 요청되었는지 (1번 이상 Ctrl+C)"""
        return self._event.is_set()

    @property
    def is_forced(self) -> bool:
        """강제 종료가 요청되었는지 (2번 이상 Ctrl+C)"""
        return self._force_event.is_set()

    async def wait_for_interrupt(self) -> None:
        """인터럽트가 발생할 때까지 대기합니다."""
        await self._event.wait()
