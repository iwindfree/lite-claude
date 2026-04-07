"""
컨텍스트 압축(Compaction) — 긴 대화의 토큰 관리

Claude Code는 대화가 길어지면 컨텍스트 윈도우를 초과할 수 있습니다.
이를 방지하기 위해 두 가지 압축 전략을 사용합니다:

1. Auto Compact (자동 압축):
   - 토큰 수가 임계값을 넘으면 LLM에게 이전 대화를 요약하도록 요청
   - 요약 결과로 이전 메시지를 대체하여 토큰 수를 줄임
   - query.ts의 shouldAutoCompact() → doAutoCompact() 흐름

2. Micro Compact (미세 압축):
   - 오래된 턴의 도구 결과를 "[cleared]"로 교체
   - LLM 호출 없이 즉시 토큰 절약
   - 최근 N턴의 결과만 유지

참조:
- src/query.ts:414-543                — shouldAutoCompact(), doAutoCompact()
- src/services/compact/prompt.ts      — BASE_COMPACT_PROMPT (요약 지시 프롬프트)
- src/services/compact/compactService.ts — compactConversation()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .llm_client import LLMClient


# ─── 토큰 추정 ─────────────────────────────────────────────────
#
# Claude Code는 정확한 토큰 카운터를 사용하지만,
# 워크숍에서는 tiktoken 또는 간단한 단어 수 기반 추정을 사용합니다.

def estimate_tokens(messages: list[dict]) -> int:
    """
    메시지 목록의 대략적 토큰 수를 추정.

    tiktoken이 설치되어 있으면 사용하고, 아니면 단어 수 기반으로 추정합니다.
    영어 기준 약 1 word ≈ 1.3 tokens, 한국어는 더 높을 수 있습니다.

    Claude Code는 Anthropic의 공식 토큰 카운터를 사용합니다.
    여기서는 워크숍용 간소화 버전입니다.
    """
    text = _messages_to_text(messages)

    # tiktoken 시도 (더 정확)
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        pass

    # fallback: 단어 수 × 1.3 (대략적 추정)
    word_count = len(text.split())
    return int(word_count * 1.3)


def _messages_to_text(messages: list[dict]) -> str:
    """메시지 목록을 하나의 텍스트로 변환 (토큰 추정용)"""
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            # content가 블록 배열인 경우 (text 블록만 추출)
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
        # tool_calls 내용도 포함
        for tc in msg.get("tool_calls", []):
            parts.append(str(tc.get("input", "")))
        # text 필드 (assistant 메시지)
        if msg.get("text"):
            parts.append(msg["text"])
    return "\n".join(parts)


# ─── Compact 프롬프트 ──────────────────────────────────────────
#
# Claude Code의 BASE_COMPACT_PROMPT (compact/prompt.ts)는
# LLM에게 대화를 요약하는 방법을 지시합니다.

COMPACT_PROMPT = """\
You are a conversation summarizer. Your task is to create a concise summary \
of the conversation so far, preserving key information needed for the assistant \
to continue helping the user.

Instructions:
- Summarize the user's original request and any clarifications
- List all files that were read, created, or modified (with paths)
- Note any important decisions, errors encountered, and their resolutions
- Preserve any pending tasks or next steps
- Keep technical details (function names, variable names, error messages) exact
- Be concise but complete — the assistant will only have this summary as context

Format your summary as a structured overview, not a conversation replay.
"""

# Claude Code에서 사용자가 커스텀 프롬프트를 추가할 수 있듯이,
# 여기서도 추가 지시를 합칠 수 있습니다.
COMPACT_PROMPT_SUFFIX = """\

If the user provided a custom focus for the summary, prioritize that topic.
"""


# ─── Auto Compact ───────────────────────────────────────────────
#
# Claude Code의 doAutoCompact() (query.ts:414-543):
# 1. 현재 토큰 수가 threshold를 넘는지 확인
# 2. 넘으면 LLM에게 이전 대화를 요약하도록 요청
# 3. 요약으로 이전 메시지를 대체

@dataclass
class CompactResult:
    """압축 결과"""
    original_tokens: int
    compacted_tokens: int
    messages: list[dict]
    summary: str = ""
    was_compacted: bool = False


async def auto_compact(
    messages: list[dict],
    client: LLMClient,
    *,
    threshold: int = 80_000,
    target_ratio: float = 0.5,
    custom_instructions: str = "",
) -> CompactResult:
    """
    자동 컨텍스트 압축 — Claude Code의 doAutoCompact() (query.ts:480)

    토큰 수가 threshold를 넘으면 LLM에게 요약을 요청하고,
    이전 메시지를 요약으로 대체합니다.

    Args:
        messages: 현재 대화 메시지
        client: LLM 클라이언트 (요약 생성용)
        threshold: 압축 시작 토큰 수 (기본 80,000)
        target_ratio: 압축 후 목표 비율 (기본 0.5 = 50%)
        custom_instructions: 사용자 커스텀 요약 지시

    Returns:
        CompactResult — 압축 결과
    """
    current_tokens = estimate_tokens(messages)

    # 임계값 미만이면 압축 불필요
    if current_tokens < threshold:
        return CompactResult(
            original_tokens=current_tokens,
            compacted_tokens=current_tokens,
            messages=messages,
            was_compacted=False,
        )

    # 요약 프롬프트 구성
    system_prompt = COMPACT_PROMPT
    if custom_instructions:
        system_prompt += f"\n\nCustom focus: {custom_instructions}"
    system_prompt += COMPACT_PROMPT_SUFFIX

    # 대화 내용을 텍스트로 변환하여 요약 요청
    conversation_text = _format_conversation_for_summary(messages)

    summary_response = await client.query(
        messages=[{
            "role": "user",
            "content": f"다음 대화를 요약해주세요:\n\n{conversation_text}",
        }],
        system=system_prompt,
        max_tokens=2048,
    )

    summary = summary_response.text or "[요약 생성 실패]"

    # 요약으로 이전 메시지 대체
    # 마지막 사용자 메시지는 유지하고, 나머지를 요약으로 대체
    compacted_messages = [
        {
            "role": "user",
            "content": (
                f"[이전 대화 요약]\n{summary}\n\n"
                "[현재 대화 계속]"
            ),
        },
    ]

    # 마지막 사용자 메시지가 있으면 추가
    last_user_msg = _find_last_user_message(messages)
    if last_user_msg:
        compacted_messages.append(last_user_msg)

    compacted_tokens = estimate_tokens(compacted_messages)

    return CompactResult(
        original_tokens=current_tokens,
        compacted_tokens=compacted_tokens,
        messages=compacted_messages,
        summary=summary,
        was_compacted=True,
    )


def _format_conversation_for_summary(messages: list[dict]) -> str:
    """대화를 요약용 텍스트로 포맷팅"""
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        text = msg.get("text", "")

        if isinstance(content, str) and content:
            lines.append(f"[{role}] {content}")
        elif text:
            lines.append(f"[{role}] {text}")

        # 도구 호출 정보
        for tc in msg.get("tool_calls", []):
            lines.append(f"  -> tool: {tc.get('name', '?')}({tc.get('input', '')})")

    return "\n".join(lines)


def _find_last_user_message(messages: list[dict]) -> dict | None:
    """마지막 사용자 메시지를 찾아 반환"""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg
    return None


# ─── Micro Compact ──────────────────────────────────────────────
#
# LLM 호출 없이 오래된 도구 결과를 "[cleared]"로 교체하여
# 즉시 토큰을 절약하는 간단한 전략입니다.
#
# Claude Code에서는 도구 결과가 매우 클 수 있으므로 (파일 내용 등)
# 오래된 결과를 제거하면 상당한 토큰 절약이 가능합니다.

CLEARED_MARKER = "[cleared]"


def micro_compact(
    messages: list[dict],
    *,
    max_age_turns: int = 6,
) -> CompactResult:
    """
    미세 압축 — 오래된 도구 결과를 "[cleared]"로 교체

    LLM 호출 없이 즉시 토큰을 절약합니다.
    최근 max_age_turns 턴의 도구 결과만 유지합니다.

    Args:
        messages: 현재 대화 메시지
        max_age_turns: 유지할 최근 턴 수 (기본 6)

    Returns:
        CompactResult — 압축 결과
    """
    original_tokens = estimate_tokens(messages)

    # 턴 경계 파악: assistant 메시지가 하나의 턴
    turn_boundaries = _find_turn_boundaries(messages)
    total_turns = len(turn_boundaries)

    if total_turns <= max_age_turns:
        return CompactResult(
            original_tokens=original_tokens,
            compacted_tokens=original_tokens,
            messages=messages,
            was_compacted=False,
        )

    # 오래된 턴의 메시지 인덱스 수집
    old_turn_count = total_turns - max_age_turns
    old_message_indices: set[int] = set()
    for i, (start, end) in enumerate(turn_boundaries):
        if i < old_turn_count:
            old_message_indices.update(range(start, end + 1))

    # 메시지 복사하면서 오래된 tool_result를 [cleared]로 교체
    compacted: list[dict] = []
    for idx, msg in enumerate(messages):
        if idx in old_message_indices and msg.get("role") == "tool":
            # tool result 메시지를 [cleared]로 교체
            compacted.append({
                **msg,
                "content": CLEARED_MARKER,
            })
        elif idx in old_message_indices and "tool_calls" in msg:
            # assistant의 tool_call은 유지하되, 큰 입력은 축약
            compacted.append(msg)  # tool_call 자체는 보통 작음
        else:
            compacted.append(msg)

    compacted_tokens = estimate_tokens(compacted)

    return CompactResult(
        original_tokens=original_tokens,
        compacted_tokens=compacted_tokens,
        messages=compacted,
        was_compacted=compacted_tokens < original_tokens,
    )


def _find_turn_boundaries(messages: list[dict]) -> list[tuple[int, int]]:
    """
    턴 경계를 찾아 (시작 인덱스, 끝 인덱스) 튜플 목록으로 반환.

    하나의 턴 = assistant 메시지 + 뒤따르는 tool result 메시지들
    """
    boundaries: list[tuple[int, int]] = []
    i = 0
    while i < len(messages):
        if messages[i].get("role") == "assistant":
            start = i
            end = i
            # 뒤따르는 tool result 메시지들을 같은 턴으로
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                end = j
                j += 1
            boundaries.append((start, end))
            i = j
        else:
            i += 1
    return boundaries


# ─── 압축 필요성 판단 ──────────────────────────────────────────

def should_auto_compact(
    messages: list[dict],
    *,
    threshold: int = 80_000,
) -> bool:
    """
    자동 압축이 필요한지 판단 — Claude Code의 shouldAutoCompact() (query.ts:414)

    현재 토큰 수가 threshold를 넘으면 True.
    """
    return estimate_tokens(messages) >= threshold


def should_micro_compact(
    messages: list[dict],
    *,
    max_age_turns: int = 6,
) -> bool:
    """미세 압축이 가능한지 판단"""
    turn_boundaries = _find_turn_boundaries(messages)
    return len(turn_boundaries) > max_age_turns
