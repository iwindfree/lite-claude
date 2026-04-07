"""
에이전트 루프 — Claude Code의 심장부

Claude Code의 핵심은 src/query.ts의 queryLoop() 함수입니다.
놀랍도록 단순한 while(true) 루프가 전체 에이전트를 구동합니다:

    while (true) {
        response = await callModel(messages)
        if (response에 tool_use가 없으면) return  // 종료
        results = await runTools(response.tool_calls)
        messages = [...messages, response, ...results]
        // 다음 반복으로
    }

이 모듈은 이 패턴을 Python으로 구현합니다.

참조:
- src/query.ts:204-217  — State 타입 (불변 상태)
- src/query.ts:241-307  — queryLoop() while(true) 루프
- src/query.ts:557-558  — needsFollowUp 판단
- src/query.ts:1715-1728 — 다음 턴 state 생성
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator, Callable, Awaitable

from .llm_client import (
    LLMClient,
    LLMResponse,
    StreamEvent,
    ToolCall,
    tool_result_message,
)


# ─── State: 루프 상태를 불변 객체로 관리 ─────────────────────────
#
# Claude Code는 매 반복마다 새로운 State 객체를 생성합니다 (query.ts:1715).
# 이전 상태를 수정하지 않으므로, 에러 발생 시 이전 상태로 쉽게 돌아갈 수 있습니다.
#
# Python에서는 frozen dataclass로 이를 구현합니다.
# 상태를 변경하려면 replace()로 새 객체를 만듭니다.

@dataclass(frozen=True)
class State:
    """에이전트 루프의 불변 상태 — Claude Code의 State 타입에 대응 (query.ts:204)"""
    messages: tuple[dict, ...]  # 대화 히스토리 (불변을 위해 tuple 사용)
    turn_count: int = 1
    transition: str = ""  # 이전 반복의 종료 이유: "", "tool_use", "max_turns", "completed", "error"

    def with_messages(self, new_messages: list[dict]) -> State:
        """메시지를 추가한 새 State 반환"""
        return replace(self, messages=tuple(list(self.messages) + new_messages))

    def next_turn(self, new_messages: list[dict], transition: str = "tool_use") -> State:
        """다음 턴을 위한 새 State 반환 — query.ts:1715-1728에 대응"""
        return replace(
            self,
            messages=tuple(list(self.messages) + new_messages),
            turn_count=self.turn_count + 1,
            transition=transition,
        )


# ─── 에이전트 루프 이벤트 ────────────────────────────────────────

@dataclass
class AgentEvent:
    """에이전트 루프가 yield하는 이벤트"""
    type: str  # "text", "tool_call", "tool_result", "turn_start", "turn_end", "done"
    text: str = ""
    tool_call: ToolCall | None = None
    tool_result: str = ""
    state: State | None = None
    response: LLMResponse | None = None


# ─── 도구 실행 함수 타입 ─────────────────────────────────────────

# 도구를 실행하는 콜백. Step 2에서 정식 도구 시스템을 만들기 전에
# 간단한 딕셔너리 기반 실행기를 사용합니다.
ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[str]]


# ─── 핵심 에이전트 루프 ──────────────────────────────────────────

async def agent_loop(
    *,
    client: LLMClient,
    messages: list[dict],
    system: str = "",
    tools: list[dict] | None = None,
    tool_executor: ToolExecutor | None = None,
    max_turns: int = 20,
) -> AsyncIterator[AgentEvent]:
    """
    Claude Code의 queryLoop() (query.ts:307)에 대응하는 핵심 에이전트 루프.

    async generator로 구현하여, 호출자가 async for로 이벤트를 소비합니다.
    이것이 Claude Code가 사용하는 AsyncGenerator 패턴입니다.

    Args:
        client: LLM API 클라이언트
        messages: 초기 대화 히스토리
        system: 시스템 프롬프트
        tools: 도구 스키마 목록 (API 포맷)
        tool_executor: 도구 실행 콜백 (name, input) -> result
        max_turns: 최대 루프 반복 횟수 (무한 루프 방지)

    Yields:
        AgentEvent — 텍스트, 도구 호출, 도구 결과, 턴 시작/종료 등
    """
    # 초기 State 생성 — query.ts:268-279
    state = State(messages=tuple(messages))

    # ── while (true) — query.ts:307 ──
    while True:
        yield AgentEvent(type="turn_start", state=state)

        # 1. LLM API 호출 — query.ts:549
        response = await client.query(
            messages=list(state.messages),
            system=system,
            tools=tools,
            max_tokens=4096,
        )

        # 응답 텍스트가 있으면 yield
        if response.text:
            yield AgentEvent(type="text", text=response.text, response=response)

        # 2. tool_use가 없으면 종료 — query.ts:1062
        #    Claude Code에서는 needsFollowUp 변수로 판단합니다.
        if not response.has_tool_calls:
            final_state = replace(state, transition="completed")
            yield AgentEvent(type="done", state=final_state, response=response)
            return

        # 3. 도구 실행 — query.ts의 runTools() 호출
        #    각 tool_call에 대해 실행하고 결과를 수집합니다.
        new_messages: list[dict] = []

        # assistant 메시지 (tool_calls 포함)를 히스토리에 추가
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "text": response.text,
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "input": tc.input}
                for tc in response.tool_calls
            ],
        }
        new_messages.append(assistant_msg)

        # 각 도구 실행
        for tc in response.tool_calls:
            yield AgentEvent(type="tool_call", tool_call=tc)

            if tool_executor:
                try:
                    result = await tool_executor(tc.name, tc.input)
                except Exception as e:
                    result = f"Error: {e}"
            else:
                result = f"[도구 실행기가 없습니다: {tc.name}]"

            yield AgentEvent(type="tool_result", tool_call=tc, tool_result=result)
            new_messages.append(tool_result_message(tc.id, result))

        # 4. max_turns 확인 — query.ts:1704-1712
        if state.turn_count >= max_turns:
            final_state = state.next_turn(new_messages, transition="max_turns")
            yield AgentEvent(type="done", state=final_state)
            return

        # 5. 다음 턴으로 — query.ts:1715-1728
        #    새 State 객체를 만들어 다음 반복에 사용합니다.
        state = state.next_turn(new_messages, transition="tool_use")

        yield AgentEvent(type="turn_end", state=state)


# ─── 스트리밍 에이전트 루프 ──────────────────────────────────────

async def agent_loop_streaming(
    *,
    client: LLMClient,
    messages: list[dict],
    system: str = "",
    tools: list[dict] | None = None,
    tool_executor: ToolExecutor | None = None,
    max_turns: int = 20,
) -> AsyncIterator[AgentEvent]:
    """
    스트리밍 버전의 에이전트 루프.

    Claude Code는 API 응답을 스트리밍하면서 동시에 도구 실행을 시작합니다
    (StreamingToolExecutor, query.ts:562). 여기서는 스트리밍 텍스트 출력을
    실시간으로 yield하는 간소화된 버전을 구현합니다.
    """
    state = State(messages=tuple(messages))

    while True:
        yield AgentEvent(type="turn_start", state=state)

        # 스트리밍 API 호출
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        async for event in client.stream(
            messages=list(state.messages),
            system=system,
            tools=tools,
            max_tokens=4096,
        ):
            if event.type == "text_delta":
                text_parts.append(event.text)
                yield AgentEvent(type="text", text=event.text)
            elif event.type == "tool_use" and event.tool_call:
                tool_calls.append(event.tool_call)

        full_text = "".join(text_parts)

        # tool_use가 없으면 종료
        if not tool_calls:
            final_state = replace(state, transition="completed")
            yield AgentEvent(type="done", state=final_state)
            return

        # 도구 실행
        new_messages: list[dict] = []

        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "text": full_text,
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "input": tc.input}
                for tc in tool_calls
            ],
        }
        new_messages.append(assistant_msg)

        for tc in tool_calls:
            yield AgentEvent(type="tool_call", tool_call=tc)

            if tool_executor:
                try:
                    result = await tool_executor(tc.name, tc.input)
                except Exception as e:
                    result = f"Error: {e}"
            else:
                result = f"[도구 실행기가 없습니다: {tc.name}]"

            yield AgentEvent(type="tool_result", tool_call=tc, tool_result=result)
            new_messages.append(tool_result_message(tc.id, result))

        if state.turn_count >= max_turns:
            final_state = state.next_turn(new_messages, transition="max_turns")
            yield AgentEvent(type="done", state=final_state)
            return

        state = state.next_turn(new_messages, transition="tool_use")
        yield AgentEvent(type="turn_end", state=state)


# ─── 편의 함수: 한 번에 실행하고 최종 텍스트 반환 ────────────────

async def run_agent(
    *,
    client: LLMClient,
    prompt: str,
    system: str = "",
    tools: list[dict] | None = None,
    tool_executor: ToolExecutor | None = None,
    max_turns: int = 20,
    verbose: bool = False,
) -> str:
    """
    에이전트를 실행하고 최종 텍스트 응답을 반환하는 편의 함수.

    verbose=True이면 도구 호출/결과를 출력합니다.
    """
    messages = [{"role": "user", "content": prompt}]
    final_text = ""

    async for event in agent_loop(
        client=client,
        messages=messages,
        system=system,
        tools=tools,
        tool_executor=tool_executor,
        max_turns=max_turns,
    ):
        if event.type == "text":
            final_text = event.text
        elif event.type == "tool_call" and verbose:
            tc = event.tool_call
            print(f"  🔧 {tc.name}({tc.input})")
        elif event.type == "tool_result" and verbose:
            print(f"  ← {event.tool_result[:200]}")
        elif event.type == "turn_start" and verbose and event.state:
            print(f"  ── Turn {event.state.turn_count} ──")

    return final_text
