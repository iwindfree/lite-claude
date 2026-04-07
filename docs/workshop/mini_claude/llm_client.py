"""
LLM API 추상화 레이어

Claude Code는 Anthropic API를 직접 사용하지만, 이 워크샵에서는
Anthropic과 OpenAI를 모두 지원하는 추상화 레이어를 만듭니다.

참조: src/services/api/claude.ts — queryModel() 함수
"""

from __future__ import annotations

import os
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Any


# ─── 통합 응답 타입 ────────────────────────────────────────────────

@dataclass
class ToolCall:
    """LLM이 요청한 도구 호출 하나를 표현"""
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class StreamEvent:
    """스트리밍 중 발생하는 이벤트"""
    type: str  # "text_delta", "tool_use", "message_start", "message_end"
    text: str = ""
    tool_call: ToolCall | None = None


@dataclass
class LLMResponse:
    """LLM 응답 하나를 표현 — 텍스트와 도구 호출을 모두 포함할 수 있음"""
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


# ─── 메시지 헬퍼 ──────────────────────────────────────────────────

def user_message(content: str) -> dict:
    return {"role": "user", "content": content}


def assistant_message(content: str) -> dict:
    return {"role": "assistant", "content": content}


def tool_result_message(tool_use_id: str, content: str) -> dict:
    """도구 실행 결과를 LLM에 전달할 메시지로 변환"""
    return {
        "role": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }


# ─── 도구 스키마 변환 ─────────────────────────────────────────────

def tool_to_anthropic_schema(name: str, description: str, parameters: dict) -> dict:
    return {
        "name": name,
        "description": description,
        "input_schema": parameters,
    }


def tool_to_openai_schema(name: str, description: str, parameters: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


# ─── 추상 클라이언트 ─────────────────────────────────────────────

class LLMClient(ABC):
    """
    LLM API 추상화. Claude Code의 queryModel() (src/services/api/claude.ts)에 대응.

    두 가지 호출 방식을 제공합니다:
    - query(): 전체 응답을 한 번에 반환
    - stream(): AsyncIterator로 스트리밍 이벤트를 하나씩 yield
    """

    @abstractmethod
    async def query(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """동기식 전체 응답"""
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamEvent]:
        """스트리밍 응답 — Claude Code의 AsyncGenerator 패턴에 대응"""
        ...


# ─── Anthropic 구현 ──────────────────────────────────────────────

class AnthropicClient(LLMClient):
    """Anthropic Claude API 클라이언트"""

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str | None = None):
        import anthropic
        self.model = model
        self.client = anthropic.AsyncAnthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    async def query(self, messages, system="", tools=None, max_tokens=4096) -> LLMResponse:
        # Anthropic API는 tool_result를 user 메시지 안에 넣음
        api_messages = _convert_messages_for_anthropic(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": api_messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        response = await self.client.messages.create(**kwargs)
        return _parse_anthropic_response(response)

    async def stream(self, messages, system="", tools=None, max_tokens=4096) -> AsyncIterator[StreamEvent]:
        api_messages = _convert_messages_for_anthropic(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": api_messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        async with self.client.messages.stream(**kwargs) as stream:
            async for event in stream:
                parsed = _parse_anthropic_stream_event(event)
                if parsed:
                    yield parsed


# ─── OpenAI 구현 ─────────────────────────────────────────────────

class OpenAIClient(LLMClient):
    """OpenAI API 클라이언트"""

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        import openai
        self.model = model
        self.client = openai.AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    async def query(self, messages, system="", tools=None, max_tokens=4096) -> LLMResponse:
        api_messages = _convert_messages_for_openai(messages, system)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": api_messages,
        }
        if tools:
            kwargs["tools"] = tools

        response = await self.client.chat.completions.create(**kwargs)
        return _parse_openai_response(response)

    async def stream(self, messages, system="", tools=None, max_tokens=4096) -> AsyncIterator[StreamEvent]:
        api_messages = _convert_messages_for_openai(messages, system)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": api_messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools

        # OpenAI 스트리밍에서 tool call은 청크로 나뉘어 옴 — 조립 필요
        tool_call_chunks: dict[int, dict] = {}

        response = await self.client.chat.completions.create(**kwargs)
        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue

            if delta.content:
                yield StreamEvent(type="text_delta", text=delta.content)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_call_chunks:
                        tool_call_chunks[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:
                        tool_call_chunks[idx]["id"] = tc.id
                    if tc.function and tc.function.name:
                        tool_call_chunks[idx]["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        tool_call_chunks[idx]["arguments"] += tc.function.arguments

            if chunk.choices and chunk.choices[0].finish_reason == "stop":
                yield StreamEvent(type="message_end")
            elif chunk.choices and chunk.choices[0].finish_reason == "tool_calls":
                # 모든 tool call 청크가 모였으므로 한번에 yield
                for tc_data in tool_call_chunks.values():
                    try:
                        args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                    except json.JSONDecodeError:
                        args = {}
                    yield StreamEvent(
                        type="tool_use",
                        tool_call=ToolCall(id=tc_data["id"], name=tc_data["name"], input=args),
                    )
                yield StreamEvent(type="message_end")


# ─── 팩토리 ──────────────────────────────────────────────────────

def create_client(provider: str = "auto", **kwargs) -> LLMClient:
    """
    API 키 환경변수를 보고 적절한 클라이언트를 생성합니다.

    provider: "anthropic", "openai", "auto" (환경변수에서 자동 감지)
    """
    if provider == "auto":
        if os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        else:
            raise ValueError(
                "API 키를 찾을 수 없습니다. "
                "ANTHROPIC_API_KEY 또는 OPENAI_API_KEY 환경변수를 설정하세요."
            )

    if provider == "anthropic":
        return AnthropicClient(**kwargs)
    elif provider == "openai":
        return OpenAIClient(**kwargs)
    else:
        raise ValueError(f"지원하지 않는 provider: {provider}")


# ─── 내부 변환 함수들 ────────────────────────────────────────────

def _convert_messages_for_anthropic(messages: list[dict]) -> list[dict]:
    """
    통합 메시지 포맷 → Anthropic API 포맷.

    Anthropic은 tool_result를 user 메시지의 content 배열 안에 넣어야 합니다.
    연속된 tool_result 메시지는 하나의 user 메시지로 합칩니다.
    """
    result = []
    i = 0
    while i < len(messages):
        msg = messages[i]

        if msg["role"] == "tool_result":
            # 연속된 tool_result를 하나의 user 메시지로 합침
            tool_results = []
            while i < len(messages) and messages[i]["role"] == "tool_result":
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": messages[i]["tool_use_id"],
                    "content": messages[i]["content"],
                })
                i += 1
            result.append({"role": "user", "content": tool_results})

        elif msg["role"] == "assistant" and "tool_calls" in msg:
            # 내부 assistant + tool_calls → Anthropic content 블록으로 변환
            content = []
            if msg.get("text"):
                content.append({"type": "text", "text": msg["text"]})
            for tc in msg["tool_calls"]:
                content.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc["input"],
                })
            result.append({"role": "assistant", "content": content})
            i += 1

        else:
            result.append(msg)
            i += 1

    return result


def _convert_messages_for_openai(messages: list[dict], system: str = "") -> list[dict]:
    """통합 메시지 포맷 → OpenAI API 포맷"""
    result = []

    if system:
        result.append({"role": "system", "content": system})

    for msg in messages:
        if msg["role"] == "tool_result":
            result.append({
                "role": "tool",
                "tool_call_id": msg["tool_use_id"],
                "content": msg["content"],
            })
        elif msg["role"] == "assistant" and "tool_calls" in msg:
            openai_msg: dict[str, Any] = {"role": "assistant", "content": msg.get("text") or None}
            openai_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["input"]),
                    },
                }
                for tc in msg["tool_calls"]
            ]
            result.append(openai_msg)
        else:
            result.append(msg)

    return result


def _parse_anthropic_response(response) -> LLMResponse:
    """Anthropic API 응답 → 통합 LLMResponse"""
    text_parts = []
    tool_calls = []

    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(id=block.id, name=block.name, input=block.input))

    return LLMResponse(
        text="".join(text_parts),
        tool_calls=tool_calls,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        stop_reason=response.stop_reason or "",
    )


def _parse_anthropic_stream_event(event) -> StreamEvent | None:
    """Anthropic 스트림 이벤트 → 통합 StreamEvent"""
    event_type = getattr(event, "type", None)

    if event_type == "content_block_delta":
        delta = event.delta
        if hasattr(delta, "text"):
            return StreamEvent(type="text_delta", text=delta.text)
    elif event_type == "content_block_start":
        block = event.content_block
        if hasattr(block, "type") and block.type == "tool_use":
            return StreamEvent(
                type="tool_use",
                tool_call=ToolCall(id=block.id, name=block.name, input={}),
            )
    elif event_type == "content_block_stop":
        pass  # 개별 블록 종료 — 무시
    elif event_type == "message_start":
        return StreamEvent(type="message_start")
    elif event_type == "message_stop":
        return StreamEvent(type="message_end")
    elif event_type == "message_delta":
        pass  # usage 업데이트 — 무시 (query()에서 처리)

    return None


def _parse_openai_response(response) -> LLMResponse:
    """OpenAI API 응답 → 통합 LLMResponse"""
    choice = response.choices[0]
    msg = choice.message
    tool_calls = []

    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, input=args))

    return LLMResponse(
        text=msg.content or "",
        tool_calls=tool_calls,
        input_tokens=response.usage.prompt_tokens if response.usage else 0,
        output_tokens=response.usage.completion_tokens if response.usage else 0,
        stop_reason=choice.finish_reason or "",
    )
