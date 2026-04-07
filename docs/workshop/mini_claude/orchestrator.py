"""
도구 실행 오케스트레이터 — 직렬/병렬 자동 결정

Claude Code의 핵심 오케스트레이션 로직:
1. LLM이 한 턴에 여러 도구를 호출할 수 있음
2. 읽기전용 + 병렬안전한 도구는 동시에 실행 (빠름)
3. 쓰기 가능한 도구는 하나씩 순차 실행 (안전)

이 분류는 각 도구의 is_concurrency_safe() 반환값으로 결정됩니다.

참조:
- src/services/tools/toolOrchestration.ts:19-82   — runTools()
- src/services/tools/toolOrchestration.ts:91-116   — partitionToolCalls()
- src/services/tools/toolExecution.ts:337-390       — runToolUse()
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from .tool_base import Tool, ToolResult
from .tool_registry import ToolRegistry
from .llm_client import ToolCall, tool_result_message


# ─── 도구 실행 결과 ──────────────────────────────────────────────

@dataclass
class ToolExecutionResult:
    """단일 도구 실행의 결과"""
    tool_call: ToolCall
    result: str
    elapsed_ms: float
    was_parallel: bool = False


# ─── 배치 분류 ───────────────────────────────────────────────────

@dataclass
class Batch:
    """
    도구 호출의 배치 — Claude Code의 Batch 타입 (toolOrchestration.ts:84)

    is_concurrent: True이면 이 배치의 도구들을 동시 실행
    calls: 이 배치에 속하는 도구 호출들
    """
    is_concurrent: bool
    calls: list[ToolCall]


def partition_tool_calls(
    tool_calls: list[ToolCall],
    registry: ToolRegistry,
) -> list[Batch]:
    """
    도구 호출을 배치로 분류 — Claude Code의 partitionToolCalls() (toolOrchestration.ts:91)

    연속된 병렬안전 도구는 하나의 배치로 합치고,
    비안전 도구는 각각 별도 배치로 만듭니다.

    예시: [Read, Read, Bash, Read] → [[Read, Read](병렬), [Bash](직렬), [Read](병렬)]
    """
    batches: list[Batch] = []

    for tc in tool_calls:
        tool = registry.find_by_name(tc.name)
        is_safe = tool.is_concurrency_safe(tc.input) if tool else False

        # 이전 배치와 같은 종류(병렬)이면 합침
        if is_safe and batches and batches[-1].is_concurrent:
            batches[-1].calls.append(tc)
        else:
            batches.append(Batch(is_concurrent=is_safe, calls=[tc]))

    return batches


# ─── 단일 도구 실행 ──────────────────────────────────────────────

async def execute_single_tool(
    tc: ToolCall,
    registry: ToolRegistry,
) -> ToolExecutionResult:
    """
    단일 도구 실행 — Claude Code의 runToolUse() (toolExecution.ts:337)의 간소화 버전

    전체 라이프사이클:
    1. 도구 찾기 (findToolByName)
    2. 입력 검증 (validateInput)
    3. 실행 (call)
    4. 결과 크기 제한
    """
    start = time.monotonic()

    tool = registry.find_by_name(tc.name)
    if not tool:
        return ToolExecutionResult(
            tool_call=tc,
            result=f"Error: 알 수 없는 도구 '{tc.name}'",
            elapsed_ms=(time.monotonic() - start) * 1000,
        )

    # 입력 검증
    error = await tool.validate_input(tc.input)
    if error:
        return ToolExecutionResult(
            tool_call=tc,
            result=f"Error: {error}",
            elapsed_ms=(time.monotonic() - start) * 1000,
        )

    # 실행
    try:
        result = await tool.call(tc.input)
        content = result.content

        # 결과 크기 제한 (maxResultSizeChars)
        if len(content) > tool.max_result_size:
            content = content[:tool.max_result_size] + "\n... [결과가 잘렸습니다]"

        return ToolExecutionResult(
            tool_call=tc,
            result=content,
            elapsed_ms=(time.monotonic() - start) * 1000,
        )
    except Exception as e:
        return ToolExecutionResult(
            tool_call=tc,
            result=f"Error: {type(e).__name__}: {e}",
            elapsed_ms=(time.monotonic() - start) * 1000,
        )


# ─── 오케스트레이터 ──────────────────────────────────────────────

async def run_tools(
    tool_calls: list[ToolCall],
    registry: ToolRegistry,
    max_concurrency: int = 10,
) -> list[ToolExecutionResult]:
    """
    도구 호출을 오케스트레이션 — Claude Code의 runTools() (toolOrchestration.ts:19)

    1. partitionToolCalls()로 배치 분류
    2. 병렬 배치는 asyncio.gather()로 동시 실행
    3. 직렬 배치는 하나씩 순차 실행

    Args:
        tool_calls: LLM이 요청한 도구 호출 목록
        registry: 도구 레지스트리
        max_concurrency: 최대 동시 실행 수 (기본 10, Claude Code와 동일)
    """
    batches = partition_tool_calls(tool_calls, registry)
    all_results: list[ToolExecutionResult] = []

    for batch in batches:
        if batch.is_concurrent and len(batch.calls) > 1:
            # 병렬 실행 — Claude Code의 runToolsConcurrently()
            semaphore = asyncio.Semaphore(max_concurrency)

            async def _run_with_limit(tc: ToolCall) -> ToolExecutionResult:
                async with semaphore:
                    result = await execute_single_tool(tc, registry)
                    result.was_parallel = True
                    return result

            results = await asyncio.gather(
                *[_run_with_limit(tc) for tc in batch.calls]
            )
            all_results.extend(results)
        else:
            # 직렬 실행 — Claude Code의 runToolsSerially()
            for tc in batch.calls:
                result = await execute_single_tool(tc, registry)
                all_results.append(result)

    return all_results


def results_to_messages(results: list[ToolExecutionResult]) -> list[dict]:
    """도구 실행 결과를 API tool_result 메시지로 변환"""
    return [
        tool_result_message(r.tool_call.id, r.result)
        for r in results
    ]


# ─── 에이전트 루프 통합용 executor ───────────────────────────────

def create_orchestrated_executor(
    registry: ToolRegistry,
    max_concurrency: int = 10,
):
    """
    오케스트레이터를 에이전트 루프의 tool_executor로 사용할 수 있게 래핑.

    반환되는 함수는 (name, input) -> str 시그니처로,
    agent_loop의 tool_executor 인터페이스에 맞습니다.

    단, 이 래퍼는 개별 도구를 하나씩 실행합니다.
    전체 배치 오케스트레이션은 Step 3 이후 에이전트 루프를 수정할 때 적용합니다.
    """
    async def executor(name: str, args: dict[str, Any]) -> str:
        return await registry.execute(name, args)
    return executor
