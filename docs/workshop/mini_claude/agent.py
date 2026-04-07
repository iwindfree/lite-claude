"""
통합 Agent 클래스 — 모든 서브시스템을 하나로 연결

이 모듈은 Claude Code의 전체 파이프라인을 하나의 Agent 클래스로 통합합니다.
각 서브시스템(LLM 클라이언트, 도구, 스킬, 브리지, 복구)을 초기화하고,
run() 메서드로 전체 파이프라인을 실행합니다.

실행 파이프라인:
1. 컨텍스트 구성 (시스템 프롬프트 + 도구 목록)
2. 컨텍스트 크기 확인 → 필요시 컴팩션
3. LLM API 호출
4. 응답 처리 (텍스트 출력, 도구 호출)
5. 권한 확인 (필요시)
6. 도구 실행
7. 상태 업데이트
8. 에러 복구 (필요시)
9. 반복 또는 종료

참조:
- src/entrypoints/cli.tsx — CLI 진입점
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .llm_client import LLMClient, create_client
from .agent_loop import agent_loop, AgentEvent, run_agent
from .tool_base import Tool
from .tool_registry import ToolRegistry
from .orchestrator import create_orchestrated_executor
from .skills.loader import SkillRegistry, create_default_registry as create_default_skill_registry
from .tools.skill_tool import SkillTool
from .tools.agent_tool import AgentTool, AgentRunner
from .recovery import (
    RetryConfig,
    GracefulShutdown,
    retry_with_backoff,
    handle_max_output_tokens,
)


# ─── 에이전트 설정 ──────────────────────────────────────────────

@dataclass
class AgentConfig:
    """
    에이전트 설정.

    YAML 파일에서 로드하거나 직접 생성할 수 있습니다.
    """
    # LLM 설정
    provider: str = "auto"
    model: str | None = None
    max_tokens: int = 4096

    # 시스템 프롬프트
    system_prompt: str = (
        "당신은 유능한 AI 코딩 어시스턴트입니다. "
        "사용자의 요청에 따라 코드를 읽고, 분석하고, 수정할 수 있습니다. "
        "도구를 사용하여 파일 시스템과 상호작용하세요."
    )

    # 에이전트 루프
    max_turns: int = 30

    # 복구
    retry_config: RetryConfig = field(default_factory=RetryConfig)

    # 스킬
    custom_skills_dir: str | None = None

    # 브리지
    bridge_enabled: bool = False
    bridge_host: str = "localhost"
    bridge_port: int = 8765


def load_config(path: str | Path) -> AgentConfig:
    """
    YAML 파일에서 설정을 로드합니다.

    파일 형식:
        provider: anthropic
        model: claude-sonnet-4-20250514
        max_turns: 30
        system_prompt: "..."
    """
    path = Path(path)
    if not path.exists():
        return AgentConfig()

    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except ImportError:
        # yaml이 없으면 기본 설정 반환
        return AgentConfig()

    config = AgentConfig()
    for key, value in data.items():
        if hasattr(config, key):
            if key == "retry_config" and isinstance(value, dict):
                setattr(config, key, RetryConfig(**value))
            else:
                setattr(config, key, value)
    return config


# ─── Agent 클래스 ────────────────────────────────────────────────

class Agent:
    """
    통합 에이전트 — Claude Code의 전체 파이프라인을 하나로

    모든 서브시스템을 초기화하고, run()으로 에이전트를 실행하고,
    repl()로 대화형 세션을 시작합니다.

    사용법:
        agent = Agent()
        # 또는 agent = Agent(config_path="config.yaml")
        # 또는 agent = Agent(provider="anthropic", model="claude-sonnet-4-20250514")

        # 단일 실행
        result = await agent.run("현재 디렉토리의 파일 목록을 보여주세요")

        # 대화형 REPL
        await agent.repl()
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        **kwargs: Any,
    ):
        # 설정 로드
        if config_path:
            self.config = load_config(config_path)
        else:
            self.config = AgentConfig()

        # kwargs로 설정 오버라이드
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # 서브시스템 초기화
        self._client: LLMClient | None = None
        self._tool_registry = ToolRegistry()
        self._skill_registry = create_default_skill_registry()
        self._shutdown = GracefulShutdown()

        # 기본 도구 등록
        self._register_default_tools()

        # 커스텀 스킬 로드
        if self.config.custom_skills_dir:
            self._skill_registry.load_from_dir(self.config.custom_skills_dir)

    def _get_client(self) -> LLMClient:
        """LLM 클라이언트를 지연 초기화하여 반환합니다."""
        if self._client is None:
            client_kwargs: dict[str, Any] = {}
            if self.config.model:
                client_kwargs["model"] = self.config.model
            self._client = create_client(provider=self.config.provider, **client_kwargs)
        return self._client

    def _register_default_tools(self) -> None:
        """기본 도구들을 등록합니다."""
        # 기존 빌트인 도구들 (bash, file_read, file_write)
        try:
            from .tools.bash_tool import BashTool
            self._tool_registry.register(BashTool)
        except ImportError:
            pass

        try:
            from .tools.file_read_tool import FileReadTool
            self._tool_registry.register(FileReadTool)
        except ImportError:
            pass

        try:
            from .tools.file_write_tool import FileWriteTool
            self._tool_registry.register(FileWriteTool)
        except ImportError:
            pass

        # 스킬 도구
        skill_tool = SkillTool(self._skill_registry)
        self._tool_registry.register(skill_tool)

        # 에이전트 도구 (서브 에이전트 runner는 자기 자신의 run 사용)
        agent_tool = AgentTool(
            runner=self._sub_agent_runner,
            parent_system=self.config.system_prompt,
        )
        self._tool_registry.register(agent_tool)

    async def _sub_agent_runner(
        self,
        prompt: str,
        system: str,
        tools: list[str] | None,
        model: str | None,
    ) -> str:
        """서브 에이전트용 runner — 자기 자신의 에이전트 루프를 재활용합니다."""
        client = self._get_client()

        # 도구 필터링
        if tools:
            tool_schemas = [
                t for t in self._tool_registry.to_anthropic_schemas()
                if t["name"] in tools
            ]
        else:
            tool_schemas = self._tool_registry.to_anthropic_schemas()

        executor = create_orchestrated_executor(self._tool_registry)

        result = await run_agent(
            client=client,
            prompt=prompt,
            system=system,
            tools=tool_schemas if tool_schemas else None,
            tool_executor=executor,
            max_turns=self.config.max_turns // 2,  # 서브 에이전트는 턴 제한을 절반으로
        )
        return result

    def register_tool(self, tool: Tool) -> None:
        """커스텀 도구를 등록합니다."""
        self._tool_registry.register(tool)

    # ── 실행 ─────────────────────────────────────────────────────

    async def run(self, prompt: str, *, verbose: bool = False) -> str:
        """
        에이전트를 실행하고 최종 응답을 반환합니다.

        전체 파이프라인:
        1. 컨텍스트 구성
        2. 에이전트 루프 실행 (API 호출 → 도구 실행 → 반복)
        3. 에러 복구 (재시도, max_tokens 처리)
        4. 최종 텍스트 반환
        """
        client = self._get_client()
        tool_schemas = self._tool_registry.to_anthropic_schemas()
        executor = create_orchestrated_executor(self._tool_registry)

        # 재시도 래핑된 실행
        async def _run() -> str:
            return await run_agent(
                client=client,
                prompt=prompt,
                system=self.config.system_prompt,
                tools=tool_schemas if tool_schemas else None,
                tool_executor=executor,
                max_turns=self.config.max_turns,
                verbose=verbose,
            )

        result = await retry_with_backoff(
            _run,
            config=self.config.retry_config,
            retryable_errors=(ConnectionError, TimeoutError, OSError),
            on_retry=lambda attempt, err, delay: print(
                f"  [재시도 {attempt}] {err} — {delay:.1f}초 후 재시도"
            ),
        )
        return result

    # ── 대화형 REPL ──────────────────────────────────────────────

    async def repl(self) -> None:
        """
        대화형 REPL을 시작합니다 — Claude Code의 CLI 모드에 대응.

        사용자 입력을 받아 에이전트를 실행하고, 결과를 출력합니다.
        /quit으로 종료, /skills로 스킬 목록, /tools로 도구 목록을 확인할 수 있습니다.
        """
        self._shutdown.install()

        print("=" * 60)
        print("  mini_claude REPL")
        print("  /quit: 종료  /tools: 도구 목록  /skills: 스킬 목록")
        print("=" * 60)
        print()

        client = self._get_client()
        tool_schemas = self._tool_registry.to_anthropic_schemas()
        executor = create_orchestrated_executor(self._tool_registry)

        # 대화 히스토리 유지
        messages: list[dict] = []

        while True:
            self._shutdown.reset()

            # 사용자 입력
            try:
                user_input = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n종료합니다.")
                break

            if not user_input:
                continue

            # 내장 명령어 처리
            if user_input == "/quit":
                print("종료합니다.")
                break
            elif user_input == "/tools":
                print("등록된 도구:")
                for name in self._tool_registry.get_names():
                    tool = self._tool_registry.find_by_name(name)
                    desc = tool.description[:60] if tool else ""
                    print(f"  - {name}: {desc}")
                print()
                continue
            elif user_input == "/skills":
                print("등록된 스킬:")
                for skill in self._skill_registry.list_skills():
                    print(f"  - {skill.name} [{skill.context}]: {skill.description[:60]}")
                print()
                continue

            # 메시지 추가
            messages.append({"role": "user", "content": user_input})

            # 에이전트 루프 실행
            try:
                final_text = ""
                async for event in agent_loop(
                    client=client,
                    messages=messages,
                    system=self.config.system_prompt,
                    tools=tool_schemas if tool_schemas else None,
                    tool_executor=executor,
                    max_turns=self.config.max_turns,
                ):
                    if self._shutdown.is_requested:
                        print("\n[중단됨]")
                        break

                    if event.type == "text":
                        final_text = event.text
                    elif event.type == "tool_call" and event.tool_call:
                        tc = event.tool_call
                        print(f"  [도구] {tc.name}({_truncate(str(tc.input), 80)})")
                    elif event.type == "tool_result":
                        print(f"  [결과] {_truncate(event.tool_result, 120)}")

                if final_text:
                    print(f"\nassistant> {final_text}\n")
                    messages.append({"role": "assistant", "content": final_text})

            except Exception as e:
                print(f"\n[에러] {type(e).__name__}: {e}\n")


# ─── 유틸리티 ────────────────────────────────────────────────────

def _truncate(text: str, max_len: int) -> str:
    """텍스트를 최대 길이로 잘라냅니다."""
    text = text.replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


# ─── CLI 진입점 ──────────────────────────────────────────────────

async def main() -> None:
    """CLI에서 직접 실행할 때의 진입점."""
    import argparse

    parser = argparse.ArgumentParser(description="mini_claude 에이전트")
    parser.add_argument("--config", "-c", help="설정 파일 경로 (YAML)")
    parser.add_argument("--provider", "-p", default="auto", help="LLM 제공자 (anthropic/openai/auto)")
    parser.add_argument("--model", "-m", help="모델 이름")
    parser.add_argument("--prompt", help="단일 프롬프트 (지정하면 REPL 대신 한번 실행)")
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 출력")
    args = parser.parse_args()

    kwargs: dict[str, Any] = {"provider": args.provider}
    if args.model:
        kwargs["model"] = args.model

    agent = Agent(config_path=args.config, **kwargs)

    if args.prompt:
        result = await agent.run(args.prompt, verbose=args.verbose)
        print(result)
    else:
        await agent.repl()


if __name__ == "__main__":
    asyncio.run(main())
