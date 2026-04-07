"""
컨텍스트 관리 — 시스템 프롬프트 조립과 실행 환경 정보 수집

Claude Code는 LLM에 보내는 시스템 프롬프트를 여러 조각으로 나누어 동적으로
조립합니다. 이 모듈은 그 패턴을 간소화하여 구현합니다.

프롬프트 구성 요소:
  1. 정적 프롬프트    — 역할, 행동 지침 등 고정 텍스트
  2. 시스템 컨텍스트  — OS 정보, CWD, git 상태 (세션 시작 시 캐시)
  3. 사용자 컨텍스트  — CLAUDE.md 파일에서 로드한 프로젝트별 지시사항
  4. 도구 설명       — 등록된 도구의 이름/설명/스키마

참조:
- src/context.ts         — getSystemPrompt(), getToolsContext() 등
- src/constants/prompts.ts — MAIN_PROMPT, TOOL_USE_RULES 등
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from mini_claude.tool_base import Tool


# ─── 정적 프롬프트 ──────────────────────────────────────────────

STATIC_PROMPT = """\
You are an interactive CLI tool that helps users with software engineering tasks.

Guidelines:
- Be direct and concise in responses.
- When editing files, always read the file first to understand existing content.
- Prefer editing existing files over creating new ones.
- When running commands, explain what you're doing and why.
- If a task is ambiguous, ask for clarification.
- Always respect the user's project structure and conventions.
"""

TOOL_USE_RULES = """\
Tool Use Rules:
- Use tools to accomplish tasks. Do not guess at file contents or command outputs.
- Always read a file before editing it.
- Prefer targeted edits over full file rewrites.
- When running shell commands, handle errors gracefully.
"""


# ─── ToolUseContext: 도구 실행에 필요한 모든 컨텍스트 ──────────

@dataclass
class ToolUseContext:
    """
    도구가 실행될 때 필요한 환경 정보를 캡슐화합니다.

    Claude Code에서 도구는 call(input, context) 형태로 호출되는데,
    이 context가 도구에 CWD, 환경변수, 권한 정보 등을 전달합니다.

    참조: src/Tool.ts의 ToolUseContext
    """
    cwd: str = field(default_factory=os.getcwd)
    env: dict[str, str] = field(default_factory=lambda: dict(os.environ))
    is_git_repo: bool = False
    git_branch: str = ""


# ─── ContextManager: 시스템 프롬프트 조립 ──────────────────────

class ContextManager:
    """
    시스템 프롬프트를 동적으로 조립하는 관리자.

    Claude Code의 getSystemPrompt()는 여러 소스에서 정보를 모아
    하나의 시스템 프롬프트를 만듭니다. 이 클래스가 그 역할을 합니다.

    사용법:
        ctx = ContextManager(cwd="/path/to/project")
        system_prompt = ctx.get_system_prompt(tools=[...])
    """

    def __init__(self, cwd: str | None = None):
        self.cwd = cwd or os.getcwd()

    def get_system_prompt(self, tools: list[Tool] | None = None) -> str:
        """
        전체 시스템 프롬프트를 조립하여 반환합니다.

        구성 순서:
          1) 정적 프롬프트 (역할/지침)
          2) 도구 사용 규칙
          3) 시스템 컨텍스트 (OS, CWD, git)
          4) 사용자 컨텍스트 (CLAUDE.md)
          5) 도구 목록 설명

        이 순서는 Claude Code의 getSystemPrompt()를 따릅니다.
        """
        sections: list[str] = []

        # 1. Static prompt
        sections.append(STATIC_PROMPT)

        # 2. Tool use rules
        sections.append(TOOL_USE_RULES)

        # 3. System context (OS, CWD, git info)
        sys_ctx = self.get_system_context()
        sections.append(sys_ctx)

        # 4. User context (CLAUDE.md)
        user_ctx = self.get_user_context()
        if user_ctx:
            sections.append(f"# User Instructions (from CLAUDE.md)\n{user_ctx}")

        # 5. Tools description
        if tools:
            tools_section = self._build_tools_section(tools)
            sections.append(tools_section)

        return "\n\n".join(sections)

    # ── 시스템 컨텍스트 ──

    @lru_cache(maxsize=1)
    def get_system_context(self) -> str:
        """
        실행 환경 정보를 수집합니다 (세션 동안 캐시됨).

        수집 항목:
          - OS 이름/버전
          - 현재 작업 디렉토리
          - git 브랜치 (git repo인 경우)

        Claude Code에서도 이 정보는 시스템 프롬프트에 포함되어
        LLM이 환경에 맞는 명령어를 생성할 수 있게 합니다.
        """
        lines = [
            "# System Context",
            f"- OS: {platform.system()} {platform.release()}",
            f"- CWD: {self.cwd}",
            f"- Shell: {os.environ.get('SHELL', 'unknown')}",
        ]

        # Git info
        git_branch = self._get_git_branch()
        if git_branch:
            lines.append(f"- Git branch: {git_branch}")
            lines.append(f"- Is git repo: Yes")
        else:
            lines.append(f"- Is git repo: No")

        return "\n".join(lines)

    # ── 사용자 컨텍스트 ──

    @lru_cache(maxsize=1)
    def get_user_context(self) -> str:
        """
        CLAUDE.md 파일들을 찾아 내용을 합칩니다 (세션 동안 캐시됨).

        Claude Code는 여러 위치에서 CLAUDE.md를 찾습니다:
          1) 프로젝트 루트 (CWD)의 CLAUDE.md
          2) 홈 디렉토리의 ~/.claude/CLAUDE.md
          3) 각 부모 디렉토리의 CLAUDE.md (git root까지)

        이 간소화 버전에서는 CWD와 홈 디렉토리만 확인합니다.
        """
        contents: list[str] = []

        # Check CWD/CLAUDE.md
        project_claude = Path(self.cwd) / "CLAUDE.md"
        if project_claude.is_file():
            text = project_claude.read_text(encoding="utf-8").strip()
            if text:
                contents.append(f"## Project CLAUDE.md\n{text}")

        # Check ~/.claude/CLAUDE.md
        home_claude = Path.home() / ".claude" / "CLAUDE.md"
        if home_claude.is_file():
            text = home_claude.read_text(encoding="utf-8").strip()
            if text:
                contents.append(f"## User CLAUDE.md\n{text}")

        return "\n\n".join(contents)

    # ── ToolUseContext 생성 ──

    def build_tool_context(self) -> ToolUseContext:
        """도구 실행에 필요한 ToolUseContext를 생성합니다."""
        git_branch = self._get_git_branch()
        return ToolUseContext(
            cwd=self.cwd,
            is_git_repo=bool(git_branch),
            git_branch=git_branch or "",
        )

    # ── 도구 설명 생성 ──

    @staticmethod
    def _build_tools_section(tools: list[Tool]) -> str:
        """
        도구 목록을 사람/LLM이 읽을 수 있는 텍스트로 변환합니다.

        Claude Code에서는 도구 설명이 시스템 프롬프트의 일부로 포함되어
        LLM이 어떤 도구를 사용할 수 있는지 알 수 있게 합니다.
        """
        lines = ["# Available Tools"]
        for tool in tools:
            if not tool.is_enabled():
                continue
            lines.append(f"\n## {tool.name}")
            lines.append(tool.description)

            # Show required parameters
            schema = tool.input_schema
            props = schema.get("properties", {})
            required = set(schema.get("required", []))
            if props:
                lines.append("Parameters:")
                for param_name, param_info in props.items():
                    req_marker = " (required)" if param_name in required else ""
                    desc = param_info.get("description", "")
                    lines.append(f"  - {param_name}{req_marker}: {desc}")

        return "\n".join(lines)

    # ── 내부 헬퍼 ──

    def _get_git_branch(self) -> str | None:
        """현재 git 브랜치를 반환합니다. git repo가 아니면 None."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return None
