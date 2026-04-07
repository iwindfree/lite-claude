"""
스킬 시스템 — 프롬프트 기반 워크플로우 조합

Claude Code의 스킬은 "재사용 가능한 프롬프트 템플릿"입니다.
마크다운 파일에 YAML 프론트매터로 메타데이터를 정의하고,
본문에 프롬프트를 작성하면 하나의 스킬이 됩니다.

스킬의 핵심 아이디어:
- 복잡한 워크플로우를 프롬프트로 캡슐화
- 도구 코드 없이 새로운 기능을 추가할 수 있음
- inline(현재 대화에 주입) 또는 fork(서브 에이전트로 실행) 모드 지원

참조:
- src/skills/bundledSkills.ts  — 번들된 스킬 정의
- src/types/command.ts         — PromptCommand 타입
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# ─── 스킬 정의 ──────────────────────────────────────────────────

@dataclass
class SkillDefinition:
    """
    하나의 스킬을 정의하는 데이터 — Claude Code의 PromptCommand에 대응

    name: 스킬 이름 (사용자가 호출할 때 사용)
    description: 스킬 설명 (LLM이 스킬 선택 시 참고)
    prompt_template: 실행할 프롬프트 텍스트 (변수 치환 가능)
    context: 실행 모드
        - 'inline': 현재 대화에 프롬프트를 주입 (빠르고 컨텍스트 공유)
        - 'fork': 새 서브 에이전트를 생성하여 독립 실행 (격리)
    allowed_tools: 이 스킬이 사용할 수 있는 도구 목록 (None이면 모든 도구)
    when_to_use: LLM에게 이 스킬을 언제 사용해야 하는지 알려주는 힌트
    """
    name: str
    description: str
    prompt_template: str
    context: Literal["inline", "fork"] = "inline"
    allowed_tools: list[str] | None = None
    when_to_use: str = ""

    def render(self, **kwargs: str) -> str:
        """프롬프트 템플릿에 변수를 치환하여 최종 프롬프트 생성"""
        result = self.prompt_template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", value)
        return result


# ─── 마크다운 파일에서 스킬 로드 ────────────────────────────────

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """
    마크다운 파일에서 YAML 프론트매터와 본문을 분리.

    파일 형식:
        ---
        name: my_skill
        description: ...
        context: inline
        ---
        프롬프트 본문...

    YAML 파싱은 yaml 라이브러리가 있으면 사용하고, 없으면 간단한 파서를 사용합니다.
    """
    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text

    frontmatter_text = parts[1].strip()
    body = parts[2].strip()

    # YAML 파싱 시도
    try:
        import yaml
        metadata = yaml.safe_load(frontmatter_text) or {}
    except ImportError:
        # yaml이 없으면 간단한 key: value 파서 사용
        metadata = {}
        for line in frontmatter_text.splitlines():
            line = line.strip()
            if ":" in line:
                key, _, value = line.partition(":")
                value = value.strip()
                # 간단한 리스트 처리: [a, b, c]
                if value.startswith("[") and value.endswith("]"):
                    value = [v.strip() for v in value[1:-1].split(",")]
                metadata[key.strip()] = value

    return metadata, body


def load_skills_from_dir(path: str | Path) -> list[SkillDefinition]:
    """
    디렉토리에서 .md 파일들을 읽어 스킬로 변환.

    각 .md 파일은 YAML 프론트매터에 메타데이터, 본문에 프롬프트를 포함합니다.
    Claude Code도 동일한 패턴으로 커스텀 슬래시 커맨드를 로드합니다.
    """
    path = Path(path)
    skills: list[SkillDefinition] = []

    if not path.is_dir():
        return skills

    for md_file in sorted(path.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        metadata, body = _parse_frontmatter(text)

        name = metadata.get("name", md_file.stem)
        description = metadata.get("description", "")
        context = metadata.get("context", "inline")
        allowed_tools = metadata.get("allowed_tools", None)
        when_to_use = metadata.get("when_to_use", "")

        if context not in ("inline", "fork"):
            context = "inline"

        skills.append(SkillDefinition(
            name=name,
            description=description,
            prompt_template=body,
            context=context,
            allowed_tools=allowed_tools if isinstance(allowed_tools, list) else None,
            when_to_use=when_to_use,
        ))

    return skills


# ─── 스킬 레지스트리 ────────────────────────────────────────────

class SkillRegistry:
    """
    스킬 레지스트리 — 스킬을 등록하고 이름으로 조회

    Claude Code의 스킬 시스템과 동일하게:
    - 번들 스킬 (내장)
    - 커스텀 스킬 (.md 파일에서 로드)
    을 통합 관리합니다.
    """

    def __init__(self):
        self._skills: dict[str, SkillDefinition] = {}

    def register(self, skill: SkillDefinition) -> None:
        """스킬을 레지스트리에 등록"""
        self._skills[skill.name] = skill

    def register_many(self, skills: list[SkillDefinition]) -> None:
        """여러 스킬을 한번에 등록"""
        for skill in skills:
            self.register(skill)

    def find_by_name(self, name: str) -> SkillDefinition | None:
        """이름으로 스킬 조회"""
        return self._skills.get(name)

    def list_skills(self) -> list[SkillDefinition]:
        """등록된 모든 스킬 반환"""
        return list(self._skills.values())

    def list_names(self) -> list[str]:
        """등록된 모든 스킬 이름 반환"""
        return list(self._skills.keys())

    def load_from_dir(self, path: str | Path) -> int:
        """디렉토리에서 스킬을 로드하여 등록. 로드된 스킬 수를 반환."""
        skills = load_skills_from_dir(path)
        self.register_many(skills)
        return len(skills)

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills


# ─── 번들 스킬 — Claude Code 내장 스킬의 간소화 버전 ────────────
#
# Claude Code의 bundledSkills.ts에 정의된 스킬들을 Python으로 재현합니다.
# 실제 Claude Code에는 commit, review-pr, simplify 등의 스킬이 있습니다.

BUNDLED_SKILLS: list[SkillDefinition] = [
    SkillDefinition(
        name="commit",
        description="변경사항을 분석하고 커밋 메시지를 작성하여 커밋합니다",
        prompt_template=(
            "현재 Git 저장소의 변경사항을 분석하고 커밋을 생성해주세요.\n\n"
            "단계:\n"
            "1. `git status`와 `git diff`로 변경사항을 확인하세요\n"
            "2. `git log --oneline -5`로 최근 커밋 스타일을 확인하세요\n"
            "3. 변경사항을 요약하는 커밋 메시지를 작성하세요\n"
            "4. 적절한 파일을 staging하고 커밋하세요\n\n"
            "추가 인자: {args}"
        ),
        context="inline",
        allowed_tools=["Bash"],
        when_to_use="사용자가 커밋을 요청하거나 /commit을 입력할 때",
    ),
    SkillDefinition(
        name="review",
        description="코드 변경사항을 리뷰하고 피드백을 제공합니다",
        prompt_template=(
            "코드 변경사항을 리뷰해주세요.\n\n"
            "확인 사항:\n"
            "- 버그나 논리 오류가 없는지\n"
            "- 코드 스타일이 일관적인지\n"
            "- 테스트가 충분한지\n"
            "- 성능 이슈가 없는지\n\n"
            "리뷰 대상: {args}"
        ),
        context="fork",
        when_to_use="사용자가 코드 리뷰를 요청하거나 /review를 입력할 때",
    ),
    SkillDefinition(
        name="simplify",
        description="변경된 코드에서 재사용 기회, 품질 문제, 효율성을 검토하고 수정합니다",
        prompt_template=(
            "변경된 코드를 분석하고 다음을 확인해주세요:\n\n"
            "1. 코드 재사용 기회 — 중복된 로직이 있으면 함수/클래스로 추출\n"
            "2. 코드 품질 — 가독성, 네이밍, 구조 개선\n"
            "3. 효율성 — 불필요한 연산, 개선 가능한 알고리즘\n\n"
            "문제를 발견하면 직접 수정해주세요.\n\n"
            "대상: {args}"
        ),
        context="inline",
        allowed_tools=["Bash", "Read", "Edit"],
        when_to_use="사용자가 코드 정리/단순화를 요청하거나 /simplify를 입력할 때",
    ),
]


def create_default_registry() -> SkillRegistry:
    """번들 스킬이 등록된 기본 레지스트리를 생성합니다."""
    registry = SkillRegistry()
    registry.register_many(BUNDLED_SKILLS)
    return registry
