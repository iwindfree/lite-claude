"""
영속 메모리 시스템 — 세션 간 기억을 유지하는 MEMORY.md 기반 시스템

Claude Code는 프로젝트에 대한 지식을 세션 간에 유지하기 위해
MEMORY.md 파일과 개별 토픽 파일을 사용합니다:

    MEMORY.md (인덱스 파일):
        - 각 메모리 토픽의 이름, 설명, 파일 경로를 나열
        - 최대 200줄 / 25,000바이트 제한

    topics/ (개별 토픽 파일):
        - YAML frontmatter + 마크다운 본문
        - 예: topics/project-architecture.md

시스템 프롬프트에 메모리를 주입하여 LLM이 이전 세션의 맥락을 알 수 있게 합니다.

참조:
- src/memdir/memdir.ts              — MemDir 클래스 (메모리 디렉토리 관리)
- src/memdir/memdir.ts:MAX_ENTRYPOINT_LINES, MAX_ENTRYPOINT_BYTES
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# ─── 상수 ───────────────────────────────────────────────────────
#
# Claude Code의 memdir.ts에서 가져온 제한값

MAX_ENTRYPOINT_LINES = 200     # MEMORY.md 최대 줄 수
MAX_ENTRYPOINT_BYTES = 25_000  # MEMORY.md 최대 바이트 수


# ─── 메모리 타입 ────────────────────────────────────────────────

class MemoryType(Enum):
    """메모리 항목의 유형"""
    USER = "user"             # 사용자가 직접 저장한 메모리
    FEEDBACK = "feedback"     # 사용자 피드백에서 추출한 메모리
    PROJECT = "project"       # 프로젝트 구조/설정 관련 메모리
    REFERENCE = "reference"   # 참조 문서/코드 관련 메모리


# ─── 메모리 항목 ────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """
    단일 메모리 항목 — Claude Code의 MemDir entry에 대응

    name: 고유 이름 (파일명으로도 사용, 예: "project-architecture")
    description: 간단한 설명 (MEMORY.md 인덱스에 표시)
    type: 메모리 유형
    content: 마크다운 본문 내용
    file_path: 토픽 파일 경로 (자동 생성)
    """
    name: str
    description: str
    type: MemoryType = MemoryType.USER
    content: str = ""
    file_path: str = ""

    def to_frontmatter(self) -> str:
        """YAML frontmatter 생성"""
        return (
            f"---\n"
            f"name: {self.name}\n"
            f"description: {self.description}\n"
            f"type: {self.type.value}\n"
            f"---\n"
        )

    def to_file_content(self) -> str:
        """토픽 파일 전체 내용 생성 (frontmatter + 본문)"""
        return self.to_frontmatter() + "\n" + self.content + "\n"

    def to_index_line(self) -> str:
        """MEMORY.md 인덱스의 한 줄"""
        return f"- **{self.name}** ({self.type.value}): {self.description}"


# ─── YAML Frontmatter 파싱 ──────────────────────────────────────

_FRONTMATTER_RE = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n",
    re.DOTALL,
)


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """
    YAML frontmatter를 파싱하여 (메타데이터, 본문) 반환.

    간소화된 파서 — PyYAML 없이 key: value 형태만 처리합니다.
    실제 Claude Code에서도 간단한 frontmatter만 사용합니다.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    frontmatter_text = match.group(1)
    body = text[match.end():].lstrip("\n")

    metadata: dict[str, str] = {}
    for line in frontmatter_text.split("\n"):
        line = line.strip()
        if ":" in line:
            key, _, value = line.partition(":")
            metadata[key.strip()] = value.strip()

    return metadata, body


def _entry_from_file(file_path: str, text: str) -> MemoryEntry | None:
    """파일 내용으로부터 MemoryEntry를 생성"""
    metadata, body = parse_frontmatter(text)
    name = metadata.get("name", "")
    if not name:
        # 파일명에서 이름 추출
        name = Path(file_path).stem

    description = metadata.get("description", "")
    type_str = metadata.get("type", "user")
    try:
        mem_type = MemoryType(type_str)
    except ValueError:
        mem_type = MemoryType.USER

    return MemoryEntry(
        name=name,
        description=description,
        type=mem_type,
        content=body.strip(),
        file_path=file_path,
    )


# ─── 메모리 매니저 ──────────────────────────────────────────────

class MemoryManager:
    """
    영속 메모리 관리자 — Claude Code의 MemDir 클래스 (memdir.ts)

    디렉토리 구조:
        memory_dir/
        ├── MEMORY.md           (인덱스 파일)
        └── topics/
            ├── topic-a.md      (개별 토픽)
            └── topic-b.md

    주요 기능:
    - load(): MEMORY.md + topics/ 디렉토리에서 모든 메모리 로드
    - save(entry): 토픽 파일 저장 + MEMORY.md 인덱스 업데이트
    - delete(name): 토픽 삭제 + 인덱스 업데이트
    - build_prompt(): 시스템 프롬프트에 주입할 메모리 텍스트 생성
    """

    def __init__(self, memory_dir: str | Path):
        self.memory_dir = Path(memory_dir)
        self.topics_dir = self.memory_dir / "topics"
        self.index_path = self.memory_dir / "MEMORY.md"
        self._entries: dict[str, MemoryEntry] = {}

    # ── 초기화 ──

    def _ensure_dirs(self) -> None:
        """필요한 디렉토리 생성"""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.topics_dir.mkdir(exist_ok=True)

    # ── Load ──

    def load(self) -> list[MemoryEntry]:
        """
        모든 메모리 로드 — MEMORY.md 인덱스 + topics/ 파일들

        MEMORY.md 인덱스는 참조용이고, 실제 데이터는 topics/ 파일에서 로드합니다.
        """
        self._entries.clear()

        if not self.topics_dir.exists():
            return []

        for topic_file in sorted(self.topics_dir.glob("*.md")):
            try:
                text = topic_file.read_text(encoding="utf-8")
                entry = _entry_from_file(str(topic_file), text)
                if entry:
                    entry.file_path = str(topic_file)
                    self._entries[entry.name] = entry
            except (OSError, UnicodeDecodeError):
                continue  # 읽기 실패한 파일은 건너뜀

        return list(self._entries.values())

    # ── Save ──

    def save(self, entry: MemoryEntry) -> str:
        """
        메모리 항목 저장 — 토픽 파일 + MEMORY.md 인덱스 업데이트

        Args:
            entry: 저장할 메모리 항목

        Returns:
            저장된 파일 경로
        """
        self._ensure_dirs()

        # 파일명 생성 (이름에서 안전한 파일명으로)
        safe_name = _slugify(entry.name)
        file_path = self.topics_dir / f"{safe_name}.md"
        entry.file_path = str(file_path)

        # 토픽 파일 저장
        file_path.write_text(entry.to_file_content(), encoding="utf-8")

        # 메모리에 추가/업데이트
        self._entries[entry.name] = entry

        # MEMORY.md 인덱스 업데이트
        self._update_index()

        return str(file_path)

    # ── Delete ──

    def delete(self, name: str) -> bool:
        """
        메모리 항목 삭제 — 토픽 파일 + MEMORY.md 인덱스 업데이트

        Args:
            name: 삭제할 항목 이름

        Returns:
            True=삭제 성공, False=항목 없음
        """
        entry = self._entries.pop(name, None)
        if not entry:
            return False

        # 토픽 파일 삭제
        if entry.file_path:
            try:
                Path(entry.file_path).unlink(missing_ok=True)
            except OSError:
                pass

        # MEMORY.md 업데이트
        self._update_index()
        return True

    # ── Query ──

    def get(self, name: str) -> MemoryEntry | None:
        """이름으로 메모리 항목 조회"""
        return self._entries.get(name)

    def list_entries(self, type_filter: MemoryType | None = None) -> list[MemoryEntry]:
        """모든 메모리 항목 목록 (선택적 타입 필터)"""
        entries = list(self._entries.values())
        if type_filter is not None:
            entries = [e for e in entries if e.type == type_filter]
        return entries

    # ── System Prompt 주입 ──

    def build_prompt(self) -> str:
        """
        시스템 프롬프트에 주입할 메모리 텍스트 생성.

        Claude Code는 시스템 프롬프트에 메모리를 포함하여
        LLM이 이전 세션의 맥락을 알 수 있게 합니다.
        """
        entries = list(self._entries.values())
        if not entries:
            return ""

        lines: list[str] = [
            "# Project Memory",
            "",
            "다음은 이전 세션에서 저장된 프로젝트 관련 메모리입니다:",
            "",
        ]

        for entry in entries:
            lines.append(f"## {entry.name}")
            if entry.description:
                lines.append(f"*{entry.description}*")
            lines.append("")
            if entry.content:
                lines.append(entry.content)
            lines.append("")

        result = "\n".join(lines)

        # MAX_ENTRYPOINT_BYTES 제한 적용
        if len(result.encode("utf-8")) > MAX_ENTRYPOINT_BYTES:
            result = result.encode("utf-8")[:MAX_ENTRYPOINT_BYTES].decode(
                "utf-8", errors="ignore"
            )
            result += "\n\n... [메모리가 잘렸습니다 — MAX_ENTRYPOINT_BYTES 초과]"

        # MAX_ENTRYPOINT_LINES 제한 적용
        result_lines = result.split("\n")
        if len(result_lines) > MAX_ENTRYPOINT_LINES:
            result = "\n".join(result_lines[:MAX_ENTRYPOINT_LINES])
            result += "\n\n... [메모리가 잘렸습니다 — MAX_ENTRYPOINT_LINES 초과]"

        return result

    # ── MEMORY.md 인덱스 관리 ──

    def _update_index(self) -> None:
        """MEMORY.md 인덱스 파일 업데이트"""
        self._ensure_dirs()

        lines: list[str] = [
            "# MEMORY",
            "",
            "이 파일은 자동 생성됩니다. 직접 수정하지 마세요.",
            "",
            f"총 {len(self._entries)}개 항목",
            "",
        ]

        for entry in self._entries.values():
            lines.append(entry.to_index_line())

        content = "\n".join(lines) + "\n"

        # 크기 제한 확인
        if len(content.encode("utf-8")) > MAX_ENTRYPOINT_BYTES:
            # 항목이 너무 많으면 설명을 축약
            lines_short = lines[:6]  # 헤더 유지
            for entry in self._entries.values():
                lines_short.append(f"- **{entry.name}** ({entry.type.value})")
            content = "\n".join(lines_short) + "\n"

        self.index_path.write_text(content, encoding="utf-8")


# ─── 유틸리티 ───────────────────────────────────────────────────

def _slugify(name: str) -> str:
    """
    이름을 파일명으로 안전하게 변환.

    예: "Project Architecture" → "project-architecture"
    """
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = slug.strip("-")
    return slug or "untitled"
