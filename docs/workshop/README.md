# Claude Code 아키텍처 기반 에이전트 시스템 워크샵

Anthropic의 Claude Code CLI 소스 코드를 분석하고, 그 핵심 아키텍처를 Python으로 직접 구현하며 배우는 워크샵입니다.

## 이 워크샵에서 만드는 것

Claude Code의 핵심 패턴을 재현한 **mini_claude** — 도구 호출, 컨텍스트 관리, 서브에이전트, MCP 통합 등을 갖춘 에이전트 시스템을 13단계에 걸쳐 점진적으로 구축합니다.

## 사전 준비

```bash
# Python 3.11+
python --version

# 패키지 설치
pip install anthropic openai pydantic tiktoken pyyaml websockets mcp

# Jupyter 설치 (아직 없다면)
pip install jupyterlab

# API 키 설정 (둘 중 하나 이상)
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

## 워크샵 구조

각 Step은 Jupyter Notebook으로, **Claude Code 소스 분석**과 **Python 구현**을 번갈아 진행합니다.
Step이 진행될수록 `mini_claude/` 패키지에 코드가 누적됩니다.

### 기초 — 에이전트의 심장

| Step | 주제 | 만드는 것 |
|------|------|-----------|
| 1 | [에이전트 루프 + 스트리밍](step_01_agent_loop.ipynb) | `llm_client.py`, `agent_loop.py` |
| 2 | [도구 정의 시스템](step_02_tool_system.ipynb) | `tool_base.py`, `tool_registry.py`, `tools/` |
| 3 | [도구 실행 오케스트레이션](step_03_tool_orchestration.ipynb) | `orchestrator.py` |
| 4 | [MCP 외부 도구 통합](step_04_mcp_integration.ipynb) | `mcp/`, `tools/mcp_tool.py` |

### 인프라 — 견고한 에이전트

| Step | 주제 | 만드는 것 |
|------|------|-----------|
| 5 | [컨텍스트 관리](step_05_context_management.ipynb) | `context.py` |
| 6 | [Hooks 런타임 확장](step_06_hooks_system.ipynb) | `hooks.py` |
| 7 | [권한 시스템](step_07_permission_system.ipynb) | `permissions.py` |
| 8 | [컨텍스트 압축 + 에러 복구](step_08_context_compaction.ipynb) | `compaction.py`, `state.py`, `recovery.py` |
| 9 | [세션 간 메모리](step_09_memory_system.ipynb) | `memory.py` |

### 고급 — 확장과 통합

| Step | 주제 | 만드는 것 |
|------|------|-----------|
| 10 | [Skill 시스템](step_10_skill_system.ipynb) | `skills/`, `tools/skill_tool.py` |
| 11 | [서브에이전트 + 코디네이터](step_11_sub_agent.ipynb) | `tools/agent_tool.py` |
| 12 | [Bridge IDE 연동](step_12_bridge_system.ipynb) | `bridge/` |
| 13 | [통합 — 완성된 에이전트](step_13_integration.ipynb) | `agent.py` |

## mini_claude 패키지 구조

```
mini_claude/
├── llm_client.py          # LLM API 추상화 (Anthropic + OpenAI)
├── agent_loop.py          # 핵심 while(true) 에이전트 루프
├── tool_base.py           # Tool 프로토콜 + build_tool 팩토리
├── tool_registry.py       # 도구 레지스트리
├── orchestrator.py        # 직렬/병렬 도구 실행
├── context.py             # 시스템 프롬프트 + 컨텍스트 관리
├── hooks.py               # 라이프사이클 훅 시스템
├── permissions.py         # 다단계 권한 시스템
├── compaction.py          # 컨텍스트 압축
├── state.py               # 불변 상태 관리
├── recovery.py            # 에러 복구
├── memory.py              # 영속 메모리
├── agent.py               # 통합 Agent 클래스
├── tools/                 # 도구 구현체
│   ├── bash_tool.py
│   ├── file_read_tool.py
│   ├── file_write_tool.py
│   ├── mcp_tool.py
│   ├── skill_tool.py
│   └── agent_tool.py
├── mcp/                   # MCP 클라이언트
├── skills/                # 스킬 시스템
└── bridge/                # IDE 브릿지
```

## Notebook 사용법

```bash
cd docs/workshop
jupyter lab
```

각 Notebook은 순서대로 진행하세요. 이전 Step의 `mini_claude/` 코드를 import하므로,
앞선 Step의 Python 파일이 먼저 작성되어 있어야 합니다.

## Claude Code 소스 참조

이 워크샵은 `src/` 디렉토리의 Claude Code 소스를 분석합니다.
Notebook에서 `src/query.ts:307` 같은 참조가 나오면, 리포지토리 루트 기준 경로입니다.
