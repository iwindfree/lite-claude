# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

This is the leaked source code of Anthropic's Claude Code CLI (leaked 2026-03-31). The `src/` directory is the original source and should not be modified. Contributions target documentation, the MCP server (`mcp-server/`), and exploration tooling only.

## Build & Dev Commands

```bash
# Install dependencies (requires Bun >=1.1.0)
bun install

# Build
bun run build                # Development build
bun run build:prod           # Production build (minified)
bun run build:watch          # Watch mode

# Web build
bun run build:web
bun run build:web:prod

# Lint & Type Check
bun run lint                 # Biome lint (src/)
bun run lint:fix             # Biome lint with auto-fix
bun run format               # Biome format
bun run typecheck            # TypeScript type check (tsc --noEmit)
bun run check                # Both lint + typecheck

# MCP Server (uses Node.js, not Bun)
cd mcp-server && npm install && npm run dev    # Dev mode (tsx)
cd mcp-server && npm run build                 # Compile to dist/
```

## Architecture

**Pipeline:** User Input → CLI Parser (Commander.js) → Query Engine → Anthropic LLM API → Tool Execution Loop → React/Ink Terminal UI

### Key Source Files

- `src/main.tsx` — Entrypoint: CLI parser + React/Ink renderer, parallel prefetch on startup
- `src/QueryEngine.ts` (~46K lines) — Core LLM engine: streaming, tool-call loops, thinking mode, retries, token counting
- `src/Tool.ts` (~29K lines) — Base tool types/interfaces, input schemas, permissions, progress state
- `src/commands.ts` (~25K lines) — Command registration with conditional per-environment imports
- `src/tools.ts` — Tool registry
- `src/context.ts` — System/user context collection
- `src/entrypoints/cli.tsx` — CLI session orchestration (actual entrypoint per package.json)

### Subsystems

- **Tools** (`src/tools/`) — ~40 self-contained tool modules (BashTool, FileEditTool, AgentTool, etc.), each with Zod input schema, permission model, execution logic, and UI components
- **Commands** (`src/commands/`) — ~50 slash commands; three types: PromptCommand (sends to LLM), LocalCommand (returns text), LocalJSXCommand (returns JSX)
- **Components** (`src/components/`, ~140) — Functional React components using Ink primitives (`Box`, `Text`, `useInput()`)
- **Hooks** (`src/hooks/`, ~80) — Permission checks, IDE integration, input handling, session management
- **Services** (`src/services/`) — API client, MCP connections, OAuth, LSP, analytics/GrowthBook, compaction, memory extraction
- **Bridge** (`src/bridge/`) — Bidirectional IDE integration (VS Code, JetBrains) via `bridgeMain.ts`, `bridgeMessaging.ts`
- **State** (`src/state/`) — `AppStateStore.ts` global mutable state, React context providers, change observers
- **Coordinator** (`src/coordinator/`) — Multi-agent orchestration
- **Skills** (`src/skills/`) — Reusable workflow system, executed via SkillTool

### Build System

- Runtime: **Bun** (not Node.js) — native JSX/TSX, ES modules with `.js` extensions
- Feature flags via `bun:bundle` for dead-code elimination at build time (PROACTIVE, KAIROS, BRIDGE_MODE, DAEMON, VOICE_MODE, etc.)
- Heavy modules (OpenTelemetry ~400KB, gRPC ~700KB) lazy-loaded via dynamic `import()`

## Code Style

- **Formatter:** Biome — tabs for indentation, line width 100, single quotes, semicolons as needed
- **JSON:** spaces (2) for indentation
- **TypeScript:** strict mode, ESNext target, bundler module resolution, JSX via react-jsx
- `noExplicitAny: off` — `any` is allowed
- `noNonNullAssertion: off` — `!` assertions are allowed
