# Pydantic AI Agent Alignment Design

**Date:** 2026-01-18
**Scope:** Align `distributed_rag_agent.py` and `enhanced_rag_agent.py` with current Pydantic AI interfaces and models.

## Goals
- Use Pydantic AI's current model/provider configuration for OpenAI-compatible APIs.
- Move all LLM calls (including content analysis) under Pydantic AI.
- Update run result access from `result.data` to `result.output` per current docs.
- Keep retrieval logic and tool behaviors unchanged.

## Non-Goals
- Changing retrieval ranking or PageIndex search algorithms.
- Altering CLI behavior or output formats beyond the Pydantic AI interface alignment.
- Introducing new models or providers beyond OpenAI-compatible endpoints.

## Architecture Overview
Both agents will construct `OpenAIChatModel` using `OpenAIProvider` with explicit `api_key` and optional `base_url`. The enhanced agent will include a dedicated analysis sub-agent to replace direct OpenAI client calls. Tools remain standard Pydantic AI function tools with `RunContext` dependencies.

Key elements:
- Model config: `OpenAIChatModel(model_name, provider=OpenAIProvider(api_key=..., base_url=...))`
- Output access: `RunResult.output`
- Enhanced analysis: a sub-agent called from `ask_llm_about_content` tool

## Components and Data Flow

### Distributed Agent
1. Build `OpenAIChatModel` via `OpenAIProvider` in `create_rag_agent`.
2. `Agent.run_sync` returns `RunResult`.
3. All response reads use `result.output`.

### Enhanced Agent
1. Build primary `OpenAIChatModel` via `OpenAIProvider` in `create_enhanced_agent`.
2. Add an analysis sub-agent configured with the same provider/model.
3. `ask_llm_about_content` tool calls the sub-agent and returns `result.output`.
4. The main agent continues to orchestrate file search and section navigation via tools.

## Configuration and Environment
- API key resolution order: explicit param > `OPENAI_API_KEY` > `CHATGPT_API_KEY`.
- Base URL resolution: explicit param > `OPENAI_BASE_URL`.
- Missing API key raises `ValueError` with clear instructions.
- Logging continues to redact keys.

## Error Handling
- Tools keep try/except blocks and return compact error messages.
- Sub-agent invocation failures are handled inside the tool so the agent can surface them.
- Retrieval errors remain unchanged.

## Testing Strategy (TDD)
- Add unit tests that use a stub model to avoid network calls.
- Verify `run_sync`/`run` results read `output`, not `data`.
- Verify API key resolution order and base URL propagation into `OpenAIProvider`.
- Verify `ask_llm_about_content` delegates to the analysis sub-agent and returns its output.

## Trade-offs and Rationale
- Sub-agent delegation keeps all LLM usage inside Pydantic AI and avoids direct OpenAI client calls.
- Slightly more code for the analysis agent, but clearer separation and consistent configuration.
- Minimal behavior changes; retrieval and prompts remain intact.

## Open Questions
- Preferred location for new tests (`tests/` vs `labbook/tests/`).
- Whether to standardize env variable names across the repo (deferred).
