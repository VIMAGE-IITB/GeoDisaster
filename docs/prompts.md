# Prompts

## Prompt locations

- **Multi-agent (executable):** [`code/multi_agent/prompts.py`](../code/multi_agent/prompts.py) —
  the orchestrator and specialist system prompts plus the message builders that assemble each turn.
- **Single-agent (executable):** [`code/single_agent/prompts.py`](../code/single_agent/prompts.py) —
  the single-agent system prompt and user-message builders.
- **Markdown references:** [`code/prompts/`](../code/prompts/) — human-readable references for
  `multi_agent_role_contracts.md`, `single_agent_system_prompt.md`, and
  `tool_and_question_spec.md`. Every released role-SFT row carries its exact system-prompt
  snapshot; the dataset also includes the standalone single-agent prompt and multi-agent
  role/tool context.

## Prompt organization

The multi-agent prompts are split by role and decision point:

| Constant | Used for |
|---|---|
| `ORC_PLAN_SYSTEM` | the orchestrator writing the initial plan |
| `ORC_DECIDE_SYSTEM` | the orchestrator deciding the next move after a specialist returns |
| `GA_SYSTEM` / `PA_SYSTEM` / `VRA_SYSTEM` | the three specialists (built from a shared `_SPECIALIST_BASE`) |
| `SYSTEM_PROMPTS` | role → system-prompt lookup used by the runtime and trainers |

Specialist prompts list each role's owned tools and enforce the turn contract (`thought` + a single
`action`, where the action is a tool call or a `TaskComplete`).

## How prompts are used by agents

At each turn the runtime builds a fresh user message from current episode state — the plan block,
shared memory, the last observation, and the handles/artifacts available to the current step —
via the `build_orc_plan_message`, `build_orc_decide_message`, and `build_specialist_message`
helpers, then prepends the role's system prompt. Messages are reconstructed from typed state each
turn rather than appended as unconstrained free text. Released SFT rows contain the prompt snapshot
used to build their targets, and their emitted tool calls validate against the same runtime schemas.
The executable prompt builders and tool registry are authoritative at inference.
