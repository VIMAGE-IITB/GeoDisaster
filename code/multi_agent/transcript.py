"""Render a conversation_<i>.json into a clean user/agent transcript.

The output is a single .md file showing:

* the user question
* every turn in order with role tag (USER / AGENT-ORC / AGENT-GA / ...)
* the user message the runtime fed in
* the model's thought + action
* the tool result (when applicable)
* the final answer

Usage::

    python -m multi_agent.transcript runs/m9_second_run/conversation_3.json
    python -m multi_agent.transcript runs/m9_second_run/   # all files in dir
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any


_HR = "-" * 72
_DHR = "=" * 72


def _indent(text: str, prefix: str = "  ") -> str:
    if text is None:
        return prefix + "(none)"
    return "\n".join(prefix + line for line in str(text).splitlines())


def _format_action(action: dict[str, Any]) -> str:
    tool = action.get("tool", "?")
    if tool == "TaskComplete":
        return f"TaskComplete  status={action.get('status')}  summary={action.get('summary')!r}"
    if tool == "CallAgent":
        return f"CallAgent  step_id={action.get('step_id')}"
    if tool == "Terminate":
        return f"Terminate  answer_ref={action.get('answer_ref')!r}"
    if tool == "Plan":
        steps = action.get("steps", [])
        rendered = "\n".join(
            f"      {s.get('id')} {s.get('agent')} | {s.get('subtask')}  needs={s.get('needs')}"
            for s in steps
        )
        return f"Plan  goal={action.get('goal')!r}\n    steps:\n{rendered}"
    args = action.get("args", {})
    return f"{tool}({json.dumps(args, ensure_ascii=False)})"


def _render_question(log: dict[str, Any]) -> list[str]:
    return [
        _DHR,
        f"QUESTION (idx={log.get('idx')}):",
        _indent(log["question"]),
        f"images: {log.get('images') or '(none)'}",
        f"question_facts: {log.get('question_facts') or {}}",
        _DHR,
        "",
    ]


def _render_event_user(event: dict[str, Any]) -> list[str]:
    payload = event.get("payload", {})
    user_msg = payload.get("user_message")
    if not user_msg:
        return []
    role_tag = f"USER -> {event['role']}"
    if event.get("step_id"):
        role_tag += f" (step {event['step_id']})"
    if event.get("turn") is not None:
        role_tag += f" turn {event['turn']}"
    return [
        f">>> {role_tag}:",
        _indent(user_msg, prefix="    "),
        "",
    ]


def _render_event_agent(event: dict[str, Any]) -> list[str]:
    payload = event.get("payload", {})
    role = event["role"]
    role_tag = f"AGENT {role}"
    if event.get("step_id"):
        role_tag += f" (step {event['step_id']})"
    if event.get("turn") is not None:
        role_tag += f" turn {event['turn']}"

    out: list[str] = [f"<<< {role_tag}:"]
    if event.get("phase") == "plan":
        plan = payload.get("plan")
        if plan is not None:
            out.append(f"    thought: (Plan)")
            out.append(f"    action : Plan  goal={plan.get('goal')!r}")
            out.append(f"    steps  :")
            for step in plan.get("steps", []):
                out.append(
                    f"      {step.get('id')} {step.get('agent')} | "
                    f"{step.get('subtask')}  needs={step.get('needs')}"
                )
        elif payload.get("error"):
            out.append(f"    ERROR  : {payload['error']}")
    elif event.get("phase") in ("decide", "delegate", "replan", "terminate"):
        decision = payload.get("decision")
        if decision is not None:
            out.append(f"    action : {_format_action(decision)}")
        elif payload.get("answer_ref"):
            out.append(f"    action : Terminate  answer_ref={payload['answer_ref']!r}")
        elif payload.get("error"):
            out.append(f"    ERROR  : {payload['error']}")
    elif event.get("phase") == "step":
        turn = payload.get("turn")
        if turn is not None:
            out.append(f"    thought: {turn.get('thought')}")
            out.append(f"    action : {_format_action(turn.get('action', {}))}")
        elif payload.get("error"):
            out.append(f"    ERROR  : {payload['error']}")
    elif event.get("phase") == "complete":
        out.append(f"    status : {payload.get('status')}")
        out.append(f"    summary: {payload.get('summary')}")

    out.append("")
    return out


def _render_event_tool(event: dict[str, Any]) -> list[str]:
    payload = event.get("payload", {})
    role_tag = "TOOL"
    if event.get("step_id"):
        role_tag += f" (step {event['step_id']})"
    if event.get("turn") is not None:
        role_tag += f" turn {event['turn']}"
    out = [
        f"--- {role_tag}: {payload.get('tool')}",
        f"    input  : {json.dumps(payload.get('args', {}), ensure_ascii=False)}",
        f"    ok     : {payload.get('ok')}",
        f"    text   : {payload.get('text')}",
    ]
    produced = payload.get("produced") or {}
    if produced.get("artifacts"):
        out.append(
            "    artifacts: " + ", ".join(
                f"{a.get('handle')}[{a.get('kind')}]" for a in produced["artifacts"]
            )
        )
    if produced.get("layers"):
        out.append(
            "    layers : " + ", ".join(
                f"{lay.get('gpkg')}/{lay.get('name')}" for lay in produced["layers"]
            )
        )
    if produced.get("fields"):
        out.append("    fields : " + ", ".join(f"{k}={v}" for k, v in produced["fields"].items()))
    out.append("")
    return out


def render(log: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.extend(_render_question(log))

    history = log.get("trace", [])
    for event in history:
        role = event.get("role")
        phase = event.get("phase")
        if role == "user" and phase == "question":
            continue  # already rendered as the QUESTION header

        # First the user message (if this turn had one)
        lines.extend(_render_event_user(event))

        # Then the agent action / tool result
        if role == "tool":
            lines.extend(_render_event_tool(event))
        elif role in ("ORC", "GA", "PA", "VRA"):
            lines.extend(_render_event_agent(event))
        else:
            # system events etc.
            lines.append(f"[{role} :: {phase}] {json.dumps(event.get('payload', {}), default=str)[:200]}")
            lines.append("")
        lines.append(_HR)

    lines.extend(
        [
            "",
            _DHR,
            f"FINAL ANSWER : {log.get('final_answer')!r}",
            f"STATUS       : {log.get('status')}  abort_reason={log.get('abort_reason')}",
            f"WALL TIME    : {log.get('wall_time_s')}s",
            f"EXPECTED KW  : {log.get('expected_keywords')}",
            _DHR,
        ]
    )
    return "\n".join(lines)


def _collect_paths(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(target.glob("conversation_*.json"))
    raise FileNotFoundError(f"no such path: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render conversation JSON as user/agent transcript.")
    parser.add_argument("path", type=str, help="conversation_<i>.json or a directory of them")
    parser.add_argument("--out", type=str, default="", help="output file (single input) or directory (multi). Default: write next to source")
    args = parser.parse_args()

    paths = _collect_paths(Path(args.path))
    if not paths:
        print(f"no conversation_*.json found under {args.path}", file=sys.stderr)
        sys.exit(1)

    for src in paths:
        log = json.loads(src.read_text(encoding="utf-8"))
        text = render(log)
        if args.out:
            out_path = Path(args.out)
            if len(paths) > 1:
                out_path.mkdir(parents=True, exist_ok=True)
                out_path = out_path / (src.stem + ".transcript.md")
        else:
            out_path = src.with_suffix(".transcript.md")
        out_path.write_text(text, encoding="utf-8")
        print(f"  wrote {out_path}  ({len(text)} bytes)")


if __name__ == "__main__":
    main()
