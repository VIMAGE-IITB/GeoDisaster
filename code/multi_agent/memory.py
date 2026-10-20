"""Token-budget guard for rendered prompt sections.

We use a deterministic word-based estimator instead of pulling a tokenizer.
Real tokenization is at most ~1.3x word count for English; word-based bounds
are a strict overestimate, which is what we want for safety.
"""

from __future__ import annotations

import re

_TOKEN_RATIO = 1.3  # word -> token estimator
_TRUNCATION_MARKER = "  ... (truncated, {n} more lines not shown)"


def estimate_tokens(text: str) -> int:
    words = re.findall(r"\S+", text or "")
    return int(len(words) * _TOKEN_RATIO) + 1


def cap_section(body: str, max_tokens: int) -> str:
    """Truncate ``body`` line-wise until it fits under ``max_tokens``.

    Always keeps the first line (header). Appends an explicit truncation marker
    on overflow so the model knows context is partial.
    """
    if estimate_tokens(body) <= max_tokens:
        return body
    lines = body.splitlines()
    if not lines:
        return body
    head = [lines[0]]
    rest = lines[1:]
    kept_tail: list[str] = []
    # Keep newest lines (end) — most relevant.
    for line in reversed(rest):
        candidate = "\n".join(head + [_TRUNCATION_MARKER.format(n=0)] + list(reversed(kept_tail + [line])))
        if estimate_tokens(candidate) > max_tokens:
            break
        kept_tail.append(line)
    dropped = len(rest) - len(kept_tail)
    return "\n".join(
        head
        + ([_TRUNCATION_MARKER.format(n=dropped)] if dropped > 0 else [])
        + list(reversed(kept_tail))
    )


# Section budgets. Tuning knobs only.
BUDGETS = {
    "PRIOR ARTIFACTS": 1500,
    "Δ STATE SINCE LAST TURN": 500,
    "SHARED MEMORY": 2000,
    "LAST OBSERVATION": 1000,
}
