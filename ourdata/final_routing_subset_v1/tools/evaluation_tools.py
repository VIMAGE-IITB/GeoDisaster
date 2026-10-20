from __future__ import annotations


def Terminate(ans: str, **_: object) -> dict:
    return {"final_answer": ans, "terminated": True}

