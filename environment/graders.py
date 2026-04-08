"""
Deterministic graders for all three tasks.
Each grader returns (score: float, breakdown: dict, feedback: str).
All scores are strictly in (0.0, 1.0) — endpoints excluded.
"""
from __future__ import annotations
import re
from typing import Any, Dict, Tuple

from .models import Action

_SCORE_MIN = 0.01
_SCORE_MAX = 0.99


def _clip(score: float) -> float:
    """Clamp score to strictly open interval (0, 1) as required by the evaluator."""
    return max(_SCORE_MIN, min(_SCORE_MAX, score))


def _clip_breakdown(breakdown: Dict[str, float]) -> Dict[str, float]:
    """Ensure breakdown components and their sum are strictly in (0, 1).
    1. Floor each component at _SCORE_MIN (eliminates exact 0.0).
    2. If the sum >= _SCORE_MAX, scale all components proportionally
       so the sum equals _SCORE_MAX (eliminates exact 1.0 on the sum).
    """
    floored = {k: max(_SCORE_MIN, v) for k, v in breakdown.items()}
    total = sum(floored.values())
    if total >= _SCORE_MAX:
        factor = _SCORE_MAX / total
        return {k: round(v * factor, 6) for k, v in floored.items()}
    return floored

# Priority adjacency — being one level off earns partial credit
_PRIORITY_RANK: Dict[str, int] = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


def _priority_score(predicted: str | None, expected: str, full_weight: float) -> float:
    """Return full_weight for exact match, half for adjacent, 0 otherwise."""
    if predicted is None:
        return 0.0
    if predicted == expected:
        return full_weight
    if abs(_PRIORITY_RANK.get(predicted, -99) - _PRIORITY_RANK.get(expected, -99)) == 1:
        return full_weight * 0.5
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — Quick Triage  (easy)
# Weights: priority 0.50, department 0.50
# ─────────────────────────────────────────────────────────────────────────────
def grade_task1(action: Action, expected: Dict[str, Any]) -> Tuple[float, Dict[str, float], str]:
    breakdown: Dict[str, float] = {}

    # Priority (0.50)
    breakdown["priority"] = _priority_score(action.priority, expected["priority"], 0.50)

    # Department (0.50)
    breakdown["department"] = 0.50 if action.department == expected["department"] else 0.0

    score = _clip(sum(breakdown.values()))
    parts = []
    parts.append(
        f"priority={'✓' if breakdown['priority'] == 0.50 else ('~' if breakdown['priority'] > 0.001 else '✗')} "
        f"(got={action.priority}, want={expected['priority']})"
    )
    parts.append(
        f"department={'✓' if breakdown['department'] == 0.50 else '✗'} "
        f"(got={action.department}, want={expected['department']})"
    )
    feedback = " | ".join(parts)
    return round(score, 6), _clip_breakdown(breakdown), feedback


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — Smart Routing  (medium)
# Weights: priority 0.30, department 0.30, sentiment 0.20, escalate 0.20
# ─────────────────────────────────────────────────────────────────────────────
def grade_task2(action: Action, expected: Dict[str, Any]) -> Tuple[float, Dict[str, float], str]:
    breakdown: Dict[str, float] = {}

    # Priority (0.30)
    breakdown["priority"] = _priority_score(action.priority, expected["priority"], 0.30)

    # Department (0.30)
    breakdown["department"] = 0.30 if action.department == expected["department"] else 0.0

    # Sentiment (0.20)
    breakdown["sentiment"] = 0.20 if action.sentiment == expected["sentiment"] else 0.0

    # Escalate (0.20)
    got_escalate = action.escalate
    exp_escalate = expected["escalate"]
    breakdown["escalate"] = 0.20 if got_escalate == exp_escalate else 0.0

    score = _clip(sum(breakdown.values()))
    parts = [
        f"priority={'✓' if breakdown['priority'] == 0.30 else ('~' if breakdown['priority'] > 0.001 else '✗')}({action.priority}/{expected['priority']})",
        f"dept={'✓' if breakdown['department'] == 0.30 else '✗'}({action.department}/{expected['department']})",
        f"sentiment={'✓' if breakdown['sentiment'] == 0.20 else '✗'}({action.sentiment}/{expected['sentiment']})",
        f"escalate={'✓' if breakdown['escalate'] == 0.20 else '✗'}({got_escalate}/{exp_escalate})",
    ]
    feedback = " | ".join(parts)
    return round(score, 6), _clip_breakdown(breakdown), feedback


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — Full Resolution  (hard)
# Weights: priority 0.10, department 0.10, response_quality 0.80
#
# Response quality sub-scores (total 0.80):
#   greeting        0.06  — opens with greeting / thank-you
#   acknowledgment  0.10  — explicitly acknowledges the reported issue
#   solution        0.30  — covers ≥50% of expected solution keywords
#   next_steps      0.14  — explains what will happen next / what to do
#   closing         0.08  — closes professionally
#   length          0.12  — 80–300 words (proportional within ±50%)
# ─────────────────────────────────────────────────────────────────────────────
def grade_task3(action: Action, expected: Dict[str, Any]) -> Tuple[float, Dict[str, float], str]:
    breakdown: Dict[str, float] = {}

    # Priority (0.10)
    breakdown["priority"] = _priority_score(action.priority, expected["priority"], 0.10)

    # Department (0.10)
    breakdown["department"] = 0.10 if action.department == expected["department"] else 0.0

    # ── Response quality ─────────────────────────────────────────────────────
    response = action.response_draft or ""
    rl = response.lower()
    words = len(response.split())

    # Greeting (0.06)
    greeting_hits = any(
        phrase in rl for phrase in [
            "dear ", "hello", "hi ", "hi,", "thank you for contacting",
            "thank you for reaching out", "thanks for reaching out",
            "thank you for your message",
        ]
    )
    breakdown["resp_greeting"] = 0.06 if greeting_hits else 0.0

    # Acknowledgment (0.10)
    ack_hits = any(
        phrase in rl for phrase in [
            "apologize", "sorry", "understand", "acknowledge",
            "received your", "we hear you", "i understand",
            "we understand", "frustrat",
        ]
    )
    breakdown["resp_acknowledgment"] = 0.10 if ack_hits else 0.0

    # Solution keyword coverage (0.30)
    solution_keywords = expected.get("solution_keywords", [])
    if solution_keywords:
        hits = sum(1 for kw in solution_keywords if kw.lower() in rl)
        coverage = hits / len(solution_keywords)
        # Partial credit: proportional above 0.5 threshold
        if coverage >= 0.5:
            breakdown["resp_solution"] = round(0.30 * coverage, 4)
        elif coverage >= 0.25:
            breakdown["resp_solution"] = round(0.15 * coverage / 0.5, 4)
        else:
            breakdown["resp_solution"] = 0.0
    else:
        breakdown["resp_solution"] = 0.15  # no keywords to check → half credit

    # Next steps (0.14)
    next_step_hits = any(
        phrase in rl for phrase in [
            "next step", "please ", "you can ", "we will ", "we'll ",
            "follow up", "follow-up", "let us know", "reach out",
            "contact us", "business day", "within ", "our team will",
        ]
    )
    breakdown["resp_next_steps"] = 0.14 if next_step_hits else 0.0

    # Closing (0.08)
    closing_hits = any(
        phrase in rl for phrase in [
            "sincerely", "regards", "best regards", "kind regards",
            "thank you", "thanks,", "please don't hesitate",
            "happy to help", "here to help",
        ]
    )
    breakdown["resp_closing"] = 0.08 if closing_hits else 0.0

    # Length score (0.12) — proportional within [80, 300] ± leniency
    if 80 <= words <= 300:
        breakdown["resp_length"] = 0.12
    elif 50 <= words < 80:
        breakdown["resp_length"] = round(0.12 * (words - 50) / 30, 4)
    elif 300 < words <= 400:
        breakdown["resp_length"] = round(0.12 * (400 - words) / 100, 4)
    else:
        breakdown["resp_length"] = 0.0

    score = _clip(sum(breakdown.values()))

    resp_score = sum(v for k, v in breakdown.items() if k.startswith("resp_"))
    parts = [
        f"priority={'✓' if breakdown['priority'] == 0.10 else '~' if breakdown['priority'] > 0.001 else '✗'}",
        f"dept={'✓' if breakdown['department'] == 0.10 else '✗'}",
        f"response={resp_score:.2f}/0.80 "
        f"[greet={'✓' if breakdown['resp_greeting'] > 0.001 else '✗'} "
        f"ack={'✓' if breakdown['resp_acknowledgment'] > 0.001 else '✗'} "
        f"sol={breakdown['resp_solution']:.2f}/0.30 "
        f"steps={'✓' if breakdown['resp_next_steps'] > 0.001 else '✗'} "
        f"close={'✓' if breakdown['resp_closing'] > 0.001 else '✗'} "
        f"len={words}w={'✓' if breakdown['resp_length'] == 0.12 else '~' if breakdown['resp_length'] > 0.001 else '✗'}]",
    ]
    feedback = " | ".join(parts)
    return round(score, 6), _clip_breakdown(breakdown), feedback


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────
GRADERS = {
    "task1_triage": grade_task1,
    "task2_routing": grade_task2,
    "task3_resolution": grade_task3,
}
