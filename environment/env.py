"""
SupportTriageEnv — Customer Support Ticket Triage OpenEnv environment.

Implements the full OpenEnv interface:
    reset(task_id)  → Observation
    step(action)    → StepResult  (observation, reward, done, info)
    state()         → EnvironmentState
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

from .models import (
    Action,
    EnvironmentState,
    Observation,
    Reward,
    StepResult,
    TicketContext,
)
from .data import TASK_TICKETS, TASK_DESCRIPTIONS, ACTION_SCHEMAS
from .graders import GRADERS

VALID_TASKS = list(TASK_TICKETS.keys())


class SupportTriageEnv:
    """
    Customer Support Ticket Triage environment.

    An agent receives one support ticket per step and must correctly
    triage it according to the active task's requirements.  Episodes
    consist of a fixed queue of tickets; the environment resets cleanly
    between episodes.

    Tasks
    -----
    task1_triage      (easy)   — classify priority + department
    task2_routing     (medium) — + sentiment + escalation decision
    task3_resolution  (hard)   — + draft a full professional response
    """

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self._task_id: str = "task1_triage"
        self._tickets: List[Dict[str, Any]] = []
        self._step: int = 0
        self._episode: int = 0
        self._step_scores: List[float] = []
        self._total_reward: float = 0.0
        self._done: bool = True  # starts as done until reset() is called

    def reset(self, task_id: str = "task1_triage") -> Observation:
        """
        Reset the environment for a new episode.

        Parameters
        ----------
        task_id : str
            One of task1_triage | task2_routing | task3_resolution

        Returns
        -------
        Observation
            The first ticket in the episode queue.
        """
        if task_id not in VALID_TASKS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. Valid tasks: {VALID_TASKS}"
            )

        self._task_id = task_id
        self._tickets = TASK_TICKETS[task_id]
        self._step = 0
        self._episode += 1
        self._step_scores = []
        self._total_reward = 0.0
        self._done = False

        return self._make_observation()

    def step(self, action: Action) -> StepResult:
        """
        Submit an action for the current ticket.

        The environment grades the action, advances to the next ticket,
        and returns the result.  Calling step() after done=True raises
        RuntimeError.
        """
        if self._done:
            raise RuntimeError(
                "Episode is finished. Call reset() to start a new episode."
            )

        ticket_expected = self._tickets[self._step]["expected"]
        grader = GRADERS[self._task_id]
        score, breakdown, feedback = grader(action, ticket_expected)

        self._step_scores.append(score)
        self._total_reward += score
        self._step += 1

        is_done = self._step >= len(self._tickets)
        self._done = is_done

        next_obs: Optional[Observation] = None
        if not is_done:
            next_obs = self._make_observation()

        reward = Reward(
            score=score,
            breakdown=breakdown,
            feedback=feedback,
            done=is_done,
        )

        mean_so_far = self._total_reward / self._step
        info: Dict[str, Any] = {
            "ticket_id": self._tickets[self._step - 1]["ticket_id"],
            "step": self._step - 1,
            "episode": self._episode,
            "mean_reward": round(max(0.001, min(0.999, mean_so_far)), 6),
            "expected": ticket_expected,
        }
        if is_done:
            raw = self._total_reward / len(self._tickets)
            info["episode_score"] = round(max(0.001, min(0.999, raw)), 6)

        return StepResult(
            observation=next_obs,
            reward=reward,
            done=is_done,
            info=info,
        )

    def state(self) -> EnvironmentState:
        """Return a full snapshot of the current environment state."""
        return EnvironmentState(
            task_id=self._task_id,
            episode=self._episode,
            step=self._step,
            max_steps=len(self._tickets) if self._tickets else 0,
            total_reward=round(max(0.001, min(0.999, self._total_reward / max(1, self._step))), 6),
            done=self._done,
            step_scores=list(self._step_scores),
            current_ticket_idx=self._step,
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _make_observation(self) -> Observation:
        raw = self._tickets[self._step]
        ticket = TicketContext(
            ticket_id=raw["ticket_id"],
            subject=raw["subject"],
            body=raw["body"],
            sender=raw["sender"],
            sender_tier=raw["sender_tier"],
            product=raw["product"],
            timestamp=raw["timestamp"],
        )
        return Observation(
            ticket=ticket,
            task_id=self._task_id,
            step=self._step,
            max_steps=len(self._tickets),
            episode=self._episode,
            task_description=TASK_DESCRIPTIONS[self._task_id],
            action_schema=ACTION_SCHEMAS[self._task_id],
        )

    # ── Convenience ──────────────────────────────────────────────────────────

    @property
    def task_ids(self) -> List[str]:
        return VALID_TASKS

    def __repr__(self) -> str:
        return (
            f"SupportTriageEnv(task={self._task_id}, "
            f"step={self._step}/{len(self._tickets)}, "
            f"episode={self._episode}, "
            f"done={self._done})"
        )
