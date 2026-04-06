"""
Typed Pydantic models for the Customer Support Ticket Triage (CSTT) OpenEnv environment.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TicketContext(BaseModel):
    """The raw ticket the agent receives each step."""
    ticket_id: str
    subject: str
    body: str
    sender: str
    sender_tier: str = Field(..., description="free | basic | enterprise")
    product: str
    timestamp: str


class Observation(BaseModel):
    """
    Full observation returned by reset() and step().
    Contains the current ticket plus task metadata.
    """
    ticket: TicketContext
    task_id: str = Field(..., description="task1_triage | task2_routing | task3_resolution")
    step: int = Field(..., description="Current step index (0-based)")
    max_steps: int = Field(..., description="Total tickets in this episode")
    episode: int = Field(default=1)
    task_description: str
    action_schema: Dict[str, Any] = Field(..., description="Required fields for this task")


class Action(BaseModel):
    """
    Agent action submitted via step().
    Not all fields are required for every task — see action_schema in Observation.
    """
    priority: Optional[str] = Field(
        None,
        description="Ticket severity: P0 (critical) | P1 (high) | P2 (medium) | P3 (low)"
    )
    department: Optional[str] = Field(
        None,
        description="Routing target: engineering | billing | support | security | product"
    )
    sentiment: Optional[str] = Field(
        None,
        description="Customer emotional tone: positive | neutral | negative | angry"
    )
    escalate: Optional[bool] = Field(
        None,
        description="Whether ticket requires immediate human manager escalation"
    )
    response_draft: Optional[str] = Field(
        None,
        description="Full drafted email response to send to the customer"
    )


class Reward(BaseModel):
    """Per-step reward with granular breakdown and human-readable feedback."""
    score: float = Field(..., ge=0.0, le=1.0)
    breakdown: Dict[str, float]
    feedback: str
    done: bool


class StepResult(BaseModel):
    """Return type of step() — follows (obs, reward, done, info) convention."""
    observation: Optional[Observation]   # None when episode ends
    reward: Reward
    done: bool
    info: Dict[str, Any]


class EnvironmentState(BaseModel):
    """Snapshot returned by state()."""
    task_id: str
    episode: int
    step: int
    max_steps: int
    total_reward: float
    done: bool
    step_scores: List[float]
    current_ticket_idx: int
