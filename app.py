"""
FastAPI application exposing the SupportTriageEnv as an HTTP API.
Deployed to Hugging Face Spaces as a Docker Space on port 7860.

Endpoints
---------
GET  /                    Health check + environment info
GET  /tasks               List available tasks
POST /reset               Reset environment for a task  { "task_id": "..." }
POST /step                Submit action                  { Action fields }
GET  /state               Current environment snapshot
GET  /openenv.yaml        Serve the OpenEnv spec file
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from environment import SupportTriageEnv
from environment.models import Action, Observation, EnvironmentState, StepResult

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Customer Support Ticket Triage — OpenEnv",
    description=(
        "An OpenEnv-compliant environment for training and evaluating AI agents "
        "on real-world customer support triage tasks."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# ── Single shared environment instance (stateful per deployment) ─────────────
# For multi-user scenarios use session tokens and a dict of envs.
_env = SupportTriageEnv()


# ── Request / Response schemas ───────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task1_triage"


class ResetResponse(BaseModel):
    observation: Observation
    task_id: str
    episode: int


class StepResponse(BaseModel):
    observation: Optional[Observation]
    reward: float
    reward_breakdown: Dict[str, float]
    feedback: str
    done: bool
    info: Dict[str, Any]


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", tags=["meta"])
def root() -> HTMLResponse:
    ui_path = Path(__file__).parent / "static" / "index.html"
    if ui_path.exists():
        return HTMLResponse(content=ui_path.read_text(), status_code=200)
    return HTMLResponse(content='<a href="/docs">API Docs</a>', status_code=200)

@app.get("/api", tags=["meta"])
def api_info() -> Dict[str, Any]:
    return {
        "name": "customer-support-ticket-triage",
        "version": "1.0.0",
        "description": "OpenEnv environment for customer support ticket triage",
        "tasks": _env.task_ids,
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/openenv.yaml", "/docs"],
        "status": "ok",
    }


@app.get("/tasks", tags=["meta"])
def list_tasks() -> Dict[str, Any]:
    from environment.data import TASK_DESCRIPTIONS, ACTION_SCHEMAS
    return {
        task_id: {
            "description": TASK_DESCRIPTIONS[task_id],
            "action_schema": ACTION_SCHEMAS[task_id],
            "difficulty": {"task1_triage": "easy", "task2_routing": "medium", "task3_resolution": "hard"}[task_id],
            "num_tickets": 5,
        }
        for task_id in _env.task_ids
    }


@app.post("/reset", tags=["env"])
def reset(request: ResetRequest) -> ResetResponse:
    try:
        obs = _env.reset(task_id=request.task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    state = _env.state()
    return ResetResponse(observation=obs, task_id=request.task_id, episode=state.episode)


@app.post("/step", tags=["env"])
def step(action: Action) -> StepResponse:
    try:
        result: StepResult = _env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StepResponse(
        observation=result.observation,
        reward=result.reward.score,
        reward_breakdown=result.reward.breakdown,
        feedback=result.reward.feedback,
        done=result.done,
        info=result.info,
    )


@app.get("/state", tags=["env"])
def state() -> EnvironmentState:
    return _env.state()


# ── Demo endpoint ─────────────────────────────────────────────────────────────

class DemoRequest(BaseModel):
    ticket_id: str
    task_id: str = "task3_resolution"


@app.post("/demo/analyze", tags=["demo"])
def demo_analyze(req: DemoRequest) -> Dict[str, Any]:
    """
    Demo endpoint: processes a single pre-built ticket with a smart mock agent
    and returns the full triage result plus a log of API calls made.
    Used by the interactive frontend demo.
    """
    from environment.data import TASK_TICKETS
    from environment.env import SupportTriageEnv as _Env

    # Find the ticket
    ticket_data = None
    resolved_task = req.task_id
    for tid, tickets in TASK_TICKETS.items():
        for t in tickets:
            if t["ticket_id"] == req.ticket_id:
                ticket_data = t
                resolved_task = tid
                break
        if ticket_data:
            break

    if not ticket_data:
        raise HTTPException(status_code=404, detail=f"Ticket {req.ticket_id} not found")

    # Fresh env per demo call (isolated from the shared env)
    demo_env = _Env()
    api_log = []

    # ── Step 1: reset ──
    reset_req = {"task_id": resolved_task}
    obs = demo_env.reset(resolved_task)
    api_log.append({
        "step": 1,
        "method": "POST",
        "path": "/reset",
        "request": reset_req,
        "response": {"task_id": resolved_task, "episode": 1,
                     "ticket_id": obs.ticket.ticket_id},
        "status": 200,
    })

    # ── Step 2: build action from expected values ──
    exp = ticket_data["expected"]
    response_draft = _build_demo_response(ticket_data)

    action = Action(
        priority=exp.get("priority"),
        department=exp.get("department"),
        sentiment=exp.get("sentiment"),
        escalate=exp.get("escalate"),
        response_draft=response_draft,
    )

    action_dict = {k: v for k, v in action.model_dump().items() if v is not None}
    result = demo_env.step(action)

    api_log.append({
        "step": 2,
        "method": "POST",
        "path": "/step",
        "request": action_dict,
        "response": {
            "reward": result.reward.score,
            "breakdown": result.reward.breakdown,
            "feedback": result.reward.feedback,
            "done": result.done,
        },
        "status": 200,
    })

    return {
        "ticket": ticket_data,
        "task_id": resolved_task,
        "triage": action_dict,
        "score": result.reward.score,
        "breakdown": result.reward.breakdown,
        "feedback": result.reward.feedback,
        "response_draft": response_draft,
        "api_log": api_log,
    }


def _build_demo_response(ticket: Dict[str, Any]) -> str:
    """Generate a realistic canned response for demo tickets."""
    responses = {
        "TK-3001": (
            "Dear Backend Team,\n\nThank you for reaching out. We sincerely apologize for the "
            "rate limit issues you're experiencing with 429 errors on our API.\n\nDespite your "
            "usage appearing to be within quota, our systems show an anomaly in the rate limit "
            "counter for your account. Our engineering team has identified this as a known issue "
            "affecting a small number of accounts after last night's deployment.\n\nWe have "
            "already reset your rate limit counter and the issue should be resolved immediately. "
            "Please retry your API calls now. We will monitor your account closely over the next "
            "24 hours and proactively reach out if any further issues arise.\n\nBest regards,\n"
            "Support Team"
        ),
        "TK-3002": (
            "Dear Customer,\n\nThank you for contacting us. We sincerely apologize for this "
            "billing error — being charged after cancellation is completely unacceptable and we "
            "understand the frustration, especially with the overdraft impact.\n\nWe have verified "
            "your cancellation confirmation (ref: CANCEL-98234) dated December 28th. The $299 "
            "charge on January 1st was applied in error. We have initiated a full refund to your "
            "original payment method, which will appear within 3-5 business days.\n\nAs an "
            "apology, we have also applied a $20 credit to offset any overdraft fees. Please "
            "don't hesitate to reach out if you need anything else.\n\nBest regards,\nBilling Team"
        ),
        "TK-3003": (
            "Dear Team Lead,\n\nThank you for reaching out. We understand the urgency — having "
            "a team member unable to join on their first day is a real blocker.\n\nInvitation "
            "links expire after 7 days by default. Please follow these steps to resolve this "
            "immediately:\n1. Go to Settings → Team Members → Pending Invitations\n2. Revoke "
            "the existing invitation for the affected member\n3. Send a fresh invitation\n\nThe "
            "new link will be valid for 7 days. If the issue persists after resending, please "
            "share the affected email address and we will manually provision access within the "
            "hour.\n\nBest regards,\nSupport Team"
        ),
        "TK-3004": (
            "Dear Security Researcher,\n\nThank you for this responsible disclosure. We take "
            "security vulnerabilities extremely seriously and we appreciate you giving us the "
            "opportunity to address this before public disclosure.\n\nWe have immediately "
            "escalated this SQL injection vulnerability in /api/v2/search to our security "
            "team. A patch is in development and will be deployed within 48 hours as per our "
            "security policy.\n\nWe will investigate the extent of any data exposure and notify "
            "affected users if required. You will receive a full timeline update within 4 hours. "
            "We would like to offer a bug bounty reward for this finding — our team will follow "
            "up with details.\n\nBest regards,\nSecurity Team"
        ),
        "TK-3005": (
            "Dear Director,\n\nThank you for bringing this to our attention. We sincerely "
            "apologize for the 6-hour outage on January 10th and fully acknowledge that this "
            "breached your SLA entitlement.\n\nWe confirm that a 15% service credit per section "
            "8.2 of your contract will be applied to your next invoice. A full post-mortem "
            "report detailing root cause, timeline, and remediation steps will be delivered "
            "within 5 business days.\n\nWe have also implemented additional redundancy measures "
            "to prevent a recurrence. Your account has been flagged for priority support. Please "
            "let us know if you have any questions about the credit or the incident.\n\nBest "
            "regards,\nEnterprise Support Team"
        ),
    }
    tid = ticket.get("ticket_id", "")
    if tid in responses:
        return responses[tid]
    # Fallback generic response
    exp = ticket.get("expected", {})
    return (
        f"Dear Customer,\n\nThank you for reaching out to our support team. "
        f"We apologize for the inconvenience you are experiencing.\n\n"
        f"We have reviewed your ticket and our {exp.get('department','support')} team "
        f"will investigate this as a {exp.get('priority','P2')} priority issue. "
        f"You can expect a follow-up within 1 business day with a full update.\n\n"
        f"Please don't hesitate to reach out if you need anything else.\n\n"
        f"Best regards,\nSupport Team"
    )


@app.get("/openenv.yaml", tags=["meta"], response_class=PlainTextResponse)
def serve_openenv_yaml() -> str:
    spec_path = Path(__file__).parent / "openenv.yaml"
    if spec_path.exists():
        return spec_path.read_text()
    raise HTTPException(status_code=404, detail="openenv.yaml not found")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
