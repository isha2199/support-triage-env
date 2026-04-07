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
