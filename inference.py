"""
Baseline inference script for the Customer Support Ticket Triage OpenEnv.

Runs an LLM agent against all 3 tasks and emits structured logs in the
required [START] / [STEP] / [END] format for automated evaluation.

Environment variables (required)
---------------------------------
API_BASE_URL   The OpenAI-compatible API base URL for the LLM.
MODEL_NAME     The model identifier (e.g. "gpt-4o", "meta-llama/Llama-3-70b-instruct").
HF_TOKEN       API key used to authenticate with the LLM endpoint.

Optional
--------
OPENENV_MODE   "local" (default) uses the env class directly.
               "remote" calls a running HTTP server at OPENENV_API_URL.
OPENENV_API_URL  Base URL of a running SupportTriageEnv server (default: http://localhost:7860).
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ── Environment import (local mode) ──────────────────────────────────────────
from environment import SupportTriageEnv
from environment.models import Action, Observation

# ── Configuration ─────────────────────────────────────────────────────────────
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: str = os.environ.get("HF_TOKEN")
LOCAL_IMAGE_NAME: str = os.environ.get("LOCAL_IMAGE_NAME")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set.")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

TASKS: List[str] = ["task1_triage", "task2_routing", "task3_resolution"]
MAX_RETRIES = 3


# ── Prompt construction ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert customer support triage specialist with 10+ years of experience.
You analyze support tickets and make precise triage decisions.

You must respond ONLY with a valid JSON object — no markdown, no explanation, no code blocks.
The JSON must contain exactly the fields specified in the task instructions."""


def build_user_prompt(obs: Observation) -> str:
    ticket = obs.ticket
    schema_str = json.dumps(obs.action_schema, indent=2)
    return f"""TASK: {obs.task_description}

TICKET DETAILS:
  ID:          {ticket.ticket_id}
  Subject:     {ticket.subject}
  From:        {ticket.sender} [{ticket.sender_tier} tier]
  Product:     {ticket.product}
  Timestamp:   {ticket.timestamp}

MESSAGE:
{ticket.body}

REQUIRED ACTION SCHEMA:
{schema_str}

Respond with a JSON object containing exactly these fields. For response_draft (if required),
write a complete, professional email response (80-300 words).

JSON response:"""


# ── LLM call with retry ───────────────────────────────────────────────────────

def call_llm(obs: Observation) -> Dict[str, Any]:
    """Call the LLM and parse the JSON action. Retries up to MAX_RETRIES times."""
    user_prompt = build_user_prompt(obs)
    last_error: Optional[Exception] = None

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            content = response.choices[0].message.content or ""
            content = content.strip()

            # Strip markdown code fences if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            return json.loads(content)

        except json.JSONDecodeError as e:
            last_error = e
            time.sleep(1)
        except Exception as e:
            last_error = e
            time.sleep(2)

    # Fallback: return empty action on persistent failure
    print(
        f"[WARN] LLM call failed after {MAX_RETRIES} retries: {last_error}",
        file=sys.stderr,
    )
    return {}


def dict_to_action(raw: Dict[str, Any]) -> Action:
    """Convert raw LLM JSON dict to a validated Action object."""
    try:
        return Action(**{k: v for k, v in raw.items() if k in Action.model_fields})
    except Exception:
        return Action()


# ── Structured logging helpers ────────────────────────────────────────────────

def log_start(task: str, episode: int) -> None:
    print(f"[START] task={task} episode={episode}", flush=True)


def log_step(
    step: int,
    action: Dict[str, Any],
    observation: Optional[Dict[str, Any]],
    reward: float,
    done: bool,
) -> None:
    obs_str = json.dumps(observation) if observation else "null"
    action_str = json.dumps(action)
    print(
        f"[STEP] step={step} action={action_str} observation={obs_str} "
        f"reward={reward:.4f} done={str(done).lower()}",
        flush=True,
    )


def log_end(task: str, episode: int, total_reward: float, steps: int) -> None:
    print(
        f"[END] task={task} episode={episode} total_reward={total_reward:.4f} steps={steps}",
        flush=True,
    )


# ── Single episode runner ─────────────────────────────────────────────────────

def run_episode(env: SupportTriageEnv, task_id: str, episode: int) -> float:
    """Run one episode and return the mean score across all steps."""
    obs: Observation = env.reset(task_id=task_id)
    log_start(task_id, episode)

    step_idx = 0
    total_reward = 0.0
    done = False

    while not done:
        # Agent: call LLM
        raw_action = call_llm(obs)
        action = dict_to_action(raw_action)

        # Step environment
        result = env.step(action)
        reward = result.reward.score
        done = result.done
        total_reward += reward

        next_obs_dict = (
            result.observation.model_dump() if result.observation else None
        )

        log_step(
            step=step_idx,
            action=raw_action,
            observation=next_obs_dict,
            reward=reward,
            done=done,
        )

        if result.observation is not None:
            obs = result.observation
        step_idx += 1

    mean_score = total_reward / step_idx if step_idx > 0 else 0.001
    mean_score = max(0.01, min(0.99, mean_score))
    log_end(task_id, episode, mean_score, step_idx)
    return mean_score


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    env = SupportTriageEnv()
    results: Dict[str, float] = {}

    print("=" * 60, flush=True)
    print(f"Customer Support Ticket Triage — Baseline Inference", flush=True)
    print(f"Model : {MODEL_NAME}", flush=True)
    print(f"API   : {API_BASE_URL}", flush=True)
    print("=" * 60, flush=True)

    for i, task_id in enumerate(TASKS, start=1):
        score = run_episode(env, task_id=task_id, episode=i)
        results[task_id] = round(max(0.01, min(0.99, score)), 4)
        print(f"  → {task_id}: {score:.4f}", flush=True)

    print("=" * 60, flush=True)
    overall = max(0.01, min(0.99, sum(results.values()) / len(results)))
    print(f"OVERALL MEAN SCORE: {overall:.4f}", flush=True)
    print("TASK SCORES:", json.dumps(results, indent=2), flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
