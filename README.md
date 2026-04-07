---
title: Customer Support Ticket Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - environment
  - reinforcement-learning
  - nlp
  - customer-support
---


# Customer Support Ticket Triage (CSTT) — OpenEnv

A real-world OpenEnv environment where AI agents triage customer support tickets —
classifying priority, routing to departments, detecting sentiment, and drafting
professional responses.

## Why this domain?

Every software company processes thousands of support tickets daily. Misrouted or
misprioritized tickets cost real money: SLA breaches, churn, engineer time lost to
billing queries. This environment directly models that workflow, making it immediately
useful for evaluating agents in enterprise support automation.

---

## Environment Overview

| Property | Value |
|---|---|
| **Domain** | Customer Support Operations |
| **Tasks** | 3 (easy → medium → hard) |
| **Steps per episode** | 5 tickets |
| **Reward range** | 0.0 – 1.0 per step |
| **Reward type** | Dense (partial credit at every step) |
| **Deterministic** | Yes — fixed ticket order for reproducibility |
| **API** | REST (FastAPI) + Python class |

---

## Observation Space

Each step the agent receives one support ticket plus task metadata:

```json
{
  "ticket": {
    "ticket_id": "TK-1001",
    "subject": "URGENT: Production API completely down",
    "body": "Our entire platform has been down for 25 minutes...",
    "sender": "cto@enterprise-client.com",
    "sender_tier": "enterprise",
    "product": "API Platform",
    "timestamp": "2024-01-15T09:32:00Z"
  },
  "task_id": "task1_triage",
  "step": 0,
  "max_steps": 5,
  "episode": 1,
  "task_description": "Quick Triage: classify priority + department",
  "action_schema": {
    "priority": "string — P0 | P1 | P2 | P3 (required)",
    "department": "string — engineering | billing | support | security | product (required)"
  }
}
```

### Fields

| Field | Type | Description |
|---|---|---|
| `ticket_id` | string | Unique ticket identifier |
| `subject` | string | Email subject line |
| `body` | string | Full ticket message body |
| `sender` | string | Customer email address |
| `sender_tier` | enum | `free` \| `basic` \| `enterprise` |
| `product` | string | Affected product area |
| `timestamp` | ISO 8601 | When ticket was received |
| `task_id` | enum | Active task identifier |
| `step` | int | Current step (0-indexed) |
| `max_steps` | int | Total steps in episode (5) |
| `action_schema` | object | Task-specific required action fields |

---

## Action Space

Not all fields are required for every task. The `action_schema` in each observation
specifies exactly which fields are needed.

```json
{
  "priority": "P1",
  "department": "engineering",
  "sentiment": "negative",
  "escalate": true,
  "response_draft": "Dear ..., Thank you for contacting us..."
}
```

| Field | Type | Values | Used in |
|---|---|---|---|
| `priority` | enum | `P0` (critical/down) \| `P1` (high/major) \| `P2` (medium/degraded) \| `P3` (low/question) | All tasks |
| `department` | enum | `engineering` \| `billing` \| `support` \| `security` \| `product` \| `sales` | All tasks |
| `sentiment` | enum | `positive` \| `neutral` \| `negative` \| `angry` | Task 2, 3 |
| `escalate` | bool | `true` / `false` | Task 2 |
| `response_draft` | string | 80–300 word email response | Task 3 |

---

## Tasks

### Task 1 — Quick Triage *(easy)*

**Objective:** Given a support ticket, correctly classify its urgency and route it to
the right team.

**Required action fields:** `priority`, `department`

**Reward weights:**

| Component | Weight | Partial credit |
|---|---|---|
| Priority correct | 0.50 | 0.25 for adjacent level (e.g., P1 when P0 correct) |
| Department correct | 0.50 | No partial credit |

**Difficulty signals:** Tickets clearly indicate urgency (e.g. "production down",
"data breach", "feature request"). Suitable for smoke-testing basic comprehension.

**Expected baseline score:** ~0.82 (GPT-4o-mini)

---

### Task 2 — Smart Routing *(medium)*

**Objective:** Extend Task 1 with customer sentiment detection and escalation decision.
Requires reasoning about business context, customer tier, and urgency signals beyond
surface keywords.

**Required action fields:** `priority`, `department`, `sentiment`, `escalate`

**Reward weights:**

| Component | Weight |
|---|---|
| Priority | 0.30 |
| Department | 0.30 |
| Sentiment | 0.20 |
| Escalation | 0.20 |

**Difficulty signals:** Tickets mix emotional language with technical issues. A ticket
from an enterprise customer threatening to cancel requires escalation; an enthusiastic
upgrade inquiry does not. The sentiment-escalation correlation is non-trivial.

**Expected baseline score:** ~0.58 (GPT-4o-mini)

---

### Task 3 — Full Resolution *(hard)*

**Objective:** Complete triage plus drafting a full, professional email response that
addresses the customer's specific concerns.

**Required action fields:** `priority`, `department`, `response_draft`

**Response graded on 6 dimensions:**

| Sub-component | Weight | Criterion |
|---|---|---|
| Greeting | 0.06 | Opens with appropriate greeting |
| Acknowledgment | 0.10 | Explicitly acknowledges the specific issue |
| Solution | 0.30 | Covers ≥50% of expected technical/procedural keywords |
| Next steps | 0.14 | Explains what will happen next / what customer should do |
| Closing | 0.08 | Professional sign-off |
| Length | 0.12 | 80–300 words (proportional partial credit for ±50%) |

**Difficulty signals:** Response must be both technically accurate (mentions rate limits,
SQL injection, SLA section numbers, refund timelines) AND professionally formatted.
Generic responses score poorly on the solution dimension.

**Expected baseline score:** ~0.41 (GPT-4o-mini)

---

## Reward Function

### Properties

- **Dense**: Every step yields a score — no sparse end-of-episode signals.
- **Partial credit**: Adjacent priority levels earn 50% priority weight.
  Keyword coverage is proportional (not binary).
- **Penalty-free**: The environment does not subtract points; minimum score per step is 0.0.
- **Episode score**: Mean of per-step scores (not sum), normalizing for episode length.

### Reward signal example (Task 3, Step 1)

```
score=0.71
breakdown={
  "priority": 0.10,        # P1 correct
  "department": 0.10,      # engineering correct
  "resp_greeting": 0.06,   # "Dear ..." present
  "resp_acknowledgment": 0.10,
  "resp_solution": 0.24,   # 8/10 keywords matched → 0.30 × 0.8
  "resp_next_steps": 0.14,
  "resp_closing": 0.08,
  "resp_length": 0.09      # 320 words → partial
}
feedback: "priority=✓ | dept=✓ | response=0.71/0.80 [greet=✓ ack=✓ sol=0.24/0.30 steps=✓ close=✓ len=320w=~]"
```

---

## API Reference

The environment exposes a REST API (FastAPI) on port 7860.

### `POST /reset`

Start a new episode.

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1_triage"}'
```

### `POST /step`

Submit an action.

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"priority": "P0", "department": "engineering"}'
```

### `GET /state`

Get current environment state.

```bash
curl http://localhost:7860/state
```

### `GET /tasks`

List all tasks with descriptions and schemas.

### Interactive docs: `GET /docs`

---

## Setup & Usage

### Local Python

```bash
pip install -r requirements.txt

# Use the environment directly
python - <<'EOF'
from environment import SupportTriageEnvcd /Users/isha.isha/support-triage-env 
from environment.models import Action

env = SupportTriageEnv()
obs = env.reset("task1_triage")
print(obs.ticket.subject)

result = env.step(Action(priority="P0", department="engineering"))
print(result.reward.score, result.reward.feedback)
EOF
```

### Docker

```bash
docker build -t cstt-env .
docker run -p 7860:7860 cstt-env

# Test
curl http://localhost:7860/
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" \
  -d '{"task_id": "task2_routing"}'
```

### Baseline Inference Script

```bash
export HF_TOKEN="your-api-key"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"

python inference.py
```

Expected output format:
```
[START] task=task1_triage episode=1
[STEP] step=0 action={"priority":"P0","department":"engineering"} observation={...} reward=1.0000 done=false
...
[END] task=task1_triage episode=1 total_reward=0.8200 steps=5
```

---

## Baseline Scores

Measured with `gpt-4o-mini` at `temperature=0`:

| Task | Difficulty | Score |
|---|---|---|
| `task1_triage` | Easy | 0.82 |
| `task2_routing` | Medium | 0.58 |
| `task3_resolution` | Hard | 0.41 |
| **Overall mean** | — | **0.60** |

---

## Pre-submission Validation

```bash
# Validate openenv.yaml structure
python -c "
import yaml, sys
with open('openenv.yaml') as f:
    spec = yaml.safe_load(f)
required = ['name','version','tasks','observation_space','action_space','reward']
missing = [k for k in required if k not in spec]
if missing:
    print('MISSING:', missing); sys.exit(1)
print('openenv.yaml OK — tasks:', [t['id'] for t in spec['tasks']])
"

# Validate environment functional interface
python -c "
from environment import SupportTriageEnv
from environment.models import Action

env = SupportTriageEnv()
for task in env.task_ids:
    obs = env.reset(task)
    assert obs.task_id == task
    scores = []
    done = False
    while not done:
        a = Action(priority='P1', department='engineering',
                   sentiment='neutral', escalate=False,
                   response_draft='Dear customer, thank you for reaching out. '
                                  'We apologize for the inconvenience. Our team will '
                                  'investigate and follow up within 1 business day. '
                                  'Please let us know if you need anything else. '
                                  'Best regards, Support Team')
        r = env.step(a)
        assert 0.0 <= r.reward.score <= 1.0, f'Score out of range: {r.reward.score}'
        scores.append(r.reward.score)
        done = r.done
    print(f'{task}: {len(scores)} steps, scores={[round(s,2) for s in scores]}')
print('All tasks PASS')
"

# Validate Docker build
docker build -t cstt-env-test . && echo "Docker build OK"
```

---

## Project Structure

```
support-triage-env/
├── README.md               # This file (also HF Space README with YAML front matter)
├── Dockerfile              # Container definition (port 7860)
├── openenv.yaml            # OpenEnv spec
├── requirements.txt
├── app.py                  # FastAPI server
├── inference.py            # Baseline inference script
└── environment/
    ├── __init__.py
    ├── models.py           # Pydantic: Observation, Action, Reward, StepResult
    ├── env.py              # SupportTriageEnv class (step/reset/state)
    ├── graders.py          # Deterministic graders for all 3 tasks
    └── data.py             # 15 annotated tickets (5 per task)
```

---

## License

MIT
