"""
Deterministic ticket datasets for all three tasks.
Each ticket carries an `expected` dict used by the grader.
"""

# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — Quick Triage  (easy)
# Agent must correctly classify: priority + department
# ─────────────────────────────────────────────────────────────────────────────
TASK1_TICKETS = [
    {
        "ticket_id": "TK-1001",
        "subject": "URGENT: Production API completely down — all requests 503",
        "body": (
            "Our entire platform has been down for 25 minutes. Every API call returns 503. "
            "We have 50,000 active users affected and are losing ~$8,000/minute in revenue. "
            "Our CTO is on a call with our investors right now. Please escalate immediately."
        ),
        "sender": "cto@enterprise-client.com",
        "sender_tier": "enterprise",
        "product": "API Platform",
        "timestamp": "2024-01-15T09:32:00Z",
        "expected": {"priority": "P0", "department": "engineering"},
    },
    {
        "ticket_id": "TK-1002",
        "subject": "How do I reset my password?",
        "body": (
            "Hi there, I forgot my password and can't log in. "
            "I tried the 'Forgot password' link twice but never received an email. "
            "Could you help me regain access to my account? Thanks!"
        ),
        "sender": "jane.doe@gmail.com",
        "sender_tier": "free",
        "product": "Web App",
        "timestamp": "2024-01-15T10:15:00Z",
        "expected": {"priority": "P3", "department": "support"},
    },
    {
        "ticket_id": "TK-1003",
        "subject": "Invoice overcharge — billed $1,500 instead of $1,000",
        "body": (
            "I just received my monthly invoice and I've been charged $1,500 "
            "instead of the contracted $1,000. I have been on the Growth plan for "
            "6 months at the same price. Please investigate and issue a credit note "
            "or refund the $500 difference as soon as possible."
        ),
        "sender": "finance@medium-business.com",
        "sender_tier": "basic",
        "product": "Billing",
        "timestamp": "2024-01-15T11:20:00Z",
        "expected": {"priority": "P2", "department": "billing"},
    },
    {
        "ticket_id": "TK-1004",
        "subject": "Feature request: dark mode for the dashboard",
        "body": (
            "Hi team! Love the product — it's been super helpful for our workflow. "
            "I just wanted to suggest adding a dark mode option to the main dashboard. "
            "I work late nights and the bright white background strains my eyes. "
            "I've also seen this requested multiple times on the community forum. "
            "Would love to see this in a future release!"
        ),
        "sender": "dev@startup.io",
        "sender_tier": "basic",
        "product": "Dashboard",
        "timestamp": "2024-01-15T13:45:00Z",
        "expected": {"priority": "P3", "department": "product"},
    },
    {
        "ticket_id": "TK-1005",
        "subject": "Possible data breach — unauthorized admin login at 03:47 GMT",
        "body": (
            "Our security team detected an unauthorized admin-level login at 03:47 AM GMT today. "
            "The source IP (185.220.101.45) is a known Tor exit node. "
            "Audit logs show read access to the users table and export of ~12,000 records. "
            "We have not authorized any access at that time. "
            "We are treating this as a critical security incident. Please respond immediately."
        ),
        "sender": "ciso@financial-corp.com",
        "sender_tier": "enterprise",
        "product": "Authentication",
        "timestamp": "2024-01-15T07:02:00Z",
        "expected": {"priority": "P0", "department": "security"},
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — Smart Routing  (medium)
# Agent must classify: priority + department + sentiment + escalate (bool)
# ─────────────────────────────────────────────────────────────────────────────
TASK2_TICKETS = [
    {
        "ticket_id": "TK-2001",
        "subject": "Third billing error this month — I am done with this company",
        "body": (
            "I cannot believe this is happening AGAIN. For the third month in a row you have "
            "overcharged my credit card. Each time I have to spend hours on support to get it "
            "fixed. My CFO is furious. If this is not resolved TODAY with a full refund AND a "
            "written explanation, we are cancelling our enterprise contract worth $120k/yr and "
            "going public on Twitter about your billing practices. This is completely unacceptable."
        ),
        "sender": "vp-finance@angry-enterprise.com",
        "sender_tier": "enterprise",
        "product": "Billing",
        "timestamp": "2024-01-16T08:14:00Z",
        "expected": {
            "priority": "P1",
            "department": "billing",
            "sentiment": "angry",
            "escalate": True,
        },
    },
    {
        "ticket_id": "TK-2002",
        "subject": "Question about exporting data to CSV",
        "body": (
            "Hello! Quick question — is there a way to export my dashboard data to CSV? "
            "I've looked through the settings and the help docs but couldn't find the option. "
            "Thanks in advance for your help!"
        ),
        "sender": "alice@small-team.com",
        "sender_tier": "free",
        "product": "Dashboard",
        "timestamp": "2024-01-16T11:30:00Z",
        "expected": {
            "priority": "P3",
            "department": "support",
            "sentiment": "neutral",
            "escalate": False,
        },
    },
    {
        "ticket_id": "TK-2003",
        "subject": "Webhook integration broken after last night's deployment",
        "body": (
            "Since your deployment at ~02:00 UTC, our webhook endpoints are no longer receiving "
            "events. Our monitoring shows zero callbacks in the last 6 hours. "
            "We process 4,000 orders/hour through this webhook — our fulfilment pipeline is "
            "completely stalled. We have confirmed our endpoint is healthy. "
            "This is a severity-1 issue on our end. Please investigate urgently."
        ),
        "sender": "platform-eng@ecommerce-giant.com",
        "sender_tier": "enterprise",
        "product": "Webhooks",
        "timestamp": "2024-01-16T09:05:00Z",
        "expected": {
            "priority": "P1",
            "department": "engineering",
            "sentiment": "negative",
            "escalate": True,
        },
    },
    {
        "ticket_id": "TK-2004",
        "subject": "We just hit 1M API calls — excited to upgrade our plan!",
        "body": (
            "Hi! Just wanted to let you know we've been growing fast and hit our 1 million "
            "API call milestone this month! We'd love to explore upgrading to the Enterprise "
            "plan. Could someone from your team reach out to walk us through the options? "
            "We're particularly interested in the SLA guarantees and dedicated support."
        ),
        "sender": "founder@growth-startup.io",
        "sender_tier": "basic",
        "product": "API Platform",
        "timestamp": "2024-01-16T14:22:00Z",
        "expected": {
            "priority": "P2",
            "department": "sales",
            "sentiment": "positive",
            "escalate": False,
        },
    },
    {
        "ticket_id": "TK-2005",
        "subject": "Suspicious login attempts — MFA codes being requested without my action",
        "body": (
            "Over the past two hours I have received 11 MFA authentication requests on my phone "
            "that I did not initiate. Someone appears to be attempting to access my account "
            "with my correct password. I have not shared my credentials with anyone. "
            "Please immediately lock my account and investigate. I'm worried my data may "
            "already be compromised."
        ),
        "sender": "worried-user@company.com",
        "sender_tier": "basic",
        "product": "Authentication",
        "timestamp": "2024-01-16T16:48:00Z",
        "expected": {
            "priority": "P1",
            "department": "security",
            "sentiment": "negative",
            "escalate": True,
        },
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 — Full Resolution  (hard)
# Agent must classify: priority + department + response_draft
# Response graded on: greeting, acknowledgment, technical accuracy,
# next steps, closing, appropriate length.
# ─────────────────────────────────────────────────────────────────────────────
TASK3_TICKETS = [
    {
        "ticket_id": "TK-3001",
        "subject": "API rate limit errors — 429 on all endpoints",
        "body": (
            "We keep hitting 429 Too Many Requests errors across all your API endpoints. "
            "Our current plan says 1,000 req/min but we're well below that according to our "
            "own metrics. The errors started 3 hours ago. We've tried backing off and "
            "retrying with exponential backoff but still seeing the issue. "
            "Error header: X-RateLimit-Remaining: 0, X-RateLimit-Reset: 1705320000"
        ),
        "sender": "backend@tech-company.com",
        "sender_tier": "basic",
        "product": "API Platform",
        "timestamp": "2024-01-17T10:00:00Z",
        "expected": {
            "priority": "P1",
            "department": "engineering",
            "solution_keywords": [
                "rate limit", "429", "quota", "investigate", "logs", "reset", "escalate"
            ],
        },
    },
    {
        "ticket_id": "TK-3002",
        "subject": "Charged for cancelled subscription — need immediate refund",
        "body": (
            "I cancelled my subscription on December 28th, confirmed by your cancellation email "
            "(ref: CANCEL-98234), yet I was still charged $299 on January 1st. "
            "I have the cancellation confirmation and my bank statement showing the charge. "
            "I need this refunded immediately as it's causing an overdraft fee on my account."
        ),
        "sender": "customer@personal.com",
        "sender_tier": "free",
        "product": "Billing",
        "timestamp": "2024-01-17T11:30:00Z",
        "expected": {
            "priority": "P2",
            "department": "billing",
            "solution_keywords": [
                "refund", "apologize", "investigate", "cancellation", "3-5 business days",
                "confirmation", "billing team"
            ],
        },
    },
    {
        "ticket_id": "TK-3003",
        "subject": "Team member can't accept organization invitation — link expired?",
        "body": (
            "I invited three new engineers to our organization last week but one of them "
            "says the invitation link has expired when they try to click it. "
            "The other two joined fine. I've re-sent the invitation twice and they're "
            "getting the same error: 'This invitation link has expired or is invalid.' "
            "We need them onboarded today as they're starting a critical project."
        ),
        "sender": "team-lead@consulting-firm.com",
        "sender_tier": "basic",
        "product": "Team Management",
        "timestamp": "2024-01-17T09:15:00Z",
        "expected": {
            "priority": "P2",
            "department": "support",
            "solution_keywords": [
                "invitation", "expire", "resend", "link", "workaround", "account",
                "investigate", "email"
            ],
        },
    },
    {
        "ticket_id": "TK-3004",
        "subject": "SQL injection vulnerability found in your search endpoint",
        "body": (
            "During a routine security audit of your platform, our penetration tester discovered "
            "a SQL injection vulnerability in the /api/v2/search endpoint. "
            "Using the payload: ' OR '1'='1 we were able to retrieve records from other tenants. "
            "We have proof-of-concept code and screenshots. We're disclosing this responsibly "
            "and expect it to be patched within 48 hours as per your stated security policy. "
            "Please acknowledge receipt and provide a timeline."
        ),
        "sender": "security-researcher@audit-firm.com",
        "sender_tier": "enterprise",
        "product": "API Platform",
        "timestamp": "2024-01-17T08:00:00Z",
        "expected": {
            "priority": "P0",
            "department": "security",
            "solution_keywords": [
                "thank", "security", "patch", "acknowledge", "investigate", "timeline",
                "responsible", "vulnerability", "team", "48 hours"
            ],
        },
    },
    {
        "ticket_id": "TK-3005",
        "subject": "SLA breach — 6-hour outage not covered by your 99.9% uptime guarantee",
        "body": (
            "Your platform was completely unavailable from 14:00–20:00 UTC on January 10th "
            "(6 hours). Our enterprise contract guarantees 99.9% monthly uptime (~43 min "
            "allowed downtime). This single incident consumed our entire monthly allowance "
            "and exceeded it by 5+ hours. "
            "Per section 8.2 of our contract, we are entitled to a 15% service credit. "
            "Please confirm the credit will appear on our next invoice and provide a "
            "post-mortem report."
        ),
        "sender": "director-it@large-enterprise.com",
        "sender_tier": "enterprise",
        "product": "API Platform",
        "timestamp": "2024-01-17T13:00:00Z",
        "expected": {
            "priority": "P1",
            "department": "billing",
            "solution_keywords": [
                "apologize", "credit", "SLA", "post-mortem", "incident", "contract",
                "confirm", "15%", "invoice", "acknowledge"
            ],
        },
    },
]

TASK_TICKETS = {
    "task1_triage": TASK1_TICKETS,
    "task2_routing": TASK2_TICKETS,
    "task3_resolution": TASK3_TICKETS,
}

TASK_DESCRIPTIONS = {
    "task1_triage": (
        "Quick Triage (Easy): Analyze the support ticket and determine:\n"
        "  1. priority — ticket severity (P0=critical/system-down, P1=high/major-impact, "
        "P2=medium/degraded, P3=low/question-or-feature)\n"
        "  2. department — correct routing team (engineering, billing, support, security, product)"
    ),
    "task2_routing": (
        "Smart Routing (Medium): Analyze the ticket and determine:\n"
        "  1. priority — P0/P1/P2/P3 as above\n"
        "  2. department — engineering, billing, support, security, product, or sales\n"
        "  3. sentiment — customer emotional tone (positive, neutral, negative, angry)\n"
        "  4. escalate — true if ticket requires immediate manager escalation, else false"
    ),
    "task3_resolution": (
        "Full Resolution (Hard): Analyze the ticket and determine:\n"
        "  1. priority — P0/P1/P2/P3\n"
        "  2. department — engineering, billing, support, security, product\n"
        "  3. response_draft — a complete, professional email response to the customer that:\n"
        "     • Opens with a greeting\n"
        "     • Acknowledges their specific issue\n"
        "     • Provides accurate technical/procedural next steps\n"
        "     • States what happens next (follow-up timeline)\n"
        "     • Closes professionally\n"
        "     • Is 80–300 words"
    ),
}

ACTION_SCHEMAS = {
    "task1_triage": {
        "priority": "string — P0 | P1 | P2 | P3 (required)",
        "department": "string — engineering | billing | support | security | product (required)",
    },
    "task2_routing": {
        "priority": "string — P0 | P1 | P2 | P3 (required)",
        "department": "string — engineering | billing | support | security | product | sales (required)",
        "sentiment": "string — positive | neutral | negative | angry (required)",
        "escalate": "boolean — true or false (required)",
    },
    "task3_resolution": {
        "priority": "string — P0 | P1 | P2 | P3 (required)",
        "department": "string — engineering | billing | support | security | product (required)",
        "response_draft": "string — full professional email response 80-300 words (required)",
    },
}
