"""
ApplyFit backend — corrected, deploy-ready version for Render.

Start command:
    uvicorn main:app --host 0.0.0.0 --port $PORT
"""

# =============================================================
# IMPORTS  (these were missing before — that caused the crash)
# =============================================================
import os
import json
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import anthropic
import stripe
import httpx
from bs4 import BeautifulSoup
from pymongo import MongoClient

# =============================================================
# CONFIG
# =============================================================
load_dotenv()  # reads .env locally; on Render, env vars come from the dashboard

ANTHROPIC_API_KEY     = os.getenv("ANTHROPIC_API_KEY")
STRIPE_SECRET_KEY     = os.getenv("STRIPE_SECRET_KEY")
STRIPE_PRICE_ID       = os.getenv("STRIPE_PRICE_ID")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
MONGODB_URI           = os.getenv("MONGODB_URI")

if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY is required")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
stripe.api_key = STRIPE_SECRET_KEY
db = MongoClient(MONGODB_URI).applyfit if MONGODB_URI else None

# =============================================================
# APP + CORS
# =============================================================
app = FastAPI(title="ApplyFit", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================
# MODELS
# =============================================================
class JobInput(BaseModel):
    url: Optional[str] = None
    text: Optional[str] = None

class TailorRequest(BaseModel):
    job: JobInput
    resume: str
    user_id: Optional[str] = None

class CheckoutRequest(BaseModel):
    user_id: str

# =============================================================
# HELPERS
# =============================================================
def fetch_job_text(job: JobInput) -> str:
    if job.text:
        return job.text
    if not job.url:
        raise HTTPException(400, "Provide either job.url or job.text")
    try:
        r = httpx.get(
            job.url, timeout=15, follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (ApplyFit)"},
        )
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(400, f"Failed to fetch job URL: {e}")
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return " ".join(soup.get_text(" ").split())[:12000]

def claude_json(system: str, user: str, max_tokens: int = 2000) -> dict:
    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())

# =============================================================
# ROUTES
# =============================================================
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/extract")
def extract_requirements(job: JobInput):
    text = fetch_job_text(job)
    return claude_json(
        system=(
            "You extract structured job requirements. "
            "Return JSON only with keys: title, company, "
            "must_have (list), nice_to_have (list), keywords (list), "
            "seniority (string), summary (string)."
        ),
        user=f"Job posting:\n\n{text}",
    )

@app.post("/score")
def score_resume(req: TailorRequest):
    job_text = fetch_job_text(req.job)
    return claude_json(
        system=(
            "You are a resume-match scorer. Return JSON only with keys: "
            "score (0-100 int), matched (list), gaps (list), "
            "recommendations (list of strings)."
        ),
        user=f"RESUME:\n{req.resume}\n\nJOB:\n{job_text}",
    )

@app.post("/tailor")
def tailor(req: TailorRequest):
    job_text = fetch_job_text(req.job)
    data = claude_json(
        system=(
            "You are an expert resume writer and cover-letter author. "
            "Return JSON only with keys: "
            "tailored_resume (markdown string), "
            "cover_letter (markdown string), "
            "change_log (list of strings). "
            "Never fabricate experience — rephrase and emphasize only."
        ),
        user=f"RESUME:\n{req.resume}\n\nJOB:\n{job_text}",
        max_tokens=4000,
    )
    if db is not None and req.user_id:
        db.tailorings.insert_one({
            "user_id": req.user_id,
            "job_snippet": job_text[:500],
            "result": data,
        })
    return data

# =============================================================
# STRIPE
# =============================================================
@app.post("/billing/checkout")
def create_checkout(req: CheckoutRequest):
    if not STRIPE_SECRET_KEY or not STRIPE_PRICE_ID:
        raise HTTPException(500, "Stripe not configured")
    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
        success_url="https://your-frontend.example.com/success?session_id={CHECKOUT_SESSION_ID}",
        cancel_url="https://your-frontend.example.com/cancel",
        client_reference_id=req.user_id,
    )
    return {"url": session.url, "id": session.id}

@app.post("/billing/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")

    # --- Signature verification (handles both old and new Stripe SDK paths) ---
    try:
        event = stripe.Webhook.construct_event(
            payload, sig, STRIPE_WEBHOOK_SECRET
        )
    except ValueError as e:
        raise HTTPException(400, f"Invalid payload: {e}")
    except Exception as e:
        # Newer stripe-python exposes SignatureVerificationError at
        # stripe.SignatureVerificationError; older at stripe.error.SignatureVerificationError.
        name = type(e).__name__
        if name == "SignatureVerificationError":
            raise HTTPException(400, f"Invalid signature: {e}")
        raise

    if event["type"] == "checkout.session.completed":
        s = event["data"]["object"]
        if db is not None:
            db.subscriptions.update_one(
                {"user_id": s.get("client_reference_id")},
                {"$set": {"status": "active", "customer": s.get("customer")}},
                upsert=True,
            )
    return {"received": True}
