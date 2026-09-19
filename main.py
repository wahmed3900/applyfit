"""
ApplyFit — FastAPI backend (final, diagnostic build).
Routes and JSON shapes match frontend_index.html exactly.

Render deploy settings:
    Build:  pip install -r requirements.txt
    Start:  uvicorn main:app --host 0.0.0.0 --port $PORT
"""

# ============================================================
# IMPORTS
# ============================================================
import os
import json
import traceback
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

import anthropic
import stripe
import httpx
from bs4 import BeautifulSoup
from pymongo import MongoClient

# ============================================================
# CONFIG
# ============================================================
load_dotenv()

ANTHROPIC_API_KEY     = os.getenv("ANTHROPIC_API_KEY")
STRIPE_SECRET_KEY     = os.getenv("STRIPE_SECRET_KEY")
STRIPE_PRICE_ID       = os.getenv("STRIPE_PRICE_ID")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
MONGODB_URI           = os.getenv("MONGODB_URI")

if not ANTHROPIC_API_KEY:
    log.warning("ANTHROPIC_API_KEY not set — will fail at call time")
ANTHROPIC_API_KEY = ""
if ANTHROPIC_API_KEY:
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=60.0)
else:
    anthropic_client = None
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
stripe.api_key = STRIPE_SECRET_KEY
db = MongoClient(MONGODB_URI).applyfit if MONGODB_URI else None

# ============================================================
# CLAUDE MODEL
# ============================================================
# Universally-available model ID. If you have access to a newer one,
# you can change this — use GET /debug/models to list available IDs.
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"

# ============================================================
# APP + CORS
# ============================================================
app = FastAPI(title="ApplyFit", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your Vercel domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# GLOBAL EXCEPTION HANDLER
# ============================================================
# Turns "Internal Server Error" into a readable JSON body so you can
# see the real cause without digging through Render logs.
@app.exception_handler(Exception)
async def debug_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    print("UNHANDLED EXCEPTION:\n", tb)   # appears in Render logs
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__},
    )

# ============================================================
# REQUEST MODELS  (shapes match the frontend exactly)
# ============================================================
class ExtractURLRequest(BaseModel):
    job_url: str

class AnalyzeRequest(BaseModel):
    job_description: str
    resume_text: str

class GenerateRequest(BaseModel):
    job_description: str
    resume_text: str
    company_name: Optional[str] = None
    tone: str = "professional"
    email: str

class CheckoutRequest(BaseModel):
    email: str
    success_url: str
    cancel_url: str

# ============================================================
# HELPERS
# ============================================================
def claude_json(system: str, user: str, max_tokens: int = 2000) -> dict:
    """Call Claude, strip markdown fences, parse JSON."""
    resp = client.messages.create(
        model=CLAUDE_MODEL,
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


def has_active_subscription(email: str) -> bool:
    if db is None:
        return True  # dev mode: no DB → allow
    return db.subscriptions.find_one(
        {"email": email.lower(), "status": "active"}
    ) is not None

# ============================================================
# ROUTES
# ============================================================
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def root():
    return {"service": "ApplyFit API", "status": "running"}


@app.get("/debug/models")
def debug_models():
    """List every Claude model your Anthropic key can access."""
    try:
        models = client.models.list()
        return {"models": [m.id for m in models.data]}
    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}


@app.post("/extract-job-url")
def extract_job_url(req: ExtractURLRequest):
    """Frontend 'Fetch' button. Returns { job_description }."""
    try:
        r = httpx.get(
            req.job_url,
            timeout=15,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (ApplyFit)"},
        )
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(400, f"Failed to fetch URL: {e}")

    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = " ".join(soup.get_text(" ").split())

    try:
        cleaned = claude_json(
            system=(
                "Extract just the job posting body from this scraped text. "
                'Return JSON only: {"job_description": string}. '
                "Strip navigation, cookie banners, and boilerplate."
            ),
            user=text[:15000],
        )
        return {"job_description": cleaned["job_description"]}
    except Exception:
        return {"job_description": text[:12000]}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """Frontend 'Analyze Match'. Returns match_score, summary, matched_keywords, missing_keywords."""
    data = claude_json(
        system=(
            "You compare a resume against a job description. "
            "Return JSON only with keys: "
            "match_score (0-100 integer), summary (2-3 sentence string), "
            "matched_keywords (list of short strings), "
            "missing_keywords (list of short strings)."
        ),
        user=f"RESUME:\n{req.resume_text}\n\nJOB:\n{req.job_description}",
    )
    return {
        "match_score":      int(data.get("match_score", 0)),
        "summary":          data.get("summary", ""),
        "matched_keywords": data.get("matched_keywords", []),
        "missing_keywords": data.get("missing_keywords", []),
    }


@app.post("/generate")
def generate(req: GenerateRequest):
    """Frontend 'Generate' button (Pro). Returns tailored_bullets + cover_letter."""
    if not has_active_subscription(req.email):
        raise HTTPException(402, "Active subscription required")

    company = req.company_name or "the company"
    data = claude_json(
        system=(
            "You are an expert resume writer and cover-letter author. "
            "Return JSON only with keys: "
            "tailored_bullets (list of 6-10 strong resume bullet strings, "
            "each starting with an action verb and including metrics where possible), "
            "cover_letter (a single plain-text string with \\n\\n paragraph breaks). "
            f"Tone: {req.tone}. Address the letter to {company}. "
            "Never fabricate experience — rephrase and emphasize only."
        ),
        user=f"RESUME:\n{req.resume_text}\n\nJOB:\n{req.job_description}",
        max_tokens=4000,
    )

    if db is not None:
        db.generations.insert_one({"email": req.email.lower(), "result": data})

    return {
        "tailored_bullets": data.get("tailored_bullets", []),
        "cover_letter":     data.get("cover_letter", ""),
    }


@app.post("/create-checkout-session")
def create_checkout_session(req: CheckoutRequest):
    """Frontend 'Subscribe' button. Returns { checkout_url }."""
    if not STRIPE_SECRET_KEY or not STRIPE_PRICE_ID:
        raise HTTPException(500, "Stripe not configured")

    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
        customer_email=req.email,
        success_url=req.success_url,
        cancel_url=req.cancel_url,
        client_reference_id=req.email.lower(),
    )
    return {"checkout_url": session.url, "id": session.id}


@app.post("/stripe-webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except ValueError:
        raise HTTPException(400, "Invalid payload")
    except Exception as e:
        if type(e).__name__ == "SignatureVerificationError":
            raise HTTPException(400, f"Invalid signature: {e}")
        raise

    if event["type"] == "checkout.session.completed":
        s = event["data"]["object"]
        email = (s.get("customer_email") or s.get("client_reference_id") or "").lower()
        if db is not None and email:
            db.subscriptions.update_one(
                {"email": email},
                {"$set": {"status": "active", "customer_id": s.get("customer")}},
                upsert=True,
            )
    elif event["type"] in ("customer.subscription.deleted", "customer.subscription.paused"):
        cust = event["data"]["object"].get("customer")
        if db is not None and cust:
            db.subscriptions.update_one(
                {"customer_id": cust}, {"$set": {"status": "canceled"}}
            )
    return {"received": True}
