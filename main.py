"""
ApplyFit — robust FastAPI backend.
- Correct CORS handling (works with browsers + exception responses)
- Global exception handler that PRESERVES CORS headers
- Retries + timeouts on Claude calls
- Graceful startup (warns for missing optional env vars instead of crashing)
- Per-request logging so Render logs are useful
- Matches the frontend's route names and JSON shapes exactly

Render:
    Build:  pip install -r requirements.txt
    Start:  uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import os
import json
import time
import uuid
import logging
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
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("applyfit")

# ============================================================
# CONFIG
# ============================================================
load_dotenv()

ANTHROPIC_API_KEY     = os.getenv("ANTHROPIC_API_KEY")
STRIPE_SECRET_KEY     = os.getenv("STRIPE_SECRET_KEY")
STRIPE_PRICE_ID       = os.getenv("STRIPE_PRICE_ID")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
MONGODB_URI           = os.getenv("MONGODB_URI")

# Required for anything to work
if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY is required")

# Warn (don't crash) about missing optional integrations
if not STRIPE_SECRET_KEY:
    log.warning("STRIPE_SECRET_KEY missing — /create-checkout-session will fail")
if not STRIPE_PRICE_ID:
    log.warning("STRIPE_PRICE_ID missing — /create-checkout-session will fail")
if not STRIPE_WEBHOOK_SECRET:
    log.warning("STRIPE_WEBHOOK_SECRET missing — /stripe-webhook signature check will fail")
if not MONGODB_URI:
    log.warning("MONGODB_URI missing — subscription checks will be bypassed")

# --- Clients ---
anthropic_client = anthropic.Anthropic(
    api_key=ANTHROPIC_API_KEY,
    timeout=60.0,
)
if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

if MONGODB_URI:
    try:
        db = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000).applyfit
        # ping to fail fast if URI is wrong
        db.command("ping")
        log.info("MongoDB connected")
    except Exception as e:
        log.warning(f"MongoDB connection failed: {e} — continuing without DB")
        db = None
else:
    db = None

# ============================================================
# MODEL
# ============================================================
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"

# ============================================================
# APP + CORS
# ============================================================
app = FastAPI(title="ApplyFit", version="0.1.0")

# CORS: allow everything during dev; lock down for production.
ALLOWED_ORIGINS = [
    "https://applyfit-frontend.vercel.app",
    "https://applyfit-frontend-git-main-wahmed3900s-projects.vercel.app",
    "https://applyfit-frontend-wahmed3900s-projects.vercel.app",
    "http://localhost:3000",
    "http://localhost:5500",
    "http://localhost:8000",
]

# Easiest reliable option: allow all origins (no credentials). 
# If you use cookies later, swap in ALLOWED_ORIGINS and set allow_credentials=True.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ============================================================
# REQUEST LOGGING MIDDLEWARE
# ============================================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    rid = uuid.uuid4().hex[:8]
    start = time.time()
    try:
        response = await call_next(request)
        elapsed = (time.time() - start) * 1000
        log.info(f"[{rid}] {request.method} {request.url.path} -> {response.status_code} ({elapsed:.0f}ms)")
        response.headers["X-Request-Id"] = rid
        return response
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        log.error(f"[{rid}] {request.method} {request.url.path} -> EXCEPTION ({elapsed:.0f}ms): {e}")
        raise

# ============================================================
# EXCEPTION HANDLER — preserves CORS headers
# ============================================================
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    log.error(f"UNHANDLED EXCEPTION on {request.url.path}:\n{tb}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "type": type(exc).__name__,
        },
        # Manually add CORS headers so the browser can actually read this
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        },
    )

# ============================================================
# REQUEST MODELS
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
# CLAUDE HELPER (with retries + robust JSON parsing)
# ============================================================
def claude_json(system: str, user: str, max_tokens: int = 2000, retries: int = 2) -> dict:
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = anthropic_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            raw = resp.content[0].text.strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                parts = raw.split("```")
                if len(parts) >= 2:
                    raw = parts[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
            raw = raw.strip()

            # Try strict parse
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                # Fallback: find first { ... last }
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end != -1 and end > start:
                    return json.loads(raw[start:end + 1])
                raise

        except Exception as e:
            last_err = e
            log.warning(f"Claude call failed (attempt {attempt + 1}/{retries + 1}): {type(e).__name__}: {e}")
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))

    # All retries exhausted — re-raise so the exception handler returns JSON
    raise last_err if last_err else RuntimeError("Claude call failed")


def has_active_subscription(email: str) -> bool:
    if db is None:
        return True  # no DB → allow (dev mode)
    try:
        return db.subscriptions.find_one(
            {"email": email.lower(), "status": "active"}
        ) is not None
    except Exception as e:
        log.warning(f"Subscription check failed: {e} — allowing")
        return True

# ============================================================
# ROUTES
# ============================================================
@app.get("/health")
def health():
    return {"ok": True, "model": CLAUDE_MODEL, "db": db is not None}


@app.get("/")
def root():
    return {"service": "ApplyFit API", "status": "running"}


@app.get("/debug/models")
def debug_models():
    """List every Claude model your key can access."""
    try:
        models = anthropic_client.models.list()
        return {"models": [m.id for m in models.data]}
    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}


@app.post("/extract-job-url")
def extract_job_url(req: ExtractURLRequest):
    """Frontend 'Fetch' button → { job_description }."""
    if not req.job_url.strip():
        raise HTTPException(400, "job_url is required")

    try:
        r = httpx.get(
            req.job_url,
            timeout=20.0,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(400, f"Failed to fetch URL: {e}")

    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    text = " ".join(soup.get_text(" ").split())

    if not text:
        raise HTTPException(400, "No readable text found at that URL")

    try:
        cleaned = claude_json(
            system=(
                "Extract just the job posting body from this scraped text. "
                'Return JSON only: {"job_description": string}. '
                "Strip navigation, cookie banners, and boilerplate."
            ),
            user=text[:15000],
        )
        return {"job_description": cleaned.get("job_description", text[:12000])}
    except Exception as e:
        log.warning(f"Claude cleanup failed, returning raw text: {e}")
        return {"job_description": text[:12000]}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """Frontend 'Analyze Match' → match_score, summary, matched_keywords, missing_keywords."""
    if not req.job_description.strip() or not req.resume_text.strip():
        raise HTTPException(400, "job_description and resume_text are required")

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

    # Normalize keys, tolerate Claude returning unexpected types
    try:
        score = int(data.get("match_score", 0))
    except (TypeError, ValueError):
        score = 0

    return {
        "match_score":      max(0, min(100, score)),
        "summary":          str(data.get("summary", "")),
        "matched_keywords": list(data.get("matched_keywords", []) or []),
        "missing_keywords": list(data.get("missing_keywords", []) or []),
    }


@app.post("/generate")
def generate(req: GenerateRequest):
    """Frontend 'Generate' (Pro) → tailored_bullets + cover_letter."""
    if not req.job_description.strip() or not req.resume_text.strip():
        raise HTTPException(400, "job_description and resume_text are required")
    if not req.email.strip():
        raise HTTPException(400, "email is required")

    if not has_active_subscription(req.email):
        raise HTTPException(402, "Active subscription required")

    company = (req.company_name or "the company").strip()
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

    # Persist if DB available (non-fatal if it fails)
    if db is not None:
        try:
            db.generations.insert_one({
                "email": req.email.lower(),
                "company": company,
                "tone": req.tone,
                "result": data,
                "created_at": time.time(),
            })
        except Exception as e:
            log.warning(f"Failed to save generation: {e}")

    return {
        "tailored_bullets": list(data.get("tailored_bullets", []) or []),
        "cover_letter":     str(data.get("cover_letter", "")),
    }


@app.post("/create-checkout-session")
def create_checkout_session(req: CheckoutRequest):
    """Frontend 'Subscribe' → { checkout_url }."""
    if not STRIPE_SECRET_KEY or not STRIPE_PRICE_ID:
        raise HTTPException(500, "Stripe is not configured on the server")
    if not req.email.strip():
        raise HTTPException(400, "email is required")

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            customer_email=req.email,
            success_url=req.success_url,
            cancel_url=req.cancel_url,
            client_reference_id=req.email.lower(),
        )
        return {"checkout_url": session.url, "id": session.id}
    except stripe.error.StripeError as e:
        raise HTTPException(400, f"Stripe error: {e}")


@app.post("/stripe-webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")

    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(500, "Stripe webhook secret not configured")

    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except ValueError:
        raise HTTPException(400, "Invalid payload")
    except Exception as e:
        if type(e).__name__ == "SignatureVerificationError":
            raise HTTPException(400, f"Invalid signature: {e}")
        raise

    try:
        if event["type"] == "checkout.session.completed":
            s = event["data"]["object"]
            email = (s.get("customer_email") or s.get("client_reference_id") or "").lower()
            if db is not None and email:
                db.subscriptions.update_one(
                    {"email": email},
                    {"$set": {"status": "active", "customer_id": s.get("customer")}},
                    upsert=True,
                )
                log.info(f"Subscription activated for {email}")
        elif event["type"] in ("customer.subscription.deleted", "customer.subscription.paused"):
            cust = event["data"]["object"].get("customer")
            if db is not None and cust:
                db.subscriptions.update_one(
                    {"customer_id": cust}, {"$set": {"status": "canceled"}}
                )
                log.info(f"Subscription canceled for customer {cust}")
    except Exception as e:
        log.error(f"Webhook processing error: {e}")
        # Still return 200 so Stripe doesn't retry endlessly

    return {"received": True}
