"""
ApplyFit — FastAPI backend with multi-provider LLM fallback.

LLM chain: Anthropic Claude -> Google Gemini -> Groq
If one provider fails (bad key, quota, timeout, outage), the next is tried.
You only need to set the API key(s) for the provider(s) you want in the
chain — missing keys are skipped, not treated as errors, as long as at
least one provider is configured.

Required:  at least ONE of ANTHROPIC_API_KEY / GEMINI_API_KEY / GROQ_API_KEY
Optional:  STRIPE_SECRET_KEY, STRIPE_PRICE_ID, STRIPE_WEBHOOK_SECRET, MONGODB_URI

NEVER hardcode any of these keys in this file. Set them in
Render -> your service -> Environment, and nowhere else.

Render deploy settings:
    Build:  pip install -r requirements.txt
    Start:  uvicorn main:app --host 0.0.0.0 --port $PORT
"""

# ============================================================
# IMPORTS
# ============================================================
import os
import json
import time
import uuid
import logging
import traceback
from typing import Optional, Tuple

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
# LOGGING (must come before any log.* call)
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("applyfit")

# ============================================================
# CONFIG — every secret comes from the environment
# ============================================================
load_dotenv()

ANTHROPIC_API_KEY     = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY        = os.getenv("GEMINI_API_KEY")     # aistudio.google.com/app/apikey
GROQ_API_KEY          = os.getenv("GROQ_API_KEY")       # console.groq.com/keys
STRIPE_SECRET_KEY     = os.getenv("STRIPE_SECRET_KEY")
STRIPE_PRICE_ID       = os.getenv("STRIPE_PRICE_ID")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
MONGODB_URI           = os.getenv("MONGODB_URI")

# Need at least one LLM provider or the app can't do anything useful.
if not (ANTHROPIC_API_KEY or GEMINI_API_KEY or GROQ_API_KEY):
    raise RuntimeError(
        "No LLM provider configured. Set at least one of: "
        "ANTHROPIC_API_KEY, GEMINI_API_KEY, GROQ_API_KEY in Render's Environment tab."
    )

if not ANTHROPIC_API_KEY:
    log.warning("ANTHROPIC_API_KEY missing — Claude will be skipped in the fallback chain")
if not GEMINI_API_KEY:
    log.warning("GEMINI_API_KEY missing — Gemini will be skipped in the fallback chain")
if not GROQ_API_KEY:
    log.warning("GROQ_API_KEY missing — Groq will be skipped in the fallback chain")
if not STRIPE_SECRET_KEY:
    log.warning("STRIPE_SECRET_KEY missing — /create-checkout-session will fail")
if not STRIPE_PRICE_ID:
    log.warning("STRIPE_PRICE_ID missing — /create-checkout-session will fail")
if not STRIPE_WEBHOOK_SECRET:
    log.warning("STRIPE_WEBHOOK_SECRET missing — webhook signature check will fail")
if not MONGODB_URI:
    log.warning("MONGODB_URI missing — subscription checks are bypassed (dev mode)")

# ============================================================
# CLIENTS
# ============================================================
anthropic_client = None
if ANTHROPIC_API_KEY:
    try:
        anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=60.0)
    except Exception as e:
        log.warning(f"Anthropic init failed: {e}")

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

db = None
if MONGODB_URI:
    try:
        db = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000).applyfit
        db.command("ping")
        log.info("MongoDB connected")
    except Exception as e:
        log.warning(f"MongoDB connection failed: {e} — continuing without DB")
        db = None

# ============================================================
# MODEL CONFIG
# ============================================================
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
GEMINI_MODEL = "gemini-flash-latest"
GROQ_MODEL   = "llama-3.3-70b-versatile"

# ============================================================
# APP + CORS
# ============================================================
app = FastAPI(title="ApplyFit", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your Vercel domain in production
    allow_methods=["*"],
    allow_headers=["*"],
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
# GLOBAL EXCEPTION HANDLER
# ============================================================
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    log.error(f"UNHANDLED EXCEPTION on {request.url.path}:\n{tb}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__},
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
# JSON PARSING HELPER
# ============================================================
def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 2:
            raw = parts[1]
            if raw.lstrip().startswith("json"):
                raw = raw.lstrip()[4:]
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start:end + 1])
        raise

# ============================================================
# PROVIDER 1 — ANTHROPIC CLAUDE
# ============================================================
def _claude_json(system: str, user: str, max_tokens: int) -> dict:
    if anthropic_client is None:
        raise RuntimeError("Anthropic not configured")
    resp = anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return _parse_json(resp.content[0].text)

# ============================================================
# PROVIDER 2 — GOOGLE GEMINI
# ============================================================
def _gemini_json(system: str, user: str, max_tokens: int) -> dict:
    if not GEMINI_API_KEY:
        raise RuntimeError("Gemini not configured")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"parts": [{"text": user}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
            "temperature": 0.4,
        },
    }
    r = httpx.post(url, json=payload, timeout=60.0)
    if r.status_code != 200:
        raise RuntimeError(f"Gemini {r.status_code}: {r.text[:200]}")
    data = r.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Gemini unexpected shape: {data}") from e
    return _parse_json(text)

# ============================================================
# PROVIDER 3 — GROQ
# ============================================================
def _groq_json(system: str, user: str, max_tokens: int) -> dict:
    if not GROQ_API_KEY:
        raise RuntimeError("Groq not configured")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.4,
        "response_format": {"type": "json_object"},
    }
    r = httpx.post(url, headers=headers, json=payload, timeout=60.0)
    if r.status_code != 200:
        raise RuntimeError(f"Groq {r.status_code}: {r.text[:200]}")
    data = r.json()
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Groq unexpected shape: {data}") from e
    return _parse_json(text)

# ============================================================
# FALLBACK DISPATCHER
# ============================================================
def llm_json(system: str, user: str, max_tokens: int = 2000) -> Tuple[dict, str]:
    """Try Claude -> Gemini -> Groq. Returns (parsed_json, provider_name_used)."""
    chain = []
    if anthropic_client is not None:
        chain.append(("claude", _claude_json))
    if GEMINI_API_KEY:
        chain.append(("gemini", _gemini_json))
    if GROQ_API_KEY:
        chain.append(("groq", _groq_json))

    if not chain:
        raise HTTPException(500, "No LLM providers configured on the server.")

    last_err = None
    for name, fn in chain:
        try:
            t0 = time.time()
            result = fn(system, user, max_tokens)
            elapsed = (time.time() - t0) * 1000
            log.info(f"LLM '{name}' succeeded in {elapsed:.0f}ms")
            return result, name
        except Exception as e:
            log.warning(f"LLM '{name}' failed: {type(e).__name__}: {e}")
            last_err = e
            continue

    raise HTTPException(502, f"All configured LLM providers failed. Last error: {last_err}")

# ============================================================
# SUBSCRIPTION CHECK
# ============================================================
def has_active_subscription(email: str) -> bool:
    if db is None:
        return True
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
    return {
        "ok": True,
        "providers": {
            "claude": anthropic_client is not None,
            "gemini": bool(GEMINI_API_KEY),
            "groq":   bool(GROQ_API_KEY),
        },
        "db": db is not None,
        "stripe": bool(STRIPE_SECRET_KEY and STRIPE_PRICE_ID),
    }


@app.get("/")
def root():
    return {"service": "ApplyFit API", "status": "running"}


@app.get("/debug/models")
def debug_models():
    out = {}

    if anthropic_client is not None:
        try:
            models = anthropic_client.models.list()
            out["claude"] = [m.id for m in models.data]
        except Exception as e:
            out["claude"] = f"ERROR: {type(e).__name__}: {e}"
    else:
        out["claude"] = "not configured"

    if GEMINI_API_KEY:
        try:
            r = httpx.get(
                f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}",
                timeout=15.0,
            )
            if r.status_code == 200:
                out["gemini"] = [m["name"].split("/")[-1] for m in r.json().get("models", [])]
            else:
                out["gemini"] = f"ERROR {r.status_code}: {r.text[:200]}"
        except Exception as e:
            out["gemini"] = f"ERROR: {type(e).__name__}: {e}"
    else:
        out["gemini"] = "not configured"

    if GROQ_API_KEY:
        try:
            r = httpx.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                timeout=15.0,
            )
            if r.status_code == 200:
                out["groq"] = [m["id"] for m in r.json().get("data", [])]
            else:
                out["groq"] = f"ERROR {r.status_code}: {r.text[:200]}"
        except Exception as e:
            out["groq"] = f"ERROR: {type(e).__name__}: {e}"
    else:
        out["groq"] = "not configured"

    return out


@app.post("/extract-job-url")
def extract_job_url(req: ExtractURLRequest):
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
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
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
        cleaned, provider = llm_json(
            system=(
                "Extract just the job posting body from this scraped text. "
                'Return JSON only: {"job_description": string}. '
                "Strip navigation, cookie banners, and boilerplate."
            ),
            user=text[:15000],
        )
        return {
            "job_description": cleaned.get("job_description", text[:12000]),
            "provider": provider,
        }
    except Exception as e:
        log.warning(f"LLM cleanup failed, returning raw text: {e}")
        return {"job_description": text[:12000], "provider": "raw-scrape"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if not req.job_description.strip() or not req.resume_text.strip():
        raise HTTPException(400, "job_description and resume_text are required")

    data, provider = llm_json(
        system=(
            "You compare a resume against a job description. "
            "Return JSON only with keys: "
            "match_score (0-100 integer), summary (2-3 sentence string), "
            "matched_keywords (list of short strings), "
            "missing_keywords (list of short strings)."
        ),
        user=f"RESUME:\n{req.resume_text}\n\nJOB:\n{req.job_description}",
    )

    try:
        score = int(data.get("match_score", 0))
    except (TypeError, ValueError):
        score = 0

    return {
        "match_score":      max(0, min(100, score)),
        "summary":          str(data.get("summary", "")),
        "matched_keywords": list(data.get("matched_keywords", []) or []),
        "missing_keywords": list(data.get("missing_keywords", []) or []),
        "provider":         provider,
    }


@app.post("/generate")
def generate(req: GenerateRequest):
    if not req.job_description.strip() or not req.resume_text.strip():
        raise HTTPException(400, "job_description and resume_text are required")
    if not req.email.strip():
        raise HTTPException(400, "email is required")

    if not has_active_subscription(req.email):
        raise HTTPException(
            status_code=402,
            detail="An active subscription is required to generate tailored resumes and cover letters. "
                   "Use /create-checkout-session to subscribe.",
        )

    company = (req.company_name or "the company").strip()
    data, provider = llm_json(
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
        try:
            db.generations.insert_one({
                "email": req.email.lower(),
                "company": company,
                "tone": req.tone,
                "provider": provider,
                "result": data,
                "created_at": time.time(),
            })
        except Exception as e:
            log.warning(f"Failed to save generation: {e}")

    return {
        "tailored_bullets": list(data.get("tailored_bullets", []) or []),
        "cover_letter":     str(data.get("cover_letter", "")),
        "provider":         provider,
    }


@app.post("/create-checkout-session")
def create_checkout_session(req: CheckoutRequest):
    if not STRIPE_SECRET_KEY or not STRIPE_PRICE_ID:
        raise HTTPException(500, "Stripe is not configured on the server.")
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
    except stripe.error.StripeError as e:
        raise HTTPException(400, f"Stripe error: {e}")

    return {"checkout_url": session.url, "id": session.id}


@app.post("/stripe-webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(500, "Stripe webhook secret not configured on the server.")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except ValueError:
        raise HTTPException(400, "Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        raise HTTPException(400, f"Invalid signature: {e}")

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
    except Exception as e:
        log.error(f"Webhook processing error: {e}")

    return {"received": True}
