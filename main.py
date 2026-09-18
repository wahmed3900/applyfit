"""
ApplyFit — Job Application Autopilot
FastAPI backend: extracts job requirements, scores resume match,
and generates a tailored resume + cover letter using the Claude API.

Run locally:
    pip install -r requirements.txt --break-system-packages
    Create a .env file with ANTHROPIC_API_KEY, STRIPE_SECRET_KEY,
    STRIPE_PRICE_ID, STRIPE_WEBHOOK_SECRET, and MONGODB_URI.
    uvicorn main:app --reload --port 8000
"""

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

load_dotenv()  # reads a local .env file if present; harmless no-op if it isn't (e.g. on Railway)

app = FastAPI(title="ApplyFit API")

# Allow the frontend (any origin for local dev — lock this down in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-6"

# ---------- Stripe setup ----------
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")
STRIPE_PRICE_ID = os.environ.get("STRIPE_PRICE_ID")  # the recurring Price ID from your Stripe dashboard
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET")

# ---------- MongoDB setup ----------
# Subscriber records persist here instead of a local JSON file, so they
# survive redeploys. Connection string comes from the MONGODB_URI env var —
# never hardcode it in this file.
MONGODB_URI = os.environ.get("MONGODB_URI")
_mongo_client = MongoClient(MONGODB_URI) if MONGODB_URI else None
_db = _mongo_client["applyfit"] if _mongo_client else None
_subscribers_collection = _db["subscribers"] if _db is not None else None


def _get_subscriber(email: str) -> Optional[dict]:
    if _subscribers_collection is None:
        raise HTTPException(status_code=500, detail="Database is not configured on the server.")
    return _subscribers_collection.find_one({"email": email.lower().strip()})


def _upsert_subscriber(email: str, active: bool, customer_id: Optional[str] = None) -> None:
    if _subscribers_collection is None:
        raise HTTPException(status_code=500, detail="Database is not configured on the server.")
    update = {"active": active}
    if customer_id:
        update["customer_id"] = customer_id
    _subscribers_collection.update_one(
        {"email": email.lower().strip()},
        {"$set": update},
        upsert=True,
    )


def _set_active_by_customer_id(customer_id: str, active: bool) -> None:
    if _subscribers_collection is None:
        raise HTTPException(status_code=500, detail="Database is not configured on the server.")
    _subscribers_collection.update_many(
        {"customer_id": customer_id},
        {"$set": {"active": active}},
    )


def is_subscribed(email: str) -> bool:
    record = _get_subscriber(email)
    return bool(record and record.get("active", False))


# ---------- Request/response models ----------

class AnalyzeRequest(BaseModel):
    job_description: Optional[str] = None
    job_url: Optional[str] = None
    resume_text: str


class MatchResult(BaseModel):
    match_score: int
    matched_keywords: list[str]
    missing_keywords: list[str]
    summary: str


class GenerateRequest(BaseModel):
    job_description: Optional[str] = None
    job_url: Optional[str] = None
    resume_text: str
    company_name: Optional[str] = None
    tone: Optional[str] = "professional"  # professional | friendly | formal
    email: str  # used to check subscription status


class GenerateResult(BaseModel):
    tailored_bullets: list[str]
    cover_letter: str


class CheckoutRequest(BaseModel):
    email: str
    success_url: str
    cancel_url: str


class CheckoutResult(BaseModel):
    checkout_url: str


class SubscriptionStatus(BaseModel):
    email: str
    active: bool


class JobUrlRequest(BaseModel):
    job_url: str


class JobUrlResult(BaseModel):
    job_description: str


# ---------- Helpers ----------

def call_claude_json(system: str, user_content: str) -> dict:
    """Call Claude and parse a strict-JSON response. Raises on malformed output."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        system=system,
        messages=[{"role": "user", "content": user_content}],
    )
    raw = "".join(block.text for block in response.content if block.type == "text")
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"Model returned non-JSON output: {e}")


def fetch_job_text(job_url: str) -> str:
    """Fetch a job posting URL and extract its main readable text.

    This is a best-effort HTML scrape — some sites (notably LinkedIn) block
    server-side scraping or require login, in which case this raises a clear
    error so the frontend can fall back to asking the user to paste the text.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    }
    try:
        resp = httpx.get(job_url, headers=headers, timeout=10, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Couldn't fetch that job URL ({e}). Some sites block automated "
                    "fetching — try pasting the job description text directly instead.",
        )

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = "\n".join(lines)

    if len(cleaned) < 200:
        raise HTTPException(
            status_code=422,
            detail="That page didn't return enough readable text — it may require "
                    "login or block automated fetching. Try pasting the job description instead.",
        )
    # Cap length so we don't send an entire page (nav clutter, unrelated content) to Claude.
    return cleaned[:8000]


def resolve_job_description(job_description: Optional[str], job_url: Optional[str]) -> str:
    """Given either pasted text or a URL, return the job description text to use."""
    if job_description and job_description.strip():
        return job_description.strip()
    if job_url and job_url.strip():
        return fetch_job_text(job_url.strip())
    raise HTTPException(status_code=422, detail="Provide either job_description or job_url.")


# ---------- Endpoints ----------

@app.post("/analyze", response_model=MatchResult)
def analyze(req: AnalyzeRequest):
    """Extract required skills from the job post and score the resume against them."""
    job_description = resolve_job_description(req.job_description, req.job_url)
    system = (
        "You are a precise resume-matching engine. Compare the job description "
        "against the resume. Respond ONLY with raw JSON, no preamble, no markdown "
        "fences, matching exactly this shape:\n"
        '{"match_score": <0-100 integer>, '
        '"matched_keywords": [<strings>], '
        '"missing_keywords": [<strings>], '
        '"summary": "<2-3 sentence plain-English assessment>"}'
    )
    user_content = (
        f"JOB DESCRIPTION:\n{job_description}\n\n"
        f"RESUME:\n{req.resume_text}\n\n"
        "Score how well the resume matches the job's required skills, tools, "
        "and experience level. List concrete keywords/skills that ARE present "
        "in the resume and ones that are required but MISSING."
    )
    data = call_claude_json(system, user_content)
    return MatchResult(**data)


@app.post("/generate", response_model=GenerateResult)
def generate(req: GenerateRequest):
    """Generate tailored resume bullets and a cover letter for this specific job.

    Gated behind an active subscription — the free tier is /analyze only.
    """
    if not is_subscribed(req.email):
        raise HTTPException(
            status_code=402,
            detail="An active subscription is required to generate tailored resumes and cover letters. "
                   "Use /create-checkout-session to subscribe.",
        )
    job_description = resolve_job_description(req.job_description, req.job_url)
    system = (
        "You are an expert resume writer and career coach. Respond ONLY with "
        "raw JSON, no preamble, no markdown fences, matching exactly this shape:\n"
        '{"tailored_bullets": [<3-6 rewritten resume bullet strings>], '
        '"cover_letter": "<full cover letter text, 3-4 paragraphs>"}\n\n'
        "Rules: Never invent experience, skills, or achievements the resume "
        "doesn't support. Only rephrase and reprioritize what's already there "
        "to match the job's language and priorities. Keep bullets factual."
    )
    company_line = f"Company: {req.company_name}\n" if req.company_name else ""
    user_content = (
        f"{company_line}Tone: {req.tone}\n\n"
        f"JOB DESCRIPTION:\n{job_description}\n\n"
        f"RESUME:\n{req.resume_text}\n\n"
        "Rewrite the most relevant resume bullets to mirror this job's "
        "priorities and keywords (staying 100% truthful to the original "
        "content), and write a tailored cover letter."
    )
    data = call_claude_json(system, user_content)
    return GenerateResult(**data)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract-job-url", response_model=JobUrlResult)
def extract_job_url(req: JobUrlRequest):
    """Fetch a job posting URL and return its extracted text, for previewing before analysis."""
    return JobUrlResult(job_description=fetch_job_text(req.job_url))


# ---------- Stripe endpoints ----------

@app.post("/create-checkout-session", response_model=CheckoutResult)
def create_checkout_session(req: CheckoutRequest):
    """Create a Stripe Checkout session for a subscription and return its URL."""
    if not stripe.api_key or not STRIPE_PRICE_ID:
        raise HTTPException(status_code=500, detail="Stripe is not configured on the server.")
    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            payment_method_types=["card"],
            billing_address_collection="auto",
            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            customer_email=req.email,
            success_url=req.success_url,
            cancel_url=req.cancel_url,
            metadata={"email": req.email},
        )
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return CheckoutResult(checkout_url=session.url)


@app.post("/webhook")
async def stripe_webhook(request: Request):
    """Stripe calls this when subscription events happen. Keeps the subscriber list current."""
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except (ValueError, stripe.error.SignatureVerificationError):
        raise HTTPException(status_code=400, detail="Invalid webhook signature or payload.")

    event_type = event["type"]
    data = event["data"]["object"]

    if event_type == "checkout.session.completed":
        email = (data.get("customer_email") or data.get("metadata", {}).get("email") or "").lower().strip()
        if email:
            _upsert_subscriber(email, active=True, customer_id=data.get("customer"))

    elif event_type in ("customer.subscription.deleted", "customer.subscription.updated"):
        customer_id = data.get("customer")
        status = data.get("status")
        if customer_id:
            _set_active_by_customer_id(customer_id, active=(status == "active"))

    return {"received": True}


@app.get("/subscription-status", response_model=SubscriptionStatus)
def subscription_status(email: str):
    """Lets the frontend check whether a given email currently has an active subscription."""
    return SubscriptionStatus(email=email, active=is_subscribed(email))
