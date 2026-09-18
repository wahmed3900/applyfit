"""
ApplyFit — Job Application Autopilot
FastAPI backend: extracts job requirements, scores resume match,
and generates a tailored resume + cover letter using the Claude API.

Run locally:
    pip install -r requirements.txt --break-system-packages
    export ANTHROPIC_API_KEY=sk-ant-...
    uvicorn main:app --reload --port 8000
"""

import os
import json
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic

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


# ---------- Request/response models ----------

class AnalyzeRequest(BaseModel):
    job_description: str
    resume_text: str


class MatchResult(BaseModel):
    match_score: int
    matched_keywords: list[str]
    missing_keywords: list[str]
    summary: str


class GenerateRequest(BaseModel):
    job_description: str
    resume_text: str
    company_name: Optional[str] = None
    tone: Optional[str] = "professional"  # professional | friendly | formal


class GenerateResult(BaseModel):
    tailored_bullets: list[str]
    cover_letter: str


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


# ---------- Endpoints ----------

@app.post("/analyze", response_model=MatchResult)
def analyze(req: AnalyzeRequest):
    """Extract required skills from the job post and score the resume against them."""
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
        f"JOB DESCRIPTION:\n{req.job_description}\n\n"
        f"RESUME:\n{req.resume_text}\n\n"
        "Score how well the resume matches the job's required skills, tools, "
        "and experience level. List concrete keywords/skills that ARE present "
        "in the resume and ones that are required but MISSING."
    )
    data = call_claude_json(system, user_content)
    return MatchResult(**data)


@app.post("/generate", response_model=GenerateResult)
def generate(req: GenerateRequest):
    """Generate tailored resume bullets and a cover letter for this specific job."""
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
        f"JOB DESCRIPTION:\n{req.job_description}\n\n"
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
