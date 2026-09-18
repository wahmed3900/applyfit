# ApplyFit — Job Application Autopilot (MVP backend)

Paste a job description + your resume → get a match score, missing
keywords, tailored resume bullets, and a cover letter.

## Setup

```bash
pip install -r requirements.txt --break-system-packages
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn main:app --reload --port 8000
```

## Endpoints

### `POST /analyze`
Scores the resume against the job description.

Request:
```json
{
  "job_description": "We're looking for a Python developer with FastAPI...",
  "resume_text": "5 years experience building REST APIs in Python..."
}
```

Response:
```json
{
  "match_score": 78,
  "matched_keywords": ["Python", "REST APIs"],
  "missing_keywords": ["FastAPI", "Docker"],
  "summary": "Strong overall fit but the resume doesn't mention FastAPI or Docker directly."
}
```

### `POST /generate`
Generates tailored resume bullets + a cover letter.

Request:
```json
{
  "job_description": "...",
  "resume_text": "...",
  "company_name": "Acme Corp",
  "tone": "professional"
}
```

Response:
```json
{
  "tailored_bullets": ["...", "..."],
  "cover_letter": "Dear Hiring Manager, ..."
}
```

### `GET /health`
Simple liveness check.

## Notes on truthfulness
The `/generate` prompt explicitly instructs the model to only rephrase
and reprioritize existing resume content — never invent skills or
experience. Worth spot-checking outputs before relying on it, since
LLMs can still drift.

## Next steps
- Add a simple frontend (paste boxes + results view).
- Add docx/PDF export of the tailored resume and cover letter.
- Add job-post URL scraping (currently expects pasted text).
- Add auth + saved history if you want a persistent product instead of a stateless tool.
- Deploy to Railway (same flow as your other projects) — set `ANTHROPIC_API_KEY` as an environment variable there.
