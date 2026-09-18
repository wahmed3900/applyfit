# ApplyFit

**Tailor your resume to any job description — instantly, and honestly.**

ApplyFit analyzes a job posting and your resume, then shows you exactly where you match, where you're missing keywords, and how to rephrase your experience to fit the role — without inventing anything.

> Built for job seekers who are tired of guessing what recruiters' filters want.

---

## Why ApplyFit?

Most "resume optimizers" either:
- Stuff keywords until your resume reads like spam, or
- Rewrite your experience into fiction.

ApplyFit does neither. It maps **what you've actually done** to **what the job actually asks for**, and tells you where the gaps are.

---

## Features

- 🎯 **Match Score** — A 0–100 fit score between your resume and the job description
- 🔑 **Keyword Gap Analysis** — Missing skills and terms, ranked by importance
- ✍️ **Bullet Rewriter** — Rephrases your existing bullets using the job's language (no fabrication)
- 📊 **Section Breakdown** — Feedback per resume section: summary, experience, skills, education
- 📄 **PDF & DOCX Support** — Upload your resume in either format
- 🔒 **Local-First Option** — Run fully offline with a local LLM

---

## Demo

```bash
$ applyfit analyze --resume resume.pdf --job job_posting.txt

Match Score: 74/100

✅ Strong matches:
   - Python, SQL, data pipelines
   - Cross-functional collaboration

⚠️  Missing keywords:
   - "dbt" (mentioned 4x in JD)
   - "stakeholder management"
   - "A/B testing"

✍️  Suggested rewrite:
   Before: "Worked with marketing team on reports"
   After:  "Partnered with marketing stakeholders to deliver
            A/B tested reporting pipelines in SQL"
```

---

## Disclaimer

ApplyFit does not fabricate experience. It only rephrases what you've already written. You are responsible for the accuracy of your final resume.
