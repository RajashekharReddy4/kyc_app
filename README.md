# KYC Authenticity Analyzer — MERN Monorepo (Option A)

This is a **monorepo** with:
- `client/` (React + Vite) — upload UI + report viewer
- `server/` (Node + Express) — API gateway, file uploads, MongoDB storage, orchestrates analysis
- `worker/` (Python + FastAPI) — runs the forensic pipeline and returns a report

> Why a Python worker? Tools like **OCRmyPDF, Tesseract, FFmpeg, ExifTool, qpdf** are easiest to run from Python and/or shell.
Node orchestrates; Python analyzes.

## Quick start (Docker)
1. Install Docker Desktop
2. From repo root:
```bash
docker compose up --build
```
3. Open:
- Client: http://localhost:5173
- Server health: http://localhost:4000/health
- Worker health: http://localhost:8000/health

## Quick start (Local, no Docker)
### 1) Worker (Python)
```bash
cd worker
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 2) Server (Node)
```bash
cd server
npm install
npm run dev
```

### 3) Client (React)
```bash
cd client
npm install
npm run dev
```

## External tools (recommended)
Install these on the **worker** host for best coverage:
- qpdf, exiftool, ocrmypdf, tesseract, poppler-utils (pdftoppm/pdftotext), ffmpeg, sox

If they are missing, the worker will mark related checks as **skipped**.

## Data model (MongoDB)
- Cases and reports are stored as JSON files under `server/uploads/<caseId>/`.

Artifacts are stored on disk under `server/uploads/<caseId>/`.

## Security notes
- Treat this as a demo/starter. Add authentication, rate limits, AV scanning, and S3 storage for production.
- This produces **risk signals**, not proof. Always include human review.
