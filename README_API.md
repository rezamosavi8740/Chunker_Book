# Chunker Book API

Pipeline:

`PDF -> OCR API -> OCR schema normalization -> Full-LLM semantic chunker -> chunks.jsonl`

The API is job-based so the UI can display clean progress without keeping one browser request open for a long book.

## Files to add to the repository

```text
Chunker_Book/
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .env.example
├── requirements-api.txt
├── api/
│   ├── __init__.py
│   └── main.py
└── src/
    └── persian_ocr_chunker_full_llm_v14.py
```

## Configure

```bash
cp .env.example .env
nano .env
```

Never commit `.env` or API keys.

At minimum set:

```env
OCR_BASE_URL=http://192.168.130.77:8091
OCR_API_KEY=...
CHUNKER_LLM_URL=http://YOUR_LLM_HOST:PORT/v1/chat/completions
CHUNKER_LLM_MODEL=auto
```

## Build and run

```bash
docker compose build
docker compose up -d
docker compose logs -f chunker-api
```

Docs:

```text
http://SERVER_IP:8080/docs
```

Liveness:

```bash
curl http://127.0.0.1:8080/health
```

Readiness/config check:

```bash
curl http://127.0.0.1:8080/ready
```

## Submit a PDF

```bash
curl -sS -X POST http://127.0.0.1:8080/v1/jobs \
  -F "pdf_file=@$HOME/data/books_pdf/zist.pdf"
```

Response:

```json
{
  "job_id": "...",
  "status": "queued",
  "status_url": "http://127.0.0.1:8080/v1/jobs/...",
  "events_url": "http://127.0.0.1:8080/v1/jobs/.../events"
}
```

## Check progress

```bash
curl -sS http://127.0.0.1:8080/v1/jobs/JOB_ID | jq
```

The UI-oriented stages are:

```text
queued
ocr
normalizing
book_plan
atomizing
planning
validating
finalizing
completed
```

On failure, the same endpoint returns a structured object such as:

```json
{
  "status": "failed",
  "stage": "planning",
  "error": {
    "code": "CHUNKER_FAILED",
    "stage": "planning",
    "message": "..."
  }
}
```

## Live UI events (SSE)

```bash
curl -N http://127.0.0.1:8080/v1/jobs/JOB_ID/events
```

Each event is a compact JSON payload containing `stage`, `progress`, `level`, and a human-readable `message`.

## Get chunks as JSON

```bash
curl -sS http://127.0.0.1:8080/v1/jobs/JOB_ID/chunks > chunks.json
```

Response shape:

```json
{
  "job_id": "...",
  "count": 123,
  "chunks": [
    {"id": "...", "text": "..."}
  ]
}
```

## Download JSONL

```bash
curl -fLO http://127.0.0.1:8080/v1/jobs/JOB_ID/download/chunks.jsonl
```

Other debug/audit artifacts:

```text
GET /v1/jobs/{job_id}/download/ocr.json
GET /v1/jobs/{job_id}/download/chunker.log
```

## Important design choices

- The OCR service currently returns `pages[].cells`; the chunker expects page content under `blocks/items/layout/...`. The API stores raw OCR unchanged and writes a normalized copy with `cells -> blocks` for the chunker.
- Full-LLM mode is always invoked with `--preset sft_dpo_quality`.
- A chunker failure does not return fake/partial chunks. The job becomes `failed` and the UI gets a structured stage/error.
- `chunks.jsonl` is preserved as the canonical artifact; `/chunks` is only a convenient JSON-array view for the UI.
- Job data is persisted under `/data/jobs`, so completed results survive container restarts.
- An in-flight job is marked failed with `SERVER_RESTARTED` after an API restart; no silent success is reported.
- Keep Uvicorn at one worker for this first version because the queue is in-process. For multi-node/high-throughput deployment, replace the in-process executor with Redis/Celery/RQ/Kubernetes jobs rather than simply increasing Uvicorn workers.
