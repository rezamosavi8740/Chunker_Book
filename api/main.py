from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel


APP_VERSION = "0.1.0"
DATA_ROOT = Path(os.getenv("DATA_ROOT", "/data/jobs")).resolve()
CHUNKER_SCRIPT = Path(
    os.getenv("CHUNKER_SCRIPT", "/app/src/persian_ocr_chunker_full_llm_v14.py")
).resolve()
OCR_BASE_URL = os.getenv("OCR_BASE_URL", "http://192.168.130.77:8091").rstrip("/")
OCR_API_KEY = os.getenv("OCR_API_KEY", "")
OCR_TIMEOUT_SECONDS = int(os.getenv("OCR_TIMEOUT_SECONDS", "7200"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
JOB_WORKERS = max(1, int(os.getenv("JOB_WORKERS", "1")))
CORS_ORIGINS = [x.strip() for x in os.getenv("CORS_ORIGINS", "*").split(",") if x.strip()]

CHUNKER_LLM_URL = os.getenv("CHUNKER_LLM_URL", "")
CHUNKER_LLM_URLS = os.getenv("CHUNKER_LLM_URLS", "")
CHUNKER_LLM_HOST = os.getenv("CHUNKER_LLM_HOST", "")
CHUNKER_LLM_PORTS = os.getenv("CHUNKER_LLM_PORTS", "")
CHUNKER_LLM_MODEL = os.getenv("CHUNKER_LLM_MODEL", "auto")
CHUNKER_LLM_API_KEY = os.getenv("CHUNKER_LLM_API_KEY", "")
CHUNKER_LLM_TIMEOUT = int(os.getenv("CHUNKER_LLM_TIMEOUT", "240"))
CHUNKER_LLM_CONCURRENCY = max(1, min(64, int(os.getenv("CHUNKER_LLM_CONCURRENCY", "16"))))
CHUNKER_LLM_MAX_OUTPUT_TOKENS = int(os.getenv("CHUNKER_LLM_MAX_OUTPUT_TOKENS", "1024"))
ATOMIZER_OVERLAP_LINES = int(os.getenv("ATOMIZER_OVERLAP_LINES", "8"))
PLANNER_OVERLAP_UNITS = int(os.getenv("PLANNER_OVERLAP_UNITS", "4"))

TERMINAL = {"completed", "failed"}
EXECUTOR = ThreadPoolExecutor(max_workers=JOB_WORKERS, thread_name_prefix="chunker-job")
STATUS_LOCK = threading.Lock()

DATA_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Persian Book Chunker API",
    version=APP_VERSION,
    description=(
        "PDF → OCR → Full-LLM BookPlan/Atomizer/Planner → chunks.jsonl. "
        "Designed as a job API so a UI can show readable progress and errors."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False if CORS_ORIGINS == ["*"] else True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class JobAccepted(BaseModel):
    job_id: str
    status: str
    status_url: str
    events_url: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def job_dir(job_id: str) -> Path:
    # UUID validation avoids arbitrary path traversal.
    try:
        normalized = str(uuid.UUID(job_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    return DATA_ROOT / normalized


def paths_for(job_id: str) -> Dict[str, Path]:
    root = job_dir(job_id)
    return {
        "root": root,
        "source": root / "source" / "book.pdf",
        "ocr_raw": root / "ocr" / "ocr_raw.json",
        "ocr_book": root / "ocr" / "book",
        "ocr_normalized": root / "ocr" / "book" / "ocr.json",
        "chunker_root": root / "chunker",
        "status": root / "status.json",
        "events": root / "events.jsonl",
        "chunker_log": root / "chunker.log",
    }


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_status(job_id: str) -> Dict[str, Any]:
    p = paths_for(job_id)["status"]
    if not p.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return read_json(p)


def write_status(job_id: str, **changes: Any) -> Dict[str, Any]:
    p = paths_for(job_id)["status"]
    with STATUS_LOCK:
        current = read_json(p) if p.exists() else {"job_id": job_id, "created_at": utc_now()}
        current.update(changes)
        current["updated_at"] = utc_now()
        atomic_write_json(p, current)
    return current


def emit_event(
    job_id: str,
    *,
    level: str,
    stage: str,
    message: str,
    progress: Optional[int] = None,
    code: Optional[str] = None,
) -> None:
    p = paths_for(job_id)["events"]
    event: Dict[str, Any] = {
        "ts": utc_now(),
        "level": level,
        "stage": stage,
        "message": message,
    }
    if progress is not None:
        event["progress"] = max(0, min(100, int(progress)))
    if code:
        event["code"] = code
    p.parent.mkdir(parents=True, exist_ok=True)
    with STATUS_LOCK:
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def set_stage(job_id: str, stage: str, progress: int, message: str) -> None:
    status = read_status(job_id)
    # Avoid noisy duplicate UI events.
    duplicate = (
        status.get("stage") == stage
        and status.get("progress") == progress
        and status.get("message") == message
    )
    write_status(job_id, status="processing", stage=stage, progress=progress, message=message)
    if not duplicate:
        emit_event(job_id, level="info", stage=stage, progress=progress, message=message)


def fail_job(job_id: str, stage: str, code: str, message: str, details: Optional[str] = None) -> None:
    err: Dict[str, Any] = {"code": code, "stage": stage, "message": message}
    if details:
        err["details"] = details[-12000:]
    write_status(
        job_id,
        status="failed",
        stage=stage,
        progress=100,
        message=message,
        error=err,
        completed_at=utc_now(),
    )
    emit_event(job_id, level="error", stage=stage, progress=100, message=message, code=code)


def normalize_ocr_for_chunker(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize this OCR service's pages[].cells schema to pages[].blocks.

    Raw OCR is stored separately. This function does not summarize or rewrite
    prose. For Table cells only, plain_text is preferred when available so
    downstream structural checks do not receive avoidable HTML markup.
    """
    pages = raw.get("pages")
    if not isinstance(pages, list):
        raise ValueError("OCR response does not contain a pages array")

    normalized_pages = []
    for index, page in enumerate(pages):
        if not isinstance(page, dict):
            continue
        cells = page.get("cells")
        if not isinstance(cells, list):
            cells = page.get("blocks")
        if not isinstance(cells, list):
            cells = []

        blocks = []
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            category = str(cell.get("category") or "Text")
            text = str(cell.get("text") or "")
            if category.lower() == "table" and str(cell.get("plain_text") or "").strip():
                text = str(cell.get("plain_text"))
            block = dict(cell)
            block["category"] = category
            block["text"] = text
            blocks.append(block)

        normalized_pages.append(
            {
                "page_no": page.get("page_no", index),
                "blocks": blocks,
            }
        )

    if not normalized_pages:
        raise ValueError("OCR response contains no usable pages")
    return {"pages": normalized_pages}


def call_ocr(job_id: str, pdf_path: Path) -> Dict[str, Any]:
    if not OCR_API_KEY:
        raise RuntimeError("OCR_API_KEY is not configured")

    url = f"{OCR_BASE_URL}/ocr/pdf"
    headers = {"Authorization": f"Bearer {OCR_API_KEY}"}
    data = {
        "prompt_mode": "prompt_layout_all_en",
        "post_correct": "false",
        "quality_retry": "true",
        "quality_max_retries": "1",
    }
    timeout = httpx.Timeout(
        connect=20.0,
        read=float(OCR_TIMEOUT_SECONDS),
        write=float(OCR_TIMEOUT_SECONDS),
        pool=20.0,
    )
    with pdf_path.open("rb") as f, httpx.Client(timeout=timeout) as client:
        response = client.post(
            url,
            headers=headers,
            data=data,
            files={"pdf_file": (pdf_path.name, f, "application/pdf")},
        )
    if response.status_code >= 400:
        detail = response.text[:6000]
        raise RuntimeError(f"OCR HTTP {response.status_code}: {detail}")
    try:
        obj = response.json()
    except Exception as exc:
        raise RuntimeError(f"OCR returned non-JSON response: {response.text[:2000]}") from exc
    if not isinstance(obj, dict):
        raise RuntimeError("OCR returned an unexpected JSON value")
    return obj


def chunker_command(input_folder: Path, output_root: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(CHUNKER_SCRIPT),
        "--input-folders",
        str(input_folder),
        "--output-root",
        str(output_root),
        "--workers",
        "1",
        "--preset",
        "sft_dpo_quality",
        "--output-mode",
        "minimal",
        "--force",
        "--llm-concurrency",
        str(CHUNKER_LLM_CONCURRENCY),
        "--llm-timeout",
        str(CHUNKER_LLM_TIMEOUT),
        "--llm-max-output-tokens",
        str(CHUNKER_LLM_MAX_OUTPUT_TOKENS),
        "--atomizer-overlap-lines",
        str(ATOMIZER_OVERLAP_LINES),
        "--planner-overlap-units",
        str(PLANNER_OVERLAP_UNITS),
        "--llm-model",
        CHUNKER_LLM_MODEL,
    ]
    if CHUNKER_LLM_URLS:
        cmd += ["--llm-urls", CHUNKER_LLM_URLS]
    elif CHUNKER_LLM_HOST and CHUNKER_LLM_PORTS:
        cmd += ["--llm-host", CHUNKER_LLM_HOST, "--llm-ports", CHUNKER_LLM_PORTS]
    elif CHUNKER_LLM_URL:
        cmd += ["--llm-url", CHUNKER_LLM_URL]
    if CHUNKER_LLM_API_KEY:
        cmd += ["--llm-api-key", CHUNKER_LLM_API_KEY]
    return cmd


PWIN_RE = re.compile(r"pwin=(\d+)/(\d+)")


def user_friendly_chunker_progress(job_id: str, line: str, memory: Dict[str, Any]) -> None:
    """Convert verbose CLI logs to low-noise user-facing progress events."""
    stage = None
    progress = None
    message = None

    if "[BOOK_PLAN_LLM_CALL]" in line:
        stage, progress, message = "book_plan", 40, "Analyzing the book structure"
    elif "[BOOK_PLAN_LLM_DONE]" in line:
        stage, progress, message = "book_plan", 45, "Book strategy is ready"
    elif "[ATOMIZER_PARALLEL]" in line or "[ATOMIZER_LLM_CALL]" in line:
        stage, progress, message = "atomizing", 48, "Finding semantic atomic units"
    elif "[ATOMIZER_LLM_DONE]" in line:
        m = PWIN_RE.search(line)
        if m:
            i, n = int(m.group(1)), max(1, int(m.group(2)))
            progress = 48 + round(18 * i / n)
        else:
            progress = 60
        stage, message = "atomizing", "Semantic atomization in progress"
    elif "[PLANNER_PARALLEL]" in line or "[PLANNER_LLM_CALL]" in line:
        stage, progress, message = "planning", 70, "Building coherent chunks from atomic units"
    elif "[PLANNER_LLM_DONE]" in line:
        m = PWIN_RE.search(line)
        if m:
            i, n = int(m.group(1)), max(1, int(m.group(2)))
            progress = 70 + round(17 * i / n)
        else:
            progress = 82
        stage, message = "planning", "Chunk planning in progress"
    elif "[GROUP_COVERAGE]" in line:
        stage, progress, message = "validating", 91, "Validating exact text coverage"
    elif "[DONE]" in line:
        stage, progress, message = "finalizing", 96, "Chunking completed; preparing output"

    if stage is None:
        return
    # Parallel windows may finish out of order. Keep UI progress monotonic.
    stage_key = f"max_progress:{stage}"
    progress = max(int(progress), int(memory.get(stage_key, 0)))
    memory[stage_key] = progress
    key = (stage, progress, message)
    if memory.get("last") == key:
        return
    memory["last"] = key
    set_stage(job_id, stage, progress, str(message))


def run_chunker(job_id: str, input_folder: Path, output_root: Path) -> Path:
    p = paths_for(job_id)
    output_root.mkdir(parents=True, exist_ok=True)
    cmd = chunker_command(input_folder, output_root)
    progress_memory: Dict[str, Any] = {}

    with p["chunker_log"].open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            user_friendly_chunker_progress(job_id, line, progress_memory)
        rc = proc.wait()

    if rc != 0:
        manifest = output_root / "run_manifest.json"
        details = ""
        if manifest.exists():
            try:
                details = json.dumps(read_json(manifest), ensure_ascii=False, indent=2)
            except Exception:
                details = ""
        if not details and p["chunker_log"].exists():
            lines = p["chunker_log"].read_text(encoding="utf-8", errors="replace").splitlines()
            details = "\n".join(lines[-80:])
        raise RuntimeError(f"Chunker exited with code {rc}\n{details}")

    candidates = sorted(output_root.glob("*/chunks.jsonl"))
    candidates = [x for x in candidates if "_llm_cache" not in x.parts]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one chunks.jsonl, found {len(candidates)}")
    return candidates[0]


def count_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def run_job(job_id: str) -> None:
    p = paths_for(job_id)
    try:
        write_status(job_id, status="processing", started_at=utc_now(), error=None)

        set_stage(job_id, "ocr", 5, "Sending PDF to OCR service")
        raw = call_ocr(job_id, p["source"])
        p["ocr_raw"].parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(p["ocr_raw"], raw)
        pages = raw.get("pages") if isinstance(raw.get("pages"), list) else []
        set_stage(job_id, "ocr", 30, f"OCR completed ({len(pages)} pages)")

        set_stage(job_id, "normalizing", 34, "Preparing OCR output for semantic chunking")
        normalized = normalize_ocr_for_chunker(raw)
        p["ocr_book"].mkdir(parents=True, exist_ok=True)
        atomic_write_json(p["ocr_normalized"], normalized)

        set_stage(job_id, "chunking", 38, "Starting Full-LLM semantic chunker")
        chunks_path = run_chunker(job_id, p["ocr_book"], p["chunker_root"])
        chunk_count = count_jsonl(chunks_path)

        if chunk_count <= 0:
            raise RuntimeError("Chunker completed but produced zero chunks")

        relative_chunks = str(chunks_path.relative_to(p["root"]))
        write_status(
            job_id,
            status="completed",
            stage="completed",
            progress=100,
            message=f"Completed successfully ({chunk_count} chunks)",
            completed_at=utc_now(),
            result={
                "chunk_count": chunk_count,
                "chunks_path": relative_chunks,
                "chunks_json_url": f"/v1/jobs/{job_id}/chunks",
                "chunks_jsonl_download_url": f"/v1/jobs/{job_id}/download/chunks.jsonl",
                "ocr_download_url": f"/v1/jobs/{job_id}/download/ocr.json",
                "log_download_url": f"/v1/jobs/{job_id}/download/chunker.log",
            },
        )
        emit_event(
            job_id,
            level="success",
            stage="completed",
            progress=100,
            message=f"Completed successfully ({chunk_count} chunks)",
        )
    except Exception as exc:
        status = read_status(job_id)
        stage = str(status.get("stage") or "unknown")
        if stage == "ocr":
            code = "OCR_FAILED"
            user_message = "OCR processing failed"
        elif stage == "normalizing":
            code = "OCR_NORMALIZATION_FAILED"
            user_message = "OCR output could not be prepared for chunking"
        else:
            code = "CHUNKER_FAILED"
            user_message = "Semantic chunking failed"
        fail_job(job_id, stage, code, user_message, details=str(exc))


def artifact_path(job_id: str, kind: str) -> Path:
    p = paths_for(job_id)
    status = read_status(job_id)
    if kind == "ocr":
        result = p["ocr_raw"]
    elif kind == "log":
        result = p["chunker_log"]
    elif kind == "chunks":
        rel = ((status.get("result") or {}).get("chunks_path"))
        if not rel:
            raise HTTPException(status_code=409, detail="Chunks are not available yet")
        result = p["root"] / rel
    else:
        raise HTTPException(status_code=404, detail="Artifact not found")
    if not result.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return result


@app.on_event("startup")
def startup_reconcile() -> None:
    """Mark stale in-process jobs as failed after a container/server restart."""
    for status_path in DATA_ROOT.glob("*/status.json"):
        try:
            status = read_json(status_path)
            if status.get("status") in {"queued", "processing"}:
                jid = str(status.get("job_id") or status_path.parent.name)
                fail_job(
                    jid,
                    str(status.get("stage") or "server"),
                    "SERVER_RESTARTED",
                    "The API restarted while this job was running. Submit the PDF again.",
                )
        except Exception:
            continue


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": APP_VERSION,
        "job_workers": JOB_WORKERS,
    }


@app.get("/ready")
def ready() -> Dict[str, Any]:
    llm_configured = bool(CHUNKER_LLM_URLS or CHUNKER_LLM_URL or (CHUNKER_LLM_HOST and CHUNKER_LLM_PORTS))
    checks = {
        "chunker_script": CHUNKER_SCRIPT.exists(),
        "ocr_api_key": bool(OCR_API_KEY),
        "llm_endpoint": llm_configured,
        "data_root": DATA_ROOT.exists(),
    }
    if not all(checks.values()):
        raise HTTPException(status_code=503, detail={"status": "not_ready", "checks": checks})
    return {"status": "ready", "checks": checks}


@app.post("/v1/jobs", response_model=JobAccepted, status_code=202)
async def create_job(request: Request, pdf_file: UploadFile = File(...)) -> JobAccepted:
    name = (pdf_file.filename or "book.pdf").lower()
    if not name.endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are accepted")

    jid = str(uuid.uuid4())
    p = paths_for(jid)
    p["source"].parent.mkdir(parents=True, exist_ok=True)

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    try:
        with p["source"].open("wb") as out:
            while True:
                chunk = await pdf_file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(status_code=413, detail=f"PDF exceeds {MAX_UPLOAD_MB} MB limit")
                out.write(chunk)
    except Exception:
        shutil.rmtree(p["root"], ignore_errors=True)
        raise
    finally:
        await pdf_file.close()

    with p["source"].open("rb") as f:
        if f.read(5) != b"%PDF-":
            shutil.rmtree(p["root"], ignore_errors=True)
            raise HTTPException(status_code=400, detail="Uploaded file does not have a valid PDF signature")

    base = str(request.base_url).rstrip("/")
    initial = {
        "job_id": jid,
        "filename": pdf_file.filename or "book.pdf",
        "size_bytes": size,
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "message": "PDF uploaded; waiting for processing",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "error": None,
        "result": None,
    }
    atomic_write_json(p["status"], initial)
    emit_event(jid, level="info", stage="queued", progress=0, message=initial["message"])
    EXECUTOR.submit(run_job, jid)

    return JobAccepted(
        job_id=jid,
        status="queued",
        status_url=f"{base}/v1/jobs/{jid}",
        events_url=f"{base}/v1/jobs/{jid}/events",
    )


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    return read_status(job_id)


@app.get("/v1/jobs/{job_id}/chunks")
def get_chunks(job_id: str) -> Dict[str, Any]:
    status = read_status(job_id)
    if status.get("status") == "failed":
        raise HTTPException(status_code=409, detail={"message": "Job failed", "error": status.get("error")})
    if status.get("status") != "completed":
        raise HTTPException(status_code=409, detail={"message": "Job is not completed", "status": status.get("status"), "stage": status.get("stage"), "progress": status.get("progress")})
    path = artifact_path(job_id, "chunks")
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return {"job_id": job_id, "count": len(chunks), "chunks": chunks}


@app.get("/v1/jobs/{job_id}/download/chunks.jsonl")
def download_chunks(job_id: str) -> FileResponse:
    path = artifact_path(job_id, "chunks")
    return FileResponse(path, media_type="application/x-ndjson", filename=f"{job_id}_chunks.jsonl")


@app.get("/v1/jobs/{job_id}/download/ocr.json")
def download_ocr(job_id: str) -> FileResponse:
    path = artifact_path(job_id, "ocr")
    return FileResponse(path, media_type="application/json", filename=f"{job_id}_ocr.json")


@app.get("/v1/jobs/{job_id}/download/chunker.log")
def download_log(job_id: str) -> FileResponse:
    path = artifact_path(job_id, "log")
    return FileResponse(path, media_type="text/plain", filename=f"{job_id}_chunker.log")


@app.get("/v1/jobs/{job_id}/events")
async def stream_events(job_id: str) -> StreamingResponse:
    p = paths_for(job_id)
    # Validate job now, before returning the stream.
    read_status(job_id)

    async def event_stream() -> Iterable[str]:
        offset = 0
        while True:
            if p["events"].exists():
                with p["events"].open("r", encoding="utf-8") as f:
                    f.seek(offset)
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        offset = f.tell()
                        yield f"data: {line.strip()}\n\n"
            status = read_status(job_id)
            if status.get("status") in TERMINAL:
                break
            yield ": keep-alive\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
