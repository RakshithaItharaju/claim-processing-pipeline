"""FastAPI entrypoint: POST /api/process"""
from __future__ import annotations

import logging
import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from .graph import build_graph
from .llm import LLMConfigError, get_llm
from .pdf_utils import PdfError, extract_pages

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("claims")

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "100"))
CLAIM_ID_RE = re.compile(r"^[A-Za-z0-9._\-]{1,64}$")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.graph = None
    app.state.startup_error = None
    try:
        app.state.graph = build_graph(get_llm())
    except LLMConfigError as exc:  # service still starts so /health can explain what is wrong
        app.state.startup_error = str(exc)
        log.error("LLM not configured: %s", exc)
    yield


app = FastAPI(title="Claim Processing Pipeline", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    ready = app.state.graph is not None
    return {"status": "ok" if ready else "degraded", "detail": app.state.startup_error}


@app.post("/api/process")
async def process(claim_id: str = Form(...), file: UploadFile = File(...)):
    if app.state.graph is None:
        raise HTTPException(503, f"LLM not configured: {app.state.startup_error}")
    if not CLAIM_ID_RE.match(claim_id):
        raise HTTPException(422, "claim_id must be 1-64 chars: letters, digits, '.', '_' or '-'")

    data = await file.read()
    if not data:
        raise HTTPException(400, "Uploaded file is empty")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")

    try:
        pages = await run_in_threadpool(extract_pages, data)
    except PdfError as exc:
        raise HTTPException(400, str(exc)) from exc
    if len(pages) > MAX_PAGES:
        raise HTTPException(413, f"PDF has {len(pages)} pages; limit is {MAX_PAGES}")

    log.info("claim=%s pages=%d", claim_id, len(pages))
    try:
        state = await run_in_threadpool(
            app.state.graph.invoke,
            {"claim_id": claim_id, "pages": pages, "warnings": []},
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("pipeline failed for claim %s", claim_id)
        raise HTTPException(502, f"Processing failed: {exc}") from exc
    return state["result"]
