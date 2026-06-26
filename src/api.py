"""Phase 4 — FastAPI search service.

Endpoints:
  GET  /health              service status + index size
  POST /search/text         {query, k?, rewrite?} -> ranked results
  POST /search/image        multipart image upload -> ranked results
  GET  /image/{id}          stream an image by id (for the UI)

Run:  uvicorn src.api:app --host 0.0.0.0 --port 8000
Set MMSEARCH_CONFIG to point at a serve config (default configs/serve.yaml).
"""

from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel

from .common import load_config, setup_logging
from .llm import QueryAssistant
from .search import Searcher

log = logging.getLogger("mmsearch")

CONFIG_PATH = os.environ.get("MMSEARCH_CONFIG", "configs/serve.yaml")
STATE: dict = {}


MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB cap before decoding an untrusted image


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    cfg = load_config(CONFIG_PATH)
    STATE["cfg"] = cfg
    # Build the searcher/llm unless they were pre-injected (e.g. by a test).
    if STATE.get("searcher") is None:
        STATE["searcher"] = Searcher.from_config(cfg)
    if STATE.get("llm") is None:
        STATE["llm"] = QueryAssistant(cfg.get("llm", {}))
    STATE["version"] = STATE.get("version", 0) + 1  # bump invalidates the response cache
    log.info("Service ready (%d images indexed).", len(STATE["searcher"].manifest))
    yield
    STATE.clear()


app = FastAPI(title="Multimodal Image Search", version="0.1.0", lifespan=lifespan)


class TextQuery(BaseModel):
    query: str
    k: int | None = None
    rewrite: bool = False


def _searcher() -> Searcher:
    s = STATE.get("searcher")
    if s is None:
        raise HTTPException(503, "Searcher not ready")
    return s


def _resolve_k(k: int | None, default: int) -> int:
    if k is None:
        return default
    if k <= 0:
        raise HTTPException(422, "k must be a positive integer")
    return k


@lru_cache(maxsize=512)
def _cached_text_search(version: int, used: str, k: int):
    # `version` keys the cache to the loaded index so a reload invalidates it.
    return STATE["searcher"].search_text(used, k)


@app.get("/health")
def health():
    s = STATE.get("searcher")
    return {"status": "ok" if s else "loading", "n_images": len(s.manifest) if s else 0}


@app.post("/search/text")
def search_text(body: TextQuery):
    s = _searcher()
    used = STATE["llm"].rewrite(body.query) if body.rewrite else body.query
    k = _resolve_k(body.k, s.top_k)
    results = _cached_text_search(STATE.get("version", 0), used, k)
    return {"query": body.query, "used_query": used, "results": results}


@app.post("/search/image")
async def search_image(file: UploadFile = File(...), k: int | None = None):
    s = _searcher()
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image too large")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image upload")
    return {"results": s.search_image(img, _resolve_k(k, s.top_k))}


@app.get("/image/{image_id}")
def get_image(image_id: int):
    s = _searcher()
    rec = s.manifest.get(image_id)
    if not rec:
        raise HTTPException(404, "Unknown image id")
    if not os.path.isfile(rec["path"]):
        raise HTTPException(404, "Image file is missing")
    return FileResponse(rec["path"])
