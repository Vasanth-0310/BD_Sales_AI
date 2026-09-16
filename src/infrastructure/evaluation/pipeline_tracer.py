"""
Non-blocking pipeline data collector for RAGAS evaluation.

Appends JSON-lines samples to data/eval_samples.jsonl every time a live
profile-match or project-match API call completes.  Always runs as a
background asyncio.create_task — never delays the API response.

Toggle off via: RAGAS_CAPTURE_ENABLED=false in .env
"""
import asyncio
import json
import os
import time
from pathlib import Path

from src.common.config import settings
from src.common.logger import get_logger

logger = get_logger(__name__)

_EVAL_FILE = Path("data/eval_samples.jsonl")
_MAX_SAMPLES = 200
_write_lock = asyncio.Lock()


# ─── Public capture helpers ───────────────────────────────────────────────────

async def capture_profile_sample(
    question: str,
    retrieval_contexts: list[str],
    answer: str,
    retrieval_meta: dict,
    generation_meta: dict,
) -> None:
    """Persist one profile-match evaluation sample (non-blocking)."""
    if not settings.ragas_capture_enabled:
        return
    sample = {
        "pipeline": "profile_match",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question": question,
        "retrieval_contexts": retrieval_contexts,
        "answer": answer,
        "retrieval_meta": retrieval_meta,
        "generation_meta": generation_meta,
    }
    await _append_sample(sample)


async def capture_project_sample(
    question: str,
    stage1_contexts: list[str],
    stage2_contexts: list[str],
    answer: str,
    retrieval_meta: dict,
) -> None:
    """Persist one project-match evaluation sample (non-blocking)."""
    if not settings.ragas_capture_enabled:
        return
    sample = {
        "pipeline": "project_match",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question": question,
        # RAGAS 'contexts' = the deepest evidence shown to the LLM
        "retrieval_contexts": stage2_contexts,
        # Stage-1 summaries kept separately for retrieval-phase analysis
        "stage1_contexts": stage1_contexts,
        "answer": answer,
        "retrieval_meta": retrieval_meta,
    }
    await _append_sample(sample)


# ─── Internal helpers ─────────────────────────────────────────────────────────

async def _append_sample(sample: dict) -> None:
    try:
        async with _write_lock:
            await asyncio.to_thread(_write_sample_sync, sample)
        logger.debug(
            "RAGAS tracer: sample captured  |  pipeline=%s", sample.get("pipeline")
        )
    except Exception as exc:
        logger.warning("RAGAS tracer: failed to capture sample (non-fatal): %s", exc)


def _write_sample_sync(sample: dict) -> None:
    """Atomically append sample; enforces rolling _MAX_SAMPLES cap."""
    _EVAL_FILE.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    if _EVAL_FILE.exists():
        with open(_EVAL_FILE, "r", encoding="utf-8") as fh:
            lines = [ln for ln in fh.readlines() if ln.strip()]

    # Rolling cap: keep the most recent _MAX_SAMPLES - 1 so new one fits
    if len(lines) >= _MAX_SAMPLES:
        lines = lines[-(_MAX_SAMPLES - 1):]

    lines.append(json.dumps(sample, ensure_ascii=False) + "\n")

    tmp = _EVAL_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.writelines(lines)
    os.replace(tmp, _EVAL_FILE)
