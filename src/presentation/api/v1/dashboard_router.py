"""
Dashboard router — exposes two read-only endpoints:

  GET /api/v1/dashboard/metrics
      → API call counts, token totals, INR cost

  GET /api/v1/dashboard/logs
      → Tail of rag_pipeline.log
"""

from pathlib import Path
import re

from fastapi import APIRouter, HTTPException, Query, Request

from src.common.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/dashboard",
    tags=["Dashboard"],
)

_LOG_FILE = Path("rag_pipeline.log")

_DEFAULT_LINES = 500
_MAX_LINES = 5000


# =============================================================================
# API 1: METRICS
# =============================================================================

@router.get(
    "/metrics",
    summary="Get API usage metrics (calls, tokens, cost)",
)
async def get_metrics(request: Request):
    """
    Returns aggregated Gemini API usage metrics persisted in
    data/api_metrics.json.
    """

    metrics_repo = getattr(
        request.app.state,
        "metrics",
        None,
    )

    if metrics_repo is None:
        raise HTTPException(
            status_code=503,
            detail="Metrics repository not initialized.",
        )

    data = await metrics_repo.get_metrics()

    return {
        "status": "success",
        "metrics": data,
    }


# =============================================================================
# API 2: LOGS
# =============================================================================

def _tail_file(
    path: Path,
    n: int,
) -> list[str]:
    """
    Memory-safe tail.

    Reads the last `n` lines from a potentially huge log file without
    loading the entire file into memory.
    """

    if not path.exists():
        return []

    chunk_size = 8192
    lines: list[bytes] = []
    buffer = b""

    with open(path, "rb") as f:
        f.seek(0, 2)

        file_size = f.tell()
        pos = file_size

        while pos > 0 and len(lines) < n + 1:
            read_size = min(
                chunk_size,
                pos,
            )

            pos -= read_size

            f.seek(pos)

            chunk = f.read(read_size)

            buffer = chunk + buffer

            lines = buffer.split(b"\n")

    if len(lines) > n:
        lines = lines[-n:]

    return [
        line.decode(
            "utf-8",
            errors="replace",
        )
        for line in lines
        if line
    ]


# =============================================================================
# LOG PARSER
# =============================================================================

def _parse_log_lines(
    raw_lines: list[str],
) -> list[dict]:
    """
    Parse structured application log lines.

    New log format:

      2026-08-21 10:50:10 |
      INFO |
      src.application.use_cases.scrape_job_url |
      user_id=123 |
      action=scrape |
      Scrape requested for domain 'wellfound.com'

    Result:

      {
          "timestamp": "2026-08-21 10:50:10",
          "level": "INFO",
          "module": "scrape_job_url",
          "user_id": "123",
          "action": "scrape",
          "message": "Scrape requested for domain 'wellfound.com'"
      }

    System logs will contain:

      user_id=- | action=-

    and are converted to:

      user_id=None
      action=None
    """

    header_pattern = re.compile(
        r"^"
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
        r"\s*\|\s*"
        r"([A-Z]+)"
        r"\s*\|\s*"
        r"(.*?)"
        r"\s*\|\s*"
        r"user_id\s*=\s*([^|]*)"
        r"\s*\|\s*"
        r"action\s*=\s*([^|]*)"
        r"\s*\|\s*"
        r"(?:section\s*=\s*([^|]*))?"
        r"\s*\|\s*"
        r"(.*)"
        r"$"
    )

    # Fallback parser for old log entries that do not have the new
    # user_id/action fields.
    old_header_pattern = re.compile(
        r"^"
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
        r"\s*\|\s*"
        r"([A-Z]+)"
        r"\s*\|\s*"
        r"(.*?)"
        r"\s*\|\s*"
        r"(.*)"
        r"$"
    )

    log_objects: list[dict] = []

    for line in raw_lines:
        line = line.rstrip("\r")

        # ---------------------------------------------------------------------
        # New structured format
        # ---------------------------------------------------------------------
        match = header_pattern.match(line)

        if match:
            (
                timestamp,
                level,
                module_path,
                user_id,
                action,
                section,
                message,
            ) = match.groups()

            short_module = (
                module_path.strip().split(".")[-1]
            )

            user_id = user_id.strip()
            action = action.strip()
            section = (section or "-").strip() or "-"
            message = message.strip()

            # "-" means there was no request/user context.
            if user_id in ("", "-"):
                user_id = None

            if action in ("", "-"):
                action = None

            # Internal plumbing lines carry no business section — they stay
            # in the log file for developers but never reach the frontend.
            if section in ("", "-"):
                continue

            # The business section IS the module name the frontend displays.
            short_module = section

            log_objects.append(
                {
                    "timestamp": timestamp.strip(),
                    "level": level.strip(),
                    "module": short_module,
                    "user_id": user_id,
                    "action": action,
                    "message": message,
                }
            )

            continue

        # ---------------------------------------------------------------------
        # Old log format — legacy lines carry no business section, so they
        # are hidden from the frontend view (still present in the file).
        # ---------------------------------------------------------------------
        old_match = old_header_pattern.match(line)

        if old_match:
            continue

        # ---------------------------------------------------------------------
        # Continuation/raw line
        # ---------------------------------------------------------------------
        if log_objects:
            log_objects[-1]["message"] += (
                f"\n{line}"
            )

        else:
            log_objects.append(
                {
                    "timestamp": None,
                    "level": "RAW",
                    "module": None,
                    "user_id": None,
                    "action": None,
                    "message": line,
                }
            )

    return log_objects


# =============================================================================
# GET LOGS
# =============================================================================

@router.get(
    "/logs",
    summary="View recent application logs",
)
async def get_logs(
    lines: int = Query(
        default=_DEFAULT_LINES,
        ge=1,
        le=_MAX_LINES,
        description=(
            f"Number of recent log lines to return "
            f"(default {_DEFAULT_LINES}, "
            f"max {_MAX_LINES})."
        ),
    ),
):
    """
    Returns the last N lines from rag_pipeline.log as structured objects.

    Each log contains:

      timestamp
      level
      module
      user_id
      action
      message
    """

    try:
        raw_lines = _tail_file(
            _LOG_FILE,
            lines,
        )

        log_objects = _parse_log_lines(
            raw_lines
        )

        return {
            "status": "success",
            "returned_lines": len(log_objects),
            "logs": log_objects,
        }

    except Exception as e:
        logger.error(
            f"[Dashboard] Failed to read log file: {e}"
        )

        # No internal details in the response — full error is in the log.
        raise HTTPException(
            status_code=500,
            detail="Could not read the application log file. "
                   "Check server logs for details.",
        )