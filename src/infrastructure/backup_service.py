"""Service to backup failed ingestions to local JSON files (Dead Letter Queue).

Writes are ATOMIC (tmp file + os.replace) and collision-proof (nanosecond
timestamp + uuid suffix + sanitized IDs), so a crash mid-write or two
failures in the same second can never corrupt or overwrite DLQ entries.
"""

import json
import os
import re
import time
import uuid
from pathlib import Path
from src.common.logger import get_logger

logger = get_logger(__name__)


def _sanitize_filename(value: str) -> str:
    """Strip characters Windows filenames cannot contain (IDs may be raw)."""
    return re.sub(r"[^A-Za-z0-9_\-]", "_", value)[:80]


def _atomic_write_json(filepath: Path, data: dict) -> None:
    tmp = filepath.with_name(filepath.name + f".tmp{uuid.uuid4().hex[:8]}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, filepath)  # atomic on Windows/NTFS and POSIX


class BackupService:
    """Handles writing failed Qdrant payloads to the local filesystem."""

    _BASE_DIR = Path("src/data/failed_ingestions")

    @classmethod
    def backup_failed_profile(
        cls, candidate_id: str, variant_id: str, payload: dict, error_msg: str
    ) -> None:
        """Backup a failed profile variant ingestion."""
        out_dir = cls._BASE_DIR / "profiles"
        out_dir.mkdir(parents=True, exist_ok=True)

        stamp = f"{time.time_ns()}_{uuid.uuid4().hex[:8]}"
        filename = f"failed_{stamp}_{_sanitize_filename(variant_id)}.json"
        filepath = out_dir / filename

        data = {
            "timestamp": time.time_ns(),
            "candidate_id": candidate_id,
            "variant_id": variant_id,
            "error": error_msg,
            "payload": payload,
        }

        try:
            _atomic_write_json(filepath, data)
            logger.info("Backed up failed profile to %s", filepath)
        except Exception as e:
            logger.error("CRITICAL: Failed to write DLQ backup for profile: %s", e)

    @classmethod
    def backup_failed_project(
        cls,
        project_id: str,
        project_name: str,
        error_msg: str,
        *,
        orphaned_chunks: list[dict] | None = None,
        **kwargs,
    ) -> None:
        """Backup a failed project ingestion.

        orphaned_chunks: full chunk records (id/vector/payload) captured
        BEFORE the pre-ingest delete ran, so a failed re-ingest is fully
        recoverable from disk instead of losing the project forever.
        """
        out_dir = cls._BASE_DIR / "projects"
        out_dir.mkdir(parents=True, exist_ok=True)

        stamp = f"{time.time_ns()}_{uuid.uuid4().hex[:8]}"
        filename = f"failed_{stamp}_{_sanitize_filename(project_id)}.json"
        filepath = out_dir / filename

        data = {
            "timestamp": time.time_ns(),
            "project_id": project_id,
            "project_name": project_name,
            "error": error_msg,
            "orphaned_chunks": orphaned_chunks or [],
            "context": kwargs,
        }

        try:
            _atomic_write_json(filepath, data)
            logger.info("Backed up failed project to %s", filepath)
        except Exception as e:
            logger.error("CRITICAL: Failed to write DLQ backup for project: %s", e)
