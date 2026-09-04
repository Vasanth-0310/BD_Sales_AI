"""Service to backup failed ingestions to local JSON files (Dead Letter Queue)."""

import json
import time
from pathlib import Path
from src.common.logger import get_logger

logger = get_logger(__name__)


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

        timestamp = int(time.time())
        filename = f"failed_{timestamp}_{variant_id}.json"
        filepath = out_dir / filename

        data = {
            "timestamp": timestamp,
            "candidate_id": candidate_id,
            "variant_id": variant_id,
            "error": error_msg,
            "payload": payload,
        }

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info("Backed up failed profile to %s", filepath)
        except Exception as e:
            logger.error("CRITICAL: Failed to write DLQ backup for profile: %s", e)

    @classmethod
    def backup_failed_project(
        cls, project_id: str, project_name: str, error_msg: str, **kwargs
    ) -> None:
        """Backup a failed project ingestion."""
        out_dir = cls._BASE_DIR / "projects"
        out_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        filename = f"failed_{timestamp}_{project_id}.json"
        filepath = out_dir / filename

        data = {
            "timestamp": timestamp,
            "project_id": project_id,
            "project_name": project_name,
            "error": error_msg,
            "context": kwargs,
        }

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info("Backed up failed project to %s", filepath)
        except Exception as e:
            logger.error("CRITICAL: Failed to write DLQ backup for project: %s", e)
