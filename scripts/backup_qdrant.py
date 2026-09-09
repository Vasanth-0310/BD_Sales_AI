"""
Qdrant Backup / Restore Utility
================================

Portable, file-based backup of ALL points (payload + vectors) from the
three collections into a single timestamped JSON file on your own disk.

ZERO-DELETION: this tool only READS from Qdrant (backup / list modes).
Restore mode only UPSERTS points into collections — it never deletes
anything that already exists in the cluster.

Usage
-----
    python scripts/backup_qdrant.py                    # backup all collections
    python scripts/backup_qdrant.py list               # show snapshots of what's stored
    python scripts/backup_qdrant.py restore <file>     # restore points from a backup file
    python scripts/backup_qdrant.py list-backups       # list backup files on disk

Configuration
-------------
Fill in QDRANT_URL / QDRANT_API_KEY below, OR leave them empty to read
them from your .env file (QDRANT_URL / QDRANT_API_KEY).
Set BACKUP_DIR to wherever you want the backup files saved.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these fields
# ══════════════════════════════════════════════════════════════════════════════

# Paste your cluster credentials here (leave "" to use the .env values):
QDRANT_URL = ""
QDRANT_API_KEY = ""

# Where backup JSON files are saved (created automatically if missing):
BACKUP_DIR = r"C:\Users\Softsuave\Documents\Demo_works\qdrant_backups"

# ══════════════════════════════════════════════════════════════════════════════
# Everything below usually needs no changes
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Windows console: force UTF-8 so no glyph can crash the run
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

from src.common.config import settings  # noqa: E402
from src.common.logger import get_logger  # noqa: E402
from qdrant_client import QdrantClient, models  # noqa: E402

logger = get_logger("QdrantBackup")

# Effective credentials: hardcoded field wins, .env is the fallback
QDRANT_URL = QDRANT_URL or settings.qdrant_url
QDRANT_API_KEY = QDRANT_API_KEY or settings.qdrant_api_key

COLLECTIONS = [
    settings.qdrant_summary_collection,          # e.g. "Summary"
    settings.qdrant_chunks_collection,           # e.g. "Chunks"
    settings.qdrant_profile_variants_collection, # e.g. "profile_variants"
]


def _client() -> QdrantClient:
    if not QDRANT_URL or not QDRANT_API_KEY:
        print("[CONFIG ERROR] QDRANT_URL / QDRANT_API_KEY are empty — "
              "fill them at the top of this script or in your .env.")
        raise SystemExit(1)
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# BACKUP
# ─────────────────────────────────────────────────────────────────────────────

def backup() -> None:
    client = _client()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(BACKUP_DIR) / f"qdrant_backup_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {c.name for c in client.get_collections().collections}
    backup_doc = {
        "created_at": datetime.now().isoformat(),
        "source_cluster": QDRANT_URL,
        "collections": {},
    }

    total_points = 0
    for name in COLLECTIONS:
        if not name:
            continue
        if name not in existing:
            print(f"  {name:30s} SKIP (collection does not exist in cluster)")
            backup_doc["collections"][name] = {
                "exists": False, "points": [], "vector_size": None,
            }
            continue

        info = client.get_collection(name)
        vector_size = None
        try:
            vector_size = info.config.params.vectors.size
        except Exception:
            vector_size = settings.qdrant_vector_size

        points = []
        offset = None
        while True:
            batch, next_offset = client.scroll(
                collection_name=name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for p in batch:
                points.append({
                    "id": str(p.id),
                    "vector": p.vector if isinstance(p.vector, list) else (p.vector or {}),
                    "payload": p.payload or {},
                })
            if next_offset is None:
                break
            offset = next_offset

        backup_doc["collections"][name] = {
            "exists": True,
            "vector_size": vector_size,
            "points": points,
        }
        total_points += len(points)
        print(f"  {name:30s} {len(points):6d} point(s) backed up")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(backup_doc, f, ensure_ascii=False, indent=1)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print("-" * 60)
    print(f"[OK] Backup saved: {out_path}")
    print(f"     {total_points} point(s) across "
          f"{sum(1 for c in backup_doc['collections'].values() if c['exists'])} collection(s), "
          f"{size_mb:.2f} MB")


# ─────────────────────────────────────────────────────────────────────────────
# RESTORE  (upsert-only — never deletes)
# ─────────────────────────────────────────────────────────────────────────────

def restore(backup_file: str) -> None:
    client = _client()
    with open(backup_file, "r", encoding="utf-8") as f:
        doc = json.load(f)

    collections = doc.get("collections", {})
    if not collections:
        print("[X] Backup file contains no collections.")
        raise SystemExit(1)

    print(f"Restoring from: {backup_file}")
    print("Mode: UPSERT-ONLY — existing points with the same ID are overwritten, "
          "nothing is deleted.")
    confirm = input("Type 'yes' to continue: ").strip().lower()
    if confirm != "yes":
        print("Aborted — nothing was written.")
        return

    total = 0
    for name, data in collections.items():
        if not data.get("exists") or not data.get("points"):
            print(f"  {name:30s} SKIP (empty in backup)")
            continue

        # Create the collection if it doesn't exist in the target cluster
        existing = {c.name for c in client.get_collections().collections}
        if name not in existing:
            vector_size = data.get("vector_size") or settings.qdrant_vector_size
            client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            print(f"  {name:30s} collection created (size={vector_size})")

        points = [
            models.PointStruct(
                id=p["id"],
                vector=p["vector"],
                payload=p.get("payload") or {},
            )
            for p in data["points"]
            if p.get("vector")
        ]

        restored = 0
        for i in range(0, len(points), 200):
            client.upsert(collection_name=name, points=points[i:i + 200], wait=True)
            restored += len(points[i:i + 200])

        total += restored
        print(f"  {name:30s} {restored:6d} point(s) restored")

    print("-" * 60)
    print(f"[OK] Restore complete: {total} point(s) upserted.")


# ─────────────────────────────────────────────────────────────────────────────
# LIST HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def list_backups() -> None:
    d = Path(BACKUP_DIR)
    if not d.is_dir():
        print(f"No backup directory at: {d}")
        return
    files = sorted(d.glob("qdrant_backup_*.json"))
    if not files:
        print(f"No backup files in: {d}")
        return
    print(f"Backup files in {d}:")
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}  ({size_mb:.2f} MB)")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "backup"

    if cmd == "backup":
        backup()
    elif cmd == "restore":
        if len(sys.argv) < 3:
            print("Usage: python scripts/backup_qdrant.py restore <backup_file.json>")
            raise SystemExit(1)
        restore(sys.argv[2])
    elif cmd == "list-backups":
        list_backups()
    else:
        print(__doc__)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
