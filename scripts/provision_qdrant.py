"""
Qdrant Production Provisioning Script
=====================================
Creates the 3 collections and all payload indices required by the BD
Automation pipeline on a FRESH Qdrant instance (e.g. a new prod cluster).

Collections created (names read from settings/.env):
    - Summary          : project summaries   (1536-dim, cosine)
    - Chunks           : case-study chunks   (1536-dim, cosine)
    - profile_variants : candidate variants  (1536-dim, cosine)

Payload indices:
    Summary:          text  -> description, project_name, domain, techstacks
                      keyword -> project_id
    Chunks:           keyword -> project_id
    profile_variants: text  -> combined_text, variant_title, tech_stacks_text
                      keyword -> candidate_id

Usage:
    # Uses QDRANT_URL / QDRANT_API_KEY from .env by default:
    python scripts/provision_qdrant.py

    # Or point at a specific instance explicitly:
    python scripts/provision_qdrant.py --url https://xxx.cloud.qdrant.io:6333 --api-key XXXX

    # Wipe and recreate collections that already exist (DESTROYS DATA):
    python scripts/provision_qdrant.py --recreate

Idempotent: existing collections are skipped unless --recreate is passed.
"""

import argparse
import sys

from qdrant_client import QdrantClient, models

# Allow running from repo root or from scripts/
sys.path.insert(0, ".")

from src.common.config import settings  # noqa: E402
from src.common.logger import get_logger  # noqa: E402

logger = get_logger("provision_qdrant")

# ══════════════════════════════════════════════════════════════════════════════
# PASTE YOUR PROD CREDENTIALS HERE
# ══════════════════════════════════════════════════════════════════════════════
PROD_QDRANT_URL = "https://dac00d78-d18f-47c6-8765-2d7b524f974a.sa-east-1-0.aws.cloud.qdrant.io"      
PROD_QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6ODk2YjQ2ZjMtYzBjMS00ZGExLTg4ZDEtM2YwYjczZWIxYjkwIn0.LNKme9-_WcqfBXIYCyTFoVijgclEF3Q5-hA2-oa_4vw"  
# ══════════════════════════════════════════════════════════════════════════════
# Just paste the two values above, save, then run:
#     python scripts/provision_qdrant.py
# If left empty, the script falls back to --url/--api-key args or your .env.

VECTOR_SIZE = settings.qdrant_vector_size  # 1536 — gemini-embedding-001 output dim
DISTANCE = models.Distance.COSINE


def provision(client: QdrantClient, recreate: bool) -> None:
    collections_spec = {
        settings.qdrant_summary_collection: {
            "text": ["description", "project_name", "domain", "techstacks"],
            "keyword": ["project_id"],
        },
        settings.qdrant_chunks_collection: {
            "text": [],
            "keyword": ["project_id"],
        },
        settings.qdrant_profile_variants_collection: {
            "text": ["combined_text", "variant_title", "tech_stacks_text"],
            "keyword": ["candidate_id"],
        },
    }

    existing = {c.name for c in client.get_collections().collections}

    friendly = {
        "text": "search index",
        "keyword": "lookup index",
    }

    for name, indices in collections_spec.items():
        if not name:
            raise SystemExit(
                "Collection name missing — set QDRANT_SUMMARY_COLLECTION / "
                "QDRANT_CHUNKS_COLLECTION / QDRANT_PROFILE_VARIANTS_COLLECTION "
                "in .env before provisioning."
            )

        if name in existing:
            if recreate:
                logger.warning(f"Storage area '{name}' already exists — recreating it (existing data will be lost)")
                client.delete_collection(name)
            else:
                logger.info(f"Storage area '{name}' already exists — reusing it")
                _ensure_indices(client, name, indices, friendly)
                continue

        logger.info(
            f"Creating storage area '{name}' for AI search"
        )
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=DISTANCE,
            ),
        )
        _ensure_indices(client, name, indices, friendly)

    logger.info("AI knowledge base setup completed successfully.")
    for c in client.get_collections().collections:
        info = client.get_collection(c.name)
        logger.info(
            f"Storage area '{c.name}' is ready "
            f"(currently holding {info.points_count or 0} records)."
        )


def _ensure_indices(client: QdrantClient, name: str, indices: dict, friendly: dict) -> None:
    """Create any missing payload indices; skip ones already present."""
    try:
        info = client.get_collection(name)
        existing_fields = {
            i.field_name: i.data_type
            for i in (info.payload_schema or {}).values()
            if hasattr(i, "field_name") and hasattr(i, "data_type")
        }
    except Exception as e:
        logger.warning(
            f"[{name}] could not read payload schema ({e}) — will attempt index creation anyway"
        )
        existing_fields = {}

    for field in indices["text"]:
        if existing_fields.get(field) == models.PayloadSchemaType.TEXT:
            continue
        logger.info(f"Adding {friendly['text']} to '{name}' so projects can be found by keywords")
        client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                lowercase=True,
            ),
        )

    for field in indices["keyword"]:
        if existing_fields.get(field) == models.PayloadSchemaType.KEYWORD:
            continue
        logger.info(f"Adding {friendly['keyword']} to '{name}' so records can be managed safely")
        client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Provision Qdrant collections + indices")
    parser.add_argument("--url", default=None, help="Qdrant cluster URL (overrides PROD_QDRANT_URL / .env)")
    parser.add_argument("--api-key", default=None, help="Qdrant API key (overrides PROD_QDRANT_API_KEY / .env)")
    parser.add_argument(
        "--recreate", action="store_true",
        help="Delete + recreate existing collections (DESTROYS ALL DATA)",
    )
    args = parser.parse_args()

    # Priority: CLI args > pasted PROD_* constants at top of this file > .env
    url = args.url or PROD_QDRANT_URL.strip() or settings.qdrant_url
    api_key = args.api_key or PROD_QDRANT_API_KEY.strip() or settings.qdrant_api_key

    if not url:
        raise SystemExit(
            "No Qdrant URL — paste credentials into PROD_QDRANT_URL / "
            "PROD_QDRANT_API_KEY at the top of this script, or pass "
            "--url/--api-key, or set QDRANT_URL in .env"
        )

    logger.info(f"Connecting to {url} ...")
    client = QdrantClient(url=url, api_key=api_key or None)

    # Mark every log line emitted here as a readable System event so it
    # shows up in the frontend AI Logs view.
    from src.common.logger import set_log_context, reset_log_context

    tokens = set_log_context(None, "Knowledge Base Setup", "System")
    try:
        provision(client, recreate=args.recreate)
    finally:
        reset_log_context(*tokens)


if __name__ == "__main__":
    main()
