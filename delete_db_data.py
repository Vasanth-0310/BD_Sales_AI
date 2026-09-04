"""
Clear all Qdrant collection data (points only — structures and configs preserved).

Collections cleared:
  - Summary          (project summaries for Stage 1 hybrid search)
  - Chunks           (project chunks for Stage 2 dense retrieval)
  - profile_variants (developer profile variants for matching)

Usage:
    python delete_db_data.py
"""

import os
from dotenv import load_dotenv
from pathlib import Path

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent / ".env")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

SUMMARY_COLLECTION = os.getenv("QDRANT_SUMMARY_COLLECTION", "Summary")
CHUNKS_COLLECTION = os.getenv("QDRANT_CHUNKS_COLLECTION", "Chunks")
PROFILE_VARIANTS_COLLECTION = os.getenv(
    "QDRANT_PROFILE_VARIANTS_COLLECTION", "profile_variants"
)

COLLECTIONS = [
    SUMMARY_COLLECTION,
    CHUNKS_COLLECTION,
    PROFILE_VARIANTS_COLLECTION,
]


async def clear_collections():
    if not QDRANT_URL or not QDRANT_API_KEY:
        print("ERROR: QDRANT_URL and QDRANT_API_KEY must be set in .env")
        return

    client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    for collection in COLLECTIONS:
        print(f"Clearing '{collection}'...")
        await client.delete(
            collection_name=collection,
            points_selector=Filter(),
        )
        print(f"  Done.")

    await client.close()
    print("All collections cleared. Structures and configs preserved.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(clear_collections())
