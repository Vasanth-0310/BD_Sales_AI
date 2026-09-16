"""Non-destructively assign a shared workspace_id to existing Qdrant points.

This script never deletes, recreates, or re-embeds a point.  It only adds or
overwrites the ``workspace_id`` payload field so read queries can share one
company talent/project library without using the uploader's ``user_id`` as a
ranking or visibility boundary.

Run dry first (the default), then explicitly pass --apply after confirming the
target Qdrant cluster and workspace value:

    python scripts/backfill_qdrant_workspace.py
    python scripts/backfill_qdrant_workspace.py --apply --workspace-id softsuave
"""

import argparse
import sys
from pathlib import Path

from qdrant_client import QdrantClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import settings  # noqa: E402


def update_collection(
    client: QdrantClient,
    collection_name: str,
    workspace_id: str,
    apply: bool,
) -> int:
    offset = None
    count = 0
    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            offset=offset,
            limit=100,
            with_payload=False,
            with_vectors=False,
        )
        if not points:
            break

        ids = [point.id for point in points]
        count += len(ids)
        if apply:
            client.set_payload(
                collection_name=collection_name,
                payload={"workspace_id": workspace_id},
                points=ids,
                wait=True,
            )
        if next_offset is None:
            break
        offset = next_offset
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Qdrant workspace payloads without deleting vectors")
    parser.add_argument("--workspace-id", default=settings.rag_workspace_id)
    parser.add_argument("--apply", action="store_true", help="Perform payload updates; omit for dry run")
    args = parser.parse_args()

    workspace_id = args.workspace_id.strip()
    if not workspace_id:
        raise SystemExit("A non-empty --workspace-id is required.")
    if not settings.qdrant_url:
        raise SystemExit("QDRANT_URL is required.")

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    collections = [
        settings.qdrant_summary_collection,
        settings.qdrant_chunks_collection,
        settings.qdrant_profile_variants_collection,
    ]
    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"[{mode}] workspace_id={workspace_id!r}")
    for collection_name in collections:
        if not collection_name:
            continue
        if not client.collection_exists(collection_name):
            print(f"{collection_name}: skipped; collection does not exist")
            continue
        count = update_collection(client, collection_name, workspace_id, args.apply)
        action = "updated" if args.apply else "would update"
        print(f"{collection_name}: {action} {count} point(s); no vectors deleted")


if __name__ == "__main__":
    main()
