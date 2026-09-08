"""One-time (idempotent) provisioning of Qdrant payload indexes.

Why: filtering on a payload field (user_id) or using MatchText keyword search
requires payload indexes on the target collection — without them Qdrant rejects
the query ("Index required but not found"). Collections were created by
provision_qdrant.py without these indexes.

Indexes created (missing ones only — safe to re-run):
  - summary collection:     user_id (keyword), project_id (keyword),
                            description/project_name/domain/techstacks (text)
  - chunks collection:      user_id (keyword), project_id (keyword)
  - profile_variants:       user_id (keyword), candidate_id (keyword),
                            combined_text/variant_title/tech_stacks_text (text)

Run:  .venv\\Scripts\\python.exe scripts\\create_payload_indexes.py
"""

import sys
sys.path.insert(0, r"C:\Users\Softsuave\Documents\Demo_works")

from qdrant_client import QdrantClient, models

from src.common.config import settings

client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)

PLAN: dict[str, list[tuple[str, models.PayloadSchemaType]]] = {
    settings.qdrant_summary_collection: [
        ("user_id", models.PayloadSchemaType.KEYWORD),
        ("project_id", models.PayloadSchemaType.KEYWORD),
        ("description", models.PayloadSchemaType.TEXT),
        ("project_name", models.PayloadSchemaType.TEXT),
        ("domain", models.PayloadSchemaType.TEXT),
        ("techstacks", models.PayloadSchemaType.TEXT),   # list field — matches array items
    ],
    settings.qdrant_chunks_collection: [
        ("user_id", models.PayloadSchemaType.KEYWORD),
        ("project_id", models.PayloadSchemaType.KEYWORD),
    ],
    settings.qdrant_profile_variants_collection: [
        ("user_id", models.PayloadSchemaType.KEYWORD),
        ("candidate_id", models.PayloadSchemaType.KEYWORD),
        ("combined_text", models.PayloadSchemaType.TEXT),
        ("variant_title", models.PayloadSchemaType.TEXT),
        ("tech_stacks_text", models.PayloadSchemaType.TEXT),
    ],
}

for collection_name, fields in PLAN.items():
    print(f"\n=== {collection_name} ===")
    try:
        info = client.get_collection(collection_name)
    except Exception as e:
        print(f"  SKIP — collection not reachable: {e}")
        continue

    existing = (info.payload_schema or {}).keys()
    for field_name, schema_type in fields:
        if field_name in existing:
            print(f"  {field_name:20s} already indexed — skipping")
            continue
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema_type,
                wait=True,
            )
            print(f"  {field_name:20s} index CREATED ({schema_type.value})")
        except Exception as e:
            msg = str(e)
            if "already exists" in msg.lower():
                print(f"  {field_name:20s} already exists (race) — skipping")
            else:
                print(f"  {field_name:20s} FAILED: {msg[:140]}")

print("\nDone.")
