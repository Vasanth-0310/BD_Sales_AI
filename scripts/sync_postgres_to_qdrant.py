"""
Differential Sync Utility: PostgreSQL (EC2) -> Qdrant (RAG Vector DB)

Compares PostgreSQL and Qdrant, and ONLY ingests data that is MISSING
from Qdrant. Never deletes or overwrites existing data.

Features:
- Standalone configuration: Uses inline credentials, bypassing .env
- Sync toggles: Control project and profile sync individually via booleans
- Differential sync: compares IDs between PostgreSQL and Qdrant
- Case study support: reads PDF/DOCX files from case_studies/ folder by project ID
- Safe: never deletes data, fail-safe per item, non-zero exit on failure

Usage:
    python scripts/sync_postgres_to_qdrant.py
    python scripts/sync_postgres_to_qdrant.py --dry-run
"""

import asyncio
import json
import os
import re
import sys
import time

# Windows consoles default to cp1252 and CRASH on any non-ASCII glyph
# (UnicodeEncodeError) - force UTF-8 output before anything prints.
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
from contextlib import closing
from urllib.parse import urlparse

# =====================================================================
# SYNC CONTROLS
# =====================================================================
SYNC_PROJECTS = True   # Set to False to skip project ingestion
SYNC_PROFILES = True  # Set to False to skip profile ingestion

# =====================================================================
# INLINE CREDENTIALS & CONFIGURATION
# =====================================================================
# Fill in your database URLs and API keys here. 
# This script will NOT read from the .env file.
SYNC_CONFIG = {
    "POSTGRES_URL": "",
    "QDRANT_URL": "",
    "QDRANT_API_KEY": "",
    "GEMINI_API_KEY": "",
    
    # Target Collections
    "QDRANT_SUMMARY_COLLECTION": "Summary",
    "QDRANT_CHUNKS_COLLECTION": "Chunks",
    "QDRANT_PROFILE_VARIANTS_COLLECTION": "profile_variants",
}

# ---------------------------------------------------------------------
# ENVIRONMENT INJECTION
# We inject these values into the system environment BEFORE importing
# any 'src' modules so that the backend configuration automatically 
# uses these inline credentials instead of looking for a .env file.
# ---------------------------------------------------------------------
for key, value in SYNC_CONFIG.items():
    if value:
        # setdefault: a value coming from the real environment / .env
        # must win — silent overrides at import time are a footgun.
        os.environ.setdefault(key, value)

# Ensure demo_works root is on sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# Guard psycopg2 import
try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("\n[ERROR] 'psycopg2' is not installed.")
    print("Please install it in your virtual environment by running:")
    print("    pip install psycopg2-binary\n")
    raise SystemExit(1)

# Now we can safely import internal application code
from src.application.dto.profile_dto import (
    IngestProfileDTO,
    VariantDTO,
    ProjectInVariantDTO,
)
from src.application.dto.project_dto import IngestProjectDTO
from src.application.use_cases.ingest_profile import IngestProfileUseCase
from src.application.use_cases.ingest_project import IngestProjectUseCase
from src.common.config import settings
from src.common.logger import get_logger
from src.infrastructure.ai.gemini_embedding_adapter import GeminiEmbeddingAdapter
from src.infrastructure.db.qdrant.qdrant_vector_store_adapter import (
    QdrantVectorStoreAdapter,
)
from src.infrastructure.db.qdrant.semantic_chunker import SemanticChunker

logger = get_logger("SyncPostgresQdrant")

# ─── Case Study Support ────────────────────────────────────────────────
CASE_STUDY_DIR = os.path.join(BASE_DIR, "case_studies")


def find_case_study_file(project_id: str) -> tuple[bytes, str] | None:
    """Look for a case study file by project ID (e.g., {uuid}.pdf or {uuid}.docx)."""
    if not os.path.isdir(CASE_STUDY_DIR):
        return None
    for ext in (".pdf", ".docx"):
        filepath = os.path.join(CASE_STUDY_DIR, f"{project_id}{ext}")
        if os.path.isfile(filepath):
            with open(filepath, "rb") as f:
                return f.read(), f"{project_id}{ext}"
    return None


# ─── Utility Functions ──────────────────────────────────────────────────


def parse_experience_years(exp_str: str) -> int:
    """Extract integer years from strings like '3-5 years', '4', '2.5'."""
    if not exp_str:
        return 0
    s = str(exp_str)
    range_match = re.search(r"(\d+)\s*[--]\s*(\d+)", s)
    if range_match:
        lo, hi = int(range_match.group(1)), int(range_match.group(2))
        return (lo + hi) // 2
    float_match = re.search(r"\d+(?:\.\d+)?", s)
    if float_match:
        return round(float(float_match.group(0)))
    return 0


def _as_str_list(value) -> list[str]:
    """Normalize a Postgres array OR a comma-separated string into a clean list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, dict):
                name = item.get("name") or item.get("title") or ""
                if name:
                    out.append(str(name).strip())
                continue
            if isinstance(item, str):
                item = item.strip()
                if item:
                    out.append(item)
            else:
                out.append(str(item))
        return out
    return [str(value)]


def _safe_links(raw_links) -> dict:
    """Safely extract a dict from a links field."""
    if isinstance(raw_links, dict):
        return raw_links
    if isinstance(raw_links, str) and raw_links.strip():
        try:
            parsed = json.loads(raw_links)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def mask_db_url(url: str) -> str:
    """Mask database password in the connection URL for secure logging."""
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return "<configured database url>"
        netloc = parsed.netloc
        if "@" in netloc:
            creds, host = netloc.split("@", 1)
            if ":" in creds:
                user = creds.split(":")[0]
                netloc = f"{user}:****@{host}"
            else:
                netloc = f"{creds}@{host}"
        return f"{parsed.scheme}://{netloc}{parsed.path}"
    except Exception:
        return "<configured database url>"


def _effective_config(key: str) -> str:
    """SYNC_CONFIG value if set, else the .env/settings value."""
    return SYNC_CONFIG.get(key) or getattr(settings, key, "") or ""


def _validate_startup_config() -> str:
    """Validate all required settings.

    SYNC_CONFIG is an OPTIONAL overlay: any empty value falls back to the
    .env / settings layer (QDRANT_URL, GEMINI_API_KEY, POSTGRES_URL, ...).
    Requiring the inline dict to be filled would make the script unusable
    for .env-based setups - and, worse, let a stale inline value override
    fresher .env credentials.
    """
    errors: list[str] = []

    required = (
        "POSTGRES_URL",
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "GEMINI_API_KEY",
        "QDRANT_SUMMARY_COLLECTION",
        "QDRANT_CHUNKS_COLLECTION",
        "QDRANT_PROFILE_VARIANTS_COLLECTION",
    )
    for key in required:
        if not _effective_config(key):
            errors.append(f"'{key}' is empty (set it in SYNC_CONFIG or .env)")

    if errors:
        print("\n[CONFIG ERROR] The following required settings are missing:")
        for e in errors:
            print(f"  - {e}")
        print("\nSet them in the SYNC_CONFIG dict at the top of the script "
              "OR in your .env file.")
        raise SystemExit(1)

    return _effective_config("POSTGRES_URL")


# ─── Main Syncer Class ──────────────────────────────────────────────────


class PostgresToQdrantSyncer:
    def __init__(self, pg_url: str):
        # The .env URL is a SQLAlchemy DSN ("postgresql+asyncpg://...").
        # psycopg2 only understands the plain scheme - strip any driver
        # suffix automatically so either form works.
        self.pg_url = re.sub(r"^postgresql\+[A-Za-z0-9_]+://", "postgresql://", pg_url.strip())
        self._masked_url = mask_db_url(pg_url)
        logger.info("[INIT] Initializing Qdrant and AI infrastructure adapters...")
        self.vector_store = QdrantVectorStoreAdapter()
        self.embedding_port = GeminiEmbeddingAdapter()
        self.chunker = SemanticChunker()
        self.profile_use_case = IngestProfileUseCase(
            embedding_port=self.embedding_port,
            vector_store_port=self.vector_store,
        )
        self.project_use_case = IngestProjectUseCase(
            embedding_port=self.embedding_port,
            vector_store_port=self.vector_store,
            chunker=self.chunker,
        )
        logger.info("[INIT] Target Qdrant Collections:")
        logger.info(f"       - Projects Summary  : {settings.qdrant_summary_collection}")
        logger.info(f"       - Projects Chunks   : {settings.qdrant_chunks_collection}")
        logger.info(f"       - Profile Variants  : {settings.qdrant_profile_variants_collection}")

    # ─── Database Helper ─────────────────────────────────────────────

    async def _fetch_from_pg(self, query: str, params=None) -> list[dict]:
        """Execute a SELECT query against PostgreSQL and return rows."""
        loop = asyncio.get_running_loop()

        def _sync_fetch():
            with closing(
                psycopg2.connect(self.pg_url, cursor_factory=psycopg2.extras.RealDictCursor)
            ) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    return cur.fetchall()

        return await loop.run_in_executor(None, _sync_fetch)

    # ─── Phase 1: Differential Project Sync ──────────────────────────

    async def sync_projects(self, dry_run: bool = False) -> bool:
        start_time = time.perf_counter()
        logger.info("\n" + "=" * 70)
        logger.info("  PHASE 1: DIFFERENTIAL PROJECT SYNC")
        logger.info("=" * 70)

        logger.info("[PROJECTS: STEP 1] Fetching published project IDs from PostgreSQL...")
        try:
            db_start = time.perf_counter()
            id_rows = await self._fetch_from_pg(
                "SELECT project_id::text FROM projects WHERE is_draft = false;"
            )
            pg_project_ids = {row["project_id"] for row in id_rows}
            logger.info(
                f"[PROJECTS: STEP 1] Found {len(pg_project_ids)} published project(s) "
                f"in PostgreSQL ({time.perf_counter() - db_start:.2f}s)."
            )
        except Exception as e:
            logger.error(
                f"[PROJECTS: ERROR] Failed to fetch project IDs: {mask_db_url(str(e))}",
                exc_info=True,
            )
            return False

        if not pg_project_ids:
            logger.info("[PROJECTS] No published projects found in PostgreSQL.")
            return True

        logger.info("[PROJECTS: STEP 2] Scrolling existing project IDs from Qdrant Summary collection...")
        try:
            qdrant_project_ids = await self.vector_store.scroll_all_point_ids(
                settings.qdrant_summary_collection
            )
            logger.info(
                f"[PROJECTS: STEP 2] Found {len(qdrant_project_ids)} project(s) already in Qdrant."
            )
        except Exception as e:
            logger.error(
                f"[PROJECTS: ERROR] Failed to scroll Qdrant Summary collection: {e}",
                exc_info=True,
            )
            return False

        missing_ids = pg_project_ids - qdrant_project_ids
        logger.info(
            f"[PROJECTS: STEP 3] Diff: {len(pg_project_ids)} in PG, "
            f"{len(qdrant_project_ids)} in Qdrant, {len(missing_ids)} missing"
        )

        if not missing_ids:
            logger.info("[PROJECTS] [OK] All projects already synced. Nothing to do.")
            elapsed = time.perf_counter() - start_time
            logger.info(f"[PROJECTS] Phase 1 completed in {elapsed:.2f}s")
            return True

        if dry_run:
            logger.info(
                f"[PROJECTS: DRY RUN] {len(missing_ids)} project(s) would be synced. "
                f"Checking case studies..."
            )

            # Fetch project names for all missing IDs (lightweight query)
            name_query = """
                SELECT project_id::text, project_name
                FROM projects
                WHERE is_draft = false
                  AND project_id::text = ANY(%s);
            """
            try:
                name_rows = await self._fetch_from_pg(name_query, (list(missing_ids),))
                name_map = {r["project_id"]: r.get("project_name") or "Unnamed" for r in name_rows}
            except Exception:
                # If name fetch fails, fall back to IDs only
                name_map = {}

            with_case_study = 0
            without_case_study = 0

            logger.info("-" * 60)
            for idx, project_id in enumerate(sorted(missing_ids), 1):
                project_name = name_map.get(project_id, f"Project-{project_id[:8]}")
                case_study = find_case_study_file(project_id)

                if case_study:
                    _, filename = case_study
                    # Get file size from disk for display
                    filepath = os.path.join(CASE_STUDY_DIR, filename)
                    try:
                        size_kb = os.path.getsize(filepath) / 1024
                        size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:.1f} MB"
                    except OSError:
                        size_str = "unknown size"
                    logger.info(
                        f"  [{idx}/{len(missing_ids)}]  {project_name[:40]:<40}  "
                        f"({project_id[:8]})"
                    )
                    logger.info(f"         Case Study: [FOUND]   {filename}  ({size_str})")
                    with_case_study += 1
                else:
                    logger.info(
                        f"  [{idx}/{len(missing_ids)}]  {project_name[:40]:<40}  "
                        f"({project_id[:8]})"
                    )
                    logger.info(f"         Case Study: [MISSING]  will use description as fallback")
                    without_case_study += 1

            logger.info("-" * 60)
            logger.info(
                f"[PROJECTS: DRY RUN] Summary: "
                f"{with_case_study} with case study file, "
                f"{without_case_study} will use description fallback."
            )
            logger.info("[PROJECTS: DRY RUN] No data was written. Run without --dry-run to ingest.")
            return True

        logger.info(f"[PROJECTS: STEP 4] Fetching full data for {len(missing_ids)} missing project(s)...")
        full_query = """
            SELECT
                p.project_id::text,
                p.project_name,
                p.description,
                p.links,
                COALESCE(pd."domain", 'General') as domain_name,
                COALESCE(
                    ARRAY_AGG(ts."techstack_name") FILTER (WHERE ts.techstack_id IS NOT NULL AND ts."techstack_name" IS NOT NULL),
                    ARRAY[]::text[]
                ) as techstacks
            FROM projects p
            LEFT JOIN project_domains pd ON p."projectDomainID" = pd.id
            LEFT JOIN project_techstacks pts ON p.project_id = pts.project_id
            LEFT JOIN tech_stacks ts ON pts.techstack_id = ts.techstack_id
            WHERE p.is_draft = false
              AND p.project_id::text = ANY(%s)
            GROUP BY p.project_id, p.project_name, p.description, p.links, pd.id, pd."domain";
        """
        try:
            rows = await self._fetch_from_pg(full_query, (list(missing_ids),))
            logger.info(f"[PROJECTS: STEP 4] Retrieved {len(rows)} project row(s) for ingestion.")
        except Exception as e:
            logger.error(
                f"[PROJECTS: ERROR] Failed to fetch project data: {mask_db_url(str(e))}",
                exc_info=True,
            )
            return False

        logger.info(f"[PROJECTS: STEP 5] Ingesting {len(rows)} missing project(s)...")
        succeeded = 0
        failed = 0

        for idx, row in enumerate(rows, 1):
            project_id = row["project_id"]
            project_name = row.get("project_name") or f"Project-{project_id[:8]}"
            domain = row.get("domain_name") or "General"
            description = row.get("description") or ""
            techstacks = [t for t in (row.get("techstacks") or []) if t]
            links = _safe_links(row.get("links"))

            logger.info("-" * 60)
            logger.info(f"[{idx}/{len(rows)}] Processing Project: '{project_name}'")
            logger.info(f"      ID         : {project_id}")
            logger.info(f"      Domain     : {domain}")
            logger.info(f"      Tech Stack : {', '.join(techstacks) if techstacks else 'None specified'}")

            case_study = find_case_study_file(project_id)
            if case_study:
                file_bytes, filename = case_study
                logger.info(f"      📄 Case study found: {filename} ({len(file_bytes):,} bytes)")
            else:
                file_bytes = description.encode("utf-8")
                filename = f"{project_name}.txt"
                logger.info(f"      ℹ️  No case study file - using description as fallback ({len(description)} chars)")

            dto = IngestProjectDTO(
                project_id=project_id,
                user_id=TENANT_USER_ID or None,
                project_name=project_name,
                domain=domain,
                techstacks=techstacks,
                description=description,
                links=links,
            )

            item_start = time.perf_counter()
            try:
                logger.info("      -> Chunking & embedding via Gemini...")
                result = await self.project_use_case.execute(
                    dto=dto,
                    file_bytes=file_bytes,
                    filename=filename,
                )
                item_elapsed = time.perf_counter() - item_start
                chunks_stored = result.get("chunks_stored", 0)
                logger.info(
                    f"      [OK] Project '{project_name}' indexed "
                    f"({chunks_stored} chunks) in {item_elapsed:.2f}s"
                )
                succeeded += 1

                delay = max(1.0, chunks_stored * 0.3)
                logger.debug(f"      -> Rate-limit cooldown: sleeping {delay:.1f}s...")
                await asyncio.sleep(delay)

            except Exception as e:
                item_elapsed = time.perf_counter() - item_start
                logger.error(
                    f"      [X] Failed to index project '{project_name}' "
                    f"after {item_elapsed:.2f}s: {e}",
                    exc_info=True,
                )
                failed += 1
                await asyncio.sleep(1.0)

        total_elapsed = time.perf_counter() - start_time
        logger.info("=" * 70)
        logger.info(f"  PHASE 1 COMPLETE: Projects Synced in {total_elapsed:.2f}s")
        logger.info(
            f"  Summary: {succeeded} Succeeded, {failed} Failed, {len(missing_ids)} Total Missing"
        )
        logger.info("=" * 70)
        return failed == 0

    # ─── Phase 2: Differential Profile Variant Sync ──────────────────

    async def sync_profiles(self, dry_run: bool = False) -> bool:
        start_time = time.perf_counter()
        logger.info("\n" + "=" * 70)
        logger.info("  PHASE 2: DIFFERENTIAL PROFILE VARIANT SYNC")
        logger.info("=" * 70)

        logger.info("[PROFILES: STEP 1] Fetching published variant IDs from PostgreSQL...")
        try:
            db_start = time.perf_counter()
            id_rows = await self._fetch_from_pg(
                "SELECT profile_variant_id::text as variant_id "
                "FROM profile_variants WHERE is_draft = false;"
            )
            pg_variant_ids = {row["variant_id"] for row in id_rows}
            logger.info(
                f"[PROFILES: STEP 1] Found {len(pg_variant_ids)} published variant(s) "
                f"in PostgreSQL ({time.perf_counter() - db_start:.2f}s)."
            )
        except Exception as e:
            logger.error(
                f"[PROFILES: ERROR] Failed to fetch variant IDs: {mask_db_url(str(e))}",
                exc_info=True,
            )
            return False

        if not pg_variant_ids:
            logger.info("[PROFILES] No published profile variants found in PostgreSQL.")
            return True

        logger.info("[PROFILES: STEP 2] Scrolling existing variant IDs from Qdrant profile_variants...")
        try:
            qdrant_variant_ids = await self.vector_store.scroll_all_point_ids(
                settings.qdrant_profile_variants_collection
            )
            logger.info(
                f"[PROFILES: STEP 2] Found {len(qdrant_variant_ids)} variant(s) already in Qdrant."
            )
        except Exception as e:
            logger.error(
                f"[PROFILES: ERROR] Failed to scroll Qdrant profile_variants: {e}",
                exc_info=True,
            )
            return False

        missing_variant_ids = pg_variant_ids - qdrant_variant_ids
        logger.info(
            f"[PROFILES: STEP 3] Diff: {len(pg_variant_ids)} in PG, "
            f"{len(qdrant_variant_ids)} in Qdrant, {len(missing_variant_ids)} missing"
        )

        if not missing_variant_ids:
            logger.info("[PROFILES] [OK] All profile variants already synced. Nothing to do.")
            elapsed = time.perf_counter() - start_time
            logger.info(f"[PROFILES] Phase 2 completed in {elapsed:.2f}s")
            return True

        if dry_run:
            logger.info(
                f"[PROFILES: DRY RUN] Would sync {len(missing_variant_ids)} variant(s). "
                f"Skipping actual ingestion."
            )
            return True

        logger.info(
            f"[PROFILES: STEP 4] Fetching full data for {len(missing_variant_ids)} missing variant(s)..."
        )
        variants_query = """
            SELECT
                pv.profile_variant_id::text as variant_id,
                pv.name as variant_title,
                pv.experience,
                pv.highlighted_skills,
                pv.certificate,
                COALESCE(jr."roleName", 'Developer') as role_name,
                u.user_id::text as candidate_id,
                u.email,
                COALESCE(
                    NULLIF(TRIM(CONCAT(COALESCE(upi.first_name, ''), ' ', COALESCE(upi.last_name, ''))), ''),
                    'Candidate'
                ) as candidate_name,
                COALESCE(us."displayName", 'On Bench') as resource_status,
                CASE
                    WHEN upi.highest_qualification IS NULL AND upi.specialization IS NULL THEN ''
                    WHEN upi.specialization IS NULL THEN COALESCE(upi.highest_qualification, '')
                    ELSE CONCAT(COALESCE(upi.highest_qualification, ''), ' (', COALESCE(upi.specialization, ''), ')')
                END as education,
                COALESCE(upi.year_of_passout, 2024) as passout_year,
                COALESCE(upi.date_of_birth::text, '') as dob,
                COALESCE(b.name, 'Main') as branch
            FROM profile_variants pv
            JOIN users u ON pv.user_id = u.user_id
            LEFT JOIN user_personal_info upi ON u.user_id = upi.user_id
            LEFT JOIN user_status us ON upi.working_status_id = us.id
            LEFT JOIN branches b ON upi.branch_id = b.id
            LEFT JOIN job_roles jr ON pv.role = jr.id
            WHERE pv.is_draft = false
              AND pv.profile_variant_id::text = ANY(%s);
        """

        projects_query = """
            SELECT
                pvp.profile_variant_id::text,
                pvp.project_id::text,
                pvp.project_name,
                pvp.techstacks,
                pvp.description,
                pvp.links,
                COALESCE(pd."domain", 'General') as domain_name
            FROM profile_variant_projects pvp
            JOIN profile_variants pv ON pvp.profile_variant_id = pv.profile_variant_id
            LEFT JOIN project_domains pd ON pvp."projectDomainID" = pd.id
            WHERE pv.is_draft = false
              AND pvp.profile_variant_id::text = ANY(%s);
        """

        missing_list = list(missing_variant_ids)
        try:
            db_start = time.perf_counter()
            variant_rows = await self._fetch_from_pg(variants_query, (missing_list,))
            proj_rows = await self._fetch_from_pg(projects_query, (missing_list,))
            logger.info(
                f"[PROFILES: STEP 4] Retrieved {len(variant_rows)} variant(s) and "
                f"{len(proj_rows)} project mapping(s) in {time.perf_counter() - db_start:.2f}s."
            )
        except Exception as e:
            logger.error(
                f"[PROFILES: ERROR] Failed to fetch profile data: {mask_db_url(str(e))}",
                exc_info=True,
            )
            return False

        if not variant_rows:
            logger.info("[PROFILES] No variant rows returned for missing IDs (possible data inconsistency).")
            return True

        logger.info("[PROFILES: STEP 5] Mapping project details to variants...")
        variant_projects_map: dict[str, list[ProjectInVariantDTO]] = {}
        for p in proj_rows:
            vid = p["profile_variant_id"]
            if vid not in variant_projects_map:
                variant_projects_map[vid] = []
            variant_projects_map[vid].append(
                ProjectInVariantDTO(
                    project_id=p["project_id"],
                    project_name=p.get("project_name") or f"Project-{p['project_id'][:8]}",
                    domain=p.get("domain_name") or "General",
                    tech_stack=_as_str_list(p.get("techstacks")),
                    links=_safe_links(p.get("links")),
                    description=p.get("description") or "",
                )
            )

        logger.info("[PROFILES: STEP 6] Grouping missing variants by candidate...")
        candidates_map: dict[str, dict] = {}
        for v in variant_rows:
            cid = v["candidate_id"]
            if cid not in candidates_map:
                raw_name = (v.get("candidate_name") or "").strip() or "Candidate"
                candidates_map[cid] = {
                    "candidate_id": cid,
                    "candidate_name": raw_name,
                    "resource_status": v["resource_status"],
                    "email": v.get("email") or "",
                    "education": v.get("education") or "",
                    "passout_year": v.get("passout_year") or 2024,
                    "dob": v.get("dob") or "",
                    "branch": v.get("branch") or "Main",
                    "variants": [],
                }

            vid = v["variant_id"]
            projs = variant_projects_map.get(vid, [])
            candidates_map[cid]["variants"].append(
                VariantDTO(
                    variant_id=vid,
                    variant_title=v.get("variant_title") or "Profile Variant",
                    role=v["role_name"],
                    experience_years=parse_experience_years(v.get("experience")),
                    no_of_projects=len(projs),
                    tech_stacks=_as_str_list(v.get("highlighted_skills")),
                    certifications=_as_str_list(v.get("certificate")),
                    projects=projs,
                )
            )

        logger.info(
            f"[PROFILES: STEP 6] Organized into {len(candidates_map)} candidate(s) "
            f"representing {len(missing_variant_ids)} missing variant(s)."
        )

        logger.info(
            f"[PROFILES: STEP 7] Ingesting candidates into Qdrant "
            f"'{settings.qdrant_profile_variants_collection}'..."
        )
        succeeded = 0
        failed = 0

        for idx, (cid, cdata) in enumerate(candidates_map.items(), 1):
            dto = IngestProfileDTO(
                candidate_id=cdata["candidate_id"],
                candidate_name=cdata["candidate_name"],
                resource_status=cdata["resource_status"],
                email=cdata["email"],
                education=cdata["education"],
                passout_year=cdata["passout_year"],
                dob=cdata["dob"],
                branch=cdata["branch"],
                user_id=TENANT_USER_ID or None,
                # ZERO-DELETION: this is a PARTIAL payload (only the variants
                # missing from Qdrant). Reconciliation would delete the
                # candidate's existing variants - must stay off for sync.
                reconcile_variants=False,
                variants=cdata["variants"],
            )

            anon_name = f"{dto.candidate_name[:1]}***" if dto.candidate_name else "***"
            logger.info("-" * 60)
            logger.info(f"[{idx}/{len(candidates_map)}] Ingesting Candidate: '{anon_name}'")
            logger.info(f"      Candidate ID   : {dto.candidate_id}")
            logger.info(f"      Variants Count : {len(dto.variants)}")

            cand_start = time.perf_counter()
            try:
                logger.info(
                    f"      -> Batch-embedding {len(dto.variants)} variant summary text(s) via Gemini..."
                )
                result = await self.profile_use_case.execute(dto)

                if hasattr(result, "status") and result.status == "FAILED":
                    error_msg = getattr(result, "error_message", "Unknown error")
                    raise RuntimeError(f"Profile ingestion returned FAILED: {error_msg}")

                cand_elapsed = time.perf_counter() - cand_start
                logger.info(
                    f"      [OK] Synced candidate '{anon_name}' "
                    f"({len(dto.variants)} variants) in {cand_elapsed:.2f}s"
                )
                succeeded += 1

            except Exception as e:
                cand_elapsed = time.perf_counter() - cand_start
                logger.error(
                    f"      [X] Failed to sync candidate '{anon_name}' "
                    f"after {cand_elapsed:.2f}s: {e}",
                    exc_info=True,
                )
                failed += 1

            delay = max(1.0, len(dto.variants) * 0.5)
            logger.debug(f"      -> Rate-limit cooldown: sleeping {delay:.1f}s...")
            await asyncio.sleep(delay)

        total_elapsed = time.perf_counter() - start_time
        logger.info("=" * 70)
        logger.info(f"  PHASE 2 COMPLETE: Profiles Synced in {total_elapsed:.2f}s")
        logger.info(
            f"  Summary: {succeeded} Succeeded, {failed} Failed, "
            f"{len(candidates_map)} Total Candidates"
        )
        logger.info("=" * 70)
        return failed == 0


# ─── Entry Point ─────────────────────────────────────────────────────────

TENANT_USER_ID = ""  # Set via --tenant-user-id

async def main():
    global TENANT_USER_ID
    print("\n" + "=" * 70)
    print("  POSTGRESQL -> QDRANT DIFFERENTIAL SYNC ENGINE")
    print("=" * 70)

    # Validate inline config
    pg_url = _validate_startup_config()
    masked_url = mask_db_url(pg_url)
    logger.info(f"[CONFIG] Target PostgreSQL: {masked_url}")

    for arg in sys.argv[1:]:
        if arg.startswith("--tenant-user-id="):
            TENANT_USER_ID = arg.split("=", 1)[1].strip()
            break
    else:
        if "--tenant-user-id" in sys.argv:
            i = sys.argv.index("--tenant-user-id")
            if i + 1 < len(sys.argv):
                TENANT_USER_ID = sys.argv[i + 1].strip()
                
    if not TENANT_USER_ID:
        logger.warning(
            "[CONFIG] --tenant-user-id not provided - synced points will be "
            "stored WITHOUT an owner and will be visible to ALL tenants."
        )

    dry_run = "--dry-run" in sys.argv
    if dry_run:
        logger.info("[MODE] DRY RUN - will compare databases but not ingest any data")

    if os.path.isdir(CASE_STUDY_DIR):
        cs_files = [f for f in os.listdir(CASE_STUDY_DIR) if f.endswith((".pdf", ".docx"))]
        logger.info(f"[CONFIG] Case study folder: {CASE_STUDY_DIR} ({len(cs_files)} file(s) found)")
    else:
        logger.info(f"[CONFIG] Case study folder not found at {CASE_STUDY_DIR} - will use description fallback")

    try:
        overall_start = time.perf_counter()
        syncer = PostgresToQdrantSyncer(pg_url)
    except Exception as e:
        print(f"\n[INIT ERROR] Failed to initialize sync infrastructure: {e}")
        print("Please check your inline Qdrant URL, API keys, and Gemini configuration.")
        raise SystemExit(1)

    projects_ok = True
    profiles_ok = True

    # Respect the boolean toggles defined at the top of the file
    if SYNC_PROJECTS:
        projects_ok = await syncer.sync_projects(dry_run=dry_run)
    else:
        logger.info("\n[MODE] SYNC_PROJECTS is False - skipping project ingestion.")

    if SYNC_PROFILES:
        profiles_ok = await syncer.sync_profiles(dry_run=dry_run)
    else:
        logger.info("\n[MODE] SYNC_PROFILES is False - skipping profile ingestion.")

    total_time = time.perf_counter() - overall_start

    print("\n" + "=" * 70)
    if projects_ok and profiles_ok:
        print(f"  [OK] ALL REQUESTED SYNC TASKS COMPLETED SUCCESSFULLY IN {total_time:.2f}s")
    else:
        failed_phases = []
        if SYNC_PROJECTS and not projects_ok:
            failed_phases.append("Projects")
        if SYNC_PROFILES and not profiles_ok:
            failed_phases.append("Profiles")
        print(f"  [WARN]  SYNC COMPLETED WITH FAILURES IN {total_time:.2f}s")
        print(f"  Failed phases: {', '.join(failed_phases)}")
    print("=" * 70 + "\n")

    if not (projects_ok and profiles_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
