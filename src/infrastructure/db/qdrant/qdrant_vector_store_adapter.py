"""Qdrant vector store adapter implementing IVectorStorePort.

Uses two separate collections — one for project summaries (Stage 1)
and one for project chunks (Stage 2) — to eliminate the need for
type-based payload filtering during search.
"""

import time
import re

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchText,
    MatchAny,
    IsEmptyCondition,
    PayloadField,
)
from tenacity import retry, stop_after_attempt, wait_exponential

from src.common.config import settings
from src.common.logger import get_logger
from src.domain.entities.project import Project, ProjectChunk
from src.domain.exceptions.rag_exceptions import VectorStoreError
from src.domain.interfaces.rag.i_vector_store_port import IVectorStorePort

logger = get_logger(__name__)


class QdrantVectorStoreAdapter(IVectorStorePort):
    """Concrete :class:`IVectorStorePort` backed by Qdrant Cloud.

    Manages two collections:

    - **Summary collection** — one point per project, used for Stage 1
      hybrid search.
    - **Chunks collection** — multiple points per project, used for
      Stage 2 dense chunk retrieval.
    """

    def __init__(self) -> None:
        self._client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        self._summary_collection: str = settings.qdrant_summary_collection
        self._chunks_collection: str = settings.qdrant_chunks_collection
        self._profile_collection: str = settings.qdrant_profile_variants_collection
        logger.info(
            "QdrantVectorStoreAdapter initialised  |  "
            "summary_collection=%s  chunks_collection=%s  profile_collection=%s",
            self._summary_collection,
            self._chunks_collection,
            self._profile_collection,
        )

    # ------------------------------------------------------------------
    # Upserts
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def upsert_project_summary(
        self, project: Project, vector: list[float],
    ) -> None:
        """Upsert a project summary point into the summaries collection."""
        start = time.perf_counter()
        try:
            payload = {
                "project_id": project.project_id,
                "project_name": project.name,
                "domain": project.domain,
                "techstacks": project.techstacks,
                "description": project.description,
                "links": project.links,
            }
            # Omit when unowned: a stored user_id=null is NOT matched by the
            # tenant filter's is_empty clause in all Qdrant versions.
            if project.user_id:
                payload["user_id"] = project.user_id

            await self._client.upsert(
                collection_name=self._summary_collection,
                points=[
                    PointStruct(
                        id=project.project_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )

            elapsed = time.perf_counter() - start
            logger.info(
                "upsert_project_summary completed in %.3fs  |  project_id=%s",
                elapsed,
                project.project_id,
            )
        except Exception as exc:
            logger.error("upsert_project_summary failed: %s", exc)
            raise VectorStoreError(reason=f"Summary upsert failed. Details logged.") from exc

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def upsert_project_chunks(
        self,
        chunks: list[ProjectChunk],
        vectors: list[list[float]],
    ) -> None:
        """Upsert chunk points into the chunks collection.

        NOTE: callers must clear prior chunks first (e.g. via delete_project).
        IngestProjectUseCase always calls delete_project() before this, so no
        internal delete is performed here — it would be a redundant round-trip
        against an already-empty filter.
        """
        if not chunks:
            return

        start = time.perf_counter()
        project_id = chunks[0].project_id

        try:
            # Build new points
            points = []
            for chunk, vector in zip(chunks, vectors):
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "project_id": chunk.project_id,
                    "project_name": chunk.project_name,
                    "domain": chunk.domain,
                    "techstacks": chunk.techstacks,
                    "text": chunk.text,
                    "token_count": chunk.token_count,
                    "sequence_index": chunk.sequence_index,
                }
                # Omit when unowned (same is_empty/null caveat as summaries).
                if chunk.user_id:
                    payload["user_id"] = chunk.user_id
                points.append(
                    PointStruct(
                        id=chunk.chunk_id,
                        vector=vector,
                        payload=payload,
                    )
                )

            await self._client.upsert(
                collection_name=self._chunks_collection,
                points=points,
            )

            elapsed = time.perf_counter() - start
            logger.info(
                "upsert_project_chunks completed in %.3fs  |  "
                "project_id=%s  chunks=%d",
                elapsed,
                project_id,
                len(points),
            )
        except Exception as exc:
            logger.error("upsert_project_chunks failed: %s", exc)
            raise VectorStoreError(
                reason=f"Chunk upsert failed for project {project_id}. Details logged."
            ) from exc

    # ------------------------------------------------------------------
    # Searches
    # ------------------------------------------------------------------

    @staticmethod
    def _tenant_filter(user_id: str | None) -> Filter | None:
        """Tenant-isolation filter for multi-user deployments.

        Matches points owned by ``user_id`` OR legacy points with no
        ``user_id`` payload at all (so data ingested before multi-tenancy
        remains searchable). Returns None when no scoping is requested.
        """
        if not user_id:
            return None
        return Filter(
            should=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                IsEmptyCondition(is_empty=PayloadField(key="user_id")),
            ]
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def search_summaries_dense(
        self, query_vector: list[float], top_k: int = 10,
        user_id: str | None = None,
    ) -> list[dict]:
        """Dense cosine-similarity search on the summaries collection."""
        start = time.perf_counter()
        try:
            results = await self._client.query_points(
                collection_name=self._summary_collection,
                query=query_vector,
                query_filter=self._tenant_filter(user_id),
                limit=top_k,
            )

            output = [
                {
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload,
                }
                for point in results.points
            ]

            elapsed = time.perf_counter() - start
            logger.info(
                "search_summaries_dense completed in %.3fs  |  results=%d",
                elapsed,
                len(output),
            )
            return output
        except Exception as exc:
            logger.error("search_summaries_dense failed: %s", exc)
            raise VectorStoreError(
                reason=f"Dense summary search failed. Details logged."
            ) from exc

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def search_summaries_keyword(
        self, query_text: str, top_k: int = 10,
        user_id: str | None = None,
    ) -> list[dict]:
        """Keyword (MatchText) search on the summaries collection.

        ``query_text`` is expected to be a bounded space-separated keyword
        string. Each token is expanded into per-field OR conditions so Qdrant
        never receives a full JD paragraph as a single MatchText expression.
        """
        start = time.perf_counter()
        try:
            tokens = self._keyword_tokens(query_text)
            if not tokens:
                return []

            should_conditions = []
            for token in tokens:
                should_conditions.append(
                    FieldCondition(key="description", match=MatchText(text=token))
                )
                should_conditions.append(
                    FieldCondition(key="project_name", match=MatchText(text=token))
                )
                should_conditions.append(
                    FieldCondition(key="domain", match=MatchText(text=token))
                )
                # techstacks is a list payload — MatchText matches against
                # array items, so a JD keyword hits projects that list the
                # skill in their stack even if the description omits it.
                should_conditions.append(
                    FieldCondition(key="techstacks", match=MatchText(text=token))
                )

            # Keyword conditions live in a nested OR-group; the tenant filter
            # (if any) is ANDed on top via `must` — putting both in `should`
            # would let tenant-owned points bypass the keyword match.
            keyword_group = Filter(should=should_conditions)
            must_clauses: list = [keyword_group]
            tenant = self._tenant_filter(user_id)
            if tenant is not None:
                must_clauses.append(tenant)
            scroll_filter = Filter(must=must_clauses)

            logger.debug(
                "search_summaries_keyword  |  tokens=%d  conditions=%d",
                len(tokens),
                len(should_conditions),
            )

            points, _next_page = await self._client.scroll(
                collection_name=self._summary_collection,
                scroll_filter=scroll_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )

            output = [
                {
                    "id": point.id,
                    "score": 1.0,  # MatchText does not return a score
                    "payload": point.payload,
                }
                for point in points
            ]

            elapsed = time.perf_counter() - start
            logger.info(
                "search_summaries_keyword completed in %.3fs  |  results=%d",
                elapsed,
                len(output),
            )
            return output
        except Exception as exc:
            logger.error("search_summaries_keyword failed: %s", exc)
            raise VectorStoreError(
                reason=f"Keyword summary search failed. Details logged."
            ) from exc

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def search_chunks_dense(
        self,
        query_vector: list[float],
        project_ids: list[str],
        top_k: int = 10,
        user_id: str | None = None,
    ) -> list[dict]:
        """Dense search on chunks, filtered to specific project IDs."""
        start = time.perf_counter()
        try:
            must_base = [
                FieldCondition(
                    key="project_id",
                    match=MatchAny(any=project_ids),
                )
            ]
            # Stage-1 already scopes project_ids per-tenant, but enforcing the
            # boundary here too keeps chunk content isolated even if a stale
            # or cross-tenant project_id sneaks into the Stage-1 output.
            tenant = self._tenant_filter(user_id)
            if tenant is not None:
                must_base.append(tenant)
            query_filter = Filter(must=must_base)

            results = await self._client.query_points(
                collection_name=self._chunks_collection,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
            )

            output = [
                {
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload,
                }
                for point in results.points
            ]

            elapsed = time.perf_counter() - start
            logger.info(
                "search_chunks_dense completed in %.3fs  |  "
                "project_ids=%s  results=%d",
                elapsed,
                project_ids,
                len(output),
            )
            return output
        except Exception as exc:
            logger.error("search_chunks_dense failed: %s", exc)
            raise VectorStoreError(
                reason=f"Dense chunk search failed. Details logged."
            ) from exc

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def delete_project(self, project_id: str, user_id: str | None = None) -> None:
        """Delete all data for a project from both collections.

        Chunks are deleted FIRST: an interrupted ingest can leave chunks with
        no summary point — deleting the summary first would still work here
        (both are filter-based), but chunks-first guarantees a summary-only
        failure can never orphan the (typically larger) chunk set.
        """
        start = time.perf_counter()
        try:
            # Tenant guard: AND the owner filter onto the project_id condition
            # (legacy points without user_id stay deletable by their owner).
            must_base = [
                FieldCondition(
                    key="project_id",
                    match=MatchValue(value=project_id),
                )
            ]
            tenant = self._tenant_filter(user_id)
            if tenant is not None:
                must_base.append(tenant)

            # Delete from chunks collection (filter by project_id payload)
            await self._client.delete(
                collection_name=self._chunks_collection,
                points_selector=models.FilterSelector(
                    filter=Filter(must=must_base)
                ),
            )

            # Delete from summary collection (filter by project_id payload)
            await self._client.delete(
                collection_name=self._summary_collection,
                points_selector=models.FilterSelector(
                    filter=Filter(must=must_base)
                ),
            )

            elapsed = time.perf_counter() - start
            logger.info(
                "delete_project completed in %.3fs  |  project_id=%s",
                elapsed,
                project_id,
            )
        except Exception as exc:
            logger.error("delete_project failed: %s", exc)
            raise VectorStoreError(
                reason=f"Delete project {project_id} failed. Details logged."
            ) from exc

    # ------------------------------------------------------------------
    # Profile Variant Operations
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def upsert_profile_variant(
        self, variant_id: str, vector: list[float], payload: dict,
    ) -> None:
        """Upsert a profile variant point using variant_id as the Qdrant point ID."""
        start = time.perf_counter()
        try:
            await self._client.upsert(
                collection_name=self._profile_collection,
                points=[
                    PointStruct(
                        id=variant_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )

            elapsed = time.perf_counter() - start
            logger.info(
                "upsert_profile_variant completed in %.3fs  |  variant_id=%s",
                elapsed,
                variant_id,
            )
        except Exception as exc:
            logger.error("upsert_profile_variant failed: %s", exc)
            raise VectorStoreError(
                reason=f"Profile variant upsert failed for variant_id={variant_id}. Details logged."
            ) from exc

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def delete_profile_variant(self, variant_id: str) -> None:
        """Delete a single profile variant point by its variant_id."""
        start = time.perf_counter()
        try:
            await self._client.delete(
                collection_name=self._profile_collection,
                points_selector=models.PointIdsList(
                    points=[variant_id],
                ),
            )

            elapsed = time.perf_counter() - start
            logger.info(
                "delete_profile_variant completed in %.3fs  |  variant_id=%s",
                elapsed,
                variant_id,
            )
        except Exception as exc:
            logger.error("delete_profile_variant failed: %s", exc)
            raise VectorStoreError(
                reason=f"Delete profile variant {variant_id} failed. Details logged."
            ) from exc

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def search_profile_variants_dense(
        self, query_vector: list[float], top_k: int = 10,
        user_id: str | None = None,
    ) -> list[dict]:
        """Dense cosine-similarity search on the profile_variants collection."""
        start = time.perf_counter()
        try:
            results = await self._client.query_points(
                collection_name=self._profile_collection,
                query=query_vector,
                query_filter=self._tenant_filter(user_id),
                limit=top_k,
            )

            output = [
                {
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload,
                }
                for point in results.points
            ]

            elapsed = time.perf_counter() - start
            logger.info(
                "search_profile_variants_dense completed in %.3fs  |  results=%d",
                elapsed,
                len(output),
            )
            return output
        except Exception as exc:
            logger.error("search_profile_variants_dense failed: %s", exc)
            raise VectorStoreError(
                reason=f"Dense profile variant search failed. Details logged."
            ) from exc

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def search_profile_variants_keyword(
        self, query_text: str, top_k: int = 10,
        user_id: str | None = None,
    ) -> list[dict]:
        """Keyword search on the profile_variants collection.

        ``query_text`` is expected to be a **space-separated list of keyword
        tokens** (e.g. ``"Python FastAPI AWS Docker"``), not a full sentence.
        Each token is expanded into three ``MatchText`` OR conditions covering
        ``combined_text``, ``variant_title``, and ``tech_stacks_text`` so that
        a profile variant matching *any* of the keywords is returned.

        Passing a full paragraph would cause Qdrant's ``MatchText`` to AND all
        tokens together, guaranteeing zero results.
        """
        start = time.perf_counter()
        try:
            # Fan out every token into per-field OR conditions
            tokens = self._keyword_tokens(query_text)
            if not tokens:
                return []

            should_conditions = []
            for token in tokens:
                should_conditions.append(
                    FieldCondition(key="combined_text", match=MatchText(text=token))
                )
                should_conditions.append(
                    FieldCondition(key="variant_title", match=MatchText(text=token))
                )
                should_conditions.append(
                    FieldCondition(key="tech_stacks_text", match=MatchText(text=token))
                )

            # Nested OR-group for keywords + ANDed tenant filter (same
            # reasoning as search_summaries_keyword).
            keyword_group = Filter(should=should_conditions)
            must_clauses: list = [keyword_group]
            tenant = self._tenant_filter(user_id)
            if tenant is not None:
                must_clauses.append(tenant)
            scroll_filter = Filter(must=must_clauses)

            logger.debug(
                "search_profile_variants_keyword  |  tokens=%d  conditions=%d",
                len(tokens),
                len(should_conditions),
            )

            points, _next_page = await self._client.scroll(
                collection_name=self._profile_collection,
                scroll_filter=scroll_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )

            output = [
                {
                    "id": point.id,
                    "score": 1.0,
                    "payload": point.payload,
                }
                for point in points
            ]

            elapsed = time.perf_counter() - start
            logger.info(
                "search_profile_variants_keyword completed in %.3fs  |  results=%d",
                elapsed,
                len(output),
            )
            return output
        except Exception as exc:
            logger.error("search_profile_variants_keyword failed: %s", exc)
            raise VectorStoreError(
                reason=f"Keyword profile variant search failed. Details logged."
            ) from exc

    @staticmethod
    def _keyword_tokens(query_text: str, max_tokens: int = 40) -> list[str]:
        """Normalize and cap keyword tokens before building Qdrant filters."""

        tokens: list[str] = []
        seen: set[str] = set()
        for raw in re.findall(r"(?:\.NET|[A-Za-z][A-Za-z0-9+#.]*)", query_text):
            token = raw.strip(".,;:()[]{}'\"").strip()
            if not token:
                continue
            key = token.lower()
            if key in seen:
                continue
            seen.add(key)
            tokens.append(token)
            if len(tokens) >= max_tokens:
                break
        return tokens

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def fetch_profile_variant_by_id(
        self, variant_id: str, user_id: str | None = None,
    ) -> dict | None:
        """Fetch a single profile variant payload directly by its variant_id.

        Since variant_id is stored as the Qdrant point ID during upsert,
        we use `retrieve` for a direct O(1) lookup — no payload index needed.

        Args:
            variant_id: The UUID string variant ID (used as the Qdrant point ID).

        Returns:
            The payload dict if found, or None if no matching variant exists.

        Raises:
            VectorStoreError: If the Qdrant query fails.
        """
        start = time.perf_counter()
        try:
            points = await self._client.retrieve(
                collection_name=self._profile_collection,
                ids=[variant_id],
                with_payload=True,
                with_vectors=False,
            )

            elapsed = time.perf_counter() - start
            if not points:
                logger.warning(
                    "fetch_profile_variant_by_id  |  variant_id=%s not found  "
                    "(%.3fs)",
                    variant_id,
                    elapsed,
                )
                return None

            payload = points[0].payload or {}

            # Tenant guard: refuse payloads owned by a different user.
            # Legacy payloads without user_id remain accessible.
            owner = payload.get("user_id")
            if user_id and owner and owner != user_id:
                logger.warning(
                    "fetch_profile_variant_by_id  |  variant_id=%s belongs to "
                    "another tenant — access denied.",
                    variant_id,
                )
                return None

            logger.info(
                "fetch_profile_variant_by_id completed in %.3fs  |  "
                "variant_id=%s  candidate=%s",
                elapsed,
                variant_id,
                payload.get("candidate_name", "unknown"),
            )
            return payload

        except Exception as exc:
            logger.error("fetch_profile_variant_by_id failed: %s", exc)
            raise VectorStoreError(
                reason=f"Fetch profile variant {variant_id} failed. Details logged."
            ) from exc

    # ------------------------------------------------------------------
    # Existence Checks
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def check_project_exists(
        self, project_id: str, user_id: str | None = None,
    ) -> bool:
        """Check whether a project exists in Summary OR Chunks by project_id.

        An ingest interrupted between chunk-write and summary-write leaves
        chunks with no summary point — checking only the Summary collection
        would make such projects undeletable (404) with orphaned vectors.
        """
        start = time.perf_counter()
        try:
            must_base = [
                FieldCondition(
                    key="project_id",
                    match=MatchValue(value=project_id),
                )
            ]
            tenant = self._tenant_filter(user_id)
            if tenant is not None:
                must_base.append(tenant)

            results, _ = await self._client.scroll(
                collection_name=self._summary_collection,
                scroll_filter=Filter(must=must_base),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            if results:
                exists = True
            else:
                chunk_results, _ = await self._client.scroll(
                    collection_name=self._chunks_collection,
                    scroll_filter=Filter(must=must_base),
                    limit=1,
                    with_payload=False,
                    with_vectors=False,
                )
                exists = len(chunk_results) > 0
                if exists:
                    logger.warning(
                        "check_project_exists  |  project_id=%s has CHUNKS but no "
                        "summary point (interrupted ingest) — treating as existing.",
                        project_id,
                    )
            logger.info(
                "check_project_exists  |  project_id=%s  exists=%s",
                project_id, exists,
            )
            return exists
        except Exception as exc:
            logger.error("check_project_exists failed: %s", exc)
            raise VectorStoreError(
                reason=f"check_project_exists failed for {project_id}. Details logged."
            ) from exc

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def check_candidate_exists(
        self, candidate_id: str, user_id: str | None = None,
    ) -> bool:
        """Check whether any profile variant exists for the given candidate_id."""
        start = time.perf_counter()
        try:
            must_base = [
                FieldCondition(
                    key="candidate_id",
                    match=MatchValue(value=candidate_id),
                )
            ]
            tenant = self._tenant_filter(user_id)
            if tenant is not None:
                must_base.append(tenant)
            results, _ = await self._client.scroll(
                collection_name=self._profile_collection,
                scroll_filter=Filter(must=must_base),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            exists = len(results) > 0
            logger.info(
                "check_candidate_exists  |  candidate_id=%s  exists=%s",
                candidate_id, exists,
            )
            return exists
        except Exception as exc:
            logger.error("check_candidate_exists failed: %s", exc)
            raise VectorStoreError(
                reason=f"check_candidate_exists failed for {candidate_id}. Details logged."
            ) from exc

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def delete_profiles_by_candidate_id(
        self, candidate_id: str, user_id: str | None = None,
    ) -> int:
        """Delete all profile variants for a candidate. Returns the count deleted."""
        start = time.perf_counter()
        try:
            must_base = [
                FieldCondition(
                    key="candidate_id",
                    match=MatchValue(value=candidate_id),
                )
            ]
            tenant = self._tenant_filter(user_id)
            if tenant is not None:
                must_base.append(tenant)
            candidate_filter = Filter(must=must_base)

            # Scroll through all matching points to count them
            all_points = []
            offset = None
            while True:
                batch, next_offset = await self._client.scroll(
                    collection_name=self._profile_collection,
                    scroll_filter=candidate_filter,
                    limit=100,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
                all_points.extend(batch)
                if next_offset is None:
                    break
                offset = next_offset
            count = len(all_points)

            # Delete all matching points
            await self._client.delete(
                collection_name=self._profile_collection,
                points_selector=models.FilterSelector(
                    filter=candidate_filter,
                ),
            )

            elapsed = time.perf_counter() - start
            logger.info(
                "delete_profiles_by_candidate_id completed in %.3fs  |  candidate_id=%s  deleted=%d",
                elapsed, candidate_id, count,
            )
            return count
        except Exception as exc:
            logger.error("delete_profiles_by_candidate_id failed: %s", exc)
            raise VectorStoreError(
                reason=f"delete_profiles_by_candidate_id failed for {candidate_id}. Details logged."
            ) from exc

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def delete_profile_variant_by_id(
        self,
        variant_id: str,
        candidate_id: str,
        user_id: str | None = None,
    ) -> bool:
        """Delete a single profile variant by its variant_id (point ID).

        Cross-checks candidate_id and tenant ownership before deletion.
        Returns True if the variant was found and deleted, False if not found.
        """
        start = time.perf_counter()
        try:
            # Retrieve the point by its ID to verify ownership
            points = await self._client.retrieve(
                collection_name=self._profile_collection,
                ids=[variant_id],
                with_payload=["candidate_id", "user_id"],
                with_vectors=False,
            )

            if not points:
                logger.warning(
                    "delete_profile_variant_by_id: variant_id=%s not found",
                    variant_id,
                )
                return False

            point = points[0]
            payload = point.payload or {}

            # Cross-check: variant must belong to the specified candidate
            stored_candidate = payload.get("candidate_id", "")
            if stored_candidate != candidate_id:
                logger.warning(
                    "delete_profile_variant_by_id: variant_id=%s belongs to "
                    "candidate '%s', not '%s' — access denied",
                    variant_id, stored_candidate, candidate_id,
                )
                return False

            # Tenant check: if user_id is provided, the point must match
            if user_id:
                stored_user = payload.get("user_id")
                if stored_user and stored_user != user_id:
                    logger.warning(
                        "delete_profile_variant_by_id: variant_id=%s belongs to "
                        "tenant '%s', not '%s' — access denied",
                        variant_id, stored_user, user_id,
                    )
                    return False

            # Safe to delete — remove the single point
            await self._client.delete(
                collection_name=self._profile_collection,
                points_selector=models.PointIdsList(points=[variant_id]),
            )

            elapsed = time.perf_counter() - start
            logger.info(
                "delete_profile_variant_by_id completed in %.3fs  |  "
                "variant_id=%s  candidate_id=%s",
                elapsed, variant_id, candidate_id,
            )
            return True

        except Exception as exc:
            logger.error("delete_profile_variant_by_id failed: %s", exc)
            raise VectorStoreError(
                reason=f"delete_profile_variant_by_id failed for {variant_id}. Details logged."
            ) from exc

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def reconcile_profile_variants(
        self, candidate_id: str, keep_variant_ids: list[str], user_id: str | None = None,
    ) -> int:
        """Delete stale variants (candidate re-ingest reconciliation).

        Called after a successful re-ingest: removes points whose variant_id
        is NOT in keep_variant_ids, so removed/renamed variants don't linger
        in the vector DB matching queries forever.
        """
        start = time.perf_counter()
        try:
            keep = set(keep_variant_ids)
            must_base = [
                FieldCondition(
                    key="candidate_id",
                    match=MatchValue(value=candidate_id),
                )
            ]
            tenant = self._tenant_filter(user_id)
            if tenant is not None:
                must_base.append(tenant)

            stale_ids: list[str] = []
            offset = None
            while True:
                points, next_offset = await self._client.scroll(
                    collection_name=self._profile_collection,
                    scroll_filter=Filter(must=must_base),
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in points:
                    vid = (point.payload or {}).get("variant_id")
                    if vid and vid not in keep:
                        stale_ids.append(str(point.id))
                if next_offset is None:
                    break
                offset = next_offset

            if not stale_ids:
                return 0

            await self._client.delete(
                collection_name=self._profile_collection,
                points_selector=models.PointIdsList(points=stale_ids),
            )

            elapsed = time.perf_counter() - start
            logger.info(
                "reconcile_profile_variants completed in %.3fs  |  candidate_id=%s  removed=%d",
                elapsed, candidate_id, len(stale_ids),
            )
            return len(stale_ids)
        except Exception as exc:
            logger.error("reconcile_profile_variants failed: %s", exc)
            raise VectorStoreError(
                reason=f"reconcile_profile_variants failed for {candidate_id}. Details logged."
            ) from exc

    # ─── Sync Support Methods ────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def scroll_all_point_ids(self, collection_name: str) -> set[str]:
        """Scroll through an entire collection and return all point IDs.

        Lightweight — does NOT transfer vectors or payloads over the network.
        Uses the same pagination pattern as delete_profiles_by_candidate_id.

        Args:
            collection_name: Name of the Qdrant collection to scroll.

        Returns:
            A set of all point IDs (strings) in the collection.
        """
        start = time.perf_counter()
        try:
            all_ids: set[str] = set()
            offset = None
            while True:
                batch, next_offset = await self._client.scroll(
                    collection_name=collection_name,
                    scroll_filter=None,
                    limit=250,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
                for point in batch:
                    all_ids.add(str(point.id))
                if next_offset is None:
                    break
                offset = next_offset

            elapsed = time.perf_counter() - start
            logger.info(
                "scroll_all_point_ids completed in %.3fs  |  collection=%s  count=%d",
                elapsed, collection_name, len(all_ids),
            )
            return all_ids
        except Exception as exc:
            logger.error("scroll_all_point_ids failed for collection %s: %s", collection_name, exc)
            raise VectorStoreError(
                reason=f"scroll_all_point_ids failed for {collection_name}. Details logged."
            ) from exc
