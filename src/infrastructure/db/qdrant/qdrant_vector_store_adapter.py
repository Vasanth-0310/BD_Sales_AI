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
                "user_id": project.user_id,
                "project_name": project.name,
                "domain": project.domain,
                "techstacks": project.techstacks,
                "description": project.description,
                "links": project.links,
            }

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
                    "user_id": chunk.user_id,
                    "project_name": chunk.project_name,
                    "domain": chunk.domain,
                    "techstacks": chunk.techstacks,
                    "text": chunk.text,
                    "token_count": chunk.token_count,
                    "sequence_index": chunk.sequence_index,
                }
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
    ) -> list[dict]:
        """Dense search on chunks, filtered to specific project IDs."""
        start = time.perf_counter()
        try:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="project_id",
                        match=MatchAny(any=project_ids),
                    )
                ]
            )

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
    async def delete_project(self, project_id: str) -> None:
        """Delete all data for a project from both collections."""
        start = time.perf_counter()
        try:
            # Delete from summary collection (filter by project_id payload)
            await self._client.delete(
                collection_name=self._summary_collection,
                points_selector=models.FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="project_id",
                                match=MatchValue(value=project_id),
                            )
                        ]
                    )
                ),
            )

            # Delete from chunks collection (filter by project_id payload)
            await self._client.delete(
                collection_name=self._chunks_collection,
                points_selector=models.FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="project_id",
                                match=MatchValue(value=project_id),
                            )
                        ]
                    )
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
    async def fetch_profile_variant_by_id(self, variant_id: str) -> dict | None:
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

            payload = points[0].payload
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
    async def check_project_exists(self, project_id: str) -> bool:
        """Check whether a project exists in the Summary collection by payload project_id."""
        try:
            results, _ = await self._client.scroll(
                collection_name=self._summary_collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="project_id",
                            match=MatchValue(value=project_id),
                        )
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            exists = len(results) > 0
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
    async def check_candidate_exists(self, candidate_id: str) -> bool:
        """Check whether any profile variant exists for the given candidate_id."""
        try:
            results, _ = await self._client.scroll(
                collection_name=self._profile_collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="candidate_id",
                            match=MatchValue(value=candidate_id),
                        )
                    ]
                ),
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
    async def delete_profiles_by_candidate_id(self, candidate_id: str) -> int:
        """Delete all profile variants for a candidate. Returns the count deleted."""
        start = time.perf_counter()
        try:
            candidate_filter = Filter(
                must=[
                    FieldCondition(
                        key="candidate_id",
                        match=MatchValue(value=candidate_id),
                    )
                ]
            )

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
