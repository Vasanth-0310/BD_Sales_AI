from abc import ABC, abstractmethod
from src.domain.entities.project import Project, ProjectChunk


class IVectorStorePort(ABC):
    """
    Port (Interface) for vector database operations.
    Implemented by QdrantVectorStoreAdapter in the infrastructure layer.
    Uses two separate collections: one for summaries (Stage 1) and
    one for chunks (Stage 2).
    """

    @abstractmethod
    async def upsert_project_summary(self, project: Project, vector: list[float]) -> None:
        """
        Upsert a project summary vector into the summaries collection.
        Uses project_id as the Qdrant point ID.

        Args:
            project: The Project entity containing metadata for the payload.
            vector: The 1536-dim embedding vector for the summary.

        Raises:
            VectorStoreError: If the upsert operation fails.
        """
        ...

    @abstractmethod
    async def upsert_project_chunks(self, chunks: list[ProjectChunk], vectors: list[list[float]]) -> None:
        """
        Upsert multiple chunk vectors into the chunks collection.
        Each chunk gets a UUID-based point ID.

        Args:
            chunks: List of ProjectChunk entities with metadata for payloads.
            vectors: List of 1536-dim embedding vectors, one per chunk.

        Raises:
            VectorStoreError: If the upsert operation fails.
        """
        ...

    @abstractmethod
    async def search_summaries_dense(self, query_vector: list[float], top_k: int = 10) -> list[dict]:
        """
        Dense (cosine similarity) search on the summaries collection.

        Args:
            query_vector: The query embedding vector.
            top_k: Maximum number of results to return.

        Returns:
            List of dicts with keys: id, score, payload.

        Raises:
            VectorStoreError: If the search operation fails.
        """
        ...

    @abstractmethod
    async def search_summaries_keyword(self, query_text: str, top_k: int = 10) -> list[dict]:
        """
        Keyword (MatchText) search on the summaries collection.

        Args:
            query_text: The raw query text for keyword matching.
            top_k: Maximum number of results to return.

        Returns:
            List of dicts with keys: id, score, payload.

        Raises:
            VectorStoreError: If the search operation fails.
        """
        ...

    @abstractmethod
    async def search_chunks_dense(self, query_vector: list[float], project_ids: list[str], top_k: int = 10) -> list[dict]:
        """
        Dense search on the chunks collection, filtered to specific project IDs.

        Args:
            query_vector: The query embedding vector.
            project_ids: List of project IDs to restrict the search to.
            top_k: Maximum number of chunk results to return.

        Returns:
            List of dicts with keys: id, score, payload.

        Raises:
            VectorStoreError: If the search operation fails.
        """
        ...

    @abstractmethod
    async def delete_project(self, project_id: str) -> None:
        """
        Delete all vectors (summary + chunks) for a given project ID
        from both collections.

        Args:
            project_id: The project ID whose data should be purged.

        Raises:
            VectorStoreError: If the delete operation fails.
        """
        ...

    # ------------------------------------------------------------------
    # Profile Variant Operations
    # ------------------------------------------------------------------

    @abstractmethod
    async def upsert_profile_variant(
        self, variant_id: str, vector: list[float], payload: dict,
    ) -> None:
        """
        Upsert a profile variant point into the profile_variants collection.
        Uses variant_id (UUID string) directly as the Qdrant point ID.

        Args:
            variant_id: The UUID string variant ID (used as Qdrant point ID).
            vector: The 1536-dim embedding vector for the variant summary.
            payload: Full variant metadata to store alongside the vector.

        Raises:
            VectorStoreError: If the upsert operation fails.
        """
        ...

    @abstractmethod
    async def delete_profile_variant(self, variant_id: str) -> None:
        """
        Delete a single profile variant point by its variant_id.

        Args:
            variant_id: The UUID string variant ID (used as Qdrant point ID).

        Raises:
            VectorStoreError: If the delete operation fails.
        """
        ...

    @abstractmethod
    async def search_profile_variants_dense(
        self, query_vector: list[float], top_k: int = 10,
    ) -> list[dict]:
        """
        Dense (cosine similarity) search on the profile_variants collection.

        Args:
            query_vector: The query embedding vector.
            top_k: Maximum number of results to return.

        Returns:
            List of dicts with keys: id, score, payload.

        Raises:
            VectorStoreError: If the search operation fails.
        """
        ...

    @abstractmethod
    async def search_profile_variants_keyword(
        self, query_text: str, top_k: int = 10,
    ) -> list[dict]:
        """
        Keyword (MatchText) search on the profile_variants collection.
        Searches across combined_text, variant_title, and tech_stacks fields.

        Args:
            query_text: The raw query text for keyword matching.
            top_k: Maximum number of results to return.

        Returns:
            List of dicts with keys: id, score, payload.

        Raises:
            VectorStoreError: If the search operation fails.
        """
        ...

    @abstractmethod
    async def fetch_profile_variant_by_id(self, variant_id: str) -> dict | None:
        """
        Fetch a single profile variant's full payload from Qdrant by its
        variant_id, without performing a vector search.

        Args:
            variant_id: The UUID string variant ID (used as Qdrant point ID).

        Returns:
            The payload dict if found, or None if no variant exists with
            that ID.

        Raises:
            VectorStoreError: If the lookup operation fails.
        """
        ...

    @abstractmethod
    async def check_project_exists(self, project_id: str) -> bool:
        """
        Check whether a project with the given project_id exists in the
        summaries collection.

        Args:
            project_id: The UUID string project ID (used as Qdrant point ID).

        Returns:
            True if the project exists, False otherwise.

        Raises:
            VectorStoreError: If the lookup fails.
        """
        ...

    @abstractmethod
    async def check_candidate_exists(self, candidate_id: str) -> bool:
        """
        Check whether any profile variant exists for the given candidate_id
        in the profile_variants collection.

        Args:
            candidate_id: The UUID string candidate ID stored in the payload.

        Returns:
            True if at least one variant exists, False otherwise.

        Raises:
            VectorStoreError: If the lookup fails.
        """
        ...

    @abstractmethod
    async def delete_profiles_by_candidate_id(self, candidate_id: str) -> int:
        """
        Delete all profile variants for a given candidate_id from the
        profile_variants collection.

        Args:
            candidate_id: The UUID string candidate ID stored in the payload.

        Returns:
            The number of variants deleted.

        Raises:
            VectorStoreError: If the delete operation fails.
        """
        ...
