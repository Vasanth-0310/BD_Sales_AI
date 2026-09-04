"""Caching decorator for :class:`IEmbeddingPort` implementations.

Stores embedding vectors in a MongoDB collection keyed by the SHA-256
hash of the input text.  Cache look-ups and writes are **best-effort**:
any MongoDB failure is logged and silently bypassed so that the inner
embedding adapter is always called as a fallback.
"""

import hashlib
import time

from motor.motor_asyncio import AsyncIOMotorDatabase

from src.common.logger import get_logger
from src.domain.interfaces.rag.i_embedding_port import IEmbeddingPort

logger = get_logger(__name__)


class CachedEmbeddingAdapter(IEmbeddingPort):
    """Transparent caching layer in front of any :class:`IEmbeddingPort`.

    Uses a ``rag_embedding_cache`` MongoDB collection with documents of
    the form::

        {"_id": "<sha256-hex>", "vector": [0.12, …]}

    Cache failures never propagate — they are caught, logged, and the
    inner adapter is used directly.

    Args:
        inner: The actual embedding adapter to delegate to on cache miss.
        db: A Motor async MongoDB database instance.
    """

    def __init__(
        self,
        inner: IEmbeddingPort,
        db: AsyncIOMotorDatabase,
    ) -> None:
        self._inner = inner
        self._db = db
        self._collection = db["rag_embedding_cache"]
        logger.info(
            "CachedEmbeddingAdapter initialised  |  collection=%s",
            self._collection.name,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_text(text: str, mode: str = "document") -> str:
        """Return the SHA-256 hex digest of *text* scoped to the embedding
        mode, model, and output dimension.

        The mode MUST be part of the key: Gemini's task types
        (RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY) produce different vectors for
        identical text. Model/dimension changes must also invalidate old
        entries, otherwise stale wrong-shaped vectors get served into Qdrant.
        """
        from src.common.config import settings

        scope = f"{settings.gemini_model}|{settings.qdrant_vector_size}|{mode}"
        return hashlib.sha256(f"{scope}::{text}".encode("utf-8")).hexdigest()

    async def _lookup(self, text_hash: str) -> list[float] | None:
        """Best-effort cache read. Returns None on miss or Mongo failure."""
        try:
            cached = await self._collection.find_one({"_id": text_hash})
            return cached["vector"] if cached is not None else None
        except Exception as exc:
            logger.warning("Cache look-up failed, falling through to inner adapter: %s", exc)
            return None

    async def _store(self, text_hash: str, vector: list[float]) -> None:
        """Best-effort cache write. Never raises."""
        try:
            await self._collection.insert_one({"_id": text_hash, "vector": vector})
        except Exception as exc:
            logger.warning("Cache write failed (non-fatal): %s", exc)

    async def _embed_cached(self, text: str, mode: str) -> list[float]:
        """Shared cache-or-compute path for embed_document/embed_query."""
        # mode MUST be part of the hash: Gemini's asymmetric task types
        # produce different vectors for identical text.
        text_hash = self._hash_text(text, mode)
        start = time.perf_counter()

        cached = await self._lookup(text_hash)
        if cached is not None:
            logger.info(
                "embed_%s CACHE HIT in %.3fs  |  hash=%s",
                mode, time.perf_counter() - start, text_hash[:12],
            )
            return cached

        vector = await getattr(self._inner, f"embed_{mode}")(text)
        await self._store(text_hash, vector)
        logger.info(
            "embed_%s completed in %.3fs  |  text_len=%d",
            mode, time.perf_counter() - start, len(text),
        )
        return vector

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def embed_document(self, text: str) -> list[float]:
        """Return the cached document embedding, or compute and cache it."""
        return await self._embed_cached(text, "document")

    async def embed_query(self, text: str) -> list[float]:
        """Return the cached query embedding, or compute and cache it."""
        return await self._embed_cached(text, "query")

    async def embed_documents_batch(
        self, texts: list[str],
    ) -> list[list[float]]:
        """Embed multiple document texts, leveraging the per-item cache.

        Cache hits are served individually; all misses are embedded with a
        SINGLE call to the inner adapter's batch API (instead of one
        round-trip per text) and written to the cache in one insert.
        """
        start = time.perf_counter()
        vectors: list[list[float] | None] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_hashes: list[str] = []
        hits = 0

        for idx, text in enumerate(texts):
            text_hash = self._hash_text(text, "document")
            cached = await self._lookup(text_hash)
            if cached is not None:
                vectors[idx] = cached
                hits += 1
            else:
                miss_indices.append(idx)
                miss_hashes.append(text_hash)

        if miss_indices:
            miss_vectors = await self._inner.embed_documents_batch(
                [texts[i] for i in miss_indices]
            )
            cache_docs = []
            for i, text_hash, vector in zip(miss_indices, miss_hashes, miss_vectors):
                vectors[i] = vector
                cache_docs.append({"_id": text_hash, "vector": vector})
            try:
                await self._collection.insert_many(cache_docs, ordered=False)
            except Exception as exc:
                logger.warning("Batch cache write failed (non-fatal): %s", exc)

        elapsed = time.perf_counter() - start
        misses = len(texts) - hits
        logger.info(
            "embed_documents_batch completed in %.3fs  |  total=%d  hits=%d  misses=%d",
            elapsed, len(texts), hits, misses,
        )
        return vectors  # type: ignore[return-value]  # all slots filled above
