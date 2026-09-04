"""Infrastructure adapter that produces text embeddings via the Gemini API.

This module wraps Google's ``genai`` client to implement the
:class:`IEmbeddingPort` domain interface.  Every public method is
protected by an exponential-backoff retry policy (3 attempts) so that
transient network / quota errors are handled transparently.
"""

import asyncio
import time

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from src.common.config import settings
from src.common.logger import get_logger
from src.domain.exceptions.rag_exceptions import EmbeddingError
from src.domain.interfaces.rag.i_embedding_port import IEmbeddingPort

logger = get_logger(__name__)


class GeminiEmbeddingAdapter(IEmbeddingPort):
    """Concrete :class:`IEmbeddingPort` backed by the Gemini Embedding API.

    Uses ``gemini-embedding-001`` for both document and query embeddings,
    differentiating only by the ``task_type`` parameter so that the
    resulting vectors are optimised for asymmetric retrieval.
    """

    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model: str = "gemini-embedding-001"
        logger.info(
            "GeminiEmbeddingAdapter initialised with model=%s",
            self._model,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def embed_document(self, text: str) -> list[float]:
        """Embed *text* for storage (``RETRIEVAL_DOCUMENT`` task type).

        Args:
            text: The document text to embed.

        Returns:
            Embedding vector as a list of floats.

        Raises:
            EmbeddingError: When the Gemini API call fails after retries.
        """
        start = time.perf_counter()
        try:
            logger.debug(f"Embedding Document Text: {text[:200]}...")
            response = await self._client.aio.models.embed_content(
                model=self._model,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=settings.qdrant_vector_size,
                ),
            )
            vector: list[float] = response.embeddings[0].values
            elapsed = time.perf_counter() - start
            logger.info(
                "embed_document completed in %.3fs  |  vector_dim=%d  |  text_len=%d",
                elapsed,
                len(vector),
                len(text),
            )
            return vector
        except Exception as exc:
            elapsed = time.perf_counter() - start
            logger.error(
                "embed_document failed after %.3fs: %s",
                elapsed,
                exc,
            )
            raise EmbeddingError(
                reason=f"Failed to embed document text: {exc}",
            ) from exc

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def embed_query(self, text: str) -> list[float]:
        """Embed *text* for search (``RETRIEVAL_QUERY`` task type).

        Args:
            text: The query text to embed.

        Returns:
            Embedding vector as a list of floats.

        Raises:
            EmbeddingError: When the Gemini API call fails after retries.
        """
        start = time.perf_counter()
        try:
            logger.debug(f"Embedding Query Text: {text[:200]}...")
            response = await self._client.aio.models.embed_content(
                model=self._model,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=settings.qdrant_vector_size,
                ),
            )
            vector: list[float] = response.embeddings[0].values
            elapsed = time.perf_counter() - start
            logger.info(
                "embed_query completed in %.3fs  |  vector_dim=%d  |  text_len=%d",
                elapsed,
                len(vector),
                len(text),
            )
            return vector
        except Exception as exc:
            elapsed = time.perf_counter() - start
            logger.error(
                "embed_query failed after %.3fs: %s",
                elapsed,
                exc,
            )
            raise EmbeddingError(
                reason=f"Failed to embed query text: {exc}",
            ) from exc

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def embed_documents_batch(
        self, texts: list[str],
    ) -> list[list[float]]:
        """Embed multiple document texts using the true Gemini batch API.

        The API accepts up to 100 texts per ``embed_content`` call, so a
        104-chunk case study becomes 2 round-trips instead of 104
        sequential ones (~0.5s each).

        Args:
            texts: A list of document texts to embed.

        Returns:
            A list of embedding vectors, one per input text.

        Raises:
            EmbeddingError: When a batch call fails after retries.
        """
        if not texts:
            return []

        start = time.perf_counter()
        vectors: list[list[float]] = []

        # Free-tier quota counts each TEXT as 1 request (limit: 100/min).
        # Using 80 per batch leaves 20 quota headroom in the same minute
        # for search queries and other embed calls that run concurrently.
        batch_size = 80
        batches = [
            texts[i:i + batch_size] for i in range(0, len(texts), batch_size)
        ]
        logger.info(
            "embed_documents_batch  |  count=%d  batches=%d",
            len(texts),
            len(batches),
        )

        try:
            for batch_idx, batch in enumerate(batches):
                # Free-tier rate limit: 100 embed_content requests per minute.
                # Each batch call counts as 1 request regardless of how many
                # texts it contains (up to 100). Pause 62s between batches so
                # the per-minute counter resets before the next call fires.
                if batch_idx > 0:
                    logger.info(
                        "embed_documents_batch  |  free-tier delay: sleeping 62s "
                        "before batch %d/%d to avoid 429 rate limit...",
                        batch_idx + 1,
                        len(batches),
                    )
                    await asyncio.sleep(62)

                response = await self._client.aio.models.embed_content(
                    model=self._model,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=settings.qdrant_vector_size,
                    ),
                )
                vectors.extend(e.values for e in response.embeddings)
        except Exception as exc:
            logger.error(
                "embed_documents_batch failed after %.3fs: %s",
                time.perf_counter() - start,
                exc,
            )
            raise EmbeddingError(
                reason=f"Failed to embed batch of {len(texts)} texts: {exc}",
            ) from exc

        elapsed = time.perf_counter() - start
        logger.info(
            "embed_documents_batch completed in %.3fs  |  count=%d",
            elapsed,
            len(vectors),
        )
        return vectors
