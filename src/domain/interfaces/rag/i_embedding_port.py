from abc import ABC, abstractmethod


class IEmbeddingPort(ABC):
    """
    Port (Interface) for text embedding services.
    Implemented by GeminiEmbeddingAdapter and CachedEmbeddingAdapter
    in the infrastructure layer.
    """

    @abstractmethod
    async def embed_document(self, text: str) -> list[float]:
        """
        Embed a document text for storage in the vector database.
        Uses task_type=RETRIEVAL_DOCUMENT for optimal retrieval performance.

        Args:
            text: The text to embed (e.g. project summary or chunk).

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            EmbeddingError: If the embedding API call fails.
        """
        ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """
        Embed a query text for searching the vector database.
        Uses task_type=RETRIEVAL_QUERY for optimal retrieval performance.

        Args:
            text: The query text (e.g. flattened job description).

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            EmbeddingError: If the embedding API call fails.
        """
        ...

    @abstractmethod
    async def embed_documents_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple document texts in a batch.

        Args:
            texts: A list of text strings to embed.

        Returns:
            A list of embedding vectors, one per input text.

        Raises:
            EmbeddingError: If any embedding API call fails.
        """
        ...
