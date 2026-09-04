class RAGBaseException(Exception):
    """Base exception for all RAG project matching errors."""
    pass


class VectorStoreError(RAGBaseException):
    """Raised when the vector database (Qdrant) encounters an error."""
    def __init__(self, reason: str):
        super().__init__(f"Vector store error: {reason}")
        self.reason = reason


class EmbeddingError(RAGBaseException):
    """Raised when the embedding API (Gemini) fails to produce vectors."""
    def __init__(self, reason: str):
        super().__init__(f"Embedding error: {reason}")
        self.reason = reason


class SynthesisError(RAGBaseException):
    """Raised when the LLM synthesis/reranking step fails."""
    def __init__(self, reason: str):
        super().__init__(f"Synthesis error: {reason}")
        self.reason = reason


class DocumentExtractionError(RAGBaseException):
    """Raised when text extraction from a .docx or .pdf file fails."""
    def __init__(self, reason: str):
        super().__init__(f"Document extraction error: {reason}")
        self.reason = reason
