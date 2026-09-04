"""Semantic text chunking with sentence-transformer similarity grouping."""

import re

from src.common.logger import get_logger

logger = get_logger(__name__)


class SemanticChunker:
    """Splits text into semantically coherent chunks.

    The chunker attempts to use a ``SentenceTransformer`` model
    (``all-MiniLM-L6-v2``) to group consecutive sentences by cosine
    similarity.  If the model cannot be loaded the chunker transparently
    falls back to character-based splitting via
    :class:`~langchain_text_splitters.RecursiveCharacterTextSplitter`.

    Parameters
    ----------
    None — the constructor automatically tries to load the model.

    Attributes
    ----------
    _use_fallback : bool
        ``True`` when the sentence-transformer model is unavailable.
    _model : SentenceTransformer | None
        The loaded model, or ``None`` when in fallback mode.
    """

    _SIMILARITY_THRESHOLD: float = 0.75
    """Minimum cosine similarity between consecutive sentences to keep
    them in the same chunk."""

    _MAX_CHUNK_WORDS: int = 300
    """Maximum number of words allowed in a single chunk before forcing
    a split."""

    _OVERLAP_WORDS: int = 50
    """Maximum number of trailing words from the previous chunk to carry
    over as overlap into the next chunk."""

    _FALLBACK_CHUNK_SIZE: int = 1200
    """Character-level chunk size for the fallback splitter (≈300 tokens)."""

    _FALLBACK_CHUNK_OVERLAP: int = 200
    """Character-level overlap for the fallback splitter."""

    def __init__(self) -> None:
        self._model = None
        self._use_fallback: bool = True

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self._use_fallback = False
            logger.info(
                "SentenceTransformer model 'all-MiniLM-L6-v2' loaded successfully"
            )
        except Exception as exc:
            logger.warning(
                "Failed to load SentenceTransformer model — falling back to "
                "RecursiveCharacterTextSplitter: %s",
                exc,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, text: str) -> list[dict]:
        """Split *text* into a list of chunk dictionaries.

        Each dictionary has the shape::

            {"text": "...", "token_count": N}

        where ``token_count`` is approximated as the whitespace-split word
        count.

        Parameters
        ----------
        text:
            The full document text to chunk.

        Returns
        -------
        list[dict]
            Ordered list of chunk dicts.
        """
        logger.debug(f"Chunking input text (length: {len(text)} chars)")
        if self._use_fallback:
            chunks = self._fallback_chunk(text)
            logger.info(
                "Produced %d chunks using fallback (RecursiveCharacterTextSplitter)",
                len(chunks),
            )
            for i, c in enumerate(chunks):
                logger.debug(f"Fallback Chunk {i+1} [{c['token_count']} words]: {c['text'][:100]}...")
            return chunks

        try:
            chunks = self._semantic_chunk(text)
            logger.info(
                "Produced %d chunks using semantic similarity grouping",
                len(chunks),
            )
            for i, c in enumerate(chunks):
                logger.debug(f"Semantic Chunk {i+1} [{c['token_count']} words]: {c['text'][:100]}...")
            return chunks
        except Exception as exc:
            logger.warning(
                "Semantic chunking failed at runtime — falling back to "
                "RecursiveCharacterTextSplitter: %s",
                exc,
            )
            chunks = self._fallback_chunk(text)
            logger.info(
                "Produced %d chunks using fallback after semantic failure",
                len(chunks),
            )
            for i, c in enumerate(chunks):
                logger.debug(f"Fallback Chunk {i+1} [{c['token_count']} words]: {c['text'][:100]}...")
            return chunks

    # ------------------------------------------------------------------
    # Semantic mode
    # ------------------------------------------------------------------

    def _semantic_chunk(self, text: str) -> list[dict]:
        """Chunk text by grouping consecutive sentences with high cosine
        similarity.

        Steps
        -----
        1. Split text into sentences via regex on ``[.!?]``.
        2. Encode sentences with the loaded ``SentenceTransformer``.
        3. Compute pairwise cosine similarity between consecutive
           sentence embeddings.
        4. Group sentences: start a new chunk when similarity drops below
           :pyattr:`_SIMILARITY_THRESHOLD` **or** the accumulated word
           count exceeds :pyattr:`_MAX_CHUNK_WORDS`.
        5. Apply overlap: when starting a new chunk, carry over trailing
           sentence(s) from the previous chunk whose total word count
           ≤ :pyattr:`_OVERLAP_WORDS`.
        """
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import-untyped]

        # 1. Sentence splitting
        sentences: list[str] = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", text)
            if s.strip()
        ]

        if not sentences:
            return []

        if len(sentences) == 1:
            return [
                {"text": sentences[0], "token_count": len(sentences[0].split())}
            ]

        # 2. Encode
        embeddings = self._model.encode(sentences)  # type: ignore[union-attr]

        # 3. Consecutive cosine similarities
        similarities: list[float] = []
        for i in range(len(sentences) - 1):
            sim = cosine_similarity(
                [embeddings[i]], [embeddings[i + 1]]
            )[0][0]
            similarities.append(float(sim))

        # 4 & 5. Group sentences into chunks with overlap
        chunks: list[dict] = []
        current_sentences: list[str] = [sentences[0]]
        current_word_count: int = len(sentences[0].split())

        for i, sim in enumerate(similarities):
            next_sentence = sentences[i + 1]
            next_word_count = len(next_sentence.split())

            should_split = (
                sim < self._SIMILARITY_THRESHOLD
                or (current_word_count + next_word_count) > self._MAX_CHUNK_WORDS
            )

            if should_split:
                # Finalise current chunk
                chunk_text = " ".join(current_sentences)
                chunks.append(
                    {"text": chunk_text, "token_count": len(chunk_text.split())}
                )

                # Compute overlap: carry trailing sentences ≤ _OVERLAP_WORDS
                overlap_sentences = self._compute_overlap(current_sentences)

                current_sentences = overlap_sentences + [next_sentence]
                current_word_count = sum(
                    len(s.split()) for s in current_sentences
                )
            else:
                current_sentences.append(next_sentence)
                current_word_count += next_word_count

        # Flush final chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(
                {"text": chunk_text, "token_count": len(chunk_text.split())}
            )

        return chunks

    def _compute_overlap(self, sentences: list[str]) -> list[str]:
        """Return trailing sentences from *sentences* whose combined word
        count does not exceed :pyattr:`_OVERLAP_WORDS`.

        Parameters
        ----------
        sentences:
            The ordered list of sentences in the chunk being finalised.

        Returns
        -------
        list[str]
            Sentences to carry over into the next chunk.
        """
        overlap: list[str] = []
        total_words = 0

        for sentence in reversed(sentences):
            word_count = len(sentence.split())
            if total_words + word_count > self._OVERLAP_WORDS:
                break
            overlap.insert(0, sentence)
            total_words += word_count

        return overlap

    # ------------------------------------------------------------------
    # Fallback mode
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_chunk(text: str) -> list[dict]:
        """Chunk text using ``RecursiveCharacterTextSplitter``.

        Uses character-based splitting with ``chunk_size=1200`` and
        ``chunk_overlap=200`` as an approximation of ~300 token chunks.

        Parameters
        ----------
        text:
            The full document text.

        Returns
        -------
        list[dict]
            List of ``{"text": ..., "token_count": ...}`` dicts.
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore[import-untyped]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            length_function=len,
        )
        raw_chunks: list[str] = splitter.split_text(text)

        return [
            {"text": chunk_text, "token_count": len(chunk_text.split())}
            for chunk_text in raw_chunks
        ]
