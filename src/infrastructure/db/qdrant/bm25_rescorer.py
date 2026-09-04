"""Client-side BM25 rescoring and Reciprocal Rank Fusion (RRF) utility."""

import re
from rank_bm25 import BM25Okapi
from src.common.logger import get_logger

logger = get_logger(__name__)


class BM25Rescorer:
    """Utility for BM25-based rescoring of Qdrant retrieval candidates.

    Builds a term-frequency corpus from candidate payloads and scores
    each candidate against a tokenised query using BM25Okapi.
    """

    @staticmethod
    def rescore(
        candidates: list[dict],
        query_text: str,
        text_fields: tuple[str, ...] = ("project_name", "domain", "description"),
        list_fields: tuple[str, ...] = ("techstacks",),
        id_field: str = "project_id",
    ) -> dict[str, float]:
        """Rescore candidates using BM25Okapi.

        Args:
            candidates: List of dicts with ``payload`` containing searchable
                metadata fields.
            query_text: The raw query string.
            text_fields: Payload keys whose string values form the corpus.
            list_fields: Payload keys holding lists (joined into the corpus).
            id_field: Payload key that identifies the candidate (returned as
                the mapping key — ``project_id`` for projects,
                ``variant_id`` for profile variants).

        Returns:
            Mapping of ``<id_field value> → bm25_score``.
        """
        if not candidates:
            return {}

        # Build corpus — one document per candidate
        corpus: list[list[str]] = []
        candidate_ids: list[str] = []

        for candidate in candidates:
            payload = candidate.get("payload", {})
            doc_parts: list[str] = []
            for key in text_fields:
                value = payload.get(key, "")
                if value:
                    doc_parts.append(str(value))
            for key in list_fields:
                value = payload.get(key, [])
                if isinstance(value, list):
                    doc_parts.append(" ".join(str(v) for v in value))
                elif value:
                    doc_parts.append(str(value))

            doc_text = " ".join(doc_parts)
            tokens = _tokenize(doc_text)
            corpus.append(tokens)
            candidate_ids.append(str(payload.get(id_field, "")))

        # Tokenize query
        tokenized_query = _tokenize(query_text)

        # BM25 scoring
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(tokenized_query)

        result = {cid: float(score) for cid, score in zip(candidate_ids, scores)}

        logger.info(
            "BM25 rescore complete  |  candidates=%d  query_tokens=%d",
            len(candidates),
            len(tokenized_query),
        )
        return result


def rrf_merge(
    dense_ranking: list[tuple],
    bm25_ranking: list[tuple],
    k: int = 60,
) -> list[tuple]:
    """Reciprocal Rank Fusion (RRF) of two ranked lists.

    Combines dense (cosine similarity) and BM25 rankings into a single
    fused ranking using the formula::

        rrf_score(d) = Σ  1 / (k + rank_i(d))

    Args:
        dense_ranking: List of ``(project_id, score)`` tuples sorted by
            score descending.
        bm25_ranking: List of ``(project_id, score)`` tuples sorted by
            score descending.
        k: RRF constant (default 60, per the original paper).

    Returns:
        A fused list of ``(project_id, rrf_score)`` tuples sorted by
        rrf_score descending.
    """
    rrf_scores: dict[str, float] = {}

    # Dense ranking contribution
    for rank, (pid, _score) in enumerate(dense_ranking, start=1):
        rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k + rank)

    # BM25 ranking contribution
    for rank, (pid, _score) in enumerate(bm25_ranking, start=1):
        rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k + rank)

    # Sort by fused score descending
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    logger.info(
        "RRF merge complete  |  dense=%d  bm25=%d  fused=%d  top_score=%.6f",
        len(dense_ranking),
        len(bm25_ranking),
        len(fused),
        fused[0][1] if fused else 0.0,
    )
    return fused


def _tokenize(text: str) -> list[str]:
    """Symbol-aware tokenizer for tech terms.

    Keeps ``+``/``#``/``.`` so that C, C++, C#, .NET and Node.js stay
    distinct tokens instead of all collapsing to ``c``/``net``/``node``
    (which made a C# JD match plain-C projects in BM25 rescoring).
    Matches the tokenizer used by the Qdrant keyword search (_keyword_tokens)
    so both retrieval sides split terms identically.
    """
    tokens = re.findall(r"(?:\.NET|[A-Za-z][A-Za-z0-9+#.]*)", text)
    return [t.lower() for t in tokens]
