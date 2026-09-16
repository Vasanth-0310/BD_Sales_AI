"""
RAGAS evaluator — scores pipeline evaluation samples with Gemini as judge.

Metrics (all reference-free — no ground-truth labels required):

  faithfulness          How well is the answer grounded in the retrieved contexts?
                        Catches Gemini hallucinating skills/evidence not in profiles.

  answer_relevancy      How relevant is the answer to the question (JD)?
                        Catches generic/irrelevant match justifications.

  ContextPrecisionWithoutReference
                        How much of the retrieved context is actually useful for
                        answering the question?
                        Catches poor retrieval: wrong profiles/chunks surfaced.

Usage:
    evaluator = RagasEvaluator()
    report = await evaluator.score_batch(samples)
    print(report.summary_table())
"""
from __future__ import annotations

import os
import sys
from types import ModuleType
from dataclasses import dataclass, field
from typing import Any

from src.common.config import settings
from src.common.logger import get_logger

logger = get_logger(__name__)


def _install_ragas_vertex_compatibility() -> None:
    """Provide RAGAS's obsolete optional Vertex import when absent.

    RAGAS 0.4.3 imports ``langchain_community.chat_models.vertexai`` during
    package initialisation solely to register a legacy Vertex type. Current
    langchain-community releases removed that module. This evaluator uses
    Gemini through ``langchain-google-genai``, never Vertex, so a local marker
    type is sufficient and avoids modifying third-party files in ``.venv``.
    """
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return

    try:
        __import__(module_name)
        return
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise

    from langchain_community import chat_models

    compatibility_module = ModuleType(module_name)

    class ChatVertexAI:  # pragma: no cover - marker for RAGAS's unused backend
        pass

    compatibility_module.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = compatibility_module
    setattr(chat_models, "vertexai", compatibility_module)


def _disable_dead_loopback_proxy() -> None:
    """Ignore only the known unusable local proxy for offline evaluations.

    A proxy at 127.0.0.1:9 is a common sandbox placeholder, not a running
    proxy. Leaving it in the process environment makes every Gemini judge
    request fail before it reaches Google. Real proxy values are preserved.
    """
    proxy_names = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
    removed = []
    for name in proxy_names:
        value = os.environ.get(name, "")
        if "127.0.0.1:9" in value or "localhost:9" in value:
            os.environ.pop(name, None)
            removed.append(name)
    if removed:
        logger.warning(
            "RagasEvaluator: ignored unreachable loopback proxy variables: %s",
            ", ".join(sorted(set(removed))),
        )


@dataclass
class EvaluationReport:
    """Holds per-sample scores and aggregated statistics."""
    pipeline: str
    samples_evaluated: int
    scores: list[dict[str, Any]] = field(default_factory=list)
    aggregates: dict[str, dict[str, float]] = field(default_factory=dict)

    def summary_table(self) -> str:
        """Return a human-readable summary table."""
        if not self.aggregates:
            return f"[{self.pipeline}] No scores available."
        lines = [
            f"\nPipeline: {self.pipeline}  ({self.samples_evaluated} samples)",
            f"{'Metric':<40} {'Min':>6} {'Max':>6} {'Mean':>6} {'Median':>8}",
            "-" * 65,
        ]
        for metric, stats in self.aggregates.items():
            lines.append(
                f"{metric:<40} "
                f"{stats.get('min', 0):.3f}  "
                f"{stats.get('max', 0):.3f}  "
                f"{stats.get('mean', 0):.3f}  "
                f"{stats.get('median', 0):.3f}"
            )
        return "\n".join(lines)


class RagasEvaluator:
    """
    Wraps RAGAS asynchronous evaluation using Gemini as judge and embeddings.

    Uses lazy import so the application server can start even if ragas is not
    installed — evaluation is an offline-only tool.
    """

    def __init__(self) -> None:
        self._api_key = settings.gemini_api_key
        if not self._api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set — required for RAGAS evaluation."
            )

    def _build_llm_and_embeddings(self):
        """Lazily configure Gemini-backed judge LLM and embeddings.

        The classic RAGAS metrics support Gemini through LangChain wrappers.
        Use base wrapper implementations directly because the public aliases
        are deprecated.
        """
        _install_ragas_vertex_compatibility()
        _disable_dead_loopback_proxy()
        try:
            from langchain_google_genai import (
                ChatGoogleGenerativeAI,
                GoogleGenerativeAIEmbeddings,
            )
            # Collection metrics require RAGAS-native dependencies when they
            # are constructed. Import base implementations, not the deprecated
            # public compatibility aliases.
            from ragas.embeddings.base import LangchainEmbeddingsWrapper
            from ragas.llms.base import LangchainLLMWrapper
        except ImportError as e:
            raise ImportError(
                f"RAGAS dependencies not installed. Run:\n"
                f"  pip install -r requirements.txt\n"
                f"Original error: {e}"
            ) from e

        judge_model = getattr(settings, "ragas_judge_model", None) or settings.gemini_model or "gemini-3.1-flash-lite"
        logger.info("RagasEvaluator: configuring judge LLM model=%s", judge_model)
        llm = LangchainLLMWrapper(
            ChatGoogleGenerativeAI(
                model=judge_model,
                google_api_key=self._api_key,
                temperature=0,
            )
        )
        embeddings = LangchainEmbeddingsWrapper(
            GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=self._api_key,
            )
        )
        return llm, embeddings

    async def score_batch(self, samples: list[dict]) -> EvaluationReport:
        """
        Evaluate a list of pipeline samples with RAGAS.

        Each sample must have at minimum:
          question           str   — the JD / input query
          retrieval_contexts list  — text strings shown to the LLM
          answer             str   — LLM output (justifications joined)
        """
        _install_ragas_vertex_compatibility()
        try:
            from ragas import aevaluate, EvaluationDataset, SingleTurnSample
            from ragas.run_config import RunConfig
            # RAGAS 0.4's collection metrics only accept its OpenAI-oriented
            # InstructorLLM. The retained classic metrics accept the Gemini
            # LangChain wrapper used by this application.
            from ragas.metrics import (
                Faithfulness,
                AnswerRelevancy,
                LLMContextPrecisionWithoutReference,
            )
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                f"RAGAS dependencies not installed. Run:\n"
                f"  pip install -r requirements.txt\n"
                f"Original error: {e}"
            ) from e

        if not samples:
            pipeline = "unknown"
            return EvaluationReport(pipeline=pipeline, samples_evaluated=0)

        pipeline = samples[0].get("pipeline", "unknown")
        logger.info(
            "RagasEvaluator: scoring %d %s samples ...", len(samples), pipeline
        )

        llm, embeddings = self._build_llm_and_embeddings()

        ragas_samples: list[SingleTurnSample] = []
        for s in samples:
            q = s.get("question", "")
            contexts = s.get("retrieval_contexts", [])
            ans = s.get("answer", "")
            if not q or not contexts or not ans:
                logger.warning("Skipping incomplete sample (missing question/contexts/answer)")
                continue
            trimmed_contexts = [
                str(context)[:settings.ragas_context_max_chars]
                for context in contexts[:settings.ragas_max_contexts]
                if str(context).strip()
            ]
            if not trimmed_contexts:
                logger.warning("Skipping sample with no usable retrieval contexts")
                continue
            # A profile-match capture may contain every intermediate candidate
            # score, while the API response exposes only the best matches.
            # Evaluate the leading result blocks so faithfulness measures the
            # user-facing answer rather than expanding dozens of low-ranked
            # candidates into hundreds of separate claims.
            answer_blocks = [
                block.strip() for block in str(ans).split("\n\n") if block.strip()
            ]
            answer_for_evaluation = "\n\n".join(answer_blocks[:5])
            trimmed_answer = answer_for_evaluation[:settings.ragas_answer_max_chars]
            if len(contexts) > len(trimmed_contexts) or len(str(ans)) > len(trimmed_answer):
                logger.info(
                    "RagasEvaluator: trimmed sample for bounded evaluation "
                    "| contexts=%d->%d answer_chars=%d->%d",
                    len(contexts), len(trimmed_contexts), len(str(ans)), len(trimmed_answer),
                )
            ragas_samples.append(
                SingleTurnSample(
                    user_input=q,
                    retrieved_contexts=trimmed_contexts,
                    response=trimmed_answer,
                )
            )

        if not ragas_samples:
            return EvaluationReport(pipeline=pipeline, samples_evaluated=0)

        dataset = EvaluationDataset(samples=ragas_samples)
        metrics = [
            Faithfulness(llm=llm),
            AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=1),
            LLMContextPrecisionWithoutReference(llm=llm),
        ]

        logger.info(
            "RagasEvaluator: running RAGAS evaluate() on %d samples — "
            "this will call Gemini as a judge (uses tokens)",
            len(ragas_samples),
        )

        result = await aevaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            run_config=RunConfig(
                timeout=settings.ragas_call_timeout_s,
                max_retries=settings.ragas_max_retries,
                max_workers=settings.ragas_max_workers,
            ),
            raise_exceptions=False,
            show_progress=False,
        )

        df = result.to_pandas()
        scores: list[dict] = df.to_dict(orient="records")

        metric_cols = [c for c in df.columns if c not in ("user_input", "retrieved_contexts", "response")]
        aggregates: dict[str, dict[str, float]] = {}
        for col in metric_cols:
            vals = df[col].dropna()
            if vals.empty:
                continue
            aggregates[col] = {
                "min": float(vals.min()),
                "max": float(vals.max()),
                "mean": float(vals.mean()),
                "median": float(vals.median()),
            }

        return EvaluationReport(
            pipeline=pipeline,
            samples_evaluated=len(ragas_samples),
            scores=scores,
            aggregates=aggregates,
        )
