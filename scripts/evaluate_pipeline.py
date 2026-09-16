#!/usr/bin/env python
"""
RAGAS Pipeline Evaluation Script
=================================
Evaluates the profile-match and project-match pipelines using RAGAS metrics.

Metrics (all reference-free - no ground truth required):
  faithfulness                      Gemini output grounded in retrieved context?
  answer_relevancy                  Gemini output relevant to the JD?
  ContextPrecisionWithoutReference     Retrieved context useful for answering the JD?

Usage:
  # Evaluate saved production samples (from data/eval_samples.jsonl)
  python scripts/evaluate_pipeline.py

  # Evaluate predefined test cases only (runs the full pipeline)
  python scripts/evaluate_pipeline.py --predefined

  # Evaluate both saved + predefined
  python scripts/evaluate_pipeline.py --all

  # Evaluate only the last N saved samples
  python scripts/evaluate_pipeline.py --last 10

  # Filter by pipeline type
  python scripts/evaluate_pipeline.py --pipeline profile_match
  python scripts/evaluate_pipeline.py --pipeline project_match
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.config import settings
from src.infrastructure.evaluation.ragas_evaluator import RagasEvaluator, EvaluationReport

_EVAL_FILE = Path("data/eval_samples.jsonl")
_TEST_CASES_FILE = Path("scripts/eval_test_cases.json")
_REPORTS_DIR = Path("data")


# ─── Sample loading ───────────────────────────────────────────────────────────

def load_saved_samples(pipeline: str | None = None, last_n: int | None = None) -> list[dict]:
    """Load samples previously captured from live API calls."""
    if not _EVAL_FILE.exists():
        print(f"[INFO] No saved samples found at {_EVAL_FILE}")
        print("[INFO] Make a profile or project match API call first to populate samples.")
        return []

    with open(_EVAL_FILE, "r", encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    samples = []
    for ln in lines:
        try:
            samples.append(json.loads(ln))
        except json.JSONDecodeError:
            continue

    if pipeline:
        samples = [s for s in samples if s.get("pipeline") == pipeline]

    if last_n:
        samples = samples[-last_n:]

    return samples


async def run_predefined_samples(pipeline_filter: str | None = None) -> list[dict]:
    """Run predefined test JDs through the live pipeline and collect samples."""
    if not _TEST_CASES_FILE.exists():
        print(f"[WARN] Test cases file not found: {_TEST_CASES_FILE}")
        return []

    with open(_TEST_CASES_FILE, "r", encoding="utf-8") as fh:
        test_cases = json.load(fh)

    if pipeline_filter:
        test_cases = [tc for tc in test_cases if tc.get("pipeline") == pipeline_filter]

    if not test_cases:
        return []

    # Import pipeline dependencies
    from src.infrastructure.ai.gemini_embedding_adapter import GeminiEmbeddingAdapter
    from src.infrastructure.db.qdrant.qdrant_vector_store_adapter import QdrantVectorStoreAdapter
    from src.infrastructure.ai.gemini_synthesizer_adapter import GeminiSynthesizerAdapter
    from src.infrastructure.metrics.metrics_repository import MetricsRepository
    from src.application.use_cases.match_profiles import MatchProfilesUseCase
    from src.application.use_cases.match_projects import MatchProjectsUseCase
    from src.application.dto.profile_dto import ProfileMatchRequestDTO
    from src.application.dto.project_dto import ProjectMatchRequestDTO
    from src.infrastructure.evaluation import pipeline_tracer as _tracer

    embedding = GeminiEmbeddingAdapter()
    vector_store = QdrantVectorStoreAdapter()
    metrics = MetricsRepository()
    synthesizer = GeminiSynthesizerAdapter(metrics_repository=metrics)

    profile_uc = MatchProfilesUseCase(embedding, vector_store, synthesizer)
    project_uc = MatchProjectsUseCase(embedding, vector_store, synthesizer)

    samples_before = _count_samples()
    print(f"[INFO] Running {len(test_cases)} predefined test case(s)...")

    for i, tc in enumerate(test_cases, 1):
        name = tc.get("name", f"case-{i}")
        pipe = tc.get("pipeline", "unknown")
        jd = tc.get("job_details", "")
        print(f"  [{i}/{len(test_cases)}] {pipe}: {name} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            if pipe == "profile_match":
                await profile_uc.execute(
                    ProfileMatchRequestDTO(job_details=jd, user_id="eval_script")
                )
            elif pipe == "project_match":
                await project_uc.execute(
                    ProjectMatchRequestDTO(job_details=jd, user_id="eval_script")
                )
            elapsed = time.perf_counter() - t0
            print(f"done ({elapsed:.1f}s)")
        except Exception as exc:
            print(f"FAILED: {exc}")

    # Wait briefly for background capture tasks to flush
    await asyncio.sleep(1.5)

    samples_after = _count_samples()
    new_count = samples_after - samples_before
    print(f"[INFO] {new_count} new sample(s) captured from predefined test cases.")
    return load_saved_samples(pipeline=pipeline_filter, last_n=new_count if new_count > 0 else None)


def _count_samples() -> int:
    if not _EVAL_FILE.exists():
        return 0
    with open(_EVAL_FILE, "r", encoding="utf-8") as fh:
        return sum(1 for ln in fh if ln.strip())


# ─── Reporting ────────────────────────────────────────────────────────────────

def save_report(reports: list[EvaluationReport]) -> Path:
    ts = time.strftime("%Y-%m-%d_%H%M%S")
    report_path = _REPORTS_DIR / f"ragas_report_{ts}.json"
    _REPORTS_DIR.mkdir(exist_ok=True)

    payload = {
        "generated_at": ts,
        "reports": [
            {
                "pipeline": r.pipeline,
                "samples_evaluated": r.samples_evaluated,
                "aggregates": r.aggregates,
                "per_sample_scores": r.scores,
            }
            for r in reports
        ],
    }
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return report_path


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS pipeline evaluation")
    parser.add_argument("--predefined", action="store_true",
                        help="Run predefined test cases through the live pipeline")
    parser.add_argument("--all", action="store_true",
                        help="Use both saved samples and predefined test cases")
    parser.add_argument("--last", type=int, metavar="N",
                        help="Only use the last N saved samples")
    parser.add_argument("--pipeline", choices=["profile_match", "project_match"],
                        help="Filter by pipeline type")
    args = parser.parse_args()

    print("\n========================================")
    print("  RAGAS Pipeline Evaluation")
    print("========================================")

    samples: list[dict] = []

    if args.predefined or args.all:
        predefined = await run_predefined_samples(pipeline_filter=args.pipeline)
        # --all loads the saved file below, which already includes samples
        # just captured by the predefined run. Adding them here would score
        # each new sample twice.
        if not args.all:
            samples.extend(predefined)

    if not args.predefined or args.all:
        saved = load_saved_samples(pipeline=args.pipeline, last_n=args.last)
        samples.extend(saved)

    if not samples:
        print("\n[WARN] No samples to evaluate.")
        print("  → Run a profile or project match API call, then retry.")
        print("  → Or use --predefined to run test cases directly.")
        return

    print(f"\n[INFO] Total samples to evaluate: {len(samples)}")

    # Split by pipeline for separate reports
    profile_samples = [s for s in samples if s.get("pipeline") == "profile_match"]
    project_samples = [s for s in samples if s.get("pipeline") == "project_match"]

    evaluator = RagasEvaluator()
    reports: list[EvaluationReport] = []

    if profile_samples:
        print(f"\n[INFO] Evaluating {len(profile_samples)} profile_match sample(s)...")
        profile_report = await evaluator.score_batch(profile_samples)
        reports.append(profile_report)
        print(profile_report.summary_table())

    if project_samples:
        print(f"\n[INFO] Evaluating {len(project_samples)} project_match sample(s)...")
        project_report = await evaluator.score_batch(project_samples)
        reports.append(project_report)
        print(project_report.summary_table())

    if reports:
        report_path = save_report(reports)
        print(f"\n[INFO] Full report saved: {report_path}")

    print("\n========================================\n")


if __name__ == "__main__":
    asyncio.run(main())
