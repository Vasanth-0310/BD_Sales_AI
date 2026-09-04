"""
MetricsRepository — thread-safe, file-backed counter for Gemini API usage.

Responsibilities:
- Track total API calls, tokens processed, and estimated cost (INR).
- Break down all metrics per model name (key = settings.gemini_model value).
- Persist to a local JSON file so data survives server restarts.
- Use asyncio.Lock to prevent race conditions when multiple coroutines
  finish Gemini calls at the same time.
"""

import json
import asyncio
from pathlib import Path
from typing import Any

from src.common.logger import get_logger

logger = get_logger(__name__)

# ── File location ──────────────────────────────────────────────────────────────
_METRICS_FILE = Path("data/api_metrics.json")

# ── Gemini pricing (USD per 1 million tokens) ──────────────────────────────────
# gemini-3.1-flash-lite is in the same Flash-Lite tier as gemini-1.5-flash-8b.
# Prices: Input $0.075/1M, Output $0.30/1M  (standard <= 128k context).
# Update this dict if you switch to a different model tier.
_MODEL_PRICING: dict[str, dict[str, float]] = {
    "default": {"input_per_million": 0.075, "output_per_million": 0.30},
    "gemini-3.1-flash-lite": {"input_per_million": 0.075, "output_per_million": 0.30},
    "gemini-1.5-flash": {"input_per_million": 0.075, "output_per_million": 0.30},
    "gemini-1.5-pro": {"input_per_million": 3.50, "output_per_million": 10.50},
    "gemini-2.0-flash": {"input_per_million": 0.10, "output_per_million": 0.40},
}

_USD_TO_INR = 83.0


def _get_pricing(model: str) -> dict[str, float]:
    """Return the pricing dict for the given model, falling back to default."""
    return _MODEL_PRICING.get(model, _MODEL_PRICING["default"])


def _calculate_cost_inr(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate the INR cost for a single Gemini API call."""
    pricing = _get_pricing(model)
    cost_usd = (
        (prompt_tokens * pricing["input_per_million"] / 1_000_000)
        + (completion_tokens * pricing["output_per_million"] / 1_000_000)
    )
    return round(cost_usd * _USD_TO_INR, 6)


def _empty_metrics() -> dict[str, Any]:
    """Return a fresh zero-state metrics structure."""
    return {
        "total": {
            "api_calls": 0,
            "total_tokens": 0,
            "estimated_cost_inr": 0.0,
        },
        "by_model": {},
    }


def _load_metrics() -> dict[str, Any]:
    """Load metrics from disk. Returns empty state if file does not exist."""
    if not _METRICS_FILE.exists():
        return _empty_metrics()
    try:
        with open(_METRICS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load metrics file — resetting to zero. Reason: {e}")
        return _empty_metrics()


def _save_metrics(data: dict[str, Any]) -> None:
    """Persist metrics dict to disk, creating parent directories if needed."""
    _METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


class MetricsRepository:
    """
    Singleton-safe, asyncio-compatible metrics counter.

    Usage (inject one instance into each AI adapter via main.py):
        repo = MetricsRepository()
        await repo.increment(
            model="gemini-3.1-flash-lite",
            prompt_tokens=1200,
            completion_tokens=350,
            operation="extract_job",
        )
        metrics = await repo.get_metrics()
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def increment(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        operation: str,
    ) -> None:
        """
        Atomically update the JSON counter for a completed Gemini call.

        Args:
            model:             The model name string (e.g. 'gemini-3.1-flash-lite').
            prompt_tokens:     Tokens consumed in the prompt / input.
            completion_tokens: Tokens generated in the completion / output.
            operation:         Human-readable label for the call (e.g. 'extract_job').
        """
        total_tokens = prompt_tokens + completion_tokens
        cost_inr = _calculate_cost_inr(model, prompt_tokens, completion_tokens)

        async with self._lock:
            data = _load_metrics()

            # ── Update totals ──────────────────────────────────────────────
            data["total"]["api_calls"] += 1
            data["total"]["total_tokens"] += total_tokens
            data["total"]["estimated_cost_inr"] = round(
                data["total"]["estimated_cost_inr"] + cost_inr, 6
            )

            # ── Update per-model bucket ────────────────────────────────────
            if model not in data["by_model"]:
                data["by_model"][model] = {
                    "api_calls": 0,
                    "total_tokens": 0,
                    "estimated_cost_inr": 0.0,
                }
            bucket = data["by_model"][model]
            bucket["api_calls"] += 1
            bucket["total_tokens"] += total_tokens
            bucket["estimated_cost_inr"] = round(
                bucket["estimated_cost_inr"] + cost_inr, 6
            )

            _save_metrics(data)

        logger.info(
            f"[Metrics] op={operation} | model={model} | "
            f"prompt={prompt_tokens} | completion={completion_tokens} | "
            f"total={total_tokens} | cost=\u20b9{cost_inr:.4f}"
        )

    async def get_metrics(self) -> dict[str, Any]:
        """Return the current metrics snapshot from disk."""
        async with self._lock:
            return _load_metrics()
