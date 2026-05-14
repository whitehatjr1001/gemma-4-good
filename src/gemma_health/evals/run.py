from __future__ import annotations

from gemma_health.config import AppConfig


def run_evaluation(config: AppConfig) -> None:
    suites = config.raw.get("eval", {}).get("suites", [])
    raise NotImplementedError(f"Evaluation is not implemented yet for suites: {suites}")
