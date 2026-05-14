from __future__ import annotations

from gemma_health.config import AppConfig


def export_for_serving(config: AppConfig) -> None:
    target = config.raw.get("serving", {}).get("target")
    runtime = config.raw.get("serving", {}).get("runtime")
    raise NotImplementedError(f"Serving export is not implemented yet for {target}/{runtime}.")
