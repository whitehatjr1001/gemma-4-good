from __future__ import annotations

from gemma_health.config import AppConfig


def run_scaling_sweep(config: AppConfig) -> None:
    sizes = config.raw.get("scaling", {}).get("probe_sizes", [])
    raise NotImplementedError(f"Scaling sweep is not implemented yet for sizes: {sizes}")
