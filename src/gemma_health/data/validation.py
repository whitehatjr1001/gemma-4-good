from __future__ import annotations

from gemma_health.config import AppConfig


def validate_dataset_weights(config: AppConfig) -> None:
    datasets = config.raw.get("datasets", [])
    total = sum(float(dataset.get("weight", 0.0)) for dataset in datasets if dataset.get("enabled"))
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"Enabled dataset weights must sum to 1.0, got {total:.3f}")
