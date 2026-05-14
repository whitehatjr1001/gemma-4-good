from __future__ import annotations

from gemma_health.config import AppConfig
from gemma_health.data.validation import validate_dataset_weights


def run_training(config: AppConfig) -> None:
    validate_dataset_weights(config)
    method = str(config.raw["training"]["method"])
    backend = str(config.raw["training"]["backend"])
    raise NotImplementedError(f"Training is not implemented yet for {backend}/{method}.")
