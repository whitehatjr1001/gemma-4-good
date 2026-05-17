from __future__ import annotations

from gemma_health.config import AppConfig
from gemma_health.data.validation import validate_dataset_weights
from gemma_health.training.unsloth_sft import SftTrainingResult, train_sft_adapter


def run_training(config: AppConfig, adapter_name: str | None = None, *, execute: bool = False) -> SftTrainingResult:
    validate_dataset_weights(config)
    method = str(config.raw["training"]["method"])
    backend = str(config.raw["training"]["backend"])
    if backend == "unsloth" and method == "sft":
        return train_sft_adapter(config, adapter_name, execute=execute)
    raise NotImplementedError(f"Training is not implemented yet for {backend}/{method}.")
