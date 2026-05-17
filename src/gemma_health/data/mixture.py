from __future__ import annotations

import random
from dataclasses import replace

from gemma_health.config import AppConfig
from gemma_health.datasets.base import DatasetConfig, load_dataset
from gemma_health.types import TrainingExample


def dataset_configs(config: AppConfig) -> list[DatasetConfig]:
    datasets = config.raw.get("datasets", [])
    if not isinstance(datasets, list):
        raise ValueError("config.datasets must be a list")
    return [DatasetConfig.from_mapping(dataset) for dataset in datasets]


def enabled_dataset_configs(config: AppConfig) -> list[DatasetConfig]:
    training = config.raw.get("training", {})
    if not isinstance(training, dict):
        raise ValueError("config.training must be a mapping")
    load_all_examples = bool(training.get("load_all_examples", False))
    total_examples = int(training.get("train_examples", 0))
    sources: list[DatasetConfig] = []
    for source in dataset_configs(config):
        if not source.enabled:
            continue
        if not load_all_examples and total_examples > 0:
            source = replace(source, max_examples=max(1, round(total_examples * source.weight)))
        sources.append(source)
    return sources


def enabled_dataset_names(config: AppConfig) -> list[str]:
    return [source.name for source in enabled_dataset_configs(config)]


def load_training_examples(config: AppConfig) -> list[TrainingExample]:
    return load_training_examples_from_sources(enabled_dataset_configs(config), config.project.seed)


def load_training_examples_from_sources(sources: list[DatasetConfig], seed: int) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    for source in sources:
        examples.extend(load_dataset(source).load())

    random.Random(seed).shuffle(examples)
    return examples
