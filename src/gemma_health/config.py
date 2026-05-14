from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    seed: int
    output_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    base_id: str
    max_seq_length: int
    load_in_4bit: bool


@dataclass(frozen=True)
class AppConfig:
    project: ProjectConfig
    model: ModelConfig
    raw: dict[str, Any]


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except OSError as err:
        raise FileNotFoundError(f"Could not read config file: {config_path}") from err

    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")

    return AppConfig(
        project=ProjectConfig(
            name=str(raw["project"]["name"]),
            seed=int(raw["project"]["seed"]),
            output_dir=Path(raw["project"]["output_dir"]),
        ),
        model=ModelConfig(
            base_id=str(raw["model"]["base_id"]),
            max_seq_length=int(raw["model"]["max_seq_length"]),
            load_in_4bit=bool(raw["model"]["load_in_4bit"]),
        ),
        raw=raw,
    )
