from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Protocol

from gemma_health.types import TrainingExample


LanguageStatus = Literal["native_telugu", "synthetic_telugu", "translation_pair", "needs_translation"]


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    enabled: bool
    weight: float
    split: str
    hf_id: str | None = None
    hf_config: str | None = None
    path: str | None = None
    max_examples: int | None = None
    language: str | None = None
    scripts: tuple[str, ...] = ()
    synthetic_telugu: bool = False
    language_status: LanguageStatus = "needs_translation"

    @classmethod
    def from_mapping(cls, value: object) -> "DatasetConfig":
        if not isinstance(value, dict):
            raise ValueError("Each dataset config entry must be a mapping")
        try:
            name = str(value["name"])
            split = str(value["split"])
        except KeyError as err:
            raise ValueError(f"Dataset config missing required field: {err.args[0]}") from err

        max_examples = value.get("max_examples")
        if max_examples is not None and int(max_examples) < 0:
            raise ValueError(f"Dataset {name!r} max_examples must be non-negative")

        return cls(
            name=name,
            enabled=bool(value.get("enabled", False)),
            weight=float(value.get("weight", 0.0)),
            split=split,
            hf_id=_optional_str(value.get("hf_id")),
            hf_config=_optional_str(value.get("hf_config")),
            path=_optional_str(value.get("path")),
            max_examples=int(max_examples) if max_examples is not None else None,
            language=_optional_str(value.get("language")),
            scripts=_string_tuple(value.get("scripts")),
            synthetic_telugu=bool(value.get("synthetic_telugu", False)),
            language_status=_language_status(value.get("language_status")),
        )


class DatasetSource(Protocol):
    name: str

    def load(self) -> list[TrainingExample]: ...

    def iter_examples(self) -> Iterator[TrainingExample]: ...


def load_dataset(source: DatasetConfig) -> DatasetSource:
    from gemma_health.datasets.loader import build_dataset

    return build_dataset(source)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("Dataset scripts must be a list when provided")
    return tuple(str(item) for item in value)


def _language_status(value: object) -> LanguageStatus:
    text = str(value or "needs_translation")
    allowed: tuple[LanguageStatus, ...] = (
        "native_telugu",
        "synthetic_telugu",
        "translation_pair",
        "needs_translation",
    )
    if text not in allowed:
        raise ValueError(f"Unsupported dataset language_status: {text}")
    return text
