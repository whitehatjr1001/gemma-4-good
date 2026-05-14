from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from gemma_health.config import AppConfig
from gemma_health.data.mixture import enabled_dataset_configs
from gemma_health.datasets.base import DatasetConfig, LanguageStatus


@dataclass(frozen=True)
class RawDatasetExport:
    source_name: str
    language_status: str
    row_count: int
    output_path: Path


def raw_export_sources(
    config: AppConfig,
    include_native_telugu: bool = False,
    language_statuses: tuple[LanguageStatus, ...] = (),
) -> list[DatasetConfig]:
    sources = enabled_dataset_configs(config)
    if language_statuses:
        sources = [source for source in sources if source.language_status in language_statuses]
    if include_native_telugu:
        return sources
    return [source for source in sources if source.language_status != "native_telugu"]


def export_raw_datasets(
    sources: list[DatasetConfig],
    output_dir: Path,
    max_rows: int | None = None,
    all_splits: bool = False,
) -> list[RawDatasetExport]:
    output_dir.mkdir(parents=True, exist_ok=True)
    exports: list[RawDatasetExport] = []
    for source in sources:
        split_sources = _split_sources(source) if all_splits else [source]
        for split_source in split_sources:
            exports.append(export_raw_dataset(split_source, output_dir, max_rows=max_rows))
    return exports


def export_raw_dataset(
    source: DatasetConfig,
    output_dir: Path,
    max_rows: int | None = None,
) -> RawDatasetExport:
    dataset = _load_hf_dataset(source)
    dataset = _filter_dataset(dataset, source)
    if max_rows is not None:
        if max_rows < 0:
            raise ValueError("max_rows must be non-negative")
        dataset = dataset.select(range(min(max_rows, len(dataset))))

    source_dir = output_dir / source.name
    source_dir.mkdir(parents=True, exist_ok=True)
    output_path = source_dir / f"{source.split}.parquet"
    dataset.to_parquet(str(output_path))
    return RawDatasetExport(
        source_name=source.name,
        language_status=source.language_status,
        row_count=len(dataset),
        output_path=output_path,
    )


def _load_hf_dataset(source: DatasetConfig) -> Any:
    if source.hf_id is None:
        raise ValueError(f"Dataset {source.name!r} requires hf_id for raw parquet export")

    try:
        import datasets as hf_datasets
    except ImportError as err:
        raise RuntimeError("Install the 'datasets' package to export Hugging Face datasets") from err

    try:
        if source.hf_config is None:
            return hf_datasets.load_dataset(source.hf_id, split=source.split)
        return hf_datasets.load_dataset(source.hf_id, source.hf_config, split=source.split)
    except Exception as err:
        raise RuntimeError(
            f"Could not load Hugging Face dataset {source.hf_id!r} split {source.split!r}"
        ) from err


def _split_sources(source: DatasetConfig) -> list[DatasetConfig]:
    if source.hf_id is None:
        raise ValueError(f"Dataset {source.name!r} requires hf_id for split discovery")

    try:
        import datasets as hf_datasets
    except ImportError as err:
        raise RuntimeError("Install the 'datasets' package to discover Hugging Face splits") from err

    try:
        if source.hf_config is None:
            split_names = hf_datasets.get_dataset_split_names(source.hf_id)
        else:
            split_names = hf_datasets.get_dataset_split_names(source.hf_id, source.hf_config)
    except Exception as err:
        raise RuntimeError(f"Could not discover splits for Hugging Face dataset {source.hf_id!r}") from err

    return [replace(source, split=split_name) for split_name in split_names]


def _filter_dataset(dataset: Any, source: DatasetConfig) -> Any:
    if source.language is None and not source.scripts:
        return dataset

    def matches(row: dict[str, object]) -> bool:
        if source.language is not None and row.get("language") != source.language:
            return False
        if source.scripts and row.get("script") not in source.scripts:
            return False
        return True

    return dataset.filter(matches)
