from __future__ import annotations

import argparse
from collections.abc import Iterable
from itertools import islice
from pathlib import Path
from typing import Any

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Hugging Face datasets to local parquet files.")
    parser.add_argument("--dataset", action="append", required=True, help="HF dataset id. Repeatable.")
    parser.add_argument("--config", action="append", default=[], help="Specific config/subset. Repeatable.")
    parser.add_argument("--split", action="append", default=[], help="Specific split. Repeatable.")
    parser.add_argument("--all-configs", action="store_true", help="Discover and download all configs/subsets.")
    parser.add_argument("--all-splits", action="store_true", help="Discover and download all splits.")
    parser.add_argument(
        "--where",
        action="append",
        default=[],
        help="Keep only rows matching column=value. Repeatable. Example: --where language=Telugu",
    )
    parser.add_argument("--max-rows", type=int, help="Smoke mode: stream and save only the first N rows.")
    parser.add_argument("--output-dir", default="data/raw/hf")
    args = parser.parse_args()

    if args.max_rows is not None and args.max_rows < 0:
        raise ValueError("--max-rows must be non-negative")
    filters = _parse_filters(args.where)

    for dataset_id in args.dataset:
        for config_name in _configs(dataset_id, args.config, args.all_configs):
            for split_name in _splits(dataset_id, config_name, args.split, args.all_splits):
                output_path = _output_path(Path(args.output_dir), dataset_id, config_name, split_name)
                row_count = _download_parquet(
                    dataset_id=dataset_id,
                    config_name=config_name,
                    split_name=split_name,
                    output_path=output_path,
                    max_rows=args.max_rows,
                    filters=filters,
                )
                print(f"{dataset_id} config={config_name or 'default'} split={split_name}: {row_count} rows -> {output_path}")


def _parse_filters(values: list[str]) -> dict[str, str]:
    filters: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--where must be column=value, got {value!r}")
        column, expected = value.split("=", 1)
        if not column.strip() or not expected.strip():
            raise ValueError(f"--where must be column=value, got {value!r}")
        filters[column.strip()] = expected.strip()
    return filters


def _configs(dataset_id: str, requested_configs: list[str], all_configs: bool) -> list[str | None]:
    if requested_configs:
        return requested_configs
    if not all_configs:
        return [None]

    try:
        import datasets as hf_datasets
    except ImportError as err:
        raise RuntimeError("Install the 'datasets' package to discover Hugging Face configs") from err

    names = hf_datasets.get_dataset_config_names(dataset_id)
    return [None if name == "default" else name for name in names]


def _splits(dataset_id: str, config_name: str | None, requested_splits: list[str], all_splits: bool) -> list[str]:
    if requested_splits:
        return requested_splits
    if not all_splits:
        return ["train"]

    try:
        import datasets as hf_datasets
    except ImportError as err:
        raise RuntimeError("Install the 'datasets' package to discover Hugging Face splits") from err

    if config_name is None:
        return list(hf_datasets.get_dataset_split_names(dataset_id))
    return list(hf_datasets.get_dataset_split_names(dataset_id, config_name))


def _download_parquet(
    *,
    dataset_id: str,
    config_name: str | None,
    split_name: str,
    output_path: Path,
    max_rows: int | None,
    filters: dict[str, str],
) -> int:
    try:
        import datasets as hf_datasets
    except ImportError as err:
        raise RuntimeError("Install the 'datasets' package to download Hugging Face datasets") from err

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if max_rows is None:
        dataset = _load_dataset(hf_datasets, dataset_id, config_name, split_name, streaming=False)
        if filters:
            dataset = dataset.filter(lambda row: _matches_filters(row, filters))
        dataset.to_parquet(str(output_path))
        return len(dataset)

    dataset = _load_dataset(hf_datasets, dataset_id, config_name, split_name, streaming=True)
    rows = list(islice(_filtered_rows(dataset, filters), max_rows))
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    return len(rows)


def _load_dataset(hf_datasets: object, dataset_id: str, config_name: str | None, split_name: str, streaming: bool) -> object:
    if config_name is None:
        return hf_datasets.load_dataset(dataset_id, split=split_name, streaming=streaming)
    return hf_datasets.load_dataset(dataset_id, config_name, split=split_name, streaming=streaming)


def _output_path(output_dir: Path, dataset_id: str, config_name: str | None, split_name: str) -> Path:
    dataset_dir = output_dir / dataset_id.replace("/", "__")
    config_dir = config_name or "default"
    return dataset_dir / config_dir / f"{split_name}.parquet"


def _filtered_rows(rows: Iterable[dict[str, Any]], filters: dict[str, str]) -> Iterable[dict[str, Any]]:
    for row in rows:
        if _matches_filters(row, filters):
            yield row


def _matches_filters(row: dict[str, Any], filters: dict[str, str]) -> bool:
    for column, expected in filters.items():
        if str(row.get(column)) != expected:
            return False
    return True


if __name__ == "__main__":
    main()
