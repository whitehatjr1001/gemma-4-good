from __future__ import annotations

import argparse
from pathlib import Path

from gemma_health.config import load_config
from gemma_health.datasets.raw_export import export_raw_datasets, raw_export_sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="data/raw/parquet")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--include-native-telugu", action="store_true")
    parser.add_argument("--language-status", action="append", default=[])
    parser.add_argument("--all-splits", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    sources = raw_export_sources(
        config,
        include_native_telugu=args.include_native_telugu,
        language_statuses=tuple(args.language_status),
    )
    exports = export_raw_datasets(
        sources,
        Path(args.output_dir),
        max_rows=args.max_rows,
        all_splits=args.all_splits,
    )

    for export in exports:
        print(
            f"{export.source_name}: {export.language_status}, "
            f"{export.row_count} rows -> {export.output_path}"
        )


if __name__ == "__main__":
    main()
