from __future__ import annotations

import argparse
from pathlib import Path

from gemma_health.data.synthetic_telugu import generate_synthetic_parquet, output_path_for


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="data/raw/parquet_english")
    parser.add_argument("--output-root", default="data/staged/synthetic_telugu")
    parser.add_argument("--model", default="qwen3:30b-a3b")
    parser.add_argument("--dataset", choices=["symptom_diagnosis", "medmcqa"])
    parser.add_argument("--split")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-predict", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    input_paths = _input_paths(input_root, dataset=args.dataset, split=args.split)
    if not input_paths:
        raise FileNotFoundError(f"No parquet files found under {input_root}")

    for input_path in input_paths:
        dataset_name = input_path.parent.name
        output_path = output_path_for(input_path, input_root, output_root)
        result = generate_synthetic_parquet(
            input_path=input_path,
            output_path=output_path,
            dataset_name=dataset_name,
            model=args.model,
            limit=args.limit,
            num_predict=args.num_predict,
            temperature=args.temperature,
            checkpoint_interval=args.checkpoint_interval,
            workers=args.workers,
            retries=args.retries,
            skip_existing=not args.overwrite,
        )
        print(f"{result.input_path} -> {result.output_path}: {result.row_count} rows via {result.model}")


def _input_paths(input_root: Path, dataset: str | None, split: str | None) -> list[Path]:
    if dataset is not None and split is not None:
        return [input_root / dataset / f"{split}.parquet"]
    if dataset is not None:
        return sorted((input_root / dataset).glob("*.parquet"))
    paths = sorted(input_root.glob("*/*.parquet"))
    if split is None:
        return paths
    return [path for path in paths if path.stem == split]


if __name__ == "__main__":
    main()
