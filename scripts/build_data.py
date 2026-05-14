from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path

from gemma_health.config import load_config
from gemma_health.data.mixture import (
    enabled_dataset_configs,
    enabled_dataset_names,
    load_training_examples,
    load_training_examples_from_sources,
)
from gemma_health.data.validation import validate_dataset_weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="data/processed/training.jsonl")
    parser.add_argument("--max-examples-per-dataset", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    validate_dataset_weights(config)
    sources = enabled_dataset_configs(config)
    if args.max_examples_per_dataset is None:
        examples = load_training_examples(config)
    else:
        if args.max_examples_per_dataset < 0:
            raise ValueError("--max-examples-per-dataset must be non-negative")
        sources = [replace(source, max_examples=args.max_examples_per_dataset) for source in sources]
        examples = load_training_examples_from_sources(sources, config.project.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")

    print("enabled datasets:", ", ".join(enabled_dataset_names(config)))
    for source in sources:
        print(f"{source.name}: {source.language_status}")
    print(f"wrote {len(examples)} examples to {output_path}")


if __name__ == "__main__":
    main()
