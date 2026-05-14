from __future__ import annotations

import argparse
from pathlib import Path

from gemma_health.config import load_config
from gemma_health.data.mixture import dataset_configs, load_training_examples_from_sources
from gemma_health.data.sft import (
    DEFAULT_SYSTEM_PROMPT,
    ResponseColumn,
    resolve_synthetic_inputs,
    synthetic_parquet_to_sft_examples,
    training_examples_to_sft,
    validate_sft_jsonl,
    write_sft_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a TRL/Unsloth SFT JSONL dataset.")
    parser.add_argument(
        "--input",
        action="append",
        help="Input staged synthetic parquet path, directory, or glob.",
    )
    parser.add_argument("--output", default="data/processed/sft/train.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--include-config-datasets",
        action="store_true",
        help="Also include enabled native Telugu and translation-pair datasets from config.yaml.",
    )
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--include-romanised",
        action="store_true",
        help="Also emit examples using romanised_telugu as the assistant response.",
    )
    args = parser.parse_args()
    if not args.input and not args.include_config_datasets:
        raise ValueError("Provide --input, --include-config-datasets, or both")

    response_columns: tuple[ResponseColumn, ...] = (
        ("telugu", "romanised_telugu") if args.include_romanised else ("telugu",)
    )
    examples = []
    if args.include_config_datasets:
        config = load_config(args.config)
        sources = [
            source
            for source in dataset_configs(config)
            if source.enabled and source.language_status in {"native_telugu", "translation_pair"}
        ]
        config_examples = load_training_examples_from_sources(sources, config.project.seed)
        if args.limit is not None:
            config_examples = config_examples[: args.limit]
        examples.extend(training_examples_to_sft(config_examples, system_prompt=args.system_prompt, variant="config"))
    if args.input:
        input_paths = resolve_synthetic_inputs(args.input)
        examples.extend(
            synthetic_parquet_to_sft_examples(
                input_paths,
                response_columns=response_columns,
                system_prompt=args.system_prompt,
                limit=args.limit,
            )
        )
    count = write_sft_jsonl(examples, Path(args.output))
    validate_sft_jsonl(Path(args.output))

    print(f"wrote {count} SFT examples to {args.output}")
    print("format: JSONL rows with source, variant, prompt, response, messages, text")


if __name__ == "__main__":
    main()
