from __future__ import annotations

import argparse

from gemma_health.config import load_config
from gemma_health.training.run import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--adapter", help="Adapter profile from training.adapters, e.g. telugu")
    parser.add_argument("--hub-dataset-id", help="Override adapter Hub dataset id for streaming training.")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--sft-jsonl", help="Use an existing local train JSONL instead of preparing one.")
    parser.add_argument("--test-jsonl", help="Use an existing local eval JSONL instead of preparing one.")
    parser.add_argument("--max-steps", type=int, help="Override training.max_steps.")
    parser.add_argument("--max-examples", type=int, help="Override selected adapter max_examples.")
    parser.add_argument("--batch-size", type=int, help="Override training.batch_size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, help="Override training.gradient_accumulation_steps.")
    parser.add_argument("--skip-eval", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume-from-checkpoint", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--output-dir", help="Override selected adapter output_dir.")
    parser.add_argument("--hub-model-id", help="Override selected adapter Hub model repo id for adapter upload.")
    parser.add_argument("--push-to-hub", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run GPU training. Without this, only prepares SFT data and prints the plan.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.hub_dataset_id:
        adapter = args.adapter or str(config.raw["training"].get("adapter", "telugu"))
        config.raw["training"]["adapters"][adapter]["hub_dataset_id"] = args.hub_dataset_id
    if args.streaming is not None:
        adapter = args.adapter or str(config.raw["training"].get("adapter", "telugu"))
        config.raw["training"]["adapters"][adapter]["streaming"] = args.streaming
    if args.sft_jsonl:
        adapter = args.adapter or str(config.raw["training"].get("adapter", "telugu"))
        config.raw["training"]["adapters"][adapter]["sft_jsonl"] = args.sft_jsonl
        config.raw["training"]["use_existing_sft_jsonl"] = True
    if args.test_jsonl:
        adapter = args.adapter or str(config.raw["training"].get("adapter", "telugu"))
        config.raw["training"]["adapters"][adapter]["test_jsonl"] = args.test_jsonl
    if args.output_dir:
        adapter = args.adapter or str(config.raw["training"].get("adapter", "telugu"))
        config.raw["training"]["adapters"][adapter]["output_dir"] = args.output_dir
    if args.hub_model_id:
        adapter = args.adapter or str(config.raw["training"].get("adapter", "telugu"))
        config.raw["training"]["adapters"][adapter]["hub_model_id"] = args.hub_model_id
    if args.max_steps is not None:
        config.raw["training"]["max_steps"] = args.max_steps
    if args.max_examples is not None:
        adapter = args.adapter or str(config.raw["training"].get("adapter", "telugu"))
        config.raw["training"]["adapters"][adapter]["max_examples"] = args.max_examples
    if args.batch_size is not None:
        config.raw["training"]["batch_size"] = args.batch_size
    if args.gradient_accumulation_steps is not None:
        config.raw["training"]["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.skip_eval is not None:
        config.raw["training"]["skip_eval"] = args.skip_eval
    if args.resume_from_checkpoint is not None:
        config.raw["training"]["resume_from_checkpoint"] = args.resume_from_checkpoint
    if args.push_to_hub is not None:
        config.raw["training"]["push_to_hub"] = args.push_to_hub
    result = run_training(config, adapter_name=args.adapter, execute=args.execute)
    action = "trained" if result.executed else "prepared"
    print(f"{action} adapter={result.adapter}")
    print(f"dataset={result.dataset_path}")
    print(f"output_dir={result.output_dir}")


if __name__ == "__main__":
    main()
