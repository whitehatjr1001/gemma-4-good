from __future__ import annotations

import argparse

from gemma_health.config import load_config
from gemma_health.evals.medmcqa_generation import medmcqa_generation_eval_config, run_medmcqa_generation_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a base or GRPO adapter on raw MedMCQA generation.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model-id", help="Base model id or LoRA adapter repo id.")
    parser.add_argument("--dataset-path")
    parser.add_argument("--hub-dataset-id")
    parser.add_argument("--split", default="test")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sample-output-limit", type=int, default=0)
    parser.add_argument("--output-json")
    parser.add_argument("--output-jsonl")
    args = parser.parse_args()

    config = load_config(args.config)
    eval_config = medmcqa_generation_eval_config(
        config,
        model_id=args.model_id,
        dataset_path=args.dataset_path,
        hub_dataset_id=args.hub_dataset_id,
        split=args.split,
        start_index=args.start_index,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
        sample_output_limit=args.sample_output_limit,
        output_json=args.output_json,
        output_jsonl=args.output_jsonl,
    )
    result = run_medmcqa_generation_eval(config, eval_config)
    print(result.to_json())


if __name__ == "__main__":
    main()
