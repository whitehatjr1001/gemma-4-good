from __future__ import annotations

import argparse

from gemma_health.config import load_config
from gemma_health.training.unsloth_grpo import grpo_policy_config, train_grpo_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GRPO LoRA policy adapter on Telugu medical QA prompts.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--base-model-id")
    parser.add_argument("--dataset-path")
    parser.add_argument("--hub-dataset-id")
    parser.add_argument("--split")
    parser.add_argument("--output-dir")
    parser.add_argument("--hub-model-id")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-generations", type=int)
    parser.add_argument("--generation-batch-size", type=int)
    parser.add_argument("--max-prompt-length", type=int)
    parser.add_argument("--max-completion-length", type=int)
    parser.add_argument("--use-vllm", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float)
    parser.add_argument("--push-to-hub", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    policy = grpo_policy_config(
        config,
        base_model_id=args.base_model_id,
        dataset_path=args.dataset_path,
        hub_dataset_id=args.hub_dataset_id,
        split=args.split,
        output_dir=args.output_dir,
        hub_model_id=args.hub_model_id,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        generation_batch_size=args.generation_batch_size,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        use_vllm=args.use_vllm,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        push_to_hub=args.push_to_hub,
    )
    result = train_grpo_policy(config, policy, execute=args.execute)
    print(result.to_json())


if __name__ == "__main__":
    main()
