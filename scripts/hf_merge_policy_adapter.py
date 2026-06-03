from __future__ import annotations

import argparse
from pathlib import Path

from gemma_health.training.hf_policy_merge import HfPolicyAdapterMergeConfig, merge_policy_adapter_with_hf_peft


def _clean_hub_id(value: str) -> str:
    return "".join(value.split())


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a GRPO LoRA adapter into a full model with HF Transformers + PEFT.")
    parser.add_argument("--base-model-id", default="google/gemma-4-E4B-it")
    parser.add_argument("--adapter-model-id", default="RohithMidigudla/gemma-health-telugu-medical-grpo-policy-v3")
    parser.add_argument("--output-dir", default="artifacts/merged_models/telugu_medical_grpo_v3_hf")
    parser.add_argument("--hub-model-id", default="RohithMidigudla/gemma-health-telugu-medical-grpo-v3-hf-merged-test")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32", "auto"], default="bfloat16")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--safe-merge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--missing-target-policy", choices=["error", "warn"], default="warn")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    merge_config = HfPolicyAdapterMergeConfig(
        base_model_id=_clean_hub_id(args.base_model_id),
        adapter_model_id=_clean_hub_id(args.adapter_model_id),
        output_dir=Path(args.output_dir),
        hub_model_id=_clean_hub_id(args.hub_model_id),
        private=args.private,
        overwrite=args.overwrite,
        dtype=args.dtype,
        max_shard_size=args.max_shard_size,
        safe_merge=args.safe_merge,
        missing_target_policy=args.missing_target_policy,
    )
    if not args.execute:
        print("dry run only; pass --execute to merge and publish")
        print(merge_config)
        return

    result = merge_policy_adapter_with_hf_peft(merge_config)
    print(result.to_json())


if __name__ == "__main__":
    main()
