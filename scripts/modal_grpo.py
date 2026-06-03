from __future__ import annotations

import argparse
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal


REMOTE_ROOT = "/workspace/gemma-4-good"


@dataclass(frozen=True)
class ModalGrpoTrainingArgs:
    base_model_id: str
    dataset_path: str | None
    hub_dataset_id: str | None
    split: str
    output_dir: str
    hub_model_id: str | None
    gpu: str
    timeout_hours: int
    max_retries: int
    max_examples: int | None
    batch_size: int | None
    gradient_accumulation_steps: int | None
    learning_rate: float | None
    max_steps: int | None
    num_generations: int | None
    generation_batch_size: int | None
    max_prompt_length: int | None
    max_completion_length: int | None
    use_vllm: bool
    vllm_gpu_memory_utilization: float | None
    push_to_hub: bool
    detach: bool
    hf_secret: str | None
    wandb_secret: str | None


@dataclass(frozen=True)
class ModalGrpoMedMcqaEvalArgs:
    model_id: str
    hub_dataset_id: str
    split: str
    gpu: str
    timeout_hours: int
    max_retries: int
    start_index: int
    max_samples: int | None
    batch_size: int
    max_new_tokens: int
    min_new_tokens: int
    temperature: float
    sample_output_limit: int
    output_root: str
    detach: bool
    hf_secret: str | None


@dataclass(frozen=True)
class ModalHfPolicyMergeArgs:
    base_model_id: str
    adapter_model_id: str
    output_dir: str
    hub_model_id: str
    dtype: str
    max_shard_size: str
    private: bool
    overwrite: bool
    safe_merge: bool
    missing_target_policy: str
    gpu: str
    timeout_hours: int
    max_retries: int
    detach: bool
    hf_secret: str | None


def _clean_hub_id(value: str) -> str:
    return "".join(value.split())


def build_grpo_training_command(plan: ModalGrpoTrainingArgs) -> list[str]:
    command = [
        "python",
        "scripts/train_grpo.py",
        "--base-model-id",
        _clean_hub_id(plan.base_model_id),
        "--split",
        plan.split,
        "--output-dir",
        plan.output_dir,
        "--execute",
    ]
    if plan.dataset_path is not None:
        command.extend(["--dataset-path", plan.dataset_path])
    if plan.hub_dataset_id is not None:
        command.extend(["--hub-dataset-id", _clean_hub_id(plan.hub_dataset_id)])
    if plan.hub_model_id is not None:
        command.extend(["--hub-model-id", _clean_hub_id(plan.hub_model_id)])
    for flag, value in (
        ("--max-examples", plan.max_examples),
        ("--batch-size", plan.batch_size),
        ("--gradient-accumulation-steps", plan.gradient_accumulation_steps),
        ("--learning-rate", plan.learning_rate),
        ("--max-steps", plan.max_steps),
        ("--num-generations", plan.num_generations),
        ("--generation-batch-size", plan.generation_batch_size),
        ("--max-prompt-length", plan.max_prompt_length),
        ("--max-completion-length", plan.max_completion_length),
    ):
        if value is not None:
            command.extend([flag, str(value)])
    command.append("--use-vllm" if plan.use_vllm else "--no-use-vllm")
    if plan.vllm_gpu_memory_utilization is not None:
        command.extend(["--vllm-gpu-memory-utilization", str(plan.vllm_gpu_memory_utilization)])
    if plan.push_to_hub:
        command.append("--push-to-hub")
    return command


def build_grpo_medmcqa_eval_command(plan: ModalGrpoMedMcqaEvalArgs) -> list[str]:
    model_id = _clean_hub_id(plan.model_id)
    hub_dataset_id = _clean_hub_id(plan.hub_dataset_id)
    slug = model_id.replace("/", "__")
    slice_suffix = f"start-{plan.start_index}" + (f"-n-{plan.max_samples}" if plan.max_samples is not None else "")
    command = [
        "python",
        "scripts/eval_grpo_medmcqa.py",
        "--model-id",
        model_id,
        "--hub-dataset-id",
        hub_dataset_id,
        "--split",
        plan.split,
        "--start-index",
        str(plan.start_index),
        "--batch-size",
        str(plan.batch_size),
        "--max-new-tokens",
        str(plan.max_new_tokens),
        "--min-new-tokens",
        str(plan.min_new_tokens),
        "--temperature",
        str(plan.temperature),
        "--sample-output-limit",
        str(plan.sample_output_limit),
        "--output-json",
        f"{plan.output_root}/{slug}-{plan.split}-{slice_suffix}-summary.json",
        "--output-jsonl",
        f"{plan.output_root}/{slug}-{plan.split}-{slice_suffix}-predictions.jsonl",
    ]
    if plan.max_samples is not None:
        command.extend(["--max-samples", str(plan.max_samples)])
    return command


def build_hf_policy_merge_command(plan: ModalHfPolicyMergeArgs) -> list[str]:
    command = [
        "python",
        "scripts/hf_merge_policy_adapter.py",
        "--base-model-id",
        _clean_hub_id(plan.base_model_id),
        "--adapter-model-id",
        _clean_hub_id(plan.adapter_model_id),
        "--output-dir",
        plan.output_dir,
        "--hub-model-id",
        _clean_hub_id(plan.hub_model_id),
        "--dtype",
        plan.dtype,
        "--max-shard-size",
        plan.max_shard_size,
        "--missing-target-policy",
        plan.missing_target_policy,
        "--execute",
    ]
    if plan.private:
        command.append("--private")
    command.append("--overwrite" if plan.overwrite else "--no-overwrite")
    command.append("--safe-merge" if plan.safe_merge else "--no-safe-merge")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GRPO train/eval/merge jobs on Modal.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_train_parser(subparsers.add_parser("train"))
    _add_eval_parser(subparsers.add_parser("eval"))
    _add_merge_parser(subparsers.add_parser("merge"))
    args = parser.parse_args()
    if args.command == "train":
        plan = _train_plan(args)
        _dry_or_run(args.execute, "grpo-train", plan, build_grpo_training_command, "unsloth")
    elif args.command == "eval":
        plan = _eval_plan(args)
        _dry_or_run(args.execute, "grpo-medmcqa-eval", plan, build_grpo_medmcqa_eval_command, "unsloth")
    elif args.command == "merge":
        plan = _merge_plan(args)
        _dry_or_run(args.execute, "hf-policy-merge", plan, build_hf_policy_merge_command, "hf")


def _add_shared_modal_args(parser: argparse.ArgumentParser, *, timeout_hours: int, max_retries: int) -> None:
    parser.add_argument("--gpu", default="H100")
    parser.add_argument("--timeout-hours", type=int, default=timeout_hours)
    parser.add_argument("--max-retries", type=int, default=max_retries)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--hf-secret")
    parser.add_argument("--execute", action="store_true")


def _add_train_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-model-id", default="RohithMidigudla/gemma-health-telugu-medical-mix-h1-30-h2-70-lora")
    parser.add_argument("--dataset-path")
    parser.add_argument("--hub-dataset-id")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="/checkpoints/adapters/grpo_policy_v3")
    parser.add_argument("--hub-model-id", default="RohithMidigudla/gemma-health-telugu-medical-grpo-policy-v3")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--generation-batch-size", type=int, default=64)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=128)
    parser.add_argument("--use-vllm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--wandb-secret")
    _add_shared_modal_args(parser, timeout_hours=8, max_retries=1)


def _add_eval_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--hub-dataset-id", default="RohithMidigudla/gemma-health-synthetic-telugu-medmcqa-grpo")
    parser.add_argument("--split", default="test")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sample-output-limit", type=int, default=0)
    parser.add_argument("--output-root", default="/checkpoints/evals/grpo_medmcqa")
    _add_shared_modal_args(parser, timeout_hours=6, max_retries=1)


def _add_merge_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-model-id", default="google/gemma-4-E4B-it")
    parser.add_argument("--adapter-model-id", default="RohithMidigudla/gemma-health-telugu-medical-grpo-policy-v3")
    parser.add_argument("--output-dir", default="/checkpoints/merged_models/telugu_medical_grpo_v3_hf")
    parser.add_argument("--hub-model-id", default="RohithMidigudla/gemma-health-telugu-medical-grpo-v3-hf-merged-test")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32", "auto"], default="bfloat16")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--safe-merge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--missing-target-policy", choices=["error", "warn"], default="warn")
    _add_shared_modal_args(parser, timeout_hours=4, max_retries=0)


def _train_plan(args: argparse.Namespace) -> ModalGrpoTrainingArgs:
    if args.dataset_path and args.hub_dataset_id:
        raise ValueError("Use only one of --dataset-path or --hub-dataset-id")
    if args.use_vllm:
        raise ValueError("--use-vllm is disabled for this Gemma4/Unsloth image; use --no-use-vllm")
    return ModalGrpoTrainingArgs(
        base_model_id=args.base_model_id,
        dataset_path=args.dataset_path or (None if args.hub_dataset_id else "/checkpoints/datasets/synthetic_telugu/medmcqa/train.parquet"),
        hub_dataset_id=args.hub_dataset_id,
        split=args.split,
        output_dir=args.output_dir,
        hub_model_id=args.hub_model_id,
        gpu=args.gpu,
        timeout_hours=args.timeout_hours,
        max_retries=args.max_retries,
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
        detach=args.detach,
        hf_secret=args.hf_secret,
        wandb_secret=args.wandb_secret,
    )


def _eval_plan(args: argparse.Namespace) -> ModalGrpoMedMcqaEvalArgs:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be at least 1")
    return ModalGrpoMedMcqaEvalArgs(
        model_id=args.model_id,
        hub_dataset_id=args.hub_dataset_id,
        split=args.split,
        gpu=args.gpu,
        timeout_hours=args.timeout_hours,
        max_retries=args.max_retries,
        start_index=args.start_index,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
        sample_output_limit=args.sample_output_limit,
        output_root=args.output_root,
        detach=args.detach,
        hf_secret=args.hf_secret,
    )


def _merge_plan(args: argparse.Namespace) -> ModalHfPolicyMergeArgs:
    return ModalHfPolicyMergeArgs(
        base_model_id=args.base_model_id,
        adapter_model_id=args.adapter_model_id,
        output_dir=args.output_dir,
        hub_model_id=args.hub_model_id,
        dtype=args.dtype,
        max_shard_size=args.max_shard_size,
        private=args.private,
        overwrite=args.overwrite,
        safe_merge=args.safe_merge,
        missing_target_policy=args.missing_target_policy,
        gpu=args.gpu,
        timeout_hours=args.timeout_hours,
        max_retries=args.max_retries,
        detach=args.detach,
        hf_secret=args.hf_secret,
    )


def _dry_or_run(
    execute: bool,
    app_suffix: str,
    plan: Any,
    command_builder: Callable[[Any], list[str]],
    image_kind: Literal["unsloth", "hf"],
) -> None:
    if not execute:
        print("dry run only; pass --execute to launch Modal")
        print(command_builder(plan))
        return
    _run_modal(app_suffix, plan, command_builder, image_kind)


def _run_modal(
    app_suffix: str,
    plan: Any,
    command_builder: Callable[[Any], list[str]],
    image_kind: Literal["unsloth", "hf"],
) -> None:
    try:
        import modal
    except ImportError as err:
        raise RuntimeError("Install modal first: uv add modal") from err

    app = modal.App(f"gemma-health-{app_suffix}")
    model_cache_volume = modal.Volume.from_name("gemma-health-unsloth-model-cache", create_if_missing=True)
    checkpoint_volume = modal.Volume.from_name("gemma-health-unsloth-checkpoints", create_if_missing=True)
    volumes = {"/model_cache": model_cache_volume, "/checkpoints": checkpoint_volume}
    if image_kind == "unsloth":
        volumes["/dataset_cache"] = modal.Volume.from_name("gemma-health-unsloth-dataset-cache", create_if_missing=True)

    image = _modal_image(image_kind, use_vllm=bool(getattr(plan, "use_vllm", False)))
    secrets = []
    if getattr(plan, "hf_secret", None):
        secrets.append(modal.Secret.from_name(plan.hf_secret))
    if getattr(plan, "wandb_secret", None):
        secrets.append(modal.Secret.from_name(plan.wandb_secret))

    @app.function(
        image=image,
        gpu=plan.gpu,
        volumes=volumes,
        secrets=secrets,
        timeout=plan.timeout_hours * 60 * 60,
        retries=modal.Retries(initial_delay=0.0, max_retries=plan.max_retries),
        single_use_containers=True,
        serialized=True,
    )
    def run_remote(remote_plan: Any) -> None:
        command = command_builder(remote_plan)
        result = subprocess.run(command, cwd=REMOTE_ROOT, text=True)
        checkpoint_volume.commit()
        if result.returncode != 0:
            raise RuntimeError(f"remote {app_suffix} command failed: command={command!r}")

    with modal.enable_output():
        with app.run(detach=plan.detach):
            if plan.detach:
                call = run_remote.spawn(plan)
                print(f"spawned detached {app_suffix} call: {call.object_id}")
            else:
                run_remote.remote(plan)


def _modal_image(image_kind: Literal["unsloth", "hf"], *, use_vllm: bool) -> Any:
    import modal

    if image_kind == "hf":
        packages = (
            "accelerate>=1.9.0",
            "huggingface_hub>=0.34.2",
            "safetensors>=0.4.5",
            "sentencepiece>=0.2.0",
            "transformers>=5.5.0",
        )
        commands = (
            "pip install --upgrade pip 'setuptools>=80' wheel",
            "pip install --force-reinstall torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128",
        )
    else:
        packages = (
            "accelerate==1.9.0",
            "datasets==3.6.0",
            "hf-transfer==0.1.9",
            "huggingface_hub==0.34.2",
            "pandas>=2.2.0",
            "peft==0.16.0",
            "pyarrow>=16.0.0",
            "pyyaml>=6.0.2",
            "safetensors>=0.4.5",
            "tqdm>=4.67.0",
            "transformers==4.54.0",
            "wandb==0.26.1",
            *(("vllm>=0.10.0",) if use_vllm else ()),
        )
        commands = (
            "pip install --upgrade pip 'setuptools>=80' wheel",
            "pip install --force-reinstall torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128",
            "pip install --no-deps git+https://github.com/unslothai/unsloth-zoo.git",
            "rm -rf /tmp/unsloth && git clone https://github.com/unslothai/unsloth.git /tmp/unsloth",
            "python -c \"from pathlib import Path; p=Path('/tmp/unsloth/pyproject.toml'); s=p.read_text(); p.write_text(s.replace('license = \\\\\\\"Apache-2.0\\\\\\\"', 'license = { text = \\\\\\\"Apache-2.0\\\\\\\" }'))\"",
            'pip install "/tmp/unsloth[cu128-torch270]" --no-build-isolation',
        )
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git")
        .uv_pip_install(*packages)
        .run_commands(*commands)
        .env(
            {
                "HF_HOME": "/model_cache",
                "HF_XET_HIGH_PERFORMANCE": "1",
                "PYTHONPATH": f"{REMOTE_ROOT}/src",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            }
        )
        .add_local_dir("src", remote_path=f"{REMOTE_ROOT}/src")
        .add_local_dir("scripts", remote_path=f"{REMOTE_ROOT}/scripts")
        .add_local_file("config.yaml", remote_path=f"{REMOTE_ROOT}/config.yaml")
    )


if __name__ == "__main__":
    main()
