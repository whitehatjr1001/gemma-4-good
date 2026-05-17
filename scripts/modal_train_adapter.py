from __future__ import annotations

import argparse
from dataclasses import dataclass


APP_NAME = "gemma-health-adapter-training"
REMOTE_ROOT = "/workspace/gemma-4-good"


@dataclass(frozen=True)
class ModalTrainingArgs:
    adapter: str
    hub_dataset_id: str
    gpu: str
    timeout_hours: int
    max_retries: int
    max_steps: int | None
    max_examples: int | None
    batch_size: int | None
    gradient_accumulation_steps: int | None
    skip_eval: bool | None
    resume_from_checkpoint: bool | None
    output_root: str
    smoke_samples: int
    hub_model_id: str | None
    push_to_hub: bool
    detach: bool
    hf_secret: str | None
    wandb_secret: str | None


def build_training_command(remote_plan: ModalTrainingArgs) -> list[str]:
    command_args = [
        "--adapter",
        remote_plan.adapter,
        "--output-dir",
        f"{remote_plan.output_root}/{remote_plan.adapter}",
        "--execute",
    ]
    if remote_plan.hub_dataset_id:
        command_args.extend(["--hub-dataset-id", remote_plan.hub_dataset_id, "--no-streaming"])
    if remote_plan.smoke_samples:
        smoke_train_path = "/tmp/gemma-health-smoke/train.jsonl"
        smoke_test_path = "/tmp/gemma-health-smoke/test.jsonl"
        command_args.extend(["--sft-jsonl", smoke_train_path, "--test-jsonl", smoke_test_path])
    if remote_plan.max_steps is not None:
        command_args.extend(["--max-steps", str(remote_plan.max_steps)])
    if remote_plan.max_examples is not None:
        command_args.extend(["--max-examples", str(remote_plan.max_examples)])
    if remote_plan.batch_size is not None:
        command_args.extend(["--batch-size", str(remote_plan.batch_size)])
    if remote_plan.gradient_accumulation_steps is not None:
        command_args.extend(["--gradient-accumulation-steps", str(remote_plan.gradient_accumulation_steps)])
    if remote_plan.skip_eval is not None:
        command_args.append("--skip-eval" if remote_plan.skip_eval else "--no-skip-eval")
    if remote_plan.resume_from_checkpoint is not None:
        command_args.append("--resume-from-checkpoint" if remote_plan.resume_from_checkpoint else "--no-resume-from-checkpoint")
    if remote_plan.max_steps is not None and remote_plan.max_steps <= 1:
        command_args.append("--skip-eval")
    if remote_plan.hub_model_id:
        command_args.extend(["--hub-model-id", remote_plan.hub_model_id])
    if remote_plan.push_to_hub:
        command_args.append("--push-to-hub")

    gpu_count = _gpu_count(remote_plan.gpu)
    if gpu_count > 1:
        return [
            "torchrun",
            "--standalone",
            "--nnodes",
            "1",
            "--nproc_per_node",
            str(gpu_count),
            "--tee",
            "3",
            "scripts/train.py",
            *command_args,
        ]
    return ["python", "scripts/train.py", *command_args]


def _gpu_count(gpu: str) -> int:
    if ":" not in gpu:
        return 1
    count = gpu.rsplit(":", maxsplit=1)[1]
    try:
        return int(count)
    except ValueError as err:
        raise ValueError(f"invalid Modal gpu count in {gpu!r}") from err


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an Unsloth adapter training job on Modal.")
    parser.add_argument("--adapter", default="telugu")
    parser.add_argument("--hub-dataset-id", help="HF dataset repo id produced by scripts/upload_sft_dataset.py.")
    parser.add_argument("--gpu", default="L40S")
    parser.add_argument("--timeout-hours", type=int, default=6)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--skip-eval", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume-from-checkpoint", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--output-root", default="/checkpoints/adapters/h1_telugu_broad_probe_001")
    parser.add_argument("--smoke-samples", type=int, default=0)
    parser.add_argument("--hub-model-id")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--hf-secret", help="Optional Modal secret name containing HF_TOKEN.")
    parser.add_argument("--wandb-secret", help="Optional Modal secret name containing WANDB_API_KEY.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually launch Modal. Without this, prints the remote plan and exits.",
    )
    args = parser.parse_args()

    if args.smoke_samples < 0:
        raise ValueError("--smoke-samples must be non-negative")
    if not args.hub_dataset_id and args.smoke_samples == 0:
        print("remote Modal training expects a Hub dataset id; upload first with scripts/upload_sft_dataset.py")
        print("example: --hub-dataset-id your-user/gemma-health-telugu-sft")
        if args.execute:
            raise SystemExit(2)
    plan = ModalTrainingArgs(
        adapter=args.adapter,
        hub_dataset_id=args.hub_dataset_id or "",
        gpu=args.gpu,
        timeout_hours=args.timeout_hours,
        max_retries=args.max_retries,
        max_steps=args.max_steps,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        skip_eval=args.skip_eval,
        resume_from_checkpoint=args.resume_from_checkpoint,
        output_root=args.output_root,
        smoke_samples=args.smoke_samples,
        hub_model_id=args.hub_model_id,
        push_to_hub=args.push_to_hub,
        detach=args.detach,
        hf_secret=args.hf_secret,
        wandb_secret=args.wandb_secret,
    )

    if not args.execute:
        print("dry run only; pass --execute to launch Modal")
        print(plan)
        return

    _run_modal(plan)


def _run_modal(plan: ModalTrainingArgs) -> None:
    try:
        import modal
    except ImportError as err:
        raise RuntimeError("Install modal first: uv add modal") from err

    app = modal.App(APP_NAME)
    model_cache_volume = modal.Volume.from_name("gemma-health-unsloth-model-cache", create_if_missing=True)
    dataset_cache_volume = modal.Volume.from_name("gemma-health-unsloth-dataset-cache", create_if_missing=True)
    checkpoint_volume = modal.Volume.from_name("gemma-health-unsloth-checkpoints", create_if_missing=True)

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git")
        .uv_pip_install(
            "accelerate==1.9.0",
            "datasets==3.6.0",
            "hf-transfer==0.1.9",
            "huggingface_hub==0.34.2",
            "pandas>=2.2.0",
            "peft==0.16.0",
            "pyarrow>=16.0.0",
            "pyyaml>=6.0.2",
            "transformers==4.54.0",
            "trl==0.19.1",
            "wandb==0.26.1",
        )
        .run_commands(
            "pip install --upgrade pip 'setuptools>=80' wheel",
            "pip install --force-reinstall torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128",
            "pip install --no-deps git+https://github.com/unslothai/unsloth-zoo.git",
            "rm -rf /tmp/unsloth && git clone https://github.com/unslothai/unsloth.git /tmp/unsloth",
            "python -c \"from pathlib import Path; p=Path('/tmp/unsloth/pyproject.toml'); s=p.read_text(); p.write_text(s.replace('license = \\\"Apache-2.0\\\"', 'license = { text = \\\"Apache-2.0\\\" }'))\"",
            'pip install "/tmp/unsloth[cu128-torch270]" --no-build-isolation',
        )
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
    secrets = []
    if plan.hf_secret:
        secrets.append(modal.Secret.from_name(plan.hf_secret))
    if plan.wandb_secret:
        secrets.append(modal.Secret.from_name(plan.wandb_secret))

    @app.function(
        image=image,
        gpu=plan.gpu,
        volumes={
            "/model_cache": model_cache_volume,
            "/dataset_cache": dataset_cache_volume,
            "/checkpoints": checkpoint_volume,
        },
        secrets=secrets,
        timeout=plan.timeout_hours * 60 * 60,
        retries=modal.Retries(initial_delay=0.0, max_retries=plan.max_retries),
        single_use_containers=True,
        serialized=True,
    )
    def train_remote(remote_plan: ModalTrainingArgs) -> None:
        import subprocess
        from pathlib import Path

        smoke_train_path = Path("/tmp/gemma-health-smoke/train.jsonl")
        smoke_test_path = Path("/tmp/gemma-health-smoke/test.jsonl")
        if remote_plan.smoke_samples:
            _write_smoke_sft_jsonl(smoke_train_path, smoke_test_path, remote_plan.smoke_samples)

        command = build_training_command(remote_plan)
        result = subprocess.run(command, cwd=REMOTE_ROOT, text=True)
        checkpoint_volume.commit()
        if result.returncode != 0:
            raise RuntimeError(f"remote training command failed: command={command!r}")

    def _write_smoke_sft_jsonl(train_path: "Path", test_path: "Path", samples: int) -> None:
        import json

        train_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        prompts = [
            ("Translate to Telugu: Drink clean water.", "శుభ్రమైన నీరు తాగండి."),
            ("Answer in Telugu: What should fever patients do?", "జ్వరం ఉంటే విశ్రాంతి తీసుకోండి, నీరు తాగండి, జ్వరం ఎక్కువైతే PHC కి వెళ్లండి."),
            ("Rewrite in Telugu: Take medicine after food.", "మందు భోజనం తర్వాత తీసుకోండి."),
            ("Translate to Telugu: Go to hospital for chest pain.", "ఛాతి నొప్పి ఉంటే వెంటనే ఆసుపత్రికి వెళ్లండి."),
            ("Answer in Telugu: When is emergency care needed?", "శ్వాస తీసుకోవడంలో ఇబ్బంది, అపస్మారం, ఛాతి నొప్పి లేదా తీవ్ర రక్తస్రావం ఉంటే అత్యవసర వైద్యం అవసరం."),
        ]
        for index in range(samples):
            prompt, response = prompts[index % len(prompts)]
            messages = [
                {"role": "system", "content": "You are a careful Telugu rural health assistant."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            text = "\n".join(f"<|{message['role']}|>\n{message['content']}" for message in messages)
            rows.append(
                {
                    "source": "modal_smoke",
                    "variant": "native_telugu",
                    "prompt": prompt,
                    "response": response,
                    "messages": messages,
                    "text": text,
                }
            )
        train_rows = rows[: max(1, samples - 1)]
        test_rows = rows[max(1, samples - 1) :] or rows[-1:]
        train_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in train_rows) + "\n", encoding="utf-8")
        test_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in test_rows) + "\n", encoding="utf-8")

    with modal.enable_output():
        with app.run(detach=plan.detach):
            if plan.detach:
                call = train_remote.spawn(plan)
                print(f"spawned detached training call: {call.object_id}")
            else:
                train_remote.remote(plan)


if __name__ == "__main__":
    main()
