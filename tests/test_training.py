from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from gemma_health.config import load_config
from gemma_health.training.run import run_training
from gemma_health.training import unsloth_sft
from gemma_health.training.unsloth_sft import adapter_training_config, adapter_dataset_sources
from scripts.modal_train_adapter import ModalTrainingArgs, build_training_command


def test_adapter_training_config_loads_telugu_profile() -> None:
    config = load_config(Path("config.yaml"))

    adapter = adapter_training_config(config, "telugu")

    assert adapter.name == "telugu"
    assert adapter.output_dir == Path("artifacts/adapters/telugu")
    assert "telugu_alpaca" in adapter.dataset_names


def test_medical_adapter_uses_ready_medmcqa_source() -> None:
    config = load_config(Path("config.yaml"))
    adapter = adapter_training_config(config, "medical")

    sources = adapter_dataset_sources(config, adapter)

    assert [source.name for source in sources] == ["medmcqa"]


def test_run_training_dry_run_prepares_sft_jsonl(tmp_path: Path) -> None:
    config = load_config(Path("config.yaml"))
    training = config.raw["training"]
    training["adapters"]["telugu"]["sft_jsonl"] = str(tmp_path / "telugu.jsonl")
    training["adapters"]["telugu"]["test_jsonl"] = str(tmp_path / "telugu_test.jsonl")
    training["adapters"]["telugu"]["max_examples"] = 3

    result = run_training(config, adapter_name="telugu", execute=False)

    assert result.executed is False
    assert result.dataset_path == tmp_path / "telugu.jsonl"
    assert result.dataset_path.exists()
    assert result.test_dataset_path.exists()
    assert len(result.dataset_path.read_text(encoding="utf-8").splitlines()) == 3
    assert len(result.test_dataset_path.read_text(encoding="utf-8").splitlines()) == 0


def test_run_training_allocates_adapter_cap_by_dataset_weight(tmp_path: Path) -> None:
    config = load_config(Path("config.yaml"))
    training = config.raw["training"]
    training["adapters"]["telugu"]["sft_jsonl"] = str(tmp_path / "telugu.jsonl")
    training["adapters"]["telugu"]["test_jsonl"] = str(tmp_path / "telugu_test.jsonl")
    training["adapters"]["telugu"]["max_examples"] = 10

    result = run_training(config, adapter_name="telugu", execute=False)

    rows = result.dataset_path.read_text(encoding="utf-8").splitlines()
    test_rows = result.test_dataset_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) + len(test_rows) <= 10
    assert len(rows) + len(test_rows) >= 5


def test_execute_with_hub_dataset_skips_local_preparation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config(Path("config.yaml"))
    training = config.raw["training"]
    training["adapters"]["telugu"]["hub_dataset_id"] = "user/gemma-health-telugu-sft"
    training["adapters"]["telugu"]["sft_jsonl"] = str(tmp_path / "should_not_exist.jsonl")
    called = []

    def fake_run_training(*args: object) -> None:
        called.append(args)

    monkeypatch.setattr(unsloth_sft, "_run_unsloth_training", fake_run_training)

    result = run_training(config, adapter_name="telugu", execute=True)

    assert result.executed is True
    assert called
    assert not result.dataset_path.exists()


def test_modal_multi_gpu_command_uses_torchrun_script_entrypoint() -> None:
    command = build_training_command(
        ModalTrainingArgs(
            adapter="telugu",
            hub_dataset_id="user/dataset",
            gpu="H100:8",
            timeout_hours=6,
            max_retries=3,
            max_steps=5500,
            max_examples=None,
            batch_size=4,
            gradient_accumulation_steps=1,
            skip_eval=True,
            resume_from_checkpoint=False,
            output_root="/checkpoints/adapters/h1",
            smoke_samples=0,
            hub_model_id="user/model",
            push_to_hub=True,
            detach=True,
            hf_secret="hf-secret",
            wandb_secret="wandb-secret",
        )
    )

    assert command[:9] == [
        "torchrun",
        "--standalone",
        "--nnodes",
        "1",
        "--nproc_per_node",
        "8",
        "--tee",
        "3",
        "scripts/train.py",
    ]
    assert "python" not in command[:10]
    assert "--batch-size" in command
    assert "--gradient-accumulation-steps" in command
    assert "--skip-eval" in command
    assert "--no-resume-from-checkpoint" in command


def test_local_rank_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert unsloth_sft._local_rank() is None

    monkeypatch.setenv("LOCAL_RANK", "3")
    assert unsloth_sft._local_rank() == 3

    monkeypatch.setenv("LOCAL_RANK", "bad")
    with pytest.raises(ValueError, match="LOCAL_RANK"):
        unsloth_sft._local_rank()
