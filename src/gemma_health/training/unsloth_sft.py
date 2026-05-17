from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from gemma_health.config import AppConfig
from gemma_health.data.mixture import dataset_configs
from gemma_health.data.sft import DEFAULT_SYSTEM_PROMPT, training_example_to_sft
from gemma_health.datasets.base import DatasetConfig, load_dataset


LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class AdapterTrainingConfig:
    name: str
    output_dir: Path
    dataset_names: tuple[str, ...]
    sft_jsonl: Path
    test_jsonl: Path
    hub_dataset_id: str | None
    hub_model_id: str | None
    dataset_split: str
    test_dataset_split: str
    streaming: bool
    max_examples: int | None
    train_split_ratio: float
    variant_multipliers: dict[str, int]
    source_max_examples: dict[str, int | None]
    system_prompt: str


@dataclass(frozen=True)
class PreparedSftDataset:
    adapter: str
    output_path: Path
    test_path: Path
    row_count: int
    train_count: int
    test_count: int
    sources: tuple[str, ...]


@dataclass(frozen=True)
class SftTrainingResult:
    adapter: str
    output_dir: Path
    dataset_path: Path
    test_dataset_path: Path
    executed: bool


def adapter_training_config(config: AppConfig, adapter_name: str | None = None) -> AdapterTrainingConfig:
    training = _training_mapping(config)
    name = str(adapter_name or training.get("adapter", "telugu"))
    adapters = training.get("adapters")
    if not isinstance(adapters, dict) or name not in adapters:
        available = ", ".join(sorted(adapters)) if isinstance(adapters, dict) else ""
        raise ValueError(f"Unknown training adapter {name!r}. Available: {available}")
    raw_adapter = adapters[name]
    if not isinstance(raw_adapter, dict):
        raise ValueError(f"training.adapters.{name} must be a mapping")

    dataset_names = raw_adapter.get("dataset_names", [])
    if not isinstance(dataset_names, list) or not dataset_names:
        raise ValueError(f"training.adapters.{name}.dataset_names must be a non-empty list")

    return AdapterTrainingConfig(
        name=name,
        output_dir=Path(str(raw_adapter.get("output_dir", f"artifacts/adapters/{name}"))),
        dataset_names=tuple(str(dataset_name) for dataset_name in dataset_names),
        sft_jsonl=Path(str(raw_adapter.get("sft_jsonl", f"data/processed/sft/{name}.jsonl"))),
        test_jsonl=Path(str(raw_adapter.get("test_jsonl", f"data/processed/sft/{name}_test.jsonl"))),
        hub_dataset_id=_optional_str(raw_adapter.get("hub_dataset_id")),
        hub_model_id=_optional_str(raw_adapter.get("hub_model_id")),
        dataset_split=str(raw_adapter.get("dataset_split", "train")),
        test_dataset_split=str(raw_adapter.get("test_dataset_split", "test")),
        streaming=bool(raw_adapter.get("streaming", True)),
        max_examples=_optional_int(raw_adapter.get("max_examples")),
        train_split_ratio=_train_split_ratio(raw_adapter.get("train_split_ratio", 0.8)),
        variant_multipliers=_variant_multipliers(raw_adapter.get("variant_multipliers")),
        source_max_examples=_source_max_examples(raw_adapter.get("source_max_examples")),
        system_prompt=str(raw_adapter.get("system_prompt", DEFAULT_SYSTEM_PROMPT)),
    )


def prepare_sft_dataset(config: AppConfig, adapter_name: str | None = None) -> PreparedSftDataset:
    adapter = adapter_training_config(config, adapter_name)
    sources = _cap_sources_for_adapter(adapter_dataset_sources(config, adapter), adapter.max_examples)
    train_count, test_count = _write_split_sft_jsonl(
        sources,
        adapter=adapter,
        seed=config.project.seed,
    )
    return PreparedSftDataset(
        adapter=adapter.name,
        output_path=adapter.sft_jsonl,
        test_path=adapter.test_jsonl,
        row_count=train_count + test_count,
        train_count=train_count,
        test_count=test_count,
        sources=tuple(source.name for source in sources),
    )


def train_sft_adapter(config: AppConfig, adapter_name: str | None = None, *, execute: bool = False) -> SftTrainingResult:
    adapter = adapter_training_config(config, adapter_name)
    training = _training_mapping(config)
    use_existing_sft_jsonl = bool(training.get("use_existing_sft_jsonl", False))
    prepared = None if execute and (adapter.hub_dataset_id or use_existing_sft_jsonl) else prepare_sft_dataset(config, adapter.name)
    if not execute:
        assert prepared is not None
        return SftTrainingResult(
            adapter=adapter.name,
            output_dir=adapter.output_dir,
            dataset_path=prepared.output_path,
            test_dataset_path=prepared.test_path,
            executed=False,
        )

    _run_unsloth_training(config, adapter, prepared.output_path if prepared is not None else adapter.sft_jsonl)
    return SftTrainingResult(
        adapter=adapter.name,
        output_dir=adapter.output_dir,
        dataset_path=prepared.output_path if prepared is not None else adapter.sft_jsonl,
        test_dataset_path=prepared.test_path if prepared is not None else adapter.test_jsonl,
        executed=True,
    )


def adapter_dataset_sources(config: AppConfig, adapter: AdapterTrainingConfig) -> list[DatasetConfig]:
    sources_by_name = {source.name: source for source in dataset_configs(config)}
    selected: list[DatasetConfig] = []
    missing: list[str] = []
    not_ready: list[str] = []
    for dataset_name in adapter.dataset_names:
        source = sources_by_name.get(dataset_name)
        if source is None:
            missing.append(dataset_name)
            continue
        if source.language_status == "needs_translation":
            not_ready.append(dataset_name)
            continue
        if dataset_name in adapter.source_max_examples:
            source = replace(source, max_examples=adapter.source_max_examples[dataset_name])
        selected.append(source)
    if missing:
        raise ValueError(f"Adapter {adapter.name!r} references unknown datasets: {', '.join(missing)}")
    if not_ready:
        raise ValueError(
            f"Adapter {adapter.name!r} references datasets that need translation first: {', '.join(not_ready)}"
        )
    return selected


def _cap_sources_for_adapter(sources: Sequence[DatasetConfig], max_examples: int | None) -> list[DatasetConfig]:
    if max_examples is None:
        return list(sources)
    total_weight = sum(source.weight for source in sources)
    if total_weight <= 0:
        raise ValueError("Adapter dataset weights must sum to a positive value")

    capped: list[DatasetConfig] = []
    remaining = max_examples
    for index, source in enumerate(sources):
        if index == len(sources) - 1:
            allocation = remaining
        else:
            allocation = max(1, round(max_examples * (source.weight / total_weight)))
            allocation = min(allocation, remaining)
        remaining -= allocation
        capped.append(replace(source, max_examples=allocation))
    return capped


def _write_split_sft_jsonl(sources: Sequence[DatasetConfig], adapter: AdapterTrainingConfig, seed: int) -> tuple[int, int]:
    adapter.sft_jsonl.parent.mkdir(parents=True, exist_ok=True)
    adapter.test_jsonl.parent.mkdir(parents=True, exist_ok=True)
    train_count = 0
    test_count = 0
    example_index = 0
    with adapter.sft_jsonl.open("w", encoding="utf-8") as train_file:
        with adapter.test_jsonl.open("w", encoding="utf-8") as test_file:
            for source in sources:
                for example in load_dataset(source).iter_examples():
                    sft_example = training_example_to_sft(
                        example,
                        system_prompt=adapter.system_prompt,
                        variant=example.variant,
                    )
                    line = json.dumps(asdict(sft_example), ensure_ascii=False) + "\n"
                    multiplier = adapter.variant_multipliers.get(example.variant, 1)
                    if _is_train_row(example_index, adapter.train_split_ratio, seed):
                        for _ in range(multiplier):
                            if _reached_output_cap(train_count + test_count, adapter.max_examples):
                                return train_count, test_count
                            train_file.write(line)
                            train_count += 1
                    else:
                        for _ in range(multiplier):
                            if _reached_output_cap(train_count + test_count, adapter.max_examples):
                                return train_count, test_count
                            test_file.write(line)
                            test_count += 1
                    example_index += 1
    return train_count, test_count


def _reached_output_cap(count: int, max_examples: int | None) -> bool:
    return max_examples is not None and count >= max_examples


def _is_train_row(index: int, train_split_ratio: float, seed: int) -> bool:
    bucket = ((index + seed) % 10_000) / 10_000
    return bucket < train_split_ratio


def _run_unsloth_training(config: AppConfig, adapter: AdapterTrainingConfig, local_dataset_path: Path) -> None:
    try:
        import unsloth  # noqa: F401
        from datasets import load_dataset
        import torch
        from trl import SFTConfig, SFTTrainer
        from unsloth import FastLanguageModel
    except ImportError as err:
        raise RuntimeError("Install unsloth, trl, transformers, and datasets on the GPU machine before training") from err

    training = _training_mapping(config)
    lora = _lora_mapping(training)
    report_to = _report_to(training.get("report_to"))
    run_name = _run_name(training, adapter.name)
    if "wandb" in report_to:
        os.environ.setdefault("WANDB_PROJECT", str(training.get("wandb_project", "gemma-health-adapters")))
        os.environ.setdefault("WANDB_NAME", run_name)
    local_rank = _local_rank()
    device_map = "auto"
    if local_rank is not None and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device_map = {"": local_rank}

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model.base_id,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=config.model.load_in_4bit,
        load_in_8bit=config.model.load_in_8bit,
        device_map=device_map,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 16)),
        lora_dropout=float(lora.get("dropout", 0.0)),
        bias=str(lora.get("bias", "none")),
        target_modules=list(LORA_TARGET_MODULES),
        use_gradient_checkpointing=str(lora.get("use_gradient_checkpointing", "unsloth")),
        random_state=config.project.seed,
        use_rslora=bool(lora.get("use_rslora", False)),
        loftq_config=None,
    )

    if adapter.hub_dataset_id:
        dataset = load_dataset(adapter.hub_dataset_id, split=adapter.dataset_split, streaming=adapter.streaming)
        dataset = _cap_hub_dataset(dataset, adapter.max_examples)
        eval_dataset = None if bool(training.get("skip_eval", False)) else load_dataset(
            adapter.hub_dataset_id,
            split=adapter.test_dataset_split,
            streaming=adapter.streaming,
        )
    else:
        dataset = load_dataset("json", data_files=str(local_dataset_path), split="train")
        eval_dataset = None if bool(training.get("skip_eval", False)) else load_dataset(
            "json",
            data_files=str(adapter.test_jsonl),
            split="train",
        )

    max_steps = int(training.get("max_steps", -1))
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            output_dir=str(adapter.output_dir),
            dataset_text_field="text",
            max_length=config.model.max_seq_length,
            packing=bool(training.get("packing", True)),
            per_device_train_batch_size=int(training.get("batch_size", 2)),
            per_device_eval_batch_size=int(training.get("eval_batch_size", 1)),
            gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 8)),
            learning_rate=float(training.get("learning_rate", 2e-4)),
            num_train_epochs=float(training.get("epochs", 1)),
            max_steps=max_steps,
            warmup_ratio=float(training.get("warmup_ratio", 0.06)),
            weight_decay=float(training.get("weight_decay", 0.01)),
            lr_scheduler_type=str(training.get("lr_scheduler_type", "cosine")),
            optim=str(training.get("optim", "adamw_8bit")),
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            save_steps=int(training.get("save_steps", 100)),
            eval_steps=int(training.get("eval_steps", 100)),
            eval_strategy="no" if eval_dataset is None else "steps",
            save_strategy="steps",
            ddp_find_unused_parameters=False,
            logging_steps=int(training.get("logging_steps", 10)),
            report_to=report_to,
            run_name=run_name,
            seed=config.project.seed,
        ),
    )
    resume_from_checkpoint = None
    if bool(training.get("resume_from_checkpoint", True)):
        resume_from_checkpoint = _latest_checkpoint(adapter.output_dir)

    try:
        trainer.train(resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None)
        if trainer.is_world_process_zero():
            model.save_pretrained(str(adapter.output_dir))
            tokenizer.save_pretrained(str(adapter.output_dir))
        trainer.accelerator.wait_for_everyone()
        if trainer.is_world_process_zero() and bool(training.get("push_to_hub", False)):
            _push_adapter_to_hub(adapter.output_dir, adapter.hub_model_id, path_in_repo=None)
    except Exception:
        if trainer.is_world_process_zero() and bool(training.get("push_on_failure", False)):
            checkpoint_path = _latest_checkpoint(adapter.output_dir)
            if checkpoint_path is not None:
                _push_adapter_to_hub(checkpoint_path, adapter.hub_model_id, path_in_repo="last-checkpoint")
        raise


def _training_mapping(config: AppConfig) -> dict[str, Any]:
    training = config.raw.get("training", {})
    if not isinstance(training, dict):
        raise ValueError("config.training must be a mapping")
    return training


def _lora_mapping(training: dict[str, Any]) -> dict[str, Any]:
    lora = training.get("lora", {})
    if not isinstance(lora, dict):
        raise ValueError("config.training.lora must be a mapping")
    return lora


def _local_rank() -> int | None:
    raw_rank = os.environ.get("LOCAL_RANK")
    if raw_rank is None:
        return None
    try:
        return int(raw_rank)
    except ValueError as err:
        raise ValueError(f"LOCAL_RANK must be an integer, got {raw_rank!r}") from err


def _cap_hub_dataset(dataset: Any, max_examples: int | None) -> Any:
    if max_examples is None:
        return dataset
    if max_examples < 0:
        raise ValueError("max_examples must be non-negative")
    if not hasattr(dataset, "select"):
        return dataset.take(max_examples) if hasattr(dataset, "take") else dataset
    return dataset.select(range(min(max_examples, len(dataset))))


def _latest_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    checkpoints = [path for path in output_dir.glob("checkpoint-*") if path.is_dir()]
    if not checkpoints:
        return None
    return max(checkpoints, key=_checkpoint_step)


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.removeprefix("checkpoint-"))
    except ValueError:
        return -1


def _push_adapter_to_hub(folder: Path, hub_model_id: str | None, *, path_in_repo: str | None) -> None:
    if hub_model_id is None:
        raise ValueError("hub_model_id is required when push_to_hub or push_on_failure is enabled")
    try:
        from huggingface_hub import HfApi
    except ImportError as err:
        raise RuntimeError("Install huggingface_hub before pushing adapters to Hugging Face") from err

    api = HfApi()
    api.create_repo(repo_id=hub_model_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=hub_model_id,
        repo_type="model",
        folder_path=str(folder),
        path_in_repo=path_in_repo,
        ignore_patterns=["checkpoint-*", "checkpoint-*/*", "runs/*", "wandb/*"],
    )


def _report_to(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ValueError("config.training.report_to must be a string or list")


def _run_name(training: dict[str, Any], adapter_name: str) -> str:
    raw_run_name = str(training.get("run_name", "run")).strip() or "run"
    return f"{raw_run_name}-{adapter_name}"


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    number = int(value)
    if number < 0:
        raise ValueError("max_examples must be non-negative")
    return number


def _variant_multipliers(value: object) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("variant_multipliers must be a mapping")

    multipliers: dict[str, int] = {}
    for variant, raw_multiplier in value.items():
        multiplier = int(raw_multiplier)
        if multiplier < 1:
            raise ValueError(f"variant_multipliers.{variant} must be at least 1")
        multipliers[str(variant)] = multiplier
    return multipliers


def _source_max_examples(value: object) -> dict[str, int | None]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("source_max_examples must be a mapping")

    overrides: dict[str, int | None] = {}
    for source_name, raw_max_examples in value.items():
        overrides[str(source_name)] = _optional_int(raw_max_examples)
    return overrides


def _train_split_ratio(value: object) -> float:
    ratio = float(value)
    if ratio <= 0 or ratio >= 1:
        raise ValueError("train_split_ratio must be between 0 and 1")
    return ratio
