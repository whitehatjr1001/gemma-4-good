from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Iterable
from typing import Any

from gemma_health.config import AppConfig
from gemma_health.rewards.telugu import telugu_density_reward
from gemma_health.training.unsloth_sft import LORA_TARGET_MODULES, _lora_mapping, _report_to, _training_mapping


OPTION_LABELS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class GrpoPolicyConfig:
    base_model_id: str
    dataset_path: Path | None
    hub_dataset_id: str | None
    split: str
    output_dir: Path
    hub_model_id: str | None
    max_examples: int | None
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    max_steps: int
    num_generations: int
    generation_batch_size: int | None
    max_prompt_length: int
    max_completion_length: int
    use_vllm: bool
    vllm_gpu_memory_utilization: float
    save_steps: int
    logging_steps: int
    push_to_hub: bool


@dataclass(frozen=True)
class GrpoTrainingResult:
    executed: bool
    base_model_id: str
    dataset: str
    output_dir: Path
    hub_model_id: str | None

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str, indent=2, sort_keys=True)


def grpo_policy_config(
    config: AppConfig,
    *,
    base_model_id: str | None,
    dataset_path: str | Path | None,
    hub_dataset_id: str | None,
    split: str | None,
    output_dir: str | Path | None,
    hub_model_id: str | None,
    max_examples: int | None,
    batch_size: int | None,
    gradient_accumulation_steps: int | None,
    learning_rate: float | None,
    max_steps: int | None,
    num_generations: int | None,
    generation_batch_size: int | None,
    max_prompt_length: int | None,
    max_completion_length: int | None,
    use_vllm: bool | None,
    vllm_gpu_memory_utilization: float | None,
    push_to_hub: bool | None,
) -> GrpoPolicyConfig:
    grpo = _grpo_mapping(config)
    if dataset_path is not None and hub_dataset_id is not None:
        raise ValueError("Use only one of --dataset-path or --hub-dataset-id")
    if hub_dataset_id is not None:
        resolved_dataset_path = None
        resolved_hub_dataset_id = hub_dataset_id
    elif dataset_path is not None:
        resolved_dataset_path = dataset_path
        resolved_hub_dataset_id = None
    else:
        resolved_dataset_path = grpo.get("dataset_path")
        resolved_hub_dataset_id = grpo.get("hub_dataset_id")
    if resolved_dataset_path is None and resolved_hub_dataset_id is None:
        raise ValueError("GRPO requires --dataset-path or --hub-dataset-id")
    if resolved_dataset_path is not None and resolved_hub_dataset_id is not None:
        raise ValueError("Use only one of --dataset-path or --hub-dataset-id")

    resolved_hub_model_id = hub_model_id if hub_model_id is not None else _optional_str(grpo.get("hub_model_id"))
    resolved_push = bool(grpo.get("push_to_hub", False)) if push_to_hub is None else push_to_hub
    if resolved_push and resolved_hub_model_id is None:
        raise ValueError("--hub-model-id is required when --push-to-hub is enabled")

    return GrpoPolicyConfig(
        base_model_id=base_model_id or str(grpo.get("base_model_id", config.model.base_id)),
        dataset_path=Path(resolved_dataset_path) if resolved_dataset_path is not None else None,
        hub_dataset_id=str(resolved_hub_dataset_id) if resolved_hub_dataset_id is not None else None,
        split=split or str(grpo.get("split", "train")),
        output_dir=Path(output_dir or grpo.get("output_dir", "artifacts/adapters/grpo_policy")),
        hub_model_id=resolved_hub_model_id,
        max_examples=max_examples if max_examples is not None else _optional_int(grpo.get("max_examples")),
        batch_size=batch_size or int(grpo.get("batch_size", 1)),
        gradient_accumulation_steps=gradient_accumulation_steps or int(grpo.get("gradient_accumulation_steps", 8)),
        learning_rate=learning_rate or float(grpo.get("learning_rate", 5e-6)),
        max_steps=max_steps if max_steps is not None else int(grpo.get("max_steps", 300)),
        num_generations=num_generations or int(grpo.get("num_generations", grpo.get("group_size", 4))),
        generation_batch_size=generation_batch_size if generation_batch_size is not None else _optional_int(grpo.get("generation_batch_size")),
        max_prompt_length=max_prompt_length or int(grpo.get("max_prompt_length", 1024)),
        max_completion_length=max_completion_length or int(grpo.get("max_completion_length", 512)),
        use_vllm=bool(grpo.get("use_vllm", False)) if use_vllm is None else use_vllm,
        vllm_gpu_memory_utilization=(
            float(grpo.get("vllm_gpu_memory_utilization", 0.3))
            if vllm_gpu_memory_utilization is None
            else vllm_gpu_memory_utilization
        ),
        save_steps=int(grpo.get("save_steps", 100)),
        logging_steps=int(grpo.get("logging_steps", 5)),
        push_to_hub=resolved_push,
    )


def train_grpo_policy(config: AppConfig, policy: GrpoPolicyConfig, *, execute: bool) -> GrpoTrainingResult:
    dataset_label = policy.hub_dataset_id or str(policy.dataset_path)
    if not execute:
        return GrpoTrainingResult(
            executed=False,
            base_model_id=policy.base_model_id,
            dataset=dataset_label,
            output_dir=policy.output_dir,
            hub_model_id=policy.hub_model_id,
        )

    try:
        import unsloth  # noqa: F401
        import torch
        from datasets import Dataset, load_dataset
        from trl import GRPOConfig, GRPOTrainer
        from unsloth import FastLanguageModel
    except ImportError as err:
        raise RuntimeError("Install unsloth, trl, torch, and datasets before GRPO training") from err

    training = _training_mapping(config)
    lora = _lora_mapping(training)
    report_to = _report_to(training.get("report_to"))
    if "wandb" in report_to:
        import os

        os.environ.setdefault("WANDB_PROJECT", str(training.get("wandb_project", "gemma-health-adapters")))
        os.environ.setdefault("WANDB_NAME", f"{training.get('run_name', 'run')}-grpo-policy")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=policy.base_model_id,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=config.model.load_in_4bit,
        load_in_8bit=config.model.load_in_8bit,
        device_map="auto",
        fast_inference=policy.use_vllm,
    )
    model = prepare_grpo_peft_model(
        fast_language_model=FastLanguageModel,
        model=model,
        lora=lora,
        seed=config.project.seed,
    )
    raw_dataset = (
        load_dataset(policy.hub_dataset_id, split=policy.split)
        if policy.hub_dataset_id is not None
        else load_dataset("parquet", data_files=str(policy.dataset_path), split="train")
    )
    if policy.max_examples is not None:
        raw_dataset = raw_dataset.select(range(min(policy.max_examples, len(raw_dataset))))
    train_rows = [grpo_row_from_medmcqa(row) for row in raw_dataset if is_usable_grpo_source_row(row)]
    if not train_rows:
        raise ValueError("GRPO dataset has no usable rows after validation")
    train_dataset = Dataset.from_list(train_rows)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            medical_correctness_reward,
            telugu_script_reward,
            reference_overlap_reward,
            concise_medical_answer_reward,
            medical_guardrail_reward,
        ],
        args=GRPOConfig(
            output_dir=str(policy.output_dir),
            per_device_train_batch_size=policy.batch_size,
            gradient_accumulation_steps=policy.gradient_accumulation_steps,
            learning_rate=policy.learning_rate,
            max_steps=policy.max_steps,
            num_generations=policy.num_generations,
            generation_batch_size=policy.generation_batch_size,
            max_prompt_length=policy.max_prompt_length,
            max_completion_length=policy.max_completion_length,
            use_vllm=policy.use_vllm,
            vllm_gpu_memory_utilization=policy.vllm_gpu_memory_utilization,
            save_steps=policy.save_steps,
            logging_steps=policy.logging_steps,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            report_to=report_to,
            seed=config.project.seed,
        ),
        train_dataset=train_dataset,
    )
    trainer.train()
    model.save_pretrained(str(policy.output_dir))
    tokenizer.save_pretrained(str(policy.output_dir))
    if policy.push_to_hub:
        _push_grpo_adapter(policy.output_dir, policy.hub_model_id)

    return GrpoTrainingResult(
        executed=True,
        base_model_id=policy.base_model_id,
        dataset=dataset_label,
        output_dir=policy.output_dir,
        hub_model_id=policy.hub_model_id,
    )


def prepare_grpo_peft_model(
    *,
    fast_language_model: Any,
    model: Any,
    lora: dict[str, Any],
    seed: int,
) -> Any:
    if _has_loaded_peft_adapter(model):
        if hasattr(fast_language_model, "for_training"):
            fast_language_model.for_training(model)
        return model
    return fast_language_model.get_peft_model(
        model,
        r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 16)),
        lora_dropout=float(lora.get("dropout", 0.0)),
        bias=str(lora.get("bias", "none")),
        target_modules=list(LORA_TARGET_MODULES),
        use_gradient_checkpointing=str(lora.get("use_gradient_checkpointing", "unsloth")),
        random_state=seed,
        use_rslora=bool(lora.get("use_rslora", False)),
        loftq_config=None,
    )


def _has_loaded_peft_adapter(model: Any) -> bool:
    peft_config = getattr(model, "peft_config", None)
    if isinstance(peft_config, dict) and peft_config:
        return True
    return bool(getattr(model, "active_adapters", None) or getattr(model, "active_adapter", None))


def grpo_row_from_medmcqa(row: dict[str, Any]) -> dict[str, Any]:
    correct_label = correct_label_from_row(row)
    return {
        "prompt": grpo_prompt_from_medmcqa(row),
        "answer": correct_label,
        "correct_label": correct_label,
        "correct_text": _option_text(row, correct_label),
        "reference_explanation": str(row.get("exp") or ""),
        "reference_telugu": str(row.get("telugu") or row.get("synthetic_telugu") or ""),
        "subject_name": str(row.get("subject_name") or ""),
        "topic_name": str(row.get("topic_name") or ""),
    }


def is_usable_grpo_source_row(row: dict[str, Any]) -> bool:
    try:
        correct_label_from_row(row)
        for key in ("question", "opa", "opb", "opc", "opd"):
            _required_str(row, key)
    except ValueError:
        return False
    review_fields = (row.get("telugu"), row.get("synthetic_telugu"))
    return not any(str(value).strip().upper() == "NEEDS_REVIEW" for value in review_fields if value is not None)


def grpo_prompt_from_medmcqa(row: dict[str, Any]) -> list[dict[str, str]]:
    user_prompt = "\n".join(
        [
            "ప్రశ్న:",
            _required_str(row, "question"),
            "",
            f"A. {_required_str(row, 'opa')}",
            f"B. {_required_str(row, 'opb')}",
            f"C. {_required_str(row, 'opc')}",
            f"D. {_required_str(row, 'opd')}",
            "",
            "తెలుగులో సహజంగా సమాధానం ఇవ్వండి.",
            "సరైన ఎంపిక లేదా సరైన వైద్య పదం చెప్పి, 2-4 చిన్న వాక్యాలలో కారణం వివరించండి.",
            "ప్రశ్నలో ప్రమాద సూచనలు ఉంటే భద్రతా జాగ్రత్తను కూడా చేర్చండి.",
        ]
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a careful Telugu medical tutor for ASHA training. "
                "Be medically faithful, concise, and do not invent treatment advice."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]


def correct_label_from_row(row: dict[str, Any]) -> str:
    raw = row.get("cop")
    if isinstance(raw, int) and raw in range(len(OPTION_LABELS)):
        return OPTION_LABELS[raw]
    try:
        index = int(str(raw))
    except ValueError as err:
        raise ValueError(f"Invalid cop value: {raw!r}") from err
    if index not in range(len(OPTION_LABELS)):
        raise ValueError(f"Invalid cop value: {raw!r}")
    return OPTION_LABELS[index]


def correct_option_reward(completions: list[Any], answer: list[str], **kwargs: object) -> list[float]:
    return medical_correctness_reward(completions, answer, **kwargs)


def medical_correctness_reward(
    completions: list[Any],
    answer: list[str],
    correct_text: list[str] | None = None,
    **_: object,
) -> list[float]:
    rewards = []
    option_texts = correct_text or [""] * len(completions)
    for completion, label, option_text in zip(completions, answer, option_texts):
        text = _completion_text(completion)
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        if _contains_option(first_line, label):
            rewards.append(2.5)
        elif _contains_option(text[:120], label):
            rewards.append(2.0)
        elif _contains_option_text(text, option_text):
            rewards.append(1.75)
        elif _contains_any_option(text[:160]):
            rewards.append(-1.0)
        else:
            rewards.append(-0.75)
    return rewards


def telugu_script_reward(completions: list[Any], **_: object) -> list[float]:
    rewards = []
    for completion in completions:
        text = _completion_text(completion)
        density = telugu_density_reward(text)
        english_penalty = 0.5 if _english_word_count(text) > 20 else 0.0
        rewards.append(max(-0.5, (2.0 * density) - english_penalty))
    return rewards


def reference_overlap_reward(
    completions: list[Any],
    reference_telugu: list[str] | None = None,
    reference_explanation: list[str] | None = None,
    **_: object,
) -> list[float]:
    telugu_refs = reference_telugu or [""] * len(completions)
    english_refs = reference_explanation or [""] * len(completions)
    rewards = []
    for completion, telugu_ref, english_ref in zip(completions, telugu_refs, english_refs):
        completion_terms = _meaningful_terms(_completion_text(completion))
        reference_terms = _meaningful_terms(telugu_ref) | _meaningful_terms(english_ref)
        if not completion_terms or not reference_terms:
            rewards.append(0.0)
            continue
        overlap = len(completion_terms & reference_terms) / min(len(reference_terms), 20)
        rewards.append(min(1.0, 2.0 * overlap))
    return rewards


def concise_medical_answer_reward(completions: list[Any], **_: object) -> list[float]:
    rewards = []
    for completion in completions:
        word_count = len(_completion_text(completion).split())
        if 12 <= word_count <= 80:
            rewards.append(1.0)
        elif word_count < 12:
            rewards.append(-0.25)
        else:
            rewards.append(max(-1.0, 1.0 - ((word_count - 80) / 80)))
    return rewards


def medical_guardrail_reward(completions: list[Any], **_: object) -> list[float]:
    unsafe_patterns = (
        "ఖచ్చితంగా నయం",
        "డాక్టర్ అవసరం లేదు",
        "ఆసుపత్రి అవసరం లేదు",
        "stop medicine",
        "ignore symptoms",
    )
    rewards = []
    for completion in completions:
        text = _completion_text(completion).lower()
        rewards.append(-2.0 if any(pattern in text for pattern in unsafe_patterns) else 0.5)
    return rewards


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = [item.get("content", "") for item in completion if isinstance(item, dict)]
        return "\n".join(str(part) for part in parts)
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def _contains_option(text: str, label: str) -> bool:
    escaped = re.escape(label)
    patterns = (
        rf"\b{escaped}\b",
        rf"సమాధానం\s*[:：]?\s*{escaped}",
        rf"ఎంపిక\s*[:：]?\s*{escaped}",
        rf"option\s*[:：]?\s*{escaped}",
    )
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _contains_any_option(text: str) -> bool:
    return any(_contains_option(text, label) for label in OPTION_LABELS)


def _contains_option_text(text: str, option_text: str) -> bool:
    option_terms = _meaningful_terms(option_text)
    if not option_terms:
        return False
    text_terms = _meaningful_terms(text)
    if option_terms & text_terms:
        return True
    normalized_text = _normalize_for_overlap(text)
    normalized_option = _normalize_for_overlap(option_text)
    return bool(normalized_option and normalized_option in normalized_text)


def _meaningful_terms(text: str) -> set[str]:
    normalized = _normalize_for_overlap(text)
    terms = set(re.findall(r"[\w\u0C00-\u0C7F]+", normalized, flags=re.UNICODE))
    return {term for term in terms if len(term) >= 3 and term not in _STOP_TERMS}


def _normalize_for_overlap(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _english_word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]{3,}", text))


_STOP_TERMS = {
    "answer",
    "correct",
    "option",
    "సమాధానం",
    "సరైన",
    "ఎంపిక",
    "వివరణ",
    "ప్రశ్న",
    "కారణం",
}


def _option_text(row: dict[str, Any], label: str) -> str:
    key = {"A": "opa", "B": "opb", "C": "opc", "D": "opd"}[label]
    return _required_str(row, key)


def _required_str(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Row missing non-empty field {key!r}")
    return value.strip()


def _grpo_mapping(config: AppConfig) -> dict[str, Any]:
    grpo = config.raw.get("grpo", {})
    if not isinstance(grpo, dict):
        raise ValueError("config.grpo must be a mapping")
    return grpo


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
        raise ValueError("value must be non-negative")
    return number


def _push_grpo_adapter(output_dir: Path, hub_model_id: str | None) -> None:
    if hub_model_id is None:
        raise ValueError("hub_model_id is required when push_to_hub is enabled")
    try:
        from huggingface_hub import HfApi
    except ImportError as err:
        raise RuntimeError("Install huggingface_hub before pushing GRPO adapter") from err

    api = HfApi()
    api.create_repo(repo_id=hub_model_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=hub_model_id,
        repo_type="model",
        folder_path=str(output_dir),
        ignore_patterns=["checkpoint-*", "checkpoint-*/*", "runs/*", "wandb/*"],
    )
