from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gemma_health.config import AppConfig
from gemma_health.rewards.telugu import telugu_density_reward
from gemma_health.training.unsloth_grpo import (
    OPTION_LABELS,
    correct_label_from_row,
    grpo_prompt_from_medmcqa,
    is_usable_grpo_source_row,
)


@dataclass(frozen=True)
class MedMcqaGenerationEvalConfig:
    model_id: str
    dataset_path: Path | None
    hub_dataset_id: str | None
    split: str
    start_index: int
    max_samples: int | None
    batch_size: int
    max_new_tokens: int
    min_new_tokens: int
    temperature: float
    sample_output_limit: int
    output_json: Path | None
    output_jsonl: Path | None


@dataclass(frozen=True)
class MedMcqaPrediction:
    index: int
    subject_name: str
    topic_name: str
    correct_label: str
    predicted_label: str | None
    exact_label_match: bool
    correct_option_text_hit: bool
    telugu_density: float
    token_length: int
    empty: bool
    clipped: bool
    unsafe: bool
    completion: str


@dataclass(frozen=True)
class MedMcqaSegmentSummary:
    segment: str
    value: str
    samples: int
    exact_label_accuracy: float
    option_text_hit_rate: float
    telugu_density_mean: float
    empty_rate: float
    clipped_rate: float
    unsafe_rate: float
    mean_token_length: float


@dataclass(frozen=True)
class MedMcqaGenerationEvalResult:
    model_id: str
    dataset: str
    split: str
    samples: int
    summaries: tuple[MedMcqaSegmentSummary, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True)


@dataclass(frozen=True)
class MedMcqaEvalRow:
    prompt: list[dict[str, str]]
    correct_label: str
    correct_text: str
    subject_name: str
    topic_name: str


def medmcqa_generation_eval_config(
    config: AppConfig,
    *,
    model_id: str | None,
    dataset_path: str | Path | None,
    hub_dataset_id: str | None,
    split: str | None,
    start_index: int,
    max_samples: int | None,
    batch_size: int,
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    sample_output_limit: int,
    output_json: str | Path | None,
    output_jsonl: str | Path | None,
) -> MedMcqaGenerationEvalConfig:
    grpo = config.raw.get("grpo", {})
    if not isinstance(grpo, dict):
        raise ValueError("config.grpo must be a mapping")
    if dataset_path is not None and hub_dataset_id is not None:
        raise ValueError("Use only one of --dataset-path or --hub-dataset-id")
    resolved_dataset_path = dataset_path if dataset_path is not None else grpo.get("dataset_path")
    resolved_hub_dataset_id = hub_dataset_id if hub_dataset_id is not None else grpo.get("hub_dataset_id")
    if dataset_path is not None:
        resolved_hub_dataset_id = None
    if hub_dataset_id is not None:
        resolved_dataset_path = None
    if resolved_dataset_path is None and resolved_hub_dataset_id is None:
        raise ValueError("MedMCQA generation eval requires --dataset-path or --hub-dataset-id")
    if resolved_dataset_path is not None and resolved_hub_dataset_id is not None:
        raise ValueError("Use only one of --dataset-path or --hub-dataset-id")
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if max_samples is not None and max_samples < 1:
        raise ValueError("--max-samples must be at least 1")
    if max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")
    if min_new_tokens < 0:
        raise ValueError("--min-new-tokens must be non-negative")
    if min_new_tokens > max_new_tokens:
        raise ValueError("--min-new-tokens must be <= --max-new-tokens")
    if temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if sample_output_limit < 0:
        raise ValueError("--sample-output-limit must be non-negative")

    return MedMcqaGenerationEvalConfig(
        model_id=model_id or str(grpo.get("base_model_id", config.model.base_id)),
        dataset_path=Path(resolved_dataset_path) if resolved_dataset_path is not None else None,
        hub_dataset_id=str(resolved_hub_dataset_id) if resolved_hub_dataset_id is not None else None,
        split=split or "test",
        start_index=start_index,
        max_samples=max_samples,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        temperature=temperature,
        sample_output_limit=sample_output_limit,
        output_json=Path(output_json) if output_json is not None else None,
        output_jsonl=Path(output_jsonl) if output_jsonl is not None else None,
    )


def run_medmcqa_generation_eval(
    config: AppConfig,
    eval_config: MedMcqaGenerationEvalConfig,
) -> MedMcqaGenerationEvalResult:
    try:
        import torch
        from datasets import load_dataset
        from tqdm.auto import tqdm
        from unsloth import FastLanguageModel
    except ImportError as err:
        raise RuntimeError("Install torch, datasets, tqdm, and unsloth before MedMCQA generation eval") from err

    dataset = (
        load_dataset(eval_config.hub_dataset_id, split=eval_config.split)
        if eval_config.hub_dataset_id is not None
        else load_dataset("parquet", data_files=str(eval_config.dataset_path), split="train")
    )
    rows = [row for raw_row in dataset if (row := normalize_medmcqa_eval_row(dict(raw_row))) is not None]
    if eval_config.start_index >= len(rows):
        raise ValueError(f"--start-index {eval_config.start_index} is outside usable eval rows length {len(rows)}")
    rows = rows[eval_config.start_index :]
    if eval_config.max_samples is not None:
        rows = rows[: eval_config.max_samples]
    if not rows:
        raise ValueError("MedMCQA eval dataset has no usable rows after validation")

    model, processor = FastLanguageModel.from_pretrained(
        model_name=eval_config.model_id,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=config.model.load_in_4bit,
        load_in_8bit=config.model.load_in_8bit,
        device_map="auto",
    )
    FastLanguageModel.for_inference(model)
    model.eval()
    text_tokenizer = _text_tokenizer(processor)
    _left_pad_tokenizer(text_tokenizer)

    predictions: list[MedMcqaPrediction] = []
    output_jsonl_handle = None
    if eval_config.output_jsonl is not None:
        eval_config.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        output_jsonl_handle = eval_config.output_jsonl.open("w", encoding="utf-8")
    try:
        for start in tqdm(range(0, len(rows), eval_config.batch_size)):
            batch_rows = rows[start : start + eval_config.batch_size]
            for offset, row in enumerate(batch_rows):
                input_ids = _encode_single_chat_prompt(text_tokenizer, row.prompt).to(model.device)
                attention_mask = torch.ones_like(input_ids)
                generation_kwargs = _generation_token_kwargs(text_tokenizer)
                with torch.inference_mode():
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=eval_config.max_new_tokens,
                        min_new_tokens=eval_config.min_new_tokens,
                        do_sample=eval_config.temperature > 0,
                        temperature=eval_config.temperature if eval_config.temperature > 0 else None,
                        **generation_kwargs,
                    )
                prompt_length = int(input_ids.shape[-1])
                completion_ids = outputs[:, prompt_length:] if int(outputs.shape[-1]) > prompt_length else outputs
                completion = _batch_decode(text_tokenizer, completion_ids)[0]
                prediction = score_medmcqa_completion(
                    eval_config.start_index + start + offset,
                    row,
                    completion,
                    eval_config.max_new_tokens,
                )
                if len(predictions) < eval_config.sample_output_limit:
                    raw_completion = _batch_decode_with_special_tokens(text_tokenizer, completion_ids)[0]
                    print(
                        json.dumps(
                            {
                                "sample_index": prediction.index,
                                "correct_label": prediction.correct_label,
                                "predicted_label": prediction.predicted_label,
                                "token_length": prediction.token_length,
                                "empty": prediction.empty,
                                "prompt_tokens": prompt_length,
                                "output_tokens": int(outputs.shape[-1]),
                                "completion_token_ids": completion_ids[0].detach().cpu().tolist()[:32],
                                "raw_completion": raw_completion,
                                "completion": prediction.completion,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                predictions.append(prediction)
                if output_jsonl_handle is not None:
                    output_jsonl_handle.write(json.dumps(asdict(prediction), ensure_ascii=False) + "\n")
                    output_jsonl_handle.flush()
                del input_ids
                del attention_mask
                del outputs
    finally:
        if output_jsonl_handle is not None:
            output_jsonl_handle.close()
    result = MedMcqaGenerationEvalResult(
        model_id=eval_config.model_id,
        dataset=eval_config.hub_dataset_id or str(eval_config.dataset_path),
        split=eval_config.split,
        samples=len(predictions),
        summaries=tuple(summarize_medmcqa_predictions(predictions)),
    )
    if eval_config.output_json is not None:
        eval_config.output_json.parent.mkdir(parents=True, exist_ok=True)
        eval_config.output_json.write_text(result.to_json() + "\n", encoding="utf-8")
    return result


def score_medmcqa_completion(
    index: int,
    row: dict[str, Any] | MedMcqaEvalRow,
    completion: str,
    max_new_tokens: int,
) -> MedMcqaPrediction:
    eval_row = row if isinstance(row, MedMcqaEvalRow) else normalize_medmcqa_eval_row(row)
    if eval_row is None:
        raise ValueError("Cannot score unusable MedMCQA row")
    correct_label = eval_row.correct_label
    predicted_label = extract_medmcqa_label(completion)
    token_length = len(completion.split())
    return MedMcqaPrediction(
        index=index,
        subject_name=eval_row.subject_name,
        topic_name=eval_row.topic_name,
        correct_label=correct_label,
        predicted_label=predicted_label,
        exact_label_match=predicted_label == correct_label,
        correct_option_text_hit=_contains_option_text(completion, eval_row.correct_text),
        telugu_density=telugu_density_reward(completion),
        token_length=token_length,
        empty=not completion.strip(),
        clipped=token_length >= max_new_tokens,
        unsafe=_has_unsafe_text(completion),
        completion=completion,
    )


def normalize_medmcqa_eval_row(row: dict[str, Any]) -> MedMcqaEvalRow | None:
    if is_usable_grpo_source_row(row):
        correct_label = correct_label_from_row(row)
        return MedMcqaEvalRow(
            prompt=grpo_prompt_from_medmcqa(row),
            correct_label=correct_label,
            correct_text=str(row.get(_option_key(correct_label)) or ""),
            subject_name=str(row.get("subject_name") or "unknown"),
            topic_name=str(row.get("topic_name") or "unknown"),
        )
    derived_label = _derived_label_from_synthetic_answer(row)
    if derived_label is not None and _has_required_mcq_fields(row):
        review_fields = (row.get("telugu"), row.get("synthetic_telugu"))
        if any(str(value).strip().upper() == "NEEDS_REVIEW" for value in review_fields if value is not None):
            return None
        return MedMcqaEvalRow(
            prompt=grpo_prompt_from_medmcqa(row),
            correct_label=derived_label,
            correct_text=str(row.get(_option_key(derived_label)) or ""),
            subject_name=str(row.get("subject_name") or "unknown"),
            topic_name=str(row.get("topic_name") or "unknown"),
        )
    prompt = row.get("prompt")
    answer = str(row.get("answer") or row.get("correct_label") or "").strip().upper()
    if not _is_chat_prompt(prompt) or answer not in OPTION_LABELS:
        return None
    review_fields = (row.get("reference_telugu"), row.get("telugu"), row.get("synthetic_telugu"))
    if any(str(value).strip().upper() == "NEEDS_REVIEW" for value in review_fields if value is not None):
        return None
    return MedMcqaEvalRow(
        prompt=[{"role": str(item["role"]), "content": str(item["content"])} for item in prompt],
        correct_label=answer,
        correct_text=str(row.get("correct_text") or ""),
        subject_name=str(row.get("subject_name") or "unknown"),
        topic_name=str(row.get("topic_name") or "unknown"),
    )


def extract_medmcqa_label(text: str) -> str | None:
    head = text.strip()[:220]
    patterns = (
        r"(?:సమాధానం|ఎంపిక|answer|option)\s*[:：]?\s*([ABCD])\b",
        r"\b([ABCD])\s*(?:\.|\)|-|:|：)",
        r"^\s*([ABCD])\b",
    )
    for pattern in patterns:
        match = re.search(pattern, head, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def summarize_medmcqa_predictions(predictions: list[MedMcqaPrediction]) -> list[MedMcqaSegmentSummary]:
    if not predictions:
        raise ValueError("predictions must not be empty")
    summaries = [_summarize_segment("overall", "all", predictions)]
    summaries.extend(_summaries_by("subject", predictions, lambda item: item.subject_name))
    summaries.extend(_summaries_by("topic", predictions, lambda item: item.topic_name))
    summaries.extend(_summaries_by("answer", predictions, lambda item: item.correct_label))
    return summaries


def _summaries_by(
    segment: str,
    predictions: list[MedMcqaPrediction],
    key_fn: Any,
) -> list[MedMcqaSegmentSummary]:
    grouped: dict[str, list[MedMcqaPrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[str(key_fn(prediction) or "unknown")].append(prediction)
    return [_summarize_segment(segment, key, rows) for key, rows in sorted(grouped.items())]


def _summarize_segment(
    segment: str,
    value: str,
    predictions: list[MedMcqaPrediction],
) -> MedMcqaSegmentSummary:
    sample_count = len(predictions)
    return MedMcqaSegmentSummary(
        segment=segment,
        value=value,
        samples=sample_count,
        exact_label_accuracy=_mean(1.0 if item.exact_label_match else 0.0 for item in predictions),
        option_text_hit_rate=_mean(1.0 if item.correct_option_text_hit else 0.0 for item in predictions),
        telugu_density_mean=_mean(item.telugu_density for item in predictions),
        empty_rate=_mean(1.0 if item.empty else 0.0 for item in predictions),
        clipped_rate=_mean(1.0 if item.clipped else 0.0 for item in predictions),
        unsafe_rate=_mean(1.0 if item.unsafe else 0.0 for item in predictions),
        mean_token_length=_mean(float(item.token_length) for item in predictions),
    )


def _format_chat_prompt(processor: Any, row: dict[str, Any]) -> str:
    messages = row.prompt if isinstance(row, MedMcqaEvalRow) else grpo_prompt_from_medmcqa(row)
    return _messages_to_text(_text_tokenizer(processor), messages)


def _text_tokenizer(processor: Any) -> Any:
    tokenizer = getattr(processor, "tokenizer", None)
    return tokenizer if tokenizer is not None else processor


def _left_pad_tokenizer(tokenizer: Any) -> None:
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"


def _encode_chat_prompts(tokenizer: Any, batch_messages: list[list[dict[str, str]]]) -> Any:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            encoded = tokenizer.apply_chat_template(
                batch_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                padding=True,
            )
            if isinstance(encoded, dict) and "input_ids" in encoded:
                return encoded
        except TypeError:
            pass
    prompts = [_messages_to_text(tokenizer, messages) for messages in batch_messages]
    return _encode_prompts(tokenizer, prompts)


def _encode_single_chat_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> Any:
    if hasattr(tokenizer, "apply_chat_template"):
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(encoded, "shape"):
            return encoded
    prompt = _messages_to_text(tokenizer, messages)
    encoded = _encode_prompts(tokenizer, [prompt])
    input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else getattr(encoded, "input_ids", None)
    if input_ids is None:
        raise ValueError("tokenizer did not return input_ids")
    return input_ids


def _messages_to_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return "\n\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\n\nassistant:"


def _encode_prompts(processor: Any, prompts: list[str]) -> Any:
    try:
        return processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
    except TypeError:
        return processor(prompts, return_tensors="pt", padding=True, truncation=True)


def _batch_decode(processor: Any, token_ids: Any) -> list[str]:
    decoder = getattr(processor, "batch_decode", None)
    if decoder is None and hasattr(processor, "tokenizer"):
        decoder = processor.tokenizer.batch_decode
    if decoder is None:
        raise ValueError("processor does not expose batch_decode")
    return list(decoder(token_ids, skip_special_tokens=True))


def _batch_decode_with_special_tokens(processor: Any, token_ids: Any) -> list[str]:
    decoder = getattr(processor, "batch_decode", None)
    if decoder is None and hasattr(processor, "tokenizer"):
        decoder = processor.tokenizer.batch_decode
    if decoder is None:
        raise ValueError("processor does not expose batch_decode")
    return list(decoder(token_ids, skip_special_tokens=False))


def _eos_token_id(processor: Any) -> int | None:
    if getattr(processor, "eos_token_id", None) is not None:
        return int(processor.eos_token_id)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "eos_token_id", None) is not None:
        return int(tokenizer.eos_token_id)
    return None


def _generation_token_kwargs(tokenizer: Any) -> dict[str, Any]:
    eos_token_id = _eos_token_id(tokenizer)
    kwargs: dict[str, Any] = {}
    if eos_token_id is not None:
        kwargs["eos_token_id"] = eos_token_id
        kwargs["pad_token_id"] = eos_token_id
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None and eos_token_id is not None and int(pad_token_id) != eos_token_id:
        kwargs["bad_words_ids"] = [[int(pad_token_id)]]
    return kwargs


def _contains_option_text(text: str, option_text: str) -> bool:
    if not option_text.strip():
        return False
    text_terms = set(re.findall(r"[\w\u0C00-\u0C7F]+", text.lower(), flags=re.UNICODE))
    option_terms = {term for term in re.findall(r"[\w\u0C00-\u0C7F]+", option_text.lower(), flags=re.UNICODE) if len(term) >= 3}
    return bool(text_terms & option_terms)


def _has_unsafe_text(text: str) -> bool:
    lowered = text.lower()
    return any(
        pattern in lowered
        for pattern in (
            "డాక్టర్ అవసరం లేదు",
            "ఆసుపత్రి అవసరం లేదు",
            "ఖచ్చితంగా నయం",
            "stop medicine",
            "ignore symptoms",
        )
    )


def _option_key(label: str) -> str:
    return {"A": "opa", "B": "opb", "C": "opc", "D": "opd"}[label]


def _derived_label_from_synthetic_answer(row: dict[str, Any]) -> str | None:
    for key in ("telugu", "synthetic_telugu", "romanised_telugu"):
        value = row.get(key)
        if isinstance(value, str):
            label = extract_medmcqa_label(value)
            if label in OPTION_LABELS:
                return label
    return None


def _has_required_mcq_fields(row: dict[str, Any]) -> bool:
    return all(isinstance(row.get(key), str) and str(row[key]).strip() for key in ("question", "opa", "opb", "opc", "opd"))


def _is_chat_prompt(value: object) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, dict)
        and isinstance(item.get("role"), str)
        and isinstance(item.get("content"), str)
        and item["role"].strip()
        and item["content"].strip()
        for item in value
    )


def _mean(values: Any) -> float:
    numbers = list(values)
    if not numbers:
        return 0.0
    return float(sum(numbers) / len(numbers))
