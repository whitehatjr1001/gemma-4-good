from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from glob import glob
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from gemma_health.types import TrainingExample


Role = Literal["system", "user", "assistant"]
ResponseColumn = Literal["telugu", "romanised_telugu"]

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful rural health assistant. Answer in the requested Telugu style, "
    "prioritize patient safety, and recommend PHC or emergency care when warning signs are present."
)


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class SftExample:
    source: str
    variant: str
    prompt: str
    response: str
    messages: list[dict[str, str]]
    text: str


def training_example_to_sft(
    example: TrainingExample,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    variant: str = "default",
) -> SftExample:
    prompt = _clean_text(example.prompt, "prompt")
    response = _clean_text(example.response, "response")
    messages = [
        ChatMessage(role="system", content=_clean_text(system_prompt, "system_prompt")),
        ChatMessage(role="user", content=prompt),
        ChatMessage(role="assistant", content=response),
    ]
    serializable_messages = [asdict(message) for message in messages]
    return SftExample(
        source=example.source,
        variant=variant,
        prompt=prompt,
        response=response,
        messages=serializable_messages,
        text=messages_to_text(serializable_messages),
    )


def training_examples_to_sft(
    examples: Iterable[TrainingExample],
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    variant: str = "default",
) -> list[SftExample]:
    return [
        training_example_to_sft(example, system_prompt=system_prompt, variant=variant)
        for example in examples
    ]


def synthetic_parquet_to_sft_examples(
    input_paths: Sequence[Path],
    *,
    response_columns: Sequence[ResponseColumn] = ("telugu",),
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    limit: int | None = None,
) -> list[SftExample]:
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if not input_paths:
        raise ValueError("At least one input parquet path is required")

    examples: list[SftExample] = []
    rows_seen = 0
    for input_path in input_paths:
        frame = pd.read_parquet(input_path)
        for row in frame.to_dict(orient="records"):
            if limit is not None and rows_seen >= limit:
                return examples
            rows_seen += 1
            source = _source_name(input_path)
            prompt = synthetic_prompt(row, source)
            for response_column in response_columns:
                response = _clean_text(row.get(response_column), response_column)
                if response == "NEEDS_REVIEW":
                    continue
                examples.append(
                    training_example_to_sft(
                        TrainingExample(prompt=prompt, response=response, source=source),
                        system_prompt=system_prompt,
                        variant=response_column,
                    )
                )
    return examples


def resolve_synthetic_inputs(values: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        candidate = Path(value)
        if candidate.is_dir():
            paths.extend(sorted(candidate.rglob("*.parquet")))
        elif any(token in value for token in ("*", "?", "[")):
            paths.extend(Path(match) for match in sorted(glob(value)))
        else:
            paths.append(candidate)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"SFT input path does not exist: {', '.join(missing)}")
    if not paths:
        raise ValueError("No parquet inputs matched")
    return paths


def synthetic_prompt(row: Mapping[str, Any], source: str) -> str:
    if source == "symptom_diagnosis":
        symptoms = _clean_text(row.get("input_text"), "input_text")
        return "\n".join(
            [
                "Patient symptoms:",
                symptoms,
                "",
                "Give a safety-first ASHA worker triage response.",
            ]
        )
    if source == "medmcqa":
        return "\n".join(
            [
                _clean_text(row.get("question"), "question"),
                f"A. {_clean_text(row.get('opa'), 'opa')}",
                f"B. {_clean_text(row.get('opb'), 'opb')}",
                f"C. {_clean_text(row.get('opc'), 'opc')}",
                f"D. {_clean_text(row.get('opd'), 'opd')}",
                "",
                "Choose the best answer and explain briefly.",
            ]
        )
    return _fallback_prompt(row)


def messages_to_text(messages: Sequence[Mapping[str, str]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = _clean_text(message.get("role"), "role")
        content = _clean_text(message.get("content"), "content")
        parts.append(f"<|{role}|>\n{content}")
    return "\n".join(parts)


def write_sft_jsonl(examples: Iterable[SftExample], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
            count += 1
    return count


def validate_sft_jsonl(input_path: Path) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"SFT JSONL does not exist: {input_path}")

    count = 0
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"SFT row {line_number} must be a JSON object")
            _validate_sft_row(row, line_number)
            count += 1
    if count == 0:
        raise ValueError(f"SFT JSONL is empty: {input_path}")
    return count


def load_sft_jsonl_sample(input_path: Path, limit: int = 3) -> list[dict[str, Any]]:
    if limit < 0:
        raise ValueError("limit must be non-negative")
    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            if len(rows) >= limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("SFT JSONL row must be a JSON object")
            rows.append(row)
    return rows


def _source_name(input_path: Path) -> str:
    if input_path.parent.name in {"train", "test", "validation"}:
        return input_path.parent.parent.name
    return input_path.parent.name


def _fallback_prompt(row: Mapping[str, Any]) -> str:
    for key in ("prompt", "question", "input", "input_text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("Synthetic row does not contain a supported prompt field")


def _clean_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing non-empty text field {field!r}")
    return value.strip()


def _validate_sft_row(row: Mapping[str, Any], line_number: int) -> None:
    for field in ("source", "variant", "prompt", "response", "text"):
        _clean_text(row.get(field), f"{field} at line {line_number}")
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"SFT row {line_number} must contain at least user and assistant messages")
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError(f"SFT row {line_number} messages must be objects")
        role = _clean_text(message.get("role"), f"message.role at line {line_number}")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"SFT row {line_number} has unsupported role: {role!r}")
        _clean_text(message.get("content"), f"message.content at line {line_number}")
