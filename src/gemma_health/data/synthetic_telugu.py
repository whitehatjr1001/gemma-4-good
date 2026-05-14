from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd


DEFAULT_MODEL = "qwen3:30b-a3b"


class OllamaClient(Protocol):
    def chat(self, **kwargs: object) -> dict[str, Any]: ...


@dataclass(frozen=True)
class SyntheticGenerationResult:
    input_path: Path
    output_path: Path
    row_count: int
    model: str


def generate_synthetic_parquet(
    input_path: Path,
    output_path: Path,
    dataset_name: str,
    model: str = DEFAULT_MODEL,
    limit: int | None = None,
    skip_existing: bool = True,
    num_predict: int = 1024,
    temperature: float = 0.2,
    checkpoint_interval: int = 25,
    workers: int = 1,
    retries: int = 2,
    client: OllamaClient | None = None,
) -> SyntheticGenerationResult:
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    if retries < 0:
        raise ValueError("retries must be non-negative")

    ollama_client = client or _default_ollama_client()
    frame = pd.read_parquet(input_path)
    if limit is not None:
        frame = frame.head(limit)

    start_index = 0
    synthetic_telugu: list[str] = []
    synthetic_romanised_telugu: list[str] = []
    synthetic_task: list[str] = []
    synthetic_model: list[str] = []
    if skip_existing and output_path.exists():
        existing = pd.read_parquet(output_path)
        if _has_synthetic_columns(existing):
            if len(existing) >= len(frame):
                return SyntheticGenerationResult(
                    input_path=input_path,
                    output_path=output_path,
                    row_count=len(existing),
                    model=model,
                )
            start_index = len(existing)
            synthetic_telugu = [str(value) for value in existing["telugu"].tolist()]
            synthetic_romanised_telugu = [str(value) for value in existing["romanised_telugu"].tolist()]
            synthetic_task = [str(value) for value in existing["synthetic_task"].tolist()]
            synthetic_model = [str(value) for value in existing["synthetic_model"].tolist()]

    rows = frame.to_dict(orient="records")
    if workers == 1:
        for index, row in enumerate(rows[start_index:], start=start_index + 1):
            task = _task_for_dataset(dataset_name)
            pair = _generate_row_with_retries(
                row,
                dataset_name,
                model,
                ollama_client,
                num_predict=num_predict,
                temperature=temperature,
                retries=retries,
            )
            synthetic_telugu.append(pair.telugu)
            synthetic_romanised_telugu.append(pair.romanised_telugu)
            synthetic_task.append(task)
            synthetic_model.append(model)
            print(f"{dataset_name}: generated {index}/{len(rows)}", flush=True)
            if index % checkpoint_interval == 0:
                _write_output(
                    frame=frame,
                    output_path=output_path,
                    synthetic_telugu=synthetic_telugu,
                    synthetic_romanised_telugu=synthetic_romanised_telugu,
                    synthetic_task=synthetic_task,
                    synthetic_model=synthetic_model,
                )
    else:
        for batch_start in range(start_index, len(rows), checkpoint_interval):
            batch_rows = rows[batch_start : batch_start + checkpoint_interval]
            pairs: list[TeluguPair | None] = [None] * len(batch_rows)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _generate_row_with_retries,
                        row,
                        dataset_name,
                        model,
                        ollama_client,
                        num_predict,
                        temperature,
                        retries,
                    ): offset
                    for offset, row in enumerate(batch_rows)
                }
                for future in as_completed(futures):
                    offset = futures[future]
                    pairs[offset] = future.result()
                    print(f"{dataset_name}: generated {batch_start + offset + 1}/{len(rows)}", flush=True)

            task = _task_for_dataset(dataset_name)
            for pair in pairs:
                if pair is None:
                    raise RuntimeError("Internal error: missing generated pair")
                synthetic_telugu.append(pair.telugu)
                synthetic_romanised_telugu.append(pair.romanised_telugu)
                synthetic_task.append(task)
                synthetic_model.append(model)
            _write_output(
                frame=frame,
                output_path=output_path,
                synthetic_telugu=synthetic_telugu,
                synthetic_romanised_telugu=synthetic_romanised_telugu,
                synthetic_task=synthetic_task,
                synthetic_model=synthetic_model,
            )

    _write_output(
        frame=frame,
        output_path=output_path,
        synthetic_telugu=synthetic_telugu,
        synthetic_romanised_telugu=synthetic_romanised_telugu,
        synthetic_task=synthetic_task,
        synthetic_model=synthetic_model,
    )
    return SyntheticGenerationResult(input_path=input_path, output_path=output_path, row_count=len(synthetic_telugu), model=model)


def _write_output(
    frame: pd.DataFrame,
    output_path: Path,
    synthetic_telugu: list[str],
    synthetic_romanised_telugu: list[str],
    synthetic_task: list[str],
    synthetic_model: list[str],
) -> None:
    output = frame.head(len(synthetic_telugu)).copy()
    output["telugu"] = synthetic_telugu
    output["romanised_telugu"] = synthetic_romanised_telugu
    output["synthetic_telugu"] = synthetic_telugu
    output["synthetic_task"] = synthetic_task
    output["synthetic_model"] = synthetic_model

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)


def _has_synthetic_columns(frame: pd.DataFrame) -> bool:
    return {"telugu", "romanised_telugu", "synthetic_task", "synthetic_model"}.issubset(frame.columns)


@dataclass(frozen=True)
class TeluguPair:
    telugu: str
    romanised_telugu: str


def generate_asha_dialogue(
    symptoms: str,
    model: str = DEFAULT_MODEL,
    client: OllamaClient | None = None,
    num_predict: int = 1024,
    temperature: float = 0.2,
) -> TeluguPair:
    ollama_client = client or _default_ollama_client()
    response = ollama_client.chat(
        model=model,
        think=False,
        format="json",
        options={"num_predict": num_predict, "temperature": temperature},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical Telugu language expert. Generate realistic ASHA worker "
                    "clinical conversations in Telugu script. Always include:\n"
                    "1. లక్షణాల విశ్లేషణ (symptom analysis)\n"
                    "2. రిస్క్ స్థాయి: LOW/MEDIUM/HIGH (risk level)\n"
                    "3. చర్య: self-care/PHC/emergency (action)\n"
                    "Also provide a romanised Telugu version in Latin script.\n"
                    "Do not claim clinical certainty.\n"
                    "Keep each value concise: 6-10 short lines maximum.\n"
                    "Respond ONLY as valid JSON with keys: telugu, romanised_telugu."
                ),
            },
            {
                "role": "user",
                "content": f"/no_think\nPatient symptoms: {symptoms}\nGenerate ASHA triage dialogue.",
            },
        ],
    )
    return _message_pair(response)


def generate_medical_qa(
    row: dict[str, Any],
    model: str = DEFAULT_MODEL,
    client: OllamaClient | None = None,
    num_predict: int = 1024,
    temperature: float = 0.2,
) -> TeluguPair:
    ollama_client = client or _default_ollama_client()
    response = ollama_client.chat(
        model=model,
        think=False,
        format="json",
        options={"num_predict": num_predict, "temperature": temperature},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Telugu medical tutor. Convert English medical exam QA into a "
                    "simple Telugu explanation for ASHA training. Keep medical terms accurate. "
                    "Also provide a romanised Telugu version in Latin script.\n"
                    "Keep each value concise: correct answer plus 3-5 short explanation lines.\n"
                    "Respond ONLY as valid JSON with keys: telugu, romanised_telugu."
                ),
            },
            {
                "role": "user",
                "content": f"/no_think\n{_format_medmcqa_prompt(row)}",
            },
        ],
    )
    return _message_pair(response)


def output_path_for(input_path: Path, input_root: Path, output_root: Path) -> Path:
    return output_root / input_path.relative_to(input_root)


def _generate_row(
    row: dict[str, Any],
    dataset_name: str,
    model: str,
    client: OllamaClient,
    num_predict: int,
    temperature: float,
) -> TeluguPair:
    if dataset_name == "symptom_diagnosis":
        return generate_asha_dialogue(
            _required_str(row, "input_text"),
            model=model,
            client=client,
            num_predict=num_predict,
            temperature=temperature,
        )
    if dataset_name == "medmcqa":
        return generate_medical_qa(
            row,
            model=model,
            client=client,
            num_predict=num_predict,
            temperature=temperature,
        )
    raise ValueError(f"Unsupported synthetic Telugu dataset: {dataset_name}")


def _generate_row_with_retries(
    row: dict[str, Any],
    dataset_name: str,
    model: str,
    client: OllamaClient,
    num_predict: int,
    temperature: float,
    retries: int,
) -> TeluguPair:
    last_error: Exception | None = None
    for _ in range(retries + 1):
        try:
            return _generate_row(
                row,
                dataset_name,
                model,
                client,
                num_predict=num_predict,
                temperature=temperature,
            )
        except ValueError as err:
            last_error = err
    assert last_error is not None
    raise last_error


def _task_for_dataset(dataset_name: str) -> str:
    if dataset_name == "symptom_diagnosis":
        return "asha_triage_dialogue"
    if dataset_name == "medmcqa":
        return "medical_qa_explanation"
    raise ValueError(f"Unsupported synthetic Telugu dataset: {dataset_name}")


def _format_medmcqa_prompt(row: dict[str, Any]) -> str:
    correct_index = row.get("cop")
    option_labels = ["A", "B", "C", "D"]
    correct_label = (
        option_labels[correct_index]
        if isinstance(correct_index, int) and correct_index in range(len(option_labels))
        else "unknown"
    )
    return "\n".join(
        [
            "Question:",
            _required_str(row, "question"),
            "",
            f"A. {_required_str(row, 'opa')}",
            f"B. {_required_str(row, 'opb')}",
            f"C. {_required_str(row, 'opc')}",
            f"D. {_required_str(row, 'opd')}",
            "",
            f"Correct answer: {correct_label}",
            f"Explanation: {_optional_str(row, 'exp')}",
            "",
            "Generate a Telugu training answer with the correct option and explanation.",
        ]
    )


def _message_pair(response: dict[str, Any]) -> TeluguPair:
    content = _message_content(response)
    start = content.find("{")
    end = content.rfind("}")
    if start == -1:
        return TeluguPair(telugu=content.strip(), romanised_telugu="NEEDS_REVIEW")
    if end == -1 or end < start:
        content = f"{content.rstrip()}" + '"}'
        end = content.rfind("}")
    try:
        payload = json.loads(content[start : end + 1])
    except json.JSONDecodeError as err:
        raise ValueError("Ollama response JSON could not be parsed") from err
    if not isinstance(payload, dict):
        raise ValueError("Ollama response JSON must be an object")
    telugu = payload.get("telugu")
    romanised_telugu = payload.get("romanised_telugu")
    if not isinstance(telugu, str) or not telugu.strip():
        raise ValueError("Ollama response JSON missing telugu")
    if not isinstance(romanised_telugu, str) or not romanised_telugu.strip():
        romanised_telugu = "NEEDS_REVIEW"
    return TeluguPair(telugu=telugu.strip(), romanised_telugu=romanised_telugu.strip())


def _message_content(response: dict[str, Any]) -> str:
    message = response.get("message") if isinstance(response, dict) else getattr(response, "message", None)
    if not isinstance(message, dict):
        content = getattr(message, "content", None)
    else:
        content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Ollama response missing message content")
    return content.strip()


def _required_str(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Row missing non-empty field {key!r}")
    return value.strip()


def _optional_str(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    return value.strip() if isinstance(value, str) else ""


def _default_ollama_client() -> OllamaClient:
    try:
        import ollama
    except ImportError as err:
        raise RuntimeError("Install the 'ollama' package to generate synthetic Telugu data") from err
    return ollama
