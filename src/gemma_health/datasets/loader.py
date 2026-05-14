from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from gemma_health.datasets.base import DatasetConfig, DatasetSource
from gemma_health.types import TrainingExample


Converter = Callable[[Mapping[str, Any], DatasetConfig], TrainingExample]


class HuggingFaceDataset:
    def __init__(self, source: DatasetConfig, converter: Converter) -> None:
        self.name = source.name
        self.source = source
        self.converter = converter

    def load(self) -> list[TrainingExample]:
        examples: list[TrainingExample] = []
        for row in load_hf_rows(self.source):
            if not _matches_filter(row, self.source):
                continue
            if self.source.max_examples is not None and len(examples) >= self.source.max_examples:
                break
            examples.append(self.converter(row, self.source))
        return examples


class JsonlDataset:
    name = "field_dialogues"

    def __init__(self, source: DatasetConfig) -> None:
        self.source = source

    def load(self) -> list[TrainingExample]:
        if self.source.path is None:
            raise ValueError(f"Dataset {self.source.name!r} requires path in config.yaml")

        path = Path(self.source.path)
        if not path.exists():
            raise FileNotFoundError(f"Field dialogues file does not exist: {path}")

        examples: list[TrainingExample] = []
        with path.open("r", encoding="utf-8") as file:
            for index, line in enumerate(file):
                if self.source.max_examples is not None and index >= self.source.max_examples:
                    break
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Field dialogues row {index + 1} must be a JSON object")
                examples.append(
                    TrainingExample(
                        prompt=required_text(row, "prompt", self.source.name),
                        response=required_text(row, "response", self.source.name),
                        source=self.source.name,
                    )
                )
        return examples


def build_dataset(source: DatasetConfig) -> DatasetSource:
    if source.name == "field_dialogues":
        return JsonlDataset(source)

    converters: dict[str, Converter] = {
        "telugu_alpaca": _telugu_alpaca,
        "symptom_diagnosis": _symptom_diagnosis,
        "medmcqa": _medmcqa,
        "prescriptions": _prescriptions,
        "indivibe_chat": _indivibe_translation,
        "indivibe_stem": _indivibe_translation,
    }
    try:
        converter = converters[source.name]
    except KeyError as err:
        available = ", ".join(sorted([*converters, "field_dialogues"]))
        raise ValueError(f"Unknown dataset {source.name!r}. Available: {available}") from err
    return HuggingFaceDataset(source, converter)


def load_hf_rows(source: DatasetConfig) -> Iterator[dict[str, Any]]:
    if source.hf_id is None:
        raise ValueError(f"Dataset {source.name!r} requires hf_id in config.yaml")

    try:
        import datasets as hf_datasets
    except ImportError as err:
        raise RuntimeError("Install the 'datasets' package to load Hugging Face datasets") from err

    try:
        if source.hf_config is None:
            dataset = hf_datasets.load_dataset(source.hf_id, split=source.split)
        else:
            dataset = hf_datasets.load_dataset(source.hf_id, source.hf_config, split=source.split)
    except Exception as err:
        raise RuntimeError(
            f"Could not load Hugging Face dataset {source.hf_id!r} split {source.split!r}"
        ) from err

    for row in dataset:
        if not isinstance(row, Mapping):
            raise ValueError(f"Expected mapping row from dataset {source.name!r}, got {type(row).__name__}")
        yield dict(row)


def required_text(row: Mapping[str, Any], field: str, dataset_name: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Dataset {dataset_name!r} row missing non-empty text field {field!r}")
    return value.strip()


def optional_text(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    return value.strip() if isinstance(value, str) else ""


def _matches_filter(row: Mapping[str, Any], source: DatasetConfig) -> bool:
    if source.language is not None and row.get("language") != source.language:
        return False
    if source.scripts and row.get("script") not in source.scripts:
        return False
    return True


def _telugu_alpaca(row: Mapping[str, Any], source: DatasetConfig) -> TrainingExample:
    instruction = required_text(row, "telugu_instruction", source.name)
    input_text = optional_text(row, "telugu_input")
    prompt = f"{instruction}\n\n{input_text}" if input_text else instruction
    return TrainingExample(
        prompt=prompt,
        response=required_text(row, "telugu_output", source.name),
        source=source.name,
    )


def _symptom_diagnosis(row: Mapping[str, Any], source: DatasetConfig) -> TrainingExample:
    symptoms = required_text(row, "input_text", source.name)
    diagnosis = required_text(row, "output_text", source.name)
    if source.synthetic_telugu:
        return TrainingExample(
            prompt=(
                "క్రింది రోగి లక్షణాలను చదవండి. సాధ్యమైన నిర్ధారణను చెప్పండి మరియు "
                "ASHA వర్కర్ కోసం భద్రతను ముందుపెట్టి చిన్న ట్రయాజ్ సూచన ఇవ్వండి.\n\n"
                f"లక్షణాలు: {symptoms}"
            ),
            response=(
                f"సంభావ్య నిర్ధారణ: {diagnosis}\n"
                "భద్రతా సూచన: లక్షణాలు తీవ్రమైతే, శ్వాస తీసుకోవడంలో ఇబ్బంది ఉంటే, "
                "అపస్మారం, గందరగోళం, నిరంతర వాంతులు, రక్తస్రావం లేదా చాలా అధిక జ్వరం ఉంటే "
                "వెంటనే PHC లేదా అత్యవసర వైద్య సేవలకు రిఫర్ చేయండి."
            ),
            source=source.name,
        )
    return TrainingExample(
        prompt=(
            "Patient symptoms:\n"
            f"{symptoms}\n\n"
            "Return the likely diagnosis and a safety-conscious triage note."
        ),
        response=f"Likely diagnosis: {diagnosis}",
        source=source.name,
    )


def _medmcqa(row: Mapping[str, Any], source: DatasetConfig) -> TrainingExample:
    options = [
        ("A", required_text(row, "opa", source.name)),
        ("B", required_text(row, "opb", source.name)),
        ("C", required_text(row, "opc", source.name)),
        ("D", required_text(row, "opd", source.name)),
    ]
    correct_index = row.get("cop")
    if not isinstance(correct_index, int) or correct_index not in range(len(options)):
        raise ValueError(f"Dataset {source.name!r} row has invalid correct option: {correct_index!r}")

    correct_label, correct_answer = options[correct_index]
    question = required_text(row, "question", source.name)
    explanation = optional_text(row, "exp")
    if source.synthetic_telugu:
        prompt = "\n".join(
            [
                "క్రింది వైద్య ప్రశ్నకు సరైన ఎంపికను ఎంచుకోండి. తరువాత చిన్న వివరణ ఇవ్వండి.",
                question,
                *(f"{label}. {text}" for label, text in options),
            ]
        )
        response = f"సరైన సమాధానం: {correct_label}. {correct_answer}"
        if explanation:
            response = f"{response}\nవివరణ: {explanation}"
        return TrainingExample(prompt=prompt, response=response, source=source.name)

    prompt = "\n".join(
        [
            question,
            *(f"{label}. {text}" for label, text in options),
            "Choose the best answer and explain briefly.",
        ]
    )
    response = f"Answer: {correct_label}. {correct_answer}"
    if explanation:
        response = f"{response}\nExplanation: {explanation}"
    return TrainingExample(prompt=prompt, response=response, source=source.name)


def _prescriptions(row: Mapping[str, Any], source: DatasetConfig) -> TrainingExample:
    return TrainingExample(
        prompt=required_text(row, "prompt", source.name),
        response=required_text(row, "response", source.name),
        source=source.name,
    )


def _indivibe_translation(row: Mapping[str, Any], source: DatasetConfig) -> TrainingExample:
    language = required_text(row, "language", source.name)
    script = required_text(row, "script", source.name)
    category = required_text(row, "category", source.name)
    prompt = required_text(row, "prompt", source.name)
    return TrainingExample(
        prompt=(
            f"Translate this {script} {language} benchmark prompt to English.\n"
            f"Category: {category}\n\n"
            f"{prompt}"
        ),
        response=required_text(row, "original_prompt", source.name),
        source=source.name,
    )
