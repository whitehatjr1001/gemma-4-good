from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from gemma_health.datasets.base import DatasetConfig, DatasetSource
from gemma_health.types import TrainingExample


Converter = Callable[[Mapping[str, Any], DatasetConfig], TrainingExample | list[TrainingExample]]


class HuggingFaceDataset:
    def __init__(self, source: DatasetConfig, converter: Converter) -> None:
        self.name = source.name
        self.source = source
        self.converter = converter

    def load(self) -> list[TrainingExample]:
        return list(self.iter_examples())

    def iter_examples(self) -> Iterator[TrainingExample]:
        count = 0
        for row in load_hf_rows(self.source):
            if not _matches_filter(row, self.source):
                continue
            if self.source.max_examples is not None and count >= self.source.max_examples:
                break
            converted = self.converter(row, self.source)
            converted_examples = converted if isinstance(converted, list) else [converted]
            for example in converted_examples:
                if self.source.max_examples is not None and count >= self.source.max_examples:
                    break
                count += 1
                yield example


class JsonlDataset:
    name = "field_dialogues"

    def __init__(self, source: DatasetConfig) -> None:
        self.source = source

    def load(self) -> list[TrainingExample]:
        return list(self.iter_examples())

    def iter_examples(self) -> Iterator[TrainingExample]:
        if self.source.path is None:
            raise ValueError(f"Dataset {self.source.name!r} requires path in config.yaml")

        path = Path(self.source.path)
        if not path.exists():
            raise FileNotFoundError(f"Field dialogues file does not exist: {path}")

        with path.open("r", encoding="utf-8") as file:
            for index, line in enumerate(file):
                if self.source.max_examples is not None and index >= self.source.max_examples:
                    break
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Field dialogues row {index + 1} must be a JSON object")
                yield TrainingExample(
                    prompt=required_text(row, "prompt", self.source.name),
                    response=required_text(row, "response", self.source.name),
                    source=self.source.name,
                )


def build_dataset(source: DatasetConfig) -> DatasetSource:
    if source.name == "field_dialogues":
        return JsonlDataset(source)

    converters: dict[str, Converter] = {
        "telugu_alpaca": _telugu_alpaca,
        "english_telugu_parallel": _english_telugu_parallel,
        "samanantar_te": _samanantar_translation,
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
    if source.path is not None:
        yield from load_parquet_rows(source)
        return

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


def load_parquet_rows(source: DatasetConfig) -> Iterator[dict[str, Any]]:
    if source.path is None:
        raise ValueError(f"Dataset {source.name!r} requires path in config.yaml")

    path = Path(source.path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset {source.name!r} parquet path does not exist: {path}")

    paths = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    emitted = 0
    for parquet_path in paths:
        for row in _iter_parquet_rows(parquet_path, source.max_examples):
            if not isinstance(row, Mapping):
                raise ValueError(f"Expected mapping row from dataset {source.name!r}, got {type(row).__name__}")
            yield dict(row)
            emitted += 1
            if source.max_examples is not None and emitted >= source.max_examples:
                return


def _iter_parquet_rows(path: Path, max_rows: int | None) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        try:
            import pandas as pd
        except ImportError as err:
            raise RuntimeError("Install 'pyarrow' or 'pandas' to load local parquet datasets") from err
        frame = pd.read_parquet(path)
        rows = frame.head(max_rows).to_dict(orient="records") if max_rows is not None else frame.to_dict(orient="records")
        yield from rows
        return

    emitted = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=1024):
        for row in batch.to_pylist():
            yield row
            emitted += 1
            if max_rows is not None and emitted >= max_rows:
                return


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


def _telugu_alpaca(row: Mapping[str, Any], source: DatasetConfig) -> list[TrainingExample]:
    return [
        TrainingExample(
            prompt=_instruction_prompt(
                instruction=required_text(row, "telugu_transliterated_instruction", source.name),
                input_text=optional_text(row, "telugu_transliterated_input"),
                script="romanised Telugu",
            ),
            response=required_text(row, "telugu_transliterated_output", source.name),
            source=source.name,
            variant="romanised_telugu",
        ),
        TrainingExample(
            prompt=_instruction_prompt(
                instruction=required_text(row, "telugu_instruction", source.name),
                input_text=optional_text(row, "telugu_input"),
                script="native Telugu script",
            ),
            response=required_text(row, "telugu_output", source.name),
            source=source.name,
            variant="native_telugu",
        ),
    ]


def _english_telugu_parallel(row: Mapping[str, Any], source: DatasetConfig) -> TrainingExample:
    return TrainingExample(
        prompt=(
            "Translate this English sentence into natural Telugu script.\n"
            "Use Telugu script, not romanised Telugu.\n\n"
            f"English:\n{required_text(row, 'english', source.name)}"
        ),
        response=required_text(row, "telugu", source.name),
        source=source.name,
        variant="native_telugu",
    )


def _samanantar_translation(row: Mapping[str, Any], source: DatasetConfig) -> TrainingExample:
    return TrainingExample(
        prompt=(
            "Translate this English sentence into natural Telugu script.\n"
            "Use Telugu script, not romanised Telugu.\n\n"
            f"English:\n{required_text(row, 'src', source.name)}"
        ),
        response=required_text(row, "tgt", source.name),
        source=source.name,
        variant="native_telugu",
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
            f"Rewrite this English benchmark prompt in {language}.\n"
            f"Expected script: {script}.\n"
            f"Category: {category}\n\n"
            f"English prompt:\n{required_text(row, 'original_prompt', source.name)}"
        ),
        response=prompt,
        source=source.name,
        variant="romanised_telugu" if script == "romanised" else "native_telugu",
    )


def _instruction_prompt(instruction: str, input_text: str, script: str) -> str:
    parts = [
        f"Answer the following instruction in {script}.",
        "",
        "Instruction:",
        instruction,
    ]
    if input_text:
        parts.extend(["", "Input:", input_text])
    return "\n".join(parts)
