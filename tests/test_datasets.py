from __future__ import annotations

from pathlib import Path

import pytest

from gemma_health.config import load_config
from gemma_health.data.mixture import enabled_dataset_configs, load_training_examples
from gemma_health.datasets.base import DatasetConfig
from gemma_health.datasets.loader import JsonlDataset
from gemma_health.datasets import loader


def test_enabled_dataset_configs_load_all_examples_by_default() -> None:
    config = load_config(Path("config.yaml"))

    sources = enabled_dataset_configs(config)

    assert [source.name for source in sources] == [
        "telugu_alpaca",
        "symptom_diagnosis",
        "medmcqa",
        "indivibe_chat",
        "indivibe_stem",
    ]
    assert all(source.max_examples is None for source in sources)
    assert {source.name: source.language_status for source in sources} == {
        "telugu_alpaca": "native_telugu",
        "symptom_diagnosis": "synthetic_telugu",
        "medmcqa": "synthetic_telugu",
        "indivibe_chat": "translation_pair",
        "indivibe_stem": "translation_pair",
    }


def test_enabled_dataset_configs_can_allocate_smoke_examples() -> None:
    config = load_config(Path("config.yaml"))
    config.raw["training"]["load_all_examples"] = False

    sources = enabled_dataset_configs(config)

    assert sum(source.max_examples or 0 for source in sources) == config.raw["training"]["train_examples"]


def test_load_training_examples_from_multiple_hf_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    rows_by_name = {
        "telugu_alpaca": [
            {
                "telugu_instruction": "లక్షణాలు చెప్పండి",
                "telugu_input": "",
                "telugu_output": "PHC కి వెళ్లండి",
            }
        ],
        "symptom_diagnosis": [{"input_text": "high fever", "output_text": "dengue"}],
        "medmcqa": [
            {
                "question": "Best test?",
                "opa": "A test",
                "opb": "B test",
                "opc": "C test",
                "opd": "D test",
                "cop": 1,
                "exp": "Because B is correct.",
            }
        ],
        "indivibe_chat": [
            {
                "prompt": "నీ రోజు ఎలా ఉంది?",
                "original_prompt": "How was your day?",
                "language": "Telugu",
                "script": "native",
                "category": "chitchat",
            }
        ],
        "indivibe_stem": [
            {
                "prompt": "Vaccine ela panichestundi?",
                "original_prompt": "How does a vaccine work?",
                "language": "Telugu",
                "script": "romanised",
                "category": "Public Health",
            }
        ],
    }

    def fake_load_hf_rows(source: DatasetConfig) -> list[dict[str, object]]:
        return rows_by_name[source.name]

    monkeypatch.setattr(loader, "load_hf_rows", fake_load_hf_rows)
    config = load_config(Path("config.yaml"))

    examples = load_training_examples(config)

    assert len(examples) == 5
    assert {example.source for example in examples} == {
        "telugu_alpaca",
        "symptom_diagnosis",
        "medmcqa",
        "indivibe_chat",
        "indivibe_stem",
    }
    symptom = next(example for example in examples if example.source == "symptom_diagnosis")
    medmcqa = next(example for example in examples if example.source == "medmcqa")
    assert "లక్షణాలు" in symptom.prompt
    assert "సంభావ్య నిర్ధారణ" in symptom.response
    assert "సరైన సమాధానం" in medmcqa.response


def test_field_dialogues_requires_existing_file() -> None:
    source = DatasetConfig(
        name="field_dialogues",
        enabled=True,
        weight=1.0,
        split="train",
        path="data/raw/missing.jsonl",
    )

    with pytest.raises(FileNotFoundError, match="Field dialogues file does not exist"):
        JsonlDataset(source).load()


def test_indivibe_filters_before_max_examples(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {
            "prompt": "Assamese prompt",
            "original_prompt": "Assamese source",
            "language": "Assamese",
            "script": "native",
            "category": "chitchat",
        },
        {
            "prompt": "తెలుగు ప్రశ్న",
            "original_prompt": "Telugu source",
            "language": "Telugu",
            "script": "native",
            "category": "chitchat",
        },
    ]

    monkeypatch.setattr(loader, "load_hf_rows", lambda source: rows)
    source = DatasetConfig(
        name="indivibe_chat",
        enabled=True,
        weight=1.0,
        split="test",
        hf_id="sarvamai/indivibe",
        hf_config="chat",
        max_examples=1,
        language="Telugu",
        scripts=("native",),
    )

    examples = loader.build_dataset(source).load()

    assert len(examples) == 1
    assert examples[0].response == "Telugu source"
