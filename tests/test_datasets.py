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
        "english_telugu_parallel",
        "samanantar_te",
        "indivibe_chat",
        "indivibe_stem",
    ]
    assert {source.name: source.max_examples for source in sources} == {
        "telugu_alpaca": None,
        "english_telugu_parallel": 20000,
        "samanantar_te": 20000,
        "indivibe_chat": None,
        "indivibe_stem": None,
    }
    assert {source.name: source.language_status for source in sources} == {
        "telugu_alpaca": "native_telugu",
        "english_telugu_parallel": "translation_pair",
        "samanantar_te": "translation_pair",
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
                "telugu_transliterated_instruction": "lakshanalu cheppandi",
                "telugu_transliterated_input": "",
                "telugu_transliterated_output": "PHC ki vellandi",
            }
        ],
        "english_telugu_parallel": [{"english": "Hello", "telugu": "హలో"}],
        "samanantar_te": [{"src": "Go to school.", "tgt": "పాఠశాలకు వెళ్ళు."}],
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

    assert len(examples) == 6
    assert {example.source for example in examples} == {
        "telugu_alpaca",
        "english_telugu_parallel",
        "samanantar_te",
        "indivibe_chat",
        "indivibe_stem",
    }
    parallel = next(example for example in examples if example.source == "english_telugu_parallel")
    samanantar = next(example for example in examples if example.source == "samanantar_te")
    alpaca = [example for example in examples if example.source == "telugu_alpaca"]
    indivibe = next(example for example in examples if example.source == "indivibe_stem")
    assert len(alpaca) == 2
    assert any("romanised Telugu" in example.prompt for example in alpaca)
    assert any(example.response == "PHC ki vellandi" for example in alpaca)
    assert "natural Telugu script" in parallel.prompt
    assert parallel.response == "హలో"
    assert samanantar.response == "పాఠశాలకు వెళ్ళు."
    assert "Expected script: romanised" in indivibe.prompt
    assert indivibe.response == "Vaccine ela panichestundi?"


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


def test_local_parquet_dataset_respects_max_examples(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    path = tmp_path / "parallel.parquet"
    pd.DataFrame(
        [
            {"english": "one", "telugu": "ఒకటి"},
            {"english": "two", "telugu": "రెండు"},
        ]
    ).to_parquet(path, index=False)
    source = DatasetConfig(
        name="english_telugu_parallel",
        enabled=True,
        weight=1.0,
        split="train",
        path=str(path),
        max_examples=1,
        language_status="translation_pair",
    )

    examples = loader.build_dataset(source).load()

    assert len(examples) == 1
    assert examples[0].response == "ఒకటి"


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
    assert examples[0].prompt.startswith("Rewrite this English benchmark prompt in Telugu.")
    assert examples[0].response == "తెలుగు ప్రశ్న"
