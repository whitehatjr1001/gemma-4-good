from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gemma_health.data.sft import (
    load_sft_jsonl_sample,
    resolve_synthetic_inputs,
    synthetic_parquet_to_sft_examples,
    training_example_to_sft,
    training_examples_to_sft,
    validate_sft_jsonl,
    write_sft_jsonl,
)
from gemma_health.types import TrainingExample


def test_training_example_to_sft_emits_messages_and_text() -> None:
    example = training_example_to_sft(
        TrainingExample(prompt="Symptoms: fever", response="రిస్క్ స్థాయి: MEDIUM", source="smoke")
    )

    assert [message["role"] for message in example.messages] == ["system", "user", "assistant"]
    assert example.messages[-1]["content"] == "రిస్క్ స్థాయి: MEDIUM"
    assert "<|assistant|>\nరిస్క్ స్థాయి: MEDIUM" in example.text


def test_synthetic_parquet_to_sft_examples_reads_telugu_and_romanised(tmp_path: Path) -> None:
    input_path = tmp_path / "symptom_diagnosis" / "train.parquet"
    input_path.parent.mkdir()
    pd.DataFrame(
        [
            {
                "input_text": "high fever",
                "output_text": "flu",
                "telugu": "లక్షణాల విశ్లేషణ: జ్వరం",
                "romanised_telugu": "lakshanala vishleshana: jvaram",
            }
        ]
    ).to_parquet(input_path, index=False)

    examples = synthetic_parquet_to_sft_examples(
        [input_path],
        response_columns=("telugu", "romanised_telugu"),
    )

    assert len(examples) == 2
    assert examples[0].source == "symptom_diagnosis"
    assert examples[0].variant == "telugu"
    assert examples[1].variant == "romanised_telugu"
    assert "Patient symptoms:" in examples[0].prompt


def test_synthetic_parquet_limit_counts_input_rows(tmp_path: Path) -> None:
    input_path = tmp_path / "symptom_diagnosis" / "train.parquet"
    input_path.parent.mkdir()
    pd.DataFrame(
        [
            {
                "input_text": "high fever",
                "telugu": "తెలుగు ఒకటి",
                "romanised_telugu": "telugu okati",
            },
            {
                "input_text": "rash",
                "telugu": "తెలుగు రెండు",
                "romanised_telugu": "telugu rendu",
            },
        ]
    ).to_parquet(input_path, index=False)

    examples = synthetic_parquet_to_sft_examples(
        [input_path],
        response_columns=("telugu", "romanised_telugu"),
        limit=1,
    )

    assert [example.variant for example in examples] == ["telugu", "romanised_telugu"]


def test_write_sft_jsonl(tmp_path: Path) -> None:
    output_path = tmp_path / "sft.jsonl"
    examples = [
        training_example_to_sft(TrainingExample(prompt="Q", response="A", source="unit")),
    ]

    count = write_sft_jsonl(examples, output_path)

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert count == 1
    assert rows[0]["messages"][1]["content"] == "Q"
    assert validate_sft_jsonl(output_path) == 1
    assert load_sft_jsonl_sample(output_path, limit=1)[0]["source"] == "unit"


def test_training_examples_to_sft_converts_multiple_examples() -> None:
    examples = training_examples_to_sft(
        [
            TrainingExample(prompt="Q1", response="A1", source="one"),
            TrainingExample(prompt="Q2", response="A2", source="two"),
        ],
        variant="native_telugu",
    )

    assert [example.source for example in examples] == ["one", "two"]
    assert {example.variant for example in examples} == {"native_telugu"}


def test_resolve_synthetic_inputs_accepts_directory(tmp_path: Path) -> None:
    input_path = tmp_path / "synthetic" / "symptom_diagnosis" / "train.parquet"
    input_path.parent.mkdir(parents=True)
    pd.DataFrame([{"input_text": "fever", "telugu": "తెలుగు"}]).to_parquet(input_path, index=False)

    paths = resolve_synthetic_inputs([str(tmp_path / "synthetic")])

    assert paths == [input_path]
