from __future__ import annotations

from pathlib import Path

import pandas as pd

from gemma_health.data.synthetic_telugu import generate_asha_dialogue, generate_synthetic_parquet
from gemma_health.data import synthetic_telugu


class FakeOllama:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, **kwargs: object) -> dict[str, object]:
        self.calls += 1
        return {
            "message": {
                "content": (
                    '{"telugu":"లక్షణాల విశ్లేషణ: జ్వరం\\nరిస్క్ స్థాయి: MEDIUM\\nచర్య: PHC",'
                    '"romanised_telugu":"lakshanala vishleshana: jvaram\\nrisk sthayi: MEDIUM\\ncharya: PHC"}'
                )
            }
        }


def test_generate_asha_dialogue_uses_ollama_client() -> None:
    client = FakeOllama()

    pair = generate_asha_dialogue("high fever", client=client)

    assert "రిస్క్ స్థాయి" in pair.telugu
    assert "risk sthayi" in pair.romanised_telugu
    assert client.calls == 1


def test_generate_synthetic_parquet_adds_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.parquet"
    output_path = tmp_path / "out.parquet"
    pd.DataFrame([{"input_text": "high fever", "output_text": "dengue"}]).to_parquet(input_path, index=False)

    result = generate_synthetic_parquet(
        input_path=input_path,
        output_path=output_path,
        dataset_name="symptom_diagnosis",
        limit=1,
        client=FakeOllama(),
    )

    frame = pd.read_parquet(output_path)
    assert result.row_count == 1
    assert frame["synthetic_task"].tolist() == ["asha_triage_dialogue"]
    assert "synthetic_telugu" in frame.columns
    assert "telugu" in frame.columns
    assert "romanised_telugu" in frame.columns


def test_generate_synthetic_parquet_resumes_partial_output(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.parquet"
    output_path = tmp_path / "out.parquet"
    pd.DataFrame(
        [
            {"input_text": "high fever", "output_text": "dengue"},
            {"input_text": "rash", "output_text": "impetigo"},
        ]
    ).to_parquet(input_path, index=False)
    pd.DataFrame(
        [
            {
                "input_text": "high fever",
                "output_text": "dengue",
                "telugu": "మొదటి",
                "romanised_telugu": "modati",
                "synthetic_telugu": "మొదటి",
                "synthetic_task": "asha_triage_dialogue",
                "synthetic_model": "qwen3:30b-a3b",
            }
        ]
    ).to_parquet(output_path, index=False)

    result = generate_synthetic_parquet(
        input_path=input_path,
        output_path=output_path,
        dataset_name="symptom_diagnosis",
        client=FakeOllama(),
        checkpoint_interval=1,
        workers=2,
    )

    frame = pd.read_parquet(output_path)
    assert result.row_count == 2
    assert frame["telugu"].tolist()[0] == "మొదటి"
    assert len(frame) == 2


def test_message_pair_falls_back_when_json_missing() -> None:
    pair = synthetic_telugu._message_pair({"message": {"content": "తెలుగు వాక్యం"}})

    assert pair.telugu == "తెలుగు వాక్యం"
    assert pair.romanised_telugu == "NEEDS_REVIEW"


def test_medmcqa_prompt_tolerates_missing_correct_option() -> None:
    prompt = synthetic_telugu._format_medmcqa_prompt(
        {
            "question": "Question?",
            "opa": "A",
            "opb": "B",
            "opc": "C",
            "opd": "D",
            "cop": -1,
            "exp": "Explanation",
        }
    )

    assert "Correct answer: unknown" in prompt
