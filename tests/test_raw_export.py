from __future__ import annotations

from pathlib import Path

from gemma_health.config import load_config
from gemma_health.datasets.raw_export import export_raw_dataset, export_raw_datasets, raw_export_sources
from gemma_health.datasets.base import DatasetConfig
from gemma_health.datasets import raw_export


class FakeDataset:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def filter(self, predicate: object) -> "FakeDataset":
        return FakeDataset([row for row in self.rows if predicate(row)])

    def select(self, indexes: range) -> "FakeDataset":
        return FakeDataset([self.rows[index] for index in indexes])

    def to_parquet(self, path: str) -> None:
        Path(path).write_text(str(len(self.rows)), encoding="utf-8")


def test_raw_export_sources_excludes_native_telugu() -> None:
    config = load_config(Path("config.yaml"))

    sources = raw_export_sources(config)

    assert [source.name for source in sources] == [
        "symptom_diagnosis",
        "medmcqa",
        "indivibe_chat",
        "indivibe_stem",
    ]


def test_raw_export_sources_can_select_synthetic_telugu_sources() -> None:
    config = load_config(Path("config.yaml"))

    sources = raw_export_sources(config, language_statuses=("synthetic_telugu",))

    assert [source.name for source in sources] == ["symptom_diagnosis", "medmcqa"]


def test_export_raw_dataset_filters_and_writes_parquet(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    dataset = FakeDataset(
        [
            {"language": "Assamese", "script": "native", "prompt": "x"},
            {"language": "Telugu", "script": "native", "prompt": "y"},
            {"language": "Telugu", "script": "romanised", "prompt": "z"},
        ]
    )
    monkeypatch.setattr(raw_export, "_load_hf_dataset", lambda source: dataset)
    source = DatasetConfig(
        name="indivibe_chat",
        enabled=True,
        weight=1.0,
        split="test",
        hf_id="sarvamai/indivibe",
        hf_config="chat",
        language="Telugu",
        scripts=("native", "romanised"),
        language_status="translation_pair",
    )

    export = export_raw_dataset(source, tmp_path, max_rows=1)

    assert export.row_count == 1
    assert export.output_path == tmp_path / "indivibe_chat" / "test.parquet"
    assert export.output_path.read_text(encoding="utf-8") == "1"


def test_export_raw_datasets_can_export_all_splits(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    monkeypatch.setattr(raw_export, "_split_sources", lambda source: [source, source])
    monkeypatch.setattr(raw_export, "_load_hf_dataset", lambda source: FakeDataset([{"text": "x"}]))
    source = DatasetConfig(
        name="symptom_diagnosis",
        enabled=True,
        weight=1.0,
        split="train",
        hf_id="gretelai/symptom_to_diagnosis",
        language_status="synthetic_telugu",
    )

    exports = export_raw_datasets([source], tmp_path, all_splits=True)

    assert len(exports) == 2
    assert all(export.row_count == 1 for export in exports)
