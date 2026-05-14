from pathlib import Path

from gemma_health.config import load_config


def test_load_config() -> None:
    config = load_config(Path("config.yaml"))
    assert config.project.name == "gemma-health"
    assert config.model.max_seq_length == 2048
