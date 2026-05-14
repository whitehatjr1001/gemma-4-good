from __future__ import annotations

from gemma_health.config import load_config


def main() -> None:
    config = load_config()
    print(f"{config.project.name}: {config.model.base_id}")
