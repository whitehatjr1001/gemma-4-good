from __future__ import annotations

import argparse

from gemma_health.config import load_config
from gemma_health.serving.export import export_for_serving


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    export_for_serving(load_config(args.config))


if __name__ == "__main__":
    main()
