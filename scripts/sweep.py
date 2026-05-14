from __future__ import annotations

import argparse

from gemma_health.config import load_config
from gemma_health.scaling.run import run_scaling_sweep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run_scaling_sweep(load_config(args.config))


if __name__ == "__main__":
    main()
