from __future__ import annotations

import argparse

from gemma_health.config import load_config
from gemma_health.serving.manifest import build_serving_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    manifest = build_serving_manifest(load_config(args.config))
    print(manifest)


if __name__ == "__main__":
    main()
