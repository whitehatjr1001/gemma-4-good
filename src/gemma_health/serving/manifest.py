from __future__ import annotations

from dataclasses import dataclass

from gemma_health.config import AppConfig


@dataclass(frozen=True)
class ServingManifest:
    model_asset: str
    runtime: str
    no_internet_required: bool


def build_serving_manifest(config: AppConfig) -> ServingManifest:
    serving = config.raw["serving"]
    android = serving["android"]
    return ServingManifest(
        model_asset=str(android["asset_model_name"]),
        runtime=str(serving["runtime"]),
        no_internet_required=bool(android["require_no_internet_permission"]),
    )
