from gemma_health.config import load_config
from gemma_health.serving.manifest import build_serving_manifest


def test_serving_manifest_uses_android_asset_config() -> None:
    manifest = build_serving_manifest(load_config())
    assert manifest.model_asset == "gemma_health.task"
    assert manifest.runtime == "litert"
    assert manifest.no_internet_required is True
