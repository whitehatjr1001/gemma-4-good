from __future__ import annotations


def build_mediapipe_preprocess_manifest() -> dict[str, bool]:
    return {
        "image_preprocess": True,
        "prescription_ocr_preprocess": True,
    }
