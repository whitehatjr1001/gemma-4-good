from __future__ import annotations


def triage_format_reward(response: str) -> float:
    has_symptoms = any(token in response for token in ("లక్షణాలు", "symptom"))
    has_risk = any(token in response for token in ("రిస్క్", "అత్యవసరం", "high", "medium", "low"))
    has_action = any(token in response for token in ("PHC", "ఆసుపత్రి", "hospital", "విశ్రాంతి"))
    return 1.0 if has_symptoms and has_risk and has_action else 0.0
