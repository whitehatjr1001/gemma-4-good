from __future__ import annotations

from gemma_health.rewards.safety import safety_escalation_reward
from gemma_health.rewards.telugu import telugu_density_reward
from gemma_health.rewards.triage import triage_format_reward


def combined_triage_reward(response: str, true_risk: str) -> float:
    word_count = len(response.split())
    brevity = 1.0 if word_count <= 100 else max(0.0, 1.0 - (word_count - 100) / 100)
    return (
        triage_format_reward(response)
        + safety_escalation_reward(response, true_risk)
        + telugu_density_reward(response)
        + brevity
    )
