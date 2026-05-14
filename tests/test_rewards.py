from gemma_health.rewards.combined import combined_triage_reward


def test_combined_triage_reward_rewards_emergency_telugu_response() -> None:
    response = "లక్షణాలు: జ్వరం. రిస్క్: high. అత్యవసరం: PHC కి వెళ్ళండి."
    assert combined_triage_reward(response, "high") > 3.0
