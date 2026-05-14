from __future__ import annotations


def telugu_density_reward(response: str) -> float:
    if not response:
        return 0.0
    telugu_chars = sum(1 for char in response if "\u0c00" <= char <= "\u0c7f")
    return min(telugu_chars / len(response), 1.0)
