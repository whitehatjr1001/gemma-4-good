from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComputeBudget:
    gpu: str
    hours: float
    max_examples: int
