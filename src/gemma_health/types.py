from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NewType


DatasetName = NewType("DatasetName", str)
RunName = NewType("RunName", str)
RiskLevel = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class TrainingExample:
    prompt: str
    response: str
    source: str
    variant: str = "default"


@dataclass(frozen=True)
class TriageLabel:
    risk: RiskLevel
    requires_referral: bool
