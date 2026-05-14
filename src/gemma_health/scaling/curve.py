from __future__ import annotations


def fit_loss_curve(points: list[tuple[int, float]]) -> tuple[float, float]:
    if len(points) < 2:
        raise ValueError("Need at least two points to fit a scaling curve.")
    raise NotImplementedError("Scaling curve fitting is not implemented yet.")
