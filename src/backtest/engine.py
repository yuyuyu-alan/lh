"""Backtest engine helpers."""

from __future__ import annotations

from collections.abc import Mapping


def compute_turnover(prev_weights: Mapping[str, float], next_weights: Mapping[str, float]) -> float:
    keys = set(prev_weights) | set(next_weights)
    return sum(abs(next_weights.get(k, 0.0) - prev_weights.get(k, 0.0)) for k in keys)


def apply_costs(gross_return: float, turnover: float, cost_bps: float) -> float:
    cost = turnover * cost_bps / 10000.0
    return gross_return - cost
