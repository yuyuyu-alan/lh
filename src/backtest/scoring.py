"""Scoring utilities for factor-based ranking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import load_yaml

logger = logging.getLogger(__name__)


def winsorize(series: pd.Series, lower_q: float = 0.05, upper_q: float = 0.95) -> pd.Series:
    if series.empty:
        return series
    lower = series.quantile(lower_q)
    upper = series.quantile(upper_q)
    return series.clip(lower, upper)


def zscore(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return series * 0
    return (series - mean) / std


def _load_factor_config(path: str | Path) -> list[dict[str, Any]]:
    config = load_yaml(path)
    factors = config.get("factors", [])
    if not factors:
        raise ValueError("No factors defined in factors config")
    return factors


def score_cross_section(
    df: pd.DataFrame,
    factor_config_path: str | Path = "configs/factors.yaml",
) -> pd.Series:
    """Return a weighted score for each row in a single-date cross section."""
    factors = _load_factor_config(factor_config_path)
    scores = pd.Series(0.0, index=df.index)

    for factor in factors:
        name = factor.get("name")
        weight = float(factor.get("weight", 0.0))
        enabled = factor.get("enabled", True)
        if not enabled or weight == 0:
            continue
        if name not in df.columns:
            logger.warning("Factor %s missing from data", name)
            continue
        cleaned = winsorize(df[name])
        scores = scores + zscore(cleaned) * weight

    return scores
