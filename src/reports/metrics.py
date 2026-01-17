"""Performance metrics for backtest reports."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Drawdown:
    value: float
    start: str
    end: str


def _daily_returns(nav: pd.Series) -> pd.Series:
    returns = nav.pct_change().dropna()
    return returns


def cagr(nav: pd.Series, trading_days: int = 252) -> float:
    if nav.empty:
        raise ValueError("nav series is empty")
    total_return = nav.iloc[-1] / nav.iloc[0] - 1.0
    periods = max(len(nav) - 1, 1)
    years = periods / trading_days
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def max_drawdown(nav: pd.Series, dates: pd.Series) -> Drawdown:
    if nav.empty:
        raise ValueError("nav series is empty")
    running_max = nav.cummax()
    drawdowns = nav / running_max - 1.0
    min_idx = drawdowns.idxmin()
    start_idx = nav[: min_idx + 1].idxmax()
    return Drawdown(
        value=float(drawdowns.loc[min_idx]),
        start=str(dates.loc[start_idx]),
        end=str(dates.loc[min_idx]),
    )


def ann_vol(nav: pd.Series, trading_days: int = 252) -> float:
    returns = _daily_returns(nav)
    if returns.empty:
        raise ValueError("not enough data to compute volatility")
    return float(returns.std(ddof=0) * math.sqrt(trading_days))


def sharpe(nav: pd.Series, risk_free_rate: float = 0.0, trading_days: int = 252) -> float:
    returns = _daily_returns(nav)
    if returns.empty:
        raise ValueError("not enough data to compute sharpe")
    excess = returns - risk_free_rate / trading_days
    std = excess.std(ddof=0)
    if std == 0 or pd.isna(std):
        return 0.0
    return float(excess.mean() / std * math.sqrt(trading_days))


def turnover_stats(turnover: pd.Series) -> dict[str, float]:
    if turnover.empty:
        raise ValueError("turnover series is empty")
    return {
        "avg": float(turnover.mean()),
        "median": float(turnover.median()),
        "max": float(turnover.max()),
    }


def yearly_returns(nav_df: pd.DataFrame) -> pd.DataFrame:
    if nav_df.empty:
        raise ValueError("nav data is empty")
    nav_df = nav_df.copy()
    nav_df["year"] = nav_df["date"].astype(str).str.slice(0, 4)
    grouped = nav_df.groupby("year", sort=True)
    summary = grouped.apply(lambda g: g["nav"].iloc[-1] / g["nav"].iloc[0] - 1.0, include_groups=False)
    return summary.reset_index(name="return")
