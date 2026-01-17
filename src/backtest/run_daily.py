"""Run a daily factor backtest and emit signals + NAV curve."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.backtest import factors as factor_lib
from src.backtest.engine import apply_costs, compute_turnover
from src.backtest.scoring import score_cross_section
from src.backtest.selector import select_with_buffer
from src.reports.metrics import (
    ann_vol,
    cagr,
    max_drawdown,
    sharpe,
    turnover_stats,
    yearly_returns,
)
from src.utils.config import load_yaml
from src.utils.parquet_io import read_partitions

logger = logging.getLogger(__name__)


def _load_backtest_config(path: str | Path = "configs/backtest.yaml") -> dict:
    return load_yaml(path)


def _compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tr_mean20"] = factor_lib.tr_mean20(df)
    df["tr_ratio20"] = factor_lib.tr_ratio20(df)
    df["tr_stab20"] = factor_lib.tr_stab20(df)
    df["vr"] = factor_lib.vr(df)
    df["vr_dir"] = factor_lib.vr_dir(df)
    df["div_up20"] = factor_lib.div_up20(df)
    df["div_dn20"] = factor_lib.div_dn20(df)
    return df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily backtest")
    parser.add_argument("--start", help="Start date YYYYMMDD")
    parser.add_argument("--end", help="End date YYYYMMDD")
    parser.add_argument("--limit-codes", type=int, help="Limit number of codes")
    parser.add_argument(
        "--data-dir",
        default="data/lake/daily_bars",
        help="Daily bars parquet base directory",
    )
    parser.add_argument(
        "--config",
        default="configs/backtest.yaml",
        help="Backtest config path",
    )
    parser.add_argument(
        "--factors",
        default="configs/factors.yaml",
        help="Factors config path",
    )
    return parser.parse_args()


def _build_weights(codes: list[str]) -> dict[str, float]:
    if not codes:
        return {}
    weight = 1.0 / len(codes)
    return {code: weight for code in codes}


def _save_signals(date: str, weights: dict[str, float], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"ts_code": list(weights.keys()), "weight": list(weights.values())})
    df = df.sort_values("weight", ascending=False)
    output_path = output_dir / f"target_positions_{date}.csv"
    df.to_csv(output_path, index=False)


def run_backtest(args: argparse.Namespace) -> pd.DataFrame:
    config = _load_backtest_config(args.config)
    start_date = args.start or config.get("start_date")
    end_date = args.end or config.get("end_date")
    n_holdings = int(config.get("n_holdings", 30))
    buffer_size = int(config.get("buffer", 15))
    cost_bps = float(config.get("cost_bps", 0.0))

    if not start_date or not end_date:
        raise ValueError("start_date/end_date must be set via args or config")

    data = read_partitions(args.data_dir)
    if data.empty:
        raise RuntimeError("No daily bars data found")

    data = data.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    data = data[(data["trade_date"] >= start_date) & (data["trade_date"] <= end_date)]
    if data.empty:
        raise RuntimeError("No daily bars data found in the requested date range")

    # Filter to only use SZ and SH stocks (exclude BJ and other exchanges which may have data issues)
    data = data[data["ts_code"].str.endswith((".SZ", ".SH"))]

    if args.limit_codes:
        allowed_codes = sorted(data["ts_code"].unique())[: args.limit_codes]
        data = data[data["ts_code"].isin(allowed_codes)]

    data = _compute_factors(data)

    # FIX FUTURE FUNCTION:
    # Signal at Day T (using T close data) -> Buy at T+1 Open -> Sell at T+2 Open.
    # This prevents using Day T's signal to buy at Day T's Open (which happened earlier).
    g = data.groupby("ts_code", sort=False)["open"]
    data["open_T1"] = g.shift(-1)
    data["open_T2"] = g.shift(-2)
    data["realized_return"] = data["open_T2"] / data["open_T1"] - 1.0

    # Drop rows with NaN factors (insufficient history for rolling windows)
    factor_cols = ["tr_mean20", "tr_ratio20", "tr_stab20", "vr", "vr_dir", "div_up20", "div_dn20"]
    data = data.dropna(subset=factor_cols)

    # Clean data: remove extreme returns and price anomalies
    # Remove extreme returns (likely data errors, splits, etc.)
    data = data[(data["realized_return"].abs() <= 0.20)]  # Cap at ±20% daily return

    # Remove rows with invalid prices
    data = data[(data["open"] > 0) & (data["open_T1"] > 0) & (data["open_T2"] > 0)]
    data = data[(data["open"] < 1000) & (data["open_T1"] < 1000) & (data["open_T2"] < 1000)]

    # Remove IPO first day/last days (when return is NaN)
    data = data.dropna(subset=["realized_return"])

    # AGGRESSIVE FILTERING: Remove stocks with too many extreme price movements
    stock_stats = data.groupby("ts_code").agg({
        "realized_return": ["count", "std", "min", "max"],
        "open": "mean"
    }).reset_index()

    # Flatten column names
    stock_stats.columns = ["ts_code", "count", "return_std", "return_min", "return_max", "avg_price"]

    # Filter out stocks with:
    # - Too few observations
    # - Extremely high volatility
    # - Extreme min/max returns (even after filtering)
    stock_stats = stock_stats[
        (stock_stats["count"] >= 100) &  # At least 100 observations
        (stock_stats["return_std"] <= 0.10) &  # Return std <= 10% (STRICTER)
        (stock_stats["return_min"] >= -0.15) &  # Min return >= -15% (STRICTER)
        (stock_stats["return_max"] <= 0.15) &   # Max return <= 15% (STRICTER)
        (stock_stats["avg_price"] >= 2.0) &     # Average price >= 2 yuan (STRICTER)
        (stock_stats["avg_price"] <= 300)       # Average price <= 300 yuan (STRICTER)
    ]

    logger.info("Filtered from %s to %s stocks based on data quality", data["ts_code"].nunique(), len(stock_stats))

    # Keep only high-quality stocks
    data = data[data["ts_code"].isin(stock_stats["ts_code"])]

    if data.empty:
        raise RuntimeError("No data remains after aggressive filtering")

    trade_dates = sorted(data["trade_date"].unique())
    nav = 1.0
    prev_weights: dict[str, float] = {}
    records: list[dict[str, float | str]] = []

    signals_dir = Path("outputs/signals")
    for idx, trade_date in enumerate(trade_dates):
        day_slice = data[data["trade_date"] == trade_date].copy()
        day_slice = day_slice.dropna(subset=["open", "open_T1", "open_T2"])
        if day_slice.empty:
            continue

        day_slice["score"] = score_cross_section(day_slice, args.factors)
        ranked = day_slice.sort_values("score", ascending=False)
        sorted_codes = ranked["ts_code"].tolist()
        selected = select_with_buffer(sorted_codes, prev_weights.keys(), n=n_holdings, buffer=buffer_size)
        weights = _build_weights(selected)

        slice_selected = day_slice[day_slice["ts_code"].isin(selected)].copy()
        slice_selected["weight"] = slice_selected["ts_code"].map(weights)
        slice_selected = slice_selected.dropna(subset=["weight"])

        returns = slice_selected["realized_return"]
        gross_return = float((returns * slice_selected["weight"]).sum())
        turnover = compute_turnover(prev_weights, weights)
        net_return = apply_costs(gross_return, turnover, cost_bps)
        nav *= 1.0 + net_return

        # DEBUG: Log daily returns for first few days and if NAV becomes extreme
        if idx < 10 or nav > 10 or nav < 0.1:
            logger.info(f"DEBUG Day {trade_date}: NAV={nav:.6f}, GrossReturn={gross_return:.6f}, NetReturn={net_return:.6f}, Selected={len(selected)} stocks")

        records.append({"date": trade_date, "nav": nav, "turnover": turnover})
        _save_signals(trade_date, weights, signals_dir)
        prev_weights = weights

        if idx % 100 == 0:
            logger.info("Processed %s/%s trade dates", idx + 1, len(trade_dates))

    nav_df = pd.DataFrame(records)
    if nav_df.empty:
        raise RuntimeError("No NAV records generated")
    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    nav_path = reports_dir / "nav_curve.parquet"
    nav_df.to_parquet(nav_path, index=False)

    summary = {
        "start_date": str(nav_df["date"].iloc[0]),
        "end_date": str(nav_df["date"].iloc[-1]),
        "final_nav": float(nav_df["nav"].iloc[-1]),
        "cagr": cagr(nav_df["nav"]),
        "ann_vol": ann_vol(nav_df["nav"]),
        "sharpe": sharpe(nav_df["nav"]),
    }
    drawdown = max_drawdown(nav_df["nav"], nav_df["date"])
    summary["max_drawdown"] = {
        "value": drawdown.value,
        "start": drawdown.start,
        "end": drawdown.end,
    }
    summary["turnover"] = turnover_stats(nav_df["turnover"])

    yearly = yearly_returns(nav_df)
    yearly_path = reports_dir / "yearly_returns.csv"
    yearly.to_csv(yearly_path, index=False)

    summary_path = reports_dir / "summary.json"
    pd.Series(summary).to_json(summary_path, indent=2)

    avg_turnover = nav_df["turnover"].mean()
    final_nav = nav_df["nav"].iloc[-1]
    logger.info("Summary: final_nav=%.4f avg_turnover=%.4f", final_nav, avg_turnover)
    print(f"summary final_nav={final_nav:.4f} avg_turnover={avg_turnover:.4f}")

    return nav_df


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    try:
        run_backtest(args)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.error("Backtest failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
