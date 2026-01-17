"""Ingest daily basic data from TuShare."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.ingest.tushare_client import get_pro
from src.utils.parquet_io import append_partition, normalize_keys
from src.utils.retry import throttle, with_retry

logger = logging.getLogger(__name__)

FIELDS = "ts_code,trade_date,turnover_rate,volume_ratio,pe,pb,total_mv,close"


def _load_trade_calendar(path: str | Path) -> pd.DataFrame:
    calendar_path = Path(path)
    if not calendar_path.exists():
        raise FileNotFoundError(f"trade_cal file not found: {calendar_path}")
    return pd.read_parquet(calendar_path)


def _iter_trade_dates(df: pd.DataFrame, start_date: str, end_date: str) -> list[str]:
    if "cal_date" not in df.columns or "is_open" not in df.columns:
        raise ValueError("trade_cal data must include cal_date and is_open")
    mask = (df["is_open"].astype(str) == "1") & (df["cal_date"] >= start_date) & (
        df["cal_date"] <= end_date
    )
    return df.loc[mask, "cal_date"].astype(str).tolist()


def _fetch_daily_basic(trade_date: str) -> pd.DataFrame:
    pro = get_pro()
    logger.info("Fetching daily_basic for %s", trade_date)
    data = pro.daily_basic(trade_date=trade_date, fields=FIELDS)
    if data is None or data.empty:
        message = f"No daily_basic data returned for {trade_date}"
        logger.error(message)
        raise RuntimeError(message)
    return data


def _warn_if_low_rows(trade_date: str, df: pd.DataFrame, threshold: int = 100) -> None:
    if len(df) < threshold:
        logger.warning("daily_basic rows for %s below threshold: %s", trade_date, len(df))


def ingest_daily_basic_full(
    start_date: str,
    end_date: str,
    trade_cal_path: str | Path = "data/lake/calendar/trade_cal.parquet",
    out_dir: str | Path = "data/lake/daily_basic",
) -> list[Path]:
    """Ingest daily basic data for all open dates in a range."""
    calendar = _load_trade_calendar(trade_cal_path)
    trade_dates = _iter_trade_dates(calendar, start_date, end_date)
    if not trade_dates:
        message = "No open trade dates found in the requested range"
        logger.error(message)
        raise RuntimeError(message)

    written: list[Path] = []
    for trade_date in trade_dates:
        data = with_retry(lambda: _fetch_daily_basic(trade_date))
        _warn_if_low_rows(trade_date, data)
        normalized = normalize_keys(data)
        written.extend(append_partition(normalized, out_dir, year_col="trade_date"))
        throttle()

    return written


def ingest_daily_basic_recent(
    n: int = 10,
    trade_cal_path: str | Path = "data/lake/calendar/trade_cal.parquet",
    out_dir: str | Path = "data/lake/daily_basic",
) -> list[Path]:
    """Ingest daily basic data for the most recent n open dates."""
    if n < 1:
        raise ValueError("n must be at least 1")

    calendar = _load_trade_calendar(trade_cal_path)
    open_dates = calendar.loc[calendar["is_open"].astype(str) == "1", "cal_date"]
    trade_dates = open_dates.astype(str).sort_values().tolist()[-n:]
    if not trade_dates:
        message = "No open trade dates found in trade_cal"
        logger.error(message)
        raise RuntimeError(message)

    written: list[Path] = []
    for trade_date in trade_dates:
        data = with_retry(lambda: _fetch_daily_basic(trade_date))
        _warn_if_low_rows(trade_date, data)
        normalized = normalize_keys(data)
        written.extend(append_partition(normalized, out_dir, year_col="trade_date"))
        throttle()

    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest daily_basic data")
    parser.add_argument("--mode", choices=["full", "recent"], required=True)
    parser.add_argument("--start", help="Start date YYYYMMDD")
    parser.add_argument("--end", help="End date YYYYMMDD")
    parser.add_argument("--n", type=int, default=10, help="Number of recent trade dates")
    parser.add_argument(
        "--trade-cal",
        default="data/lake/calendar/trade_cal.parquet",
        help="Path to trade calendar parquet",
    )
    parser.add_argument(
        "--out-dir",
        default="data/lake/daily_basic",
        help="Output directory for parquet partitions",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    try:
        if args.mode == "full":
            if not args.start or not args.end:
                raise ValueError("--start and --end are required for full mode")
            ingest_daily_basic_full(
                args.start, args.end, trade_cal_path=args.trade_cal, out_dir=args.out_dir
            )
        else:
            ingest_daily_basic_recent(
                n=args.n, trade_cal_path=args.trade_cal, out_dir=args.out_dir
            )
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.error("Daily basic ingestion failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
