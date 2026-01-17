"""Ingest daily bars from TuShare pro_bar with resume support."""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path
from typing import Iterable

import pandas as pd
import tushare as ts

from src.ingest.stock_basic import filter_main_board_codes
from src.ingest.tushare_client import get_pro
from src.utils.parquet_io import append_partition, normalize_keys
from src.utils.retry import throttle, with_retry

logger = logging.getLogger(__name__)

PROGRESS_COLUMNS = ["ts_code", "year"]


def _load_stock_codes(
    stock_basic_path: str | Path = "data/lake/basics/stock_basic.parquet",
    limit_codes: int | None = None,
) -> list[str]:
    stock_path = Path(stock_basic_path)
    if not stock_path.exists():
        raise FileNotFoundError(f"stock_basic file not found: {stock_path}")

    data = pd.read_parquet(stock_path)
    codes = filter_main_board_codes(data)
    if limit_codes is not None:
        codes = codes[:limit_codes]
    if not codes:
        raise RuntimeError("No stock codes available after filtering")
    return codes


def _load_trade_calendar(
    trade_cal_path: str | Path = "data/lake/calendar/trade_cal.parquet",
) -> pd.DataFrame:
    calendar_path = Path(trade_cal_path)
    if not calendar_path.exists():
        raise FileNotFoundError(f"trade_cal file not found: {calendar_path}")
    return pd.read_parquet(calendar_path)


def _iter_recent_trade_dates(calendar: pd.DataFrame, n: int) -> list[str]:
    if "cal_date" not in calendar.columns or "is_open" not in calendar.columns:
        raise ValueError("trade_cal data must include cal_date and is_open")
    open_dates = calendar.loc[calendar["is_open"].astype(str) == "1", "cal_date"]
    dates = open_dates.astype(str).sort_values().tolist()[-n:]
    if not dates:
        raise RuntimeError("No open trade dates found in trade_cal")
    return dates


def _load_progress(progress_path: Path) -> set[tuple[str, str]]:
    if not progress_path.exists():
        return set()
    df = pd.read_csv(progress_path, dtype=str)
    if not set(PROGRESS_COLUMNS).issubset(df.columns):
        raise ValueError("Progress file missing required columns")
    return set(zip(df["ts_code"], df["year"]))


def _append_progress(progress_path: Path, entries: Iterable[tuple[str, str]]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(entries, columns=PROGRESS_COLUMNS)
    header = not progress_path.exists()
    df.to_csv(progress_path, mode="a", header=header, index=False)


def _fetch_daily_bars(ts_code: str, start_date: str, end_date: str, adj: str) -> pd.DataFrame:
    logger.info("Fetching daily bars for %s (%s - %s)", ts_code, start_date, end_date)
    data = ts.pro_bar(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        freq="D",
        asset="E",
        adj=adj,
        adjfactor=True,
    )
    if data is None or data.empty:
        message = f"No daily bars returned for {ts_code} {start_date}-{end_date}"
        logger.error(message)
        raise RuntimeError(message)
    return data


def _year_bounds(year: int, start_date: str, end_date: str) -> tuple[str, str]:
    year_start = f"{year}0101"
    year_end = f"{year}1231"
    return max(start_date, year_start), min(end_date, year_end)


def ingest_daily_bars_full(
    start_date: str,
    end_date: str,
    years: list[int] | None = None,
    adj: str = "qfq",
    workers: int = 3,
    limit_codes: int | None = None,
    stock_basic_path: str | Path = "data/lake/basics/stock_basic.parquet",
    out_dir: str | Path = "data/lake/daily_bars",
    progress_path: str | Path = "logs/progress_daily_bars.csv",
) -> list[Path]:
    """Ingest daily bars for each code/year with resume support."""
    get_pro()
    codes = _load_stock_codes(stock_basic_path, limit_codes=limit_codes)
    if years is None:
        years = list(range(int(start_date[:4]), int(end_date[:4]) + 1))

    progress_file = Path(progress_path)
    completed = _load_progress(progress_file)
    completed_lock = threading.Lock()
    written: list[Path] = []

    tasks: list[tuple[str, int]] = [(code, year) for code in codes for year in years]

    def _handle_task(task: tuple[str, int]) -> list[Path]:
        ts_code, year = task
        if (ts_code, str(year)) in completed:
            logger.info("Skipping %s %s (already completed)", ts_code, year)
            return []
        year_start, year_end = _year_bounds(year, start_date, end_date)
        data = with_retry(lambda: _fetch_daily_bars(ts_code, year_start, year_end, adj))
        normalized = normalize_keys(data)
        paths = append_partition(normalized, out_dir, year_col="trade_date")
        with completed_lock:
            _append_progress(progress_file, [(ts_code, str(year))])
            completed.add((ts_code, str(year)))
        throttle()
        return paths

    if workers < 1:
        raise ValueError("workers must be at least 1")

    if workers == 1:
        for task in tasks:
            written.extend(_handle_task(task))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_handle_task, task) for task in tasks]
            for future in as_completed(futures):
                written.extend(future.result())

    return written


def ingest_daily_bars_recent(
    n: int = 10,
    adj: str = "qfq",
    workers: int = 3,
    limit_codes: int | None = None,
    stock_basic_path: str | Path = "data/lake/basics/stock_basic.parquet",
    trade_cal_path: str | Path = "data/lake/calendar/trade_cal.parquet",
    out_dir: str | Path = "data/lake/daily_bars",
) -> list[Path]:
    """Ingest daily bars for the most recent n trade dates for each code."""
    if n < 1:
        raise ValueError("n must be at least 1")
    if workers < 1:
        raise ValueError("workers must be at least 1")

    get_pro()
    codes = _load_stock_codes(stock_basic_path, limit_codes=limit_codes)
    calendar = _load_trade_calendar(trade_cal_path)
    dates = _iter_recent_trade_dates(calendar, n)
    start_date = dates[0]
    end_date = dates[-1]
    written: list[Path] = []

    def _handle_code(ts_code: str) -> list[Path]:
        data = with_retry(lambda: _fetch_daily_bars(ts_code, start_date, end_date, adj))
        normalized = normalize_keys(data)
        paths = append_partition(normalized, out_dir, year_col="trade_date")
        throttle()
        return paths

    if workers == 1:
        for code in codes:
            written.extend(_handle_code(code))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_handle_code, code) for code in codes]
            for future in as_completed(futures):
                written.extend(future.result())

    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest daily bars via TuShare pro_bar")
    parser.add_argument("--mode", choices=["full", "recent"], required=True)
    parser.add_argument("--start", help="Start date YYYYMMDD")
    parser.add_argument("--end", help="End date YYYYMMDD")
    parser.add_argument("--years", nargs="*", type=int, help="Years to ingest")
    parser.add_argument("--adj", default="qfq", help="Adjust type (default: qfq)")
    parser.add_argument("--workers", type=int, default=3, help="Worker threads")
    parser.add_argument("--limit-codes", type=int, help="Limit number of codes")
    parser.add_argument("--n", type=int, default=10, help="Recent trade dates to pull")
    parser.add_argument(
        "--stock-basic",
        default="data/lake/basics/stock_basic.parquet",
        help="Path to stock_basic parquet",
    )
    parser.add_argument(
        "--trade-cal",
        default="data/lake/calendar/trade_cal.parquet",
        help="Path to trade_cal parquet",
    )
    parser.add_argument(
        "--out-dir",
        default="data/lake/daily_bars",
        help="Output directory for parquet partitions",
    )
    parser.add_argument(
        "--progress",
        default="logs/progress_daily_bars.csv",
        help="Progress CSV path",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    try:
        if args.mode == "full":
            if not args.start or not args.end:
                raise ValueError("--start and --end are required for full mode")
            ingest_daily_bars_full(
                args.start,
                args.end,
                years=args.years,
                adj=args.adj,
                workers=args.workers,
                limit_codes=args.limit_codes,
                stock_basic_path=args.stock_basic,
                out_dir=args.out_dir,
                progress_path=args.progress,
            )
        else:
            ingest_daily_bars_recent(
                n=args.n,
                adj=args.adj,
                workers=args.workers,
                limit_codes=args.limit_codes,
                stock_basic_path=args.stock_basic,
                trade_cal_path=args.trade_cal,
                out_dir=args.out_dir,
            )
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.error("Daily bars ingestion failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
