"""Ingest trade calendar data from TuShare."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.ingest.tushare_client import get_pro

logger = logging.getLogger(__name__)


def ingest_trade_cal(
    start_date: str,
    end_date: str,
    exchange: str = "SSE",
    out_path: str | Path = "data/lake/calendar/trade_cal.parquet",
) -> Path:
    """Fetch trade calendar data and persist it to a parquet file."""
    pro = get_pro()
    logger.info("Fetching trade calendar: %s to %s (%s)", start_date, end_date, exchange)
    data = pro.trade_cal(exchange=exchange, start_date=start_date, end_date=end_date)
    if data is None or data.empty:
        message = "No trade calendar data returned"
        logger.error(message)
        raise RuntimeError(message)

    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output_path, index=False)
    logger.info("Saved %s rows to %s", len(data), output_path)
    return output_path


def open_trade_dates(df: pd.DataFrame) -> list[str]:
    """Return open trading dates where is_open == '1'."""
    if "is_open" not in df.columns or "cal_date" not in df.columns:
        raise ValueError("trade_cal data must include cal_date and is_open")
    open_dates = df.loc[df["is_open"].astype(str) == "1", "cal_date"]
    return open_dates.astype(str).tolist()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest trade calendar data")
    parser.add_argument("--start", required=True, help="Start date YYYYMMDD")
    parser.add_argument("--end", required=True, help="End date YYYYMMDD")
    parser.add_argument("--exchange", default="SSE", help="Exchange code (default: SSE)")
    parser.add_argument(
        "--out",
        default="data/lake/calendar/trade_cal.parquet",
        help="Output parquet path",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    try:
        ingest_trade_cal(args.start, args.end, exchange=args.exchange, out_path=args.out)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.error("Trade calendar ingestion failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
