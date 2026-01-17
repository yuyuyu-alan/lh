"""Ingest stock basic data from TuShare."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.ingest.tushare_client import get_pro

logger = logging.getLogger(__name__)


def ingest_stock_basic(
    out_path: str | Path = "data/lake/basics/stock_basic.parquet",
) -> Path:
    """Fetch stock basic data and persist it to a parquet file."""
    pro = get_pro()
    logger.info("Fetching stock basic data")
    data = pro.stock_basic(
        exchange="",
        list_status="L",
        fields="ts_code,symbol,name,area,industry,market,list_date",
    )
    if data is None or data.empty:
        message = "No stock basic data returned"
        logger.error(message)
        raise RuntimeError(message)

    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output_path, index=False)
    logger.info("Saved %s rows to %s", len(data), output_path)
    return output_path


def filter_main_board_codes(df: pd.DataFrame) -> list[str]:
    """Filter A-share main board codes using a simple prefix rule.

    Rule: keep Shanghai main board (60/601/603/605) and Shenzhen main board
    (000/001/002/003) codes that end with .SH or .SZ.
    """
    if "ts_code" not in df.columns:
        raise ValueError("stock_basic data must include ts_code")

    main_prefixes = ("000", "001", "002", "003", "600", "601", "603", "605")
    codes = df["ts_code"].astype(str)
    is_a_share = codes.str.endswith(".SH") | codes.str.endswith(".SZ")
    has_prefix = codes.str.startswith(main_prefixes)
    filtered = codes[is_a_share & has_prefix]
    return filtered.tolist()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest stock basic data")
    parser.add_argument(
        "--out",
        default="data/lake/basics/stock_basic.parquet",
        help="Output parquet path",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    try:
        ingest_stock_basic(out_path=args.out)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.error("Stock basic ingestion failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
