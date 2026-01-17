"""Smoke test for TuShare connectivity."""

from __future__ import annotations

import logging
import sys

from src.ingest.tushare_client import get_pro


logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        pro = get_pro()
        logger.info("Requesting trade calendar sample")
        data = pro.trade_cal(exchange="SSE", start_date="20240101", end_date="20240105")
        row_count = len(data)
        print(f"trade_cal rows: {row_count}")
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.error("Smoke test failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
