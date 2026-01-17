"""Minimal example for appending and compacting parquet partitions."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.parquet_io import append_partition, compact_year, read_partitions


def main() -> None:
    df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "trade_date": ["20240102", "20240102"],
            "close": [10.0, 10.5],
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        append_partition(df, tmpdir)
        append_partition(df, tmpdir)
        compact_year(tmpdir, 2024)
        combined = read_partitions(tmpdir)
        print(combined)


if __name__ == "__main__":
    main()
