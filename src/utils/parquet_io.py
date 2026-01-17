"""Parquet data lake helpers."""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize trade_date and de-duplicate by ts_code and trade_date."""
    if "trade_date" not in df.columns:
        raise ValueError("trade_date column is required")
    if "ts_code" not in df.columns:
        raise ValueError("ts_code column is required")

    normalized = df.copy()
    normalized["trade_date"] = (
        pd.to_datetime(normalized["trade_date"], errors="coerce").dt.strftime("%Y%m%d")
    )
    normalized = normalized.dropna(subset=["trade_date"])
    normalized["trade_date"] = normalized["trade_date"].astype(str)
    normalized = normalized.sort_values(["ts_code", "trade_date"]).drop_duplicates(
        ["ts_code", "trade_date"], keep="last"
    )
    return normalized.reset_index(drop=True)


def _partition_years(df: pd.DataFrame, year_col: str) -> pd.Series:
    if year_col not in df.columns:
        raise ValueError(f"{year_col} column is required")
    years = df[year_col].astype(str).str.slice(0, 4)
    if years.isna().any():
        raise ValueError("year_col contains null values")
    return years


def append_partition(df: pd.DataFrame, base_dir: str | Path, year_col: str = "trade_date") -> list[Path]:
    """Append rows into year-based partitions under base_dir/year=YYYY."""
    normalized = normalize_keys(df)
    years = _partition_years(normalized, year_col)
    base_path = Path(base_dir)
    written: list[Path] = []

    for year, group in normalized.groupby(years):
        partition_dir = base_path / f"year={year}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        file_path = partition_dir / f"part-{uuid.uuid4().hex}.parquet"
        logger.info("Appending %s rows to %s", len(group), file_path)
        group.to_parquet(file_path, index=False)
        written.append(file_path)

    return written


def read_partitions(base_dir: str | Path) -> pd.DataFrame:
    """Read all year partitions under base_dir/year=*."""
    base_path = Path(base_dir)
    files = sorted(base_path.glob("year=*/*.parquet"))
    if not files:
        logger.info("No partitions found under %s", base_dir)
        return pd.DataFrame()

    frames = [pd.read_parquet(path) for path in files]
    combined = pd.concat(frames, ignore_index=True)
    return normalize_keys(combined)


def compact_year(
    base_dir: str | Path,
    year: str | int,
    key_cols: Sequence[str] | None = None,
) -> Path:
    """Compact a single year partition into one parquet file."""
    key_cols = list(key_cols) if key_cols is not None else ["ts_code", "trade_date"]
    year_str = str(year)
    partition_dir = Path(base_dir) / f"year={year_str}"
    if not partition_dir.exists():
        raise FileNotFoundError(f"Partition not found: {partition_dir}")

    files = sorted(partition_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {partition_dir}")

    frames = [pd.read_parquet(path) for path in files]
    combined = pd.concat(frames, ignore_index=True)
    combined = normalize_keys(combined)
    if key_cols:
        combined = combined.sort_values(list(key_cols)).drop_duplicates(list(key_cols), keep="last")

    tmp_path = partition_dir / f".tmp-compact-{uuid.uuid4().hex}.parquet"
    final_path = partition_dir / "compact.parquet"
    logger.info("Compacting %s files into %s", len(files), final_path)
    combined.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, final_path)

    for path in files:
        if path != final_path:
            path.unlink(missing_ok=True)

    return final_path
