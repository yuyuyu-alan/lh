#!/usr/bin/env python3
"""Convert CSV data to Parquet format with year partitions."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from src.utils.parquet_io import append_partition, compact_year

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_csv_to_parquet(csv_path: str, output_dir: str = "data/lake/daily_bars"):
    """Convert CSV to Parquet with year partitions."""
    logger.info(f"Reading CSV file: {csv_path}")

    # Read CSV in chunks to handle large files efficiently
    chunk_size = 100000
    total_rows = 0
    chunks_written = []

    for chunk_df in pd.read_csv(csv_path, chunksize=chunk_size):
        logger.info(f"Processing chunk with {len(chunk_df)} rows...")

        # Ensure proper data types
        if 'trade_date' in chunk_df.columns:
            chunk_df['trade_date'] = chunk_df['trade_date'].astype(str)

        # Write chunk to parquet partitions
        written_files = append_partition(chunk_df, output_dir)
        chunks_written.extend(written_files)
        total_rows += len(chunk_df)

    logger.info(f"✅ Successfully converted {total_rows:,} rows to Parquet format")
    logger.info(f"📁 Output directory: {output_dir}")

    # Compact each year partition
    base_path = Path(output_dir)
    if base_path.exists():
        year_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('year=')]
        logger.info(f"Found {len(year_dirs)} year partitions")

        for year_dir in sorted(year_dirs):
            year = year_dir.name.split('=')[1]
            try:
                compact_file = compact_year(output_dir, year)
                logger.info(f"✅ Compacted year {year}: {compact_file}")
            except Exception as e:
                logger.error(f"❌ Error compacting year {year}: {e}")

    return chunks_written


def main():
    """Main entry point."""
    csv_file = "data/daily_data_SUPPLEMENTED.csv"

    if not Path(csv_file).exists():
        logger.error(f"CSV file not found: {csv_file}")
        sys.exit(1)

    convert_csv_to_parquet(csv_file)

    # Verify the conversion
    logger.info("\n" + "=" * 60)
    logger.info("Verification - Listing Parquet files:")
    logger.info("=" * 60)

    data_dir = Path("data/lake/daily_bars")
    if data_dir.exists():
        for year_dir in sorted(data_dir.iterdir()):
            if year_dir.is_dir():
                parquet_files = list(year_dir.glob("*.parquet"))
                total_size = sum(f.stat().st_size for f in parquet_files) / (1024 * 1024)  # MB
                logger.info(f"  {year_dir.name}: {len(parquet_files)} files, {total_size:.1f} MB")


if __name__ == '__main__':
    main()