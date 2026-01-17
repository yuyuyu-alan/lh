#!/usr/bin/env python3
"""Fetch remaining data with rate limiting to avoid API restrictions."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta

import pandas as pd

from src.ingest.tushare_client import get_pro

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def date_range(start_str, end_str, step_days=5):
    """Generate date ranges."""
    start = datetime.strptime(start_str, '%Y%m%d')
    end = datetime.strptime(end_str, '%Y%m%d')

    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=step_days), end)
        yield current.strftime('%Y%m%d'), chunk_end.strftime('%Y%m%d')
        current = chunk_end + timedelta(days=1)


def fetch_with_delay(pro: any, start: str, end: str, delay: float = 2.0):
    """Fetch data with delay to avoid rate limiting."""
    logger.info(f"Fetching {start} to {end} with {delay}s delay")

    chunk_data = []
    chunk_count = 0

    for chunk_start, chunk_end in date_range(start, end, step_days=5):
        chunk_count += 1
        logger.info(f"  Chunk {chunk_count}: {chunk_start} to {chunk_end}")

        try:
            # Try with pagination
            page_size = 5000
            offset = 0
            page_count = 0

            while True:
                page_count += 1

                df = pro.daily(
                    ts_code='',
                    start_date=chunk_start,
                    end_date=chunk_end,
                    fields='ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount',
                    limit=page_size,
                    offset=offset
                )

                if df.empty:
                    if page_count == 1:
                        logger.warning(f"    No data for {chunk_start}-{chunk_end}")
                    break

                chunk_data.append(df)
                logger.info(f"    Page {page_count}: {len(df)} records (offset: {offset})")

                offset += page_size

                if len(df) < page_size:
                    break

                # Add delay between pages
                time.sleep(delay)

        except Exception as e:
            logger.error(f"    Error: {e}")
            time.sleep(5)  # Longer delay on error
            continue

        # Add delay between chunks
        time.sleep(delay)

    if chunk_data:
        combined = pd.concat(chunk_data, ignore_index=True)
        return combined.sort_values(['trade_date', 'ts_code'])
    else:
        return pd.DataFrame()


def main():
    """Main entry point."""
    os.environ['TUSHARE_TOKEN'] = '3189939459350328192'

    pro = get_pro()

    # Define remaining date ranges
    remaining_ranges = [
        ('20231001', '20231231', 2.0),  # 2023年10-12月
        ('20240101', '20241231', 2.0),  # 2024年全年
        ('20250101', '20250531', 2.0),  # 2025年1-5月
    ]

    all_remaining_data = []

    for i, (start, end, delay) in enumerate(remaining_ranges, 1):
        logger.info("\n" + "=" * 60)
        logger.info(f"Range {i}/{len(remaining_ranges)}: {start} to {end}")
        logger.info("=" * 60)

        year = start[:4]
        logger.info(f"Fetching {year} data with {delay}s delay...")

        data = fetch_with_delay(pro, start, end, delay)

        if not data.empty:
            logger.info(f"✅ Successfully fetched {len(data):,} records for {year}")
            logger.info(f"   Date range: {data.trade_date.min()} - {data.trade_date.max()}")
            logger.info(f"   Unique stocks: {data.ts_code.nunique()}")
            all_remaining_data.append(data)
        else:
            logger.warning(f"❌ No data found for {year}")

        # Save intermediate results
        if all_remaining_data:
            intermediate = pd.concat(all_remaining_data, ignore_index=True)
            intermediate_file = f'data/remaining_{start[:4]}.csv'
            intermediate.to_csv(intermediate_file, index=False)
            logger.info(f"💾 Saved intermediate to {intermediate_file}")

    # Save all remaining data
    if all_remaining_data:
        all_remaining = pd.concat(all_remaining_data, ignore_index=True)
        all_remaining = all_remaining.drop_duplicates()
        all_remaining = all_remaining.sort_values(['trade_date', 'ts_code'])

        remaining_file = 'data/daily_data_remaining_all.csv'
        all_remaining.to_csv(remaining_file, index=False)
        logger.info(f"\n💾 Saved all remaining data to {remaining_file}")

        # Merge with existing complete dataset
        logger.info("\n" + "=" * 60)
        logger.info("Merging with existing complete dataset")
        logger.info("=" * 60)

        try:
            existing_complete = pd.read_csv('data/daily_data_complete_5years.csv')
            logger.info(f"Existing complete data: {len(existing_complete):,} records")

            final_combined = pd.concat([existing_complete, all_remaining], ignore_index=True)
            final_combined = final_combined.drop_duplicates()
            final_combined = final_combined.sort_values(['trade_date', 'ts_code'])

            final_file = 'data/daily_data_final_complete.csv'
            final_combined.to_csv(final_file, index=False)

            logger.info(f"\n🎉 Final complete dataset:")
            logger.info(f"   Total records: {len(final_combined):,}")
            logger.info(f"   Unique stocks: {final_combined.ts_code.nunique()}")
            logger.info(f"   Date range: {final_combined.trade_date.min()} - {final_combined.trade_date.max()}")
            logger.info(f"   Saved to: {final_file}")

            # Print yearly summary
            logger.info("\n" + "=" * 60)
            logger.info("Final Yearly Summary")
            logger.info("=" * 60)

            for year in range(2020, 2026):
                year_data = final_combined[(final_combined.trade_date >= year*10000) & (final_combined.trade_date < (year+1)*10000)]
                if not year_data.empty:
                    unique_dates = year_data.trade_date.nunique()
                    total_records = len(year_data)
                    unique_stocks = year_data.ts_code.nunique()
                    months = sorted((year_data.trade_date % 10000 // 100).unique())
                    status = "✅" if len(months) >= 10 else "⚠️"
                    print(f"{status} {year}年: {unique_dates}个交易日, {total_records:,}条记录, {unique_stocks}只股票, 月份: {months}")
                else:
                    print(f"❌ {year}年: 无数据")

        except Exception as e:
            logger.error(f"Error merging: {e}")

    else:
        logger.error("No remaining data was fetched!")


if __name__ == '__main__':
    main()
