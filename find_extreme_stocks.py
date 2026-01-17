#!/usr/bin/env python3
"""Find stocks that generate extreme returns in backtest."""

import pandas as pd
import numpy as np

def find_extreme_stocks():
    """Find stocks with extreme returns."""
    # Load all data
    all_data = []
    for year in [2020, 2021, 2022]:
        df = pd.read_parquet(f'data/lake/daily_bars/year={year}/compact.parquet')
        all_data.append(df)
    data = pd.concat(all_data, ignore_index=True)

    print(f"Total data: {len(data):,} rows")

    # Compute returns
    data = data.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    data['open_next'] = data.groupby('ts_code', sort=False)['open'].shift(-1)
    data['return'] = data['open_next'] / data['open'] - 1.0

    # Clean data (as in backtest)
    data = data[(data['return'].abs() <= 0.20)]
    data = data[(data['open'] > 0) & (data['open_next'] > 0)]
    data = data[(data['open'] < 1000) & (data['open_next'] < 1000)]
    data = data.dropna(subset=['return'])

    print(f"After cleaning: {len(data):,} rows")

    # Compute factor statistics per stock
    stock_stats = []

    for ts_code in data['ts_code'].unique():
        stock_data = data[data['ts_code'] == ts_code].copy()
        if len(stock_data) < 100:  # Skip stocks with too little data
            continue

        mean_return = stock_data['return'].mean()
        std_return = stock_data['return'].std()
        max_return = stock_data['return'].max()
        min_return = stock_data['return'].min()

        stock_stats.append({
            'ts_code': ts_code,
            'count': len(stock_data),
            'mean_return': mean_return,
            'std_return': std_return,
            'max_return': max_return,
            'min_return': min_return,
            'total_return': (1 + stock_data['return']).prod() - 1,  # Compound return
        })

    stock_df = pd.DataFrame(stock_stats)

    print("\n=== Top 20 stocks by mean return ===")
    top_mean = stock_df.nlargest(20, 'mean_return')
    print(top_mean[['ts_code', 'count', 'mean_return', 'std_return', 'total_return']].to_string())

    print("\n=== Top 20 stocks by total compound return ===")
    top_total = stock_df.nlargest(20, 'total_return')
    print(top_total[['ts_code', 'count', 'mean_return', 'total_return']].to_string())

    print("\n=== Extreme returns distribution ===")
    extreme_returns = data[(data['return'] > 0.10) | (data['return'] < -0.10)]
    print(f"Number of extreme returns (>10% or <-10%): {len(extreme_returns):,}")

    if len(extreme_returns) > 0:
        print("\nTop 20 extreme returns:")
        extreme_sorted = extreme_returns.nlargest(20, 'return')
        print(extreme_sorted[['ts_code', 'trade_date', 'open', 'open_next', 'return']].to_string())

if __name__ == '__main__':
    find_extreme_stocks()