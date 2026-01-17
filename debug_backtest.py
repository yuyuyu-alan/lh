#!/usr/bin/env python3
"""Debug backtest to find the issue."""

import pandas as pd
import numpy as np
from pathlib import Path

def debug_backtest():
    """Debug the backtest logic."""
    # Load data
    data = pd.read_parquet('data/lake/daily_bars/year=2020/compact.parquet')

    # Use only first 100 stocks for debugging
    allowed_codes = sorted(data['ts_code'].unique())[:100]
    data = data[data['ts_code'].isin(allowed_codes)]

    # Sort and filter
    data = data.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    data = data[(data['trade_date'] >= '20200102') & (data['trade_date'] <= '20200131')]

    # Compute factors
    from src.backtest import factors as factor_lib

    data['tr_mean20'] = factor_lib.tr_mean20(data)
    data['tr_ratio20'] = factor_lib.tr_ratio20(data)
    data['tr_stab20'] = factor_lib.tr_stab20(data)
    data['vr'] = factor_lib.vr(data)
    data['vr_dir'] = factor_lib.vr_dir(data)
    data['div_up20'] = factor_lib.div_up20(data)
    data['div_dn20'] = factor_lib.div_dn20(data)

    data['open_next'] = data.groupby('ts_code', sort=False)['open'].shift(-1)

    # Get trade dates
    trade_dates = sorted(data['trade_date'].unique())

    print("=== Backtest Debug ===")
    print(f"Total stocks: {data.ts_code.nunique()}")
    print(f"Date range: {trade_dates[0]} to {trade_dates[-1]}")
    print(f"Total trading days: {len(trade_dates)}")

    # Simulate first few days
    nav = 1.0
    prev_weights = {}

    for i, trade_date in enumerate(trade_dates[:10]):
        day_slice = data[data['trade_date'] == trade_date].copy()
        day_slice = day_slice.dropna(subset=['open', 'open_next'])

        if day_slice.empty:
            print(f"\nDay {trade_date}: EMPTY - skipping")
            continue

        print(f"\nDay {trade_date}: {len(day_slice)} stocks")

        # Check for extreme prices
        if (day_slice['open'] < 0.1).any() or (day_slice['open'] > 1000).any():
            print(f"  WARNING: Extreme open prices detected!")
            print(f"  Min open: {day_slice['open'].min()}")
            print(f"  Max open: {day_slice['open'].max()}")

        # Check for NaN in factors
        factor_cols = ['tr_mean20', 'tr_ratio20', 'tr_stab20', 'vr', 'vr_dir', 'div_up20', 'div_dn20']
        for col in factor_cols:
            nan_count = day_slice[col].isna().sum()
            if nan_count > 0:
                print(f"  WARNING: {col} has {nan_count} NaN values out of {len(day_slice)}")

        # Compute score
        from src.backtest.scoring import score_cross_section
        day_slice['score'] = score_cross_section(day_slice, 'configs/factors.yaml')

        # Sort and select
        ranked = day_slice.sort_values('score', ascending=False)
        selected_codes = ranked['ts_code'].tolist()[:30]  # Select top 30

        # Compute weights
        weights = {code: 1.0/len(selected_codes) for code in selected_codes}

        # Compute return
        slice_selected = day_slice[day_slice['ts_code'].isin(selected_codes)].copy()
        slice_selected['weight'] = slice_selected['ts_code'].map(weights)
        slice_selected = slice_selected.dropna(subset=['weight'])

        if len(slice_selected) == 0:
            print(f"  ERROR: No selected stocks!")
            continue

        returns = slice_selected['open_next'] / slice_selected['open'] - 1.0

        # Check for extreme returns
        max_return = returns.max()
        min_return = returns.min()
        if abs(max_return) > 0.2 or abs(min_return) > 0.2:
            print(f"  WARNING: Extreme returns! Max: {max_return:.4f}, Min: {min_return:.4f}")

        gross_return = float((returns * slice_selected['weight']).sum())
        nav *= 1.0 + gross_return

        print(f"  Selected: {len(selected_codes)} stocks")
        print(f"  Return: {gross_return:.6f} ({gross_return*100:.4f}%)")
        print(f"  NAV: {nav:.6f}")

        if nav > 10 or nav < 0.1:
            print(f"  *** NAV ABNORMAL: {nav} ***")

        prev_weights = weights

if __name__ == '__main__':
    debug_backtest()