"""Factor computations for daily bar data."""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {"ts_code", "trade_date", "open", "high", "low", "close", "vol"}


def _validate_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df.groupby("ts_code", sort=False)["close"].shift(1)
    high_low = df["high"] - df["low"]
    high_prev = (df["high"] - prev_close).abs()
    low_prev = (df["low"] - prev_close).abs()
    return pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)


def tr_mean20(df: pd.DataFrame) -> pd.Series:
    _validate_columns(df)
    tr = _true_range(df)
    return (
        tr.groupby(df["ts_code"], sort=False)
        .rolling(20, min_periods=5)
        .mean()
        .reset_index(level=0, drop=True)
    )


def tr_ratio20(df: pd.DataFrame) -> pd.Series:
    _validate_columns(df)
    tr = _true_range(df)
    tr_mean = (
        tr.groupby(df["ts_code"], sort=False)
        .rolling(20, min_periods=5)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return tr / tr_mean


def tr_stab20(df: pd.DataFrame) -> pd.Series:
    _validate_columns(df)
    tr = _true_range(df)
    tr_mean = (
        tr.groupby(df["ts_code"], sort=False)
        .rolling(20, min_periods=5)
        .mean()
        .reset_index(level=0, drop=True)
    )
    tr_std = (
        tr.groupby(df["ts_code"], sort=False)
        .rolling(20, min_periods=5)
        .std()
        .reset_index(level=0, drop=True)
    )
    return tr_std / tr_mean


def vr(df: pd.DataFrame) -> pd.Series:
    _validate_columns(df)
    vol_mean = (
        df.groupby("ts_code", sort=False)["vol"]
        .rolling(20, min_periods=5)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return df["vol"] / vol_mean


def vr_dir(df: pd.DataFrame) -> pd.Series:
    _validate_columns(df)
    returns = df.groupby("ts_code", sort=False)["close"].pct_change()
    return vr(df) * (returns > 0).astype(int) - vr(df) * (returns < 0).astype(int)


def div_up20(df: pd.DataFrame) -> pd.Series:
    _validate_columns(df)
    returns = df.groupby("ts_code", sort=False)["close"].pct_change()
    up = (returns > 0).astype(int)
    return (
        up.groupby(df["ts_code"], sort=False)
        .rolling(20, min_periods=5)
        .mean()
        .reset_index(level=0, drop=True)
    )


def div_dn20(df: pd.DataFrame) -> pd.Series:
    _validate_columns(df)
    returns = df.groupby("ts_code", sort=False)["close"].pct_change()
    dn = (returns < 0).astype(int)
    return (
        dn.groupby(df["ts_code"], sort=False)
        .rolling(20, min_periods=5)
        .mean()
        .reset_index(level=0, drop=True)
    )
