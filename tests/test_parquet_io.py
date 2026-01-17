import pandas as pd

from src.utils.parquet_io import append_partition, read_partitions


def test_append_and_read_dedup(tmp_path):
    df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "trade_date": ["20240102", "20240102"],
            "close": [10.0, 10.5],
        }
    )

    append_partition(df, tmp_path)
    append_partition(df, tmp_path)

    result = read_partitions(tmp_path)

    assert len(result) == 1
    assert result.loc[0, "ts_code"] == "000001.SZ"
    assert result.loc[0, "trade_date"] == "20240102"
