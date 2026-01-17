# LH Backtest Bootstrap

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your Tushare token in the environment:

```bash
export TUSHARE_TOKEN="your_token_here"
```

## Data layout

Parquet data is organized under `data/lake` using year partitions:

```
data/lake/
  year=2024/
    part-<uuid>.parquet
    compact.parquet
```

## Smoke workflow (placeholder)

```bash
# TODO: replace with the actual command once pipelines are implemented
python -c "from src.utils.config import load_yaml; print(load_yaml('configs/backtest.yaml').keys())"
```
