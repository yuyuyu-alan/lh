set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

ingest_recent:
    @echo "TODO: run ingestion for recent data"

backtest_smoke:
    python -m src.backtest.run_daily --start 20230101 --end 20230630 --limit-codes 100

all_smoke: ingest_recent backtest_smoke
