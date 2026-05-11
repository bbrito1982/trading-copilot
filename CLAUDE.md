# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (include ML extras for phases 2b–4)
uv venv && uv pip install -e ".[dev,ml]"

# Run tests
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_signals.py -v

# Service management (runs as systemd daemon in production)
sudo systemctl start trading-copilot
sudo systemctl stop trading-copilot
sudo systemctl restart trading-copilot   # required after code changes
sudo systemctl status trading-copilot
journalctl -u trading-copilot -f         # live logs

# Start manually (development only)
uv run python -m trading_copilot.scheduler

# Webhook only (development)
uv run uvicorn trading_copilot.api.webhook:app --reload --port 8000

# Backfill historical price data
uv run python scripts/backfill_prices.py --watchlist
uv run python scripts/backfill_prices.py --tickers AAPL MSFT --from 2015-01-01
uv run python scripts/backfill_prices.py --universe   # full S&P500 + ETFs

# Backtest (Phase 2)
uv run python scripts/run_backtest.py --watchlist --from 2020-01-01
uv run python scripts/run_backtest.py -t AAPL -t NVDA --from 2022-01-01 --threshold 0.05 --trades

# Train ML signal model (Phase 2b) — cap --to for out-of-sample validation
uv run python scripts/train_ml.py --watchlist --from 2018-01-01
uv run python scripts/train_ml.py --watchlist --from 2018-01-01 --to 2024-12-31

# Train ensemble meta-model (Phase 4) — run after train_ml.py
uv run python scripts/train_ensemble.py --watchlist --from 2018-01-01
```

## Architecture

The system is a swing trading assistant that scans stocks/ETFs for signals, sends rich notifications (chart image + action buttons) via ntfy, and tracks entered positions through to exit.

### Data flow

```
APScheduler (scheduler.py)
  ├── Daily scan (6:30 AM ET)
  │     ├── tiingo.py → DuckDB cache (OHLCV)
  │     ├── news.py → NewsAPI headlines (last 3 days)
  │     ├── sentiment/tagger.py → macro theme score per ticker
  │     ├── signals/rules.py → RSI, MACD, MA crossover, volume spike
  │     ├── signals/scorer.py → conviction (ensemble > ML > rule-based)
  │     │     ├── signals/ml/predictor.py → P(profitable) from GradientBoosting
  │     │     └── signals/ensemble.py → meta-model blend of all three layers
  │     ├── notifications/charts.py → mplfinance PNG
  │     └── notifications/ntfy.py → push alert with Enter/Skip buttons
  └── Position monitor (4 PM ET after close)
        ├── tracker/positions.py → check stop/target/reversal
        └── ntfy exit alert with Confirm button

FastAPI webhook (api/webhook.py) — receives ntfy button callbacks
  ├── POST /enter?opportunity_id=X  → creates Position in SQLite
  ├── POST /skip?opportunity_id=X   → marks Opportunity skipped
  └── POST /exit?position_id=X&price=Y → closes Position, records P&L
```

### Storage

- **DuckDB** (`data/market.duckdb`) — append-only OHLCV and news headline cache. Never modify existing rows.
- **SQLite** (`data/positions.db`) — `Opportunity` and `Position` tables via SQLModel. Every signal is saved as an Opportunity; only user-confirmed ones become Positions. This table is the ML training dataset.

### Signal pipeline

Conviction is computed in three layers; the highest available layer wins:

1. **Rule-based** (`signals/rules.py` + `signals/scorer.py`) — RSI, MACD, MA crossover, volume spike. Requires ≥2 agreeing signals. Always available.
2. **ML model** (`signals/ml/`) — GradientBoosting classifier trained on 21 indicator features predicting 10-day return direction. Trained via `scripts/train_ml.py`. CV ROC-AUC ~0.57–0.60.
3. **Ensemble** (`signals/ensemble.py`) — Logistic regression meta-model blending rule + ML + sentiment. Trained via `scripts/train_ensemble.py`. CV ROC-AUC ~0.84. Improves as live outcome data accumulates.

**Sentiment veto**: if macro sentiment strongly contradicts the technical direction (threshold 0.4), the opportunity is suppressed before reaching the user.

Conviction threshold for watchlist alerts: `config.yaml → conviction_threshold` (default 0.05).  
Discovery threshold for universe screener: `discovery_threshold` (default 0.10).

Backtest baseline (2025 out-of-sample, ML trained on 2018–2024): Sharpe 0.44, win rate 50%, expectancy 0.63%/trade, max drawdown 3%.

### Configuration

- `config.yaml` — watchlist, signal parameters, scheduler cron times, swing trade percentages
- `.env` — API keys (Tiingo, NewsAPI), ntfy topic, webhook base URL, DB paths
- `data/ml_signal_model.joblib` — trained ML model (gitignored)
- `data/ensemble_model.joblib` — trained ensemble model (gitignored)

### Completed phases

**Phase 1 — Rule-based signals** ✓  
**Phase 2 — Backtest engine** ✓ (`backtest/engine.py`, `backtest/metrics.py`, `scripts/run_backtest.py`)  
**Phase 2b — ML signal layer** ✓ (`signals/ml/features.py`, `trainer.py`, `predictor.py`, `scripts/train_ml.py`)  
**Phase 3 — Sentiment pipeline** ✓ (`sentiment/tagger.py`, 13 macro themes, NewsAPI integration, veto mechanism)  
**Phase 4 — Ensemble** ✓ (`signals/ensemble.py`, `scripts/train_ensemble.py`)

### Remaining phases

**Phase 3b — Learned ticker-contextualized embeddings**
Upgrade `sentiment/tagger.py` keyword matching to a sentence-transformer model trained on `(headline_embedding ⊕ ticker_embedding) → price_direction`. Requires GDELT historical headline backfill (paid NewsAPI or GDELT BigQuery). Fixes noisy keyword matches on the free NewsAPI tier.

**Ensemble retraining (ongoing)**
As the live system accumulates `Opportunity` records with real sentiment scores and outcomes, periodically retrain the ensemble on that data. Sentiment weight is currently 0 (no historical training data); it will become non-zero as live data accumulates.
