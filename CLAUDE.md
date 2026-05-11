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

The system is a swing trading assistant that scans stocks/ETFs for signals, monitors breaking financial news in real time, sends push notifications via ntfy with Enter/Skip action buttons, and tracks entered positions through to exit.

### Data flow

```
APScheduler (scheduler.py)
  ├── Daily scan (6:30 AM ET)
  │     ├── tiingo.py → DuckDB cache (OHLCV)
  │     ├── news.py → NewsAPI headlines (last 3 days)
  │     ├── sentiment/tagger.py → ticker + macro sentiment, blended
  │     ├── signals/rules.py → RSI, MACD, MA crossover, volume spike
  │     ├── signals/scorer.py → conviction (ensemble > ML > rule-based)
  │     │     ├── signals/ml/predictor.py → P(profitable) from GradientBoosting
  │     │     └── signals/ensemble.py → meta-model blend of all three layers
  │     ├── notifications/charts.py → mplfinance PNG (daily candles)
  │     └── notifications/ntfy.py → push alert with Enter/Skip buttons
  ├── Breaking news monitor (every 5 min, market hours only)
  │     ├── news/rss.py → RSSPoller polls 6 RSS feeds, deduplicates by URL hash
  │     ├── news/breaking.py → scores each item with neural embedder
  │     │     ├── if |score| ≥ 0.35 → fetch intraday 5-min bars (data/intraday.py)
  │     │     ├── run signal scorer on intraday bars
  │     │     └── if conviction ≥ 0.4 → generate intraday chart + ntfy alert
  │     └── notifications/charts.py → generate_intraday_chart() (5-min candles, EMA9/20)
  └── Position monitor (4 PM ET after close)
        ├── tracker/positions.py → check stop/target/reversal
        └── ntfy exit alert with Confirm button

FastAPI webhook (api/webhook.py) — receives ntfy button callbacks via nginx → port 8000
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

- `config.yaml` — watchlist, signal parameters, scheduler cron times, swing trade percentages, `breaking_conviction_threshold` (default 0.4)
- `.env` — API keys (Tiingo, NewsAPI), ntfy topic, `WEBHOOK_BASE_URL=http://143.47.48.68` (nginx proxies port 80 → 8000), DB paths
- `data/ml_signal_model.joblib` — trained ML model (gitignored)
- `data/ensemble_model.joblib` — trained ensemble model (gitignored)
- `data/fnspid_sentiment.parquet` — FNSPID-derived sentiment cache (gitignored)
- `data/fnspid_cache/` — raw FNSPID downloads: `news_small.csv`, `full_history.zip`, `prices.parquet` (gitignored)

### Networking

- nginx listens on port 80, proxies to uvicorn on port 8000
- nginx config: `/etc/nginx/sites-available/trading-copilot` (server_name `143.47.48.68`)
- ntfy button callbacks POST to `http://143.47.48.68/enter` and `/skip` — confirmed working
- ntfy notifications: text-only (no inline images on iOS ntfy app); action buttons use structured JSON actions list

### Completed phases

**Phase 1 — Rule-based signals** ✓  
**Phase 2 — Backtest engine** ✓ (`backtest/engine.py`, `backtest/metrics.py`, `scripts/run_backtest.py`)  
**Phase 2b — ML signal layer** ✓ (`signals/ml/features.py`, `trainer.py`, `predictor.py`, `scripts/train_ml.py`)  
**Phase 3 — Sentiment pipeline** ✓ (`sentiment/tagger.py`, 13 macro themes, NewsAPI integration, veto mechanism)  
**Phase 4 — Ensemble** ✓ (`signals/ensemble.py`, `scripts/train_ensemble.py`)  
**Phase 4b — FNSPID historical sentiment** ✓ (`scripts/prepare_fnspid.py`, ensemble patched to accept `--sentiment-cache`)  
**Phase 3b — Neural sentiment embedder** ✓ (`sentiment/embedder.py`, `scripts/train_embedder.py`, `score_headlines_neural()` with keyword fallback; `prepare_fnspid.py` now also outputs `fnspid_headlines.parquet`)  
**Phase 5 — Breaking news monitor** ✓ (`news/rss.py`, `news/breaking.py`, `data/intraday.py`; 5-min APScheduler job, intraday chart generation, ntfy alerts with working Enter/Skip buttons)

### FNSPID integration — conclusions

- **Dataset**: `Zihan1004/FNSPID` on HuggingFace. Used `All_external.csv` (5.7 GB). License: CC BY-NC-4.0 (personal use only).
- **Pipeline**: `scripts/prepare_fnspid.py --watchlist --small` streams the CSV in 50k-row chunks (RAM-safe), extracts per-ticker price history from the zip, scores headlines with the existing keyword tagger, and saves `data/fnspid_sentiment.parquet` + `data/fnspid_headlines.parquet` (raw headlines for neural training).
- **Retrain**: `scripts/train_ensemble.py --watchlist --sentiment-cache data/fnspid_sentiment.parquet`
- **Results**: Ensemble CV ROC-AUC 0.789. Sentiment weight went from 0 → 0.13 (non-zero for first time). Backtest unchanged: Sharpe 0.44, win rate 50%, expectancy 0.63%, max drawdown 3%.
- **Why no backtest improvement**: The keyword tagger achieves only 3.9% directional agreement with 10-day returns on FNSPID headlines — too noisy to move the needle with a 0.13 weight over 22 trades.

### Phase 3b — Neural embedder workflow

```bash
# 1. Re-run prepare_fnspid (now also outputs fnspid_headlines.parquet)
uv run python scripts/prepare_fnspid.py --watchlist --small --skip-download

# 2. Train neural embedder (requires sentence-transformers + torch)
uv pip install -e ".[sentiment]"
uv run python scripts/train_embedder.py --headlines data/fnspid_headlines.parquet

# 3. Retrain ensemble (sentiment scores unchanged, but live scoring now uses neural)
uv run python scripts/train_ensemble.py --watchlist --from 2018-01-01 \
  --sentiment-cache data/fnspid_sentiment.parquet
```

The scheduler auto-uses the neural embedder when `data/sentiment_embedder.pt` exists; falls back to keyword tagger otherwise.

### Remaining phases

**Ensemble retraining (ongoing)**  
Retrain periodically as live `Opportunity` records accumulate real sentiment scores and outcomes. Use `--sentiment-cache` to include FNSPID historical data alongside live data.

**Possible next steps**
- Web dashboard to view open positions, recent signals, and chart history
- Persist intraday charts to disk and serve from FastAPI for richer notifications
- Extend RSS feed list or add Alpaca news stream for lower-latency breaking news
- Add position sizing logic (Kelly criterion or fixed fractional) to the opportunity output
