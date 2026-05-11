# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv venv && uv pip install -e ".[dev]"

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
```

## Architecture

The system is a swing trading assistant that scans stocks/ETFs for signals, sends rich notifications (chart image + action buttons) via ntfy, and tracks entered positions through to exit.

### Data flow

```
APScheduler (scheduler.py)
  ├── Daily scan (6:30 AM ET)
  │     ├── tiingo.py → DuckDB cache (OHLCV)
  │     ├── signals/rules.py → compute RSI, MACD, MA crossover, volume spike
  │     ├── signals/scorer.py → conviction score per ticker
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

### Signal pipeline (Phase 1 — rule-based)

`signals/rules.py` computes indicators via the `ta` library and returns `Signal` objects (type, direction, strength 0–1). `signals/scorer.py` weights them by type, requires ≥2 agreeing signals, and produces an `Opportunity` with entry/stop/target prices.

Conviction threshold for watchlist alerts: `config.yaml → conviction_threshold` (default 0.6).  
Discovery threshold for universe screener suggestions: `discovery_threshold` (default 0.7).

### Configuration

- `config.yaml` — watchlist, signal parameters, scheduler cron times, swing trade percentages
- `.env` — API keys (Tiingo, NewsAPI), ntfy topic, webhook base URL, DB paths

### Planned phases

**Phase 2 — Backtest engine**
Build `backtest/engine.py`: replay DuckDB OHLCV day-by-day, apply current signal rules as if live, simulate entries/exits, output metrics (Sharpe, CAGR, max drawdown, win rate, expectancy). Entry point: `scripts/run_backtest.py --strategy rsi_macd --from 2020-01-01`. Required before ML — validates that signals have a historical edge and allows threshold tuning with evidence.

**Phase 2b — ML signal layer**
Train a scikit-learn classifier in `signals/ml/` that predicts 5–10 day return direction using rule-based indicators as features. Labels come from the `opportunities` table (live outcomes) + historical backfill. Replaces hard-coded conviction weights with learned ones.

**Phase 3 — News/sentiment pipeline**
- Backfill headlines from GDELT aligned with historical price moves → auto-labeled `(headline, ticker, outcome)` training set
- `sentiment/tagger.py`: lookup table mapping macro entities (Hormuz, Fed rate hike, oil supply, etc.) to per-ticker directional effects — same headline can be bullish for one asset and bearish for another
- Integrate sentiment score into daily scan alongside rule signals

**Phase 3b — Learned ticker-contextualized embeddings**
Upgrade `sentiment/tagger.py` lookup table to a sentence-transformer model trained on `(headline_embedding ⊕ ticker_embedding) → price_direction`. Training data from GDELT backfill. Captures non-obvious correlations the lookup table misses.

**Phase 4 — Ensemble**
Meta-model in `signals/ensemble.py` taking rule conviction + ML score + sentiment score as inputs, outputting a single weighted conviction score. Trained on the full `opportunities` outcome history.
