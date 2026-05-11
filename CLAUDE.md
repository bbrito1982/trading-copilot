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

- **Phase 2**: `signals/ml/` — scikit-learn classifier trained on `opportunities` table outcomes
- **Phase 3**: `sentiment/` — ticker-contextualized news scoring (same headline → different effect per asset via `tagger.py` lookup table, then learned embeddings)
- **Phase 4**: Ensemble combining all signal layers
