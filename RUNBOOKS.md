# Trading Copilot — Runbooks & Playbooks

Operational reference for all routine and emergency procedures. All commands run as `ubuntu` on the server unless noted.

---

## Table of Contents

1. [Service Management](#1-service-management)
2. [Watchlist Management](#2-watchlist-management)
3. [Historical Data Backfill](#3-historical-data-backfill)
4. [Model Training](#4-model-training)
5. [Monitoring & Logs](#5-monitoring--logs)
6. [Webhook & API](#6-webhook--api)
7. [Position Management](#7-position-management)
8. [Backtesting](#8-backtesting)
9. [Incident Playbooks](#9-incident-playbooks)
10. [Scheduled Jobs Reference](#10-scheduled-jobs-reference)
11. [Infrastructure Reference](#11-infrastructure-reference)

---

## 1. Service Management

### Start the service
```bash
sudo systemctl start trading-copilot
```

### Stop the service
```bash
sudo systemctl stop trading-copilot
```

### Restart after code or config changes
```bash
sudo systemctl restart trading-copilot
```
> **Always restart after editing `config.yaml`, any `.py` file, or `.env`.**

### Check status
```bash
sudo systemctl status trading-copilot
```

### Enable auto-start on boot
```bash
sudo systemctl enable trading-copilot
```

### Run manually (development only — not as daemon)
```bash
cd /home/ubuntu/projects/trading-copilot
uv run python -m trading_copilot.scheduler
```

### Run webhook only (development)
```bash
uv run uvicorn trading_copilot.api.webhook:app --reload --port 8000
```

---

## 2. Watchlist Management

### View current watchlist
```bash
grep -A 50 "^watchlist:" config.yaml | grep "^  -"
```

### Add a US-listed ticker

**Step 1** — Add to `config.yaml`:
```yaml
watchlist:
  - AAPL
  - TSLA   # new entry
```

**Step 2** — Backfill historical data:
```bash
uv run python scripts/backfill_prices.py --tickers "TSLA" --from 2020-01-01
```

**Step 3** — Restart:
```bash
sudo systemctl restart trading-copilot
```

### Add a non-US / European ticker

Non-US tickers are not on Tiingo; Yahoo Finance is used as fallback. You must provide the Yahoo Finance symbol mapping.

| Exchange | Yahoo suffix | Example |
|----------|-------------|---------|
| London (LSE) | `.L` | `HEAL.L` |
| XETRA (Frankfurt) | `.DE` | `DFEN.DE` |
| Euronext Paris | `.PA` | `AIR.PA` |
| Euronext Brussels | `.BR` | `EURN.BR` |
| Euronext Amsterdam | `.AS` | `ASML.AS` |
| Borsa Italiana | `.MI` | `ENI.MI` |

**Step 1** — Add to `config.yaml` watchlist using the local ticker name:
```yaml
watchlist:
  - DFEN.DE
```

**Step 2** — Add Yahoo Finance mapping to `trading_copilot/data/tiingo.py` → `YAHOO_FALLBACK`:
```python
YAHOO_FALLBACK: dict[str, str] = {
    ...
    "DFEN.DE": "DFEN.DE",   # same if Yahoo uses the same symbol
    "HEAL.UK": "HEAL.L",    # different suffix on Yahoo
}
```

**Step 3** — Backfill:
```bash
uv run python scripts/backfill_prices.py --tickers "DFEN.DE" --from 2020-01-01
```

**Step 4** — Restart:
```bash
sudo systemctl restart trading-copilot
```

### Remove a ticker

**Step 1** — Delete the line from `config.yaml` watchlist.

**Step 2** — Optionally remove its `YAHOO_FALLBACK` entry from `tiingo.py` if it's non-US.

**Step 3** — Restart:
```bash
sudo systemctl restart trading-copilot
```
> Historical data in DuckDB is kept (append-only). No cleanup needed.

### Verify a Yahoo Finance symbol before adding
```bash
uv run python -c "
import yfinance as yf
df = yf.download('DFEN.DE', period='5d', auto_adjust=True, progress=False)
print(df.tail())
"
```
If the output is empty or errors, the symbol is wrong or the ticker is delisted.

---

## 3. Historical Data Backfill

### Backfill the full watchlist
```bash
uv run python scripts/backfill_prices.py --watchlist --from 2020-01-01
```

### Backfill specific tickers
```bash
uv run python scripts/backfill_prices.py --tickers "AAPL" --tickers "NVDA" --from 2020-01-01
```

### Backfill from a specific date range
```bash
uv run python scripts/backfill_prices.py --watchlist --from 2018-01-01 --to 2024-12-31
```

### Backfill full S&P 500 + ETF universe (slow — ~500 tickers)
```bash
uv run python scripts/backfill_prices.py --universe
```

### Check what data is cached
```bash
uv run python -c "
import duckdb
con = duckdb.connect('data/market.duckdb')
print(con.execute(\"SELECT ticker, COUNT(*) as rows, MIN(date) as from_, MAX(date) as to_ FROM ohlcv GROUP BY ticker ORDER BY ticker\").df().to_string())
"
```

---

## 4. Model Training

Models must be retrained after significant watchlist changes or when enough new live signal data has accumulated.

### Train the ML model (GradientBoosting)
```bash
uv run python scripts/train_ml.py --watchlist --from 2018-01-01
```
Output: `data/ml_signal_model.joblib`
Expected CV ROC-AUC: ~0.60

### Train the ML model capped to out-of-sample validation date
```bash
uv run python scripts/train_ml.py --watchlist --from 2018-01-01 --to 2024-12-31
```

### Train the ensemble meta-model
Run **after** `train_ml.py` completes.
```bash
uv run python scripts/train_ensemble.py --watchlist --from 2018-01-01
```
Output: `data/ensemble_model.joblib`
Expected CV ROC-AUC: ~0.79

### Train ensemble with FNSPID historical sentiment cache
```bash
uv run python scripts/train_ensemble.py --watchlist --from 2018-01-01 \
  --sentiment-cache data/fnspid_sentiment.parquet
```

### Train neural sentiment embedder
Requires sentence-transformers and the FNSPID headlines parquet.
```bash
uv pip install -e ".[sentiment]"
uv run python scripts/train_embedder.py --headlines data/fnspid_headlines.parquet
```
Output: `data/sentiment_embedder.pt`

### Full retrain sequence
```bash
uv run python scripts/train_ml.py --watchlist --from 2018-01-01
uv run python scripts/train_ensemble.py --watchlist --from 2018-01-01
sudo systemctl restart trading-copilot
```

---

## 5. Monitoring & Logs

### Tail live logs
```bash
journalctl -u trading-copilot -f
```

### View last 100 lines
```bash
journalctl -u trading-copilot -n 100 --no-pager
```

### View logs since a specific time
```bash
journalctl -u trading-copilot --since "2026-05-17 11:00:00" --no-pager
```

### View only errors
```bash
journalctl -u trading-copilot -p err -f
```

### Check breaking news monitor is running
Look for lines like:
```
INFO apscheduler.executors.default: Job "BreakingNewsMonitor.tick ..." executed successfully
```
These should appear every 5 minutes during market hours (13:30–20:00 UTC, Mon–Fri).

### Check scan jobs ran
EU scan fires at **11:30 UTC** Mon–Fri. Look for:
```
INFO trading_copilot.scheduler: Starting EU scan
```
US scan fires at **21:30 UTC** Mon–Fri. Look for:
```
INFO trading_copilot.scheduler: Starting US scan
```

### Check nginx status (webhook proxy)
```bash
sudo systemctl status nginx --no-pager
```

### Test webhook endpoint is reachable
```bash
curl -s -o /dev/null -w "%{http_code}" http://143.47.48.68/
```
Expected: `404` or `422` (FastAPI alive, no matching route).

---

## 6. Webhook & API

### Endpoint reference

| Method | Path | Triggered by | Action |
|--------|------|-------------|--------|
| `POST` | `/enter?opportunity_id=X` | ntfy Enter button | Creates Position, sends confirmation ntfy |
| `POST` | `/skip?opportunity_id=X` | ntfy Skip button | Marks Opportunity skipped, sends ntfy |
| `POST` | `/exit?position_id=X&price=Y` | ntfy Confirm button | Closes Position, records P&L, sends ntfy |

### Manually trigger Enter (for testing)
```bash
curl -X POST "http://143.47.48.68/enter?opportunity_id=1"
```

### Manually trigger Skip
```bash
curl -X POST "http://143.47.48.68/skip?opportunity_id=1"
```

### Manually trigger Exit
```bash
curl -X POST "http://143.47.48.68/exit?position_id=1&price=185.50"
```

### Reload nginx config without downtime
```bash
sudo nginx -t && sudo systemctl reload nginx
```

### nginx config location
```
/etc/nginx/sites-available/trading-copilot
```

---

## 7. Position Management

### View open positions
```bash
uv run python -c "
from trading_copilot.config import settings
import sqlite3, pandas as pd
con = sqlite3.connect(settings.positions_db)
df = pd.read_sql('SELECT * FROM position WHERE status=\"open\"', con)
print(df.to_string())
"
```

### View all opportunities (signals sent)
```bash
uv run python -c "
from trading_copilot.config import settings
import sqlite3, pandas as pd
con = sqlite3.connect(settings.positions_db)
df = pd.read_sql('SELECT * FROM opportunity ORDER BY created_at DESC LIMIT 20', con)
print(df.to_string())
"
```

### View closed positions with P&L
```bash
uv run python -c "
from trading_copilot.config import settings
import sqlite3, pandas as pd
con = sqlite3.connect(settings.positions_db)
df = pd.read_sql('SELECT * FROM position WHERE status!=\"open\" ORDER BY closed_at DESC', con)
print(df.to_string())
"
```

---

## 8. Backtesting

### Backtest the watchlist (default signals)
```bash
uv run python scripts/run_backtest.py --watchlist --from 2020-01-01
```

### Backtest specific tickers
```bash
uv run python scripts/run_backtest.py -t AAPL -t NVDA --from 2022-01-01
```

### Backtest with custom conviction threshold and show trades
```bash
uv run python scripts/run_backtest.py --watchlist --from 2020-01-01 --threshold 0.05 --trades
```

### Baseline metrics (2025 out-of-sample, ML trained 2018–2024)
| Metric | Value |
|--------|-------|
| Sharpe ratio | 0.44 |
| Win rate | 50% |
| Expectancy | 0.63%/trade |
| Max drawdown | 3% |

---

## 9. Incident Playbooks

### Service is down

1. Check status:
   ```bash
   sudo systemctl status trading-copilot
   ```
2. Check for crash reason:
   ```bash
   journalctl -u trading-copilot -n 50 --no-pager
   ```
3. Restart:
   ```bash
   sudo systemctl restart trading-copilot
   ```
4. If it keeps crashing, run manually to see the error directly:
   ```bash
   cd /home/ubuntu/projects/trading-copilot
   uv run python -m trading_copilot.scheduler
   ```

---

### No scan notifications received

1. Check if service is running:
   ```bash
   sudo systemctl status trading-copilot
   ```
2. Check if scans fired at expected times:
   ```bash
   journalctl -u trading-copilot --since "today" | grep -i "scan\|Starting"
   ```
3. Check ntfy topic in `.env`:
   ```bash
   grep NTFY .env
   ```
4. Test ntfy manually:
   ```bash
   curl -d "Test message" https://ntfy.sh/YOUR_TOPIC
   ```
5. Check conviction threshold — if too high, no signals fire:
   ```bash
   grep conviction_threshold config.yaml
   ```
   Lower it temporarily to `0.01` for testing, then restore.

---

### Breaking news monitor not firing

1. Check logs for the monitor tick:
   ```bash
   journalctl -u trading-copilot -f | grep -i "breaking\|BreakingNews\|rss"
   ```
2. Verify it only runs during market hours (13:30–20:00 UTC Mon–Fri).
3. If outside market hours, it's expected to be silent.
4. If inside market hours and missing — restart the service.

---

### Webhook buttons not working (Enter/Skip not registering)

1. Test nginx is up:
   ```bash
   sudo systemctl status nginx
   curl -I http://143.47.48.68/
   ```
2. Test FastAPI directly:
   ```bash
   curl -X POST "http://127.0.0.1:8000/skip?opportunity_id=1"
   ```
3. Check nginx error log:
   ```bash
   sudo tail -50 /var/log/nginx/error.log
   ```
4. Check FastAPI logs:
   ```bash
   journalctl -u trading-copilot -n 50 | grep "POST\|webhook\|enter\|skip\|exit"
   ```
5. Reload nginx if config was recently changed:
   ```bash
   sudo nginx -t && sudo systemctl reload nginx
   ```

---

### Ticker returning no data

1. Check if Tiingo has it:
   ```bash
   uv run python -c "
   import httpx, os
   from dotenv import load_dotenv; load_dotenv()
   r = httpx.get(f'https://api.tiingo.com/tiingo/daily/TICKER/prices?token={os.getenv(\"TIINGO_TOKEN\")}&startDate=2024-01-01')
   print(r.status_code, r.text[:200])
   "
   ```
2. Check Yahoo Finance symbol:
   ```bash
   uv run python -c "
   import yfinance as yf
   df = yf.download('TICKER.DE', period='5d', auto_adjust=True, progress=False)
   print(df)
   "
   ```
3. If Yahoo returns data, add the correct mapping to `YAHOO_FALLBACK` in `tiingo.py`.
4. If both fail, the ticker may be delisted or use a different symbol.

---

### ML model file missing

If `data/ml_signal_model.joblib` or `data/ensemble_model.joblib` are missing, the system falls back to rule-based signals automatically. To restore ML:
```bash
uv run python scripts/train_ml.py --watchlist --from 2018-01-01
uv run python scripts/train_ensemble.py --watchlist --from 2018-01-01
sudo systemctl restart trading-copilot
```

---

### DuckDB locked / database error

The DuckDB file (`data/market.duckdb`) can only be opened by one process at a time.

1. Stop the service:
   ```bash
   sudo systemctl stop trading-copilot
   ```
2. Run the backfill or query:
   ```bash
   uv run python scripts/backfill_prices.py --watchlist
   ```
3. Restart the service:
   ```bash
   sudo systemctl start trading-copilot
   ```

---

## 10. Scheduled Jobs Reference

| Job | Schedule (UTC) | ET equivalent | Purpose |
|-----|---------------|---------------|---------|
| EU scan | 11:30 Mon–Fri | 06:30 AM | Scan EU + prior US closes |
| US scan | 21:30 Mon–Fri | 04:30 PM | Scan US end-of-day closes |
| Position monitor | 21:00 Mon–Fri | 04:00 PM | Check stop/target/reversal |
| Breaking news | every 5 min, 13:30–20:00 | market hours | RSS feed + intraday alerts |

To change schedules, edit `config.yaml`:
```yaml
scheduler:
  scan_cron: "30 11 * * 1-5"
  scan_cron_us: "30 21 * * 1-5"
  monitor_cron: "0 21 * * 1-5"
```
Then restart the service.

---

## 11. Infrastructure Reference

| Component | Detail |
|-----------|--------|
| Server IP | `143.47.48.68` |
| Project dir | `/home/ubuntu/projects/trading-copilot` |
| Virtual env | `.venv/` (managed by `uv`) |
| Config file | `config.yaml` |
| Secrets | `.env` |
| DuckDB (OHLCV cache) | `data/market.duckdb` |
| SQLite (positions) | `data/positions.db` |
| ML model | `data/ml_signal_model.joblib` |
| Ensemble model | `data/ensemble_model.joblib` |
| Neural embedder | `data/sentiment_embedder.pt` |
| nginx config | `/etc/nginx/sites-available/trading-copilot` |
| systemd unit | `/etc/systemd/system/trading-copilot.service` |
| Webhook base URL | `http://143.47.48.68` (nginx → port 8000) |

### Key API keys (stored in `.env`)
- `TIINGO_TOKEN` — primary OHLCV data
- `NEWSAPI_KEY` — headlines for sentiment
- `NTFY_TOPIC` — push notification topic
