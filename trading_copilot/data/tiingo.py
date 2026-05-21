"""Tiingo OHLCV fetcher with DuckDB cache."""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb
import httpx
import pandas as pd

from trading_copilot.config import settings

logger = logging.getLogger(__name__)

TIINGO_BASE = "https://api.tiingo.com/tiingo/daily"

# Tickers Tiingo doesn't carry → Yahoo Finance symbol mapping
YAHOO_FALLBACK: dict[str, str] = {
    "IGLN":    "IGLN.L",
    "EURN":    "EURN.BR",
    "CEA1":    "CEA1.F",
    "ALDO":    "ALDO.MI",
    "URNU.DE": "URNU.DE",
    "HYCN.DE": "HYCN.DE",
    "DTG.DE":  "DTG.DE",
    "HEAL.UK": "HEAL.L",
    "CEBT.DE": "CEBT.DE",
    "IQQQ.DE": "IQQQ.DE",
    "DFEN.DE":  "DFEN.DE",
}


def _conn(db_path: str | None = None) -> duckdb.DuckDBPyConnection:
    path = db_path or settings.duckdb_path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            ticker      TEXT,
            date        DATE,
            open        DOUBLE,
            high        DOUBLE,
            low         DOUBLE,
            close       DOUBLE,
            volume      BIGINT,
            adj_close   DOUBLE,
            adj_open    DOUBLE,
            adj_high    DOUBLE,
            adj_low     DOUBLE,
            div_cash    DOUBLE,
            split_factor DOUBLE,
            PRIMARY KEY (ticker, date)
        )
    """)
    return con


def _fetch_tiingo(
    ticker: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    if not settings.tiingo_api_key:
        raise ValueError("TIINGO_API_KEY not set")

    url = f"{TIINGO_BASE}/{ticker}/prices"
    params = {
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "token": settings.tiingo_api_key,
        "format": "json",
    }
    resp = httpx.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["ticker"] = ticker.upper()
    df = df.rename(columns={
        "adjClose": "adj_close",
        "adjOpen": "adj_open",
        "adjHigh": "adj_high",
        "adjLow": "adj_low",
        "divCash": "div_cash",
        "splitFactor": "split_factor",
    })
    # Keep only schema columns, in order
    cols = ["ticker", "date", "open", "high", "low", "close", "volume",
            "adj_close", "adj_open", "adj_high", "adj_low", "div_cash", "split_factor"]
    return df[[c for c in cols if c in df.columns]]


def _fetch_yahoo(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Fetch OHLCV from Yahoo Finance and normalise to the same schema as Tiingo."""
    import yfinance as yf
    yf_sym = YAHOO_FALLBACK.get(ticker.upper(), ticker)
    df = yf.download(yf_sym, start=start.isoformat(), end=end.isoformat(),
                     progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.rename(columns={"Date": "date", "Open": "open", "High": "high",
                             "Low": "low", "Close": "close", "Volume": "volume"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["ticker"] = ticker.upper()
    df["adj_close"] = df["close"]
    df["adj_open"]  = df["open"]
    df["adj_high"]  = df["high"]
    df["adj_low"]   = df["low"]
    df["div_cash"]      = 0.0
    df["split_factor"]  = 1.0
    cols = ["ticker", "date", "open", "high", "low", "close", "volume",
            "adj_close", "adj_open", "adj_high", "adj_low", "div_cash", "split_factor"]
    return df[[c for c in cols if c in df.columns]]


def _latest_cached_date(con: duckdb.DuckDBPyConnection, ticker: str) -> date | None:
    row = con.execute(
        "SELECT MAX(date) FROM ohlcv WHERE ticker = ?", [ticker.upper()]
    ).fetchone()
    return row[0] if row and row[0] else None


def get_ohlcv(
    ticker: str,
    start: date | None = None,
    end: date | None = None,
    db_path: str | None = None,
) -> pd.DataFrame:
    """Return adjusted OHLCV for ticker, fetching missing data from Tiingo."""
    ticker = ticker.upper()
    end = end or date.today()
    con = _conn(db_path)

    latest = _latest_cached_date(con, ticker)
    fetch_start = start or date(2010, 1, 1)

    # Consider cache fresh if latest is within 1 calendar day of end (covers today not yet published)
    if latest is not None and latest >= end - timedelta(days=1):
        logger.debug("%s: fully cached up to %s", ticker, latest)
    else:
        remote_start = (latest + timedelta(days=1)) if latest else fetch_start
        logger.info("%s: fetching %s → %s from Tiingo", ticker, remote_start, end)
        df_new = pd.DataFrame()
        try:
            df_new = _fetch_tiingo(ticker, remote_start, end)
        except Exception as exc:
            logger.warning("%s: Tiingo fetch failed: %s — trying Yahoo Finance", ticker, exc)

        if df_new.empty and ticker.upper() in YAHOO_FALLBACK:
            try:
                df_new = _fetch_yahoo(ticker, remote_start, end)
                if not df_new.empty:
                    logger.info("%s: fetched %d rows from Yahoo Finance", ticker, len(df_new))
            except Exception as exc2:
                logger.warning("%s: Yahoo Finance fallback also failed: %s", ticker, exc2)

        if not df_new.empty:
            con.execute("INSERT OR REPLACE INTO ohlcv SELECT * FROM df_new")
            logger.info("%s: cached %d new rows", ticker, len(df_new))

    query = "SELECT * FROM ohlcv WHERE ticker = ? AND date >= ? AND date <= ? ORDER BY date"
    df = con.execute(query, [ticker, fetch_start, end]).df()
    con.close()
    return df


def get_ohlcv_cached_only(
    ticker: str,
    start: date | None = None,
    end: date | None = None,
    db_path: str | None = None,
) -> pd.DataFrame:
    """Return only cached OHLCV — never hits Tiingo. Used for universe screener."""
    ticker = ticker.upper()
    end = end or date.today()
    fetch_start = start or date(2010, 1, 1)
    con = _conn(db_path)
    df = con.execute(
        "SELECT * FROM ohlcv WHERE ticker = ? AND date >= ? AND date <= ? ORDER BY date",
        [ticker, fetch_start, end],
    ).df()
    con.close()
    return df


def get_ohlcv_multi(
    tickers: list[str],
    start: date | None = None,
    end: date | None = None,
    db_path: str | None = None,
    delay: float = 0.25,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for multiple tickers with rate-limit delay between requests."""
    import time
    results = {}
    for t in tickers:
        results[t] = get_ohlcv(t, start=start, end=end, db_path=db_path)
        time.sleep(delay)
    return results
