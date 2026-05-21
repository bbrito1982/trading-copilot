"""Tiingo IEX intraday OHLCV fetcher."""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone

import httpx
import pandas as pd

from trading_copilot.config import settings

logger = logging.getLogger(__name__)

_BASE = "https://api.tiingo.com/iex"
_FREQ = "5min"


def get_intraday_bars(ticker: str, lookback_bars: int = 78) -> pd.DataFrame:
    """Fetch today's 5-min OHLCV bars for *ticker* from Tiingo IEX.

    Returns a DataFrame with columns:
        date (UTC datetime), open, high, low, close, volume (if available)

    lookback_bars: how many 5-min bars to return (78 = full trading day).
    """
    today = date.today().isoformat()
    url = f"{_BASE}/{ticker.upper()}/prices"
    params = {
        "startDate": today,
        "resampleFreq": _FREQ,
        "token": settings.tiingo_api_key,
    }
    try:
        r = httpx.get(url, params=params, timeout=15, follow_redirects=True)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        logger.warning("Intraday fetch failed for %s: %s", ticker, exc)
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df = df.rename(columns={"close": "adj_close", "open": "adj_open",
                             "high": "adj_high", "low": "adj_low"})
    for col in ("adj_open", "adj_high", "adj_low", "adj_close"):
        if col not in df.columns:
            df[col] = df.get(col.replace("adj_", ""), float("nan"))

    if "volume" not in df.columns:
        df["volume"] = 0

    df = df[["date", "adj_open", "adj_high", "adj_low", "adj_close", "volume"]].dropna()
    df = df.sort_values("date").tail(lookback_bars).reset_index(drop=True)
    return df
