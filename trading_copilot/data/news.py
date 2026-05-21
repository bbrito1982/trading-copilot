"""News headline fetcher with DuckDB cache."""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb
import httpx
import pandas as pd

from trading_copilot.config import settings

logger = logging.getLogger(__name__)

NEWSAPI_BASE = "https://newsapi.org/v2/everything"


def _conn(db_path: str | None = None) -> duckdb.DuckDBPyConnection:
    path = db_path or settings.duckdb_path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS news (
            id          TEXT PRIMARY KEY,
            published   TIMESTAMP,
            source      TEXT,
            title       TEXT,
            description TEXT,
            url         TEXT,
            query       TEXT
        )
    """)
    return con


def fetch_newsapi(
    query: str,
    from_date: date,
    to_date: date,
    page_size: int = 100,
) -> list[dict]:
    if not settings.news_api_key:
        logger.warning("NEWS_API_KEY not set, skipping news fetch")
        return []

    headers = {"X-Api-Key": settings.news_api_key}
    params = {
        "q": query,
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
    }
    resp = httpx.get(NEWSAPI_BASE, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("articles", [])


def store_headlines(articles: list[dict], query: str, db_path: str | None = None) -> int:
    if not articles:
        return 0
    con = _conn(db_path)
    rows = []
    for a in articles:
        pub = a.get("publishedAt", "")
        rows.append({
            "id": a.get("url", pub),
            "published": pub,
            "source": a.get("source", {}).get("name", ""),
            "title": a.get("title", ""),
            "description": a.get("description", ""),
            "url": a.get("url", ""),
            "query": query,
        })
    df = pd.DataFrame(rows)
    con.execute("INSERT OR IGNORE INTO news SELECT * FROM df")
    count = len(rows)
    con.close()
    return count


def get_headlines(
    from_date: date,
    to_date: date,
    query: str | None = None,
    db_path: str | None = None,
) -> pd.DataFrame:
    """Return cached headlines within date range."""
    con = _conn(db_path)
    if query:
        df = con.execute(
            "SELECT * FROM news WHERE published >= ? AND published <= ? AND query = ? ORDER BY published DESC",
            [from_date.isoformat(), to_date.isoformat(), query],
        ).df()
    else:
        df = con.execute(
            "SELECT * FROM news WHERE published >= ? AND published <= ? ORDER BY published DESC",
            [from_date.isoformat(), to_date.isoformat()],
        ).df()
    con.close()
    return df
