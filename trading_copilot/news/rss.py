"""RSS feed poller for breaking financial news.

Polls a curated list of financial RSS feeds every few minutes, filters
headlines that mention watchlist tickers, and yields new unseen items.

Deduplication is done by URL hash stored in memory (resets on restart)
and optionally persisted to a small SQLite set.
"""
from __future__ import annotations

import hashlib
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feed catalogue — free, no auth, ~1-3 min latency
# ---------------------------------------------------------------------------
RSS_FEEDS = [
    # Reuters
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/technologyNews",
    # Yahoo Finance
    "https://finance.yahoo.com/rss/headline",
    # MarketWatch
    "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
    # Seeking Alpha (public)
    "https://seekingalpha.com/feed.xml",
    # CNBC
    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
]

# Company name → ticker mapping for mention detection
COMPANY_NAMES: dict[str, str] = {
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL",
    "amazon": "AMZN", "nvidia": "NVDA", "nvda": "NVDA",
    "s&p 500": "SPY", "s&p500": "SPY", "nasdaq": "QQQ",
    "gold": "GLD", "treasury": "TLT", "t-bond": "TLT",
    "energy": "XLE", "oil": "XLE", "crude": "XLE",
    "bank": "XLF", "financial": "XLF", "jpmorgan": "XLF",
    # EU watchlist tickers
    "dividend": "IDVY", "european dividend": "IDVY",
    "uranium": "URNU.DE", "nuclear energy": "URNU.DE",
    "hydrogen": "HYCN.DE", "fuel cell": "HYCN.DE",
    "daimler": "DTG.DE", "daimler truck": "DTG.DE",
    "critical metal": "CEBT.DE", "essential metal": "CEBT.DE", "lithium": "CEBT.DE",
    "water": "IQQQ.DE", "water infrastructure": "IQQQ.DE",
    "defense": "DFEN.DE", "defence": "DFEN.DE", "vaneck defense": "DFEN.DE",
    "healthcare innovation": "HEAL.UK", "biotech": "HEAL.UK",
    "euronav": "EURN", "tanker": "EURN",
    "robotics": "RBOT", "automation": "RBOT",
    "quantum": "WQTM", "quantum computing": "WQTM",
    "space": "JEDI", "satellite": "JEDI",
}


@dataclass
class NewsItem:
    title: str
    url: str
    published: datetime
    source: str
    tickers: list[str] = field(default_factory=list)

    @property
    def uid(self) -> str:
        return hashlib.md5(self.url.encode()).hexdigest()


def _detect_tickers(text: str, watchlist: list[str]) -> list[str]:
    """Return watchlist tickers mentioned in text by ticker symbol or company name."""
    text_lower = text.lower()
    found = set()
    # Direct ticker mention (e.g. "$AAPL" or "AAPL")
    for ticker in watchlist:
        if re.search(rf"\b{re.escape(ticker)}\b", text, re.IGNORECASE):
            found.add(ticker)
    # Company name mention
    for name, ticker in COMPANY_NAMES.items():
        if ticker in watchlist and name in text_lower:
            found.add(ticker)
    return sorted(found)


def _parse_feed(xml_text: str, source: str, watchlist: list[str]) -> list[NewsItem]:
    items = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    # RSS 2.0
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        url = (item.findtext("link") or item.findtext("guid") or "").strip()
        pub_str = item.findtext("pubDate") or ""
        try:
            pub = parsedate_to_datetime(pub_str).astimezone(timezone.utc).replace(tzinfo=None)
        except Exception:
            pub = datetime.utcnow()
        tickers = _detect_tickers(title, watchlist)
        if tickers:
            items.append(NewsItem(title=title, url=url, published=pub,
                                  source=source, tickers=tickers))
    # Atom
    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", namespaces=ns) or "").strip()
        url = entry.find("atom:link", ns)
        url = url.get("href", "") if url is not None else ""
        pub_str = entry.findtext("atom:published", namespaces=ns) or ""
        try:
            pub = datetime.fromisoformat(pub_str.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            pub = datetime.utcnow()
        tickers = _detect_tickers(title, watchlist)
        if tickers:
            items.append(NewsItem(title=title, url=url, published=pub,
                                  source=source, tickers=tickers))
    return items


class RSSPoller:
    """Stateful RSS poller — tracks seen item UIDs to avoid duplicate alerts."""

    def __init__(self, watchlist: list[str], feeds: list[str] | None = None):
        self._watchlist = [t.upper() for t in watchlist]
        self._feeds = feeds or RSS_FEEDS
        self._seen: set[str] = set()

    def poll(self) -> list[NewsItem]:
        """Fetch all feeds and return unseen items mentioning watchlist tickers."""
        new_items: list[NewsItem] = []
        for feed_url in self._feeds:
            try:
                r = httpx.get(feed_url, timeout=10, follow_redirects=True,
                               headers={"User-Agent": "trading-copilot/1.0"})
                if r.status_code != 200:
                    continue
                items = _parse_feed(r.text, feed_url, self._watchlist)
                for item in items:
                    if item.uid not in self._seen:
                        self._seen.add(item.uid)
                        new_items.append(item)
            except Exception as exc:
                logger.debug("RSS fetch failed %s: %s", feed_url, exc)

        if new_items:
            logger.info("RSS poll: %d new items across %d tickers",
                        len(new_items), len({t for i in new_items for t in i.tickers}))
        return new_items
