"""APScheduler entry point: daily scan + position monitor."""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import date, timedelta

import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler

from trading_copilot.config import config, settings
from trading_copilot.data.tiingo import get_ohlcv, get_ohlcv_cached_only
from trading_copilot.data.news import fetch_newsapi, store_headlines
from trading_copilot.data.universe import FULL_UNIVERSE
from trading_copilot.sentiment.tagger import TICKER_QUERIES, score_headlines, score_headlines_neural, score_macro_headlines, blend_sentiment
from trading_copilot.notifications.charts import generate_chart
from trading_copilot.notifications.ntfy import (
    send_discovery_alert,
    send_exit_alert,
    send_signal_alert,
    send_text,
)
from trading_copilot.signals.scorer import score_ticker
from trading_copilot.tracker.models import create_tables
from trading_copilot.tracker.positions import (
    check_exit_conditions,
    get_open_positions,
    save_opportunity,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

signals_cfg = config.get("signals", {})
swing_cfg = config.get("swing", {})
notif_cfg = config.get("notifications", {})
chart_lookback = notif_cfg.get("chart_lookback_days", 90)
conviction_threshold = config.get("conviction_threshold", 0.6)
discovery_threshold = config.get("discovery_threshold", 0.7)
breaking_conviction_threshold = config.get("breaking_conviction_threshold", 0.4)
watchlist: list[str] = config.get("watchlist", [])


def _fetch_df(ticker: str, days: int = 200):
    end = date.today()
    start = end - timedelta(days=days)
    return get_ohlcv(ticker, start=start, end=end)


_macro_headlines_cache: list[str] | None = None

def _fetch_macro_headlines() -> list[str]:
    """Fetch market-wide macro headlines once per scan and cache for the session."""
    global _macro_headlines_cache
    if _macro_headlines_cache is not None:
        return _macro_headlines_cache
    if not settings.news_api_key:
        _macro_headlines_cache = []
        return []
    macro_query = (
        "Federal Reserve interest rates inflation recession GDP "
        "oil price OPEC dollar treasury bonds market crash rally"
    )
    try:
        from datetime import date, timedelta
        today = date.today()
        articles = fetch_newsapi(macro_query, from_date=today - timedelta(days=3), to_date=today)
        store_headlines(articles, macro_query)
        _macro_headlines_cache = [
            a.get("title", "") + " " + (a.get("description") or "") for a in articles
        ]
        logger.info("Macro headlines fetched: %d articles", len(_macro_headlines_cache))
    except Exception as exc:
        logger.warning("Macro headlines fetch failed: %s", exc)
        _macro_headlines_cache = []
    return _macro_headlines_cache


def _fetch_sentiment(ticker: str) -> tuple[float | None, list[str]]:
    """Fetch ticker-specific + macro headlines, blend, return (sentiment_score, themes)."""
    from datetime import date, timedelta
    query = TICKER_QUERIES.get(ticker)
    if not settings.news_api_key:
        return None, []
    try:
        today = date.today()

        # Ticker-specific score (neural)
        ticker_score: float | None = None
        themes: list[str] = []
        if query:
            articles = fetch_newsapi(query, from_date=today - timedelta(days=3), to_date=today)
            store_headlines(articles, query)
            headlines = [a.get("title", "") + " " + (a.get("description") or "") for a in articles]
            result = score_headlines_neural(ticker, headlines)
            ticker_score = result.score if headlines else None
            themes = result.matched_themes

        # Macro score (keyword theme table applied to market-wide headlines)
        macro_score: float | None = None
        macro_headlines = _fetch_macro_headlines()
        if macro_headlines:
            macro_scores = score_macro_headlines(macro_headlines, [ticker])
            macro_score = macro_scores.get(ticker)

        blended = blend_sentiment(ticker_score, macro_score, ticker_weight=0.6)
        if macro_score is not None and blended != ticker_score:
            themes = list(set(themes + ["macro"]))

        return blended, themes
    except Exception as exc:
        logger.warning("Sentiment fetch failed for %s: %s", ticker, exc)
        return None, []


def run_daily_scan():
    logger.info("=== Daily scan starting ===")
    send_text("🔍 Trading Copilot", "Daily scan started")

    found = 0
    for ticker in watchlist:
        try:
            df = _fetch_df(ticker)
            if df.empty:
                continue
            sentiment_score, sentiment_themes = _fetch_sentiment(ticker)
            opp = score_ticker(
                ticker, df, signals_cfg, swing_cfg,
                sentiment_score=sentiment_score,
                sentiment_themes=sentiment_themes,
            )
            if opp is None or opp.conviction < conviction_threshold:
                continue

            record = save_opportunity(opp)
            chart = generate_chart(ticker, df, opportunity=opp, cfg=signals_cfg, lookback_days=chart_lookback)
            send_signal_alert(opp, chart, record.id)
            found += 1
            logger.info("Signal: %s %s conviction=%.2f", ticker, opp.direction, opp.conviction)
            time.sleep(0.3)  # respect Tiingo rate limit
        except Exception as exc:
            logger.error("Error scanning %s: %s", ticker, exc)

    # Discovery screener: use only cached data to avoid burning API quota
    universe_extras = [t for t in FULL_UNIVERSE if t not in watchlist]
    discoveries = 0
    for ticker in universe_extras:
        try:
            df = get_ohlcv_cached_only(ticker)
            if df.empty or len(df) < 60:
                continue
            opp = score_ticker(ticker, df, signals_cfg, swing_cfg)
            if opp is None or opp.conviction < discovery_threshold:
                continue

            chart = generate_chart(ticker, df, opportunity=opp, cfg=signals_cfg, lookback_days=chart_lookback)
            reason = ", ".join(s.signal_type.replace("_", " ") for s in opp.signals)
            send_discovery_alert(ticker, reason, opp.conviction, chart)
            discoveries += 1
        except Exception as exc:
            logger.error("Error in discovery %s: %s", ticker, exc)

    send_text(
        "✅ Scan complete",
        f"Watchlist signals: {found}  |  Discoveries: {discoveries}",
    )
    logger.info("=== Daily scan done: %d signals, %d discoveries ===", found, discoveries)


def run_position_monitor():
    logger.info("=== Position monitor starting ===")
    positions = get_open_positions()
    if not positions:
        logger.info("No open positions")
        return

    for pos in positions:
        try:
            df = _fetch_df(pos.ticker)
            if df.empty:
                continue

            result = check_exit_conditions(pos, df, config)
            if result is None:
                continue

            _, reason, current_price = result
            chart = generate_chart(
                pos.ticker, df,
                cfg=signals_cfg,
                lookback_days=chart_lookback,
                entry_date=pos.entry_date,
                entry_price=pos.entry_price,
            )
            send_exit_alert(
                ticker=pos.ticker,
                position_id=pos.id,
                reason=reason,
                current_price=current_price,
                entry_price=pos.entry_price,
                chart_png=chart,
            )
            logger.info("Exit alert sent: %s reason=%s", pos.ticker, reason)
        except Exception as exc:
            logger.error("Error monitoring position %s: %s", pos.id, exc)


def main():
    create_tables()

    from trading_copilot.news.breaking import BreakingNewsMonitor
    breaking_monitor = BreakingNewsMonitor(
        watchlist=watchlist,
        signals_cfg=signals_cfg,
        swing_cfg=swing_cfg,
        conviction_threshold=breaking_conviction_threshold,
    )

    scheduler = BackgroundScheduler(timezone="UTC")
    scan_cron = config.get("scheduler", {}).get("scan_cron", "30 11 * * 1-5")
    monitor_cron = config.get("scheduler", {}).get("monitor_cron", "0 21 * * 1-5")

    # Parse "MIN HOUR DOW_OF_WEEK" style cron
    scan_parts = scan_cron.split()
    monitor_parts = monitor_cron.split()

    scheduler.add_job(
        run_daily_scan, "cron",
        minute=scan_parts[0], hour=scan_parts[1], day_of_week=scan_parts[4],
        id="daily_scan",
    )
    scheduler.add_job(
        run_position_monitor, "cron",
        minute=monitor_parts[0], hour=monitor_parts[1], day_of_week=monitor_parts[4],
        id="position_monitor",
    )
    scheduler.add_job(
        breaking_monitor.tick, "interval", minutes=5,
        id="breaking_news",
    )

    scheduler.start()
    logger.info("Scheduler started. Scan: %s UTC  Monitor: %s UTC", scan_cron, monitor_cron)

    # Run FastAPI webhook in the same process
    uvicorn.run(
        "trading_copilot.api.webhook:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )


if __name__ == "__main__":
    main()
