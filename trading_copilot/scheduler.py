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
from trading_copilot.signals.rules import compute_indicators
from trading_copilot.signals.scorer import score_ticker
from trading_copilot.tracker.models import create_tables
from trading_copilot.tracker.positions import (
    already_actioned_today,
    check_exit_conditions,
    get_open_positions,
    has_open_position,
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
        "Federal Reserve OR inflation OR recession OR \"interest rates\" OR "
        "GDP OR OPEC OR \"oil price\" OR \"treasury bonds\" OR \"stock market\""
    )
    try:
        from datetime import date, timedelta
        today = date.today()
        articles = fetch_newsapi(macro_query, from_date=today - timedelta(days=3), to_date=today)
        store_headlines(articles, macro_query)
        _macro_headlines_cache = [
            (a.get("title") or "") + " " + (a.get("description") or "") for a in articles
        ]
        logger.info("Macro headlines fetched: %d articles", len(_macro_headlines_cache))
    except Exception as exc:
        logger.warning("Macro headlines fetch failed: %s", exc)
        _macro_headlines_cache = []
    return _macro_headlines_cache


def _fetch_sentiment(ticker: str) -> tuple[float | None, list[str], list[str]]:
    """Fetch ticker-specific + macro headlines, blend, return (sentiment_score, themes, top_headlines)."""
    from datetime import date, timedelta
    query = TICKER_QUERIES.get(ticker)
    if not settings.news_api_key:
        return None, [], []
    try:
        today = date.today()

        # Ticker-specific score (neural)
        ticker_score: float | None = None
        themes: list[str] = []
        top_headlines: list[str] = []
        if query:
            articles = fetch_newsapi(query, from_date=today - timedelta(days=3), to_date=today)
            store_headlines(articles, query)
            headlines = [(a.get("title") or "") + " " + (a.get("description") or "") for a in articles]
            result = score_headlines_neural(ticker, headlines)
            ticker_score = result.score if headlines else None
            themes = result.matched_themes
            # Keep top 2 article titles for display in notifications
            top_headlines = [a["title"] for a in articles[:2] if a.get("title")]

        # Macro score (keyword theme table applied to market-wide headlines)
        macro_score: float | None = None
        macro_headlines = _fetch_macro_headlines()
        if macro_headlines:
            macro_scores = score_macro_headlines(macro_headlines, [ticker])
            macro_score = macro_scores.get(ticker)

        blended = blend_sentiment(ticker_score, macro_score, ticker_weight=0.6)
        if macro_score is not None and blended != ticker_score:
            themes = list(set(themes + ["macro"]))

        return blended, themes, top_headlines
    except Exception as exc:
        logger.warning("Sentiment fetch failed for %s: %s", ticker, exc)
        return None, [], []


def run_daily_scan():
    logger.info("=== Daily scan starting ===")
    send_text("🔍 Trading Copilot", "Daily scan started")

    found = 0
    ticker_rsi: dict[str, float] = {}
    macro_sentiment_score: float | None = None

    for ticker in watchlist:
        try:
            df = _fetch_df(ticker)
            if df.empty:
                continue

            ind = compute_indicators(df.copy(), signals_cfg)
            rsi_val = ind.iloc[-1].get("rsi")
            if rsi_val is not None:
                ticker_rsi[ticker] = round(float(rsi_val), 1)

            sentiment_score, sentiment_themes, top_headlines = _fetch_sentiment(ticker)
            if sentiment_score is not None and macro_sentiment_score is None:
                macro_sentiment_score = sentiment_score

            opp = score_ticker(
                ticker, df, signals_cfg, swing_cfg,
                sentiment_score=sentiment_score,
                sentiment_themes=sentiment_themes,
            )
            if opp is not None:
                opp.top_headlines = top_headlines
            if opp is None or opp.conviction < conviction_threshold:
                continue

            if has_open_position(ticker):
                logger.info("Skipping %s — already have an open position", ticker)
                continue
            if already_actioned_today(ticker):
                logger.info("Skipping %s — already entered or skipped today", ticker)
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
            current_price = float(df.iloc[-1].get("adj_close", df.iloc[-1].get("close", 0)))
            rsi_val = opp.indicators.get("rsi")
            send_discovery_alert(ticker, reason, opp.conviction, chart, current_price=current_price, rsi=rsi_val)
            discoveries += 1
        except Exception as exc:
            logger.error("Error in discovery %s: %s", ticker, exc)

    summary_lines = [f"Signals: {found}  |  Discoveries: {discoveries}"]

    # Market snapshot: key benchmarks + overbought/oversold counts
    key_benchmarks = ["SPY", "QQQ", "GLD", "TLT"]
    benchmark_str = "  ".join(
        f"{t} {ticker_rsi[t]:.0f}" for t in key_benchmarks if t in ticker_rsi
    )
    if benchmark_str:
        summary_lines.append(f"RSI: {benchmark_str}")

    overbought = [t for t, r in ticker_rsi.items() if r >= 70]
    oversold = [t for t, r in ticker_rsi.items() if r <= 30]
    if overbought:
        summary_lines.append(f"Overbought (≥70): {', '.join(overbought)}")
    if oversold:
        summary_lines.append(f"Oversold (≤30): {', '.join(oversold)}")
    if not overbought and not oversold:
        summary_lines.append("All tickers in neutral RSI range")

    if macro_sentiment_score is not None:
        sent_label = "bullish" if macro_sentiment_score > 0.1 else ("bearish" if macro_sentiment_score < -0.1 else "neutral")
        summary_lines.append(f"Macro sentiment: {sent_label} ({macro_sentiment_score:+.2f})")

    send_text("✅ Scan complete", "\n".join(summary_lines))
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
                entry_date=pos.entry_date,
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
    sched_cfg = config.get("scheduler", {})
    scan_cron = sched_cfg.get("scan_cron", "30 11 * * 1-5")
    scan_cron_us = sched_cfg.get("scan_cron_us", "30 21 * * 1-5")
    monitor_cron = sched_cfg.get("monitor_cron", "0 21 * * 1-5")

    def _add_cron(func, cron: str, job_id: str):
        parts = cron.split()
        scheduler.add_job(func, "cron", minute=parts[0], hour=parts[1], day_of_week=parts[4], id=job_id)

    _add_cron(run_daily_scan, scan_cron, "daily_scan_eu")
    _add_cron(run_daily_scan, scan_cron_us, "daily_scan_us")
    _add_cron(run_position_monitor, monitor_cron, "position_monitor")
    scheduler.add_job(breaking_monitor.tick, "interval", minutes=5, id="breaking_news")

    scheduler.start()
    logger.info(
        "Scheduler started. EU scan: %s UTC  US scan: %s UTC  Monitor: %s UTC",
        scan_cron, scan_cron_us, monitor_cron,
    )

    # Run FastAPI webhook in the same process
    uvicorn.run(
        "trading_copilot.api.webhook:app",
        host="127.0.0.1",
        port=8000,
        log_level="info",
    )


if __name__ == "__main__":
    main()
