"""Breaking news monitor.

Processes NewsItems from the RSS poller: scores each item with the neural
embedder (or keyword fallback), fetches intraday bars for affected tickers,
runs the signal scorer, and fires an ntfy alert when a signal is present.

Designed to be called every 5 minutes during market hours.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from trading_copilot.data.intraday import get_intraday_bars
from trading_copilot.news.rss import NewsItem, RSSPoller
from trading_copilot.notifications.charts import generate_intraday_chart
from trading_copilot.notifications.ntfy import send_breaking_news_alert
from trading_copilot.sentiment.tagger import score_headlines_neural
from trading_copilot.signals.scorer import score_ticker
from trading_copilot.tracker.positions import save_opportunity

logger = logging.getLogger(__name__)

# Minimum absolute sentiment score to trigger an intraday check
SENTIMENT_THRESHOLD = 0.35


def _is_market_hours() -> bool:
    now_et = datetime.now(tz=timezone.utc).astimezone(
        __import__("zoneinfo").ZoneInfo("America/New_York")
    )
    if now_et.weekday() >= 5:  # Saturday / Sunday
        return False
    open_h, close_h = 9, 16
    open_m = 30
    t = (now_et.hour, now_et.minute)
    return (t >= (open_h, open_m)) and (t < (close_h, 0))


def process_news_item(
    item: NewsItem,
    signals_cfg: dict,
    swing_cfg: dict,
    conviction_threshold: float = 0.4,
) -> bool:
    """Score one NewsItem; if signal found, alert and return True."""
    for ticker in item.tickers:
        try:
            result = score_headlines_neural(ticker, [item.title])
            score = result.score
            if score is None or abs(score) < SENTIMENT_THRESHOLD:
                continue

            logger.info(
                "Breaking news hit: %s score=%.2f  '%s'",
                ticker, score, item.title[:80],
            )

            df = get_intraday_bars(ticker, lookback_bars=78)
            if df.empty or len(df) < 10:
                logger.debug("Not enough intraday bars for %s", ticker)
                continue

            opp = score_ticker(
                ticker, df, signals_cfg, swing_cfg,
                sentiment_score=score,
                sentiment_themes=result.matched_themes,
            )
            if opp is None or opp.conviction < conviction_threshold:
                continue

            chart = generate_intraday_chart(ticker, df, opportunity=opp)
            record = save_opportunity(opp)
            send_breaking_news_alert(
                item=item,
                opportunity=opp,
                chart_png=chart,
                opportunity_id=record.id,
            )
            logger.info(
                "Breaking news alert sent: %s %s conviction=%.2f",
                ticker, opp.direction, opp.conviction,
            )
            return True

        except Exception as exc:
            logger.warning("Error processing breaking news for %s: %s", ticker, exc)

    return False


class BreakingNewsMonitor:
    """Stateful monitor — holds the RSS poller across scheduler ticks."""

    def __init__(
        self,
        watchlist: list[str],
        signals_cfg: dict,
        swing_cfg: dict,
        conviction_threshold: float = 0.4,
    ):
        self._poller = RSSPoller(watchlist)
        self._signals_cfg = signals_cfg
        self._swing_cfg = swing_cfg
        self._conviction_threshold = conviction_threshold

    def tick(self) -> None:
        """Called every 5 minutes by the scheduler."""
        if not _is_market_hours():
            return

        items = self._poller.poll()
        if not items:
            return

        logger.info("Breaking news: %d new items to evaluate", len(items))
        for item in items:
            process_news_item(
                item,
                self._signals_cfg,
                self._swing_cfg,
                self._conviction_threshold,
            )
