"""ntfy notification client with image and action button support."""
from __future__ import annotations

import logging
from base64 import b64encode

import httpx

from trading_copilot.config import settings
from trading_copilot.signals.scorer import Opportunity

# avoid circular import — NewsItem imported lazily inside send_breaking_news_alert

logger = logging.getLogger(__name__)


def _base_url() -> str:
    return settings.ntfy_base_url.rstrip("/")


def _topic_url() -> str:
    return f"{_base_url()}/{settings.ntfy_topic}"


def send_signal_alert(
    opportunity: Opportunity,
    chart_png: bytes,
    opportunity_id: str,
) -> bool:
    """Send a buy/sell signal notification with chart image and Enter/Skip buttons."""
    direction = opportunity.direction.upper()
    emoji = "📈" if opportunity.direction == "buy" else "📉"
    conviction_pct = f"{opportunity.conviction * 100:.0f}%"
    signal_names = ", ".join(s.signal_type.replace("_", " ") for s in opportunity.signals)

    entry = opportunity.entry_price
    stop = opportunity.stop_loss
    target = opportunity.target
    risk_pct = abs(entry - stop) / entry * 100
    gain_pct = abs(target - entry) / entry * 100
    rr = gain_pct / risk_pct if risk_pct else 0

    rsi = opportunity.indicators.get("rsi", "—")
    vol = opportunity.indicators.get("vol_ratio", "—")
    rsi_str = f"{rsi:.1f}" if isinstance(rsi, float) else str(rsi)
    vol_str = f"{vol:.1f}x" if isinstance(vol, float) else str(vol)

    sentiment_line = ""
    if opportunity.sentiment_score is not None:
        sent_label = "bullish" if opportunity.sentiment_score > 0.1 else ("bearish" if opportunity.sentiment_score < -0.1 else "neutral")
        themes = ", ".join(opportunity.sentiment_themes[:3]) if opportunity.sentiment_themes else "—"
        sentiment_line = f"\nSentiment:  {sent_label} ({themes})"
        for hl in (opportunity.top_headlines or [])[:2]:
            sentiment_line += f"\n  • {hl[:90]}"

    title = f"{emoji} {direction}: {opportunity.ticker}  [{conviction_pct} conviction]"
    body = (
        f"Entry:   ${entry:.2f}\n"
        f"Stop:    ${stop:.2f}  (−{risk_pct:.1f}%)\n"
        f"Target:  ${target:.2f}  (+{gain_pct:.1f}%)\n"
        f"R/R:     {rr:.1f} : 1  |  Hold ~10 days\n"
        f"Signals: {signal_names}\n"
        f"RSI: {rsi_str}  Vol: {vol_str}{sentiment_line}"
    )

    webhook = settings.webhook_base_url.rstrip("/")
    actions = (
        f"http, Enter, {webhook}/enter?opportunity_id={opportunity_id}, clear=true; "
        f"http, Skip, {webhook}/skip?opportunity_id={opportunity_id}, clear=true"
    )

    return _send_with_image(title, body, chart_png, actions, priority="high")


def send_exit_alert(
    ticker: str,
    position_id: str,
    reason: str,
    current_price: float,
    entry_price: float,
    chart_png: bytes,
    entry_date=None,
) -> bool:
    """Send a sell/exit notification for an open position."""
    from datetime import date as _date
    pnl_pct = (current_price - entry_price) / entry_price * 100
    sign = "+" if pnl_pct >= 0 else ""
    emoji = "🟢" if pnl_pct >= 0 else "🔴"

    reason_labels = {
        "stop": "Stop loss hit",
        "target": "Target reached",
        "signal_reversal": "Signal reversal",
        "manual": "Manual exit",
    }
    reason_str = reason_labels.get(reason, reason.replace("_", " "))

    hold_line = ""
    if entry_date is not None:
        today = _date.today()
        entry = entry_date if isinstance(entry_date, _date) else _date.fromisoformat(str(entry_date))
        hold_days = (today - entry).days
        hold_line = f"\nHeld:  {hold_days} day{'s' if hold_days != 1 else ''}"

    title = f"{emoji} EXIT: {ticker}  ({sign}{pnl_pct:.1f}%)"
    body = (
        f"Reason: {reason_str}\n"
        f"Entry:  ${entry_price:.2f}  →  Now: ${current_price:.2f}\n"
        f"P&L:    {sign}{pnl_pct:.1f}%{hold_line}"
    )

    webhook = settings.webhook_base_url.rstrip("/")
    actions = (
        f"http, Confirm exit, {webhook}/exit?position_id={position_id}&price={current_price}, clear=true"
    )

    return _send_with_image(title, body, chart_png, actions, priority="high")


def send_discovery_alert(
    ticker: str,
    reason: str,
    conviction: float,
    chart_png: bytes,
    current_price: float | None = None,
    rsi: float | None = None,
) -> bool:
    """Suggest adding a ticker to the watchlist."""
    title = f"🔭 Consider watching: {ticker}"
    extra = ""
    if current_price is not None:
        extra += f"\nPrice: ${current_price:.2f}"
    if rsi is not None:
        extra += f"  RSI: {rsi:.1f}"
    body = f"Reason: {reason}\nConviction: {conviction * 100:.0f}%{extra}"
    return _send_with_image(title, body, chart_png, priority="default")


def send_breaking_news_alert(
    item: "trading_copilot.news.rss.NewsItem",  # type: ignore[name-defined]
    opportunity: Opportunity,
    chart_png: bytes,
    opportunity_id: str,
) -> bool:
    """Send a breaking news signal alert with intraday chart."""
    from trading_copilot.news.rss import NewsItem  # noqa: F401  (type check only)

    direction = opportunity.direction.upper()
    emoji = "⚡📈" if opportunity.direction == "buy" else "⚡📉"
    conviction_pct = f"{opportunity.conviction * 100:.0f}%"

    entry = opportunity.entry_price
    stop = opportunity.stop_loss
    target = opportunity.target
    risk_pct = abs(entry - stop) / entry * 100
    gain_pct = abs(target - entry) / entry * 100
    rr = gain_pct / risk_pct if risk_pct else 0
    rsi = opportunity.indicators.get("rsi")
    rsi_str = f"  RSI: {rsi:.1f}" if isinstance(rsi, float) else ""

    title = f"{emoji} BREAKING: {opportunity.ticker} {direction}  [{conviction_pct}]"
    body = (
        f"News: {item.title[:100]}\n"
        f"Entry:  ${entry:.2f}{rsi_str}\n"
        f"Stop:   ${stop:.2f}  (−{risk_pct:.1f}%)\n"
        f"Target: ${target:.2f}  (+{gain_pct:.1f}%)\n"
        f"R/R:    {rr:.1f} : 1"
    )

    webhook = settings.webhook_base_url.rstrip("/")
    actions = (
        f"http, Enter, {webhook}/enter?opportunity_id={opportunity_id}, clear=true; "
        f"http, Skip, {webhook}/skip?opportunity_id={opportunity_id}, clear=true"
    )

    return _send_with_image(title, body, chart_png, actions, priority="urgent")


def _json_post(payload: dict) -> bool:
    """POST JSON to the ntfy publish endpoint (handles Unicode natively)."""
    try:
        resp = httpx.post(
            f"{_base_url()}/",
            json=payload,
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception as exc:
        logger.error("ntfy send failed: %s", exc)
        return False


def send_text(title: str, body: str, priority: str = "default") -> bool:
    """Send a plain text notification."""
    return _json_post({
        "topic": settings.ntfy_topic,
        "title": title,
        "message": body,
        "priority": _priority_int(priority),
    })


def _priority_int(p: str) -> int:
    return {"min": 1, "low": 2, "default": 3, "high": 4, "urgent": 5}.get(p, 3)


def _send_with_image(
    title: str,
    body: str,
    image_png: bytes,  # kept for API compatibility but not sent
    actions: str | None = None,
    priority: str = "default",
) -> bool:
    """Send text notification with action buttons (image skipped — iOS renders none)."""
    payload: dict = {
        "topic": settings.ntfy_topic,
        "title": title,
        "message": body,
        "priority": _priority_int(priority),
    }
    if actions:
        action_list = []
        for part in actions.split(";"):
            parts = [p.strip() for p in part.strip().split(",")]
            if len(parts) >= 3:
                action_list.append({
                    "action": parts[0],
                    "label": parts[1],
                    "url": parts[2],
                    "clear": "clear=true" in part,
                })
        payload["actions"] = action_list

    try:
        resp = httpx.post(f"{_base_url()}/", json=payload, timeout=15)
        resp.raise_for_status()
        logger.info("ntfy alert sent: %s", title)
        return True
    except Exception as exc:
        logger.error("ntfy send failed: %s", exc)
        return send_text(title, body, priority)
