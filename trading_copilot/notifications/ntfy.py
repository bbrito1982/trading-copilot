"""ntfy notification client with image and action button support."""
from __future__ import annotations

import logging
from base64 import b64encode

import httpx

from trading_copilot.config import settings
from trading_copilot.signals.scorer import Opportunity

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

    title = f"{emoji} {direction}: {opportunity.ticker}  [{conviction_pct} conviction]"
    body = (
        f"Entry zone: ${opportunity.entry_price:.2f}\n"
        f"Stop loss:  ${opportunity.stop_loss:.2f}\n"
        f"Target:     ${opportunity.target:.2f}\n"
        f"Signals:    {signal_names}\n"
        f"RSI: {opportunity.indicators.get('rsi', '—')}  "
        f"Vol ratio: {opportunity.indicators.get('vol_ratio', '—')}x"
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
) -> bool:
    """Send a sell/exit notification for an open position."""
    pnl_pct = (current_price - entry_price) / entry_price * 100
    sign = "+" if pnl_pct >= 0 else ""
    emoji = "🟢" if pnl_pct >= 0 else "🔴"

    title = f"{emoji} EXIT signal: {ticker}  ({sign}{pnl_pct:.1f}%)"
    body = (
        f"Reason:        {reason.replace('_', ' ')}\n"
        f"Entry price:   ${entry_price:.2f}\n"
        f"Current price: ${current_price:.2f}\n"
        f"P&L:           {sign}{pnl_pct:.1f}%"
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
) -> bool:
    """Suggest adding a ticker to the watchlist."""
    title = f"🔭 Consider watching: {ticker}"
    body = f"Reason: {reason}\nConviction: {conviction * 100:.0f}%"
    return _send_with_image(title, body, chart_png, priority="default")


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
    image_png: bytes,
    actions: str | None = None,
    priority: str = "default",
) -> bool:
    # ntfy supports image attachments via PUT with headers
    headers: dict[str, str] = {
        "X-Topic": settings.ntfy_topic,
        "X-Title": title,
        "X-Message": body.replace("\n", " | "),
        "X-Priority": str(_priority_int(priority)),
        "X-Filename": "chart.png",
        "Content-Type": "image/png",
    }
    if actions:
        headers["X-Actions"] = actions

    try:
        resp = httpx.put(
            f"{_base_url()}/{settings.ntfy_topic}",
            content=image_png,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        logger.info("ntfy alert sent: %s", title)
        return True
    except Exception as exc:
        logger.error("ntfy image send failed: %s", exc)
        # Fallback: text-only via JSON
        return send_text(title, body, priority)
