"""FastAPI webhook for ntfy action button callbacks."""
from __future__ import annotations

import logging

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse

from trading_copilot.tracker.positions import (
    enter_position,
    skip_opportunity,
    close_position,
    get_open_positions,
    get_opportunity,
)
from trading_copilot.notifications.ntfy import send_text

logger = logging.getLogger(__name__)

app = FastAPI(title="Trading Copilot Webhook")


@app.post("/enter")
async def enter(opportunity_id: str = Query(...)):
    """Called when user taps 'Enter' on a signal notification."""
    opp = get_opportunity(opportunity_id)
    if not opp:
        raise HTTPException(status_code=404, detail="Opportunity not found")
    pos = enter_position(opportunity_id)
    send_text(
        title=f"Position opened: {opp.ticker} {opp.direction.upper()}",
        body=(
            f"Entry:  ${opp.entry_price:.2f}\n"
            f"Stop:   ${opp.stop_loss:.2f}\n"
            f"Target: ${opp.target:.2f}\n"
            f"Conviction: {opp.conviction * 100:.0f}%"
        ),
        priority="high",
    )
    logger.info("Entered position %s via webhook", pos.id)
    return {"status": "ok", "position_id": pos.id}


@app.post("/skip")
async def skip(opportunity_id: str = Query(...)):
    """Called when user taps 'Skip' on a signal notification."""
    ok = skip_opportunity(opportunity_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Opportunity not found")
    send_text(
        title="Signal skipped",
        body="Opportunity marked as skipped.",
        priority="low",
    )
    return {"status": "skipped"}


@app.post("/exit")
async def exit_position(
    position_id: str = Query(...),
    price: float = Query(...),
):
    """Called when user confirms exit from a position monitor alert."""
    pos = close_position(position_id, exit_price=price, exit_reason="manual")
    if not pos:
        raise HTTPException(status_code=404, detail="Position not found or already closed")
    pnl = pos.pnl_pct or 0.0
    sign = "+" if pnl >= 0 else ""
    emoji = "UP" if pnl >= 0 else "DOWN"
    send_text(
        title=f"Position closed: {pos.ticker}  {sign}{pnl:.1f}%",
        body=(
            f"Exit:  ${pos.exit_price:.2f}\n"
            f"Entry: ${pos.entry_price:.2f}\n"
            f"P&L:   {sign}{pnl:.1f}%  ({emoji})"
        ),
        priority="high",
    )
    return {"status": "closed", "pnl_pct": pnl}


@app.get("/positions")
async def list_positions():
    """List all open positions."""
    positions = get_open_positions()
    return [
        {
            "id": p.id,
            "ticker": p.ticker,
            "entry_price": p.entry_price,
            "entry_date": str(p.entry_date),
            "shares": p.shares,
        }
        for p in positions
    ]


@app.get("/health")
async def health():
    return {"status": "ok"}
