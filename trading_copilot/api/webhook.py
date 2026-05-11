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
)
from trading_copilot.notifications.ntfy import send_text

logger = logging.getLogger(__name__)

app = FastAPI(title="Trading Copilot Webhook")


@app.post("/enter")
async def enter(opportunity_id: str = Query(...)):
    """Called when user taps 'Enter' on a signal notification."""
    pos = enter_position(opportunity_id)
    if not pos:
        raise HTTPException(status_code=404, detail="Opportunity not found")
    send_text(
        title=f"✅ Position opened: {pos.ticker}",
        body=f"Entered {pos.ticker} @ ${pos.entry_price:.2f}  |  Position ID: {pos.id}",
    )
    logger.info("Entered position %s via webhook", pos.id)
    return {"status": "ok", "position_id": pos.id}


@app.post("/skip")
async def skip(opportunity_id: str = Query(...)):
    """Called when user taps 'Skip' on a signal notification."""
    ok = skip_opportunity(opportunity_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Opportunity not found")
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
    sign = "+" if (pos.pnl_pct or 0) >= 0 else ""
    send_text(
        title=f"{'🟢' if (pos.pnl_pct or 0) >= 0 else '🔴'} Position closed: {pos.ticker}",
        body=f"Exited {pos.ticker} @ ${pos.exit_price:.2f}  |  P&L: {sign}{pos.pnl_pct:.1f}%",
    )
    return {"status": "closed", "pnl_pct": pos.pnl_pct}


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
