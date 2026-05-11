"""Position CRUD and exit condition checks."""
from __future__ import annotations

import json
import logging
from datetime import date

import pandas as pd
from sqlmodel import Session, select

from trading_copilot.tracker.models import Opportunity, Position, get_engine
from trading_copilot.signals.scorer import Opportunity as SignalOpportunity

logger = logging.getLogger(__name__)


def save_opportunity(opp: SignalOpportunity, db_engine=None) -> Opportunity:
    engine = db_engine or get_engine()
    record = Opportunity(
        ticker=opp.ticker,
        date=opp.date,
        signal_type=",".join(opp.signal_types),
        direction=opp.direction,
        conviction=opp.conviction,
        entry_price=opp.entry_price,
        stop_loss=opp.stop_loss,
        target=opp.target,
        indicators=json.dumps(opp.indicators),
    )
    with Session(engine) as session:
        session.add(record)
        session.commit()
        session.refresh(record)
    return record


def enter_position(
    opportunity_id: str,
    entry_price: float | None = None,
    shares: int | None = None,
    db_engine=None,
) -> Position | None:
    engine = db_engine or get_engine()
    with Session(engine) as session:
        opp = session.get(Opportunity, opportunity_id)
        if not opp:
            logger.error("Opportunity %s not found", opportunity_id)
            return None

        opp.action = "entered"
        price = entry_price or opp.entry_price
        pos = Position(
            opportunity_id=opportunity_id,
            ticker=opp.ticker,
            entry_price=price,
            entry_date=date.today(),
            shares=shares,
        )
        session.add(opp)
        session.add(pos)
        session.commit()
        session.refresh(pos)
        logger.info("Position entered: %s @ $%.2f", opp.ticker, price)
        return pos


def skip_opportunity(opportunity_id: str, db_engine=None) -> bool:
    engine = db_engine or get_engine()
    with Session(engine) as session:
        opp = session.get(Opportunity, opportunity_id)
        if not opp:
            return False
        opp.action = "skipped"
        session.add(opp)
        session.commit()
    return True


def close_position(
    position_id: str,
    exit_price: float,
    exit_reason: str = "manual",
    db_engine=None,
) -> Position | None:
    engine = db_engine or get_engine()
    with Session(engine) as session:
        pos = session.get(Position, position_id)
        if not pos or pos.status == "closed":
            return None
        pos.status = "closed"
        pos.exit_price = exit_price
        pos.exit_date = date.today()
        pos.exit_reason = exit_reason
        pos.pnl_pct = round((exit_price - pos.entry_price) / pos.entry_price * 100, 2)
        session.add(pos)
        session.commit()
        session.refresh(pos)
        logger.info("Position closed: %s @ $%.2f  P&L: %.1f%%", pos.ticker, exit_price, pos.pnl_pct)
        return pos


def get_open_positions(db_engine=None) -> list[Position]:
    engine = db_engine or get_engine()
    with Session(engine) as session:
        return session.exec(select(Position).where(Position.status == "open")).all()


ExitResult = tuple[str, str, float]  # (position_id, reason, current_price)


def check_exit_conditions(
    position: Position,
    df: pd.DataFrame,
    cfg: dict,
) -> ExitResult | None:
    """Return (position_id, exit_reason, current_price) if exit conditions met."""
    if df.empty:
        return None

    current_price = float(df.iloc[-1]["adj_close"])
    swing = cfg.get("swing", {})
    stop_pct = swing.get("stop_loss_pct", 0.05)
    target_pct = swing.get("take_profit_pct", 0.12)

    # Hard stop loss / take profit
    if current_price <= position.entry_price * (1 - stop_pct):
        return (position.id, "stop", current_price)
    if current_price >= position.entry_price * (1 + target_pct):
        return (position.id, "target", current_price)

    # Signal reversal: look for opposing signals
    from trading_copilot.signals.rules import detect_signals
    signals = detect_signals(df, cfg.get("signals", {}))
    sell_signals = [s for s in signals if s.direction == "sell"]
    if len(sell_signals) >= 2 and position.entry_price < current_price:
        return (position.id, "signal_reversal", current_price)

    return None
