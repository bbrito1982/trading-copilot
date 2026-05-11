"""Trade journal: query outcomes for ML training data preparation."""
from __future__ import annotations

import json
from datetime import date

import pandas as pd
from sqlmodel import Session, select

from trading_copilot.tracker.models import Opportunity, Position, get_engine


def get_labeled_outcomes(db_engine=None) -> pd.DataFrame:
    """Return all closed positions joined with their opportunity signals.

    Each row is a training sample: features at signal time + outcome label.
    """
    engine = db_engine or get_engine()
    with Session(engine) as session:
        positions = session.exec(
            select(Position).where(Position.status == "closed")
        ).all()

        rows = []
        for pos in positions:
            opp = session.get(Opportunity, pos.opportunity_id)
            if not opp:
                continue
            indicators = json.loads(opp.indicators or "{}")
            rows.append({
                "ticker": pos.ticker,
                "signal_date": opp.date,
                "signal_types": opp.signal_type,
                "direction": opp.direction,
                "conviction": opp.conviction,
                "entry_price": pos.entry_price,
                "exit_price": pos.exit_price,
                "pnl_pct": pos.pnl_pct,
                "exit_reason": pos.exit_reason,
                "profitable": (pos.pnl_pct or 0) > 0,
                **indicators,
            })

    return pd.DataFrame(rows)
