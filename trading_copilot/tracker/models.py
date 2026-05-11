"""SQLModel database models for opportunities and positions."""
from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Optional

from sqlmodel import Field, SQLModel, create_engine, Session, select
from sqlalchemy import JSON, Column

from trading_copilot.config import settings


def get_engine():
    db_url = f"sqlite:///{settings.sqlite_path}"
    from pathlib import Path
    Path(settings.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(db_url)


def create_tables():
    SQLModel.metadata.create_all(get_engine())


class Opportunity(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    ticker: str
    date: date
    signal_type: str          # comma-joined list of signal types
    direction: str            # 'buy' | 'sell'
    conviction: float
    entry_price: float
    stop_loss: float
    target: float
    indicators: Optional[str] = None   # JSON string
    news_context: Optional[str] = None # JSON string
    action: Optional[str] = None       # 'entered' | 'skipped' | None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Position(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    opportunity_id: str = Field(foreign_key="opportunity.id")
    ticker: str
    entry_price: float
    entry_date: date
    shares: Optional[int] = None
    status: str = "open"             # 'open' | 'closed'
    exit_price: Optional[float] = None
    exit_date: Optional[date] = None
    exit_reason: Optional[str] = None  # 'target' | 'stop' | 'signal_reversal' | 'manual'
    pnl_pct: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
