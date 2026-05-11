"""Tests for position tracker."""
import pytest
from datetime import date
from sqlmodel import create_engine, Session

from trading_copilot.tracker.models import SQLModel, Opportunity, Position
from trading_copilot.tracker.positions import (
    enter_position,
    skip_opportunity,
    close_position,
    get_open_positions,
)


@pytest.fixture
def engine(tmp_path):
    db_url = f"sqlite:///{tmp_path}/test.db"
    eng = create_engine(db_url)
    SQLModel.metadata.create_all(eng)
    return eng


@pytest.fixture
def sample_opportunity(engine):
    opp = Opportunity(
        ticker="AAPL",
        date=date.today(),
        signal_type="rsi_oversold,macd_bullish_cross",
        direction="buy",
        conviction=0.75,
        entry_price=150.0,
        stop_loss=142.5,
        target=168.0,
    )
    with Session(engine) as session:
        session.add(opp)
        session.commit()
        session.refresh(opp)
    return opp


def test_enter_position(engine, sample_opportunity):
    pos = enter_position(sample_opportunity.id, db_engine=engine)
    assert pos is not None
    assert pos.ticker == "AAPL"
    assert pos.status == "open"
    assert pos.entry_price == 150.0


def test_skip_opportunity(engine, sample_opportunity):
    ok = skip_opportunity(sample_opportunity.id, db_engine=engine)
    assert ok is True
    with Session(engine) as session:
        opp = session.get(Opportunity, sample_opportunity.id)
        assert opp.action == "skipped"


def test_close_position(engine, sample_opportunity):
    pos = enter_position(sample_opportunity.id, db_engine=engine)
    closed = close_position(pos.id, exit_price=165.0, exit_reason="target", db_engine=engine)
    assert closed.status == "closed"
    assert closed.exit_price == 165.0
    assert closed.pnl_pct == pytest.approx(10.0, abs=0.1)


def test_get_open_positions(engine, sample_opportunity):
    enter_position(sample_opportunity.id, db_engine=engine)
    positions = get_open_positions(db_engine=engine)
    assert len(positions) == 1
    assert positions[0].ticker == "AAPL"
