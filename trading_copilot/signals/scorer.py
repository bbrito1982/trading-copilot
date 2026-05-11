"""Combine signals into a conviction score and opportunity summary."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from trading_copilot.signals.rules import Signal, compute_indicators, detect_signals

# Weight per signal type — buy signals positive, sell signals negative
SIGNAL_WEIGHTS: dict[str, float] = {
    "rsi_oversold": 0.25,
    "rsi_overbought": -0.25,
    "macd_bullish_cross": 0.30,
    "macd_bearish_cross": -0.30,
    "ma_golden_cross": 0.25,
    "ma_death_cross": -0.25,
    "volume_spike_up": 0.20,
    "volume_spike_down": -0.20,
}


@dataclass
class Opportunity:
    ticker: str
    date: date
    direction: str          # 'buy' | 'sell'
    conviction: float       # 0–1
    signals: list[Signal]
    entry_price: float
    stop_loss: float
    target: float
    indicators: dict = field(default_factory=dict)

    @property
    def signal_types(self) -> list[str]:
        return [s.signal_type for s in self.signals]


def score_ticker(
    ticker: str,
    df: pd.DataFrame,
    cfg: dict,
    swing_cfg: dict | None = None,
) -> Opportunity | None:
    """Score a ticker and return an Opportunity if conviction is high enough."""
    signals = detect_signals(df, cfg)
    if not signals:
        return None

    swing = swing_cfg or {}
    stop_pct = swing.get("stop_loss_pct", 0.05)
    target_pct = swing.get("take_profit_pct", 0.12)

    raw_score = sum(SIGNAL_WEIGHTS.get(s.signal_type, 0) * s.strength for s in signals)
    conviction = min(1.0, abs(raw_score))
    direction = "buy" if raw_score > 0 else "sell"

    # Only surface signals where multiple signals agree
    buy_signals = [s for s in signals if s.direction == "buy"]
    sell_signals = [s for s in signals if s.direction == "sell"]
    aligned = buy_signals if direction == "buy" else sell_signals
    if len(aligned) < 2:
        return None

    last = df.iloc[-1]
    entry = float(last["adj_close"])
    if direction == "buy":
        stop = round(entry * (1 - stop_pct), 2)
        target = round(entry * (1 + target_pct), 2)
    else:
        stop = round(entry * (1 + stop_pct), 2)
        target = round(entry * (1 - target_pct), 2)

    df_ind = compute_indicators(df, cfg)
    row = df_ind.iloc[-1]
    indicators = {
        "rsi": round(row.get("rsi", 0) or 0, 2),
        "macd": round(row.get("macd", 0) or 0, 4),
        "macd_signal": round(row.get("macd_signal", 0) or 0, 4),
        "vol_ratio": round(row.get("vol_ratio", 1) or 1, 2),
        "close": round(entry, 2),
    }

    return Opportunity(
        ticker=ticker,
        date=last["date"] if isinstance(last["date"], date) else last["date"],
        direction=direction,
        conviction=round(conviction, 3),
        signals=aligned,
        entry_price=entry,
        stop_loss=stop,
        target=target,
        indicators=indicators,
    )
