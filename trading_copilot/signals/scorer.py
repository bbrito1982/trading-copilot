"""Combine signals into a conviction score and opportunity summary.

Conviction is computed in three layers and blended:
  1. Rule-based: weighted sum of signal strengths (SIGNAL_WEIGHTS).
  2. ML model (preferred over rule-based): P(profitable) from the trained
     GradientBoosting classifier. Used when model file exists.
  3. Sentiment (optional): macro keyword score from recent headlines.
     Passed in as ``sentiment_score`` (-1 to +1). Adjusts final conviction
     up or down by up to SENTIMENT_WEIGHT, and can flip an opportunity to None
     if sentiment strongly contradicts the technical direction.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from trading_copilot.signals.rules import Signal, compute_indicators, detect_signals

logger = logging.getLogger(__name__)

# Sentiment blending: sentiment adjusts conviction by up to this fraction.
# e.g. SENTIMENT_WEIGHT=0.2 means a +1.0 sentiment score adds 0.2 to conviction.
SENTIMENT_WEIGHT = 0.20
# Sentiment veto: if sentiment score contradicts direction beyond this threshold,
# suppress the opportunity entirely.
SENTIMENT_VETO_THRESHOLD = 0.4

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
    sentiment_score: float | None = None      # -1 to +1 if available
    sentiment_themes: list[str] = field(default_factory=list)

    @property
    def signal_types(self) -> list[str]:
        return [s.signal_type for s in self.signals]


def score_ticker(
    ticker: str,
    df: pd.DataFrame,
    cfg: dict,
    swing_cfg: dict | None = None,
    model_path: str | None = None,
    sentiment_score: float | None = None,
    sentiment_themes: list[str] | None = None,
) -> Opportunity | None:
    """Score a ticker and return an Opportunity if conviction is high enough.

    Parameters
    ----------
    sentiment_score:
        Optional pre-computed sentiment score in [-1, +1].  Positive means
        bullish, negative means bearish.  Passed in from the scheduler after
        fetching and tagging recent headlines.
    sentiment_themes:
        List of matched macro theme names, for display in notifications.
    """
    signals = detect_signals(df, cfg)
    if not signals:
        return None

    swing = swing_cfg or {}
    stop_pct = swing.get("stop_loss_pct", 0.05)
    target_pct = swing.get("take_profit_pct", 0.08)

    # Determine direction from rule-based net weight (used even in ML mode)
    raw_score = sum(SIGNAL_WEIGHTS.get(s.signal_type, 0) * s.strength for s in signals)
    direction = "buy" if raw_score > 0 else "sell"

    buy_signals = [s for s in signals if s.direction == "buy"]
    sell_signals = [s for s in signals if s.direction == "sell"]
    aligned = buy_signals if direction == "buy" else sell_signals
    if len(aligned) < 2:
        return None

    # --- Layer 1: rule-based conviction (always computed) ---
    rule_conviction = min(1.0, abs(raw_score))

    # --- Layer 2: ML conviction ---
    from trading_copilot.signals.ml.predictor import predict_conviction, model_exists
    kwargs = {"model_path": model_path} if model_path else {}
    ml_conviction = predict_conviction(df, signals, cfg, **kwargs) if (
        model_path is not None or model_exists()
    ) else None

    # --- Sentiment veto (before ensemble — hard gate regardless of other signals) ---
    if sentiment_score is not None:
        direction_sign = 1.0 if direction == "buy" else -1.0
        aligned_sentiment = sentiment_score * direction_sign
        if aligned_sentiment < -SENTIMENT_VETO_THRESHOLD:
            logger.info(
                "%s: opportunity vetoed by sentiment (score=%.2f contradicts %s)",
                ticker, sentiment_score, direction,
            )
            return None

    # --- Layer 3: ensemble meta-model (preferred when available) ---
    from trading_copilot.signals.ensemble import predict_ensemble, ensemble_exists
    if ensemble_exists():
        ensemble_conviction = predict_ensemble(
            rule_conviction, ml_conviction, sentiment_score, direction
        )
    else:
        ensemble_conviction = None

    # Priority: ensemble > ML > rule-based; sentiment blend applied when no ensemble
    if ensemble_conviction is not None:
        conviction = ensemble_conviction
        logger.debug(
            "%s: ensemble conviction=%.3f (rule=%.3f ml=%s sentiment=%s)",
            ticker, conviction, rule_conviction,
            f"{ml_conviction:.3f}" if ml_conviction is not None else "n/a",
            f"{sentiment_score:.3f}" if sentiment_score is not None else "n/a",
        )
    elif ml_conviction is not None:
        conviction = ml_conviction
        if sentiment_score is not None:
            direction_sign = 1.0 if direction == "buy" else -1.0
            aligned_sentiment = sentiment_score * direction_sign
            conviction = min(1.0, max(0.0, conviction + aligned_sentiment * SENTIMENT_WEIGHT))
        logger.debug("%s: ML conviction=%.3f (rule=%.3f)", ticker, conviction, rule_conviction)
    else:
        conviction = rule_conviction
        if sentiment_score is not None:
            direction_sign = 1.0 if direction == "buy" else -1.0
            aligned_sentiment = sentiment_score * direction_sign
            conviction = min(1.0, max(0.0, conviction + aligned_sentiment * SENTIMENT_WEIGHT))
        logger.debug("%s: rule conviction=%.3f", ticker, conviction)

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
        sentiment_score=round(sentiment_score, 4) if sentiment_score is not None else None,
        sentiment_themes=sentiment_themes or [],
    )
