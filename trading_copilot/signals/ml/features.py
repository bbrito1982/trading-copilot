"""Feature extraction for the ML signal layer.

A feature vector is computed from the indicator-enriched DataFrame at a given
bar. It captures the technical state of the market at signal-fire time so that
the classifier can learn which configurations actually lead to profitable moves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from trading_copilot.signals.rules import Signal, SignalType, compute_indicators

# Ordered list — must stay stable between train and predict
FEATURE_NAMES: list[str] = [
    # Momentum
    "rsi",
    "rsi_dist_oversold",    # distance from 30 (positive = below 30)
    "rsi_dist_overbought",  # distance from 70 (positive = above 70)
    # MACD
    "macd_hist_norm",       # macd_hist / adj_close
    "macd_cross_norm",      # (macd - macd_signal) / adj_close
    # Trend
    "ma_spread_norm",       # (ma_fast - ma_slow) / adj_close
    # Volume
    "vol_ratio",
    # Short-term price momentum
    "ret_5d",               # 5-bar return
    "ret_20d",              # 20-bar return
    # Signal presence flags (one per signal type)
    "sig_rsi_oversold",
    "sig_rsi_overbought",
    "sig_macd_bullish_cross",
    "sig_macd_bearish_cross",
    "sig_ma_golden_cross",
    "sig_ma_death_cross",
    "sig_volume_spike_up",
    "sig_volume_spike_down",
    # Signal aggregate stats
    "n_buy_signals",
    "n_sell_signals",
    "buy_strength_sum",
    "sell_strength_sum",
]

_SIGNAL_FLAG_NAMES: list[SignalType] = [
    "rsi_oversold",
    "rsi_overbought",
    "macd_bullish_cross",
    "macd_bearish_cross",
    "ma_golden_cross",
    "ma_death_cross",
    "volume_spike_up",
    "volume_spike_down",
]


def extract_features(df: pd.DataFrame, signals: list[Signal], cfg: dict) -> np.ndarray:
    """Return a 1-D feature vector for the last bar of *df*.

    Parameters
    ----------
    df:
        OHLCV DataFrame ending at the bar of interest, already long enough
        to compute all indicators (≥60 bars).
    signals:
        Signals detected on the last bar by ``detect_signals``.
    cfg:
        Signal config dict (``config.yaml`` ``signals`` section).
    """
    df_ind = compute_indicators(df, cfg)
    row = df_ind.iloc[-1]
    close = float(row["adj_close"])

    ma_fast_key = f"ma{cfg.get('ma_crossover', {}).get('fast', 20)}"
    ma_slow_key = f"ma{cfg.get('ma_crossover', {}).get('slow', 50)}"

    rsi = float(row.get("rsi") or 50.0)
    macd_hist = float(row.get("macd_hist") or 0.0)
    macd = float(row.get("macd") or 0.0)
    macd_signal_val = float(row.get("macd_signal") or 0.0)
    ma_fast = float(row.get(ma_fast_key) or close)
    ma_slow = float(row.get(ma_slow_key) or close)
    vol_ratio = float(row.get("vol_ratio") or 1.0)

    # Price momentum — look back in the full df
    ret_5d = float((df_ind["adj_close"].iloc[-1] / df_ind["adj_close"].iloc[-6] - 1)
                   if len(df_ind) >= 6 else 0.0)
    ret_20d = float((df_ind["adj_close"].iloc[-1] / df_ind["adj_close"].iloc[-21] - 1)
                    if len(df_ind) >= 21 else 0.0)

    sig_flags = {s.signal_type: s.strength for s in signals}
    buy_signals = [s for s in signals if s.direction == "buy"]
    sell_signals = [s for s in signals if s.direction == "sell"]

    vec = [
        rsi / 100.0,
        max(0.0, (30 - rsi) / 30),
        max(0.0, (rsi - 70) / 30),
        macd_hist / (close + 1e-9),
        (macd - macd_signal_val) / (close + 1e-9),
        (ma_fast - ma_slow) / (close + 1e-9),
        min(vol_ratio, 10.0) / 10.0,
        ret_5d,
        ret_20d,
        # signal flags
        *(sig_flags.get(st, 0.0) for st in _SIGNAL_FLAG_NAMES),
        # aggregates
        len(buy_signals),
        len(sell_signals),
        sum(s.strength for s in buy_signals),
        sum(s.strength for s in sell_signals),
    ]
    return np.array(vec, dtype=np.float32)
