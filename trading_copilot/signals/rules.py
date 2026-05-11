"""Rule-based trading signals computed from OHLCV DataFrames."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
import ta as talib
import ta.momentum
import ta.trend
import ta.volume


SignalType = Literal[
    "rsi_oversold",
    "rsi_overbought",
    "macd_bullish_cross",
    "macd_bearish_cross",
    "ma_golden_cross",
    "ma_death_cross",
    "volume_spike_up",
    "volume_spike_down",
]


@dataclass
class Signal:
    signal_type: SignalType
    direction: Literal["buy", "sell"]
    strength: float  # 0–1, how strong/clean the signal is
    indicators: dict = field(default_factory=dict)


def compute_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add RSI, MACD, MAs, and volume ratio columns to df in-place."""
    rsi_period = cfg.get("rsi", {}).get("period", 14)
    macd_fast = cfg.get("macd", {}).get("fast", 12)
    macd_slow = cfg.get("macd", {}).get("slow", 26)
    macd_signal = cfg.get("macd", {}).get("signal", 9)
    ma_fast = cfg.get("ma_crossover", {}).get("fast", 20)
    ma_slow = cfg.get("ma_crossover", {}).get("slow", 50)

    df = df.copy()
    close = df["adj_close"]
    df["rsi"] = ta.momentum.RSIIndicator(close, window=rsi_period).rsi()

    macd_obj = ta.trend.MACD(close, window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_signal)
    df["macd"] = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_hist"] = macd_obj.macd_diff()

    df[f"ma{ma_fast}"] = ta.trend.SMAIndicator(close, window=ma_fast).sma_indicator()
    df[f"ma{ma_slow}"] = ta.trend.SMAIndicator(close, window=ma_slow).sma_indicator()
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    return df


def detect_signals(df: pd.DataFrame, cfg: dict) -> list[Signal]:
    """Return signals found on the last row of df (today's bar)."""
    if len(df) < 60:
        return []

    df = compute_indicators(df, cfg)
    row = df.iloc[-1]
    prev = df.iloc[-2]
    signals: list[Signal] = []

    rsi_cfg = cfg.get("rsi", {})
    oversold = rsi_cfg.get("oversold", 30)
    overbought = rsi_cfg.get("overbought", 70)
    ma_fast_key = f"ma{cfg.get('ma_crossover', {}).get('fast', 20)}"
    ma_slow_key = f"ma{cfg.get('ma_crossover', {}).get('slow', 50)}"
    vol_threshold = cfg.get("volume_spike", {}).get("threshold", 2.0)

    rsi = row.get("rsi")

    # RSI signals
    if pd.notna(rsi):
        if rsi < oversold:
            strength = min(1.0, (oversold - rsi) / oversold)
            signals.append(Signal("rsi_oversold", "buy", strength, {"rsi": round(rsi, 2)}))
        elif rsi > overbought:
            strength = min(1.0, (rsi - overbought) / (100 - overbought))
            signals.append(Signal("rsi_overbought", "sell", strength, {"rsi": round(rsi, 2)}))

    # MACD crossover
    if all(k in df.columns for k in ["macd", "macd_signal"]):
        curr_cross = row["macd"] - row["macd_signal"]
        prev_cross = prev["macd"] - prev["macd_signal"]
        if pd.notna(curr_cross) and pd.notna(prev_cross):
            if prev_cross < 0 < curr_cross:
                strength = min(1.0, abs(curr_cross) / (abs(row["adj_close"]) * 0.01 + 1e-9))
                signals.append(Signal("macd_bullish_cross", "buy", min(strength, 1.0), {
                    "macd": round(row["macd"], 4),
                    "macd_signal": round(row["macd_signal"], 4),
                }))
            elif prev_cross > 0 > curr_cross:
                strength = min(1.0, abs(curr_cross) / (abs(row["adj_close"]) * 0.01 + 1e-9))
                signals.append(Signal("macd_bearish_cross", "sell", min(strength, 1.0), {
                    "macd": round(row["macd"], 4),
                    "macd_signal": round(row["macd_signal"], 4),
                }))

    # MA crossover
    if all(k in df.columns for k in [ma_fast_key, ma_slow_key]):
        curr_spread = row[ma_fast_key] - row[ma_slow_key]
        prev_spread = prev[ma_fast_key] - prev[ma_slow_key]
        if pd.notna(curr_spread) and pd.notna(prev_spread):
            pct_spread = abs(curr_spread) / (row["adj_close"] + 1e-9)
            if prev_spread < 0 < curr_spread:
                signals.append(Signal("ma_golden_cross", "buy", min(pct_spread * 10, 1.0), {
                    ma_fast_key: round(row[ma_fast_key], 2),
                    ma_slow_key: round(row[ma_slow_key], 2),
                }))
            elif prev_spread > 0 > curr_spread:
                signals.append(Signal("ma_death_cross", "sell", min(pct_spread * 10, 1.0), {
                    ma_fast_key: round(row[ma_fast_key], 2),
                    ma_slow_key: round(row[ma_slow_key], 2),
                }))

    # Volume spike
    vol_ratio = row.get("vol_ratio")
    if pd.notna(vol_ratio) and vol_ratio >= vol_threshold:
        direction = "buy" if row["adj_close"] >= prev["adj_close"] else "sell"
        strength = min(1.0, (vol_ratio - vol_threshold) / vol_threshold)
        signals.append(Signal(
            f"volume_spike_{'up' if direction == 'buy' else 'down'}",
            direction,
            strength,
            {"vol_ratio": round(vol_ratio, 2)},
        ))

    return signals
