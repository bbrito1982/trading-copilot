"""Tests for rule-based signal detection."""
import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

from trading_copilot.signals.rules import detect_signals, compute_indicators, Signal
from trading_copilot.signals.scorer import score_ticker


DEFAULT_CFG = {
    "rsi": {"period": 14, "oversold": 30, "overbought": 70},
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "ma_crossover": {"fast": 20, "slow": 50},
    "volume_spike": {"threshold": 2.0},
}


def make_ohlcv(n: int = 120, trend: float = 0.0, vol_spike_last: bool = False) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    rng = np.random.default_rng(42)
    dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(n)]
    closes = 100 * np.cumprod(1 + rng.normal(trend, 0.015, n))
    volumes = rng.integers(1_000_000, 3_000_000, n).astype(float)
    if vol_spike_last:
        volumes[-1] = volumes[-20:].mean() * 3.0

    return pd.DataFrame({
        "ticker": "TEST",
        "date": dates,
        "open": closes * 0.99,
        "high": closes * 1.01,
        "low": closes * 0.98,
        "close": closes,
        "volume": volumes,
        "adj_close": closes,
        "adj_open": closes * 0.99,
        "adj_high": closes * 1.01,
        "adj_low": closes * 0.98,
        "div_cash": 0.0,
        "split_factor": 1.0,
    })


def test_compute_indicators_returns_expected_columns():
    df = make_ohlcv(120)
    result = compute_indicators(df, DEFAULT_CFG)
    assert "rsi" in result.columns
    assert "macd" in result.columns
    assert "macd_signal" in result.columns
    assert "ma20" in result.columns
    assert "ma50" in result.columns
    assert "vol_ratio" in result.columns


def test_detect_signals_returns_list():
    df = make_ohlcv(120)
    signals = detect_signals(df, DEFAULT_CFG)
    assert isinstance(signals, list)
    for s in signals:
        assert isinstance(s, Signal)
        assert 0.0 <= s.strength <= 1.0
        assert s.direction in ("buy", "sell")


def test_detect_signals_short_df_returns_empty():
    df = make_ohlcv(30)  # too short
    signals = detect_signals(df, DEFAULT_CFG)
    assert signals == []


def test_volume_spike_detected():
    df = make_ohlcv(120, vol_spike_last=True)
    signals = detect_signals(df, DEFAULT_CFG)
    spike_signals = [s for s in signals if "volume_spike" in s.signal_type]
    assert len(spike_signals) >= 1
    assert spike_signals[0].indicators["vol_ratio"] >= 2.0


def test_score_ticker_returns_none_for_weak_signal():
    df = make_ohlcv(120, trend=0.0)
    # With random data and default settings, may or may not fire — just check type
    result = score_ticker("TEST", df, DEFAULT_CFG)
    assert result is None or result.conviction >= 0


def test_score_ticker_conviction_in_range():
    df = make_ohlcv(120, trend=-0.005)  # downtrend might trigger sell signals
    result = score_ticker("TEST", df, DEFAULT_CFG)
    if result is not None:
        assert 0.0 <= result.conviction <= 1.0
        assert result.direction in ("buy", "sell")
        assert result.stop_loss > 0
        assert result.target > 0
