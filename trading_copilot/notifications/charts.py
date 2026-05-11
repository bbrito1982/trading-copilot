"""Chart generation using mplfinance."""
from __future__ import annotations

import io
from datetime import date, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

from trading_copilot.signals.rules import compute_indicators
from trading_copilot.signals.scorer import Opportunity


def _prep_df(df: pd.DataFrame, lookback_days: int = 90) -> pd.DataFrame:
    """Prepare OHLCV dataframe for mplfinance (indexed by date, correct columns)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.rename(columns={
        "adj_open": "Open",
        "adj_high": "High",
        "adj_low": "Low",
        "adj_close": "Close",
        "volume": "Volume",
    })
    # Fallback to unadjusted if adjusted columns missing
    for col, fallback in [("Open", "open"), ("High", "high"), ("Low", "low"), ("Close", "close")]:
        if col not in df.columns and fallback in df.columns:
            df[col] = df[fallback]

    cutoff = df.index[-1] - timedelta(days=lookback_days)
    return df[df.index >= cutoff][["Open", "High", "Low", "Close", "Volume"]]


def generate_chart(
    ticker: str,
    df: pd.DataFrame,
    opportunity: Opportunity | None = None,
    cfg: dict | None = None,
    lookback_days: int = 90,
    entry_date: date | None = None,
    entry_price: float | None = None,
) -> bytes:
    """Return PNG bytes for a candlestick chart with indicators and signal markers."""
    cfg = cfg or {}
    ohlcv = _prep_df(df, lookback_days)

    # Compute indicators on full df then slice
    full_df = df.copy()
    full_df["date"] = pd.to_datetime(full_df["date"])
    full_df = full_df.set_index("date").sort_index()
    ind_df = compute_indicators(
        full_df.rename(columns={"adj_close": "adj_close"}).assign(
            adj_close=full_df.get("adj_close", full_df.get("close"))
        ),
        cfg,
    )
    cutoff = ohlcv.index[0]
    ind_df = ind_df[ind_df.index >= cutoff]

    ma_fast = cfg.get("ma_crossover", {}).get("fast", 20)
    ma_slow = cfg.get("ma_crossover", {}).get("slow", 50)

    add_plots = []
    if f"ma{ma_fast}" in ind_df.columns:
        add_plots.append(mpf.make_addplot(
            ind_df[f"ma{ma_fast}"].reindex(ohlcv.index),
            color="#2196F3", width=1.2, label=f"MA{ma_fast}",
        ))
    if f"ma{ma_slow}" in ind_df.columns:
        add_plots.append(mpf.make_addplot(
            ind_df[f"ma{ma_slow}"].reindex(ohlcv.index),
            color="#FF9800", width=1.2, label=f"MA{ma_slow}",
        ))
    if "rsi" in ind_df.columns:
        add_plots.append(mpf.make_addplot(
            ind_df["rsi"].reindex(ohlcv.index),
            panel=2, color="#9C27B0", ylabel="RSI", ylim=(0, 100),
        ))

    # Horizontal lines for signal levels
    hlines = {}
    if opportunity:
        hlines = {
            "hlines": [opportunity.stop_loss, opportunity.target],
            "colors": ["#F44336", "#4CAF50"],
            "linestyle": "--",
            "linewidths": 1,
        }
    elif entry_price:
        hlines = {}

    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        gridstyle=":",
        gridcolor="#333333",
        facecolor="#1a1a2e",
        figcolor="#1a1a2e",
        edgecolor="#444444",
    )

    direction_emoji = ""
    if opportunity:
        direction_emoji = "▲ BUY" if opportunity.direction == "buy" else "▼ SELL"
        conviction_pct = f"{opportunity.conviction * 100:.0f}%"
        title = f"{ticker}  {direction_emoji}  conviction {conviction_pct}"
    else:
        title = f"{ticker}  position monitor"

    fig, axes = mpf.plot(
        ohlcv,
        type="candle",
        style=style,
        title=title,
        volume=True,
        addplot=add_plots if add_plots else None,
        panel_ratios=(4, 1, 2) if any(p.panel == 2 for p in add_plots) else (4, 1),
        figsize=(12, 8),
        tight_layout=True,
        returnfig=True,
        **hlines,
    )

    # Mark entry point if tracking a position
    if entry_date and entry_price:
        ax = axes[0]
        entry_dt = pd.Timestamp(entry_date)
        if entry_dt in ohlcv.index:
            idx = ohlcv.index.get_loc(entry_dt)
            ax.axvline(x=idx, color="#FFD700", linewidth=1.5, linestyle="--", alpha=0.8)
            ax.axhline(y=entry_price, color="#FFD700", linewidth=1, linestyle=":", alpha=0.6)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
