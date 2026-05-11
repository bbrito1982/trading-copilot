"""Walk-forward backtest engine.

Replays cached OHLCV day-by-day, applies the live signal pipeline as if running
on each bar's close, simulates entries/exits, and returns an equity curve + trade log.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from trading_copilot.signals.scorer import score_ticker
from trading_copilot.backtest.metrics import BacktestMetrics, compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    ticker_results: dict[str, "TickerResult"]
    equity_curve: pd.Series          # combined portfolio equity, indexed by date
    trades: list[dict]               # all closed trade records
    metrics: BacktestMetrics


@dataclass
class TickerResult:
    ticker: str
    equity_curve: pd.Series
    trades: list[dict]
    metrics: BacktestMetrics


@dataclass
class _OpenPosition:
    ticker: str
    direction: str      # 'buy' | 'sell'
    entry_price: float
    entry_date: date
    stop_loss: float
    target: float
    hold_days: int      # max days to hold


def _lookup_sentiment(
    sentiment_cache: "pd.DataFrame | None",
    ticker: str,
    bar_date: date,
    window_days: int = 3,
) -> "float | None":
    """Return the most recent sentiment score for ticker on or before bar_date.

    Looks back up to window_days to handle weekends / sparse news coverage.
    """
    if sentiment_cache is None:
        return None
    ts = pd.Timestamp(bar_date)
    mask = (
        (sentiment_cache["ticker"] == ticker)
        & (sentiment_cache["date"] <= ts)
        & (sentiment_cache["date"] >= ts - pd.Timedelta(days=window_days))
    )
    rows = sentiment_cache.loc[mask]
    if rows.empty:
        return None
    return float(rows.sort_values("date").iloc[-1]["sentiment_score"])


def _simulate_ticker(
    ticker: str,
    df: pd.DataFrame,
    cfg: dict,
    swing_cfg: dict,
    conviction_threshold: float,
    starting_capital: float,
    sentiment_cache: "pd.DataFrame | None" = None,
) -> TickerResult:
    """Simulate signal detection and position management for one ticker."""
    hold_days = swing_cfg.get("hold_days", 10)
    equity: dict[date, float] = {}
    trades: list[dict] = []
    capital = starting_capital
    open_pos: _OpenPosition | None = None
    dates = sorted(df["date"].unique())

    for i, bar_date in enumerate(dates):
        if i < 60:
            equity[bar_date] = capital
            continue

        # Slice everything up to and including this bar (no lookahead)
        window = df[df["date"] <= bar_date].copy()

        # --- Check existing position exit ---
        if open_pos is not None:
            today_row = window[window["date"] == bar_date]
            if today_row.empty:
                equity[bar_date] = capital
                continue

            price = float(today_row["adj_close"].iloc[0])
            days_held = sum(
                1 for d in dates[: i + 1] if d > open_pos.entry_date and d <= bar_date
            )

            exit_reason = None
            if open_pos.direction == "buy":
                if price <= open_pos.stop_loss:
                    exit_reason = "stop"
                elif price >= open_pos.target:
                    exit_reason = "target"
            else:  # short / sell signal
                if price >= open_pos.stop_loss:
                    exit_reason = "stop"
                elif price <= open_pos.target:
                    exit_reason = "target"

            if exit_reason is None and days_held >= hold_days:
                exit_reason = "timeout"

            if exit_reason:
                if open_pos.direction == "buy":
                    pnl_pct = (price - open_pos.entry_price) / open_pos.entry_price
                else:
                    pnl_pct = (open_pos.entry_price - price) / open_pos.entry_price

                capital *= 1 + pnl_pct
                trades.append({
                    "ticker": ticker,
                    "direction": open_pos.direction,
                    "entry_date": open_pos.entry_date,
                    "entry_price": open_pos.entry_price,
                    "exit_date": bar_date,
                    "exit_price": price,
                    "exit_reason": exit_reason,
                    "pnl_pct": pnl_pct,
                })
                logger.debug(
                    "%s: closed %s @ %.2f (%s) pnl=%.2f%%",
                    ticker, open_pos.direction, price, exit_reason, pnl_pct * 100,
                )
                open_pos = None

        # --- Look for new entry ---
        if open_pos is None:
            sentiment_score = _lookup_sentiment(sentiment_cache, ticker, bar_date)
            opp = score_ticker(ticker, window, cfg, swing_cfg, sentiment_score=sentiment_score)
            if opp is not None and opp.conviction >= conviction_threshold:
                open_pos = _OpenPosition(
                    ticker=ticker,
                    direction=opp.direction,
                    entry_price=opp.entry_price,
                    entry_date=bar_date,
                    stop_loss=opp.stop_loss,
                    target=opp.target,
                    hold_days=hold_days,
                )
                logger.debug(
                    "%s: opened %s @ %.2f (conviction=%.2f)",
                    ticker, opp.direction, opp.entry_price, opp.conviction,
                )

        equity[bar_date] = capital

    # Force-close any open position at last bar
    if open_pos is not None and dates:
        last_date = dates[-1]
        last_row = df[df["date"] == last_date]
        if not last_row.empty:
            price = float(last_row["adj_close"].iloc[0])
            if open_pos.direction == "buy":
                pnl_pct = (price - open_pos.entry_price) / open_pos.entry_price
            else:
                pnl_pct = (open_pos.entry_price - price) / open_pos.entry_price
            capital *= 1 + pnl_pct
            trades.append({
                "ticker": ticker,
                "direction": open_pos.direction,
                "entry_date": open_pos.entry_date,
                "entry_price": open_pos.entry_price,
                "exit_date": last_date,
                "exit_price": price,
                "exit_reason": "end_of_data",
                "pnl_pct": pnl_pct,
            })
            equity[last_date] = capital

    eq_series = pd.Series(equity, name=ticker)
    eq_series.index = pd.to_datetime(eq_series.index)
    eq_series = eq_series.sort_index()

    metrics = compute_metrics(eq_series, trades)
    return TickerResult(ticker=ticker, equity_curve=eq_series, trades=trades, metrics=metrics)


def run_backtest(
    tickers: list[str],
    ohlcv_data: dict[str, pd.DataFrame],
    cfg: dict,
    swing_cfg: dict | None = None,
    conviction_threshold: float = 0.6,
    starting_capital: float = 10_000.0,
    sentiment_cache: "pd.DataFrame | None" = None,
) -> BacktestResult:
    """Run a walk-forward backtest over pre-loaded OHLCV data.

    Parameters
    ----------
    tickers:
        List of tickers to backtest.
    ohlcv_data:
        Dict mapping ticker → DataFrame (same schema as DuckDB ohlcv table).
    cfg:
        Signal config dict (same as config.yaml ``signals`` section).
    swing_cfg:
        Swing trade config dict (stop_loss_pct, take_profit_pct, hold_days).
    conviction_threshold:
        Minimum conviction to open a position (0–1).
    starting_capital:
        Notional capital per ticker (used only to compute equity curves).
    sentiment_cache:
        Optional DataFrame with columns (ticker, date, sentiment_score).
        When provided, each bar's entry signal is augmented with the most
        recent sentiment score from this cache (look-back window: 3 days).
    """
    swing = swing_cfg or {}
    ticker_results: dict[str, TickerResult] = {}

    for ticker in tickers:
        df = ohlcv_data.get(ticker)
        if df is None or df.empty:
            logger.warning("%s: no data, skipping", ticker)
            continue
        logger.info("Backtesting %s (%d bars) …", ticker, len(df))
        ticker_results[ticker] = _simulate_ticker(
            ticker, df, cfg, swing, conviction_threshold, starting_capital,
            sentiment_cache=sentiment_cache,
        )

    if not ticker_results:
        raise ValueError("No data available for any of the requested tickers.")

    # Combined portfolio equity: equally-weighted sum of all ticker equity curves
    all_equity = pd.concat(
        [r.equity_curve for r in ticker_results.values()], axis=1
    ).ffill()
    combined = all_equity.sum(axis=1)
    combined.name = "portfolio"

    all_trades = [t for r in ticker_results.values() for t in r.trades]
    portfolio_metrics = compute_metrics(combined, all_trades)

    return BacktestResult(
        ticker_results=ticker_results,
        equity_curve=combined,
        trades=all_trades,
        metrics=portfolio_metrics,
    )
