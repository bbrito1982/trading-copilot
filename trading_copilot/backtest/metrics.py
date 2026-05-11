"""Performance metrics computed from a backtest equity curve and trade list."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    total_trades: int
    win_rate: float         # fraction of closed trades that were profitable
    avg_win_pct: float      # average gain on winning trades
    avg_loss_pct: float     # average loss on losing trades
    expectancy: float       # avg P&L per trade (weighted by win/loss rate)
    sharpe: float           # annualised Sharpe ratio (risk-free = 0)
    cagr: float             # compound annual growth rate
    max_drawdown: float     # peak-to-trough as a fraction (positive = loss)
    total_return: float     # total return as a fraction

    def __str__(self) -> str:  # noqa: D105
        lines = [
            f"  Total trades  : {self.total_trades}",
            f"  Win rate      : {self.win_rate:.1%}",
            f"  Avg win       : {self.avg_win_pct:.2%}",
            f"  Avg loss      : {self.avg_loss_pct:.2%}",
            f"  Expectancy    : {self.expectancy:.2%}",
            f"  Sharpe        : {self.sharpe:.2f}",
            f"  CAGR          : {self.cagr:.2%}",
            f"  Max drawdown  : {self.max_drawdown:.2%}",
            f"  Total return  : {self.total_return:.2%}",
        ]
        return "\n".join(lines)


def compute_metrics(equity_curve: pd.Series, trades: list[dict]) -> BacktestMetrics:
    """Derive metrics from an equity curve (indexed by date) and trade records."""
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0

    # CAGR
    n_days = (equity_curve.index[-1] - equity_curve.index[0]).days
    years = max(n_days / 365.25, 1 / 365.25)
    cagr = (1 + total_return) ** (1 / years) - 1

    # Daily returns for Sharpe
    daily_ret = equity_curve.pct_change().dropna()
    if daily_ret.std() > 0:
        sharpe = (daily_ret.mean() / daily_ret.std()) * math.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown
    roll_max = equity_curve.cummax()
    drawdowns = (equity_curve - roll_max) / roll_max
    max_drawdown = float(drawdowns.min())  # negative number; will negate for display

    closed = [t for t in trades if t.get("pnl_pct") is not None]
    total_trades = len(closed)

    if closed:
        pnls = [t["pnl_pct"] for t in closed]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / total_trades
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
    else:
        win_rate = avg_win = avg_loss = expectancy = 0.0

    return BacktestMetrics(
        total_trades=total_trades,
        win_rate=win_rate,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        expectancy=expectancy,
        sharpe=sharpe,
        cagr=cagr,
        max_drawdown=-max_drawdown,  # return as positive magnitude
        total_return=total_return,
    )
