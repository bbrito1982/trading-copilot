#!/usr/bin/env python
"""CLI entry point for the Phase 2 backtest engine.

Examples
--------
    uv run python scripts/run_backtest.py --from 2020-01-01
    uv run python scripts/run_backtest.py --tickers AAPL MSFT --from 2022-01-01 --to 2024-01-01
    uv run python scripts/run_backtest.py --watchlist --from 2020-01-01 --threshold 0.55
    uv run python scripts/run_backtest.py --watchlist --from 2020-01-01 --trades
"""
from __future__ import annotations

import logging
import sys
from datetime import date, datetime
from pathlib import Path

# Ensure project root is on the path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from rich.console import Console
from rich.table import Table

from trading_copilot.config import load_config, settings
from trading_copilot.data.tiingo import get_ohlcv_cached_only
from trading_copilot.backtest.engine import run_backtest

console = Console()


def _parse_date(ctx, param, value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise click.BadParameter(f"Expected YYYY-MM-DD, got '{value}'")


@click.command()
@click.option("--from", "start", default="2020-01-01", callback=_parse_date, metavar="YYYY-MM-DD",
              help="Backtest start date (default: 2020-01-01).")
@click.option("--to", "end", default=None, callback=lambda c, p, v: _parse_date(c, p, v) if v else date.today(),
              metavar="YYYY-MM-DD", help="Backtest end date (default: today).")
@click.option("--tickers", "-t", multiple=True, metavar="TICKER",
              help="Tickers to backtest (repeatable: -t AAPL -t MSFT). Overrides --watchlist.")
@click.option("--watchlist", is_flag=True, default=False,
              help="Use watchlist from config.yaml.")
@click.option("--threshold", default=None, type=float,
              help="Conviction threshold (default: config.yaml conviction_threshold).")
@click.option("--capital", default=10_000.0, show_default=True,
              help="Notional starting capital per ticker.")
@click.option("--trades", "show_trades", is_flag=True, default=False,
              help="Print individual trade log after summary.")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Enable DEBUG logging.")
def main(
    start: date,
    end: date,
    tickers: tuple[str, ...],
    watchlist: bool,
    threshold: float | None,
    capital: float,
    show_trades: bool,
    verbose: bool,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cfg_all = load_config()
    signal_cfg = cfg_all.get("signals", {})
    swing_cfg = cfg_all.get("swing", {})
    conviction_threshold = threshold if threshold is not None else cfg_all.get("conviction_threshold", 0.6)

    if tickers:
        ticker_list = [t.upper() for t in tickers]
    elif watchlist:
        ticker_list = [t.upper() for t in cfg_all.get("watchlist", [])]
    else:
        console.print("[red]Specify --tickers or --watchlist.[/red]")
        raise SystemExit(1)

    console.print(
        f"\n[bold]Backtest[/bold] {start} → {end}  "
        f"tickers={ticker_list}  threshold={conviction_threshold}\n"
    )

    # Load OHLCV from local DuckDB cache (never hits Tiingo during backtest)
    console.print("Loading cached OHLCV …")
    ohlcv_data: dict = {}
    for ticker in ticker_list:
        df = get_ohlcv_cached_only(ticker, start=start, end=end)
        if df.empty:
            console.print(f"  [yellow]⚠ {ticker}: no cached data (run backfill_prices.py first)[/yellow]")
        else:
            ohlcv_data[ticker] = df
            console.print(f"  {ticker}: {len(df)} bars")

    if not ohlcv_data:
        console.print("[red]No data loaded. Aborting.[/red]")
        raise SystemExit(1)

    console.print("\nRunning simulation …\n")

    try:
        result = run_backtest(
            tickers=list(ohlcv_data.keys()),
            ohlcv_data=ohlcv_data,
            cfg=signal_cfg,
            swing_cfg=swing_cfg,
            conviction_threshold=conviction_threshold,
            starting_capital=capital,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1)

    # --- Per-ticker results table ---
    table = Table(title="Per-ticker results", show_lines=False)
    table.add_column("Ticker", style="cyan")
    table.add_column("Trades", justify="right")
    table.add_column("Win %", justify="right")
    table.add_column("Avg win", justify="right")
    table.add_column("Avg loss", justify="right")
    table.add_column("Expectancy", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("CAGR", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("Total rtn", justify="right")

    for ticker, tr in sorted(result.ticker_results.items()):
        m = tr.metrics
        table.add_row(
            ticker,
            str(m.total_trades),
            f"{m.win_rate:.0%}",
            f"{m.avg_win_pct:.2%}",
            f"{m.avg_loss_pct:.2%}",
            f"{m.expectancy:.2%}",
            f"{m.sharpe:.2f}",
            f"{m.cagr:.2%}",
            f"{m.max_drawdown:.2%}",
            f"{m.total_return:.2%}",
        )

    console.print(table)

    # --- Portfolio summary ---
    m = result.metrics
    console.print("\n[bold]Portfolio summary[/bold]")
    console.print(str(m))
    console.print()

    # --- Trade log ---
    if show_trades and result.trades:
        trade_table = Table(title="Trade log", show_lines=False)
        trade_table.add_column("Ticker", style="cyan")
        trade_table.add_column("Dir")
        trade_table.add_column("Entry date")
        trade_table.add_column("Entry $", justify="right")
        trade_table.add_column("Exit date")
        trade_table.add_column("Exit $", justify="right")
        trade_table.add_column("Reason")
        trade_table.add_column("P&L %", justify="right")

        for t in sorted(result.trades, key=lambda x: x["entry_date"]):
            pnl = t.get("pnl_pct", 0)
            color = "green" if pnl > 0 else "red"
            trade_table.add_row(
                t["ticker"],
                t["direction"],
                str(t["entry_date"]),
                f"{t['entry_price']:.2f}",
                str(t["exit_date"]),
                f"{t['exit_price']:.2f}",
                t["exit_reason"],
                f"[{color}]{pnl:.2%}[/{color}]",
            )

        console.print(trade_table)


if __name__ == "__main__":
    main()
