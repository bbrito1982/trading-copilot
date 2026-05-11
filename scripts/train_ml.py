#!/usr/bin/env python
"""Train the ML signal classifier from cached historical OHLCV data.

Examples
--------
    uv run python scripts/train_ml.py --watchlist
    uv run python scripts/train_ml.py --tickers AAPL MSFT NVDA --from 2018-01-01
    uv run python scripts/train_ml.py --watchlist --forward-days 5 --min-signals 1
"""
from __future__ import annotations

import logging
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from rich.console import Console

from trading_copilot.config import load_config
from trading_copilot.data.tiingo import get_ohlcv_cached_only
from trading_copilot.signals.ml.trainer import train, DEFAULT_MODEL_PATH
from trading_copilot.signals.ml.predictor import invalidate_cache

console = Console()


def _parse_date(ctx, param, value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise click.BadParameter(f"Expected YYYY-MM-DD, got '{value}'")


@click.command()
@click.option("--from", "start", default="2018-01-01", callback=_parse_date, metavar="YYYY-MM-DD",
              help="Start of training window (default: 2018-01-01).")
@click.option("--to", "end", default=None,
              callback=lambda c, p, v: _parse_date(c, p, v) if v else date.today(),
              metavar="YYYY-MM-DD", help="End of training window (default: today).")
@click.option("--tickers", "-t", multiple=True, metavar="TICKER",
              help="Tickers to train on (repeatable). Overrides --watchlist.")
@click.option("--watchlist", is_flag=True, default=False,
              help="Use watchlist from config.yaml.")
@click.option("--forward-days", default=10, show_default=True,
              help="Bars ahead to measure trade outcome.")
@click.option("--min-signals", default=1, show_default=True,
              help="Minimum signals required to include a bar as a training sample.")
@click.option("--model-path", default=DEFAULT_MODEL_PATH, show_default=True,
              help="Where to save the trained model.")
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(
    start: date,
    end: date,
    tickers: tuple[str, ...],
    watchlist: bool,
    forward_days: int,
    min_signals: int,
    model_path: str,
    verbose: bool,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cfg_all = load_config()
    signal_cfg = cfg_all.get("signals", {})

    if tickers:
        ticker_list = [t.upper() for t in tickers]
    elif watchlist:
        ticker_list = [t.upper() for t in cfg_all.get("watchlist", [])]
    else:
        console.print("[red]Specify --tickers or --watchlist.[/red]")
        raise SystemExit(1)

    console.print(f"\n[bold]ML trainer[/bold] — tickers={ticker_list}  forward_days={forward_days}\n")

    console.print(f"Loading cached OHLCV  {start} → {end} …")
    ohlcv_data: dict = {}
    for ticker in ticker_list:
        df = get_ohlcv_cached_only(ticker, start=start, end=end)
        if df.empty:
            console.print(f"  [yellow]⚠ {ticker}: no cached data[/yellow]")
        else:
            ohlcv_data[ticker] = df
            console.print(f"  {ticker}: {len(df)} bars")

    if not ohlcv_data:
        console.print("[red]No data. Run backfill_prices.py first.[/red]")
        raise SystemExit(1)

    console.print("\nBuilding dataset and training …\n")
    try:
        pipeline, cv_scores = train(
            ohlcv_data,
            signal_cfg,
            forward_days=forward_days,
            min_signals=min_signals,
            model_path=model_path,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1)

    invalidate_cache()

    console.print(f"[green]✓ Model saved → {model_path}[/green]")
    console.print(f"  CV ROC-AUC : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    console.print(
        "\nThe scorer will now use ML conviction automatically.\n"
        "Re-run the backtest to compare:\n"
        "  uv run python scripts/run_backtest.py --watchlist --from 2020-01-01\n"
    )


if __name__ == "__main__":
    main()
