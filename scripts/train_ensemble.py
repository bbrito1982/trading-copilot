#!/usr/bin/env python
"""Train the Phase 4 ensemble meta-model.

Examples
--------
    uv run python scripts/train_ensemble.py --watchlist --from 2018-01-01
    uv run python scripts/train_ensemble.py --watchlist --from 2018-01-01 --to 2024-12-31
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
from trading_copilot.signals.ensemble import train_ensemble, DEFAULT_ENSEMBLE_PATH, invalidate_cache

console = Console()


def _parse_date(ctx, param, value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise click.BadParameter(f"Expected YYYY-MM-DD, got '{value}'")


@click.command()
@click.option("--from", "start", default="2018-01-01", callback=_parse_date, metavar="YYYY-MM-DD")
@click.option("--to", "end", default=None,
              callback=lambda c, p, v: _parse_date(c, p, v) if v else date.today(),
              metavar="YYYY-MM-DD")
@click.option("--tickers", "-t", multiple=True, metavar="TICKER")
@click.option("--watchlist", is_flag=True, default=False)
@click.option("--forward-days", default=10, show_default=True)
@click.option("--ensemble-path", default=DEFAULT_ENSEMBLE_PATH, show_default=True)
@click.option("--sentiment-cache", default=None, metavar="PATH",
              help="Path to fnspid_sentiment.parquet produced by prepare_fnspid.py.")
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(start, end, tickers, watchlist, forward_days, ensemble_path, sentiment_cache, verbose):
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

    console.print(f"\n[bold]Ensemble trainer[/bold] — {start} → {end}\n")

    ohlcv_data = {}
    for ticker in ticker_list:
        df = get_ohlcv_cached_only(ticker, start=start, end=end)
        if not df.empty:
            ohlcv_data[ticker] = df
            console.print(f"  {ticker}: {len(df)} bars")
        else:
            console.print(f"  [yellow]⚠ {ticker}: no cached data[/yellow]")

    if not ohlcv_data:
        console.print("[red]No data. Run backfill_prices.py first.[/red]")
        raise SystemExit(1)

    if sentiment_cache:
        console.print(f"  Sentiment cache  : {sentiment_cache}")
    console.print("\nTraining ensemble …\n")
    try:
        pipeline, cv_scores = train_ensemble(
            ohlcv_data, signal_cfg,
            forward_days=forward_days,
            ensemble_path=ensemble_path,
            sentiment_cache_path=sentiment_cache,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1)

    invalidate_cache()

    # Show learned feature weights
    coef = pipeline.named_steps["clf"].coef_[0]
    from trading_copilot.signals.ensemble import ENSEMBLE_FEATURE_NAMES
    console.print(f"[green]✓ Ensemble saved → {ensemble_path}[/green]")
    console.print(f"  CV ROC-AUC : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}\n")
    console.print("  [bold]Learned feature weights:[/bold]")
    for name, w in zip(ENSEMBLE_FEATURE_NAMES, coef):
        bar = "█" * int(abs(w) * 20)
        sign = "+" if w >= 0 else "-"
        console.print(f"    {name:<24s} {sign}{abs(w):.4f}  {bar}")
    console.print()
    console.print(
        "Re-run the backtest to compare:\n"
        "  uv run python scripts/run_backtest.py --watchlist --from 2025-01-01\n"
    )


if __name__ == "__main__":
    main()
