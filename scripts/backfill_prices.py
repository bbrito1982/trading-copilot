"""Seed DuckDB with historical OHLCV data from Tiingo.

Usage:
    python scripts/backfill_prices.py --tickers AAPL MSFT SPY --from 2015-01-01
    python scripts/backfill_prices.py --watchlist          # use config.yaml watchlist
    python scripts/backfill_prices.py --universe           # full universe
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from datetime import date
from rich.console import Console
from rich.progress import track

from trading_copilot.config import config
from trading_copilot.data.tiingo import get_ohlcv
from trading_copilot.data.universe import FULL_UNIVERSE

console = Console()


@click.command()
@click.option("--tickers", multiple=True, help="Specific tickers to backfill (repeat or space-separate: --tickers AAPL MSFT)")
@click.option("--watchlist", is_flag=True, help="Use watchlist from config.yaml")
@click.option("--universe", is_flag=True, help="Backfill full universe")
@click.option("--from", "from_date", default="2015-01-01", help="Start date (YYYY-MM-DD)")
@click.option("--to", "to_date", default=None, help="End date (YYYY-MM-DD, default: today)")
def main(tickers, watchlist, universe, from_date, to_date):
    start = date.fromisoformat(from_date)
    end = date.fromisoformat(to_date) if to_date else date.today()

    if universe:
        ticker_list = FULL_UNIVERSE
    elif watchlist:
        ticker_list = config.get("watchlist", [])
    elif tickers:
        ticker_list = list(tickers)
    else:
        console.print("[red]Specify --tickers, --watchlist, or --universe[/red]")
        raise SystemExit(1)

    console.print(f"Backfilling [bold]{len(ticker_list)}[/bold] tickers from {start} to {end}")

    ok, failed = 0, []
    for ticker in track(ticker_list, description="Fetching..."):
        try:
            df = get_ohlcv(ticker, start=start, end=end)
            if df.empty:
                console.print(f"  [yellow]⚠ {ticker}: no data[/yellow]")
                failed.append(ticker)
            else:
                ok += 1
        except Exception as exc:
            console.print(f"  [red]✗ {ticker}: {exc}[/red]")
            failed.append(ticker)

    console.print(f"\n[green]✓ {ok} tickers cached[/green]")
    if failed:
        console.print(f"[red]✗ Failed: {', '.join(failed)}[/red]")


if __name__ == "__main__":
    main()
