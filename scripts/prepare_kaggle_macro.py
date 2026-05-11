#!/usr/bin/env python
"""Process the Kaggle S&P 500 headlines dataset as a macro sentiment source.

Reads data/kaggle_cache/sp500_headlines_2008_2024.csv (already downloaded),
applies the keyword theme table to derive per-ticker macro scores for each date,
then merges with the existing ticker-specific sentiment cache to produce a
blended data/merged_sentiment.parquet covering 2008–2024.

Usage
-----
    uv run python scripts/prepare_kaggle_macro.py --watchlist
    uv run python scripts/prepare_kaggle_macro.py --watchlist --ticker-weight 0.6
"""
from __future__ import annotations

import logging
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import track

from trading_copilot.config import load_config
from trading_copilot.data.tiingo import get_ohlcv_cached_only
from trading_copilot.sentiment.tagger import score_macro_headlines, blend_sentiment

console = Console()
logger = logging.getLogger(__name__)

KAGGLE_CSV = Path("data/kaggle_cache/sp500_headlines_2008_2024.csv")
MACRO_OUT = Path("data/kaggle_macro_sentiment.parquet")
MERGED_OUT = Path("data/merged_sentiment.parquet")
FORWARD_DAYS = 10


def _load_prices(tickers: list[str], start: date) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        df = get_ohlcv_cached_only(ticker, start=start, end=date.today())
        if df.empty:
            continue
        df = df[["date", "adj_close"]].copy()
        df["ticker"] = ticker
        df["date"] = pd.to_datetime(df["date"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["ticker", "date", "adj_close"]
    )


def _compute_forward_return(prices: pd.DataFrame, ticker: str, signal_date: pd.Timestamp) -> float | None:
    tp = prices[prices["ticker"] == ticker].set_index("date")["adj_close"]
    if signal_date not in tp.index:
        return None
    future = tp.index[tp.index > signal_date]
    if len(future) < FORWARD_DAYS:
        return None
    return float((tp[future[FORWARD_DAYS - 1]] - tp[signal_date]) / tp[signal_date])


def _build_macro_cache(tickers: list[str], start: date) -> pd.DataFrame:
    if not KAGGLE_CSV.exists():
        console.print(f"[red]Kaggle CSV not found: {KAGGLE_CSV}[/red]")
        console.print("Run: uv run kaggle datasets download dyutidasmahaptra/s-and-p-500-with-financial-news-headlines-20082024 -p data/kaggle_cache --unzip")
        raise SystemExit(1)

    df = pd.read_csv(KAGGLE_CSV)
    df["date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])
    df = df[df["date"] >= pd.Timestamp(start)]
    console.print(f"  Kaggle CSV: {len(df):,} rows after {start}")

    prices = _load_prices(tickers, start)
    console.print(f"  Price bars: {len(prices):,}")

    records = []
    daily = df.groupby("date")["Title"].apply(list)
    console.print(f"  Scoring {len(daily):,} dates across {len(tickers)} tickers …")

    for signal_date, headlines in track(daily.items(), description="Macro scoring"):
        macro_scores = score_macro_headlines(headlines, tickers)
        for ticker in tickers:
            score = macro_scores.get(ticker, 0.0)
            fwd = _compute_forward_return(prices, ticker, signal_date)
            records.append({
                "ticker": ticker,
                "date": signal_date,
                "macro_score": score,
                "fwd_return_10d": fwd,
            })

    result = pd.DataFrame(records)
    result["date"] = pd.to_datetime(result["date"])
    # Drop rows where both score is 0 and no fwd return (no signal, no label)
    result = result[result["macro_score"] != 0.0].copy()
    return result


def _merge_with_ticker_cache(macro_df: pd.DataFrame, ticker_weight: float) -> pd.DataFrame:
    """Blend macro scores with existing ticker-specific sentiment cache."""
    ticker_path = Path("data/multisource_sentiment.parquet")
    fnspid_path = Path("data/fnspid_sentiment_neural.parquet")

    frames = []
    for p in (fnspid_path, ticker_path):
        if p.exists():
            df = pd.read_parquet(p)
            df["date"] = pd.to_datetime(df["date"])
            df["ticker"] = df["ticker"].str.upper()
            frames.append(df)
            console.print(f"  Loaded {p.name}: {len(df):,} rows")

    # Pivot macro to (ticker, date, macro_score)
    macro_pivot = macro_df[["ticker", "date", "macro_score", "fwd_return_10d"]].copy()

    if not frames:
        # No ticker-specific cache — use macro only
        console.print("  No ticker-specific cache found, using macro scores only")
        result = macro_pivot.rename(columns={"macro_score": "sentiment_score"})
        result["label"] = result["fwd_return_10d"].apply(
            lambda x: int(x > 0) if x is not None and not np.isnan(x) else None
        )
        return result

    ticker_df = pd.concat(frames, ignore_index=True)
    ticker_df = ticker_df.drop_duplicates(subset=["ticker", "date"], keep="last")

    # Merge on (ticker, date) — outer join to keep rows from either source
    merged = ticker_df.merge(macro_pivot[["ticker", "date", "macro_score"]],
                              on=["ticker", "date"], how="outer")

    # Blend scores — treat NaN ticker score as absent (None)
    merged["sentiment_score"] = merged.apply(
        lambda row: blend_sentiment(
            None if pd.isna(row.get("sentiment_score")) else row.get("sentiment_score"),
            None if pd.isna(row.get("macro_score")) else row.get("macro_score"),
            ticker_weight,
        ),
        axis=1,
    )

    # Prefer fwd_return from ticker cache; fill from macro where missing
    if "fwd_return_10d_x" in merged.columns:
        merged["fwd_return_10d"] = merged["fwd_return_10d_x"].fillna(merged.get("fwd_return_10d_y"))
        merged = merged.drop(columns=["fwd_return_10d_x", "fwd_return_10d_y"], errors="ignore")

    merged = merged.drop(columns=["macro_score"], errors="ignore")
    merged["label"] = merged["fwd_return_10d"].apply(
        lambda x: int(x > 0) if x is not None and not np.isnan(float(x) if x is not None else float("nan")) else None
    )
    merged = merged.drop_duplicates(subset=["ticker", "date"], keep="last")
    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Per-ticker z-score normalisation — removes static bias so the signal
    # reflects "today's sentiment relative to this ticker's baseline" rather
    # than a fixed bullish/bearish tilt baked in during training.
    def _normalise(group):
        s = group["sentiment_score"]
        mu, sigma = s.mean(), s.std()
        if sigma < 1e-6:
            group["sentiment_score"] = 0.0
        else:
            group["sentiment_score"] = ((s - mu) / sigma).clip(-3, 3) / 3  # → [-1, +1]
        return group

    merged = merged.groupby("ticker", group_keys=False).apply(_normalise)
    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
    return merged


def _parse_date(ctx, param, value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise click.BadParameter(f"Expected YYYY-MM-DD, got '{value}'")


@click.command()
@click.option("--from", "start", default="2018-01-01", callback=_parse_date, metavar="YYYY-MM-DD")
@click.option("--tickers", "-t", multiple=True, metavar="TICKER")
@click.option("--watchlist", is_flag=True, default=False)
@click.option("--ticker-weight", default=0.6, show_default=True,
              help="Weight for ticker-specific score when blending (0–1). Remainder goes to macro.")
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(start, tickers, watchlist, ticker_weight, verbose):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config()
    if tickers:
        ticker_list = [t.upper() for t in tickers]
    elif watchlist:
        ticker_list = [t.upper() for t in cfg.get("watchlist", [])]
    else:
        console.print("[red]Specify --tickers or --watchlist.[/red]")
        raise SystemExit(1)

    console.print(f"\n[bold]prepare_kaggle_macro[/bold] — tickers: {ticker_list}")
    console.print(f"  Start: {start}  |  Ticker weight: {ticker_weight}\n")

    console.print("[bold]Step 1 — Build macro sentiment cache[/bold]")
    macro_df = _build_macro_cache(ticker_list, start)
    MACRO_OUT.parent.mkdir(parents=True, exist_ok=True)
    macro_df.to_parquet(MACRO_OUT, index=False)
    console.print(f"[green]✓ Macro cache → {MACRO_OUT}[/green]  ({len(macro_df):,} rows)\n")

    console.print("[bold]Step 2 — Merge with ticker-specific cache[/bold]")
    merged = _merge_with_ticker_cache(macro_df, ticker_weight)
    merged.to_parquet(MERGED_OUT, index=False)

    agree = (
        ((merged["sentiment_score"] > 0) == (merged["fwd_return_10d"] > 0))
        .dropna().mean()
        if "fwd_return_10d" in merged.columns else float("nan")
    )
    console.print(f"\n[green]✓ Merged → {MERGED_OUT}[/green]")
    console.print(f"  Rows      : {len(merged):,}")
    console.print(f"  Date range: {merged['date'].min().date()} → {merged['date'].max().date()}")
    console.print(f"  Dir agree : {agree:.1%}" if not np.isnan(agree) else "")

    console.print(
        "\nNext steps:\n"
        "  Retrain ensemble:\n"
        "    uv run python scripts/train_ensemble.py --watchlist --from 2018-01-01 \\\n"
        f"        --to 2021-12-31 --sentiment-cache {MERGED_OUT}\n"
        "  Backtest:\n"
        f"    uv run python scripts/run_backtest.py --watchlist --from 2022-01-01 \\\n"
        f"        --to 2023-12-31 --sentiment-cache {MERGED_OUT}\n"
    )


if __name__ == "__main__":
    main()
