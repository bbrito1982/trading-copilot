#!/usr/bin/env python
"""Download and preprocess Brianferrell787/financial-news-multisource into a
sentiment cache compatible with the ensemble and backtest pipelines.

Downloads selected subset Parquet files directly (no datasets library needed),
filters to watchlist tickers via the extra_fields.stocks JSON array, scores
headlines with the trained neural embedder, and writes:

  data/multisource_headlines.parquet  — raw headlines for retraining embedder
  data/multisource_sentiment.parquet  — neural sentiment scores per (ticker, date)
  data/merged_sentiment.parquet       — merged with FNSPID neural cache

Requires HF_TOKEN env var (dataset is gated):
  export HF_TOKEN=hf_...

Usage
-----
    uv run python scripts/prepare_multisource.py --watchlist
    uv run python scripts/prepare_multisource.py -t AAPL -t NVDA --from 2021-01-01
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import httpx
import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn, DownloadColumn, Progress, SpinnerColumn,
    TextColumn, TransferSpeedColumn,
)

from trading_copilot.config import load_config
from trading_copilot.data.tiingo import get_ohlcv_cached_only

console = Console()
logger = logging.getLogger(__name__)

_HF_BASE = "https://huggingface.co/datasets/Brianferrell787/financial-news-multisource/resolve/main"

# Subset parquet files: (subset_name, [relative paths])
SUBSET_FILES: dict[str, list[str]] = {
    "reddit_finance_sp500": [
        "data/reddit_finance_sp500/reddit_finance_sp500.000.parquet",
    ],
    "nyt_articles_2000_present": [
        "data/nyt_articles_2000_present/nyt_articles_2000_present.000.parquet",
        "data/nyt_articles_2000_present/nyt_articles_2000_present.001.parquet",
        "data/nyt_articles_2000_present/nyt_articles_2000_present.002.parquet",
        "data/nyt_articles_2000_present/nyt_articles_2000_present.003.parquet",
        "data/nyt_articles_2000_present/nyt_articles_2000_present.004.parquet",
    ],
    "yahoo_finance_felixdrinkall": [
        "data/yahoo_finance_felixdrinkall/yahoo_finance_felixdrinkall.000.parquet",
    ],
}

DEFAULT_SUBSETS = ["reddit_finance_sp500", "nyt_articles_2000_present", "yahoo_finance_felixdrinkall"]

CACHE_DIR = Path("data/multisource_cache")
HEADLINES_OUT = Path("data/multisource_headlines.parquet")
SENTIMENT_OUT = Path("data/multisource_sentiment.parquet")
MERGED_OUT = Path("data/merged_sentiment.parquet")
FORWARD_DAYS = 10
CHUNK_SIZE = 1 << 20  # 1 MB


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _get_headers() -> dict[str, str]:
    from trading_copilot.config import settings
    token = os.environ.get("HF_TOKEN", "") or settings.hf_token
    if not token:
        console.print("[red]HF_TOKEN not set. Add it to .env or export it.[/red]")
        raise SystemExit(1)
    return {"Authorization": f"Bearer {token}"}


def _download_parquet(rel_path: str, headers: dict) -> Path:
    dest = CACHE_DIR / rel_path
    if dest.exists():
        console.print(f"  [dim]↩ cached {dest.name}[/dim]")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{_HF_BASE}/{rel_path}"
    tmp = dest.with_suffix(".tmp")

    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with Progress(
            SpinnerColumn(), TextColumn(f"[cyan]{dest.name}"),
            BarColumn(), DownloadColumn(), TransferSpeedColumn(),
            console=console,
        ) as prog:
            task = prog.add_task("", total=total or None)
            with tmp.open("wb") as fh:
                for chunk in r.iter_bytes(CHUNK_SIZE):
                    fh.write(chunk)
                    prog.advance(task, len(chunk))

    tmp.rename(dest)
    return dest


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _extract_tickers(extra_fields_raw) -> list[str]:
    if not extra_fields_raw:
        return []
    try:
        obj = json.loads(extra_fields_raw) if isinstance(extra_fields_raw, str) else extra_fields_raw
        stocks = obj.get("stocks") or obj.get("tickers") or obj.get("ticker") or []
        if isinstance(stocks, str):
            stocks = [stocks]
        return [s.upper().strip() for s in stocks if s]
    except Exception:
        return []


def _is_english(text: str) -> bool:
    try:
        return len(text) > 0 and sum(ord(c) < 128 for c in text) / len(text) > 0.85
    except Exception:
        return False


def _filter_parquet(path: Path, ticker_set: set[str], start_ts: pd.Timestamp) -> pd.DataFrame:
    """Read one parquet file, filter to tickers + date range."""
    df = pd.read_parquet(path)

    # Normalise date column
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if date_col is None:
        return pd.DataFrame()
    df["_date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=["_date"])
    df = df[df["_date"] >= start_ts]
    if df.empty:
        return pd.DataFrame()

    # Text column
    text_col = next((c for c in df.columns if c.lower() in ("text", "title", "headline", "article")), None)
    if text_col is None:
        return pd.DataFrame()

    # extra_fields for tickers
    ef_col = next((c for c in df.columns if "extra" in c.lower()), None)

    rows = []
    for _, row in df.iterrows():
        tickers = _extract_tickers(row.get(ef_col) if ef_col else None)
        matched = [t for t in tickers if t in ticker_set]
        if not matched:
            continue
        text = str(row.get(text_col, "")).strip()
        if not text or not _is_english(text):
            continue
        for ticker in matched:
            rows.append({"ticker": ticker, "date": row["_date"], "headline": text})

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["ticker", "date", "headline"])


# ---------------------------------------------------------------------------
# Forward returns + scoring
# ---------------------------------------------------------------------------

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
    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "adj_close"])
    return pd.concat(frames, ignore_index=True)


def _compute_forward_return(prices: pd.DataFrame, ticker: str, signal_date: pd.Timestamp) -> float | None:
    tp = prices[prices["ticker"] == ticker].set_index("date")["adj_close"]
    if signal_date not in tp.index:
        return None
    future = tp.index[tp.index > signal_date]
    if len(future) < FORWARD_DAYS:
        return None
    return float((tp[future[FORWARD_DAYS - 1]] - tp[signal_date]) / tp[signal_date])


def _build_caches(news: pd.DataFrame, prices: pd.DataFrame, tickers: list[str]):
    from trading_copilot.sentiment.tagger import score_headlines_neural

    headline_records, sentiment_records = [], []

    for ticker in tickers:
        tn = news[news["ticker"] == ticker]
        if tn.empty:
            console.print(f"  [yellow]⚠ {ticker}: no rows[/yellow]")
            continue

        daily = tn.groupby("date")["headline"].apply(list)
        console.print(f"  {ticker}: {len(daily)} days with headlines")

        for signal_date, headlines in daily.items():
            fwd = _compute_forward_return(prices, ticker, signal_date)
            result = score_headlines_neural(ticker, headlines)
            score = result.score
            label = int(fwd > 0) if fwd is not None else None
            headline_records.append({
                "ticker": ticker, "date": signal_date,
                "headlines_json": json.dumps(headlines),
                "fwd_return_10d": fwd, "label": label,
            })
            sentiment_records.append({
                "ticker": ticker, "date": signal_date,
                "sentiment_score": score,
                "fwd_return_10d": fwd, "label": label,
            })

    hl_df = pd.DataFrame(headline_records)
    sent_df = pd.DataFrame(sentiment_records)
    for df in (hl_df, sent_df):
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
    return hl_df, sent_df


def _merge_with_fnspid(new_df: pd.DataFrame) -> pd.DataFrame:
    fnspid_path = Path("data/fnspid_sentiment_neural.parquet")
    if not fnspid_path.exists():
        console.print("  [yellow]FNSPID neural cache not found, skipping merge[/yellow]")
        return new_df
    fnspid = pd.read_parquet(fnspid_path)
    fnspid["date"] = pd.to_datetime(fnspid["date"])
    fnspid["ticker"] = fnspid["ticker"].str.upper()
    combined = pd.concat([fnspid, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_date(ctx, param, value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise click.BadParameter(f"Expected YYYY-MM-DD, got '{value}'")


@click.command()
@click.option("--from", "start", default="2020-06-12", callback=_parse_date, metavar="YYYY-MM-DD",
              help="Start date (default: day after FNSPID ends).")
@click.option("--tickers", "-t", multiple=True, metavar="TICKER")
@click.option("--watchlist", is_flag=True, default=False, help="Use tickers from config.yaml.")
@click.option("--subsets", default=",".join(DEFAULT_SUBSETS), show_default=True,
              help="Comma-separated subset names.")
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(start, tickers, watchlist, subsets, verbose):
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

    subset_list = [s.strip() for s in subsets.split(",") if s.strip()]
    headers = _get_headers()

    console.print(f"\n[bold]prepare_multisource[/bold] — tickers: {ticker_list}")
    console.print(f"  Date from : {start}")
    console.print(f"  Subsets   : {subset_list}\n")

    # Step 1 — download
    console.print("[bold]Step 1 — Download[/bold]")
    local_files: list[Path] = []
    for subset in subset_list:
        files = SUBSET_FILES.get(subset, [])
        if not files:
            console.print(f"  [yellow]Unknown subset: {subset}[/yellow]")
            continue
        for rel in files:
            local_files.append(_download_parquet(rel, headers))

    # Step 2 — filter
    console.print("\n[bold]Step 2 — Filter to tickers + date range[/bold]")
    ticker_set = set(ticker_list)
    start_ts = pd.Timestamp(start)
    frames = []
    for path in local_files:
        df = _filter_parquet(path, ticker_set, start_ts)
        if not df.empty:
            frames.append(df)
        console.print(f"  {path.name}: {len(df):,} matching rows")

    if not frames:
        console.print("[red]No matching rows found.[/red]")
        raise SystemExit(1)

    news = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ticker", "date", "headline"])
    console.print(f"\n  Total unique rows: {len(news):,}\n")

    # Step 3 — prices + score
    console.print("[bold]Step 3 — Load prices[/bold]")
    prices = _load_prices(ticker_list, start)
    console.print(f"  {len(prices):,} price bars\n")

    console.print("[bold]Step 4 — Score sentiment[/bold]")
    hl_df, sent_df = _build_caches(news, prices, ticker_list)

    HEADLINES_OUT.parent.mkdir(parents=True, exist_ok=True)
    hl_df.to_parquet(HEADLINES_OUT, index=False)
    sent_df.to_parquet(SENTIMENT_OUT, index=False)
    console.print(f"\n[green]✓ Headlines → {HEADLINES_OUT}[/green]")
    console.print(f"[green]✓ Sentiment → {SENTIMENT_OUT}[/green]")

    # Step 5 — merge
    console.print("\n[bold]Step 5 — Merge with FNSPID neural cache[/bold]")
    merged = _merge_with_fnspid(sent_df)
    merged.to_parquet(MERGED_OUT, index=False)

    agree = (
        ((merged["sentiment_score"] > 0) == (merged["fwd_return_10d"] > 0))
        .dropna().mean()
        if "fwd_return_10d" in merged.columns else float("nan")
    )
    console.print(f"[green]✓ Merged → {MERGED_OUT}[/green]")
    console.print(f"  Rows      : {len(merged):,}")
    console.print(f"  Date range: {merged['date'].min().date()} → {merged['date'].max().date()}")
    console.print(f"  Dir agree : {agree:.1%}" if not np.isnan(agree) else "")

    console.print(
        "\nNext steps:\n"
        "  Retrain ensemble:\n"
        "    uv run python scripts/train_ensemble.py --watchlist --from 2018-01-01 \\\n"
        f"        --sentiment-cache {MERGED_OUT}\n"
        "  Backtest 2025+:\n"
        f"    uv run python scripts/run_backtest.py --watchlist --from 2025-01-01 \\\n"
        f"        --sentiment-cache {MERGED_OUT}\n"
    )


if __name__ == "__main__":
    main()
