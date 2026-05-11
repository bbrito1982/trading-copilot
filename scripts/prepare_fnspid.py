#!/usr/bin/env python
"""Download and preprocess FNSPID data into a sentiment cache for ensemble training.

Downloads:
  - FNSPID news CSV (HuggingFace: Zihan1004/FNSPID — Stock_news/nasdaq_exteral_data.csv, 23 GB)
    or the smaller All_external.csv (5.7 GB) via --small flag
  - FNSPID full price history zip (Stock_price/full_history.zip, ~590 MB)

Outputs:
  data/fnspid_sentiment.parquet  — columns: ticker, date, sentiment_score

Usage
-----
    uv run python scripts/prepare_fnspid.py --watchlist
    uv run python scripts/prepare_fnspid.py --watchlist --small        # use 5 GB file instead
    uv run python scripts/prepare_fnspid.py -t AAPL -t MSFT --from 2018-01-01
    uv run python scripts/prepare_fnspid.py --watchlist --skip-download  # reprocess cached files
"""
from __future__ import annotations

import io
import logging
import sys
import zipfile
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import httpx
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn

from trading_copilot.config import load_config
from trading_copilot.sentiment.tagger import score_headlines

console = Console()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FNSPID HuggingFace URLs
# ---------------------------------------------------------------------------
_HF_BASE = "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main"
# Large file (23 GB) — full NASDAQ news; small alternative (5.7 GB) via --small flag
_NEWS_URL_LARGE = f"{_HF_BASE}/Stock_news/nasdaq_exteral_data.csv"
_NEWS_URL_SMALL = f"{_HF_BASE}/Stock_news/All_external.csv"
_PRICE_URL = f"{_HF_BASE}/Stock_price/full_history.zip"

_CACHE_DIR = Path("data/fnspid_cache")
_NEWS_CACHE_LARGE = _CACHE_DIR / "news_large.csv"
_NEWS_CACHE_SMALL = _CACHE_DIR / "news_small.csv"
_PRICE_CACHE = _CACHE_DIR / "prices.parquet"
OUTPUT_PATH = Path("data/fnspid_sentiment.parquet")
HEADLINES_PATH = Path("data/fnspid_headlines.parquet")

FORWARD_DAYS = 10
CHUNK_SIZE = 1 << 20  # 1 MB


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path, label: str) -> None:
    """Stream download directly to disk — never buffers the full file in RAM."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        console.print(f"  [dim]↩ cached {dest}[/dim]")
        return

    tmp = dest.with_suffix(".tmp")
    console.print(f"  ↓ {label} …")
    with httpx.stream("GET", url, follow_redirects=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written = 0
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[cyan]{label}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            console=console,
        ) as prog:
            task = prog.add_task("", total=total or None)
            with tmp.open("wb") as fh:
                for chunk in r.iter_bytes(CHUNK_SIZE):
                    fh.write(chunk)
                    written += len(chunk)
                    prog.advance(task, len(chunk))

    tmp.rename(dest)
    console.print(f"  [green]✓ saved {dest} ({written / 1e6:.1f} MB)[/green]")


def _download_price_zip() -> None:
    zip_path = _CACHE_DIR / "full_history.zip"
    _download(_PRICE_URL, zip_path, "FNSPID prices (zip)")

    if _PRICE_CACHE.exists():
        return

    console.print("  Extracting price CSVs …")
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path) as zf:
        # Skip __MACOSX metadata entries; real CSVs live under full_history/
        csv_names = [n for n in zf.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX")]
        for name in csv_names:
            ticker = Path(name).stem.upper()
            try:
                df = pd.read_csv(io.BytesIO(zf.read(name)))
                df.columns = [c.lower().strip() for c in df.columns]
                # column is "adj close" (space) not "adj_close"
                adj_col = next((c for c in df.columns if c.startswith("adj")), None)
                if adj_col is None or "date" not in df.columns:
                    continue
                df["ticker"] = ticker
                df = df.rename(columns={adj_col: "adj_close"})[["ticker", "date", "adj_close"]].dropna()
                frames.append(df)
            except Exception:
                pass

    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices.to_parquet(_PRICE_CACHE, index=False)
    console.print(f"  [green]✓ price cache: {len(prices):,} rows[/green]")


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def _detect_col_map(columns: list[str]) -> dict[str, str]:
    cols = [c.lower().strip() for c in columns]
    col_map: dict[str, str] = {}
    for cand in ("stock_ticker", "ticker", "symbol", "stock_symbol"):
        if cand in cols:
            col_map[columns[cols.index(cand)]] = "ticker"
            break
    for cand in ("date", "datetime", "published_date", "publisheddate"):
        if cand in cols:
            col_map[columns[cols.index(cand)]] = "date"
            break
    for cand in ("title", "headline", "article_title", "article"):
        if cand in cols:
            col_map[columns[cols.index(cand)]] = "headline"
            break
    return col_map


def _is_english(text: str) -> bool:
    try:
        return sum(ord(c) < 128 for c in text) / len(text) > 0.9
    except Exception:
        return False


def _load_news(tickers: list[str], start: date, small: bool = False) -> pd.DataFrame:
    """Read the news CSV in 50k-row chunks, filtering as we go to stay within RAM."""
    path = _NEWS_CACHE_SMALL if small else _NEWS_CACHE_LARGE
    console.print(f"  Streaming {path} in chunks …")

    ticker_set = set(tickers)
    start_ts = pd.Timestamp(start)
    kept: list[pd.DataFrame] = []
    col_map: dict[str, str] | None = None
    total_rows = 0
    dropped_lang = 0

    for chunk in pd.read_csv(path, chunksize=50_000, low_memory=False):
        total_rows += len(chunk)

        if col_map is None:
            col_map = _detect_col_map(list(chunk.columns))

        chunk = chunk.rename(columns=col_map)
        missing = {"ticker", "date", "headline"} - set(chunk.columns)
        if missing:
            logger.warning("Could not map columns %s — skipping chunk. Available: %s", missing, list(chunk.columns))
            continue

        chunk = chunk[["ticker", "date", "headline"]].dropna(subset=["headline"])
        chunk["ticker"] = chunk["ticker"].astype(str).str.upper().str.strip()
        chunk = chunk[chunk["ticker"].isin(ticker_set)]
        if chunk.empty:
            continue

        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
        chunk = chunk.dropna(subset=["date"])
        chunk = chunk[chunk["date"] >= start_ts]
        if chunk.empty:
            continue

        eng_mask = chunk["headline"].apply(_is_english)
        dropped_lang += (~eng_mask).sum()
        chunk = chunk[eng_mask]
        if not chunk.empty:
            kept.append(chunk)

    if dropped_lang:
        logger.info("Dropped %d non-English headlines", dropped_lang)

    if not kept:
        return pd.DataFrame(columns=["ticker", "date", "headline"])

    news = pd.concat(kept, ignore_index=True)
    console.print(f"  Scanned {total_rows:,} rows → kept {len(news):,} matching rows")
    return news


def _load_prices(tickers: list[str], start: date) -> pd.DataFrame:
    prices = pd.read_parquet(_PRICE_CACHE)
    prices = prices[prices["ticker"].isin(tickers)]
    prices = prices[prices["date"] >= pd.Timestamp(start)]
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    return prices


def _compute_forward_return(prices: pd.DataFrame, ticker: str, signal_date: pd.Timestamp) -> float | None:
    """10-day forward return for *ticker* starting at *signal_date*."""
    tp = prices[prices["ticker"] == ticker].set_index("date")["adj_close"]
    if signal_date not in tp.index:
        return None
    future_dates = tp.index[tp.index > signal_date]
    if len(future_dates) < FORWARD_DAYS:
        return None
    exit_date = future_dates[FORWARD_DAYS - 1]
    return float((tp[exit_date] - tp[signal_date]) / tp[signal_date])


def _build_sentiment_cache(
    news: pd.DataFrame,
    prices: pd.DataFrame,
    tickers: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each (ticker, date) with headlines, score sentiment and attach 10-day label.

    Returns (sentiment_df, headlines_df) where headlines_df stores raw headline
    lists as JSON strings for neural embedder training.
    """
    import json

    records: list[dict] = []
    headline_records: list[dict] = []

    for ticker in tickers:
        tn = news[news["ticker"] == ticker]
        if tn.empty:
            console.print(f"  [yellow]⚠ {ticker}: no news rows[/yellow]")
            continue

        daily = tn.groupby("date")["headline"].apply(list)
        console.print(f"  {ticker}: {len(daily)} trading days with headlines")

        for signal_date, headlines in daily.items():
            result = score_headlines(ticker, headlines)
            fwd = _compute_forward_return(prices, ticker, signal_date)
            records.append({
                "ticker": ticker,
                "date": signal_date,
                "sentiment_score": result.score,
                "matched_themes": ",".join(result.matched_themes),
                "headline_count": result.headline_count,
                "fwd_return_10d": fwd,
                "label": (
                    int((result.score > 0 and fwd > 0) or (result.score < 0 and fwd < 0))
                    if fwd is not None else None
                ),
            })
            headline_records.append({
                "ticker": ticker,
                "date": signal_date,
                "headlines_json": json.dumps(headlines),
                "fwd_return_10d": fwd,
                # Binary direction label: 1 = price went up, 0 = down
                "label": int(fwd > 0) if fwd is not None else None,
            })

    sentiment_df = pd.DataFrame(records)
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])

    headlines_df = pd.DataFrame(headline_records)
    headlines_df["date"] = pd.to_datetime(headlines_df["date"])

    return sentiment_df, headlines_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_date(ctx, param, value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise click.BadParameter(f"Expected YYYY-MM-DD, got '{value}'")


@click.command()
@click.option("--from", "start", default="2018-01-01", callback=_parse_date, metavar="YYYY-MM-DD",
              help="Earliest date to include.")
@click.option("--tickers", "-t", multiple=True, metavar="TICKER")
@click.option("--watchlist", is_flag=True, default=False, help="Use tickers from config.yaml.")
@click.option("--skip-download", is_flag=True, default=False,
              help="Skip download if cached files already exist.")
@click.option("--small", is_flag=True, default=False,
              help="Use All_external.csv (5.7 GB) instead of nasdaq_exteral_data.csv (23 GB).")
@click.option("--output", default=str(OUTPUT_PATH), show_default=True)
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(start, tickers, watchlist, skip_download, small, output, verbose):
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

    console.print(f"\n[bold]FNSPID prepare[/bold] — tickers: {ticker_list}\n")

    news_url = _NEWS_URL_SMALL if small else _NEWS_URL_LARGE
    news_cache = _NEWS_CACHE_SMALL if small else _NEWS_CACHE_LARGE
    news_label = "FNSPID news small/All_external.csv (5.7 GB)" if small else "FNSPID news large/nasdaq_exteral_data.csv (23 GB)"

    if not skip_download:
        console.print("[bold]Step 1 — Download[/bold]")
        _download(news_url, news_cache, news_label)
        _download_price_zip()
    else:
        if not news_cache.exists() or not _PRICE_CACHE.exists():
            console.print("[red]--skip-download set but cache files missing. Remove the flag.[/red]")
            raise SystemExit(1)

    console.print("\n[bold]Step 2 — Load & filter[/bold]")
    news = _load_news(ticker_list, start, small=small)
    prices = _load_prices(ticker_list, start)

    if news.empty:
        console.print("[red]No news rows matched the tickers/date range.[/red]")
        raise SystemExit(1)

    console.print("\n[bold]Step 3 — Score sentiment[/bold]")
    cache, headlines_df = _build_sentiment_cache(news, prices, ticker_list)

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache.to_parquet(out, index=False)

    hl_out = out.parent / HEADLINES_PATH.name
    headlines_df.to_parquet(hl_out, index=False)

    # Summary
    total = len(cache)
    with_label = cache["label"].notna().sum()
    agree = cache.dropna(subset=["label"])["label"].mean()
    console.print(f"\n[green]✓ Sentiment cache saved → {out}[/green]")
    console.print(f"[green]✓ Headlines cache saved → {hl_out}[/green]")
    console.print(f"  Total rows       : {total:,}")
    console.print(f"  Rows with label  : {with_label:,}")
    console.print(f"  Directional agree: {agree:.1%}" if with_label else "  (no labels)")
    console.print(
        "\nNext steps:\n"
        "  1. Train neural sentiment embedder:\n"
        f"     uv run python scripts/train_embedder.py --headlines {hl_out}\n"
        "  2. Retrain ensemble:\n"
        "     uv run python scripts/train_ensemble.py --watchlist --from 2018-01-01 "
        f"--sentiment-cache {out}\n"
    )


if __name__ == "__main__":
    main()
