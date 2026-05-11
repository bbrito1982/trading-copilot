#!/usr/bin/env python
"""Train the neural sentiment embedder (Phase 3b).

Loads fnspid_headlines.parquet, encodes each (ticker, date) group's headlines
with a frozen sentence-transformer, then trains ticker embeddings + MLP head
to predict 10-day return direction.

Usage
-----
    uv run python scripts/train_embedder.py --headlines data/fnspid_headlines.parquet
    uv run python scripts/train_embedder.py --headlines data/fnspid_headlines.parquet --epochs 20
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rich.console import Console
from rich.progress import track
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer

from trading_copilot.sentiment.embedder import (
    _build_model, _ST_MODEL_NAME, DEFAULT_EMBEDDER_PATH
)

console = Console()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _encode_headlines(
    df: pd.DataFrame,
    st_model: SentenceTransformer,
    ticker_to_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (embeddings, ticker_indices, labels) arrays."""
    embs, tickers, labels = [], [], []

    for _, row in track(df.iterrows(), total=len(df), description="Encoding headlines"):
        headlines = json.loads(row["headlines_json"])
        if not headlines:
            continue
        label = row["label"]
        if label is None or (isinstance(label, float) and np.isnan(label)):
            continue

        raw = st_model.encode(headlines, convert_to_numpy=True, show_progress_bar=False)
        mean_emb = raw.mean(axis=0)

        embs.append(mean_emb)
        tickers.append(ticker_to_idx.get(str(row["ticker"]).upper(), 0))
        labels.append(int(label))

    return np.array(embs, dtype=np.float32), np.array(tickers, dtype=np.int64), np.array(labels, dtype=np.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train_fold(
    X_emb: np.ndarray,
    X_tick: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    n_tickers: int,
    epochs: int,
    lr: float,
) -> tuple[float, dict]:
    """Train one CV fold; return (val_auc, best_state_dict)."""
    model = _build_model(n_tickers)

    X_emb_t = torch.tensor(X_emb)
    X_tick_t = torch.tensor(X_tick)
    y_t = torch.tensor(y)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_auc, best_state = 0.0, None
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(X_emb_t[train_idx], X_tick_t[train_idx])
        loss = criterion(logits, y_t[train_idx])
        loss.backward()
        opt.step()

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_emb_t[val_idx], X_tick_t[val_idx]).numpy()
            val_probs = 1 / (1 + np.exp(-val_logits))
            try:
                auc = roc_auc_score(y[val_idx], val_probs)
            except ValueError:
                auc = 0.5
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    return best_auc, best_state or model.state_dict()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--headlines", "headlines_path", default="data/fnspid_headlines.parquet",
              show_default=True, help="Headlines parquet from prepare_fnspid.py.")
@click.option("--output", default=DEFAULT_EMBEDDER_PATH, show_default=True)
@click.option("--epochs", default=30, show_default=True)
@click.option("--lr", default=1e-3, show_default=True)
@click.option("--folds", default=5, show_default=True)
@click.option("--verbose", "-v", is_flag=True)
def main(headlines_path, output, epochs, lr, folds, verbose):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    p = Path(headlines_path)
    if not p.exists():
        console.print(f"[red]Headlines file not found: {p}[/red]")
        console.print("Run prepare_fnspid.py first.")
        raise SystemExit(1)

    df = pd.read_parquet(p)
    df = df.dropna(subset=["label"])
    console.print(f"\n[bold]train_embedder[/bold] — {len(df):,} rows, "
                  f"{df['ticker'].nunique()} tickers\n")

    tickers = sorted(df["ticker"].str.upper().unique())
    ticker_to_idx = {t: i + 1 for i, t in enumerate(tickers)}  # 0 = unknown/padding
    n_tickers = len(tickers)

    console.print("Loading sentence-transformer …")
    st_model = SentenceTransformer(_ST_MODEL_NAME)

    console.print("Encoding all (ticker, date) groups …")
    X_emb, X_tick, y = _encode_headlines(df, st_model, ticker_to_idx)
    console.print(f"  Dataset: {len(y)} samples, {y.mean():.1%} positive (price up)\n")

    if len(y) < 50:
        console.print("[yellow]Warning: very few samples — model may not generalise.[/yellow]")

    # Cross-validation
    skf = StratifiedKFold(n_splits=min(folds, len(y) // 10 or 2), shuffle=True, random_state=42)
    aucs = []
    best_overall_auc = 0.0
    best_overall_state = None

    console.print(f"[bold]Cross-validation ({folds} folds, {epochs} epochs each)[/bold]")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_emb, y)):
        auc, state = _train_fold(X_emb, X_tick, y, train_idx, val_idx,
                                  n_tickers, epochs, lr)
        console.print(f"  Fold {fold + 1}: val AUC = {auc:.4f}")
        aucs.append(auc)
        if auc > best_overall_auc:
            best_overall_auc = auc
            best_overall_state = state

    mean_auc = float(np.mean(aucs))
    console.print(f"\n  Mean CV AUC: {mean_auc:.4f}  (best fold: {best_overall_auc:.4f})")

    # Retrain on full data using best fold's state as init
    console.print("\n[bold]Retraining on full dataset …[/bold]")
    final_model = _build_model(n_tickers)
    if best_overall_state:
        final_model.load_state_dict(best_overall_state)

    X_emb_t = torch.tensor(X_emb)
    X_tick_t = torch.tensor(X_tick)
    y_t = torch.tensor(y)
    opt = torch.optim.Adam(final_model.parameters(), lr=lr * 0.5, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    for _ in track(range(epochs // 2), description="Final training"):
        final_model.train()
        opt.zero_grad()
        logits = final_model(X_emb_t, X_tick_t)
        loss = criterion(logits, y_t)
        loss.backward()
        opt.step()

    # Directional agreement on full set
    final_model.eval()
    with torch.no_grad():
        all_logits = final_model(X_emb_t, X_tick_t).numpy()
    preds = (all_logits > 0).astype(int)
    dir_agree = (preds == y.astype(int)).mean()
    console.print(f"  Full-set directional agreement: {dir_agree:.1%}  "
                  f"(keyword baseline: 3.9%)")

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "ticker_to_idx": ticker_to_idx,
        "model_state": final_model.state_dict(),
        "cv_auc": mean_auc,
        "dir_agree": float(dir_agree),
        "st_model_name": _ST_MODEL_NAME,
    }
    torch.save(checkpoint, out)
    console.print(f"\n[green]✓ Embedder saved → {out}[/green]")

    # --- Emit neural sentiment parquet for use in backtest / ensemble ---
    # Score every row in the full (labelled + unlabelled) headlines file so
    # the backtest can look up the sentiment for any (ticker, date).
    console.print("\nScoring all headlines with neural embedder …")
    df_full = pd.read_parquet(p)  # includes rows without a forward return label

    # Re-encode the full set (we already have X_emb for the labelled subset;
    # re-encode from scratch to include unlabelled rows too).
    all_embs, all_tickers_idx, all_rows = [], [], []
    for _, row in track(df_full.iterrows(), total=len(df_full), description="Re-encoding"):
        headlines = json.loads(row["headlines_json"])
        if not headlines:
            continue
        raw = st_model.encode(headlines, convert_to_numpy=True, show_progress_bar=False)
        mean_emb = raw.mean(axis=0)
        tidx = ticker_to_idx.get(str(row["ticker"]).upper(), 0)
        all_embs.append(mean_emb)
        all_tickers_idx.append(tidx)
        all_rows.append(row)

    emb_t = torch.tensor(np.array(all_embs, dtype=np.float32))
    tick_t = torch.tensor(np.array(all_tickers_idx, dtype=np.int64))

    final_model.eval()
    with torch.no_grad():
        logits = final_model(emb_t, tick_t).numpy()

    probs_up = 1 / (1 + np.exp(-logits))
    neural_scores = ((probs_up - 0.5) * 2).round(4)

    neural_df = pd.DataFrame([
        {
            "ticker": row["ticker"],
            "date": row["date"],
            "sentiment_score": float(score),
            "fwd_return_10d": row.get("fwd_return_10d"),
            "label": row.get("label"),
        }
        for row, score in zip(all_rows, neural_scores)
    ])
    neural_df["date"] = pd.to_datetime(neural_df["date"])

    neural_out = out.parent / "fnspid_sentiment_neural.parquet"
    neural_df.to_parquet(neural_out, index=False)
    neural_agree = (
        ((neural_df["sentiment_score"] > 0) == (neural_df["fwd_return_10d"] > 0))
        .dropna().mean()
        if "fwd_return_10d" in neural_df else float("nan")
    )
    console.print(f"[green]✓ Neural sentiment cache → {neural_out}[/green]")
    console.print(f"  Rows: {len(neural_df):,}  |  Directional agree: {neural_agree:.1%}")

    console.print(
        "\nNext steps:\n"
        "  Backtest with sentiment:\n"
        f"    uv run python scripts/run_backtest.py --watchlist --from 2025-01-01 "
        f"--sentiment-cache {neural_out}\n"
        "  Retrain ensemble:\n"
        "    uv run python scripts/train_ensemble.py --watchlist --from 2018-01-01 "
        f"--sentiment-cache {neural_out}\n"
    )


if __name__ == "__main__":
    main()
