"""Generate labelled training data from historical OHLCV and train the classifier.

Label definition
----------------
For every bar where ≥1 signal fires:
  - direction = 'buy' if net signal weight > 0, else 'sell'
  - label = 1 if the trade would have been profitable over ``forward_days`` bars
             (i.e. price went up for a buy, down for a sell)
  - label = 0 otherwise

The trained model outputs P(label=1 | features), which replaces the hand-tuned
SIGNAL_WEIGHTS conviction score in scorer.py.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from trading_copilot.signals.rules import detect_signals
from trading_copilot.signals.ml.features import FEATURE_NAMES, extract_features

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "data/ml_signal_model.joblib"


def build_dataset(
    ohlcv_data: dict[str, pd.DataFrame],
    cfg: dict,
    forward_days: int = 10,
    min_signals: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Walk all tickers day-by-day and produce (X, y, sample_meta) arrays.

    Parameters
    ----------
    ohlcv_data:
        Dict of ticker → OHLCV DataFrame (from DuckDB cache).
    cfg:
        Signal config dict.
    forward_days:
        How many bars ahead to measure outcome.
    min_signals:
        Minimum number of signals required to include a bar as a sample.
        Set to 1 to maximise data; set to 2 to match live scorer behaviour.
    """
    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    meta: list[str] = []

    for ticker, df in ohlcv_data.items():
        if df.empty or len(df) < 60 + forward_days:
            continue
        dates = sorted(df["date"].unique())

        for i in range(60, len(dates) - forward_days):
            window = df[df["date"] <= dates[i]].copy()
            signals = detect_signals(window, cfg)
            if len(signals) < min_signals:
                continue

            # Determine intended trade direction from net signal weight
            from trading_copilot.signals.scorer import SIGNAL_WEIGHTS
            raw = sum(SIGNAL_WEIGHTS.get(s.signal_type, 0) * s.strength for s in signals)
            direction = "buy" if raw >= 0 else "sell"

            # Label: was the direction correct?
            entry_price = float(window["adj_close"].iloc[-1])
            future_row = df[df["date"] == dates[i + forward_days]]
            if future_row.empty:
                continue
            exit_price = float(future_row["adj_close"].iloc[0])
            fwd_return = (exit_price - entry_price) / entry_price

            label = 1 if (
                (direction == "buy" and fwd_return > 0) or
                (direction == "sell" and fwd_return < 0)
            ) else 0

            try:
                vec = extract_features(window, signals, cfg)
            except Exception as exc:
                logger.debug("Feature extraction failed %s %s: %s", ticker, dates[i], exc)
                continue

            X_rows.append(vec)
            y_rows.append(label)
            meta.append(f"{ticker}@{dates[i]}")

    if not X_rows:
        raise ValueError("No training samples generated. Run backfill_prices.py first.")

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)
    logger.info("Dataset: %d samples, %d features, %.1f%% positive", len(y), X.shape[1], y.mean() * 100)
    return X, y, meta


def train(
    ohlcv_data: dict[str, pd.DataFrame],
    cfg: dict,
    forward_days: int = 10,
    min_signals: int = 1,
    model_path: str = DEFAULT_MODEL_PATH,
) -> Pipeline:
    """Build dataset, train GradientBoosting classifier, save to disk."""
    X, y, meta = build_dataset(ohlcv_data, cfg, forward_days=forward_days, min_signals=min_signals)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )),
    ])

    # Cross-validate first to report honest accuracy
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")
    logger.info("CV ROC-AUC: %.3f ± %.3f", cv_scores.mean(), cv_scores.std())

    pipeline.fit(X, y)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "feature_names": FEATURE_NAMES, "forward_days": forward_days}, model_path)
    logger.info("Model saved → %s", model_path)

    return pipeline, cv_scores
