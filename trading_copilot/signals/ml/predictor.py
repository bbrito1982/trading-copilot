"""Load the trained model and predict conviction for a live bar."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from trading_copilot.signals.rules import Signal
from trading_copilot.signals.ml.features import extract_features
from trading_copilot.signals.ml.trainer import DEFAULT_MODEL_PATH

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model(model_path: str) -> dict | None:
    if not Path(model_path).exists():
        return None
    return joblib.load(model_path)


def predict_conviction(
    df: pd.DataFrame,
    signals: list[Signal],
    cfg: dict,
    model_path: str = DEFAULT_MODEL_PATH,
) -> float | None:
    """Return model's P(profitable) for the current bar, or None if no model.

    Returns a float in [0, 1] representing the classifier's confidence that
    the signal will produce a profitable outcome, regardless of direction.
    The caller is responsible for checking direction via the signals list.
    """
    bundle = _load_model(model_path)
    if bundle is None:
        return None

    try:
        vec = extract_features(df, signals, cfg)
        proba = bundle["pipeline"].predict_proba(vec.reshape(1, -1))[0][1]
        return float(proba)
    except Exception as exc:
        logger.warning("ML prediction failed: %s", exc)
        return None


def model_exists(model_path: str = DEFAULT_MODEL_PATH) -> bool:
    return Path(model_path).exists()


def invalidate_cache() -> None:
    """Call after retraining to force model reload."""
    _load_model.cache_clear()
