"""Neural sentiment embedder — Phase 3b.

Replaces the keyword tagger with a learned model:
    sentence-transformer(headline) ⊕ ticker_embedding → MLP → P(price_up)

The sentence-transformer is frozen; only the ticker embeddings and MLP head
are trained on FNSPID (ticker, date, headlines[], fwd_return_10d) data.

Output: sentiment score in [-1, +1] (positive = bullish, negative = bearish).
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDER_PATH = "data/sentiment_embedder.pt"

# Sentence-transformer model — small, fast, good quality
_ST_MODEL_NAME = "all-MiniLM-L6-v2"
_ST_DIM = 384
_TICKER_DIM = 16
_HIDDEN = 128


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def _build_model(n_tickers: int):
    """Return (model, ticker_to_idx) — requires torch."""
    import torch
    import torch.nn as nn

    class SentimentHead(nn.Module):
        def __init__(self, n_tickers: int):
            super().__init__()
            self.ticker_emb = nn.Embedding(n_tickers + 1, _TICKER_DIM, padding_idx=0)
            self.mlp = nn.Sequential(
                nn.Linear(_ST_DIM + _TICKER_DIM, _HIDDEN),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(_HIDDEN, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, headline_emb, ticker_idx):
            te = self.ticker_emb(ticker_idx)
            x = torch.cat([headline_emb, te], dim=-1)
            return self.mlp(x).squeeze(-1)

    return SentimentHead(n_tickers)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class SentimentEmbedder:
    """Loaded model for inference."""

    def __init__(self, checkpoint: dict):
        import torch
        from sentence_transformers import SentenceTransformer

        self._st = SentenceTransformer(_ST_MODEL_NAME)
        self._ticker_to_idx: dict[str, int] = checkpoint["ticker_to_idx"]
        n_tickers = len(self._ticker_to_idx)
        self._model = _build_model(n_tickers)
        self._model.load_state_dict(checkpoint["model_state"])
        self._model.eval()
        self._device = torch.device("cpu")

    def predict_score(self, ticker: str, headlines: list[str]) -> float:
        """Return sentiment score in [-1, +1].  Positive = bullish."""
        import torch

        if not headlines:
            return 0.0

        embs = self._st.encode(headlines, convert_to_numpy=True, show_progress_bar=False)
        mean_emb = embs.mean(axis=0)

        ticker_idx = self._ticker_to_idx.get(ticker.upper(), 0)  # 0 = unknown

        with torch.no_grad():
            h = torch.tensor(mean_emb, dtype=torch.float32).unsqueeze(0)
            t = torch.tensor([ticker_idx], dtype=torch.long)
            logit = self._model(h, t).item()

        # Map logit → [-1, +1]: sigmoid gives P(up), centre at 0.5
        prob_up = 1.0 / (1.0 + np.exp(-logit))
        return float(round((prob_up - 0.5) * 2, 4))


@lru_cache(maxsize=1)
def _load_cached(path: str) -> SentimentEmbedder | None:
    import torch
    p = Path(path)
    if not p.exists():
        return None
    try:
        checkpoint = torch.load(p, map_location="cpu", weights_only=False)
        return SentimentEmbedder(checkpoint)
    except Exception as exc:
        logger.warning("Could not load sentiment embedder from %s: %s", path, exc)
        return None


def load_embedder(path: str = DEFAULT_EMBEDDER_PATH) -> SentimentEmbedder | None:
    """Return a cached SentimentEmbedder, or None if not trained yet."""
    try:
        return _load_cached(path)
    except Exception:
        return None


def embedder_exists(path: str = DEFAULT_EMBEDDER_PATH) -> bool:
    return Path(path).exists()
