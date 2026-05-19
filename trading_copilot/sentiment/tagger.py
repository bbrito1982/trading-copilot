"""Keyword-based macro sentiment tagger.

Maps headline text → macro themes → per-ticker directional sentiment score.

Same headline can be bullish for one asset and bearish for another
(e.g. "Fed hikes rates" → TLT bearish, GLD neutral, XLF bullish).

Phase 3b will replace the keyword matching with sentence-transformer embeddings,
but the theme → ticker effect table stays the same.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Theme definitions
# Each theme has:
#   - keywords: list of regex patterns (case-insensitive OR match)
#   - effects:  dict[ticker, float] where +1 = strongly bullish, -1 = strongly bearish
#               Tickers not listed → no effect (0)
# ---------------------------------------------------------------------------

@dataclass
class Theme:
    name: str
    keywords: list[str]
    effects: dict[str, float]   # ticker → directional effect (-1 to +1)
    _pattern: re.Pattern = field(init=False, repr=False)

    def __post_init__(self):
        joined = "|".join(self.keywords)
        self._pattern = re.compile(joined, re.IGNORECASE)

    def matches(self, text: str) -> bool:
        return bool(self._pattern.search(text))


THEMES: list[Theme] = [
    Theme(
        name="fed_rate_hike",
        keywords=[
            r"fed\b.*hike", r"rate hike", r"raises? rates?", r"tightening",
            r"hawkish", r"50bp", r"75bp", r"interest rate.*rise",
        ],
        effects={
            "SPY": -0.3, "QQQ": -0.4, "AAPL": -0.3, "MSFT": -0.3,
            "GOOGL": -0.3, "AMZN": -0.3, "NVDA": -0.4,
            "TLT": -0.7,   # bonds sell off hard on hikes
            "GLD": -0.2,
            "XLF": +0.4,   # banks benefit from higher rates
            "XLE": -0.1,
        },
    ),
    Theme(
        name="fed_rate_cut",
        keywords=[
            r"fed\b.*cut", r"rate cut", r"cuts? rates?", r"dovish",
            r"easing", r"pivot", r"lower rates?",
        ],
        effects={
            "SPY": +0.3, "QQQ": +0.4, "AAPL": +0.3, "MSFT": +0.3,
            "GOOGL": +0.3, "AMZN": +0.3, "NVDA": +0.4,
            "TLT": +0.6,
            "GLD": +0.3,
            "XLF": -0.3,
            "XLE": +0.1,
        },
    ),
    Theme(
        name="recession_fear",
        keywords=[
            r"recession", r"economic contraction", r"gdp.*decline",
            r"hard landing", r"downturn", r"slowdown",
        ],
        effects={
            "SPY": -0.5, "QQQ": -0.4, "AAPL": -0.3, "MSFT": -0.3,
            "GOOGL": -0.3, "AMZN": -0.4, "NVDA": -0.4,
            "TLT": +0.5,   # flight to safety
            "GLD": +0.4,
            "XLF": -0.5,
            "XLE": -0.4,
        },
    ),
    Theme(
        name="inflation_high",
        keywords=[
            r"inflation.*surge", r"cpi.*high", r"pce.*hot",
            r"price.*surge", r"inflation.*jump", r"inflationary",
        ],
        effects={
            "SPY": -0.2, "QQQ": -0.3,
            "TLT": -0.5,
            "GLD": +0.5,   # inflation hedge
            "XLE": +0.3,   # energy benefits from inflation
            "XLF": -0.2,
        },
    ),
    Theme(
        name="oil_supply_cut",
        keywords=[
            r"opec.*cut", r"oil.*supply.*cut", r"production.*cut",
            r"saudi.*cut", r"output.*reduction", r"oil.*shortage",
        ],
        effects={
            "XLE": +0.7,
            "SPY": -0.2, "QQQ": -0.2,
            "AMZN": -0.2,   # logistics cost
            "GLD": +0.1,
        },
    ),
    Theme(
        name="oil_price_drop",
        keywords=[
            r"oil.*plunge", r"oil.*crash", r"crude.*fall",
            r"opec.*increase", r"oil.*glut", r"energy.*selloff",
        ],
        effects={
            "XLE": -0.7,
            "SPY": +0.1,
            "AMZN": +0.2,
            "GLD": -0.1,
        },
    ),
    Theme(
        name="geopolitical_tension",
        keywords=[
            r"war", r"conflict", r"sanctions", r"strait of hormuz",
            r"middle east.*tension", r"ukraine", r"taiwan.*strait",
            r"military.*strike", r"nato",
        ],
        effects={
            "SPY": -0.3, "QQQ": -0.3,
            "GLD": +0.5,
            "TLT": +0.2,   # flight to safety
            "XLE": +0.3,   # supply disruption risk
        },
    ),
    Theme(
        name="dollar_strength",
        keywords=[
            r"dollar.*strong", r"usd.*rally", r"dollar.*surge",
            r"dxy.*high", r"greenback.*rises?",
        ],
        effects={
            "GLD": -0.4,   # gold inversely correlated to dollar
            "AMZN": -0.1,
            "GOOGL": -0.1,
            "MSFT": -0.1,  # overseas revenue headwind
        },
    ),
    Theme(
        name="dollar_weakness",
        keywords=[
            r"dollar.*weak", r"usd.*fall", r"dollar.*decline",
            r"dxy.*low", r"greenback.*drop",
        ],
        effects={
            "GLD": +0.4,
            "AMZN": +0.1,
            "GOOGL": +0.1,
            "MSFT": +0.1,
        },
    ),
    Theme(
        name="tech_earnings_beat",
        keywords=[
            r"beats? estimates?", r"beats? expectations?",
            r"earnings.*beat", r"eps.*above", r"revenue.*surpass",
        ],
        effects={
            "AAPL": +0.3, "MSFT": +0.3, "GOOGL": +0.3,
            "AMZN": +0.3, "NVDA": +0.4, "QQQ": +0.2, "SPY": +0.1,
        },
    ),
    Theme(
        name="tech_earnings_miss",
        keywords=[
            r"misses? estimates?", r"misses? expectations?",
            r"earnings.*miss", r"eps.*below", r"guidance.*cut",
            r"revenue.*disappoint",
        ],
        effects={
            "AAPL": -0.3, "MSFT": -0.3, "GOOGL": -0.3,
            "AMZN": -0.3, "NVDA": -0.4, "QQQ": -0.2, "SPY": -0.1,
        },
    ),
    Theme(
        name="bank_stress",
        keywords=[
            r"bank.*fail", r"banking.*crisis", r"credit crunch",
            r"liquidity.*crisis", r"svb", r"bank run",
        ],
        effects={
            "XLF": -0.8,
            "SPY": -0.4, "QQQ": -0.3,
            "TLT": +0.4,
            "GLD": +0.3,
        },
    ),
    Theme(
        name="ai_boom",
        keywords=[
            r"ai.*breakthrough", r"artificial intelligence.*boom",
            r"chatgpt", r"llm.*demand", r"gpu.*demand",
            r"nvidia.*order", r"ai chip",
        ],
        effects={
            "NVDA": +0.7,
            "MSFT": +0.4,
            "GOOGL": +0.3,
            "AMZN": +0.2,
            "QQQ": +0.2,
        },
    ),
]

# Default ticker queries for NewsAPI (what to search when scanning a ticker)
TICKER_QUERIES: dict[str, str] = {
    "AAPL": "Apple stock",
    "MSFT": "Microsoft stock",
    "GOOGL": "Google Alphabet stock",
    "AMZN": "Amazon stock",
    "NVDA": "Nvidia stock",
    "SPY": "S&P 500 market",
    "QQQ": "Nasdaq tech market",
    "GLD": "gold price",
    "TLT": "treasury bonds yield",
    "XLE": "energy sector oil",
    "XLF": "financial sector banks",
    "IDVY": "European dividend stocks",
    "IGLN": "gold bullion physical gold",
    "JEDI": "space innovation satellite",
    "WQTM": "quantum computing technology",
    "RBOT": "robotics automation industry",
    "EURN": "Euronav crude oil tanker shipping",
    "HEAL.UK": "healthcare innovation biotech",
    "URNU.DE": "uranium mining nuclear energy",
    "HYCN.DE": "hydrogen energy fuel cell",
    "DTG.DE": "Daimler Truck",
    "CEBT.DE": "essential metals mining copper lithium",
    "IQQQ.DE": "water infrastructure utility",
    "DFEN.DE": "defense sector military spending",
}


@dataclass
class SentimentResult:
    ticker: str
    score: float            # net sentiment -1 to +1
    matched_themes: list[str]
    headline_count: int


def score_macro_headlines(headlines: list[str], tickers: list[str]) -> dict[str, float]:
    """Apply generic financial headlines to all tickers using the theme effects table.

    Macro headlines (Fed decisions, inflation prints, geopolitical events) affect
    every ticker differently according to the theme→ticker effects table. This is
    the right tool for market-wide news; use score_headlines_neural() for
    company-specific news.

    Returns a dict of ticker → score in [-1, +1], omitting tickers with zero effect.
    """
    result = {}
    for ticker in tickers:
        r = score_headlines(ticker, headlines)
        if r.score != 0.0:
            result[ticker] = r.score
    return result


def blend_sentiment(ticker_score: float | None, macro_score: float | None,
                    ticker_weight: float = 0.6) -> float | None:
    """Blend ticker-specific and macro sentiment scores.

    If only one source is available, returns it directly (no blending penalty).
    """
    if ticker_score is not None and macro_score is not None:
        return round(ticker_weight * ticker_score + (1 - ticker_weight) * macro_score, 4)
    return ticker_score if ticker_score is not None else macro_score


def score_headlines_neural(ticker: str, headlines: list[str]) -> SentimentResult:
    """Score headlines using the trained neural embedder (Phase 3b).

    Falls back to keyword tagger if the embedder model is not available.
    """
    from trading_copilot.sentiment.embedder import load_embedder, embedder_exists

    if not embedder_exists():
        return score_headlines(ticker, headlines)

    embedder = load_embedder()
    if embedder is None:
        return score_headlines(ticker, headlines)

    score = embedder.predict_score(ticker, headlines)
    return SentimentResult(
        ticker=ticker,
        score=score,
        matched_themes=["neural"],
        headline_count=len(headlines),
    )


def score_headlines(ticker: str, headlines: list[str]) -> SentimentResult:
    """Score a list of headline strings for a given ticker.

    Returns a SentimentResult with a net score in [-1, +1].
    """
    if not headlines:
        return SentimentResult(ticker=ticker, score=0.0, matched_themes=[], headline_count=0)

    theme_scores: dict[str, float] = {}

    for headline in headlines:
        for theme in THEMES:
            if theme.matches(headline):
                effect = theme.effects.get(ticker, 0.0)
                if effect != 0.0:
                    # Accumulate — multiple headlines on same theme strengthen signal
                    theme_scores[theme.name] = theme_scores.get(theme.name, 0.0) + effect

    if not theme_scores:
        return SentimentResult(ticker=ticker, score=0.0, matched_themes=[], headline_count=len(headlines))

    # Normalise: cap each theme's contribution, then average across *matched* themes.
    # Dividing by total themes would dilute to near-zero even for strong signals.
    capped = {t: max(-1.0, min(1.0, s)) for t, s in theme_scores.items()}
    net_score = sum(capped.values()) / len(capped)
    net_score = max(-1.0, min(1.0, net_score))

    return SentimentResult(
        ticker=ticker,
        score=round(net_score, 4),
        matched_themes=sorted(theme_scores.keys()),
        headline_count=len(headlines),
    )
