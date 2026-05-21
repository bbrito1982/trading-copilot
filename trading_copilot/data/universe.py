"""Broad universe of tickers for the discovery screener."""

# Major broad-market ETFs always included
CORE_ETFS = [
    "SPY", "QQQ", "IWM", "DIA", "VTI",
    "GLD", "SLV", "TLT", "HYG", "LQD",
    "USO", "XLE", "XLF", "XLK", "XLV",
    "XLI", "XLP", "XLU", "XLB", "XLRE",
    "IAU", "GDX", "EEM", "EFA",
]

# S&P 500 large-cap subset for screening (top ~100 by market cap)
SP500_LARGE_CAP = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "TSLA",
    "BRK-B", "JPM", "UNH", "XOM", "V", "LLY", "JNJ", "MA", "PG",
    "AVGO", "HD", "MRK", "CVX", "ABBV", "COST", "ORCL", "PEP",
    "ADBE", "AMD", "CRM", "KO", "MCD", "BAC", "WMT", "ACN", "LIN",
    "TMO", "CSCO", "ABT", "DHR", "TXN", "NEE", "NFLX", "PM",
    "INTU", "WFC", "AMGN", "IBM", "QCOM", "RTX", "GE", "SPGI",
    "CAT", "AMAT", "GS", "BLK", "ISRG", "SYP", "MDT", "AXP",
    "BKNG", "T", "LOW", "GILD", "DE", "VRTX", "ADI", "ADP",
    "CVS", "REGN", "SLB", "CI", "MO", "MDLZ", "MMC", "TJX",
    "BSX", "ETN", "CB", "PANW", "LRCX", "SNPS", "ZTS", "PLD",
    "SO", "DUK", "CL", "PYPL", "BDX", "EQIX", "ITW", "APD",
    "CME", "AON", "FI", "ECL", "HCA", "ICE", "NOC", "GD",
]

FULL_UNIVERSE = list(dict.fromkeys(CORE_ETFS + SP500_LARGE_CAP))
