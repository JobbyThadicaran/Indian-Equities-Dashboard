"""
config.py — Central Configuration for Indian Long/Short Equity Research System
==============================================================================
Defines fallback universes, factor weights, market metadata, date ranges,
and all tunable system parameters.
"""

from datetime import datetime, timedelta


# ============================================================================
# DATE CONFIGURATION
# ============================================================================
def get_date_range():
    """Return (start_date, end_date, lookback_1y) computed at call time."""
    end = datetime.now()
    return (
        (end - timedelta(days=400)).strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        (end - timedelta(days=365)).strftime("%Y-%m-%d"),
    )


# ============================================================================
# MARKET / UNIVERSE METADATA
# ============================================================================
MARKET_NAME = "Indian"
UNIVERSE_DESCRIPTION = "NIFTY 50 + F&O-Eligible Stocks"
SHORT_BOOK_DESCRIPTION = "Short book restricted to F&O-eligible stocks"

NIFTY_50_FALLBACK = [
    "ADANIENT",
    "ADANIPORTS",
    "APOLLOHOSP",
    "ASIANPAINT",
    "AXISBANK",
    "BAJAJ-AUTO",
    "BAJFINANCE",
    "BAJAJFINSV",
    "BEL",
    "BHARTIARTL",
    "BPCL",
    "BRITANNIA",
    "CIPLA",
    "COALINDIA",
    "DRREDDY",
    "EICHERMOT",
    "ETERNAL",
    "GRASIM",
    "HCLTECH",
    "HDFCBANK",
    "HINDALCO",
    "HINDUNILVR",
    "ICICIBANK",
    "INDUSINDBK",
    "INFY",
    "ITC",
    "JSWSTEEL",
    "KOTAKBANK",
    "LT",
    "M&M",
    "MARUTI",
    "NESTLEIND",
    "NTPC",
    "ONGC",
    "POWERGRID",
    "RELIANCE",
    "SBI",
    "SBILIFE",
    "SHRIRAMFIN",
    "SUNPHARMA",
    "TATACONSUM",
    "TATAMOTORS",
    "TATASTEEL",
    "TCS",
    "TECHM",
    "TITAN",
    "TRENT",
    "ULTRACEMCO",
    "WIPRO",
    "BAJAJHLDNG",
]

SECTOR_FALLBACK = {
    "ADANIENT": "Conglomerate",
    "ADANIPORTS": "Logistics",
    "APOLLOHOSP": "Healthcare",
    "ASIANPAINT": "Consumer",
    "AXISBANK": "Financials",
    "BAJAJ-AUTO": "Automobile",
    "BAJFINANCE": "Financials",
    "BAJAJFINSV": "Financials",
    "BEL": "Capital Goods",
    "BHARTIARTL": "Telecom",
    "BPCL": "Energy",
    "BRITANNIA": "FMCG",
    "CIPLA": "Healthcare",
    "COALINDIA": "Mining",
    "DRREDDY": "Healthcare",
    "EICHERMOT": "Automobile",
    "ETERNAL": "Consumer Services",
    "GRASIM": "Materials",
    "HCLTECH": "IT",
    "HDFCBANK": "Financials",
    "HINDALCO": "Metals",
    "HINDUNILVR": "FMCG",
    "ICICIBANK": "Financials",
    "INDUSINDBK": "Financials",
    "INFY": "IT",
    "ITC": "FMCG",
    "JSWSTEEL": "Metals",
    "KOTAKBANK": "Financials",
    "LT": "Capital Goods",
    "M&M": "Automobile",
    "MARUTI": "Automobile",
    "NESTLEIND": "FMCG",
    "NTPC": "Power",
    "ONGC": "Energy",
    "POWERGRID": "Power",
    "RELIANCE": "Energy",
    "SBI": "Financials",
    "SBILIFE": "Financials",
    "SHRIRAMFIN": "Financials",
    "SUNPHARMA": "Healthcare",
    "TATACONSUM": "FMCG",
    "TATAMOTORS": "Automobile",
    "TATASTEEL": "Metals",
    "TCS": "IT",
    "TECHM": "IT",
    "TITAN": "Consumer Durables",
    "TRENT": "Retail",
    "ULTRACEMCO": "Materials",
    "WIPRO": "IT",
}

# Fallback default used when live/public universe discovery is unavailable.
FULL_UNIVERSE = [f"{symbol}.NS" for symbol in NIFTY_50_FALLBACK]


# ============================================================================
# INDEX BENCHMARKS
# ============================================================================
INDEX_TICKERS = {
    "NIFTY 50": "^NSEI",
    "NIFTY Bank": "^NSEBANK",
    "BSE Sensex": "^BSESN",
}
INDEX_QUOTE_KEYS = {
    "NIFTY 50": "NSE:NIFTY 50",
    "NIFTY Bank": "NSE:NIFTY BANK",
    "BSE Sensex": "BSE:SENSEX",
}
GIFT_NIFTY_SPOT_URL = "https://www1.nseix.com/api/nifty-market-rate"
GIFT_NIFTY_FUTURES_URL = "https://www1.nseix.com/api/market-rate?type=derivative"
LIVE_REFRESH_SECONDS = 300
ZERODHA_QUOTE_BATCH_SIZE = 200

COUNTRY_MAP = {ticker: "India" for ticker in FULL_UNIVERSE}


# ============================================================================
# FACTOR MODEL WEIGHTS
# ============================================================================
FACTOR_WEIGHTS = {
    "pe_rank": 0.12,
    "ev_ebitda_rank": 0.12,
    "fcf_yield_rank": 0.11,
    "roic_rank": 0.12,
    "ebitda_margin_rank": 0.12,
    "leverage_rank": 0.11,
    "mom_3m_rank": 0.15,
    "mom_6m_rank": 0.15,
}

LONG_THRESHOLD_PERCENTILE = 90
SHORT_THRESHOLD_PERCENTILE = 10


# ============================================================================
# SENTIMENT ANALYSIS CONFIGURATION
# ============================================================================
RSS_FEEDS = {
    "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
    "Economic Times Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "Mint Markets": "https://www.livemint.com/rss/markets",
    "Investing.com": "https://www.investing.com/rss/news.rss",
}

EVENT_KEYWORDS = {
    "earnings_beat": [
        "earnings beat",
        "beat estimates",
        "beat expectations",
        "better than expected",
        "topped estimates",
        "surpassed expectations",
    ],
    "earnings_miss": [
        "earnings miss",
        "missed estimates",
        "missed expectations",
        "worse than expected",
        "fell short",
        "below expectations",
    ],
    "guidance_upgrade": [
        "guidance upgrade",
        "raised guidance",
        "raised outlook",
        "upgraded forecast",
        "lifted guidance",
        "raised full-year",
    ],
    "guidance_downgrade": [
        "guidance downgrade",
        "lowered guidance",
        "cut outlook",
        "downgraded forecast",
        "reduced guidance",
        "lowered full-year",
    ],
    "margin_expansion": [
        "margin expansion",
        "margin improvement",
        "widening margins",
        "margin beat",
        "improving profitability",
    ],
    "profit_warning": [
        "profit warning",
        "earnings warning",
        "revenue warning",
        "issued warning",
        "downside warning",
    ],
}


# ============================================================================
# DATA PATHS
# ============================================================================
DATA_DIR = "data"
REPORTS_DIR = "reports"
CACHE_EXPIRY_HOURS = 12


# ============================================================================
# PDF REPORT SETTINGS
# ============================================================================
REPORT_TITLE = "Indian Long/Short Equity Report"
REPORT_SUBTITLE = "NIFTY 50 + F&O Systematic Factor Research"
TOP_N_LONG = 3
TOP_N_SHORT = 3
