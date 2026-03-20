"""
sentiment_analysis.py — News Fetching & Sentiment Analysis Module
=================================================================
Fetches financial news via RSS feeds, performs VADER sentiment analysis,
and detects key event keywords (earnings beat/miss, guidance changes, etc.)
for each stock in the European equity universe.
"""

import re
import time
from datetime import datetime, timedelta
from urllib.parse import quote
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import feedparser
import requests
from bs4 import BeautifulSoup

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer as NltkVader
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

from config import RSS_FEEDS, EVENT_KEYWORDS, FULL_UNIVERSE, CACHE_EXPIRY_HOURS
from utils import (
    setup_logger, save_to_cache, load_from_cache,
    ticker_to_name, clean_ticker
)

logger = setup_logger("sentiment")
SENTIMENT_CACHE_KEY = "sentiment_scores_v2"
TICKER_SENTIMENT_CACHE_PREFIX = "ticker_sentiment_v2"
CORPORATE_SUFFIXES = {
    "adr", "ag", "co", "company", "corp", "corporation", "di", "group",
    "holding", "holdings", "inc", "limited", "ltd", "nv", "ordinary",
    "ord", "plc", "s", "sa", "se", "shares", "spa", "stock", "the",
}
GENERIC_SINGLE_WORD_NAMES = {"next"}

# ============================================================================
# SENTIMENT ANALYZER INITIALISATION
# ============================================================================

def get_analyzer():
    """
    Initialise the best available sentiment analyzer.
    Priority: vaderSentiment → nltk VADER
    
    Returns:
        SentimentIntensityAnalyzer instance
    """
    if VADER_AVAILABLE:
        logger.info("Using vaderSentiment for sentiment analysis")
        return SentimentIntensityAnalyzer()
    elif NLTK_AVAILABLE:
        try:
            nltk.download("vader_lexicon", quiet=True)
            logger.info("Using NLTK VADER for sentiment analysis")
            return NltkVader()
        except Exception:
            pass
    
    logger.warning("No sentiment analyzer available — install vaderSentiment or nltk")
    return None


# ============================================================================
# NEWS FETCHING
# ============================================================================

def fetch_rss_news(
    feeds: Dict[str, str] = None,
    max_articles_per_feed: int = 50
) -> List[Dict]:
    """
    Fetch news articles from RSS feeds.
    
    Args:
        feeds: Dictionary mapping feed name → RSS URL
        max_articles_per_feed: Maximum articles to process per feed
    
    Returns:
        List of article dicts with keys: 'title', 'summary', 'link',
            'source', 'published', 'full_text'
    """
    if feeds is None:
        feeds = RSS_FEEDS
    
    logger.info(f"Fetching news from {len(feeds)} RSS feeds")
    articles = []
    
    for source_name, url in feeds.items():
        try:
            # Use requests with a browser-like User-Agent to avoid blocks/429s
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code != 200:
                logger.warning(f"  ✗ {source_name}: HTTP {response.status_code}")
                continue
                
            feed = feedparser.parse(response.content)
            count = 0
            
            for entry in feed.entries[:max_articles_per_feed]:
                article = {
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", entry.get("description", "")),
                    "link": entry.get("link", ""),
                    "source": source_name,
                    "published": entry.get("published", ""),
                }
                
                # Clean HTML from summary
                if article["summary"]:
                    soup = BeautifulSoup(article["summary"], "html.parser")
                    article["summary"] = soup.get_text(strip=True)
                
                # Combine title + summary for analysis
                article["full_text"] = f"{article['title']}. {article['summary']}"
                
                articles.append(article)
                count += 1
            
            logger.info(f"  ✓ {source_name}: {count} articles")
            
        except Exception as e:
            logger.warning(f"  ✗ {source_name}: {e}")
    
    logger.info(f"Total articles fetched: {len(articles)}")
    return articles


def fetch_ticker_news(
    ticker: str,
    max_articles: int = 20,
    ticker_name: str = None
) -> List[Dict]:
    """
    Fetch news specifically for a single ticker using yfinance and RSS.
    
    Args:
        ticker: Yahoo Finance ticker symbol
        max_articles: Maximum number of articles to retrieve
    
    Returns:
        List of article dicts
    """
    articles = []
    
    # Method 1: Use yfinance news feed
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        news = stock.news or []
        
        for item in news[:max_articles]:
            articles.append({
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "link": item.get("link", ""),
                "source": item.get("publisher", "yfinance"),
                "published": datetime.fromtimestamp(
                    item.get("providerPublishTime", 0)
                ).strftime("%Y-%m-%d %H:%M") if item.get("providerPublishTime") else "",
                "full_text": f"{item.get('title', '')}. {item.get('summary', '')}",
            })
    except Exception as e:
        logger.debug(f"yfinance news fetch failed for {ticker}: {e}")
    
    # Method 2: Google News RSS
    try:
        company_name = build_search_name(ticker_name or ticker_to_name(ticker), ticker)
        search_query = f"\"{company_name}\" stock"
        google_rss_url = f"https://news.google.com/rss/search?q={quote(search_query)}+when:7d&hl=en"
        
        # Google News blocks direct feedparser requests, use requests with User-Agent
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        res = requests.get(google_rss_url, headers=headers, timeout=10)
        feed = feedparser.parse(res.content)
        
        if not feed.entries:
            logger.warning(f"Google News returned 0 entries for {ticker} (possible block/rate limit)")
            
        for entry in feed.entries[:max_articles]:
            articles.append({
                "title": entry.get("title", ""),
                "summary": entry.get("summary", ""),
                "link": entry.get("link", ""),
                "source": "Google News",
                "published": entry.get("published", ""),
                "full_text": f"{entry.get('title', '')}. {entry.get('summary', '')}",
            })
    except Exception as e:
        logger.debug(f"Google News fetch failed for {ticker}: {e}")
    
    filtered = [
        article for article in dedupe_articles(articles)
        if article_matches_ticker(
            article,
            ticker,
            ticker_name=ticker_name,
            allow_generic_single_word=True
        )
    ]
    return filtered


# ============================================================================
# SENTIMENT SCORING
# ============================================================================

def score_sentiment(
    text: str,
    analyzer=None
) -> Dict[str, float]:
    """
    Compute VADER sentiment scores for a given text.
    
    Args:
        text: Input text (headline or article)
        analyzer: Pre-initialised SentimentIntensityAnalyzer
    
    Returns:
        Dict with keys: 'neg', 'neu', 'pos', 'compound'
    """
    if analyzer is None or not text:
        return {"neg": 0, "neu": 1, "pos": 0, "compound": 0}
    
    try:
        scores = analyzer.polarity_scores(text)
        return scores
    except Exception:
        return {"neg": 0, "neu": 1, "pos": 0, "compound": 0}


def detect_events(text: str) -> Dict[str, bool]:
    """
    Detect key financial event keywords in text.
    
    Checks for: earnings beat/miss, guidance upgrade/downgrade,
    margin expansion, profit warning.
    
    Args:
        text: Input text to search
    
    Returns:
        Dict mapping event_type → boolean (True if detected)
    """
    text_lower = text.lower()
    events = {}
    
    for event_type, keywords in EVENT_KEYWORDS.items():
        events[event_type] = any(kw.lower() in text_lower for kw in keywords)
    
    return events


def normalise_text(text: str) -> str:
    """Lowercase and strip punctuation for matching/search."""
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def extract_company_tokens(name: str) -> List[str]:
    """Reduce noisy quote/security names to core company tokens."""
    tokens = []
    for token in normalise_text(name).split():
        if token in CORPORATE_SUFFIXES:
            continue
        if token.isdigit():
            continue
        if re.fullmatch(r"[a-z]*\d+[a-z]*", token):
            continue
        if len(token) < 2:
            continue
        tokens.append(token)
    return tokens


def build_search_name(name: str, ticker: str) -> str:
    """Build a cleaner company name for ticker-specific news search."""
    raw_tokens = [
        token for token in normalise_text(name).split()
        if token and not token.isdigit() and token not in {"ord", "ordinary", "shares", "stock"}
    ]
    tokens = extract_company_tokens(name)
    if len(tokens) == 1 and tokens[0] in GENERIC_SINGLE_WORD_NAMES and len(raw_tokens) >= 2:
        return " ".join(raw_tokens[:2])
    if tokens:
        return " ".join(tokens[:4])
    return clean_ticker(ticker)


def build_ticker_patterns(
    ticker: str,
    ticker_name: str = None,
    allow_generic_single_word: bool = False
) -> List[str]:
    """Build conservative article-match patterns to reduce false positives."""
    base_ticker = clean_ticker(ticker).lower()
    patterns = []
    raw_name = ticker_name or ticker_to_name(ticker)
    raw_tokens = [
        token for token in normalise_text(raw_name).split()
        if token and not token.isdigit() and token not in {"ord", "ordinary", "shares", "stock"}
    ]
    tokens = extract_company_tokens(raw_name)

    if len(tokens) >= 2:
        patterns.append(" ".join(tokens[:4]))
        patterns.append(" ".join(tokens[:2]))
    elif len(tokens) == 1:
        token = tokens[0]
        if allow_generic_single_word and token in GENERIC_SINGLE_WORD_NAMES and len(raw_tokens) >= 2:
            patterns.append(" ".join(raw_tokens[:2]))
        if len(token) >= 5 or allow_generic_single_word:
            if allow_generic_single_word or token not in GENERIC_SINGLE_WORD_NAMES:
                patterns.append(token)

    if len(base_ticker) >= 4:
        if allow_generic_single_word or base_ticker not in GENERIC_SINGLE_WORD_NAMES:
            patterns.append(base_ticker)

    deduped = []
    seen = set()
    for pattern in patterns:
        if pattern and pattern not in seen:
            deduped.append(pattern)
            seen.add(pattern)
    return deduped


def dedupe_articles(articles: List[Dict]) -> List[Dict]:
    """Remove duplicate articles by link/title while preserving order."""
    unique = []
    seen = set()

    for article in articles:
        if not article.get("title") and not article.get("summary"):
            continue
        key = (
            article.get("link", "").strip(),
            normalise_text(article.get("title", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(article)

    return unique


def article_matches_ticker(
    article: Dict,
    ticker: str,
    ticker_name: str = None,
    allow_generic_single_word: bool = False
) -> bool:
    """Check if an article text/title matches the intended company."""
    patterns = build_ticker_patterns(
        ticker,
        ticker_name=ticker_name,
        allow_generic_single_word=allow_generic_single_word
    )
    if not patterns:
        return True

    title = article.get("title", "").lower()
    text = article.get("full_text", "").lower()
    for pattern in patterns:
        if len(pattern) >= 3 and (
            re.search(r"\b" + re.escape(pattern) + r"\b", title) or
            re.search(r"\b" + re.escape(pattern) + r"\b", text)
        ):
            return True
    return False


def build_sentiment_record(
    ticker: str,
    articles: List[Dict],
    analyzer=None
) -> Dict:
    """Aggregate article-level sentiment into a single ticker record."""
    record = {"ticker": ticker}
    ticker_arts = dedupe_articles(articles)

    if not ticker_arts:
        record["sentiment_score"] = 0.0
        record["sentiment_count"] = 0
        record["sentiment_positive"] = 0.0
        record["sentiment_negative"] = 0.0
        for event_type in EVENT_KEYWORDS:
            record[f"event_{event_type}"] = False
        record["headlines"] = []
        return record

    scores = []
    all_events = defaultdict(bool)
    headlines = []

    for art in ticker_arts:
        text = art.get("full_text", "")
        sent = score_sentiment(text, analyzer)
        scores.append(sent["compound"])

        events = detect_events(text)
        for event_type, detected in events.items():
            if detected:
                all_events[event_type] = True

        if art.get("title"):
            headlines.append({
                "title": art["title"],
                "source": art.get("source", ""),
                "published": art.get("published", ""),
                "link": art.get("link", ""),
                "sentiment": sent["compound"],
            })

    scores_arr = np.array(scores)
    record["sentiment_score"] = float(np.mean(scores_arr))
    record["sentiment_count"] = len(scores)
    record["sentiment_positive"] = float(np.mean(scores_arr > 0.05))
    record["sentiment_negative"] = float(np.mean(scores_arr < -0.05))

    for event_type in EVENT_KEYWORDS:
        record[f"event_{event_type}"] = all_events.get(event_type, False)

    headlines.sort(key=lambda x: abs(x.get("sentiment", 0)), reverse=True)
    record["headlines"] = headlines[:10]
    return record


# ============================================================================
# TICKER MATCHING
# ============================================================================

def match_articles_to_tickers(
    articles: List[Dict],
    tickers: List[str],
    ticker_names: Dict[str, str] = None
) -> Dict[str, List[Dict]]:
    """
    Match news articles to tickers based on company name / ticker mention.
    
    Uses both the raw ticker symbol and the human-readable name mapping
    to associate articles with stocks.
    
    Args:
        articles: List of article dicts
        tickers: List of Yahoo Finance tickers
        ticker_names: Optional mapping from ticker → company name
    
    Returns:
        Dict mapping ticker → list of matched article dicts
    """
    ticker_articles = defaultdict(list)
    
    # Build search patterns for each ticker
    ticker_patterns = {
        ticker: build_ticker_patterns(
            ticker,
            ticker_names.get(ticker) if ticker_names else None
        )
        for ticker in tickers
    }
    
    for article in articles:
        text = article.get("full_text", "").lower()
        title = article.get("title", "").lower()
        
        for ticker, patterns in ticker_patterns.items():
            for pattern in patterns:
                # Use word boundary check to avoid matching substrings (e.g., 'BNP' in 'ABNP')
                if len(pattern) >= 3 and (
                    re.search(r'\b' + re.escape(pattern) + r'\b', title) or 
                    re.search(r'\b' + re.escape(pattern) + r'\b', text)
                ):
                    ticker_articles[ticker].append(article)
                    break
    
    return dict(ticker_articles)


# ============================================================================
# AGGREGATE SENTIMENT PER TICKER
# ============================================================================

def compute_ticker_sentiment(
    tickers: List[str],
    articles: List[Dict] = None,
    fetch_individual: bool = False,
    ticker_names: Dict[str, str] = None
) -> pd.DataFrame:
    """
    Compute aggregated sentiment score and event flags for each ticker.
    
    For each stock, combines sentiment from:
    1. Matched articles from general RSS feeds
    2. Ticker-specific news (yfinance)
    
    Args:
        tickers: List of tickers to analyse
        articles: Pre-fetched general articles (optional)
        fetch_individual: Whether to also fetch per-ticker news
        ticker_names: Optional mapping from ticker → company name
    
    Returns:
        DataFrame with sentiment data indexed by ticker
    """
    cache_key = SENTIMENT_CACHE_KEY
    
    analyzer = get_analyzer()
    
    # Fetch general news if not provided
    if articles is None:
        articles = fetch_rss_news()
    
    # Match articles to tickers
    matched = match_articles_to_tickers(articles, tickers, ticker_names=ticker_names)
    
    logger.info(f"Computing sentiment for {len(tickers)} tickers")
    records = []
    
    for ticker in tickers:
        # Collect all articles for this ticker
        ticker_arts = matched.get(ticker, [])
        
        # Optionally fetch ticker-specific news
        if fetch_individual and len(ticker_arts) < 3:
            try:
                specific = fetch_ticker_news(
                    ticker,
                    max_articles=10,
                    ticker_name=ticker_names.get(ticker) if ticker_names else None
                )
                ticker_arts.extend(specific)
            except Exception as e:
                logger.debug(f"fetch_ticker_news failed for {ticker}: {e}")

        records.append(build_sentiment_record(ticker, ticker_arts, analyzer))
    
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).set_index("ticker")
    
    # Cache results
    save_to_cache(cache_key, df)
    
    logger.info(f"Sentiment computed for {len(df)} tickers")
    logger.info(f"  Avg sentiment: {df['sentiment_score'].mean():.3f}")
    logger.info(f"  Tickers with news: {(df['sentiment_count'] > 0).sum()}")
    
    return df


# ============================================================================
# SENTIMENT SUMMARY
# ============================================================================

def get_sentiment_summary(sentiment_df: pd.DataFrame) -> Dict:
    """
    Generate a high-level sentiment summary across the universe.
    
    Returns:
        Dict with overall market sentiment statistics
    """
    if sentiment_df.empty or "sentiment_count" not in sentiment_df.columns:
        summary = {
            "total_stocks_analysed": 0,
            "stocks_with_news": 0,
            "avg_sentiment": 0.0,
            "median_sentiment": 0.0,
            "most_positive": None,
            "most_negative": None,
            "pct_positive": 0.0,
            "pct_negative": 0.0,
        }
        for event_type in EVENT_KEYWORDS:
            summary[f"count_{event_type}"] = 0
        return summary

    active = sentiment_df[sentiment_df["sentiment_count"] > 0]
    
    # Drop NaNs to safely compute min/max
    scores = active["sentiment_score"].dropna()
    
    summary = {
        "total_stocks_analysed": len(sentiment_df),
        "stocks_with_news": len(active),
        "avg_sentiment": float(active["sentiment_score"].mean()) if len(active) > 0 else 0,
        "median_sentiment": float(active["sentiment_score"].median()) if len(active) > 0 else 0,
        "most_positive": scores.idxmax() if len(scores) > 0 else None,
        "most_negative": scores.idxmin() if len(scores) > 0 else None,
        "pct_positive": float((active["sentiment_score"] > 0.05).mean()) if len(active) > 0 else 0,
        "pct_negative": float((active["sentiment_score"] < -0.05).mean()) if len(active) > 0 else 0,
    }
    
    # Count events
    event_cols = [c for c in sentiment_df.columns if c.startswith("event_")]
    for col in event_cols:
        event_name = col.replace("event_", "")
        summary[f"count_{event_name}"] = int(sentiment_df[col].sum()) if col in sentiment_df.columns else 0
    
    return summary


# ============================================================================
# MAIN SENTIMENT PIPELINE
# ============================================================================

def run_sentiment_pipeline(
    tickers: List[str] = None,
    fetch_individual: bool = False,
    ticker_names: Dict[str, str] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Execute the full sentiment analysis pipeline.
    
    Args:
        tickers: List of tickers (defaults to FULL_UNIVERSE)
        fetch_individual: Fetch per-ticker news (slower but more comprehensive)
        ticker_names: Optional dict mapping ticker -> company name for better matching
    """
    if tickers is None:
        tickers = FULL_UNIVERSE
    
    logger.info("=" * 60)
    logger.info("RUNNING SENTIMENT ANALYSIS PIPELINE")
    logger.info(f"Universe: {len(tickers)} tickers")
    logger.info("=" * 60)
    
    # Check cache first to avoid wasteful RSS fetches
    cached = load_from_cache(SENTIMENT_CACHE_KEY, max_age_hours=CACHE_EXPIRY_HOURS)
    if cached is not None and len(cached) > 0:
        logger.info(f"Loaded sentiment data from cache ({len(cached)} tickers)")
        return cached, get_sentiment_summary(cached)
        
    # Fetch general news
    articles = fetch_rss_news()
    
    # Compute per-ticker sentiment
    sentiment_df = compute_ticker_sentiment(
        tickers, articles, fetch_individual=fetch_individual,
        ticker_names=ticker_names
    )
    
    # Summary
    summary = get_sentiment_summary(sentiment_df)
    
    logger.info("=" * 60)
    logger.info("SENTIMENT ANALYSIS COMPLETE")
    logger.info(f"  Market sentiment: {summary['avg_sentiment']:.3f}")
    logger.info(f"  Positive: {summary['pct_positive']:.0%} | Negative: {summary['pct_negative']:.0%}")
    logger.info("=" * 60)
    
    return sentiment_df, summary


def get_ticker_sentiment_snapshot(
    ticker: str,
    ticker_name: str = None,
    max_articles: int = 10,
    max_age_hours: int = 4
) -> Dict:
    """Fetch and cache a live sentiment snapshot for a single ticker."""
    cache_key = f"{TICKER_SENTIMENT_CACHE_PREFIX}_{clean_ticker(ticker).lower()}"
    cached = load_from_cache(cache_key, max_age_hours=max_age_hours)
    if cached is not None:
        return cached

    analyzer = get_analyzer()
    articles = fetch_ticker_news(
        ticker,
        max_articles=max_articles,
        ticker_name=ticker_name
    )
    snapshot = build_sentiment_record(ticker, articles, analyzer)
    save_to_cache(cache_key, snapshot)
    return snapshot


def get_ticker_sentiment_snapshots(
    tickers: List[str],
    ticker_names: Dict[str, str] = None,
    max_articles: int = 10,
    max_workers: int = 6
) -> pd.DataFrame:
    """Fetch live sentiment snapshots for a small list of priority tickers."""
    if not tickers:
        return pd.DataFrame()

    records = []

    def _fetch_snapshot(ticker: str) -> Dict:
        return get_ticker_sentiment_snapshot(
            ticker,
            ticker_name=ticker_names.get(ticker) if ticker_names else None,
            max_articles=max_articles,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_snapshot, ticker): ticker for ticker in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:
                logger.warning(f"Live sentiment fetch failed for {ticker}: {exc}")

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).set_index("ticker")


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    from utils import export_to_csv
    
    sentiment_df, summary = run_sentiment_pipeline(fetch_individual=False)
    
    print("\n=== SENTIMENT SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    
    print("\n=== TOP POSITIVE SENTIMENT ===")
    top_pos = sentiment_df.nlargest(10, "sentiment_score")
    print(top_pos[["sentiment_score", "sentiment_count"]].to_string())
    
    print("\n=== TOP NEGATIVE SENTIMENT ===")
    top_neg = sentiment_df.nsmallest(10, "sentiment_score")
    print(top_neg[["sentiment_score", "sentiment_count"]].to_string())
    
    # Export
    export_cols = [c for c in sentiment_df.columns if c != "headlines"]
    path = export_to_csv(sentiment_df[export_cols], "sentiment_scores")
    print(f"\nSentiment data exported to: {path}")
