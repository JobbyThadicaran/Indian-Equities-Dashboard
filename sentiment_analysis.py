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
    max_articles: int = 20
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
        company_name = ticker_to_name(ticker)
        search_query = f"{company_name} stock"
        google_rss_url = f"https://news.google.com/rss/search?q={quote(search_query)}+when:7d&hl=en"
        
        feed = feedparser.parse(google_rss_url)
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
    
    return articles


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
    ticker_patterns = {}
    for ticker in tickers:
        name = ticker_names.get(ticker) if ticker_names else ticker_to_name(ticker)
        # Fallback to ticker if name is None or empty
        name_val = str(name) if name else ticker
        clean = clean_ticker(ticker)
        patterns = [name_val.lower(), clean.lower()]
        # Add common variations
        if "." in ticker and len(ticker.split(".")[0]) >= 4:
            patterns.append(ticker.split(".")[0].lower())
        ticker_patterns[ticker] = patterns
    
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
    cache_key = "sentiment_scores"
    
    analyzer = get_analyzer()
    
    # Fetch general news if not provided
    if articles is None:
        articles = fetch_rss_news()
    
    # Match articles to tickers
    matched = match_articles_to_tickers(articles, tickers, ticker_names=ticker_names)
    
    logger.info(f"Computing sentiment for {len(tickers)} tickers")
    records = []
    
    for ticker in tickers:
        record = {"ticker": ticker}
        
        # Collect all articles for this ticker
        ticker_arts = matched.get(ticker, [])
        
        # Optionally fetch ticker-specific news
        if fetch_individual and len(ticker_arts) < 3:
            try:
                specific = fetch_ticker_news(ticker, max_articles=10)
                ticker_arts.extend(specific)
            except Exception as e:
                logger.debug(f"fetch_ticker_news failed for {ticker}: {e}")
        
        if not ticker_arts:
            record["sentiment_score"] = 0.0
            record["sentiment_count"] = 0
            record["sentiment_positive"] = 0.0
            record["sentiment_negative"] = 0.0
            for event_type in EVENT_KEYWORDS:
                record[f"event_{event_type}"] = False
            record["headlines"] = []
            records.append(record)
            continue
        
        # Score each article
        scores = []
        all_events = defaultdict(bool)
        headlines = []
        
        for art in ticker_arts:
            text = art.get("full_text", "")
            
            # Sentiment
            sent = score_sentiment(text, analyzer)
            scores.append(sent["compound"])
            
            # Event detection
            events = detect_events(text)
            for event_type, detected in events.items():
                if detected:
                    all_events[event_type] = True
            
            # Headlines
            if art.get("title"):
                headlines.append({
                    "title": art["title"],
                    "source": art.get("source", ""),
                    "published": art.get("published", ""),
                    "sentiment": sent["compound"],
                })
        
        # Aggregate sentiment
        scores_arr = np.array(scores)
        record["sentiment_score"] = float(np.mean(scores_arr))
        record["sentiment_count"] = len(scores)
        record["sentiment_positive"] = float(np.mean(scores_arr > 0.05))
        record["sentiment_negative"] = float(np.mean(scores_arr < -0.05))
        
        # Event flags
        for event_type in EVENT_KEYWORDS:
            record[f"event_{event_type}"] = all_events.get(event_type, False)
        
        # Top headlines (sorted by absolute sentiment)
        headlines.sort(key=lambda x: abs(x.get("sentiment", 0)), reverse=True)
        record["headlines"] = headlines[:10]
        
        records.append(record)
    
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
    cached = load_from_cache("sentiment_scores", max_age_hours=CACHE_EXPIRY_HOURS)
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
