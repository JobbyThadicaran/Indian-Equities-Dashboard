"""
factor_model.py — Multi-Factor Scoring Engine for Long/Short Portfolio Construction
====================================================================================
Implements a systematic factor model combining Value, Quality, and Momentum
factors. Scores the Indian working universe, ranks stocks on a percentile
basis, and constructs long/short portfolios.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

from config import (
    FACTOR_WEIGHTS, LONG_THRESHOLD_PERCENTILE,
    SHORT_THRESHOLD_PERCENTILE
)
from utils import (
    setup_logger, percentile_rank, winsorize,
    save_to_cache, load_from_cache, export_to_csv, ticker_to_name
)

logger = setup_logger("factor_model")

# ============================================================================
# FACTOR COMPUTATION
# ============================================================================

def compute_value_factors(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Value factor scores (lower valuation = better score).
    
    Factors:
        - P/E Ratio (inverted rank — lower P/E → higher score)
        - EV/EBITDA (inverted rank — lower EV/EBITDA → higher score)
        - FCF Yield (direct rank — higher FCF yield → higher score)
    
    Args:
        metrics: DataFrame with 'pe_ratio', 'ev_ebitda', 'fcf_yield' columns
    
    Returns:
        DataFrame with value factor rank columns added
    """
    df = metrics.copy()
    
    # P/E rank — lower is better, so we invert the rank
    # Filter out negative P/E (unprofitable) and extreme values
    pe = pd.to_numeric(df["pe_ratio"], errors="coerce").copy()
    pe = pe.where((pe > 0) & (pe < 200))  # Exclude negative & extreme P/E
    pe = winsorize(pe)
    df["pe_rank"] = 100 - percentile_rank(pe)  # Invert: low P/E → high rank
    
    # EV/EBITDA rank — lower is better, so we invert
    ev_ebitda = pd.to_numeric(df["ev_ebitda"], errors="coerce").copy()
    ev_ebitda = ev_ebitda.where((ev_ebitda > 0) & (ev_ebitda < 100))
    ev_ebitda = winsorize(ev_ebitda)
    df["ev_ebitda_rank"] = 100 - percentile_rank(ev_ebitda)  # Invert: low → high rank
    
    # FCF Yield rank — higher is better, direct rank
    fcf_yield = df["fcf_yield"].copy()
    fcf_yield = winsorize(fcf_yield)
    df["fcf_yield_rank"] = percentile_rank(fcf_yield)
    
    logger.info("Value factors computed (P/E, EV/EBITDA, FCF Yield)")
    return df


def compute_quality_factors(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Quality factor scores (higher quality = better score).
    
    Factors:
        - ROIC proxy (direct rank — higher return on invested capital → higher score)
        - EBITDA Margin (direct rank — higher margins → higher score)
        - Leverage (inverted rank — lower Net Debt/EBITDA → higher score)
    
    Args:
        metrics: DataFrame with 'roic', 'ebitda_margin', 'leverage' columns
    
    Returns:
        DataFrame with quality factor rank columns added
    """
    df = metrics.copy()
    
    # ROIC rank — higher is better
    roic = df["roic"].copy()
    roic = winsorize(roic)
    df["roic_rank"] = percentile_rank(roic)
    
    # EBITDA Margin rank — higher is better
    ebitda_m = df["ebitda_margin"].copy()
    ebitda_m = winsorize(ebitda_m)
    df["ebitda_margin_rank"] = percentile_rank(ebitda_m)
    
    # Leverage rank — lower is better, so we invert
    leverage = df["leverage"].copy()
    leverage = winsorize(leverage)
    df["leverage_rank"] = 100 - percentile_rank(leverage)  # Low leverage → high rank
    
    logger.info("Quality factors computed (ROIC, EBITDA Margin, Leverage)")
    return df


def compute_momentum_factors(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Momentum factor scores (stronger momentum = better score).
    
    Factors:
        - 3-month return (direct rank)
        - 6-month return (direct rank)
    
    Args:
        metrics: DataFrame with 'mom_3m', 'mom_6m' columns
    
    Returns:
        DataFrame with momentum factor rank columns added
    """
    df = metrics.copy()
    
    # 3-month momentum rank — higher return → higher rank
    mom3 = df["mom_3m"].copy()
    mom3 = winsorize(mom3)
    df["mom_3m_rank"] = percentile_rank(mom3)
    
    # 6-month momentum rank
    mom6 = df["mom_6m"].copy()
    mom6 = winsorize(mom6)
    df["mom_6m_rank"] = percentile_rank(mom6)
    
    logger.info("Momentum factors computed (3M, 6M returns)")
    return df


# ============================================================================
# COMPOSITE SCORE
# ============================================================================

def compute_composite_score(
    metrics: pd.DataFrame,
    weights: Dict[str, float] = None
) -> pd.DataFrame:
    """
    Compute the weighted composite factor score for each stock.
    
    The composite score is a weighted sum of all individual factor
    percentile ranks. Higher score → more attractive for LONG position.
    
    Args:
        metrics: DataFrame with all factor rank columns
        weights: Dictionary of factor_name → weight (default: FACTOR_WEIGHTS from config)
    
    Returns:
        DataFrame with 'composite_score' column and factor breakdown
    """
    if weights is None:
        weights = FACTOR_WEIGHTS
    
    df = metrics.copy()
    factor_cols = [factor for factor in weights if factor in df.columns]
    composite = pd.Series(0.0, index=df.index, dtype=float)
    weight_sum = 0.0
    
    if not factor_cols:
        logger.warning("No factor rank columns available; assigning neutral composite scores")
        df["composite_score"] = 50.0
        df["value_score"] = 50.0
        df["quality_score"] = 50.0
        df["momentum_score"] = 50.0
        return df
    
    # Count NaNs across factor columns to flag low-quality data
    nan_counts = df[factor_cols].isna().sum(axis=1)
    heavy_nan_tickers = nan_counts[nan_counts > 3].index.tolist()
    if heavy_nan_tickers:
        logger.warning(f"{len(heavy_nan_tickers)} stocks missing >3 factors (neutral score assigned): {heavy_nan_tickers[:5]}...")
    
    for factor, weight in weights.items():
        if factor in df.columns:
            # Fill NaN factor ranks with 50 (neutral) to avoid penalising missing data
            factor_vals = df[factor].fillna(50)
            composite += factor_vals * weight
            weight_sum += weight
    
    # Normalise by actual weight sum (in case some factors are missing)
    if weight_sum > 0:
        composite = composite / weight_sum  # Scale to 0–100 (since ranks are 0-100)
    
    df["composite_score"] = composite
    
    # Add sub-category scores for breakdown display
    value_cols = ["pe_rank", "ev_ebitda_rank", "fcf_yield_rank"]
    quality_cols = ["roic_rank", "ebitda_margin_rank", "leverage_rank"]
    momentum_cols = ["mom_3m_rank", "mom_6m_rank"]
    
    for label, cols in [("value_score", value_cols), ("quality_score", quality_cols), ("momentum_score", momentum_cols)]:
        available = [c for c in cols if c in df.columns]
        if available:
            df[label] = df[available].fillna(50).mean(axis=1)
    
    logger.info(f"Composite scores computed for {len(df)} stocks")
    logger.info(f"  Score range: {composite.min():.1f} — {composite.max():.1f}")
    logger.info(f"  Mean: {composite.mean():.1f}, Median: {composite.median():.1f}")
    
    return df


# ============================================================================
# PORTFOLIO CONSTRUCTION
# ============================================================================

def construct_portfolio(
    scored_df: pd.DataFrame,
    long_pct: float = LONG_THRESHOLD_PERCENTILE,
    short_pct: float = SHORT_THRESHOLD_PERCENTILE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Construct long/short portfolios from scored universe.
    
    Top percentile stocks → LONG book
    Bottom percentile stocks → SHORT book
    
    Args:
        scored_df: DataFrame with 'composite_score' column
        long_pct: Percentile threshold for long positions (default: 90 → top 10%)
        short_pct: Percentile threshold for short positions (default: 10 → bottom 10%)
    
    Returns:
        Tuple of (full_df, long_book, short_book)
    """
    df = scored_df.copy()
    
    # Filter out stocks without a valid score
    valid = df.dropna(subset=["composite_score"])
    
    if len(valid) == 0:
        logger.error("No valid composite scores — cannot construct portfolio")
        return df, pd.DataFrame(), pd.DataFrame()
    
    # Determine thresholds
    long_threshold = np.percentile(valid["composite_score"], long_pct)
    short_candidates = valid
    if "is_fo_eligible" in valid.columns:
        eligible = valid[valid["is_fo_eligible"].fillna(False)]
        if not eligible.empty:
            short_candidates = eligible
        else:
            logger.warning("No F&O-eligible names available; short book left empty")
            short_candidates = pd.DataFrame(columns=valid.columns)

    # Assign portfolio positions
    df["position"] = "NEUTRAL"
    df.loc[df["composite_score"] >= long_threshold, "position"] = "LONG"
    if not short_candidates.empty:
        short_threshold = np.percentile(short_candidates["composite_score"], short_pct)
        short_names = short_candidates[short_candidates["composite_score"] <= short_threshold].index
        df.loc[df.index.isin(short_names), "position"] = "SHORT"
    else:
        short_threshold = np.nan
    
    long_book = df[df["position"] == "LONG"].sort_values("composite_score", ascending=False)
    short_book = df[df["position"] == "SHORT"].sort_values("composite_score", ascending=True)
    
    logger.info(f"Portfolio constructed:")
    logger.info(f"  LONG:    {len(long_book)} positions (score ≥ {long_threshold:.1f})")
    if not np.isnan(short_threshold):
        logger.info(f"  SHORT:   {len(short_book)} positions (score ≤ {short_threshold:.1f})")
    else:
        logger.info("  SHORT:   0 positions (no F&O-eligible short universe)")
    logger.info(f"  NEUTRAL: {len(df) - len(long_book) - len(short_book)} positions")
    
    return df, long_book, short_book


# ============================================================================
# MAIN FACTOR MODEL PIPELINE
# ============================================================================

def run_factor_model(metrics: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Execute the full factor model pipeline:
    1. Compute Value factors
    2. Compute Quality factors
    3. Compute Momentum factors
    4. Compute Composite score
    5. Construct Long/Short portfolio
    
    Args:
        metrics: DataFrame from data_ingestion.extract_key_metrics()
    
    Returns:
        Tuple of (scored_universe, long_book, short_book)
    """
    logger.info("=" * 60)
    
    if metrics.empty:
        logger.warning("Metrics DataFrame is empty — returning empty results")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Step 1: Compute all factor rankings
    df = compute_value_factors(metrics)
    df = compute_quality_factors(df)
    df = compute_momentum_factors(df)
    
    # Step 2: Composite score
    df = compute_composite_score(df)
    
    # Step 3: Portfolio construction
    scored_universe, long_book, short_book = construct_portfolio(df)
    
    # Cache results
    save_to_cache("scored_universe", scored_universe)
    save_to_cache("long_book", long_book)
    save_to_cache("short_book", short_book)
    
    logger.info("=" * 60)
    logger.info("FACTOR MODEL COMPLETE")
    logger.info("=" * 60)
    
    return scored_universe, long_book, short_book


# ============================================================================
# RELATIVE TRADE SUGGESTIONS
# ============================================================================

def suggest_relative_trades(
    long_book: pd.DataFrame,
    short_book: pd.DataFrame,
    n_pairs: int = 5
) -> List[Dict]:
    """
    Suggest relative value (pair) trades: Long X / Short Y.
    
    Pairs are formed by matching stocks in the same sector
    or, if not possible, by largest score differential.
    
    Args:
        long_book: DataFrame of LONG positions
        short_book: DataFrame of SHORT positions
        n_pairs: Number of pairs to suggest
    
    Returns:
        List of dicts with keys: 'long_ticker', 'short_ticker',
            'long_score', 'short_score', 'score_spread', 'sector'
    """
    required_cols = {"composite_score"}
    if (
        long_book is None
        or short_book is None
        or long_book.empty
        or short_book.empty
        or not required_cols.issubset(long_book.columns)
        or not required_cols.issubset(short_book.columns)
    ):
        return []

    pairs = []
    used_long = set()
    used_short = set()
    
    # First pass: sector-matched pairs
    if "sector" in long_book.columns and "sector" in short_book.columns:
        for sector in long_book["sector"].unique():
            sector_longs = long_book[
                (long_book["sector"] == sector) &
                (~long_book.index.isin(used_long))
            ]
            sector_shorts = short_book[
                (short_book["sector"] == sector) &
                (~short_book.index.isin(used_short))
            ]
            
            if not sector_longs.empty and not sector_shorts.empty:
                l_ticker = sector_longs.index[0]
                s_ticker = sector_shorts.index[0]
                pairs.append({
                    "long_ticker": l_ticker,
                    "long_name": sector_longs.loc[l_ticker].get("name", ticker_to_name(l_ticker)),
                    "short_ticker": s_ticker,
                    "short_name": sector_shorts.loc[s_ticker].get("name", ticker_to_name(s_ticker)),
                    "long_score": sector_longs.loc[l_ticker, "composite_score"],
                    "short_score": sector_shorts.loc[s_ticker, "composite_score"],
                    "score_spread": (
                        sector_longs.loc[l_ticker, "composite_score"] -
                        sector_shorts.loc[s_ticker, "composite_score"]
                    ),
                    "sector": sector,
                })
                used_long.add(l_ticker)
                used_short.add(s_ticker)
    
    # Second pass: fill remaining with highest spread
    remaining_longs = long_book[~long_book.index.isin(used_long)].sort_values("composite_score", ascending=False)
    remaining_shorts = short_book[~short_book.index.isin(used_short)].sort_values("composite_score", ascending=True)
    
    for i in range(min(len(remaining_longs), len(remaining_shorts))):
        if len(pairs) >= n_pairs:
            break
        
        l_ticker = remaining_longs.index[i]
        s_ticker = remaining_shorts.index[i]
        
        pairs.append({
            "long_ticker": l_ticker,
            "long_name": remaining_longs.loc[l_ticker].get("name", ticker_to_name(l_ticker)),
            "short_ticker": s_ticker,
            "short_name": remaining_shorts.loc[s_ticker].get("name", ticker_to_name(s_ticker)),
            "long_score": remaining_longs.loc[l_ticker, "composite_score"],
            "short_score": remaining_shorts.loc[s_ticker, "composite_score"],
            "score_spread": (
                remaining_longs.loc[l_ticker, "composite_score"] -
                remaining_shorts.loc[s_ticker, "composite_score"]
            ),
            "sector": remaining_longs.loc[l_ticker].get("sector", "Cross-Sector"),
        })
    
    # Sort by widest score spread
    pairs.sort(key=lambda x: x["score_spread"], reverse=True)
    
    return pairs[:n_pairs]


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    from data_ingestion import run_data_pipeline
    
    # Run data pipeline
    price_data, financial_data, metrics_df, index_data = run_data_pipeline()
    
    # Run factor model
    scored, longs, shorts = run_factor_model(metrics_df)
    
    # Display results
    print("\n" + "=" * 60)
    print("TOP LONG IDEAS")
    print("=" * 60)
    display_cols = ["name", "composite_score", "value_score", "quality_score", "momentum_score"]
    available_cols = [c for c in display_cols if c in longs.columns]
    print(longs[available_cols].head(10).to_string())
    
    print("\n" + "=" * 60)
    print("TOP SHORT IDEAS")
    print("=" * 60)
    available_cols = [c for c in display_cols if c in shorts.columns]
    print(shorts[available_cols].head(10).to_string())
    
    # Relative trades
    pairs = suggest_relative_trades(longs, shorts)
    print("\n" + "=" * 60)
    print("RELATIVE TRADE SUGGESTIONS")
    print("=" * 60)
    for i, pair in enumerate(pairs, 1):
        print(f"  {i}. Long {pair['long_name']} / Short {pair['short_name']}"
              f"  (spread: {pair['score_spread']:.1f}, sector: {pair['sector']})")
    
    # Export
    path = export_to_csv(scored, "scored_universe")
    print(f"\nScored universe exported to: {path}")
