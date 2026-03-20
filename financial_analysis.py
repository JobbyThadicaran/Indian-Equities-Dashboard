"""
financial_analysis.py — Financial Analysis Module
==================================================
Computes comprehensive financial metrics for each stock in the universe:
Growth, Profitability, Risk, and Valuation analytics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from config import INDEX_TICKERS, START_DATE, END_DATE
from utils import (
    setup_logger, safe_divide, save_to_cache, load_from_cache, export_to_csv
)

logger = setup_logger("financial_analysis")

# ============================================================================
# GROWTH METRICS
# ============================================================================

def compute_growth_metrics(financial_data: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compute year-over-year growth metrics from financial statements.
    
    Metrics:
        - Revenue growth (YoY)
        - EBITDA growth (YoY)
        - Earnings growth (YoY)
        - Revenue CAGR (if multi-year data available)
    
    Args:
        financial_data: Dict from data_ingestion.fetch_financial_data()
    
    Returns:
        DataFrame with growth metrics indexed by ticker
    """
    logger.info("Computing growth metrics")
    records = []
    
    for ticker, data in financial_data.items():
        record = {"ticker": ticker}
        income = data.get("income_stmt", pd.DataFrame())
        
        if not income.empty and income.shape[1] >= 2:
            latest = income.columns[0]
            prev = income.columns[1]
            
            # Revenue Growth
            for rev_key in ["Total Revenue", "Revenue", "Operating Revenue"]:
                if rev_key in income.index:
                    curr_rev = income.loc[rev_key, latest]
                    prev_rev = income.loc[rev_key, prev]
                    record["revenue_current"] = curr_rev
                    record["revenue_previous"] = prev_rev
                    record["revenue_growth"] = safe_divide(
                        (curr_rev - prev_rev), abs(prev_rev)
                    )
                    break
            
            # EBITDA Growth
            for key in ["EBITDA", "Normalized EBITDA"]:
                if key in income.index:
                    curr = income.loc[key, latest]
                    prev_val = income.loc[key, prev]
                    record["ebitda_current"] = curr
                    record["ebitda_previous"] = prev_val
                    record["ebitda_growth"] = safe_divide(
                        (curr - prev_val), abs(prev_val)
                    )
                    break
            
            # Net Income Growth
            for key in ["Net Income", "Net Income Common Stockholders"]:
                if key in income.index:
                    curr = income.loc[key, latest]
                    prev_val = income.loc[key, prev]
                    record["net_income_current"] = curr
                    record["net_income_previous"] = prev_val
                    record["earnings_growth"] = safe_divide(
                        (curr - prev_val), abs(prev_val)
                    )
                    break
            
            # Revenue CAGR (if 3+ years of data)
            if income.shape[1] >= 3:
                oldest = income.columns[-1]
                for rev_key in ["Total Revenue", "Revenue", "Operating Revenue"]:
                    if rev_key in income.index:
                        start_rev = pd.to_numeric(income.loc[rev_key, oldest], errors="coerce")
                        end_rev = pd.to_numeric(income.loc[rev_key, latest], errors="coerce")
                        n_years = income.shape[1] - 1
                        if pd.notna(start_rev) and start_rev > 0 and pd.notna(end_rev) and end_rev > 0:
                            record["revenue_cagr"] = (end_rev / start_rev) ** (1 / n_years) - 1
                        break
        
        records.append(record)
    
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).set_index("ticker")
    logger.info(f"Growth metrics computed for {len(df)} stocks")
    return df


# ============================================================================
# PROFITABILITY METRICS
# ============================================================================

def compute_profitability_metrics(financial_data: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compute profitability metrics from financial statements.
    
    Metrics:
        - EBITDA Margin
        - Net Margin
        - Gross Margin
        - Return on Equity (ROE)
        - Return on Assets (ROA)
        - ROIC proxy
    
    Args:
        financial_data: Dict from data_ingestion.fetch_financial_data()
    
    Returns:
        DataFrame with profitability metrics indexed by ticker
    """
    logger.info("Computing profitability metrics")
    records = []
    
    for ticker, data in financial_data.items():
        record = {"ticker": ticker}
        info = data.get("info", {})
        income = data.get("income_stmt", pd.DataFrame())
        balance = data.get("balance_sheet", pd.DataFrame())
        
        if not income.empty and income.shape[1] >= 1:
            latest = income.columns[0]
            
            # Get key P&L items
            revenue = None
            for key in ["Total Revenue", "Revenue"]:
                if key in income.index:
                    revenue = income.loc[key, latest]
                    break
            
            ebitda = None
            for key in ["EBITDA", "Normalized EBITDA"]:
                if key in income.index:
                    ebitda = income.loc[key, latest]
                    break
            
            net_income = None
            for key in ["Net Income", "Net Income Common Stockholders"]:
                if key in income.index:
                    net_income = income.loc[key, latest]
                    break
            
            gross_profit = None
            if "Gross Profit" in income.index:
                gross_profit = income.loc["Gross Profit", latest]
            
            # Margins
            record["gross_margin"] = safe_divide(gross_profit, revenue)
            record["ebitda_margin"] = safe_divide(ebitda, revenue)
            record["net_margin"] = safe_divide(net_income, revenue)
        
        # Balance sheet ratios
        if not balance.empty and balance.shape[1] >= 1:
            latest_bs = balance.columns[0]
            
            total_equity = None
            for key in ["Stockholders Equity", "Total Equity Gross Minority Interest"]:
                if key in balance.index:
                    total_equity = balance.loc[key, latest_bs]
                    break
            
            total_assets = None
            if "Total Assets" in balance.index:
                total_assets = balance.loc["Total Assets", latest_bs]
            
            record["roe"] = safe_divide(net_income, total_equity) if 'net_income' in dir() else None
            record["roa"] = safe_divide(net_income, total_assets) if 'net_income' in dir() else None
        
        # From yfinance info (as backup)
        record["roe_info"] = info.get("returnOnEquity")
        record["roa_info"] = info.get("returnOnAssets")
        record["profit_margins"] = info.get("profitMargins")
        record["operating_margins"] = info.get("operatingMargins")
        
        records.append(record)
    
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).set_index("ticker")
    logger.info(f"Profitability metrics computed for {len(df)} stocks")
    return df


# ============================================================================
# RISK METRICS
# ============================================================================

def compute_risk_metrics(
    price_data: Dict[str, pd.DataFrame],
    index_data: Dict[str, pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Compute risk metrics from historical price data.
    
    Metrics:
        - Annualised Volatility
        - Max Drawdown
        - Max Drawdown Duration (days)
        - Beta (vs STOXX 600 or FTSE 100 benchmark)
        - Sharpe Ratio (annualised, rf=0 assumed)
        - Sortino Ratio
        - Skewness
        - Kurtosis
    
    Args:
        price_data: Dict mapping ticker → price DataFrame
        index_data: Dict mapping index name → price DataFrame (for beta calc)
    
    Returns:
        DataFrame with risk metrics indexed by ticker
    """
    logger.info("Computing risk metrics")
    records = []
    
    # Get benchmark returns for beta calculation
    benchmark_returns = None
    if index_data:
        for bench_name in ["STOXX 600", "Euro STOXX 50", "FTSE 100"]:
            if bench_name in index_data:
                benchmark_returns = index_data[bench_name]["Close"].pct_change().dropna()
                logger.info(f"Using {bench_name} as benchmark for beta calculation")
                break
    
    for ticker, df in price_data.items():
        record = {"ticker": ticker}
        
        try:
            prices = df["Close"].dropna()
            if len(prices) < 20:
                continue
            
            daily_returns = prices.pct_change().dropna()
            
            # Annualised Volatility
            record["volatility"] = daily_returns.std() * np.sqrt(252)
            
            # Annualised Return
            total_return = prices.iloc[-1] / prices.iloc[0] - 1
            n_days = (prices.index[-1] - prices.index[0]).days
            record["annualised_return"] = (1 + total_return) ** (365 / max(n_days, 1)) - 1
            
            # Max Drawdown
            cummax = prices.cummax()
            drawdown = (prices - cummax) / cummax
            record["max_drawdown"] = drawdown.min()
            
            # Max Drawdown Duration
            is_drawdown = drawdown < 0
            if is_drawdown.any():
                # Find the longest consecutive drawdown period
                dd_groups = (~is_drawdown).cumsum()
                dd_lengths = is_drawdown.groupby(dd_groups).sum()
                record["max_dd_duration_days"] = dd_lengths.max()
            
            # Sharpe Ratio (assuming risk-free rate = 0 for simplicity)
            vol = pd.to_numeric(record.get("volatility"), errors="coerce")
            ann_ret = pd.to_numeric(record.get("annualised_return"), errors="coerce")
            if pd.notna(vol) and vol > 0 and pd.notna(ann_ret):
                record["sharpe_ratio"] = ann_ret / vol
            
            # Sortino Ratio (downside deviation only)
            downside_returns = daily_returns[daily_returns < 0]
            downside_vol = downside_returns.std() * np.sqrt(252)
            if downside_vol and downside_vol > 0:
                record["sortino_ratio"] = record["annualised_return"] / downside_vol
            
            # Higher moments
            record["skewness"] = daily_returns.skew()
            record["kurtosis"] = daily_returns.kurtosis()
            
            # Beta vs benchmark
            if benchmark_returns is not None:
                # Align dates
                aligned = pd.DataFrame({
                    "stock": daily_returns,
                    "bench": benchmark_returns
                }).dropna()
                
                if len(aligned) > 20:
                    cov = aligned["stock"].cov(aligned["bench"])
                    var_bench = aligned["bench"].var()
                    if var_bench and var_bench > 0:
                        record["beta"] = cov / var_bench
                        record["alpha"] = (
                            record["annualised_return"] -
                            record["beta"] * (aligned["bench"].mean() * 252)
                        )
        
        except Exception as e:
            logger.warning(f"Error computing risk metrics for {ticker}: {e}")
            continue
        
        records.append(record)
    
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).set_index("ticker")
    logger.info(f"Risk metrics computed for {len(df)} stocks")
    return df


# ============================================================================
# VALUATION METRICS (COMPREHENSIVE)
# ============================================================================

def compute_valuation_metrics(
    financial_data: Dict[str, Dict],
    price_data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Compute comprehensive valuation metrics.
    
    Metrics:
        - P/E (trailing)
        - Forward P/E
        - EV/EBITDA
        - EV/Revenue
        - Price/Book
        - FCF Yield
        - Dividend Yield
        - PEG Ratio
    
    Args:
        financial_data: Financial statement data
        price_data: Historical price data
    
    Returns:
        DataFrame with valuation metrics indexed by ticker
    """
    logger.info("Computing valuation metrics")
    records = []
    
    for ticker, data in financial_data.items():
        record = {"ticker": ticker}
        info = data.get("info", {})
        
        record["pe_trailing"] = info.get("trailingPE")
        record["pe_forward"] = info.get("forwardPE")
        record["ev_ebitda"] = info.get("enterpriseToEbitda")
        record["ev_revenue"] = info.get("enterpriseToRevenue")
        record["price_book"] = info.get("priceToBook")
        record["peg_ratio"] = info.get("pegRatio")
        record["dividend_yield"] = info.get("dividendYield")
        record["market_cap"] = info.get("marketCap")
        record["enterprise_value"] = info.get("enterpriseValue")
        
        # FCF Yield
        fcf = info.get("freeCashflow")
        mkt_cap = info.get("marketCap")
        record["fcf_yield"] = safe_divide(fcf, mkt_cap)
        
        # Earnings yield (inverse of P/E)
        pe = pd.to_numeric(info.get("trailingPE"), errors="coerce")
        if pd.notna(pe) and pe > 0:
            record["earnings_yield"] = 1 / pe
        
        records.append(record)
    
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).set_index("ticker")
    logger.info(f"Valuation metrics computed for {len(df)} stocks")
    return df


# ============================================================================
# COMBINED ANALYSIS
# ============================================================================

def run_financial_analysis(
    financial_data: Dict[str, Dict],
    price_data: Dict[str, pd.DataFrame],
    index_data: Dict[str, pd.DataFrame] = None
) -> Dict[str, pd.DataFrame]:
    """
    Run the complete financial analysis pipeline.
    
    Args:
        financial_data: Financial statement data
        price_data: Historical price data
        index_data: Index benchmark data
    
    Returns:
        Dictionary with keys: 'growth', 'profitability', 'risk', 'valuation', 'combined'
    """
    logger.info("=" * 60)
    logger.info("RUNNING FINANCIAL ANALYSIS")
    logger.info("=" * 60)
    
    # Compute each category
    growth = compute_growth_metrics(financial_data)
    profitability = compute_profitability_metrics(financial_data)
    risk = compute_risk_metrics(price_data, index_data)
    valuation = compute_valuation_metrics(financial_data, price_data)
    
    # Combine into a single DataFrame
    combined = pd.DataFrame(index=growth.index)
    
    # Select key columns from each category
    growth_cols = ["revenue_growth", "earnings_growth", "revenue_cagr"]
    prof_cols = ["gross_margin", "ebitda_margin", "net_margin", "roe", "roa"]
    risk_cols = ["volatility", "max_drawdown", "beta", "sharpe_ratio", "sortino_ratio"]
    val_cols = ["pe_trailing", "pe_forward", "ev_ebitda", "fcf_yield", "dividend_yield", "price_book"]
    
    for cols, source in [
        (growth_cols, growth),
        (prof_cols, profitability),
        (risk_cols, risk),
        (val_cols, valuation)
    ]:
        available = [c for c in cols if c in source.columns]
        if available:
            combined = combined.join(source[available], how="outer")
    
    results = {
        "growth": growth,
        "profitability": profitability,
        "risk": risk,
        "valuation": valuation,
        "combined": combined
    }
    
    # Cache combined analysis
    save_to_cache("financial_analysis", results)
    
    logger.info(f"Combined analysis: {len(combined)} stocks, {len(combined.columns)} metrics")
    logger.info("=" * 60)
    logger.info("FINANCIAL ANALYSIS COMPLETE")
    logger.info("=" * 60)
    
    return results


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    from data_ingestion import run_data_pipeline
    
    price_data, financial_data, metrics_df, index_data = run_data_pipeline()
    results = run_financial_analysis(financial_data, price_data, index_data)
    
    print("\n=== GROWTH METRICS SAMPLE ===")
    print(results["growth"].head(10))
    
    print("\n=== PROFITABILITY SAMPLE ===")
    print(results["profitability"].head(10))
    
    print("\n=== RISK METRICS SAMPLE ===")
    print(results["risk"].head(10))
    
    print("\n=== VALUATION SAMPLE ===")
    print(results["valuation"].head(10))
    
    path = export_to_csv(results["combined"], "financial_analysis_combined")
    print(f"\nCombined analysis exported to: {path}")
