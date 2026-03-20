"""
data_ingestion.py — Data Ingestion Module for European Equity Research
======================================================================
Fetches historical prices, financial statements, and valuation metrics
for the European equity universe. Uses yfinance as primary source with
akshare as fallback. Implements disk caching for offline use.
"""

import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

try:
    import akshare as ak
except Exception:
    ak = None

from config import (
    FULL_UNIVERSE, get_date_range, DATA_DIR,
    INDEX_TICKERS, CACHE_EXPIRY_HOURS
)
from utils import (
    setup_logger, ensure_directories, save_to_cache,
    load_from_cache, safe_divide
)

warnings.filterwarnings("ignore")
logger = setup_logger("data_ingestion")

# ============================================================================
# PRICE DATA FETCHING
# ============================================================================

def fetch_price_data(
    tickers: List[str],
    start: str = None,
    end: str = None,
    max_workers: int = 5
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical OHLCV price data for a list of tickers via yfinance.
    
    Uses threading for parallel downloads and caches results to disk.
    Falls back to individual downloads if batch fails.
    
    Args:
        tickers: List of Yahoo Finance ticker symbols
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format
        max_workers: Number of parallel download threads
    
    Returns:
        Dictionary mapping ticker → DataFrame with OHLCV columns
    """
    if start is None or end is None:
        start, end, _ = get_date_range()
    cache_key = f"prices_{start}_{end}"
    cached = load_from_cache(cache_key)
    if cached is not None and len(cached) > 0:
        logger.info(f"Loaded price data from cache ({len(cached)} tickers)")
        return cached
    
    logger.info(f"Fetching price data for {len(tickers)} tickers from {start} to {end}")
    price_data = {}
    failed_tickers = []
    
    def _fetch_single(ticker: str) -> Tuple[str, Optional[pd.DataFrame]]:
        """Download price data for a single ticker with error handling."""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start, end=end, auto_adjust=True)
            if df is not None and not df.empty and len(df) > 20:
                # Standardise column names
                df.index = pd.to_datetime(df.index)
                df.index = df.index.tz_localize(None)  # Remove timezone info
                return ticker, df
            else:
                return ticker, None
        except Exception as e:
            return ticker, None
    
    # Parallel download with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_single, t): t for t in tickers}
        for future in tqdm(
            as_completed(futures),
            total=len(tickers),
            desc="Downloading prices",
            ncols=80
        ):
            ticker, df = future.result()
            if df is not None:
                price_data[ticker] = df
            else:
                failed_tickers.append(ticker)
    
    logger.info(f"Successfully downloaded: {len(price_data)}/{len(tickers)} tickers")
    if failed_tickers:
        logger.warning(f"Failed tickers ({len(failed_tickers)}): {failed_tickers[:20]}...")
    
    # Cache the results
    save_to_cache(cache_key, price_data)
    
    return price_data


def fetch_index_data(
    start: str = None,
    end: str = None
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for European market indices.
    
    Uses yfinance primarily and akshare as a fallback.
    
    Args:
        start: Start date
        end: End date
    
    Returns:
        Dictionary mapping index name → DataFrame
    """
    if start is None or end is None:
        start, end, _ = get_date_range()
    cache_key = f"indices_{start}_{end}"
    cached = load_from_cache(cache_key)
    if cached is not None and len(cached) > 0:
        logger.info("Loaded index data from cache")
        return cached
    
    logger.info("Fetching index benchmark data")
    index_data = {}
    
    # Mapping for Akshare symbols (requires Chinese names for global indices)
    ak_idx_map = {
        "英国富时100指数": "FTSE 100",
        "德国DAX 30种股价指数": "DAX",
        "法CAC40指数": "CAC 40",
        "欧洲Stoxx50指数": "Euro STOXX 50"
    }
    
    for name, ticker in INDEX_TICKERS.items():
        try:
            logger.info(f"Fetching {name} via yfinance...")
            df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index)
                df.index = df.index.tz_localize(None)
                index_data[name] = df
                logger.info(f"  ✓ {name} (yf): {len(df)} data points")
            else:
                raise ValueError("Empty yf dataframe")
        except Exception as e:
            logger.warning(f"  ✗ {name} (yf) failed: {e}. Trying Akshare...")
            if ak is None:
                logger.warning(f"  ✗ {name}: Akshare unavailable in this environment")
                continue
            try:
                # Find Akshare symbol
                ak_sym = None
                for sym, label in ak_idx_map.items():
                    if label == name:
                        ak_sym = sym
                        break
                
                if ak_sym:
                    df_ak = ak.index_global_hist_sina(symbol=ak_sym)
                    if df_ak is not None and not df_ak.empty:
                        # Akshare returns [date, open, high, low, close, volume]
                        df_ak = df_ak.rename(columns={
                            df_ak.columns[0]: "Date",
                            df_ak.columns[1]: "Open", df_ak.columns[2]: "High",
                            df_ak.columns[3]: "Low",  df_ak.columns[4]: "Close",
                            df_ak.columns[5]: "Volume"
                        })
                        df_ak = df_ak.set_index("Date")
                        df_ak.index = pd.to_datetime(df_ak.index)
                        
                        # Filter by date
                        df_ak = df_ak.loc[start:end]
                        index_data[name] = df_ak
                        logger.info(f"  ✓ {name} (ak): {len(df_ak)} data points")
                    else:
                        logger.warning(f"  ✗ {name} (ak) also empty")
                else:
                    logger.warning(f"  ✗ No Akshare symbol mapped for {name}")
            except Exception as e2:
                logger.error(f"  ✗ {name} (ak) fatal error: {e2}")
    
    save_to_cache(cache_key, index_data)
    return index_data


# ============================================================================
# FINANCIAL STATEMENT DATA
# ============================================================================

def fetch_financial_data(
    tickers: List[str],
    max_workers: int = 5
) -> Dict[str, Dict]:
    """
    Fetch fundamental financial data for each ticker:
    income statement, balance sheet, cash flow, and key statistics.
    
    Args:
        tickers: List of ticker symbols
        max_workers: Number of parallel threads
    
    Returns:
        Dictionary mapping ticker → dict with keys:
            'info', 'income_stmt', 'balance_sheet', 'cashflow',
            'quarterly_income', 'quarterly_balance', 'quarterly_cashflow'
    """
    cache_key = "financials_data"
    cached = load_from_cache(cache_key)
    if cached is not None and len(cached) > 0:
        logger.info(f"Loaded financial data from cache ({len(cached)} tickers)")
        return cached
    
    logger.info(f"Fetching financial statements for {len(tickers)} tickers")
    financial_data = {}
    
    def _fetch_financials(ticker: str) -> Tuple[str, Optional[Dict]]:
        """Fetch all financial data for a single ticker."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}
            
            # Fetch annual statements
            income_stmt = stock.income_stmt
            balance_sheet = stock.balance_sheet
            cashflow = stock.cashflow
            
            # Fetch quarterly statements
            q_income = stock.quarterly_income_stmt
            q_balance = stock.quarterly_balance_sheet
            q_cashflow = stock.quarterly_cashflow
            
            data = {
                "info": info,
                "income_stmt": income_stmt if income_stmt is not None else pd.DataFrame(),
                "balance_sheet": balance_sheet if balance_sheet is not None else pd.DataFrame(),
                "cashflow": cashflow if cashflow is not None else pd.DataFrame(),
                "quarterly_income": q_income if q_income is not None else pd.DataFrame(),
                "quarterly_balance": q_balance if q_balance is not None else pd.DataFrame(),
                "quarterly_cashflow": q_cashflow if q_cashflow is not None else pd.DataFrame(),
            }
            return ticker, data
        except Exception as e:
            return ticker, None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_financials, t): t for t in tickers}
        for future in tqdm(
            as_completed(futures),
            total=len(tickers),
            desc="Downloading financials",
            ncols=80
        ):
            ticker, data = future.result()
            if data is not None:
                financial_data[ticker] = data
    
    logger.info(f"Successfully fetched financials: {len(financial_data)}/{len(tickers)}")
    save_to_cache(cache_key, financial_data)
    
    return financial_data


# ============================================================================
# METRICS EXTRACTION
# ============================================================================

def extract_key_metrics(
    financial_data: Dict[str, Dict],
    price_data: Dict[str, pd.DataFrame],
    start: str = None,
    end: str = None
) -> pd.DataFrame:
    """
    Extract and compute key metrics from raw financial + price data.
    
    For each stock computes:
      - Valuation: P/E, EV/EBITDA, FCF Yield
      - Quality: ROIC proxy, EBITDA margin, Net Debt/EBITDA
      - Growth: Revenue growth YoY, Earnings growth YoY
      - Risk: Volatility, volume
      - Momentum: 3M return, 6M return, 12M return
    
    Args:
        financial_data: Output of fetch_financial_data()
        price_data: Output of fetch_price_data()
    
    Returns:
        DataFrame with tickers as index and metrics as columns
    """
    logger.info("Extracting key metrics from financial data")
    
    # Check cache first
    if start is None or end is None:
        start, end, _ = get_date_range()
    cache_key = f"key_metrics_{start}_{end}"
    cached = load_from_cache(cache_key)
    if cached is not None and len(cached) > 0:
        logger.info(f"Loaded key metrics from cache ({len(cached)} stocks)")
        return cached
    
    records = []
    
    for ticker in financial_data:
        try:
            info = financial_data[ticker].get("info", {})
            income = financial_data[ticker].get("income_stmt", pd.DataFrame())
            balance = financial_data[ticker].get("balance_sheet", pd.DataFrame())
            cashflow = financial_data[ticker].get("cashflow", pd.DataFrame())
            
            record = {"ticker": ticker}
            
            # ------------------------------------------------------------------
            # VALUATION METRICS
            # ------------------------------------------------------------------
            record["pe_ratio"] = info.get("trailingPE") or info.get("forwardPE")
            record["ev_ebitda"] = info.get("enterpriseToEbitda")
            record["market_cap"] = info.get("marketCap")
            record["enterprise_value"] = info.get("enterpriseValue")
            
            # FCF Yield = Free Cash Flow / Market Cap
            fcf = info.get("freeCashflow")
            mkt_cap = info.get("marketCap")
            record["fcf"] = fcf
            record["fcf_yield"] = safe_divide(fcf, mkt_cap) if fcf and mkt_cap else None
            
            # ------------------------------------------------------------------
            # PROFITABILITY METRICS
            # ------------------------------------------------------------------
            # Revenue and EBITDA from income statement
            if not income.empty and income.shape[1] >= 1:
                latest_col = income.columns[0]
                
                # Revenue
                revenue = None
                for rev_key in ["Total Revenue", "Revenue", "Operating Revenue"]:
                    if rev_key in income.index:
                        revenue = income.loc[rev_key, latest_col]
                        break
                record["revenue"] = revenue
                
                # EBITDA
                ebitda = None
                for ebitda_key in ["EBITDA", "Normalized EBITDA"]:
                    if ebitda_key in income.index:
                        ebitda = income.loc[ebitda_key, latest_col]
                        break
                record["ebitda"] = ebitda
                
                # Net Income
                net_income = None
                for ni_key in ["Net Income", "Net Income Common Stockholders"]:
                    if ni_key in income.index:
                        net_income = income.loc[ni_key, latest_col]
                        break
                record["net_income"] = net_income
                
                # EBITDA Margin
                record["ebitda_margin"] = safe_divide(ebitda, revenue)
                
                # Net Margin
                record["net_margin"] = safe_divide(net_income, revenue)
                
                # YoY Revenue Growth
                if income.shape[1] >= 2:
                    prev_col = income.columns[1]
                    prev_revenue = None
                    for rev_key in ["Total Revenue", "Revenue", "Operating Revenue"]:
                        if rev_key in income.index:
                            prev_revenue = income.loc[rev_key, prev_col]
                            break
                    record["revenue_growth_yoy"] = safe_divide(
                        (revenue - prev_revenue) if revenue is not None and prev_revenue is not None else None,
                        prev_revenue
                    )
                    
                    # YoY Earnings Growth
                    prev_ni = None
                    for ni_key in ["Net Income", "Net Income Common Stockholders"]:
                        if ni_key in income.index:
                            prev_ni = income.loc[ni_key, prev_col]
                            break
                    record["earnings_growth_yoy"] = safe_divide(
                        (net_income - prev_ni) if net_income is not None and prev_ni is not None else None,
                        prev_ni
                    )
            
            # ------------------------------------------------------------------
            # BALANCE SHEET METRICS
            # ------------------------------------------------------------------
            if not balance.empty and balance.shape[1] >= 1:
                latest_bs = balance.columns[0]
                
                # Total Debt
                total_debt = None
                for debt_key in ["Total Debt", "Long Term Debt", "Total Non Current Liabilities Net Minority Interest"]:
                    if debt_key in balance.index:
                        total_debt = balance.loc[debt_key, latest_bs]
                        break
                record["total_debt"] = total_debt
                
                # Cash and equivalents
                cash = None
                for cash_key in ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash Financial"]:
                    if cash_key in balance.index:
                        cash = balance.loc[cash_key, latest_bs]
                        break
                record["cash"] = cash
                
                # Net Debt
                if total_debt is not None and cash is not None:
                    record["net_debt"] = total_debt - cash
                else:
                    record["net_debt"] = total_debt
                
                # Leverage: Net Debt / EBITDA
                ebitda_val = record.get("ebitda")
                record["leverage"] = safe_divide(
                    record.get("net_debt"), ebitda_val
                )
                
                # Total Equity for ROIC proxy
                total_equity = None
                for eq_key in ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"]:
                    if eq_key in balance.index:
                        total_equity = balance.loc[eq_key, latest_bs]
                        break
                record["total_equity"] = total_equity
                
                # ROIC Proxy = Net Income / (Equity + Net Debt)
                invested_capital = None
                if total_equity is not None:
                    nd = record.get("net_debt", 0) or 0
                    invested_capital = total_equity + nd
                record["roic"] = safe_divide(
                    record.get("net_income"), invested_capital
                )
            
            # ------------------------------------------------------------------
            # MOMENTUM & RISK (from price data)
            # ------------------------------------------------------------------
            if ticker in price_data:
                prices = price_data[ticker]["Close"]
                if len(prices) > 0:
                    current_price = prices.iloc[-1]
                    record["current_price"] = current_price
                    
                    # Momentum returns
                    if len(prices) >= 63:  # ~3 months of trading days
                        record["mom_3m"] = (current_price / prices.iloc[-63] - 1)
                    if len(prices) >= 126:  # ~6 months
                        record["mom_6m"] = (current_price / prices.iloc[-126] - 1)
                    if len(prices) >= 252:  # ~12 months
                        record["mom_12m"] = (current_price / prices.iloc[-252] - 1)
                    
                    # Daily returns for volatility
                    daily_returns = prices.pct_change().dropna()
                    record["volatility"] = daily_returns.std() * np.sqrt(252)  # Annualised
                    
                    # Average volume (20-day)
                    if "Volume" in price_data[ticker].columns:
                        record["avg_volume_20d"] = price_data[ticker]["Volume"].tail(20).mean()
                    
                    # Max Drawdown
                    cummax = prices.cummax()
                    drawdown = (prices - cummax) / cummax
                    record["max_drawdown"] = drawdown.min()
            
            # Additional info fields
            record["sector"] = info.get("sector", "Unknown")
            record["industry"] = info.get("industry", "Unknown")
            record["name"] = info.get("shortName") or info.get("longName", ticker)
            
            records.append(record)
            
        except Exception as e:
            logger.warning(f"Error extracting metrics for {ticker}: {e}")
            continue
    
    if not records:
        logger.warning("No records extracted — returning empty metrics DataFrame")
        return pd.DataFrame()
    
    df = pd.DataFrame(records).set_index("ticker")
    
    # Ensure numeric columns are indeed numeric (yfinance sometimes returns strings/None)
    numeric_cols = [
        "pe_ratio", "ev_ebitda", "market_cap", "enterprise_value", "fcf", "fcf_yield",
        "revenue", "ebitda", "net_income", "ebitda_margin", "net_margin",
        "revenue_growth_yoy", "earnings_growth_yoy", "total_debt", "cash",
        "net_debt", "leverage", "roic", "current_price", "mom_3m", "mom_6m",
        "mom_12m", "volatility", "avg_volume_20d", "max_drawdown"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    logger.info(f"Extracted metrics for {len(df)} stocks")
    # Cache the metrics
    save_to_cache(cache_key, df)
    
    return df


# ============================================================================
# MAIN DATA PIPELINE
# ============================================================================

def run_data_pipeline(
    universe: List[str] = None,
    start: str = None,
    end: str = None
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict], pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Execute the full data ingestion pipeline:
    1. Fetch price data
    2. Fetch index data
    3. Fetch financial statements
    4. Extract key metrics
    
    Args:
        universe: List of tickers (defaults to FULL_UNIVERSE from config)
        start: Start date
        end: End date
    
    Returns:
        Tuple of (price_data, financial_data, metrics_df, index_data)
    """
    ensure_directories()
    if start is None or end is None:
        start, end, _ = get_date_range()
    
    if universe is None:
        universe = FULL_UNIVERSE
    
    logger.info("=" * 60)
    logger.info("STARTING DATA INGESTION PIPELINE")
    logger.info(f"Universe: {len(universe)} tickers")
    logger.info(f"Period: {start} to {end}")
    logger.info("=" * 60)
    
    # Step 1: Price data
    price_data = fetch_price_data(universe, start, end)
    
    # Step 2: Index data
    index_data = fetch_index_data(start, end)
    
    # Step 3: Financial statements (only for tickers with price data)
    valid_tickers = list(price_data.keys())
    financial_data = fetch_financial_data(valid_tickers)
    
    # Step 4: Extract metrics
    metrics_df = extract_key_metrics(financial_data, price_data, start, end)
    
    logger.info("=" * 60)
    logger.info("DATA INGESTION COMPLETE")
    logger.info(f"Stocks with prices: {len(price_data)}")
    logger.info(f"Stocks with financials: {len(financial_data)}")
    logger.info(f"Stocks with metrics: {len(metrics_df)}")
    logger.info("=" * 60)
    
    return price_data, financial_data, metrics_df, index_data


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    price_data, financial_data, metrics_df, index_data = run_data_pipeline()
    
    # Export metrics to CSV
    from utils import export_to_csv
    path = export_to_csv(metrics_df, "key_metrics")
    print(f"\nMetrics exported to: {path}")
    print(f"\nSample metrics:\n{metrics_df.head(10)}")
