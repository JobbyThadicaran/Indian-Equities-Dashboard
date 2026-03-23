"""
data_ingestion.py — Data Ingestion Module for Indian Equity Research
====================================================================
Fetches historical prices, financial statements, and valuation metrics
for the Indian working universe. Prices and fundamentals come from
yfinance, while the universe definition is built from NIFTY 50 plus
F&O-eligible stocks discovered via Zerodha/public files.
"""

import hashlib
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm

from config import (
    FULL_UNIVERSE, get_date_range, DATA_DIR,
    INDEX_TICKERS, INDEX_QUOTE_KEYS, CACHE_EXPIRY_HOURS,
    GIFT_NIFTY_SPOT_URL, GIFT_NIFTY_FUTURES_URL, ZERODHA_QUOTE_BATCH_SIZE
)
from market_universe import DEFAULT_USER_AGENT, ZERODHA_API_BASE, build_india_universe
from utils import (
    setup_logger, ensure_directories, save_to_cache,
    load_from_cache, safe_divide
)

warnings.filterwarnings("ignore")
logger = setup_logger("data_ingestion")


def _is_financial_company(
    sector: Optional[object] = None,
    industry: Optional[object] = None,
    info: Optional[Dict] = None,
) -> bool:
    """Flag banks, insurers, and similar financials where EV/EBITDA and ROIC are not comparable."""
    info = info or {}
    values = [
        sector,
        industry,
        info.get("sector"),
        info.get("industry"),
        info.get("quoteType"),
    ]
    text = " ".join(str(value).lower() for value in values if value)
    keywords = [
        "financial",
        "bank",
        "insurance",
        "capital markets",
        "credit",
        "asset management",
        "nbfc",
        "finserv",
    ]
    return any(keyword in text for keyword in keywords)


def _universe_fingerprint(tickers: List[str]) -> str:
    """Generate a stable cache suffix for a specific ticker set."""
    unique = sorted(set(tickers))
    if not unique:
        return "empty"
    digest = hashlib.sha1(",".join(unique).encode("utf-8")).hexdigest()[:12]
    return f"{len(unique)}_{digest}"


def _chunked(items: List[str], size: int) -> List[List[str]]:
    return [items[idx:idx + size] for idx in range(0, len(items), size)]


def _parse_live_timestamp(snapshot: Dict) -> Optional[pd.Timestamp]:
    for key in ("timestamp", "last_trade_time"):
        value = snapshot.get(key)
        if value:
            try:
                ts = pd.to_datetime(value)
                if getattr(ts, "tzinfo", None) is not None:
                    ts = ts.tz_convert(None)
                return ts
            except Exception:
                continue
    return None


def _normalise_symbol_from_ticker(ticker: str, universe_metadata: Optional[pd.DataFrame]) -> str:
    if universe_metadata is not None and not universe_metadata.empty and ticker in universe_metadata.index:
        value = universe_metadata.loc[ticker].get("symbol")
        if pd.notna(value):
            return str(value)
    return ticker.split(".")[0]


def _apply_live_snapshot_to_frame(frame: pd.DataFrame, snapshot: Dict) -> pd.DataFrame:
    """Merge a live quote snapshot into a daily OHLCV frame using today's date."""
    if not snapshot or snapshot.get("last_price") in (None, 0):
        return frame

    base = frame.copy() if frame is not None else pd.DataFrame()
    ohlc = snapshot.get("ohlc", {}) or {}
    trade_ts = _parse_live_timestamp(snapshot) or pd.Timestamp.now()
    row_date = trade_ts.normalize()
    last_price = float(snapshot.get("last_price"))

    row = {
        "Open": float(ohlc.get("open", last_price)),
        "High": float(ohlc.get("high", last_price)),
        "Low": float(ohlc.get("low", last_price)),
        "Close": last_price,
    }
    if "Volume" in base.columns or snapshot.get("volume") is not None:
        row["Volume"] = snapshot.get("volume")

    for column in base.columns:
        if column in row:
            continue
        if column in {"Dividends", "Stock Splits"}:
            row[column] = 0.0
        else:
            row[column] = np.nan

    live_row = pd.DataFrame([row], index=[row_date])
    if not base.empty and pd.to_datetime(base.index[-1]).normalize() == row_date:
        base = base.iloc[:-1]
    merged = pd.concat([base, live_row], axis=0)
    merged.index = pd.to_datetime(merged.index).tz_localize(None)
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    return merged


def fetch_live_zerodha_quotes(
    tickers: List[str],
    universe_metadata: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[str, Dict], Dict[str, Dict], Optional[str]]:
    """Fetch live quote snapshots for equities and benchmark indices from Zerodha."""
    api_key = os.getenv("ZERODHA_API_KEY", "")
    access_token = os.getenv("ZERODHA_ACCESS_TOKEN", "")
    if not (api_key and access_token):
        return {}, {}, "missing Zerodha API key or access token"

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": DEFAULT_USER_AGENT,
            "X-Kite-Version": "3",
            "Authorization": f"token {api_key}:{access_token}",
        }
    )

    stock_request_map = {}
    for ticker in tickers:
        symbol = _normalise_symbol_from_ticker(ticker, universe_metadata)
        if symbol:
            stock_request_map[f"NSE:{symbol}"] = ticker

    request_keys = list(stock_request_map.keys()) + list(INDEX_QUOTE_KEYS.values())
    stock_quotes: Dict[str, Dict] = {}
    index_quotes: Dict[str, Dict] = {}

    try:
        for batch in _chunked(request_keys, ZERODHA_QUOTE_BATCH_SIZE):
            params = [("i", key) for key in batch]
            response = session.get(f"{ZERODHA_API_BASE}/quote", params=params, timeout=15)
            response.raise_for_status()
            payload = response.json()
            if payload.get("status") != "success":
                raise ValueError(payload.get("message") or "quote API returned non-success status")
            quote_data = payload.get("data", {})
            for request_key, snapshot in quote_data.items():
                if request_key in stock_request_map:
                    stock_quotes[stock_request_map[request_key]] = snapshot
                else:
                    for name, key in INDEX_QUOTE_KEYS.items():
                        if request_key == key:
                            index_quotes[name] = snapshot
                            break
    except Exception as exc:
        logger.warning("Live Zerodha quote fetch failed: %s", exc)
        return {}, {}, str(exc)

    logger.info(
        "Fetched Zerodha live quotes for %s equities and %s indices",
        len(stock_quotes),
        len(index_quotes),
    )
    return stock_quotes, index_quotes, None


def fetch_gift_nifty_snapshot() -> Tuple[Optional[Dict], Optional[str]]:
    """Fetch the current GIFT NIFTY snapshot from official NSE IX endpoints."""
    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"})

    try:
        response = session.get(GIFT_NIFTY_FUTURES_URL, timeout=12)
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        nifty_rows = [row for row in rows if row.get("SYMBOL") == "NIFTY" and row.get("INSTRUMENTTYPE") == "FUTIDX"]
        if nifty_rows:
            nifty_rows.sort(key=lambda row: pd.to_datetime(row.get("EXPIRYDATE"), errors="coerce"))
            row = nifty_rows[0]
            expiry = row.get("EXPIRYDATE")
            return {
                "label": "GIFT NIFTY",
                "last_price": float(row.get("LASTPRICE", 0)),
                "change": float(row.get("DAYCHANGE_1", 0)),
                "change_pct": float(row.get("PERCHANGE", 0)) / 100.0,
                "timestamp": row.get("TIMESTMP"),
                "source": f"NSE IX front-month futures ({expiry})" if expiry else "NSE IX front-month futures",
                "expiry": expiry,
            }, None
    except Exception as exc:
        logger.debug("GIFT NIFTY futures snapshot failed: %s", exc)

    try:
        response = session.get(GIFT_NIFTY_SPOT_URL, timeout=12)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list) and payload:
            row = payload[0]
            label = str(
                row.get("OI_INDEX_NAME")
                or row.get("INDEX_NAME")
                or row.get("name")
                or ""
            ).strip()
            if "gift" in label.lower():
                last_price = float(str(row.get("CURRVALUE", "0")).replace(",", ""))
                change = float(row.get("CHANGE", 0))
                change_pct = float(row.get("PERCHANGE", 0)) / 100.0
                return {
                    "label": label or "GIFT NIFTY",
                    "last_price": last_price,
                    "change": change,
                    "change_pct": change_pct,
                    "timestamp": row.get("FULLTIMESTAMP") or row.get("TIMESTAMP"),
                    "source": "NSE IX market snapshot",
                }, None
    except Exception as exc:
        logger.warning("GIFT NIFTY snapshot fetch failed: %s", exc)
        return None, str(exc)

    unmatched_label = None
    try:
        response = session.get(GIFT_NIFTY_SPOT_URL, timeout=12)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list) and payload:
            row = payload[0]
            unmatched_label = row.get("OI_INDEX_NAME") or row.get("INDEX_NAME") or row.get("name")
    except Exception:
        pass

    if unmatched_label:
        return None, f"official NSE IX spot snapshot returned '{unmatched_label}' instead of an explicit GIFT NIFTY row"
    return None, "official NSE IX endpoints returned no explicit GIFT NIFTY row"

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
    cache_key = f"prices_{_universe_fingerprint(tickers)}_{start}_{end}"
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
    Fetch historical data for Indian market indices via yfinance.
    
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

    for name, ticker in INDEX_TICKERS.items():
        try:
            logger.info(f"Fetching {name} via yfinance")
            df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index)
                df.index = df.index.tz_localize(None)
                index_data[name] = df
                logger.info(f"  ✓ {name}: {len(df)} data points")
            else:
                logger.warning(f"  ✗ {name}: empty dataframe")
        except Exception as e:
            logger.warning(f"  ✗ {name}: {e}")
    
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
    cache_key = f"financials_data_{_universe_fingerprint(tickers)}"
    cached = load_from_cache(cache_key)
    financial_data = {}
    missing_tickers = list(tickers)
    if cached is not None and len(cached) > 0:
        financial_data = {ticker: cached[ticker] for ticker in tickers if ticker in cached}
        missing_tickers = [ticker for ticker in tickers if ticker not in financial_data]
        logger.info(
            "Loaded financial data from cache (%s tickers); refetching %s missing tickers",
            len(financial_data),
            len(missing_tickers),
        )
        if not missing_tickers:
            return financial_data
    else:
        logger.info(f"Fetching financial statements for {len(tickers)} tickers")
    
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
        futures = {executor.submit(_fetch_financials, t): t for t in missing_tickers}
        for future in tqdm(
            as_completed(futures),
            total=len(missing_tickers),
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
    universe_metadata: Optional[pd.DataFrame] = None,
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
    cache_key = f"key_metrics_v2_{_universe_fingerprint(list(financial_data.keys()))}_{start}_{end}"
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

                if (
                    pd.isna(record.get("ev_ebitda"))
                    and record.get("enterprise_value") is not None
                    and ebitda is not None
                    and not pd.isna(ebitda)
                    and ebitda > 0
                ):
                    record["ev_ebitda"] = safe_divide(record.get("enterprise_value"), ebitda)
            
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
                if ebitda_val is not None and not pd.isna(ebitda_val) and ebitda_val > 0:
                    record["leverage"] = safe_divide(
                        record.get("net_debt"), ebitda_val
                    )
                else:
                    record["leverage"] = None
                
                # Total Equity for ROIC proxy
                total_equity = None
                for eq_key in ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"]:
                    if eq_key in balance.index:
                        total_equity = balance.loc[eq_key, latest_bs]
                        break
                record["total_equity"] = total_equity
                
                # ROIC Proxy = Net Income / (Equity + Net Debt)
                invested_capital = None
                if total_equity is not None and not pd.isna(total_equity):
                    nd = record.get("net_debt", 0) or 0
                    invested_capital = total_equity + nd
                if invested_capital is not None and not pd.isna(invested_capital) and invested_capital > 0:
                    record["roic"] = safe_divide(
                        record.get("net_income"), invested_capital
                    )
                else:
                    record["roic"] = None
            
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

            if _is_financial_company(record.get("sector"), record.get("industry"), info):
                for field in [
                    "fcf",
                    "fcf_yield",
                    "ebitda",
                    "ev_ebitda",
                    "ebitda_margin",
                    "net_debt",
                    "leverage",
                    "roic",
                ]:
                    record[field] = None
            
            records.append(record)
            
        except Exception as e:
            logger.warning(f"Error extracting metrics for {ticker}: {e}")
            continue
    
    if not records:
        logger.warning("No records extracted — returning empty metrics DataFrame")
        return pd.DataFrame()
    
    df = pd.DataFrame(records).set_index("ticker")

    if universe_metadata is not None and not universe_metadata.empty:
        metadata = universe_metadata.reindex(df.index)
        for col in ["symbol", "company_name", "country", "is_nifty50", "is_fo_eligible", "universe_bucket", "universe_source"]:
            if col in metadata.columns:
                df[col] = metadata[col]

        if "sector" in metadata.columns:
            if "sector" in df.columns:
                df["sector"] = df["sector"].replace("Unknown", pd.NA).fillna(metadata["sector"])
            else:
                df["sector"] = metadata["sector"]

        if "company_name" in metadata.columns:
            missing_name = df["name"].isna() | df["name"].eq(df.index)
            df.loc[missing_name, "name"] = metadata.loc[missing_name, "company_name"]
    
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

    universe_metadata = build_india_universe()
    if universe is None:
        universe = universe_metadata.index.tolist()
    else:
        universe = list(universe)
        if universe_metadata is not None and not universe_metadata.empty:
            universe_metadata = universe_metadata.reindex(universe)

    if not universe:
        universe = FULL_UNIVERSE
        universe_metadata = build_india_universe().reindex(universe)
    
    logger.info("=" * 60)
    logger.info("STARTING DATA INGESTION PIPELINE")
    logger.info(f"Universe: {len(universe)} tickers")
    logger.info(f"Period: {start} to {end}")
    logger.info("=" * 60)
    
    # Step 1: Price data
    price_data = fetch_price_data(universe, start, end)
    
    # Step 2: Index data
    index_data = fetch_index_data(start, end)

    # Step 2b: Overlay live quote snapshots when Zerodha credentials are available
    live_stock_quotes, live_index_quotes, live_quote_error = fetch_live_zerodha_quotes(
        universe,
        universe_metadata=universe_metadata,
    )
    if live_stock_quotes:
        price_data = {
            ticker: _apply_live_snapshot_to_frame(frame, live_stock_quotes.get(ticker, {}))
            for ticker, frame in price_data.items()
        }
    if live_index_quotes:
        index_data = {
            name: _apply_live_snapshot_to_frame(frame, live_index_quotes.get(name, {}))
            for name, frame in index_data.items()
        }
    live_quote_mode = bool(live_stock_quotes or live_index_quotes)

    gift_nifty_snapshot, gift_nifty_error = fetch_gift_nifty_snapshot()
    
    # Step 3: Financial statements (only for tickers with price data)
    valid_tickers = list(price_data.keys())
    financial_data = fetch_financial_data(valid_tickers)
    
    # Step 4: Extract metrics
    metrics_df = extract_key_metrics(
        financial_data,
        price_data,
        universe_metadata=universe_metadata,
        start=start,
        end=end,
    )
    metrics_df.attrs["live_quote_mode"] = live_quote_mode
    metrics_df.attrs["live_quote_error"] = live_quote_error
    metrics_df.attrs["live_quote_source"] = "Zerodha quote snapshots" if live_quote_mode else None
    metrics_df.attrs["live_index_quotes"] = live_index_quotes
    metrics_df.attrs["live_quote_timestamp"] = max(
        [
            _parse_live_timestamp(snapshot)
            for snapshot in list(live_stock_quotes.values()) + list(live_index_quotes.values())
            if _parse_live_timestamp(snapshot) is not None
        ],
        default=None,
    )
    metrics_df.attrs["gift_nifty_snapshot"] = gift_nifty_snapshot
    metrics_df.attrs["gift_nifty_error"] = gift_nifty_error
    
    logger.info("=" * 60)
    logger.info("DATA INGESTION COMPLETE")
    logger.info(f"Stocks with prices: {len(price_data)}")
    logger.info(f"Stocks with financials: {len(financial_data)}")
    logger.info(f"Stocks with metrics: {len(metrics_df)}")
    if live_quote_mode:
        logger.info("Live quote overlay: enabled via Zerodha")
    elif live_quote_error:
        logger.info("Live quote overlay unavailable: %s", live_quote_error)
    if gift_nifty_snapshot:
        logger.info("GIFT NIFTY snapshot: %s @ %s", gift_nifty_snapshot.get("last_price"), gift_nifty_snapshot.get("timestamp"))
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
