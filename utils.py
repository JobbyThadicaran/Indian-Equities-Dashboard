"""
utils.py — Shared Utilities for European Long/Short Equity Research System
==========================================================================
Provides caching, formatting, CSV export, logging, and common helper
functions used across all modules.
"""

import os
import logging
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd
import numpy as np

from config import DATA_DIR, REPORTS_DIR, CACHE_EXPIRY_HOURS

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create a configured logger with console + file output.
    
    Args:
        name: Logger name (typically module name)
        level: Logging level (default: INFO)
    
    Returns:
        Configured logging.Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers on repeated calls
    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fmt = logging.Formatter(
            "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        
        # File handler
        os.makedirs("logs", exist_ok=True)
        fh = logging.FileHandler(f"logs/{name}.log")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    
    return logger


# ============================================================================
# DIRECTORY MANAGEMENT
# ============================================================================

def ensure_directories():
    """Create all required output directories if they don't exist."""
    for d in [DATA_DIR, REPORTS_DIR, "logs", "exports"]:
        os.makedirs(d, exist_ok=True)


# ============================================================================
# CACHING UTILITIES
# ============================================================================

def get_cache_path(key: str) -> str:
    """
    Generate a deterministic cache file path from a key string.
    
    Args:
        key: Unique identifier for the cached data
    
    Returns:
        Absolute path to the cache file
    """
    # Sanitise key for filesystem safety
    safe_key = "".join(c if c.isalnum() or c in "._-" else "_" for c in key)
    return os.path.join(DATA_DIR, f"cache_{safe_key}.pkl")


def save_to_cache(key: str, data: Any) -> None:
    """
    Persist data to disk cache using pickle.
    
    Args:
        key: Cache identifier
        data: Any picklable Python object
    """
    ensure_directories()
    path = get_cache_path(key)
    with open(path, "wb") as f:
        pickle.dump({"timestamp": datetime.now(), "data": data}, f)


def load_from_cache(key: str, max_age_hours: int = CACHE_EXPIRY_HOURS) -> Optional[Any]:
    """
    Load data from cache if it exists and hasn't expired.
    
    Args:
        key: Cache identifier
        max_age_hours: Maximum cache age in hours before it's considered stale
    
    Returns:
        Cached data if valid, None if expired or missing
    """
    path = get_cache_path(key)
    if not os.path.exists(path):
        return None
    
    try:
        with open(path, "rb") as f:
            cached = pickle.load(f)
        
        age = datetime.now() - cached["timestamp"]
        if age > timedelta(hours=max_age_hours):
            return None  # Cache is stale
        
        return cached["data"]
    except (EOFError, KeyError, pickle.UnpicklingError):
        return None


# ============================================================================
# DATA EXPORT
# ============================================================================

def export_to_csv(df: pd.DataFrame, filename: str, directory: str = "exports") -> str:
    """
    Export a DataFrame to CSV with timestamp in the filename.
    
    Args:
        df: DataFrame to export
        filename: Base filename (without extension)
        directory: Output directory
    
    Returns:
        Full path to the exported file
    """
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(directory, f"{filename}_{timestamp}.csv")
    df.to_csv(filepath, index=True)
    return filepath


# ============================================================================
# FORMATTING UTILITIES
# ============================================================================

def fmt_pct(value: float, decimals: int = 1) -> str:
    """Format a decimal as a percentage string (e.g., 0.154 → '15.4%')."""
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def fmt_number(value: float, decimals: int = 2) -> str:
    """Format a number with commas and specified decimal places."""
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def fmt_large_number(value: float) -> str:
    """Format large numbers with B/M/K suffixes (e.g., 3.2B, 450M)."""
    if pd.isna(value):
        return "N/A"
    abs_val = abs(value)
    sign = "-" if value < 0 else ""
    if abs_val >= 1e9:
        return f"{sign}{abs_val / 1e9:.1f}B"
    elif abs_val >= 1e6:
        return f"{sign}{abs_val / 1e6:.1f}M"
    elif abs_val >= 1e3:
        return f"{sign}{abs_val / 1e3:.1f}K"
    else:
        return f"{sign}{abs_val:.0f}"


def fmt_ratio(value: float, decimals: int = 1) -> str:
    """Format a ratio/multiple (e.g., P/E of 14.2x)."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}x"


# ============================================================================
# STATISTICAL HELPERS
# ============================================================================

def safe_divide(numerator: float, denominator: float) -> Optional[float]:
    """Safe division that returns None instead of raising on zero/NaN."""
    try:
        if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
            return None
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return None


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """
    Clip extreme values to specified percentiles to reduce outlier impact.
    
    Args:
        series: Input Series
        lower: Lower percentile threshold
        upper: Upper percentile threshold
    
    Returns:
        Winsorized Series
    """
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    return series.clip(lower=lower_bound, upper=upper_bound)


def percentile_rank(series: pd.Series) -> pd.Series:
    """
    Compute percentile rank (0–100) for each value in a Series.
    Higher values get higher ranks.
    
    Args:
        series: Input numeric Series
    
    Returns:
        Series of percentile ranks
    """
    return series.rank(pct=True, na_option="keep") * 100


# ============================================================================
# TICKER UTILITIES
# ============================================================================

def clean_ticker(ticker: str) -> str:
    """Extract the company name part from a Yahoo Finance ticker."""
    return ticker.split(".")[0]


def ticker_to_name(ticker: str) -> str:
    """
    Convert a ticker to a more readable name.
    Falls back to the ticker symbol if no mapping exists.
    """
    # Common European equity name mappings
    name_map = {
        "MC.PA": "LVMH", "OR.PA": "L'Oréal", "TTE.PA": "TotalEnergies",
        "SAN.PA": "Sanofi", "AIR.PA": "Airbus", "BNP.PA": "BNP Paribas",
        "SU.PA": "Schneider Electric", "ACA.PA": "Crédit Agricole",
        "RMS.PA": "Hermès", "KER.PA": "Kering", "SAF.PA": "Safran",
        "DG.PA": "Vinci", "RI.PA": "Pernod Ricard", "EL.PA": "EssilorLuxottica",
        "BN.PA": "Danone", "CS.PA": "AXA", "ENGI.PA": "Engie",
        "CAP.PA": "Capgemini", "GLE.PA": "Société Générale",
        "SAP.DE": "SAP", "SIE.DE": "Siemens", "ALV.DE": "Allianz",
        "MBG.DE": "Mercedes-Benz", "DTE.DE": "Deutsche Telekom",
        "BAS.DE": "BASF", "BMW.DE": "BMW", "BAYN.DE": "Bayer",
        "MUV2.DE": "Munich Re", "ADS.DE": "Adidas", "IFX.DE": "Infineon",
        "DBK.DE": "Deutsche Bank", "VOW3.DE": "Volkswagen",
        "RWE.DE": "RWE", "HEN3.DE": "Henkel",
        "SHEL.L": "Shell", "AZN.L": "AstraZeneca", "HSBA.L": "HSBC",
        "ULVR.L": "Unilever", "BP.L": "BP", "GSK.L": "GSK",
        "RIO.L": "Rio Tinto", "BARC.L": "Barclays", "DGE.L": "Diageo",
        "LLOY.L": "Lloyds", "GLEN.L": "Glencore", "LSEG.L": "LSEG",
        "NXT.L": "Next", "RKT.L": "Reckitt", "VOD.L": "Vodafone",
        "BATS.L": "BAT", "PRU.L": "Prudential", "CRH.L": "CRH",
        "EXPN.L": "Experian",
        "NESN.SW": "Nestlé", "NOVN.SW": "Novartis", "ROG.SW": "Roche",
        "NOVO-B.CO": "Novo Nordisk", "ASML.AS": "ASML",
        "PHIA.AS": "Philips", "UNA.AS": "Unilever NV",
        "INGA.AS": "ING", "ABI.BR": "AB InBev",
        "ENEL.MI": "Enel", "ISP.MI": "Intesa Sanpaolo",
        "UCG.MI": "UniCredit", "ENI.MI": "ENI", "RACE.MI": "Ferrari",
        "SAN.MC": "Santander", "IBE.MC": "Iberdrola", "TEF.MC": "Telefónica",
        "BBVA.MC": "BBVA", "ITX.MC": "Inditex",
    }
    return name_map.get(ticker, clean_ticker(ticker))
