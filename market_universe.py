"""
market_universe.py — India universe discovery and Zerodha helpers
=================================================================
Builds the live Indian equity universe used by the factor model:
NIFTY 50 plus F&O-eligible stocks. Prefers Zerodha instruments when a
live access token is available and falls back to public NSE/Nifty files.
"""

from __future__ import annotations

import hashlib
import io
import os
from typing import Iterable, Optional, Tuple

import pandas as pd
import requests

from config import CACHE_EXPIRY_HOURS, NIFTY_50_FALLBACK, SECTOR_FALLBACK
from utils import load_from_cache, save_to_cache, setup_logger

logger = setup_logger("market_universe")

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
NIFTY_50_URLS = [
    "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
    "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
]
FO_SYMBOL_URLS = [
    "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv",
    "https://archives.nseindia.com/content/fo/fo_mktlots.csv",
    "https://www.nseindia.com/content/fo/fo_mktlots.csv",
]
ZERODHA_API_BASE = "https://api.kite.trade"
INDEX_UNDERLYINGS = {
    "BANKNIFTY",
    "BANKEX",
    "FINNIFTY",
    "MIDCPNIFTY",
    "NIFTY",
    "NIFTYNXT50",
    "SENSEX",
}


def _normalise_symbol(value: object) -> str:
    text = str(value or "").strip().upper()
    return text.replace("*", "")


def _to_yahoo_ticker(symbol: str) -> str:
    return f"{symbol}.NS"


def _read_csv_with_header_detection(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    header_idx = 0
    for idx, line in enumerate(lines[:10]):
        upper = line.upper()
        if "," in line and ("SYMBOL" in upper or "COMPANY NAME" in upper):
            header_idx = idx
            break
    return pd.read_csv(io.StringIO(text), skiprows=header_idx)


def _new_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "text/csv,application/json,text/plain,*/*",
            "Referer": "https://www.nseindia.com/",
        }
    )
    return session


def _prime_nse_session(session: requests.Session) -> None:
    try:
        session.get("https://www.nseindia.com/", timeout=8)
    except Exception:
        pass


def _fetch_csv_from_urls(urls: Iterable[str], *, session: Optional[requests.Session] = None) -> Optional[pd.DataFrame]:
    working_session = session or _new_session()
    _prime_nse_session(working_session)
    for url in urls:
        try:
            response = working_session.get(url, timeout=12)
            response.raise_for_status()
            return _read_csv_with_header_detection(response.text)
        except Exception as exc:
            logger.debug("CSV fetch failed for %s: %s", url, exc)
            continue
    return None


def _build_fallback_nifty50() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": NIFTY_50_FALLBACK,
            "company_name": NIFTY_50_FALLBACK,
            "sector": [SECTOR_FALLBACK.get(symbol, "Unknown") for symbol in NIFTY_50_FALLBACK],
        }
    )


def get_nifty50_constituents() -> pd.DataFrame:
    """Return NIFTY 50 constituents from public sources with cached fallback."""
    cache_key = "india_nifty50_constituents_v1"
    cached = load_from_cache(cache_key, max_age_hours=CACHE_EXPIRY_HOURS)
    if cached is not None and not cached.empty:
        return cached

    frame = _fetch_csv_from_urls(NIFTY_50_URLS)
    if frame is None or frame.empty:
        fallback = _build_fallback_nifty50()
        save_to_cache(cache_key, fallback)
        return fallback

    frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
    frame = frame.rename(
        columns={
            "symbol": "symbol",
            "company_name": "company_name",
            "company_name_": "company_name",
            "industry": "sector",
        }
    )
    if "symbol" not in frame.columns:
        fallback = _build_fallback_nifty50()
        save_to_cache(cache_key, fallback)
        return fallback

    frame["symbol"] = frame["symbol"].map(_normalise_symbol)
    if "company_name" not in frame.columns:
        frame["company_name"] = frame["symbol"]
    if "sector" not in frame.columns:
        frame["sector"] = frame["symbol"].map(SECTOR_FALLBACK).fillna("Unknown")
    frame = frame[["symbol", "company_name", "sector"]]
    frame = frame[frame["symbol"] != ""].drop_duplicates("symbol").sort_values("symbol")
    save_to_cache(cache_key, frame)
    return frame


def _extract_public_fo_symbols(frame: pd.DataFrame) -> list[str]:
    if frame is None or frame.empty:
        return []

    working = frame.copy()
    working.columns = [str(column).strip().lower().replace(" ", "_") for column in working.columns]
    symbol_column = None
    for candidate in ("symbol", "underlying", "underlying_symbol", "security"):
        if candidate in working.columns:
            symbol_column = candidate
            break
    if symbol_column is None:
        symbol_column = working.columns[0]

    symbols = (
        working[symbol_column]
        .map(_normalise_symbol)
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    filtered = []
    for symbol in symbols:
        if symbol in INDEX_UNDERLYINGS:
            continue
        if not any(ch.isalpha() for ch in symbol):
            continue
        filtered.append(symbol)
    return sorted(set(filtered))


def get_public_fo_symbols() -> list[str]:
    """Return F&O-eligible stock symbols from the NSE market-lot file."""
    cache_key = "india_public_fo_symbols_v1"
    cached = load_from_cache(cache_key, max_age_hours=CACHE_EXPIRY_HOURS)
    if cached:
        return cached

    frame = _fetch_csv_from_urls(FO_SYMBOL_URLS)
    symbols = _extract_public_fo_symbols(frame)
    if not symbols:
        logger.warning("F&O public file unavailable; falling back to NIFTY 50 symbols")
        symbols = sorted(set(NIFTY_50_FALLBACK))
    save_to_cache(cache_key, symbols)
    return symbols


def _extract_zerodha_fo_symbols(frame: pd.DataFrame) -> list[str]:
    if frame is None or frame.empty:
        return []

    working = frame.copy()
    for column in ("exchange", "segment", "name", "tradingsymbol", "instrument_type"):
        if column in working.columns:
            working[column] = working[column].astype(str)

    mask = pd.Series(True, index=working.index)
    if "exchange" in working.columns:
        mask &= working["exchange"].str.upper().eq("NFO")
    elif "segment" in working.columns:
        mask &= working["segment"].str.upper().str.startswith("NFO")

    if "instrument_type" in working.columns:
        mask &= working["instrument_type"].str.upper().isin({"FUT", "CE", "PE"})

    subset = working.loc[mask].copy()
    if subset.empty:
        return []

    if "name" in subset.columns:
        raw_symbols = subset["name"].map(_normalise_symbol)
    else:
        raw_symbols = subset["tradingsymbol"].map(_normalise_symbol)

    symbols = []
    for symbol in raw_symbols.dropna().tolist():
        if not symbol or symbol in INDEX_UNDERLYINGS:
            continue
        if not any(ch.isalpha() for ch in symbol):
            continue
        symbols.append(symbol)
    return sorted(set(symbols))


def get_zerodha_fo_symbols(api_key: Optional[str] = None, access_token: Optional[str] = None) -> list[str]:
    """Return F&O-eligible stock symbols from Zerodha instruments."""
    resolved_api_key = api_key or os.getenv("ZERODHA_API_KEY", "")
    resolved_access_token = access_token or os.getenv("ZERODHA_ACCESS_TOKEN", "")
    if not (resolved_api_key and resolved_access_token):
        return []

    cache_suffix = hashlib.sha1(resolved_api_key.encode("utf-8")).hexdigest()[:8]
    cache_key = f"india_zerodha_fo_symbols_v1_{cache_suffix}"
    cached = load_from_cache(cache_key, max_age_hours=CACHE_EXPIRY_HOURS)
    if cached:
        return cached

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": DEFAULT_USER_AGENT,
            "X-Kite-Version": "3",
            "Authorization": f"token {resolved_api_key}:{resolved_access_token}",
        }
    )
    response = session.get(f"{ZERODHA_API_BASE}/instruments", timeout=15)
    response.raise_for_status()
    frame = pd.read_csv(io.StringIO(response.text))
    symbols = _extract_zerodha_fo_symbols(frame)
    if symbols:
        save_to_cache(cache_key, symbols)
    return symbols


def build_india_universe(api_key: Optional[str] = None, access_token: Optional[str] = None) -> pd.DataFrame:
    """
    Build the working India universe metadata indexed by Yahoo ticker.

    Columns include:
      symbol, company_name, sector, country,
      is_nifty50, is_fo_eligible, universe_bucket, universe_source
    """
    mode = "live" if (api_key or os.getenv("ZERODHA_API_KEY")) and (access_token or os.getenv("ZERODHA_ACCESS_TOKEN")) else "public"
    cache_key = f"india_market_universe_v2_{mode}"
    cached = load_from_cache(cache_key, max_age_hours=CACHE_EXPIRY_HOURS)
    if cached is not None and not cached.empty:
        return cached

    nifty50 = get_nifty50_constituents()
    nifty_symbols = set(nifty50["symbol"].map(_normalise_symbol).tolist())

    fo_source = "NSE public"
    try:
        fo_symbols = set(get_zerodha_fo_symbols(api_key=api_key, access_token=access_token))
        if fo_symbols:
            fo_source = "Zerodha instruments"
        else:
            fo_symbols = set(get_public_fo_symbols())
    except Exception as exc:
        logger.warning("Zerodha F&O universe unavailable: %s", exc)
        fo_symbols = set(get_public_fo_symbols())

    all_symbols = sorted(nifty_symbols | fo_symbols)
    if not all_symbols:
        all_symbols = sorted(set(NIFTY_50_FALLBACK))
        fo_symbols = set(all_symbols)
        fo_source = "Fallback"

    frame = pd.DataFrame({"symbol": all_symbols})
    frame["company_name"] = frame["symbol"]
    frame["sector"] = frame["symbol"].map(SECTOR_FALLBACK).fillna("Unknown")

    nifty_lookup = nifty50.set_index("symbol")
    in_nifty = frame["symbol"].isin(nifty_lookup.index)
    frame.loc[in_nifty, "company_name"] = frame.loc[in_nifty, "symbol"].map(nifty_lookup["company_name"])
    frame.loc[in_nifty, "sector"] = frame.loc[in_nifty, "symbol"].map(nifty_lookup["sector"]).fillna(frame.loc[in_nifty, "sector"])

    frame["is_nifty50"] = frame["symbol"].isin(nifty_symbols)
    frame["is_fo_eligible"] = frame["symbol"].isin(fo_symbols)
    frame["country"] = "India"
    frame["universe_source"] = fo_source
    frame["universe_bucket"] = frame.apply(
        lambda row: (
            "NIFTY 50 + F&O"
            if row["is_nifty50"] and row["is_fo_eligible"]
            else "NIFTY 50"
            if row["is_nifty50"]
            else "F&O"
        ),
        axis=1,
    )
    frame["ticker"] = frame["symbol"].map(_to_yahoo_ticker)
    frame = frame.set_index("ticker").sort_index()
    save_to_cache(cache_key, frame)
    return frame


def build_zerodha_login_url(api_key: Optional[str] = None) -> str:
    """Return the Zerodha connect login URL for the provided API key."""
    resolved_api_key = api_key or os.getenv("ZERODHA_API_KEY", "")
    return f"https://kite.zerodha.com/connect/login?v=3&api_key={resolved_api_key}"


def generate_zerodha_session(
    api_key: str,
    api_secret: str,
    request_token: str,
) -> dict:
    """
    Exchange a request token for a live Zerodha access token.
    """
    checksum = hashlib.sha256(f"{api_key}{request_token}{api_secret}".encode("utf-8")).hexdigest()
    response = requests.post(
        f"{ZERODHA_API_BASE}/session/token",
        data={
            "api_key": api_key,
            "request_token": request_token,
            "checksum": checksum,
        },
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "X-Kite-Version": "3",
        },
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("data", payload)
