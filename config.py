"""
config.py — Central Configuration for European Long/Short Equity Research System
================================================================================
Defines investment universe, factor weights, sector/country mappings,
date ranges, and all tunable system parameters.
"""

from datetime import datetime, timedelta

# ============================================================================
# DATE CONFIGURATION
# ============================================================================
def get_date_range():
    """Returns (start_date, end_date, lookback_1y) computed at call time."""
    end = datetime.now()
    return (
        (end - timedelta(days=400)).strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        (end - timedelta(days=365)).strftime("%Y-%m-%d")
    )

# ============================================================================
# EUROPEAN EQUITY UNIVERSE
# ============================================================================

# CAC 40 — France (Paris)
CAC_40 = [
    "AI.PA", "AIR.PA", "ALO.PA", "MT.AS", "CS.PA", "BNP.PA", "EN.PA",
    "CAP.PA", "CA.PA", "ACA.PA", "BN.PA", "DSY.PA", "ENGI.PA", "EL.PA",
    "ERF.PA", "RMS.PA", "KER.PA", "LR.PA", "OR.PA", "MC.PA", "ML.PA",
    "ORA.PA", "RI.PA", "PUB.PA", "RNO.PA", "SAF.PA", "SGO.PA", "SAN.PA",
    "SU.PA", "GLE.PA", "STLAM.PA", "STMPA.PA", "TEP.PA", "HO.PA",
    "TTE.PA", "URW.PA", "VIE.PA", "DG.PA", "VIV.PA", "WLN.PA"
]

# DAX 40 — Germany (Frankfurt)
DAX_40 = [
    "1COV.DE", "ADS.DE", "AIR.DE", "ALV.DE", "BAS.DE", "BAYN.DE",
    "BEI.DE", "BMW.DE", "BNR.DE", "CON.DE", "DB1.DE",
    "DBK.DE", "DHL.DE", "DTE.DE", "DTG.DE", "ENR.DE", "FRE.DE",
    "HEI.DE", "HEN3.DE", "IFX.DE", "MBG.DE", "MRK.DE",
    "MTX.DE", "MUV2.DE", "PAH3.DE", "P911.DE", "QIA.DE", "RHM.DE",
    "RWE.DE", "SAP.DE", "SRT3.DE", "SIE.DE", "SY1.DE", "VOW3.DE",
    "VNA.DE", "ZAL.DE", "HNR1.DE", "SHL.DE", "CBK.DE"
]

# FTSE 100 — United Kingdom (London)
FTSE_100 = [
    "AAL.L", "ABF.L", "ADM.L", "AHT.L", "ANTO.L", "AV.L", "AVST.L",
    "AZN.L", "BA.L", "BARC.L", "BATS.L", "BDEV.L", "BKG.L", "BME.L",
    "BNZL.L", "BP.L", "BRBY.L", "BT-A.L", "CCH.L", "CNA.L", "CPG.L",
    "CRDA.L", "CRH.L", "DCC.L", "DGE.L", "ENT.L", "EXPN.L", "FERG.L",
    "FLTR.L", "GLEN.L", "GSK.L", "HIK.L", "HLMA.L", "HSBA.L", "HSX.L",
    "IAG.L", "IHG.L", "III.L", "IMB.L", "INF.L", "ITRK.L", "JD.L",
    "KGF.L", "LAND.L", "LGEN.L", "LLOY.L", "LSEG.L", "MNG.L", "MRO.L",
    "MNDI.L", "NG.L", "NWG.L", "NXT.L", "PHNX.L", "PRU.L", "PSH.L",
    "PSN.L", "REL.L", "RIO.L", "RKT.L", "RMV.L", "RR.L", "RS1.L",
    "RTO.L", "SBRY.L", "SDR.L", "SGE.L", "SGRO.L", "SHEL.L", "SKG.L",
    "SMDS.L", "SMIN.L", "SMT.L", "SN.L", "SPX.L", "SSE.L", "STAN.L",
    "SVT.L", "TSCO.L", "TW.L", "ULVR.L", "UTG.L", "UU.L", "VOD.L",
    "WPP.L", "WTB.L"
]

# STOXX 600 representative subset (selected large-caps across Europe)
STOXX_SUBSET = [
    "NESN.SW", "NOVN.SW", "ROG.SW", "NOVO-B.CO", "ASML.AS",
    "PHIA.AS", "INGA.AS", "UNA.AS", "ABI.BR", "UCB.BR",
    "NOKIA.HE", "NESTE.HE", "ENEL.MI", "ISP.MI", "UCG.MI",
    "ENI.MI", "RACE.MI", "IBE.MC", "SAN.MC", "TEF.MC",
    "BBVA.MC", "ITX.MC", "EDP.LS", "GALP.LS"
]

# Full universe (de-duplicated)
FULL_UNIVERSE = list(set(CAC_40 + DAX_40 + FTSE_100 + STOXX_SUBSET))

# ============================================================================
# INDEX BENCHMARKS (for market overview & beta calculation)
# ============================================================================
INDEX_TICKERS = {
    "STOXX 600": "STOXX6E.SW",
    "CAC 40": "^FCHI",
    "DAX": "^GDAXI",
    "FTSE 100": "^FTSE",
    "Euro STOXX 50": "^STOXX50E"
}

# ============================================================================
# COUNTRY & SECTOR MAPPINGS
# ============================================================================
COUNTRY_MAP = {}
for t in CAC_40:
    COUNTRY_MAP[t] = "France"
for t in DAX_40:
    COUNTRY_MAP[t] = "Germany"
for t in FTSE_100:
    COUNTRY_MAP[t] = "United Kingdom"
# STOXX subset — assign by exchange suffix
for t in STOXX_SUBSET:
    suffix = t.split(".")[-1]
    country_by_suffix = {
        "SW": "Switzerland", "CO": "Denmark", "AS": "Netherlands",
        "BR": "Belgium", "HE": "Finland", "MI": "Italy",
        "MC": "Spain", "LS": "Portugal"
    }
    COUNTRY_MAP[t] = country_by_suffix.get(suffix, "Europe")

# ============================================================================
# FACTOR MODEL WEIGHTS
# ============================================================================
FACTOR_WEIGHTS = {
    # Value factors (total = 0.35)
    "pe_rank": 0.12,
    "ev_ebitda_rank": 0.12,
    "fcf_yield_rank": 0.11,
    # Quality factors (total = 0.35)
    "roic_rank": 0.12,
    "ebitda_margin_rank": 0.12,
    "leverage_rank": 0.11,
    # Momentum factors (total = 0.30)
    "mom_3m_rank": 0.15,
    "mom_6m_rank": 0.15,
}

# Portfolio construction thresholds
LONG_THRESHOLD_PERCENTILE = 90   # Top 10% → LONG
SHORT_THRESHOLD_PERCENTILE = 10  # Bottom 10% → SHORT

# ============================================================================
# SENTIMENT ANALYSIS CONFIGURATION
# ============================================================================
RSS_FEEDS = {
    "Yahoo Finance EU": "https://finance.yahoo.com/news/rssindex",
    "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
    "MarketWatch Europe": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "CNBC Europe": "https://www.cnbc.com/id/19794221/device/rss/rss.html",
    "Investing.com": "https://www.investing.com/rss/news.rss",
}

# Keywords for event detection
EVENT_KEYWORDS = {
    "earnings_beat": ["earnings beat", "beat estimates", "beat expectations",
                      "better than expected", "topped estimates", "surpassed expectations"],
    "earnings_miss": ["earnings miss", "missed estimates", "missed expectations",
                      "worse than expected", "fell short", "below expectations"],
    "guidance_upgrade": ["guidance upgrade", "raised guidance", "raised outlook",
                         "upgraded forecast", "lifted guidance", "raised full-year"],
    "guidance_downgrade": ["guidance downgrade", "lowered guidance", "cut outlook",
                           "downgraded forecast", "reduced guidance", "lowered full-year"],
    "margin_expansion": ["margin expansion", "margin improvement", "widening margins",
                         "margin beat", "improving profitability"],
    "profit_warning": ["profit warning", "earnings warning", "revenue warning",
                       "issued warning", "downside warning"],
}

# ============================================================================
# DATA PATHS
# ============================================================================
DATA_DIR = "data"
REPORTS_DIR = "reports"
CACHE_EXPIRY_HOURS = 12  # Re-fetch data if cache is older than this

# ============================================================================
# PDF REPORT SETTINGS
# ============================================================================
REPORT_TITLE = "European Long/Short Equity Report"
REPORT_SUBTITLE = "Systematic Factor-Based Research"
TOP_N_LONG = 3   # Number of top long ideas in the report
TOP_N_SHORT = 3  # Number of top short ideas in the report
