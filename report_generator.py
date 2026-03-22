"""
report_generator.py - Markdown-first Indian equity research report generator
===========================================================================
Builds a detailed markdown research report and a styled PDF companion from
the live factor model outputs used by the dashboard.
"""

from __future__ import annotations

import hashlib
import html
import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from config import REPORTS_DIR, TOP_N_LONG, TOP_N_SHORT
from utils import ensure_directories, fmt_large_number, setup_logger, ticker_to_name

logger = setup_logger("report_generator")


_ORIGINAL_MD5 = hashlib.md5


def _compat_md5(*args, **kwargs):
    """Ignore `usedforsecurity` on Python/OpenSSL builds that reject it."""
    try:
        return _ORIGINAL_MD5(*args, **kwargs)
    except TypeError:
        kwargs.pop("usedforsecurity", None)
        return _ORIGINAL_MD5(*args, **kwargs)


hashlib.md5 = _compat_md5

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    REPORTLAB_AVAILABLE = True
    REPORTLAB_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional runtime dependency guard
    REPORTLAB_AVAILABLE = False
    REPORTLAB_IMPORT_ERROR = exc


def _coerce_float(value: object, default: float = 50.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _display_name(row: pd.Series, fallback: str) -> str:
    value = row.get("name")
    return str(value) if pd.notna(value) and str(value).strip() else ticker_to_name(fallback)


def _fmt_pct(value: object, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.{decimals}f}%"


def _fmt_score(value: object) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.1f}"


def _fmt_multiple(value: object) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.1f}x"


def _fmt_plain_number(value: object, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):,.{decimals}f}"


def _safe_cell(value: object) -> str:
    if value is None or pd.isna(value):
        text = "N/A"
    else:
        text = str(value)
    return text.replace("|", "&#124;").replace("\n", "<br>")


def _markdown_table(headers: List[str], rows: List[List[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_safe_cell(value) for value in row) + " |")
    return "\n".join(lines)


def _data_as_of(price_data: Dict[str, pd.DataFrame], index_data: Dict[str, pd.DataFrame]) -> str:
    dates = []
    for frame in list(price_data.values()) + list(index_data.values()):
        if frame is not None and not frame.empty and len(frame.index) > 0:
            dates.append(pd.to_datetime(frame.index[-1]).date())
    return max(dates).isoformat() if dates else datetime.now().date().isoformat()


def _ytd_return(frame: pd.DataFrame) -> Optional[float]:
    if frame is None or frame.empty or "Close" not in frame.columns:
        return None
    closes = frame["Close"].dropna()
    if closes.empty:
        return None
    current_year = closes.index[-1].year
    this_year = closes[closes.index.year == current_year]
    base = this_year.iloc[0] if not this_year.empty else closes.iloc[0]
    if base == 0:
        return None
    return closes.iloc[-1] / base - 1


def _compute_sector_snapshot(scored_universe: pd.DataFrame) -> pd.DataFrame:
    if scored_universe.empty or "sector" not in scored_universe.columns:
        return pd.DataFrame()
    grouped = (
        scored_universe.groupby("sector", dropna=False)
        .agg(
            stocks=("sector", "size"),
            avg_3m_return=("mom_3m", "mean"),
            avg_6m_return=("mom_6m", "mean"),
            median_score=("composite_score", "median"),
            median_pe=("pe_ratio", "median"),
        )
        .reset_index()
        .sort_values(["avg_3m_return", "median_score"], ascending=[False, False])
    )
    return grouped


def _universe_statistics(scored_universe: pd.DataFrame) -> List[List[str]]:
    fo_count = int(scored_universe["is_fo_eligible"].fillna(False).sum()) if "is_fo_eligible" in scored_universe.columns else 0
    return [
        ["Stocks in scored universe", str(len(scored_universe))],
        ["F&O-eligible stocks", str(fo_count)],
        ["Median P/E", _fmt_multiple(scored_universe["pe_ratio"].median()) if "pe_ratio" in scored_universe.columns else "N/A"],
        ["Median EV/EBITDA", _fmt_multiple(scored_universe["ev_ebitda"].median()) if "ev_ebitda" in scored_universe.columns else "N/A"],
        ["Average volatility", _fmt_pct(scored_universe["volatility"].mean()) if "volatility" in scored_universe.columns else "N/A"],
        ["Median ROIC", _fmt_pct(scored_universe["roic"].median()) if "roic" in scored_universe.columns else "N/A"],
        ["Median 3M return", _fmt_pct(scored_universe["mom_3m"].median()) if "mom_3m" in scored_universe.columns else "N/A"],
        ["Median 6M return", _fmt_pct(scored_universe["mom_6m"].median()) if "mom_6m" in scored_universe.columns else "N/A"],
    ]


def _score_medians(scored_universe: pd.DataFrame) -> Dict[str, float]:
    medians = {}
    for column in [
        "pe_ratio",
        "ev_ebitda",
        "ebitda_margin",
        "net_margin",
        "revenue_growth_yoy",
        "roic",
        "leverage",
        "volatility",
        "mom_3m",
        "mom_6m",
    ]:
        medians[column] = pd.to_numeric(scored_universe.get(column), errors="coerce").median() if column in scored_universe.columns else None
    return medians


def _is_financial_entity(row: pd.Series) -> bool:
    text = " ".join(
        str(value).lower()
        for value in [row.get("sector"), row.get("industry")]
        if value is not None and not pd.isna(value)
    )
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


def _metric_rows(row: pd.Series) -> List[List[str]]:
    return [
        ["Market Cap", fmt_large_number(row.get("market_cap"))],
        ["P/E", _fmt_multiple(row.get("pe_ratio"))],
        ["EV/EBITDA", _fmt_multiple(row.get("ev_ebitda"))],
        ["EBITDA margin", _fmt_pct(row.get("ebitda_margin"))],
        ["Net margin", _fmt_pct(row.get("net_margin"))],
        ["Revenue Growth (YoY)", _fmt_pct(row.get("revenue_growth_yoy"))],
        ["ROIC", _fmt_pct(row.get("roic"))],
        ["Leverage (ND/EBITDA)", _fmt_multiple(row.get("leverage"))],
        ["Volatility", _fmt_pct(row.get("volatility"))],
        ["3M return", _fmt_pct(row.get("mom_3m"))],
        ["6M return", _fmt_pct(row.get("mom_6m"))],
    ]


def _driver_label(long_row: pd.Series, short_row: pd.Series) -> str:
    drivers = {
        "Value": _coerce_float(long_row.get("value_score")) - _coerce_float(short_row.get("value_score")),
        "Quality": _coerce_float(long_row.get("quality_score")) - _coerce_float(short_row.get("quality_score")),
        "Momentum": _coerce_float(long_row.get("momentum_score")) - _coerce_float(short_row.get("momentum_score")),
    }
    return max(drivers, key=lambda key: abs(drivers[key]))


def _long_strength_text(row: pd.Series) -> str:
    strengths = {
        "value": _coerce_float(row.get("value_score")),
        "quality": _coerce_float(row.get("quality_score")),
        "momentum": _coerce_float(row.get("momentum_score")),
    }
    dominant = max(strengths, key=strengths.get)
    is_financial = _is_financial_entity(row)
    if dominant == "value":
        if is_financial:
            return (
                f"Valuation is doing the work here: P/E at {_fmt_multiple(row.get('pe_ratio'))} leaves the stock looking inexpensive against the broader market without leaning on industrial-company multiples."
            )
        return (
            f"Valuation does most of the heavy lifting here: P/E at {_fmt_multiple(row.get('pe_ratio'))} "
            f"and EV/EBITDA at {_fmt_multiple(row.get('ev_ebitda'))} keep the stock competitive even before "
            "leaning on a rerating argument."
        )
    if dominant == "quality":
        if is_financial:
            return (
                f"For a financial name, the cleaner signal is earnings quality rather than ROIC: net margin is {_fmt_pct(row.get('net_margin'))}, while the quality bucket is still scoring {_fmt_score(row.get('quality_score'))}/100."
            )
        return (
            f"Quality is the standout driver with EBITDA margin at {_fmt_pct(row.get('ebitda_margin'))}, "
            f"net margin at {_fmt_pct(row.get('net_margin'))}, and ROIC at {_fmt_pct(row.get('roic'))}, "
            "which points to stronger operating discipline than most peers."
        )
    return (
        f"Momentum is already confirming the setup, with 3-month and 6-month returns of "
        f"{_fmt_pct(row.get('mom_3m'))} and {_fmt_pct(row.get('mom_6m'))}. That improves the odds that "
        "incremental flows keep chasing the winner."
    )


def _short_weakness_text(row: pd.Series) -> str:
    weaknesses = {
        "value": _coerce_float(100 - _coerce_float(row.get("value_score"))),
        "quality": _coerce_float(100 - _coerce_float(row.get("quality_score"))),
        "momentum": _coerce_float(100 - _coerce_float(row.get("momentum_score"))),
    }
    dominant = max(weaknesses, key=weaknesses.get)
    is_financial = _is_financial_entity(row)
    if dominant == "value":
        if is_financial:
            return (
                f"The valuation cushion is thin for a financial name: P/E at {_fmt_multiple(row.get('pe_ratio'))} is still not low enough to offset a weak overall ranking."
            )
        return (
            f"The valuation cushion is thin: P/E at {_fmt_multiple(row.get('pe_ratio'))} and EV/EBITDA at "
            f"{_fmt_multiple(row.get('ev_ebitda'))} still look demanding for a stock with a weak overall score."
        )
    if dominant == "quality":
        if is_financial:
            return (
                f"Fundamental quality is not helping. Net margin is {_fmt_pct(row.get('net_margin'))}, revenue growth is {_fmt_pct(row.get('revenue_growth_yoy'))}, and the quality bucket sits at only {_fmt_score(row.get('quality_score'))}/100."
            )
        return (
            f"Operating quality is the core problem. EBITDA margin is {_fmt_pct(row.get('ebitda_margin'))}, "
            f"net margin is {_fmt_pct(row.get('net_margin'))}, and ROIC is {_fmt_pct(row.get('roic'))}, "
            "which leaves little margin for execution mistakes."
        )
    return (
        f"Price action is already weak with 3-month and 6-month returns of {_fmt_pct(row.get('mom_3m'))} "
        f"and {_fmt_pct(row.get('mom_6m'))}. That keeps the stock exposed if risk appetite stays selective."
    )


def _build_long_thesis(row: pd.Series, medians: Dict[str, float]) -> List[str]:
    name = _display_name(row, str(row.name))
    sector = row.get("sector", "its sector")
    is_financial = _is_financial_entity(row)
    valuation_line = (
        f"{name} is one of the cleaner longs in {sector} because it pairs a composite score of "
        f"{_fmt_score(row.get('composite_score'))}/100 with valuation at P/E {_fmt_multiple(row.get('pe_ratio'))}, "
        f"versus a universe median of {_fmt_multiple(medians.get('pe_ratio'))}."
        if is_financial
        else (
            f"{name} is one of the cleaner longs in {sector} because it pairs a composite score of "
            f"{_fmt_score(row.get('composite_score'))}/100 with valuation at P/E {_fmt_multiple(row.get('pe_ratio'))} "
            f"and EV/EBITDA {_fmt_multiple(row.get('ev_ebitda'))}, versus universe medians of "
            f"{_fmt_multiple(medians.get('pe_ratio'))} and {_fmt_multiple(medians.get('ev_ebitda'))}."
        )
    )
    balance_line = (
        f"For a financial business, the cleaner read is earnings quality and growth: net margin is {_fmt_pct(row.get('net_margin'))} and revenue growth is {_fmt_pct(row.get('revenue_growth_yoy'))}, which is enough to keep the factor signal intact."
        if is_financial
        else (
            f"Balance-sheet risk is contained with leverage at {_fmt_multiple(row.get('leverage'))}, while "
            f"revenue growth of {_fmt_pct(row.get('revenue_growth_yoy'))} gives the market a fundamental reason "
            "to keep rewarding the factor signal."
        )
    )
    thesis = [
        valuation_line,
        _long_strength_text(row),
        balance_line,
    ]
    return thesis


def _build_short_thesis(row: pd.Series, medians: Dict[str, float]) -> List[str]:
    name = _display_name(row, str(row.name))
    sector = row.get("sector", "its sector")
    is_financial = _is_financial_entity(row)
    valuation_line = (
        f"Compared with a universe median of {_fmt_multiple(medians.get('pe_ratio'))} P/E, the stock does not offer enough valuation support for a financial name with a weak factor profile."
        if is_financial
        else (
            f"Compared with universe medians of {_fmt_multiple(medians.get('pe_ratio'))} P/E and "
            f"{_fmt_multiple(medians.get('ev_ebitda'))} EV/EBITDA, the stock does not offer enough valuation support "
            "to offset soft fundamentals or poor tape action."
        )
    )
    thesis = [
        (
            f"{name} falls into the short book with a composite score of {_fmt_score(row.get('composite_score'))}/100, "
            f"which is weak inside {sector} and sits well below the quality of names making the long side."
        ),
        _short_weakness_text(row),
        valuation_line,
    ]
    return thesis


def _build_catalysts(row: pd.Series, side: str) -> List[str]:
    sector = row.get("sector", "sector")
    catalysts = []

    revenue_growth = row.get("revenue_growth_yoy")
    if pd.notna(revenue_growth):
        if side == "LONG":
            catalysts.append(
                f"The next earnings print can validate current revenue growth of {_fmt_pct(revenue_growth)} and reinforce the market's confidence in the {sector} earnings path."
            )
        else:
            catalysts.append(
                f"The next earnings print is a risk event: if revenue growth stays around {_fmt_pct(revenue_growth)} or slips further, the market could keep cutting expectations for this {sector} name."
            )

    sentiment_count = row.get("sentiment_count", 0)
    sentiment_score = row.get("sentiment_score")
    if pd.notna(sentiment_score) and sentiment_count and float(sentiment_count) > 0:
        if side == "LONG":
            catalysts.append(
                f"News flow is supportive with sentiment at {_fmt_plain_number(sentiment_score, 2)} across {int(sentiment_count)} recent items, which can help extend the rerating."
            )
        else:
            catalysts.append(
                f"News flow is only mildly supportive at {_fmt_plain_number(sentiment_score, 2)} across {int(sentiment_count)} items; any deterioration would quickly reinforce the short case."
            )
    elif pd.notna(row.get("mom_6m")):
        if side == "LONG":
            catalysts.append(
                f"Six-month price strength of {_fmt_pct(row.get('mom_6m'))} can keep drawing incremental flows if sector leadership remains intact."
            )
        else:
            catalysts.append(
                f"The weak six-month tape of {_fmt_pct(row.get('mom_6m'))} can extend if investors continue rotating away from lower-ranked cyclicals and balance-sheet stories."
            )

    return catalysts[:2]


def _build_risks(row: pd.Series, side: str, medians: Dict[str, float]) -> List[str]:
    risks = []
    is_financial = _is_financial_entity(row)
    if side == "LONG":
        risks.append(
            f"Volatility at {_fmt_pct(row.get('volatility'))} means even a high-ranked long can mean-revert sharply in a market-wide risk-off move."
        )
        if is_financial:
            risks.append(
                f"For a financial name, the key risk is a deterioration in earnings quality: net margin is {_fmt_pct(row.get('net_margin'))} today, and any pressure on that line could weaken the thesis quickly."
            )
        else:
            risks.append(
                f"If margins or ROIC soften from current levels of {_fmt_pct(row.get('ebitda_margin'))} and {_fmt_pct(row.get('roic'))}, the premium embedded in the factor score could compress quickly."
            )
        if pd.notna(row.get("pe_ratio")) and pd.notna(medians.get("pe_ratio")) and row.get("pe_ratio") > medians.get("pe_ratio"):
            risks.append(
                "This is not a deep-value long, so a miss on earnings or guidance can still trigger valuation de-rating."
            )
    else:
        if is_financial:
            risks.append(
                f"A positive surprise on growth or profitability would matter because the stock is already carrying weak expectations at net margin {_fmt_pct(row.get('net_margin'))} and revenue growth {_fmt_pct(row.get('revenue_growth_yoy'))}."
            )
        else:
            risks.append(
                f"A positive surprise on margins or growth would matter because the stock is already carrying weak expectations at EBITDA margin {_fmt_pct(row.get('ebitda_margin'))} and revenue growth {_fmt_pct(row.get('revenue_growth_yoy'))}."
            )
        risks.append(
            f"Volatility of {_fmt_pct(row.get('volatility'))} leaves room for sharp countertrend rallies that can hurt short timing even when the broader thesis stays intact."
        )
        if pd.notna(row.get("sentiment_score")) and row.get("sentiment_score", 0) > 0.1:
            risks.append(
                f"Sentiment is not outright bearish at {_fmt_plain_number(row.get('sentiment_score'), 2)}, so the short can squeeze if the market latches onto incremental good news."
            )
    return risks[:3]


def _top_names(book: pd.DataFrame, count: int) -> str:
    if book is None or book.empty:
        return ""
    names = []
    for ticker, row in book.head(count).iterrows():
        names.append(_display_name(row, ticker))
    return ", ".join(names)


def _executive_summary_paragraphs(
    scored_universe: pd.DataFrame,
    long_book: pd.DataFrame,
    short_book: pd.DataFrame,
    index_data: Dict[str, pd.DataFrame],
) -> List[str]:
    index_returns = []
    for name, frame in index_data.items():
        value = _ytd_return(frame)
        if value is not None:
            index_returns.append(f"{name} {_fmt_pct(value)}")
    index_context = ", ".join(index_returns) if index_returns else "index performance unavailable"

    avg_vol = _fmt_pct(scored_universe["volatility"].mean()) if "volatility" in scored_universe.columns and not scored_universe.empty else "N/A"
    median_pe = _fmt_multiple(scored_universe["pe_ratio"].median()) if "pe_ratio" in scored_universe.columns and not scored_universe.empty else "N/A"
    median_ev = _fmt_multiple(scored_universe["ev_ebitda"].median()) if "ev_ebitda" in scored_universe.columns and not scored_universe.empty else "N/A"
    top_longs = _top_names(long_book, TOP_N_LONG)
    top_shorts = _top_names(short_book, TOP_N_SHORT)

    return [
        (
            f"Indian equities are trading through a mixed but still investable backdrop, with {index_context}. "
            f"Across the screened universe, average volatility is {avg_vol}, median P/E is {median_pe}, and median EV/EBITDA is {median_ev}, which points to a market that is not broadly cheap but still offers meaningful stock-level dispersion."
        ),
        (
            f"The long book is led by {top_longs or 'the highest-ranked names'}, where valuation, operating quality, and momentum are reinforcing each other. "
            f"The short book is led by {top_shorts or 'the weakest-ranked names'}, where lower-quality fundamentals and softer price action are leaving certain F&O-eligible stocks exposed."
        ),
        (
            "The practical takeaway is that the opportunity set supports both outright longs and cleaner relative-value trades inside the same sectors. "
            "The report below focuses on names where factor alignment is strongest rather than offering a generic market recap."
        ),
    ]


def _index_rows(index_data: Dict[str, pd.DataFrame]) -> List[List[str]]:
    rows = []
    for name, frame in index_data.items():
        latest = frame["Close"].iloc[-1] if frame is not None and not frame.empty and "Close" in frame.columns else None
        rows.append([name, _fmt_pct(_ytd_return(frame)), _fmt_plain_number(latest, 0)])
    return rows or [["N/A", "N/A", "N/A"]]


def _sector_rows(scored_universe: pd.DataFrame) -> List[List[str]]:
    snapshot = _compute_sector_snapshot(scored_universe)
    rows = []
    if not snapshot.empty:
        for _, row in snapshot.iterrows():
            rows.append(
                [
                    row["sector"],
                    str(int(row["stocks"])),
                    _fmt_pct(row["avg_3m_return"]),
                    _fmt_pct(row["avg_6m_return"]),
                    _fmt_score(row["median_score"]),
                    _fmt_multiple(row["median_pe"]),
                ]
            )
    return rows or [["N/A", "0", "N/A", "N/A", "N/A", "N/A"]]


def _appendix_rows(scored_universe: pd.DataFrame) -> List[List[str]]:
    top20 = (
        scored_universe.sort_values("composite_score", ascending=False).head(20)
        if not scored_universe.empty and "composite_score" in scored_universe.columns
        else pd.DataFrame()
    )
    rows = []
    if not top20.empty:
        for ticker, row in top20.iterrows():
            rows.append(
                [
                    row.get("symbol", ticker),
                    _display_name(row, ticker),
                    row.get("sector", "Unknown"),
                    _fmt_score(row.get("composite_score")),
                    _fmt_score(row.get("value_score")),
                    _fmt_score(row.get("quality_score")),
                    _fmt_score(row.get("momentum_score")),
                    _fmt_multiple(row.get("pe_ratio")),
                    _fmt_multiple(row.get("ev_ebitda")),
                    _fmt_pct(row.get("mom_3m")),
                    _fmt_pct(row.get("mom_6m")),
                ]
            )
    return rows or [["N/A"] * 11]


def _pair_quality_gap_text(long_row: pd.Series, short_row: pd.Series) -> str:
    if pd.notna(long_row.get("roic")) and pd.notna(short_row.get("roic")):
        return (
            f"The long carries EBITDA margin {_fmt_pct(long_row.get('ebitda_margin'))} and ROIC "
            f"{_fmt_pct(long_row.get('roic'))} against {_fmt_pct(short_row.get('ebitda_margin'))} and "
            f"{_fmt_pct(short_row.get('roic'))} on the short."
        )
    if pd.notna(long_row.get("net_margin")) and pd.notna(short_row.get("net_margin")):
        return (
            f"On comparable profitability metrics, the long posts net margin {_fmt_pct(long_row.get('net_margin'))} "
            f"versus {_fmt_pct(short_row.get('net_margin'))} on the short, while quality scores are "
            f"{_fmt_score(long_row.get('quality_score'))} versus {_fmt_score(short_row.get('quality_score'))}."
        )
    return (
        f"The quality bucket still favours the long at {_fmt_score(long_row.get('quality_score'))} versus "
        f"{_fmt_score(short_row.get('quality_score'))} on the short."
    )


def _relative_trade_payloads(
    scored_universe: pd.DataFrame,
    long_book: pd.DataFrame,
    short_book: pd.DataFrame,
    pairs: List[Dict],
) -> List[Dict[str, str]]:
    payloads = []
    for pair in pairs[:5]:
        long_ticker = pair.get("long_ticker")
        short_ticker = pair.get("short_ticker")
        if long_ticker in getattr(long_book, "index", []):
            long_row = long_book.loc[long_ticker]
        elif long_ticker in scored_universe.index:
            long_row = scored_universe.loc[long_ticker]
        else:
            continue
        if short_ticker in getattr(short_book, "index", []):
            short_row = short_book.loc[short_ticker]
        elif short_ticker in scored_universe.index:
            short_row = scored_universe.loc[short_ticker]
        else:
            continue
        driver = _driver_label(long_row, short_row)
        payloads.append(
            {
                "title": f"Long {pair['long_name']} / Short {pair['short_name']}",
                "sector": pair.get("sector", "Cross-Sector"),
                "ratio": "1:1",
                "driver": driver,
                "score_gap": (
                    f"{pair['long_name']} scores {_fmt_score(long_row.get('composite_score'))} versus "
                    f"{_fmt_score(short_row.get('composite_score'))} for {pair['short_name']}, with {driver.lower()} "
                    "showing the widest separation."
                ),
                "quality_gap": _pair_quality_gap_text(long_row, short_row),
                "momentum_gap": (
                    f"On momentum, the long is at {_fmt_pct(long_row.get('mom_6m'))} over 6 months versus "
                    f"{_fmt_pct(short_row.get('mom_6m'))} for the short, giving the spread a clear tape-confirmation angle."
                ),
                "chart": f"Chart: 12-month relative return spread of {pair['long_name']} minus {pair['short_name']}.",
            }
        )
    return payloads


def _stock_markdown_section(row: pd.Series, side: str, medians: Dict[str, float], index_name: str = "NIFTY 50") -> str:
    name = _display_name(row, str(row.name))
    symbol = row.get("symbol", row.name)
    thesis = _build_long_thesis(row, medians) if side == "LONG" else _build_short_thesis(row, medians)
    catalysts = _build_catalysts(row, side)
    risks = _build_risks(row, side, medians)

    lines = [
        f"### {name} ({symbol})",
        f"Chart: {name} price versus {index_name} over the last 6 months, with volume and factor-score callouts.",
        "",
        "**Investment Thesis**",
    ]
    lines.extend(f"- {item}" for item in thesis)
    lines.extend(["", _markdown_table(["Metric", "Value"], _metric_rows(row)), "", "**Catalysts**"])
    lines.extend(
        f"- {item}" for item in (catalysts or ["Near-term earnings, sector rotation, and price confirmation will determine whether the current setup strengthens or fades."])
    )
    lines.extend(["", "**Risks**"])
    lines.extend(f"- {item}" for item in risks)
    lines.append("")
    return "\n".join(lines)


def _idea_markdown_section(title: str, book: pd.DataFrame, medians: Dict[str, float], side: str) -> str:
    lines = [title, ""]
    if book.empty:
        lines.append("No qualifying ideas were available for this side of the book.")
        lines.append("")
        return "\n".join(lines)

    limit = TOP_N_LONG if side == "LONG" else TOP_N_SHORT
    for _, row in book.head(limit).iterrows():
        lines.append(_stock_markdown_section(row, side, medians))
    return "\n".join(lines)


def _markdown_report_content(
    scored_universe: pd.DataFrame,
    long_book: pd.DataFrame,
    short_book: pd.DataFrame,
    price_data: Dict[str, pd.DataFrame],
    index_data: Dict[str, pd.DataFrame],
    pairs: List[Dict],
) -> str:
    as_of = _data_as_of(price_data, index_data)
    medians = _score_medians(scored_universe)
    executive = _executive_summary_paragraphs(scored_universe, long_book, short_book, index_data)
    relative_payloads = _relative_trade_payloads(scored_universe, long_book, short_book, pairs)

    sections = [
        f"# Indian Equity Research Report\n\n**Data as of:** {as_of}\n",
        "## 1. Executive Summary\n\n" + "\n\n".join(executive) + "\n",
        "\n".join(
            [
                "## 2. Market Overview",
                "Chart: YTD normalized performance of NIFTY 50, NIFTY Bank, and Sensex.",
                "",
                "### YTD Index Returns",
                _markdown_table(["Index", "YTD Return", "Latest Level"], _index_rows(index_data)),
                "",
                "### Sector Performance Snapshot",
                _markdown_table(
                    ["Sector", "Stocks", "Avg 3M Return", "Avg 6M Return", "Median Composite", "Median P/E"],
                    _sector_rows(scored_universe),
                ),
                "",
                "### Universe Statistics",
                _markdown_table(["Statistic", "Value"], _universe_statistics(scored_universe)),
                "",
            ]
        ),
        _idea_markdown_section("## 3. Top Long Ideas", long_book, medians, "LONG"),
        _idea_markdown_section("## 4. Top Short Ideas", short_book, medians, "SHORT"),
    ]

    relative_lines = ["## 5. Relative Trades", ""]
    if relative_payloads:
        for payload in relative_payloads:
            relative_lines.extend(
                [
                    f"### {payload['title']}",
                    f"- Sector: {payload['sector']}",
                    f"- Notional ratio: {payload['ratio']}",
                    f"- Key spread driver: {payload['driver']}",
                    f"- {payload['score_gap']}",
                    f"- {payload['quality_gap']}",
                    f"- {payload['momentum_gap']}",
                    payload["chart"],
                    "",
                ]
            )
    else:
        relative_lines.append("No sector-matched pair trades were available from the current books.")
        relative_lines.append("")
    sections.append("\n".join(relative_lines))

    sections.append(
        "\n".join(
            [
                "## 6. Appendix",
                "### Top 20 Universe Ranking",
                _markdown_table(
                    ["Symbol", "Name", "Sector", "Composite", "Value", "Quality", "Momentum", "P/E", "EV/EBITDA", "3M", "6M"],
                    _appendix_rows(scored_universe),
                ),
                "",
                "### Factor Definitions",
                "- Value: rewards cheaper valuation and higher free-cash-flow yield. Higher score means the stock looks cheaper on the current metrics set.",
                "- Quality: rewards stronger margins, better ROIC, and lower leverage. Higher score means stronger operating quality and cleaner balance-sheet profile.",
                "- Momentum: rewards stronger 3-month and 6-month price performance. Higher score means the market is already validating the story.",
                "- Scoring range: each factor and the composite are normalized to a 0-100 scale. Around 50 is neutral, the long book is built from the top decile, and the short book is built from the bottom decile of F&O-eligible names.",
                "- Composite methodology: the composite score is a weighted blend of the underlying sub-factor ranks, so the displayed Value, Quality, and Momentum bucket scores do not average exactly to the composite.",
                "",
            ]
        )
    )
    sections.append(
        "## 7. Disclaimer\nThis report is for informational purposes only and reflects a systematic interpretation of market data, factor signals, and recent price action. It is not investment advice, not a solicitation to buy or sell securities, and should not be used as the sole basis for any investment decision. Market conditions can change quickly, data may be revised, and all investing involves risk, including the risk of capital loss.\n"
    )

    return "\n".join(section.strip() for section in sections if section).strip() + "\n"


if REPORTLAB_AVAILABLE:
    NAVY = colors.HexColor("#11253d")
    BLUE = colors.HexColor("#1f4e79")
    TEAL = colors.HexColor("#2e8b8b")
    LIGHT_GREY = colors.HexColor("#f5f7fa")
    MID_GREY = colors.HexColor("#d9dee7")
    DARK_GREY = colors.HexColor("#55606e")


    def _get_styles() -> Dict[str, ParagraphStyle]:
        base = getSampleStyleSheet()
        return {
            "Title": ParagraphStyle(
                "Title",
                parent=base["Title"],
                fontName="Helvetica-Bold",
                fontSize=22,
                leading=28,
                textColor=NAVY,
                spaceAfter=6,
            ),
            "Meta": ParagraphStyle(
                "Meta",
                parent=base["Normal"],
                fontName="Helvetica",
                fontSize=10,
                leading=14,
                textColor=DARK_GREY,
                spaceAfter=4,
            ),
            "Section": ParagraphStyle(
                "Section",
                parent=base["Heading1"],
                fontName="Helvetica-Bold",
                fontSize=15,
                leading=19,
                textColor=NAVY,
                spaceBefore=10,
                spaceAfter=6,
            ),
            "Subsection": ParagraphStyle(
                "Subsection",
                parent=base["Heading2"],
                fontName="Helvetica-Bold",
                fontSize=11,
                leading=14,
                textColor=BLUE,
                spaceBefore=8,
                spaceAfter=4,
            ),
            "StockTitle": ParagraphStyle(
                "StockTitle",
                parent=base["Heading3"],
                fontName="Helvetica-Bold",
                fontSize=11,
                leading=14,
                textColor=NAVY,
                spaceBefore=8,
                spaceAfter=4,
            ),
            "Body": ParagraphStyle(
                "Body",
                parent=base["BodyText"],
                fontName="Helvetica",
                fontSize=9,
                leading=13,
                textColor=colors.black,
                spaceAfter=5,
            ),
            "Bullet": ParagraphStyle(
                "Bullet",
                parent=base["BodyText"],
                fontName="Helvetica",
                fontSize=9,
                leading=12,
                leftIndent=10,
                firstLineIndent=0,
                textColor=colors.black,
                spaceAfter=3,
            ),
            "Caption": ParagraphStyle(
                "Caption",
                parent=base["Italic"],
                fontName="Helvetica-Oblique",
                fontSize=8,
                leading=11,
                textColor=TEAL,
                spaceAfter=5,
            ),
            "Disclaimer": ParagraphStyle(
                "Disclaimer",
                parent=base["BodyText"],
                fontName="Helvetica-Oblique",
                fontSize=7,
                leading=10,
                textColor=DARK_GREY,
                spaceAfter=3,
            ),
            "TableHeader": ParagraphStyle(
                "TableHeader",
                parent=base["BodyText"],
                fontName="Helvetica-Bold",
                fontSize=8,
                leading=10,
                textColor=colors.white,
            ),
            "TableCell": ParagraphStyle(
                "TableCell",
                parent=base["BodyText"],
                fontName="Helvetica",
                fontSize=8,
                leading=10,
                textColor=colors.black,
            ),
            "TableCellSmall": ParagraphStyle(
                "TableCellSmall",
                parent=base["BodyText"],
                fontName="Helvetica",
                fontSize=7,
                leading=9,
                textColor=colors.black,
            ),
        }


    def _escape_pdf(value: object) -> str:
        return html.escape("N/A" if value is None or pd.isna(value) else str(value))


    def _make_pdf_table(
        headers: List[str],
        rows: List[List[object]],
        styles: Dict[str, ParagraphStyle],
        col_widths: Optional[List[float]] = None,
        small: bool = False,
    ) -> Table:
        cell_style = styles["TableCellSmall"] if small else styles["TableCell"]
        data = [[Paragraph(_escape_pdf(header), styles["TableHeader"]) for header in headers]]
        for row in rows:
            data.append([Paragraph(_escape_pdf(value), cell_style) for value in row])

        table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
        table_style = [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.35, MID_GREY),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        for idx in range(1, len(data)):
            background = LIGHT_GREY if idx % 2 else colors.white
            table_style.append(("BACKGROUND", (0, idx), (-1, idx), background))
        table.setStyle(TableStyle(table_style))
        return table


    def _append_bullets(story: List[object], items: List[str], styles: Dict[str, ParagraphStyle]) -> None:
        for item in items:
            story.append(Paragraph(f"&bull; {_escape_pdf(item)}", styles["Bullet"]))


    def _draw_footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(DARK_GREY)
        canvas.drawString(doc.leftMargin, 10 * mm, "Indian Long/Short Equity Research")
        canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 10 * mm, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()


    def _append_stock_pdf_section(
        story: List[object],
        row: pd.Series,
        side: str,
        medians: Dict[str, float],
        styles: Dict[str, ParagraphStyle],
        index_name: str = "NIFTY 50",
    ) -> None:
        name = _display_name(row, str(row.name))
        symbol = row.get("symbol", row.name)
        thesis = _build_long_thesis(row, medians) if side == "LONG" else _build_short_thesis(row, medians)
        catalysts = _build_catalysts(row, side)
        risks = _build_risks(row, side, medians)

        story.append(Paragraph(f"{_escape_pdf(name)} ({_escape_pdf(symbol)})", styles["StockTitle"]))
        story.append(
            Paragraph(
                _escape_pdf(f"Chart: {name} price versus {index_name} over the last 6 months, with volume and factor-score callouts."),
                styles["Caption"],
            )
        )
        story.append(Paragraph("Investment Thesis", styles["Subsection"]))
        _append_bullets(story, thesis, styles)
        story.append(Spacer(1, 2))
        story.append(
            _make_pdf_table(
                ["Metric", "Value"],
                _metric_rows(row),
                styles,
                col_widths=[62 * mm, 52 * mm],
            )
        )
        story.append(Spacer(1, 5))
        story.append(Paragraph("Catalysts", styles["Subsection"]))
        _append_bullets(
            story,
            catalysts or ["Near-term earnings, sector rotation, and price confirmation will determine whether the current setup strengthens or fades."],
            styles,
        )
        story.append(Paragraph("Risks", styles["Subsection"]))
        _append_bullets(story, risks, styles)
        story.append(HRFlowable(width="100%", thickness=0.4, color=MID_GREY, spaceBefore=5, spaceAfter=8))


    def _build_pdf_report(
        output_path: str,
        scored_universe: pd.DataFrame,
        long_book: pd.DataFrame,
        short_book: pd.DataFrame,
        price_data: Dict[str, pd.DataFrame],
        index_data: Dict[str, pd.DataFrame],
        pairs: List[Dict],
    ) -> str:
        styles = _get_styles()
        as_of = _data_as_of(price_data, index_data)
        medians = _score_medians(scored_universe)
        executive = _executive_summary_paragraphs(scored_universe, long_book, short_book, index_data)
        relative_payloads = _relative_trade_payloads(scored_universe, long_book, short_book, pairs)

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            leftMargin=14 * mm,
            rightMargin=14 * mm,
            topMargin=15 * mm,
            bottomMargin=16 * mm,
            title="Indian Equity Research Report",
            author="Indian Long/Short Equity Research System",
        )

        story: List[object] = [
            Paragraph("Indian Equity Research Report", styles["Title"]),
            Paragraph(f"Data as of: {as_of}", styles["Meta"]),
            Paragraph("Universe: NIFTY 50 plus F&O-eligible Indian equities.", styles["Meta"]),
            Spacer(1, 6),
        ]

        story.append(Paragraph("1. Executive Summary", styles["Section"]))
        for paragraph in executive:
            story.append(Paragraph(_escape_pdf(paragraph), styles["Body"]))

        story.append(Paragraph("2. Market Overview", styles["Section"]))
        story.append(Paragraph("Chart: YTD normalized performance of NIFTY 50, NIFTY Bank, and Sensex.", styles["Caption"]))
        story.append(Paragraph("YTD Index Returns", styles["Subsection"]))
        story.append(
            _make_pdf_table(
                ["Index", "YTD Return", "Latest Level"],
                _index_rows(index_data),
                styles,
                col_widths=[62 * mm, 32 * mm, 38 * mm],
            )
        )
        story.append(Spacer(1, 6))
        story.append(Paragraph("Sector Performance Snapshot", styles["Subsection"]))
        story.append(
            _make_pdf_table(
                ["Sector", "Stocks", "Avg 3M Return", "Avg 6M Return", "Median Composite", "Median P/E"],
                _sector_rows(scored_universe),
                styles,
                col_widths=[42 * mm, 16 * mm, 25 * mm, 25 * mm, 26 * mm, 24 * mm],
                small=True,
            )
        )
        story.append(Spacer(1, 6))
        story.append(Paragraph("Universe Statistics", styles["Subsection"]))
        story.append(
            _make_pdf_table(
                ["Statistic", "Value"],
                _universe_statistics(scored_universe),
                styles,
                col_widths=[72 * mm, 42 * mm],
            )
        )

        story.append(Paragraph("3. Top Long Ideas", styles["Section"]))
        if long_book.empty:
            story.append(Paragraph("No qualifying long ideas were available in the current run.", styles["Body"]))
        else:
            for _, row in long_book.head(TOP_N_LONG).iterrows():
                _append_stock_pdf_section(story, row, "LONG", medians, styles)

        story.append(Paragraph("4. Top Short Ideas", styles["Section"]))
        if short_book.empty:
            story.append(Paragraph("No qualifying short ideas were available in the current run.", styles["Body"]))
        else:
            for _, row in short_book.head(TOP_N_SHORT).iterrows():
                _append_stock_pdf_section(story, row, "SHORT", medians, styles)

        story.append(Paragraph("5. Relative Trades", styles["Section"]))
        if relative_payloads:
            for payload in relative_payloads:
                story.append(Paragraph(_escape_pdf(payload["title"]), styles["StockTitle"]))
                _append_bullets(
                    story,
                    [
                        f"Sector: {payload['sector']}",
                        f"Notional ratio: {payload['ratio']}",
                        f"Key spread driver: {payload['driver']}",
                        payload["score_gap"],
                        payload["quality_gap"],
                        payload["momentum_gap"],
                    ],
                    styles,
                )
                story.append(Paragraph(_escape_pdf(payload["chart"]), styles["Caption"]))
                story.append(Spacer(1, 4))
        else:
            story.append(Paragraph("No sector-matched pair trades were available from the current books.", styles["Body"]))

        story.append(Paragraph("6. Appendix", styles["Section"]))
        story.append(Paragraph("Top 20 Universe Ranking", styles["Subsection"]))
        appendix_rows = []
        for row in _appendix_rows(scored_universe):
            appendix_rows.append(row[:7])
        story.append(
            _make_pdf_table(
                ["Symbol", "Name", "Sector", "Composite", "Value", "Quality", "Momentum"],
                appendix_rows or [["N/A"] * 7],
                styles,
                col_widths=[21 * mm, 46 * mm, 31 * mm, 18 * mm, 18 * mm, 18 * mm, 18 * mm],
                small=True,
            )
        )
        story.append(Spacer(1, 6))
        story.append(Paragraph("Factor Definitions", styles["Subsection"]))
        _append_bullets(
            story,
            [
                "Value rewards cheaper valuation and higher free-cash-flow yield. Higher scores indicate cheaper stocks on the current metrics set.",
                "Quality rewards stronger margins, better ROIC, and lower leverage. Higher scores indicate better operating quality and cleaner balance sheets.",
                "Momentum rewards stronger 3-month and 6-month price performance. Higher scores indicate that market action is confirming the story.",
                "Each factor and the composite are normalized to a 0-100 scale. Around 50 is neutral, the long book is drawn from the top decile, and the short book is drawn from the bottom decile of F&O-eligible names.",
                "The composite score is a weighted blend of the underlying sub-factor ranks, so the displayed Value, Quality, and Momentum bucket scores do not average exactly to the composite.",
            ],
            styles,
        )

        story.append(Paragraph("7. Disclaimer", styles["Section"]))
        story.append(
            Paragraph(
                _escape_pdf(
                    "This report is for informational purposes only and reflects a systematic interpretation of market data, factor signals, and recent price action. "
                    "It is not investment advice, not a solicitation to buy or sell securities, and should not be used as the sole basis for any investment decision. "
                    "Market conditions can change quickly, data may be revised, and all investing involves risk, including the risk of capital loss."
                ),
                styles["Disclaimer"],
            )
        )

        doc.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
        logger.info("PDF report generated successfully: %s", output_path)
        return output_path


def generate_report(
    scored_universe: pd.DataFrame,
    long_book: pd.DataFrame,
    short_book: pd.DataFrame,
    price_data: Dict[str, pd.DataFrame],
    index_data: Dict[str, pd.DataFrame],
    pairs: List[Dict] = None,
    backtest_results: Dict = None,
    output_filename: str = None,
) -> Dict[str, Optional[str]]:
    """
    Generate the research report bundle.

    Returns a dictionary with markdown and PDF artifact paths.
    """
    del backtest_results
    ensure_directories()

    if output_filename is None:
        base_name = f"indian_equity_research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        base_name = os.path.splitext(output_filename)[0]

    markdown_path = os.path.join(REPORTS_DIR, f"{base_name}.md")
    pdf_path = os.path.join(REPORTS_DIR, f"{base_name}.pdf")
    report_pairs = pairs or []

    markdown_content = _markdown_report_content(
        scored_universe=scored_universe,
        long_book=long_book,
        short_book=short_book,
        price_data=price_data,
        index_data=index_data,
        pairs=report_pairs,
    )
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write(markdown_content)
    logger.info("Markdown report generated successfully: %s", markdown_path)

    pdf_output = None
    if REPORTLAB_AVAILABLE:
        pdf_output = _build_pdf_report(
            output_path=pdf_path,
            scored_universe=scored_universe,
            long_book=long_book,
            short_book=short_book,
            price_data=price_data,
            index_data=index_data,
            pairs=report_pairs,
        )
    else:
        logger.warning("Skipping PDF generation because reportlab is unavailable: %s", REPORTLAB_IMPORT_ERROR)

    return {
        "markdown_path": markdown_path,
        "pdf_path": pdf_output,
    }


if __name__ == "__main__":
    from data_ingestion import run_data_pipeline
    from factor_model import run_factor_model, suggest_relative_trades

    price_data, financial_data, metrics_df, index_data = run_data_pipeline()
    scored, longs, shorts = run_factor_model(metrics_df)
    del financial_data
    artifacts = generate_report(
        scored_universe=scored,
        long_book=longs,
        short_book=shorts,
        price_data=price_data,
        index_data=index_data,
        pairs=suggest_relative_trades(longs, shorts),
    )
    print(f"\nMarkdown report: {artifacts['markdown_path']}")
    print(f"PDF report: {artifacts['pdf_path']}")
