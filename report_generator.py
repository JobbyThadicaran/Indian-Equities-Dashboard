"""
report_generator.py — Hedge Fund-Style PDF Report Generator
=============================================================
Generates professional PDF reports with market overview, long/short
ideas with thesis and financials, relative trades, charts, and tables.
Uses ReportLab for full layout control.
"""

import os
import io
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/headless environments
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Line

from config import (
    REPORT_TITLE, REPORT_SUBTITLE, REPORTS_DIR,
    TOP_N_LONG, TOP_N_SHORT
)
from utils import (
    setup_logger, ensure_directories, fmt_pct, fmt_number,
    fmt_large_number, fmt_ratio, ticker_to_name
)

logger = setup_logger("report_generator")

# ============================================================================
# COLOUR PALETTE
# ============================================================================
NAVY = colors.HexColor("#1a1a2e")
DARK_BLUE = colors.HexColor("#16213e")
ACCENT_BLUE = colors.HexColor("#0f3460")
ACCENT_TEAL = colors.HexColor("#53a8b6")
LIGHT_GREY = colors.HexColor("#f5f5f5")
MID_GREY = colors.HexColor("#cccccc")
GREEN = colors.HexColor("#27ae60")
RED = colors.HexColor("#e74c3c")
GOLD = colors.HexColor("#f39c12")

# ============================================================================
# CUSTOM STYLES
# ============================================================================

def get_report_styles() -> Dict[str, ParagraphStyle]:
    """Create custom paragraph styles for the report."""
    base = getSampleStyleSheet()
    
    styles = {
        "CoverTitle": ParagraphStyle(
            "CoverTitle", parent=base["Title"],
            fontSize=28, leading=34, textColor=NAVY,
            spaceAfter=6, fontName="Helvetica-Bold"
        ),
        "CoverSubtitle": ParagraphStyle(
            "CoverSubtitle", parent=base["Normal"],
            fontSize=14, leading=18, textColor=ACCENT_BLUE,
            spaceAfter=20, fontName="Helvetica"
        ),
        "CoverDate": ParagraphStyle(
            "CoverDate", parent=base["Normal"],
            fontSize=11, textColor=colors.grey,
            fontName="Helvetica-Oblique"
        ),
        "SectionTitle": ParagraphStyle(
            "SectionTitle", parent=base["Heading1"],
            fontSize=18, leading=22, textColor=NAVY,
            spaceBefore=16, spaceAfter=10,
            fontName="Helvetica-Bold",
            borderWidth=0, borderColor=ACCENT_TEAL,
            borderPadding=4
        ),
        "SubsectionTitle": ParagraphStyle(
            "SubsectionTitle", parent=base["Heading2"],
            fontSize=14, leading=17, textColor=ACCENT_BLUE,
            spaceBefore=12, spaceAfter=6,
            fontName="Helvetica-Bold"
        ),
        "StockTitle": ParagraphStyle(
            "StockTitle", parent=base["Heading3"],
            fontSize=13, leading=16, textColor=NAVY,
            spaceBefore=8, spaceAfter=4,
            fontName="Helvetica-Bold"
        ),
        "BodyText": ParagraphStyle(
            "BodyText", parent=base["Normal"],
            fontSize=10, leading=14, textColor=colors.black,
            alignment=TA_JUSTIFY, fontName="Helvetica",
            spaceAfter=6
        ),
        "BulletText": ParagraphStyle(
            "BulletText", parent=base["Normal"],
            fontSize=10, leading=13, textColor=colors.black,
            leftIndent=15, fontName="Helvetica",
            spaceAfter=3
        ),
        "MetricLabel": ParagraphStyle(
            "MetricLabel", parent=base["Normal"],
            fontSize=9, textColor=colors.grey,
            fontName="Helvetica"
        ),
        "MetricValue": ParagraphStyle(
            "MetricValue", parent=base["Normal"],
            fontSize=11, textColor=NAVY,
            fontName="Helvetica-Bold"
        ),
        "Disclaimer": ParagraphStyle(
            "Disclaimer", parent=base["Normal"],
            fontSize=7, leading=9, textColor=colors.grey,
            fontName="Helvetica-Oblique", alignment=TA_CENTER
        ),
        "Footer": ParagraphStyle(
            "Footer", parent=base["Normal"],
            fontSize=8, textColor=colors.grey,
            fontName="Helvetica", alignment=TA_CENTER
        ),
    }
    return styles


# ============================================================================
# CHART GENERATION (Matplotlib → ReportLab Image)
# ============================================================================

def create_price_chart(
    price_series: pd.Series,
    title: str = "",
    figsize: Tuple = (6, 2.5)
) -> Optional[str]:
    """
    Generate a price chart and save as a temporary PNG for PDF embedding.
    
    Args:
        price_series: Series with datetime index and price values
        title: Chart title
        figsize: Figure dimensions
    
    Returns:
        Path to the saved PNG file
    """
    try:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(price_series.index, price_series.values, color="#0f3460", linewidth=1.5)
        ax.fill_between(price_series.index, price_series.values,
                        alpha=0.1, color="#53a8b6")
        ax.set_title(title, fontsize=10, fontweight="bold", color="#1a1a2e")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(fontsize=8, rotation=45)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        
        safe_title = re.sub(r'[^a-zA-Z0-9_\-]', '_', title)[:30]
        path = os.path.join(REPORTS_DIR, f"chart_{safe_title}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception as e:
        logger.warning(f"Chart generation failed: {e}")
        return None


def create_factor_breakdown_chart(
    scores: Dict[str, float],
    title: str = "Factor Breakdown",
    figsize: Tuple = (4, 3)
) -> Optional[str]:
    """
    Create a horizontal bar chart showing factor score breakdown.
    
    Args:
        scores: Dict mapping factor name → score
        title: Chart title
        figsize: Figure dimensions
    
    Returns:
        Path to saved PNG
    """
    try:
        fig, ax = plt.subplots(figsize=figsize)
        factors = list(scores.keys())
        values = list(scores.values())
        bar_colors = ["#27ae60" if v >= 50 else "#e74c3c" for v in values]
        
        y_pos = range(len(factors))
        ax.barh(y_pos, values, color=bar_colors, height=0.6, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(factors, fontsize=8)
        ax.set_xlim(0, 100)
        ax.axvline(x=50, color="grey", linestyle="--", alpha=0.5)
        ax.set_title(title, fontsize=10, fontweight="bold", color="#1a1a2e")
        ax.set_xlabel("Score", fontsize=8)
        plt.tight_layout()
        
        safe_title = re.sub(r'[^a-zA-Z0-9_\-]', '_', title)[:30]
        path = os.path.join(REPORTS_DIR, f"factors_{safe_title}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception as e:
        logger.warning(f"Factor chart generation failed: {e}")
        return None


def create_cumulative_returns_chart(
    returns_dict: Dict[str, pd.Series],
    title: str = "Strategy Cumulative Returns",
    figsize: Tuple = (6, 3)
) -> Optional[str]:
    """
    Create a multi-line cumulative returns chart.
    
    Args:
        returns_dict: Dict mapping series name → cumulative returns Series
        title: Chart title
        figsize: Figure dimensions
    
    Returns:
        Path to saved PNG
    """
    try:
        fig, ax = plt.subplots(figsize=figsize)
        line_colors = ["#0f3460", "#27ae60", "#e74c3c", "#f39c12"]
        
        for i, (name, series) in enumerate(returns_dict.items()):
            if len(series) > 0:
                ax.plot(series.index, series.values * 100,
                        label=name, linewidth=1.5,
                        color=line_colors[i % len(line_colors)])
        
        ax.set_title(title, fontsize=10, fontweight="bold", color="#1a1a2e")
        ax.set_ylabel("Return (%)", fontsize=8)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        plt.xticks(fontsize=8, rotation=45)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        
        path = os.path.join(REPORTS_DIR, "cumulative_returns.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception as e:
        logger.warning(f"Returns chart generation failed: {e}")
        return None


# ============================================================================
# TABLE BUILDERS
# ============================================================================

def build_metrics_table(
    data: Dict[str, str],
    col_widths: List[float] = None
) -> Table:
    """
    Build a formatted two-column metrics table (label → value).
    
    Args:
        data: Dict mapping metric label → formatted value
        col_widths: Column widths in points
    
    Returns:
        ReportLab Table object
    """
    if col_widths is None:
        col_widths = [140, 100]
    
    table_data = [[k, v] for k, v in data.items()]
    t = Table(table_data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.grey),
        ("TEXTCOLOR", (1, 0), (1, -1), NAVY),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, MID_GREY),
    ]))
    return t


def build_stock_table(
    df: pd.DataFrame,
    columns: List[str],
    headers: List[str] = None,
    col_widths: List[float] = None
) -> Table:
    """
    Build a multi-column stock data table.
    
    Args:
        df: DataFrame with stock data
        columns: Column names to include
        headers: Display headers (defaults to column names)
        col_widths: Column widths
    
    Returns:
        ReportLab Table object
    """
    if headers is None:
        headers = columns
    
    available = [c for c in columns if c in df.columns]
    if not available:
        return Table([["No data available"]])
    
    table_data = [headers[:len(available)]]
    
    # Explicit mapping to prevent silent ratio misidentification
    ratio_cols = ["composite_score", "value_score", "quality_score", "momentum_score"]
    
    for idx, row in df.head(15).iterrows():
        row_data = []
        for col in available:
            val = row.get(col, "N/A")
            if pd.isna(val):
                row_data.append("N/A")
            elif isinstance(val, float):
                if col in ratio_cols:
                    row_data.append(f"{val:.1f}")
                elif abs(val) < 1:
                    row_data.append(f"{val:.2%}")
                elif abs(val) > 1e6:
                    row_data.append(fmt_large_number(val))
                else:
                    row_data.append(f"{val:.1f}")
            else:
                row_data.append(str(val)[:25])
        table_data.append(row_data)
    
    if col_widths is None:
        col_widths = [max(80, 480 / len(available))] * len(available)
    
    t = Table(table_data, colWidths=col_widths[:len(available)])
    t.setStyle(TableStyle([
        # Header row
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        # Data rows
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 1), (0, -1), "LEFT"),
        # Alternating row colors
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
        # Borders
        ("LINEBELOW", (0, 0), (-1, 0), 1, ACCENT_TEAL),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, MID_GREY),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


# ============================================================================
# REPORT SECTIONS
# ============================================================================

def build_cover_page(styles: Dict) -> List:
    """Build the cover page elements."""
    elements = []
    elements.append(Spacer(1, 80))
    elements.append(Paragraph(REPORT_TITLE, styles["CoverTitle"]))
    elements.append(Paragraph(REPORT_SUBTITLE, styles["CoverSubtitle"]))
    elements.append(Spacer(1, 10))
    
    # Horizontal rule
    elements.append(HRFlowable(
        width="60%", thickness=2, color=ACCENT_TEAL,
        spaceBefore=10, spaceAfter=20
    ))
    
    date_str = datetime.now().strftime("%B %d, %Y")
    elements.append(Paragraph(f"Generated: {date_str}", styles["CoverDate"]))
    elements.append(Paragraph("Systematic Factor-Based Analysis", styles["CoverDate"]))
    elements.append(Spacer(1, 40))
    elements.append(Paragraph(
        "This report presents a systematic long/short equity strategy "
        "applied to European markets. Stocks are scored using a multi-factor "
        "model combining Value, Quality, and Momentum signals.",
        styles["BodyText"]
    ))
    elements.append(Spacer(1, 200))
    elements.append(Paragraph(
        "CONFIDENTIAL — For Institutional Use Only",
        styles["Disclaimer"]
    ))
    elements.append(PageBreak())
    return elements


def build_market_overview(
    styles: Dict,
    index_data: Dict[str, pd.DataFrame],
    scored_universe: pd.DataFrame
) -> List:
    """Build the Market Overview section."""
    elements = []
    elements.append(Paragraph("1. Market Overview", styles["SectionTitle"]))
    elements.append(HRFlowable(width="100%", thickness=1, color=ACCENT_TEAL, spaceAfter=10))
    
    # Index performance table
    perf_data = {}
    for name, df in index_data.items():
        if len(df) > 0 and "Close" in df.columns:
            prices = df["Close"]
            ytd_ret = prices.iloc[-1] / prices.iloc[0] - 1
            perf_data[name] = fmt_pct(ytd_ret)
    
    if perf_data:
        elements.append(Paragraph("Index Performance (YTD)", styles["SubsectionTitle"]))
        elements.append(build_metrics_table(perf_data))
        elements.append(Spacer(1, 15))
    
    # Universe statistics
    if not scored_universe.empty:
        universe_stats = {
            "Total Stocks": str(len(scored_universe)),
            "Avg Composite Score": fmt_number(scored_universe["composite_score"].mean())
                if "composite_score" in scored_universe.columns else "N/A",
            "Median P/E": fmt_ratio(scored_universe["pe_ratio"].median())
                if "pe_ratio" in scored_universe.columns else "N/A",
            "Median EV/EBITDA": fmt_ratio(scored_universe["ev_ebitda"].median())
                if "ev_ebitda" in scored_universe.columns else "N/A",
            "Avg Volatility": fmt_pct(scored_universe["volatility"].mean())
                if "volatility" in scored_universe.columns else "N/A",
        }
        elements.append(Paragraph("Universe Statistics", styles["SubsectionTitle"]))
        elements.append(build_metrics_table(universe_stats))
    
    # Index price charts
    for name, df in list(index_data.items())[:3]:
        if "Close" in df.columns:
            chart_path = create_price_chart(df["Close"], title=name)
            if chart_path and os.path.exists(chart_path):
                elements.append(Spacer(1, 10))
                elements.append(Image(chart_path, width=420, height=175))
    
    elements.append(PageBreak())
    return elements


def build_stock_idea_section(
    styles: Dict,
    stock_row: pd.Series,
    ticker: str,
    price_data: Dict[str, pd.DataFrame],
    position_type: str = "LONG"
) -> List:
    """
    Build a single stock idea section with thesis, financials, and chart.
    
    Args:
        styles: Report paragraph styles
        stock_row: Series with stock metrics
        ticker: Ticker symbol
        price_data: Price data for chart generation
        position_type: "LONG" or "SHORT"
    """
    elements = []
    
    # Stock header
    name = stock_row.get("name", ticker_to_name(ticker))
    score = stock_row.get("composite_score", 0)
    color_hex = "#27ae60" if position_type == "LONG" else "#e74c3c"
    
    elements.append(Paragraph(
        f'<font color="{color_hex}">{position_type}</font> — '
        f'{name} ({ticker}) — Score: {score:.1f}',
        styles["StockTitle"]
    ))
    
    # Investment Thesis
    elements.append(Paragraph("Investment Thesis", styles["SubsectionTitle"]))
    
    # Auto-generate thesis from metrics
    thesis_points = []
    if position_type == "LONG":
        if stock_row.get("value_score", 50) > 60:
            thesis_points.append("• Attractively valued relative to sector peers")
        if stock_row.get("quality_score", 50) > 60:
            thesis_points.append("• Strong quality metrics with solid profitability")
        if stock_row.get("momentum_score", 50) > 60:
            thesis_points.append("• Positive price momentum supporting the setup")
        if stock_row.get("revenue_growth_yoy") and stock_row["revenue_growth_yoy"] > 0:
            thesis_points.append(f"• Revenue growing {fmt_pct(stock_row['revenue_growth_yoy'])} YoY")
    else:
        if stock_row.get("value_score", 50) < 40:
            thesis_points.append("• Expensive valuation relative to peers")
        if stock_row.get("quality_score", 50) < 40:
            thesis_points.append("• Weak quality metrics and declining profitability")
        if stock_row.get("momentum_score", 50) < 40:
            thesis_points.append("• Negative price momentum confirming weakness")
        if stock_row.get("leverage") and stock_row["leverage"] > 3:
            thesis_points.append(f"• High leverage at {fmt_ratio(stock_row['leverage'])}")
    
    if not thesis_points:
        thesis_points.append("• Systematic factor model identifies this stock based on composite scoring")
    
    for point in thesis_points:
        elements.append(Paragraph(point, styles["BulletText"]))
    
    elements.append(Spacer(1, 8))
    
    # Key Financials Table
    financials = {}
    metric_map = {
        "Market Cap": ("market_cap", fmt_large_number),
        "P/E Ratio": ("pe_ratio", lambda x: fmt_ratio(x)),
        "EV/EBITDA": ("ev_ebitda", lambda x: fmt_ratio(x)),
        "FCF Yield": ("fcf_yield", lambda x: fmt_pct(x)),
        "EBITDA Margin": ("ebitda_margin", lambda x: fmt_pct(x)),
        "Net Margin": ("net_margin", lambda x: fmt_pct(x)),
        "Revenue Growth": ("revenue_growth_yoy", lambda x: fmt_pct(x)),
        "ROIC": ("roic", lambda x: fmt_pct(x)),
        "Leverage (ND/EBITDA)": ("leverage", lambda x: fmt_ratio(x)),
        "Volatility": ("volatility", lambda x: fmt_pct(x)),
        "3M Return": ("mom_3m", lambda x: fmt_pct(x)),
        "6M Return": ("mom_6m", lambda x: fmt_pct(x)),
    }
    
    for label, (col, formatter) in metric_map.items():
        val = stock_row.get(col)
        if val is not None and not pd.isna(val):
            financials[label] = formatter(val)
    
    if financials:
        elements.append(Paragraph("Key Financials & Valuation", styles["SubsectionTitle"]))
        elements.append(build_metrics_table(financials))
    
    # Factor breakdown chart
    factor_scores = {}
    for label, col in [
        ("Value", "value_score"), ("Quality", "quality_score"),
        ("Momentum", "momentum_score")
    ]:
        val = stock_row.get(col)
        if val is not None and not pd.isna(val):
            factor_scores[label] = float(val)
    
    if factor_scores:
        chart_path = create_factor_breakdown_chart(factor_scores, title=f"{name} Factors")
        if chart_path and os.path.exists(chart_path):
            elements.append(Spacer(1, 5))
            elements.append(Image(chart_path, width=280, height=180))
    
    # Price chart
    if ticker in price_data:
        prices = price_data[ticker]["Close"]
        chart_path = create_price_chart(prices, title=f"{name} Price")
        if chart_path and os.path.exists(chart_path):
            elements.append(Spacer(1, 5))
            elements.append(Image(chart_path, width=420, height=175))
    
    # Catalyst & Risks
    elements.append(Paragraph("Catalyst", styles["SubsectionTitle"]))
    if position_type == "LONG":
        elements.append(Paragraph(
            "• Factor convergence: improving quality metrics and positive momentum "
            "suggest potential for mean-reversion in valuation multiple.",
            styles["BulletText"]
        ))
    else:
        elements.append(Paragraph(
            "• Deteriorating fundamentals: declining margins and negative momentum "
            "suggest further downside risk to current valuation.",
            styles["BulletText"]
        ))
    
    elements.append(Paragraph("Risks", styles["SubsectionTitle"]))
    vol = stock_row.get("volatility")
    vol_str = fmt_pct(vol) if vol and not pd.isna(vol) else "N/A"
    elements.append(Paragraph(
        f"• Annualised volatility: {vol_str}",
        styles["BulletText"]
    ))
    dd = stock_row.get("max_drawdown")
    dd_str = fmt_pct(dd) if dd and not pd.isna(dd) else "N/A"
    elements.append(Paragraph(
        f"• Maximum drawdown: {dd_str}",
        styles["BulletText"]
    ))
    
    elements.append(Spacer(1, 15))
    elements.append(HRFlowable(width="80%", thickness=0.5, color=MID_GREY, spaceAfter=10))
    
    return elements


def build_relative_trades_section(
    styles: Dict,
    pairs: List[Dict]
) -> List:
    """Build the Relative Trades section."""
    elements = []
    elements.append(Paragraph("4. Relative Trades", styles["SectionTitle"]))
    elements.append(HRFlowable(width="100%", thickness=1, color=ACCENT_TEAL, spaceAfter=10))
    
    elements.append(Paragraph(
        "The following pair trade ideas match stocks with the highest spread "
        "in composite factor scores, preferring sector-matched pairs for "
        "reduced systematic risk.",
        styles["BodyText"]
    ))
    elements.append(Spacer(1, 10))
    
    if pairs:
        # Build pairs table
        table_data = [["#", "Long", "Short", "Sector", "Score Spread"]]
        for i, pair in enumerate(pairs, 1):
            table_data.append([
                str(i),
                f"{pair['long_name']}",
                f"{pair['short_name']}",
                pair.get("sector", "Cross-Sector")[:20],
                f"{pair['score_spread']:.1f}",
            ])
        
        t = Table(table_data, colWidths=[30, 120, 120, 110, 80])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
            ("LINEBELOW", (0, 0), (-1, 0), 1, ACCENT_TEAL),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
        ]))
        elements.append(t)
    else:
        elements.append(Paragraph(
            "No relative trade pairs identified in this period.",
            styles["BodyText"]
        ))
    
    elements.append(PageBreak())
    return elements


def build_appendix(
    styles: Dict,
    scored_universe: pd.DataFrame,
    backtest_results: Dict = None
) -> List:
    """Build the Appendix section with data tables and backtest results."""
    elements = []
    elements.append(Paragraph("5. Appendix", styles["SectionTitle"]))
    elements.append(HRFlowable(width="100%", thickness=1, color=ACCENT_TEAL, spaceAfter=10))
    
    # Full universe scoring table
    elements.append(Paragraph("A. Full Universe Scores (Top 20)", styles["SubsectionTitle"]))
    
    if not scored_universe.empty:
        cols_to_show = ["name", "composite_score", "value_score", "quality_score", "momentum_score"]
        available = [c for c in cols_to_show if c in scored_universe.columns]
        if available:
            top20 = scored_universe.sort_values("composite_score", ascending=False).head(20)
            headers = ["Name", "Composite", "Value", "Quality", "Momentum"][:len(available)]
            elements.append(build_stock_table(top20, available, headers))
    
    elements.append(Spacer(1, 20))
    
    # Backtest results
    if backtest_results and "metrics" in backtest_results:
        elements.append(Paragraph("B. Backtest Performance", styles["SubsectionTitle"]))
        elements.append(build_metrics_table(backtest_results["metrics"]))
        
        # Cumulative returns chart
        returns_series = {}
        for key, label in [
            ("cumulative_returns", "L/S Portfolio"),
            ("long_cumulative", "Long Leg"),
            ("short_cumulative", "Short Leg"),
        ]:
            if key in backtest_results and len(backtest_results[key]) > 0:
                returns_series[label] = backtest_results[key]
        
        if returns_series:
            chart_path = create_cumulative_returns_chart(returns_series)
            if chart_path and os.path.exists(chart_path):
                elements.append(Spacer(1, 10))
                elements.append(Image(chart_path, width=420, height=210))
    
    # Disclaimer
    elements.append(Spacer(1, 40))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=MID_GREY, spaceAfter=5))
    elements.append(Paragraph(
        "DISCLAIMER: This report is generated algorithmically and is for informational "
        "purposes only. It does not constitute investment advice. Past performance is not "
        "indicative of future results. All investments carry risk.",
        styles["Disclaimer"]
    ))
    
    return elements


# ============================================================================
# MAIN REPORT GENERATOR
# ============================================================================

def generate_report(
    scored_universe: pd.DataFrame,
    long_book: pd.DataFrame,
    short_book: pd.DataFrame,
    price_data: Dict[str, pd.DataFrame],
    index_data: Dict[str, pd.DataFrame],
    pairs: List[Dict] = None,
    backtest_results: Dict = None,
    output_filename: str = None
) -> str:
    """
    Generate the complete hedge fund-style PDF report.
    
    Args:
        scored_universe: Full scored universe DataFrame
        long_book: Long positions DataFrame
        short_book: Short positions DataFrame
        price_data: Historical price data
        index_data: Index benchmark data
        pairs: Relative trade suggestions
        backtest_results: Backtest output (optional)
        output_filename: Custom output filename
    
    Returns:
        Path to the generated PDF file
    """
    ensure_directories()
    
    if output_filename is None:
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"european_ls_equity_report_{date_str}.pdf"
    
    output_path = os.path.join(REPORTS_DIR, output_filename)
    styles = get_report_styles()
    
    logger.info(f"Generating PDF report: {output_path}")
    
    # Create document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        topMargin=25 * mm,
        bottomMargin=20 * mm,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        title=REPORT_TITLE,
        author="Systematic Equity Research"
    )
    
    elements = []
    
    # 1. Cover Page
    elements.extend(build_cover_page(styles))
    
    # 2. Market Overview
    elements.extend(build_market_overview(styles, index_data, scored_universe))
    
    # 3. Long Ideas
    elements.append(Paragraph("2. Top Long Ideas", styles["SectionTitle"]))
    elements.append(HRFlowable(width="100%", thickness=1, color=GREEN, spaceAfter=10))
    
    for i, (ticker, row) in enumerate(long_book.head(TOP_N_LONG).iterrows()):
        elements.extend(
            build_stock_idea_section(styles, row, ticker, price_data, "LONG")
        )
    
    elements.append(PageBreak())
    
    # 4. Short Ideas
    elements.append(Paragraph("3. Top Short Ideas", styles["SectionTitle"]))
    elements.append(HRFlowable(width="100%", thickness=1, color=RED, spaceAfter=10))
    
    for i, (ticker, row) in enumerate(short_book.head(TOP_N_SHORT).iterrows()):
        elements.extend(
            build_stock_idea_section(styles, row, ticker, price_data, "SHORT")
        )
    
    elements.append(PageBreak())
    
    # 5. Relative Trades
    if pairs:
        elements.extend(build_relative_trades_section(styles, pairs))
    
    # 6. Appendix
    elements.extend(
        build_appendix(styles, scored_universe, backtest_results)
    )
    
    # Build the PDF
    try:
        doc.build(elements)
        logger.info(f"Report generated successfully: {output_path}")
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        raise
    finally:
        # Cleanup temporary PNG charts
        for file in os.listdir(REPORTS_DIR):
            if file.endswith(".png"):
                try:
                    os.remove(os.path.join(REPORTS_DIR, file))
                except Exception as e:
                    logger.debug(f"Failed to cleanup PNG {file}: {e}")
    
    return output_path


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    from data_ingestion import run_data_pipeline
    from factor_model import run_factor_model, suggest_relative_trades
    
    # Run data pipeline
    price_data, financial_data, metrics_df, index_data = run_data_pipeline()
    
    # Run factor model
    scored, longs, shorts = run_factor_model(metrics_df)
    
    # Relative trades
    pairs = suggest_relative_trades(longs, shorts)
    
    # Generate report
    report_path = generate_report(
        scored_universe=scored,
        long_book=longs,
        short_book=shorts,
        price_data=price_data,
        index_data=index_data,
        pairs=pairs
    )
    
    print(f"\nReport generated: {report_path}")
