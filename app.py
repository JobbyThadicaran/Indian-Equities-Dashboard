"""
app.py — Streamlit Dashboard for Indian Long/Short Equity Research
==================================================================
Interactive dashboard with market overview, long/short ideas,
stock drill-down, sentiment analysis, and report generation.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import MARKET_NAME, SHORT_BOOK_DESCRIPTION, UNIVERSE_DESCRIPTION
from utils import (
    fmt_pct, fmt_number, fmt_large_number, fmt_ratio,
    ticker_to_name, load_from_cache, ensure_directories
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Indian L/S Equity Research",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card h3 {
        font-size: 14px;
        color: #53a8b6;
        margin-bottom: 5px;
        font-weight: 400;
    }
    .metric-card h2 {
        font-size: 28px;
        margin: 0;
        font-weight: 700;
    }
    
    /* Position badges */
    .long-badge {
        background-color: #27ae60;
        color: white;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
    }
    .short-badge {
        background-color: #e74c3c;
        color: white;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
    }
    
    /* Section headers */
    .section-header {
        font-size: 24px;
        font-weight: 700;
        color: #1a1a2e;
        border-bottom: 3px solid #53a8b6;
        padding-bottom: 8px;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    
    /* Score bar */
    .score-bar {
        height: 8px;
        border-radius: 4px;
        background: linear-gradient(90deg, #e74c3c 0%, #f39c12 50%, #27ae60 100%);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Table styling */
    .dataframe th {
        background-color: #1a1a2e !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA LOADING (cached)
# ============================================================================

@st.cache_data(ttl=3600, show_spinner="Loading market data...", hash_funcs={dict: lambda d: str(sorted(d.keys()))})
def load_all_data():
    """
    Load all data — either from cache or by running the full pipeline.
    Returns a dict with all required data objects.
    """
    from data_ingestion import run_data_pipeline
    from factor_model import run_factor_model, suggest_relative_trades
    from sentiment_analysis import run_sentiment_pipeline
    
    ensure_directories()
    
    # Data ingestion
    price_data, financial_data, metrics_df, index_data = run_data_pipeline()
    
    # Factor model
    scored_universe, long_book, short_book = run_factor_model(metrics_df)
    
    # Financial analysis execution removed as it is entirely unused by the UI
    # Sentiment analysis
    ticker_names = metrics_df["name"].to_dict() if "name" in metrics_df.columns else {}
    sentiment_df, sentiment_summary = run_sentiment_pipeline(
        tickers=list(price_data.keys()),
        fetch_individual=False,
        ticker_names=ticker_names
    )
    
    # Relative trades
    pairs = suggest_relative_trades(long_book, short_book)
    
    # Merge sentiment into scored universe
    if not sentiment_df.empty:
        # Ensure sentiment_df uses ticker as index
        if "ticker" in sentiment_df.columns:
            sentiment_df = sentiment_df.set_index("ticker")
        
        sentiment_cols = ["sentiment_score", "sentiment_count", "headlines"]
        available = [c for c in sentiment_cols if c in sentiment_df.columns]
        if available:
            scored_universe = scored_universe.join(
                sentiment_df[available], how="left"
            )
    
    if "country" not in scored_universe.columns:
        scored_universe["country"] = MARKET_NAME
    else:
        scored_universe["country"] = scored_universe["country"].fillna(MARKET_NAME)
    
    return {
        "price_data": price_data,
        "financial_data": financial_data,
        "metrics_df": metrics_df,
        "index_data": index_data,
        "scored_universe": scored_universe,
        "long_book": long_book,
        "short_book": short_book,
        "sentiment_df": sentiment_df,
        "sentiment_summary": sentiment_summary,
        "pairs": pairs,
    }


@st.cache_data(
    ttl=1800,
    show_spinner=False,
    hash_funcs={dict: lambda d: str(sorted(d.items()))}
)
def load_live_sentiment_for_tickers(tickers, ticker_names):
    """Fetch live news/sentiment for the small set of tickers shown in the UI."""
    from sentiment_analysis import get_ticker_sentiment_snapshots

    if not tickers:
        return pd.DataFrame()

    return get_ticker_sentiment_snapshots(
        list(tickers),
        ticker_names=ticker_names,
        max_articles=8,
    )


@st.cache_data(ttl=1800, show_spinner="Loading latest company news...")
def load_live_sentiment_for_ticker(ticker, ticker_name):
    """Fetch live news/sentiment for the selected drill-down stock."""
    from sentiment_analysis import get_ticker_sentiment_snapshot

    return get_ticker_sentiment_snapshot(
        ticker,
        ticker_name=ticker_name,
        max_articles=10,
    )


def apply_live_sentiment(base_df, live_df):
    """Override cached sentiment with fresher per-ticker snapshots."""
    if base_df.empty or live_df is None or live_df.empty:
        return base_df

    updated = base_df.copy()
    common = updated.index.intersection(live_df.index)
    if len(common) == 0:
        return updated

    for col in live_df.columns:
        updated.loc[common, col] = live_df.loc[common, col]

    return updated


def render_sentiment_label(row):
    """Render a human-friendly sentiment label."""
    sentiment = row.get("sentiment_score", 0)
    sentiment_count = row.get("sentiment_count", 0)

    if sentiment_count is None or pd.isna(sentiment_count) or float(sentiment_count) <= 0:
        st.markdown("Sent: No news")
        return

    sent_color = "🟢" if sentiment > 0.05 else "🔴" if sentiment < -0.05 else "⚪"
    st.markdown(f"Sent: {sent_color} {sentiment:.2f}")


# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar(data):
    """Render sidebar with filters and controls."""
    st.sidebar.markdown("## 🎛️ Filters")
    
    scored = data["scored_universe"]
    
    selected_buckets = None
    if "universe_bucket" in scored.columns:
        buckets = sorted(scored["universe_bucket"].dropna().unique())
        selected_buckets = st.sidebar.multiselect(
            "Universe Bucket", buckets, default=buckets,
            key="universe_bucket_filter"
        )
    
    # Sector filter
    if "sector" in scored.columns:
        sectors = sorted(scored["sector"].dropna().unique())
        selected_sectors = st.sidebar.multiselect(
            "Sector", sectors, default=sectors,
            key="sector_filter"
        )
    else:
        selected_sectors = None
    
    # Score range
    if "composite_score" in scored.columns:
        score_range = st.sidebar.slider(
            "Composite Score Range",
            min_value=0.0, max_value=100.0,
            value=(0.0, 100.0),
            key="score_range"
        )
    else:
        score_range = (0, 100)

    fo_only = False
    if "is_fo_eligible" in scored.columns:
        fo_only = st.sidebar.checkbox(
            "Show F&O-eligible only",
            value=False,
            help="Limit the table and idea views to names that are shortable / optionable.",
            key="fo_only_filter"
        )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## ⚙️ Actions")
    
    # Report generation button
    generate_report = st.sidebar.button("📄 Generate PDF Report", key="gen_report")
    
    # Data refresh
    refresh_data = st.sidebar.button("🔄 Refresh Data", key="refresh")
    if refresh_data:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Universe:** {len(scored)} stocks  \n"
        f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}"
    )
    
    return {
        "universe_buckets": selected_buckets,
        "sectors": selected_sectors,
        "score_range": score_range,
        "fo_only": fo_only,
        "generate_report": generate_report,
    }


def apply_filters(scored, filters):
    """Apply sidebar filters to the scored universe."""
    filtered = scored.copy()
    
    if filters["universe_buckets"] and "universe_bucket" in filtered.columns:
        filtered = filtered[filtered["universe_bucket"].isin(filters["universe_buckets"])]
    
    # Sector filter
    if filters["sectors"] and "sector" in filtered.columns:
        filtered = filtered[filtered["sector"].isin(filters["sectors"])]

    if filters["fo_only"] and "is_fo_eligible" in filtered.columns:
        filtered = filtered[filtered["is_fo_eligible"].fillna(False)]
    
    # Score range
    if "composite_score" in filtered.columns:
        filtered = filtered[
            (filtered["composite_score"] >= filters["score_range"][0]) &
            (filtered["composite_score"] <= filters["score_range"][1])
        ]
    
    return filtered


# ============================================================================
# SECTION 1: MARKET OVERVIEW
# ============================================================================

def render_market_overview(data):
    """Render the Market Overview section."""
    st.markdown('<div class="section-header">📈 Market Overview</div>', unsafe_allow_html=True)
    
    index_data = data["index_data"]
    
    if not index_data:
        st.info("No index data available. Running data pipeline to fetch market data...")
        return
    
    # Index performance metrics
    cols = st.columns(len(index_data))
    for i, (name, df) in enumerate(index_data.items()):
        with cols[i]:
            if "Close" in df.columns and len(df) > 1:
                current = df["Close"].iloc[-1]
                prev = df["Close"].iloc[0]
                change = (current / prev - 1)
                delta_color = "normal" if change >= 0 else "inverse"
                st.metric(
                    label=name,
                    value=f"{current:,.0f}",
                    delta=f"{change:.2%}",
                    delta_color=delta_color
                )
    
    # Index performance chart
    st.markdown("#### Index Performance (Normalized)")
    fig = go.Figure()
    colors_list = ["#0f3460", "#27ae60", "#e74c3c", "#f39c12", "#9b59b6"]
    
    for i, (name, df) in enumerate(index_data.items()):
        if "Close" in df.columns and len(df) > 0:
            normalized = (df["Close"] / df["Close"].iloc[0] - 1) * 100
            fig.add_trace(go.Scatter(
                x=normalized.index,
                y=normalized.values,
                mode="lines",
                name=name,
                line=dict(width=2, color=colors_list[i % len(colors_list)])
            ))
    
    fig.update_layout(
        yaxis_title="Return (%)",
        template="plotly_white",
        height=400,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=40, b=50)
    )
    fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# SECTION 2 & 3: LONG / SHORT IDEAS
# ============================================================================

def render_portfolio_ideas(data, filtered):
    """Render Top Long and Short Ideas sections."""
    col_long, col_short = st.columns(2)
    
    # Intersection logic to handle sidebar filters while preserving factor-model rankings
    long_tickers = filtered.index.intersection(data["long_book"].index)
    long_book = data["long_book"].loc[long_tickers].copy()
    
    short_tickers = filtered.index.intersection(data["short_book"].index)
    short_book = data["short_book"].loc[short_tickers].copy()

    priority_tickers = list(long_book.head(10).index.union(short_book.head(10).index))
    ticker_names = filtered["name"].to_dict() if "name" in filtered.columns else {}
    live_sentiment = load_live_sentiment_for_tickers(priority_tickers, ticker_names)
    long_book = apply_live_sentiment(long_book, live_sentiment)
    short_book = apply_live_sentiment(short_book, live_sentiment)
    
    with col_long:
        st.markdown(
            '<div class="section-header" style="border-color: #27ae60;">'
            '🟢 Top Long Ideas</div>',
            unsafe_allow_html=True
        )
        
        if len(long_book) > 0:
            for ticker, row in long_book.head(10).iterrows():
                name = row.get("name", ticker_to_name(ticker))
                score = row.get("composite_score", 0)
                sentiment = row.get("sentiment_score", 0)
                
                with st.container():
                    c1, c2, c3 = st.columns([3, 1, 1])
                    with c1:
                        st.markdown(f"**{name}** `{ticker}`")
                    with c2:
                        st.markdown(f"Score: **{score:.0f}**")
                    with c3:
                        render_sentiment_label(row)
                    
                    # Mini factor bar
                    val = row.get("value_score", 50)
                    qual = row.get("quality_score", 50)
                    mom = row.get("momentum_score", 50)
                    
                    factor_data = pd.DataFrame({
                        "Factor": ["Value", "Quality", "Momentum"],
                        "Score": [val, qual, mom]
                    })
                    
                    fig_bar = go.Figure(go.Bar(
                        x=factor_data["Score"],
                        y=factor_data["Factor"],
                        orientation="h",
                        marker_color=["#0f3460", "#53a8b6", "#f39c12"],
                        text=[f"{v:.0f}" for v in factor_data["Score"]],
                        textposition="auto"
                    ))
                    fig_bar.update_layout(
                        height=100, margin=dict(l=0, r=0, t=0, b=0),
                        xaxis=dict(range=[0, 100], showgrid=False, showticklabels=False),
                        yaxis=dict(showgrid=False),
                        template="plotly_white",
                        showlegend=False
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                    st.markdown("---")
        else:
            st.info("No long positions in the filtered universe.")
    
    with col_short:
        st.markdown(
            '<div class="section-header" style="border-color: #e74c3c;">'
            '🔴 Top Short Ideas</div>',
            unsafe_allow_html=True
        )
        st.caption(SHORT_BOOK_DESCRIPTION)
        
        if len(short_book) > 0:
            for ticker, row in short_book.head(10).iterrows():
                name = row.get("name", ticker_to_name(ticker))
                score = row.get("composite_score", 0)
                sentiment = row.get("sentiment_score", 0)
                
                with st.container():
                    c1, c2, c3 = st.columns([3, 1, 1])
                    with c1:
                        st.markdown(f"**{name}** `{ticker}`")
                    with c2:
                        st.markdown(f"Score: **{score:.0f}**")
                    with c3:
                        render_sentiment_label(row)
                    
                    val = row.get("value_score", 50)
                    qual = row.get("quality_score", 50)
                    mom = row.get("momentum_score", 50)
                    
                    factor_data = pd.DataFrame({
                        "Factor": ["Value", "Quality", "Momentum"],
                        "Score": [val, qual, mom]
                    })
                    
                    fig_bar = go.Figure(go.Bar(
                        x=factor_data["Score"],
                        y=factor_data["Factor"],
                        orientation="h",
                        marker_color=["#e74c3c", "#c0392b", "#e67e22"],
                        text=[f"{v:.0f}" for v in factor_data["Score"]],
                        textposition="auto"
                    ))
                    fig_bar.update_layout(
                        height=100, margin=dict(l=0, r=0, t=0, b=0),
                        xaxis=dict(range=[0, 100], showgrid=False, showticklabels=False),
                        yaxis=dict(showgrid=False),
                        template="plotly_white",
                        showlegend=False
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                    st.markdown("---")
        else:
            st.info("No short positions in the filtered universe.")


# ============================================================================
# SECTION 4: STOCK DRILL-DOWN
# ============================================================================

def render_stock_drilldown(data, filtered):
    """Render the Stock Drill-down section."""
    st.markdown(
        '<div class="section-header">🔍 Stock Drill-Down</div>',
        unsafe_allow_html=True
    )
    
    # Stock selector
    ticker_names = {t: f"{ticker_to_name(t)} ({t})" for t in filtered.index}
    sorted_tickers = sorted(ticker_names.items(), key=lambda x: x[1])
    
    if not sorted_tickers:
        st.warning("No stocks available after filtering.")
        return
    
    selected_display = st.selectbox(
        "Select Stock",
        [v for _, v in sorted_tickers],
        key="stock_selector"
    )
    
    # Find the ticker from display name
    selected_ticker = None
    for t, display in sorted_tickers:
        if display == selected_display:
            selected_ticker = t
            break
    
    if selected_ticker is None:
        return
    
    row = filtered.loc[selected_ticker].copy()
    name = row.get("name", ticker_to_name(selected_ticker))
    live_snapshot = load_live_sentiment_for_ticker(selected_ticker, name)
    if isinstance(live_snapshot, dict) and live_snapshot:
        for key, value in live_snapshot.items():
            if key != "ticker":
                row[key] = value
    
    st.markdown(f"### {name} ({selected_ticker})")
    
    # Position badge
    position = row.get("position", "NEUTRAL")
    if position == "LONG":
        st.markdown('<span class="long-badge">LONG</span>', unsafe_allow_html=True)
    elif position == "SHORT":
        st.markdown('<span class="short-badge">SHORT</span>', unsafe_allow_html=True)
    
    # Key metrics row
    metric_cols = st.columns(6)
    metrics_display = [
        ("Composite Score", row.get("composite_score"), "{:.1f}"),
        ("P/E", row.get("pe_ratio"), "{:.1f}x"),
        ("EV/EBITDA", row.get("ev_ebitda"), "{:.1f}x"),
        ("FCF Yield", row.get("fcf_yield"), "{:.1%}"),
        ("Volatility", row.get("volatility"), "{:.1%}"),
        ("Sentiment", row.get("sentiment_score", 0), "{:+.2f}"),
    ]
    
    for i, (label, val, fmt) in enumerate(metrics_display):
        with metric_cols[i]:
            if label == "Sentiment" and (
                row.get("sentiment_count") is None or
                pd.isna(row.get("sentiment_count")) or
                float(row.get("sentiment_count", 0)) <= 0
            ):
                st.metric(label, "No news")
            elif val is not None and not pd.isna(val):
                st.metric(label, fmt.format(val))
            else:
                st.metric(label, "N/A")
    
    # Two-column layout: chart + details
    chart_col, detail_col = st.columns([3, 2])
    
    with chart_col:
        # Price chart
        if selected_ticker in data["price_data"]:
            price_df = data["price_data"][selected_ticker]
            
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                row_heights=[0.7, 0.3],
                vertical_spacing=0.05
            )
            
            # Candlestick
            fig.add_trace(
                go.Candlestick(
                    x=price_df.index,
                    open=price_df['Open'],
                    high=price_df['High'],
                    low=price_df['Low'],
                    close=price_df['Close'],
                    name="Price"
                ),
                row=1, col=1
            )
            
            # Vectorized volume color calculation for better performance
            if "Volume" in price_df.columns:
                vol_colors = np.where(price_df["Close"] >= price_df["Open"], "#27ae60", "#e74c3c").tolist()
                
                fig.add_trace(
                    go.Bar(
                        x=price_df.index,
                        y=price_df['Volume'],
                        marker_color=vol_colors,
                        name="Volume",
                        opacity=0.5
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(
                height=500,
                template="plotly_white",
                xaxis_rangeslider_visible=False,
                showlegend=False,
                margin=dict(l=50, r=20, t=20, b=30),
            )
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No price data available for this stock.")
    
    with detail_col:
        # Factor Score Breakdown (Radar Chart)
        st.markdown("#### Factor Breakdown")
        
        factor_values = {
            "Value": row.get("value_score", 50),
            "Quality": row.get("quality_score", 50),
            "Momentum": row.get("momentum_score", 50),
        }
        
        # Expand to include sub-factors
        sub_factors = {
            "P/E": row.get("pe_rank", 50),
            "EV/EBITDA": row.get("ev_ebitda_rank", 50),
            "FCF Yield": row.get("fcf_yield_rank", 50),
            "ROIC": row.get("roic_rank", 50),
            "Margins": row.get("ebitda_margin_rank", 50),
            "Leverage": row.get("leverage_rank", 50),
            "Mom 3M": row.get("mom_3m_rank", 50),
            "Mom 6M": row.get("mom_6m_rank", 50),
        }
        
        # Radar chart
        categories = list(sub_factors.keys())
        values = [v if not pd.isna(v) else 50 for v in sub_factors.values()]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the polygon
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(83, 168, 182, 0.3)",
            line=dict(color="#0f3460", width=2),
            name="Score"
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], showticklabels=True),
            ),
            height=350,
            margin=dict(l=50, r=50, t=30, b=30),
            showlegend=False
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Detailed metrics table
        st.markdown("#### Key Metrics")
        metrics_data = {
            "Market Cap": fmt_large_number(row.get("market_cap", 0)),
            "Revenue": fmt_large_number(row.get("revenue", 0)),
            "EBITDA": fmt_large_number(row.get("ebitda", 0)),
            "Net Income": fmt_large_number(row.get("net_income", 0)),
            "P/E Ratio": fmt_ratio(row.get("pe_ratio")),
            "EV/EBITDA": fmt_ratio(row.get("ev_ebitda")),
            "FCF Yield": fmt_pct(row.get("fcf_yield")),
            "EBITDA Margin": fmt_pct(row.get("ebitda_margin")),
            "Net Margin": fmt_pct(row.get("net_margin")),
            "ROIC": fmt_pct(row.get("roic")),
            "Leverage": fmt_ratio(row.get("leverage")),
            "Rev Growth YoY": fmt_pct(row.get("revenue_growth_yoy")),
            "Volatility": fmt_pct(row.get("volatility")),
            "Max Drawdown": fmt_pct(row.get("max_drawdown")),
            "3M Return": fmt_pct(row.get("mom_3m")),
            "6M Return": fmt_pct(row.get("mom_6m")),
        }
        
        metrics_df_display = pd.DataFrame(
            list(metrics_data.items()),
            columns=["Metric", "Value"]
        )
        st.dataframe(
            metrics_df_display,
            hide_index=True,
            use_container_width=True,
            height=400
        )
    
    # News & Sentiment Section
    st.markdown("#### 📰 News & Sentiment")
    
    headlines = row.get("headlines", [])
    if isinstance(headlines, list) and headlines:
        for h in headlines[:8]:
            if isinstance(h, dict):
                sent = h.get("sentiment", 0)
                emoji = "🟢" if sent > 0.05 else "🔴" if sent < -0.05 else "⚪"
                title = h.get("title", "")
                source = h.get("source", "")
                link = h.get("link", "")
                published = h.get("published", "")
                meta_parts = [f"_{source}_"] if source else []
                if published:
                    meta_parts.append(f"`{published}`")
                meta = " · ".join(meta_parts)
                if link:
                    if meta:
                        st.markdown(f"{emoji} [{title}]({link}) — {meta}")
                    else:
                        st.markdown(f"{emoji} [{title}]({link})")
                else:
                    if meta:
                        st.markdown(f"{emoji} **{title}** — {meta}")
                    else:
                        st.markdown(f"{emoji} **{title}**")
    else:
        st.info("No recent news headlines available for this stock.")


# ============================================================================
# SECTION 5: UNIVERSE TABLE
# ============================================================================

def render_universe_table(filtered):
    """Render the full scored universe as a sortable table."""
    st.markdown(
        '<div class="section-header">📋 Full Universe Scores</div>',
        unsafe_allow_html=True
    )
    
    display_cols = [
        "symbol", "name", "position", "composite_score", "value_score",
        "quality_score", "momentum_score", "pe_ratio", "ev_ebitda",
        "ebitda_margin", "volatility", "mom_3m", "sentiment_score",
        "universe_bucket", "is_fo_eligible"
    ]
    available = [c for c in display_cols if c in filtered.columns]
    
    if available and not filtered.empty:
        display_df = filtered[available].copy()
        display_df = display_df.sort_values("composite_score", ascending=False)
        
        # Format numeric columns
        for col in ["ebitda_margin", "volatility", "mom_3m"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
                )
        
        for col in ["composite_score", "value_score", "quality_score", "momentum_score"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
                )
        
        for col in ["pe_ratio", "ev_ebitda"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.1f}x" if pd.notna(x) else "N/A"
                )
        
        if "sentiment_score" in display_df.columns:
            display_df["sentiment_score"] = display_df["sentiment_score"].apply(
                lambda x: f"{x:+.2f}" if pd.notna(x) else "N/A"
            )
        
        # Rename columns for display
        rename_map = {
            "symbol": "Symbol", "name": "Name", "position": "Position", "composite_score": "Score",
            "value_score": "Value", "quality_score": "Quality",
            "momentum_score": "Momentum", "pe_ratio": "P/E",
            "ev_ebitda": "EV/EBITDA", "ebitda_margin": "EBITDA Margin",
            "volatility": "Volatility", "mom_3m": "3M Return",
            "sentiment_score": "Sentiment", "universe_bucket": "Universe",
            "is_fo_eligible": "F&O Eligible"
        }
        display_df = display_df.rename(columns=rename_map)
        if "F&O Eligible" in display_df.columns:
            display_df["F&O Eligible"] = display_df["F&O Eligible"].map(
                lambda value: "Yes" if pd.notna(value) and bool(value) else "No"
            )
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=500
        )
        
        # Download button
        csv = filtered[available].to_csv()
        st.download_button(
            "📥 Download Universe CSV",
            csv,
            "scored_universe.csv",
            "text/csv"
        )
    else:
        st.info("No data available for the selected filters.")


# ============================================================================
# REPORT GENERATION HANDLER
# ============================================================================

def handle_report_generation(data):
    """Handle PDF report generation via the sidebar button."""
    with st.spinner("Generating India market PDF report..."):
        try:
            from report_generator import generate_report
            from factor_model import suggest_relative_trades
            
            pairs = suggest_relative_trades(
                data["long_book"], data["short_book"]
            )
            
            # Try to load backtest results from cache
            backtest_results = load_from_cache("backtest_results")
            
            report_path = generate_report(
                scored_universe=data["scored_universe"],
                long_book=data["long_book"],
                short_book=data["short_book"],
                price_data=data["price_data"],
                index_data=data["index_data"],
                pairs=pairs,
                backtest_results=backtest_results
            )
            
            st.success(f"✅ Report generated: `{report_path}`")
            
            # Offer download
            if os.path.exists(report_path):
                with open(report_path, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_bytes,
                    file_name=os.path.basename(report_path),
                    mime="application/pdf"
                )
                st.success("Report generated successfully!")
        except Exception as e:
            st.error(f"Failed to generate report: {e}")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point."""
    # Title
    st.markdown(
        "# 📊 Indian Long/Short Equity Research"
    )
    st.markdown(
        f"_{UNIVERSE_DESCRIPTION} with Value · Quality · Momentum scoring_"
    )
    st.markdown("---")
    
    # Load data
    try:
        data = load_all_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    # Sidebar filters
    if data["scored_universe"].empty:
        st.warning("⚠️ No stock data available in cache. Fetching market data for the first time...")
        st.info("The initial NIFTY 50 + F&O universe fetch can take a few minutes. Please wait and click 'Refresh' below once complete.")
        if st.button("Refresh Now", key="refresh_empty"):
            st.cache_data.clear()
            st.rerun()
        
        # Still show market overview if available
        if data["index_data"]:
            render_market_overview(data)
        return
    
    filters = render_sidebar(data)
    
    # Apply filters
    filtered = apply_filters(data["scored_universe"], filters)
    
    # Handle report generation
    if filters["generate_report"]:
        handle_report_generation(data)
    
    # Create tabs for navigation
    tab_overview, tab_ideas, tab_drilldown, tab_universe = st.tabs([
        "📈 Market Overview",
        "💡 Long/Short Ideas",
        "🔍 Stock Drill-Down",
        "📋 Universe Table"
    ])
    
    with tab_overview:
        render_market_overview(data)
    
    with tab_ideas:
        render_portfolio_ideas(data, filtered)
    
    with tab_drilldown:
        render_stock_drilldown(data, filtered)
    
    with tab_universe:
        render_universe_table(filtered)


if __name__ == "__main__":
    main()
