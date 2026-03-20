# European Long/Short Equity Research System

A production-quality, modular Python system for systematic long/short equity research on European markets. Built as a hedge fund prototype.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard (app.py)                  │
├──────────┬──────────┬──────────────┬───────────────────────────-─┤
│ Market   │ Long/    │  Stock       │   Universe                  │
│ Overview │ Short    │  Drill-Down  │   Table                     │
├──────────┴──────────┴──────────────┴────────────────────────────-┤
│                                                                  │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────┐  │
│  │ Factor Model │  │  Financial    │  │ Sentiment Analysis   │  │
│  │ (Scoring)    │  │  Analysis     │  │ (News + NLP)         │  │
│  └──────┬───────┘  └───────┬───────┘  └──────────┬───────────┘  │
│         │                  │                      │              │
│  ┌──────┴──────────────────┴──────────────────────┴───────────┐  │
│  │              Data Ingestion (yfinance + cache)              │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────┐  ┌──────────────────────────────────────-─┐  │
│  │  Backtesting   │  │    PDF Report Generator (ReportLab)    │  │
│  └────────────────┘  └────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

## Modules

| Module | Description |
|---|---|
| `config.py` | Universe definitions (CAC 40, DAX 40, FTSE 100, STOXX subset), factor weights, sentiment keywords, constants |
| `data_ingestion.py` | Fetches prices + financials via yfinance (primary) and Akshare (fallback for indices) with parallel threading and disk caching |
| `factor_model.py` | Multi-factor scoring engine (Value, Quality, Momentum), percentile ranking, portfolio construction |
| `financial_analysis.py` | Growth, Profitability, Risk, Valuation analytics per stock |
| `sentiment_analysis.py` | RSS news fetching, VADER sentiment scoring, event keyword detection |
| `backtesting.py` | Monthly rebalance strategy simulation with performance metrics |
| `report_generator.py` | Professional PDF report generation via ReportLab |
| `app.py` | Interactive Streamlit dashboard with Plotly charts |
| `utils.py` | Shared helpers: caching, formatting, export, logging |

## Factor Model

### Scoring System

| Category | Factor | Weight | Direction |
|---|---|---|---|
| **Value** | P/E Ratio | 12% | Lower is better |
| **Value** | EV/EBITDA | 12% | Lower is better |
| **Value** | FCF Yield | 11% | Higher is better |
| **Quality** | ROIC | 12% | Higher is better |
| **Quality** | EBITDA Margin | 12% | Higher is better |
| **Quality** | Net Debt/EBITDA | 11% | Lower is better |
| **Momentum** | 3-Month Return | 15% | Higher is better |
| **Momentum** | 6-Month Return | 15% | Higher is better |

### Portfolio Construction
- **LONG**: Top 10% composite score → Stocks with strongest combined factor signal
- **SHORT**: Bottom 10% composite score → Stocks with weakest combined factor signal

## Setup

### Prerequisites
- Python 3.9+
- Internet connection (for data fetching)

### Installation

```bash
# Clone / navigate to the project
cd "hedge Fund Analysis"

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (for sentiment analysis)
python3 -c "import nltk; nltk.download('vader_lexicon')"
```

## Usage

### 1. Run the full Streamlit Dashboard

```bash
streamlit run app.py
```

This launches the interactive dashboard at `http://localhost:8501` with:
- **Market Overview** — European index performance with interactive charts
- **Long/Short Ideas** — Top-ranked stocks with factor breakdowns
- **Stock Drill-Down** — Candlestick chart, radar factor chart, sentiment, key metrics
- **Universe Table** — Sortable/filterable table of all scored stocks

### 2. Run individual modules

```bash
# Data ingestion only
python3 data_ingestion.py

# Factor model + portfolio construction
python3 factor_model.py

# Financial analysis
python3 financial_analysis.py

# Sentiment analysis
python3 sentiment_analysis.py

# Backtesting
python3 backtesting.py

# PDF report generation
python3 report_generator.py
```

### 3. Generate PDF Report

Via the dashboard sidebar button, or standalone:

```bash
python3 report_generator.py
```

Output is saved to `reports/european_ls_equity_report_<timestamp>.pdf`.

### 4. Backtesting

```bash
python3 backtesting.py
```

Runs a monthly rebalance simulation, outputs:
- Portfolio/long/short cumulative returns
- Sharpe, Sortino, Calmar ratios
- Max drawdown, hit rate
- CSV export to `exports/`

## Output Directories

| Directory | Contents |
|---|---|
| `data/` | Cached price and financial data (pickle) |
| `reports/` | Generated PDF reports and charts |
| `exports/` | CSV exports of metrics, scores, performance |
| `logs/` | Module-level log files |

## Data Sources

- **yfinance** — Historical prices, financial statements, valuation metrics
- **Akshare** — Fallback for European market indices (FTSE 100, DAX, CAC 40, Euro STOXX 50)
- **RSS Feeds** — Financial news from Yahoo Finance, Reuters, MarketWatch, CNBC
- **VADER** — Sentiment scoring via vaderSentiment library

## Notes

- First run takes 5-10 minutes to download data for the full universe (~180 stocks)
- Subsequent runs use cached data (12-hour expiry by default)
- Factor weights are configurable in `config.py`
- The system gracefully handles missing data by filling with neutral scores
