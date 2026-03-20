# Indian Long/Short Equity Research System

Systematic factor-based long/short research for Indian equities using the same value, quality, and momentum logic as the original project, now applied to:

- `NIFTY 50` constituents
- `F&O-eligible` Indian stocks
- an F&O-gated short book, so bearish ideas stay inside the tradable derivatives universe

## What Changed

- The working universe is now built dynamically as `NIFTY 50 ∪ F&O`.
- Live F&O discovery prefers `Zerodha` instruments when `ZERODHA_ACCESS_TOKEN` is available.
- If live Zerodha auth is unavailable, the app falls back to public NSE/Nifty files and then to a bundled NIFTY 50 fallback set.
- The dashboard and PDF report now speak in Indian-market terms instead of the old Europe/country setup.

## Architecture

- `market_universe.py`: Builds the India universe and contains Zerodha auth helpers.
- `data_ingestion.py`: Fetches price/fundamental data and merges universe metadata.
- `factor_model.py`: Scores the universe and restricts shorts to F&O-eligible names.
- `backtesting.py`: Applies the same F&O short-side restriction in historical simulation.
- `sentiment_analysis.py`: News and sentiment for the Indian equity universe.
- `app.py`: Streamlit dashboard.
- `report_generator.py`: PDF report generation.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 -c "import nltk; nltk.download('vader_lexicon')"
```

## Zerodha Auth

Zerodha live endpoints require an `access token`. `API key` + `API secret` alone are not enough.

1. Export your credentials:

```bash
export ZERODHA_API_KEY="your_api_key"
export ZERODHA_API_SECRET="your_api_secret"
```

2. Get the login URL:

```bash
python3 scripts/zerodha_auth.py
```

3. After Zerodha redirects you back, copy the `request_token` from the URL and exchange it:

```bash
python3 scripts/zerodha_auth.py --request-token "<request_token>"
```

4. Export the returned access token:

```bash
export ZERODHA_ACCESS_TOKEN="your_access_token"
```

If you skip this, the app still runs using public NSE/Nifty universe discovery.

## Run

```bash
streamlit run app.py
```

The dashboard includes:

- `Market Overview`: NIFTY / Bank Nifty / Sensex performance
- `Long/Short Ideas`: factor-ranked longs and F&O-eligible shorts
- `Stock Drill-Down`: chart, factor breakdown, key metrics, sentiment
- `Universe Table`: sortable view with universe bucket and F&O eligibility

## Reports

Generate from the sidebar or run:

```bash
python3 report_generator.py
```

Output is written to `reports/indian_ls_equity_report_<timestamp>.pdf`.

## Notes

- Cache keys are now universe-aware, so India runs do not collide with the old Europe caches.
- Public market files can fail or throttle; the pipeline degrades to fallback universes instead of stopping.
- Short ideas are selected only from F&O-eligible names when that metadata is available.
