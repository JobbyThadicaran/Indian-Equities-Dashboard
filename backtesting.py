"""
backtesting.py — Strategy Backtesting & Performance Tracking Module
====================================================================
Simulates the long/short factor strategy over historical data with
monthly rebalancing. Computes portfolio performance metrics and exports
results for analysis. In the Indian setup, the short leg is restricted
to F&O-eligible names when that metadata is available.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

from config import (
    FULL_UNIVERSE, FACTOR_WEIGHTS,
    LONG_THRESHOLD_PERCENTILE, SHORT_THRESHOLD_PERCENTILE
)
from utils import (
    setup_logger, save_to_cache, load_from_cache,
    export_to_csv, fmt_pct, fmt_number
)

logger = setup_logger("backtesting")

# ============================================================================
# BACKTEST ENGINE
# ============================================================================

class BacktestEngine:
    """
    Monthly rebalance backtesting engine for long/short factor strategies.
    
    Attributes:
        price_data: Dict mapping ticker → price DataFrame
        metrics_df: DataFrame with factor metrics per stock
        rebalance_freq: Rebalancing frequency ('M' for monthly)
        long_pct: Percentile threshold for LONG positions
        short_pct: Percentile threshold for SHORT positions
    """
    
    def __init__(
        self,
        price_data: Dict[str, pd.DataFrame],
        metrics_df: pd.DataFrame,
        rebalance_freq: str = "M",
        long_pct: float = LONG_THRESHOLD_PERCENTILE,
        short_pct: float = SHORT_THRESHOLD_PERCENTILE
    ):
        self.price_data = price_data
        self.metrics_df = metrics_df
        self.rebalance_freq = rebalance_freq
        self.long_pct = long_pct
        self.short_pct = short_pct
        
        # Build aligned price panel
        self.price_panel = self._build_price_panel()
        self.returns_panel = self.price_panel.pct_change()
    
    def _build_price_panel(self) -> pd.DataFrame:
        """
        Create a unified DataFrame of closing prices (columns = tickers, rows = dates).
        """
        close_prices = {}
        for ticker, df in self.price_data.items():
            if "Close" in df.columns and len(df) > 0:
                close_prices[ticker] = df["Close"]
        
        panel = pd.DataFrame(close_prices)
        panel.index = pd.to_datetime(panel.index)
        panel = panel.sort_index()
        
        # Forward fill missing prices (for non-trading days)
        panel = panel.ffill()
        
        logger.info(f"Price panel: {panel.shape[0]} days × {panel.shape[1]} tickers")
        return panel
    
    def _compute_rolling_factors(
        self,
        date: pd.Timestamp,
        lookback_days: int = 252
    ) -> pd.DataFrame:
        """
        Compute factor scores at a given point in time using only past data.
        
        This avoids look-ahead bias by using trailing momentum calculations
        and snapshot metrics.
        
        Args:
            date: Point-in-time date
            lookback_days: Number of trading days to look back
        
        Returns:
            DataFrame of factor scores at this date
        """
        # Get price history up to this date
        prices_up_to = self.price_panel.loc[:date]
        
        if len(prices_up_to) < 63:  # Need at least 3 months
            return pd.DataFrame()
        
        scores = pd.DataFrame(index=prices_up_to.columns)
        
        # Momentum factors (computed from trailing prices — no look-ahead bias)
        if len(prices_up_to) >= 63:
            scores["mom_3m"] = prices_up_to.iloc[-1] / prices_up_to.iloc[-63] - 1
        if len(prices_up_to) >= 126:
            scores["mom_6m"] = prices_up_to.iloc[-1] / prices_up_to.iloc[-126] - 1
        
        # Volatility (trailing 60-day)
        returns_trailing = prices_up_to.pct_change().tail(60)
        scores["volatility"] = returns_trailing.std() * np.sqrt(252)
        
        # Use static metrics (from most recent fundamental data)
        # In a production system, these would also be point-in-time
        for col in ["pe_ratio", "ev_ebitda", "fcf_yield", "roic", "ebitda_margin", "leverage"]:
            if col in self.metrics_df.columns:
                scores[col] = self.metrics_df[col]
        if "is_fo_eligible" in self.metrics_df.columns:
            scores["is_fo_eligible"] = self.metrics_df["is_fo_eligible"]
        
        # Rank factors
        if "pe_ratio" in scores:
            pe = scores["pe_ratio"].where((scores["pe_ratio"] > 0) & (scores["pe_ratio"] < 200))
            scores["pe_rank"] = 100 - pe.rank(pct=True) * 100
        if "ev_ebitda" in scores:
            ev = scores["ev_ebitda"].where((scores["ev_ebitda"] > 0) & (scores["ev_ebitda"] < 100))
            scores["ev_ebitda_rank"] = 100 - ev.rank(pct=True) * 100
        if "fcf_yield" in scores:
            scores["fcf_yield_rank"] = scores["fcf_yield"].rank(pct=True) * 100
        if "roic" in scores:
            scores["roic_rank"] = scores["roic"].rank(pct=True) * 100
        if "ebitda_margin" in scores:
            scores["ebitda_margin_rank"] = scores["ebitda_margin"].rank(pct=True) * 100
        if "leverage" in scores:
            scores["leverage_rank"] = 100 - scores["leverage"].rank(pct=True) * 100
        if "mom_3m" in scores:
            scores["mom_3m_rank"] = scores["mom_3m"].rank(pct=True) * 100
        if "mom_6m" in scores:
            scores["mom_6m_rank"] = scores["mom_6m"].rank(pct=True) * 100
        
        # Composite score
        composite = pd.Series(0.0, index=scores.index)
        weight_sum = 0.0
        for factor, weight in FACTOR_WEIGHTS.items():
            if factor in scores.columns:
                composite += scores[factor].fillna(50) * weight
                weight_sum += weight
        
        if weight_sum > 0:
            composite = composite / weight_sum
        
        scores["composite_score"] = composite
        
        return scores
    
    def run_backtest(self) -> Dict:
        """
        Execute the monthly rebalance backtest.
        
        Process:
        1. At each rebalance date, rank stocks by composite factor score
        2. Go LONG top decile, SHORT bottom decile
        3. Equal-weight within each leg
        4. Hold until next rebalance
        
        Returns:
            Dict with keys:
                'portfolio_returns': Series of daily portfolio returns
                'cumulative_returns': Series of cumulative returns
                'long_returns': Series for long-only leg
                'short_returns': Series for short-only leg
                'benchmark_returns': Series for equal-weight benchmark
                'rebalance_log': List of rebalance events
                'metrics': Dict of performance statistics
        """
        logger.info("=" * 60)
        logger.info("RUNNING BACKTEST")
        logger.info("=" * 60)
        
        # Generate monthly rebalance dates
        all_dates = self.price_panel.index
        # Version-safe frequency: 'ME' for pandas >= 2.2, 'M' otherwise
        freq = "ME" if pd.__version__ >= "2.2" else "M"
        
        rebalance_dates = pd.date_range(
            start=all_dates[63],  # Start after 3 months of data
            end=all_dates[-1],
            freq=freq
        )
        # Filter to dates that exist in our data
        rebalance_dates = [d for d in rebalance_dates if d <= all_dates[-1]]
        
        logger.warning("LOOK-AHEAD BIAS: fundamental factors are point-in-time as of today.")
        logger.info(f"Rebalance dates: {len(rebalance_dates)} months")
        
        # Track portfolio state
        portfolio_returns_list = []
        long_returns_list = []
        short_returns_list = []
        rebalance_log = []
        current_longs = []
        current_shorts = []
        
        for i, reb_date in enumerate(rebalance_dates):
            # Find the nearest valid trading date
            valid_dates = all_dates[all_dates <= reb_date]
            if len(valid_dates) == 0:
                continue
            reb_date_actual = valid_dates[-1]
            
            # Compute factor scores at this date
            scores = self._compute_rolling_factors(reb_date_actual)
            
            if scores.empty or "composite_score" not in scores.columns:
                continue
            
            valid_scores = scores.dropna(subset=["composite_score"])
            
            if len(valid_scores) < 10:
                continue
            
            # Select LONG and SHORT positions
            long_thresh = np.percentile(valid_scores["composite_score"], self.long_pct)
            short_scores = valid_scores
            if "is_fo_eligible" in valid_scores.columns:
                eligible = valid_scores[valid_scores["is_fo_eligible"].fillna(False)]
                if not eligible.empty:
                    short_scores = eligible
                else:
                    short_scores = pd.DataFrame(columns=valid_scores.columns)
            
            new_longs = valid_scores[
                valid_scores["composite_score"] >= long_thresh
            ].index.tolist()
            if short_scores.empty:
                new_shorts = []
            else:
                short_thresh = np.percentile(short_scores["composite_score"], self.short_pct)
                new_shorts = short_scores[
                    short_scores["composite_score"] <= short_thresh
                ].index.tolist()
            
            # Log rebalance
            rebalance_log.append({
                "date": reb_date_actual,
                "n_longs": len(new_longs),
                "n_shorts": len(new_shorts),
                "longs": new_longs[:5],
                "shorts": new_shorts[:5],
            })
            
            current_longs = new_longs
            current_shorts = new_shorts
            
            # Compute returns until next rebalance (or end of data)
            if i + 1 < len(rebalance_dates):
                next_reb = rebalance_dates[i + 1]
                next_valid = all_dates[all_dates <= next_reb]
                if len(next_valid) > 0:
                    next_date = next_valid[-1]
                else:
                    next_date = all_dates[-1]
            else:
                next_date = all_dates[-1]
            
            # Get daily returns for the holding period
            period_returns = self.returns_panel.loc[
                reb_date_actual:next_date
            ].iloc[1:]  # Skip the rebalance day itself
            
            if period_returns.empty:
                continue
            
            # Equal-weight portfolio returns
            valid_l = []
            if current_longs:
                valid_l = [l for l in current_longs if l in period_returns.columns]
                if valid_l:
                    long_returns_list.append(period_returns[valid_l].mean(axis=1))
            
            valid_s = []
            if current_shorts:
                valid_s = [s for s in current_shorts if s in period_returns.columns]
                if valid_s:
                    short_returns_list.append(-period_returns[valid_s].mean(axis=1)) # Short = negative return
            
            # Combined L/S portfolio (50% long, 50% short)
            if valid_l and valid_s: # Use already computed valid_l and valid_s
                combined = (
                    0.5 * period_returns[valid_l].mean(axis=1) +
                    0.5 * (-period_returns[valid_s].mean(axis=1)) # Use negative for short leg
                )
                portfolio_returns_list.append(combined)
        
        # Concat once after the loop for performance
        portfolio_returns = pd.concat(portfolio_returns_list) if portfolio_returns_list else pd.Series(dtype=float)
        long_returns = pd.concat(long_returns_list) if long_returns_list else pd.Series(dtype=float)
        short_returns = pd.concat(short_returns_list) if short_returns_list else pd.Series(dtype=float)
        
        # Compute cumulative returns
        if len(portfolio_returns) > 0:
            cumulative = (1 + portfolio_returns).cumprod() - 1
        else:
            cumulative = pd.Series(dtype=float)
        
        if len(long_returns) > 0:
            long_cum = (1 + long_returns).cumprod() - 1
        else:
            long_cum = pd.Series(dtype=float)
        
        if len(short_returns) > 0:
            short_cum = (1 + short_returns).cumprod() - 1
        else:
            short_cum = pd.Series(dtype=float)
        
        # Performance metrics
        metrics = self._compute_performance_metrics(
            portfolio_returns, long_returns, short_returns
        )
        
        results = {
            "portfolio_returns": portfolio_returns,
            "cumulative_returns": cumulative,
            "long_returns": long_returns,
            "long_cumulative": long_cum,
            "short_returns": short_returns,
            "short_cumulative": short_cum,
            "rebalance_log": rebalance_log,
            "metrics": metrics,
        }
        
        # Log summary
        logger.info(f"Backtest complete (Sharpe rf=0 assumed):")
        for k, v in results["metrics"].items():
            logger.info(f"  {k}: {v}")
        
        save_to_cache("backtest_results", results)
        
        return results
    
    def _compute_performance_metrics(
        self,
        portfolio_returns: pd.Series,
        long_returns: pd.Series,
        short_returns: pd.Series
    ) -> Dict:
        """
        Calculate key strategy performance statistics.
        
        Returns:
            Dict of performance metrics
        """
        metrics = {}
        
        # Look-ahead bias warning suffix
        lb = " (Look-ahead Bias)"
        
        for name, returns in [
            ("portfolio", portfolio_returns),
            ("long_leg", long_returns),
            ("short_leg", short_returns)
        ]:
            if len(returns) == 0:
                continue
            
            # Total return
            total_ret = (1 + returns).prod() - 1
            metrics[f"{name}_total_return{lb}"] = f"{total_ret:.2%}"
            
            # Annualised return
            n_days = len(returns)
            ann_ret = (1 + total_ret) ** (252 / max(n_days, 1)) - 1
            metrics[f"{name}_annual_return{lb}"] = f"{ann_ret:.2%}"
            
            # Volatility
            vol = returns.std() * np.sqrt(252)
            metrics[f"{name}_volatility{lb}"] = f"{vol:.2%}"
            
            # Sharpe Ratio (assuming rf=0)
            if vol > 0:
                sharpe = ann_ret / vol
                metrics[f"{name}_sharpe_rf0{lb}"] = f"{sharpe:.2f}"
            
            # Max Drawdown
            cum_ret = (1 + returns).cumprod()
            rolling_max = cum_ret.cummax()
            drawdown = (cum_ret - rolling_max) / rolling_max
            max_dd = drawdown.min()
            metrics[f"{name}_max_drawdown{lb}"] = f"{max_dd:.2%}"
            
            # Hit Rate (% of positive days)
            metrics[f"{name}_hit_rate"] = f"{(returns > 0).mean():.2%}"
            
            # Sortino Ratio
            downside = returns[returns < 0].std() * np.sqrt(252)
            if downside > 0:
                metrics[f"{name}_sortino"] = f"{ann_ret / downside:.2f}"
            
            # Calmar Ratio
            max_dd = abs(drawdown.min())
            if max_dd > 0:
                metrics[f"{name}_calmar"] = f"{ann_ret / max_dd:.2f}"
        
        return metrics


# ============================================================================
# PORTFOLIO TRACKING
# ============================================================================

def track_portfolio_performance(
    backtest_results: Dict,
    export_csv: bool = True
) -> pd.DataFrame:
    """
    Create a detailed portfolio performance tracking table.
    
    Args:
        backtest_results: Output from BacktestEngine.run_backtest()
        export_csv: Whether to export the tracking table to CSV
    
    Returns:
        DataFrame with daily portfolio tracking data
    """
    tracking = pd.DataFrame()
    
    if len(backtest_results.get("portfolio_returns", pd.Series())) > 0:
        tracking["daily_return"] = backtest_results["portfolio_returns"]
        tracking["cumulative_return"] = backtest_results["cumulative_returns"]
    
    if len(backtest_results.get("long_returns", pd.Series())) > 0:
        tracking["long_daily"] = backtest_results["long_returns"]
        tracking["long_cumulative"] = backtest_results["long_cumulative"]
    
    if len(backtest_results.get("short_returns", pd.Series())) > 0:
        tracking["short_daily"] = backtest_results["short_returns"]
        tracking["short_cumulative"] = backtest_results["short_cumulative"]
    
    if not tracking.empty:
        # Rolling metrics
        tracking["rolling_vol_30d"] = tracking["daily_return"].rolling(30).std() * np.sqrt(252)
        tracking["rolling_sharpe_30d"] = (
            tracking["daily_return"].rolling(30).mean() * 252 /
            (tracking["daily_return"].rolling(30).std() * np.sqrt(252))
        )
        
        if export_csv:
            path = export_to_csv(tracking, "portfolio_performance")
            logger.info(f"Portfolio tracking exported to: {path}")
    
    return tracking


# ============================================================================
# MAIN BACKTEST PIPELINE
# ============================================================================

def run_backtest_pipeline(
    price_data: Dict[str, pd.DataFrame],
    metrics_df: pd.DataFrame
) -> Tuple[Dict, pd.DataFrame]:
    """
    Run the complete backtesting pipeline.
    
    Args:
        price_data: Historical price data from data_ingestion
        metrics_df: Key metrics from data_ingestion
    
    Returns:
        Tuple of (backtest_results, tracking_df)
    """
    engine = BacktestEngine(price_data, metrics_df)
    results = engine.run_backtest()
    
    tracking = track_portfolio_performance(results)
    
    return results, tracking


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    from data_ingestion import run_data_pipeline
    
    # Run data pipeline
    price_data, financial_data, metrics_df, index_data = run_data_pipeline()
    
    # Run backtest
    results, tracking = run_backtest_pipeline(price_data, metrics_df)
    
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    for k, v in results["metrics"].items():
        print(f"  {k}: {v}")
    
    print(f"\nRebalance events: {len(results['rebalance_log'])}")
    for log in results["rebalance_log"][:5]:
        print(f"  {log['date'].strftime('%Y-%m-%d')}: "
              f"L={log['n_longs']}, S={log['n_shorts']}")
    
    if not tracking.empty:
        print(f"\nTracking table shape: {tracking.shape}")
        print(tracking.tail(10))
