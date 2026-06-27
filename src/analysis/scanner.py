"""
Market Scanner Module.

Scans a universe of stocks to find:
- Market Anomalies (high volatility, sharp drops/spikes)
- Steady Growth (consistent positive linear regression slope)

Useful for finding new investment opportunities outside the fixed watchlist.
"""

import logging
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# A small subset of popular tech/S&P symbols to scan by default
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", 
    "TSM", "AVGO", "V", "JPM", "WMT", "UNH", "MA", "PG", "JNJ", "HD",
    "ORCL", "COST", "MRK", "ABBV", "BAC", "CVX", "CRM", "AMD", "NFLX",
    "KO", "PEP", "TMO", "ADBE", "WFC", "DIS", "CSCO", "MCD", "INTC"
]

class MarketScanner:
    """Scans multiple tickers to identify actionable opportunities."""

    def __init__(self, universe: List[str] = None):
        self.universe = universe or DEFAULT_UNIVERSE

    def scan(self, period: str = "3mo") -> Dict[str, List[dict]]:
        """Run the scan over the universe.
        
        Returns a dict categorizing findings:
        - "anomalies": List of dicts describing volatile/abnormal behaviors.
        - "steady_growth": List of dicts describing consistent uptrends.
        """
        logger.info(f"Scanning universe of {len(self.universe)} tickers over {period}...")
        
        try:
            # Download bulk data (threads=True speeds it up)
            data = yf.download(self.universe, period=period, group_by="ticker", threads=True, progress=False)
        except Exception as e:
            logger.error(f"Scanner failed to fetch yfinance data: {e}")
            return {"anomalies": [], "steady_growth": []}

        anomalies = []
        steady_growth = []

        for ticker in self.universe:
            try:
                # yfinance MultiIndex handling
                if len(self.universe) > 1:
                    df = data[ticker].dropna()
                else:
                    df = data.dropna()
                
                if df.empty or len(df) < 20:
                    continue
                
                closes = df["Close"].values
                returns = pd.Series(closes).pct_change().dropna().values
                
                # 1. Anomaly Detection (Z-score of latest return)
                recent_return = returns[-1]
                std_dev = np.std(returns)
                mean_return = np.mean(returns)
                
                if std_dev > 0:
                    z_score = (recent_return - mean_return) / std_dev
                    if abs(z_score) > 2.5:
                        direction = "Drop" if z_score < 0 else "Spike"
                        anomalies.append({
                            "symbol": ticker,
                            "type": direction,
                            "z_score": float(z_score),
                            "recent_return": float(recent_return),
                            "close_price": float(closes[-1])
                        })
                
                # 2. Steady Growth (Linear Regression R^2 and positive slope)
                x = np.arange(len(closes))
                poly_coefs = np.polyfit(x, closes, 1)
                slope = poly_coefs[0]
                
                # Calculate R-squared to measure 'steadiness'
                trendline = np.polyval(poly_coefs, x)
                ss_res = np.sum((closes - trendline) ** 2)
                ss_tot = np.sum((closes - np.mean(closes)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # If slope is positive and R^2 is high (>0.80), it's steadily growing
                if slope > 0 and r_squared > 0.80:
                    steady_growth.append({
                        "symbol": ticker,
                        "slope": float(slope),
                        "r_squared": float(r_squared),
                        "close_price": float(closes[-1]),
                        "total_growth_pct": float((closes[-1] - closes[0]) / closes[0])
                    })
                    
            except Exception as e:
                logger.debug(f"Skipping {ticker} due to error: {e}")

        # Sort results
        anomalies.sort(key=lambda x: abs(x["z_score"]), reverse=True)
        steady_growth.sort(key=lambda x: x["r_squared"], reverse=True)

        return {
            "anomalies": anomalies,
            "steady_growth": steady_growth
        }
