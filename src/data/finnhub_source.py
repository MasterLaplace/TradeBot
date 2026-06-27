"""
Finnhub Data Source Module.

Provides market data from Finnhub API (free tier: 30 req/s).
Supports both REST historical candles and WebSocket real-time prices.

Finnhub covers US and European equities, ETFs, and crypto.

Usage:
    source = FinnhubRESTSource(symbol="AAPL", api_key="...", resolution="D", days=90)
    prices = source.fetch_prices()
"""

from datetime import datetime, timedelta
from typing import List, Optional
import time

import requests

from ..core.models import Price
from .sources import BaseDataSource


# =============================================================================
# FINNHUB REST DATA SOURCE
# =============================================================================

class FinnhubRESTSource(BaseDataSource):
    """
    Fetch historical candle data from Finnhub REST API.

    Free tier limits: 30 API calls/second.
    Resolutions: 1, 5, 15, 30, 60, D, W, M
    """

    BASE_URL = "https://finnhub.io/api/v1"

    RESOLUTION_MAP = {
        "1m": "1",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "1d": "D",
        "1w": "W",
        "1M": "M",
    }

    def __init__(
        self,
        symbol: str,
        api_key: str,
        resolution: str = "D",
        days: int = 90,
        symbol_b: Optional[str] = None,
    ):
        """
        Args:
            symbol: Primary ticker symbol (e.g. "AAPL", "TSLA").
            api_key: Finnhub API key.
            resolution: Candle resolution (1, 5, 15, 30, 60, D, W, M).
            days: Number of days of history to fetch.
            symbol_b: Optional second symbol for pair trading (legacy compat).
        """
        self.symbol = symbol.upper()
        self.symbol_b = symbol_b.upper() if symbol_b else None
        self.api_key = api_key
        self.resolution = self.RESOLUTION_MAP.get(resolution, resolution)
        self.days = days
        self._prices: Optional[List[Price]] = None

    def _fetch_candles(self, symbol: str) -> dict:
        """Fetch candle data from Finnhub for a single symbol.

        Returns the raw JSON response containing 'c' (close), 'h' (high),
        'l' (low), 'o' (open), 'v' (volume), 't' (timestamps), and 's' (status).
        """
        now = int(datetime.now().timestamp())
        start = int((datetime.now() - timedelta(days=self.days)).timestamp())

        params = {
            "symbol": symbol,
            "resolution": self.resolution,
            "from": start,
            "to": now,
            "token": self.api_key,
        }

        try:
            response = requests.get(
                f"{self.BASE_URL}/stock/candle",
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            if data.get("s") == "no_data":
                raise ValueError(
                    f"No data from Finnhub for {symbol} "
                    f"(resolution={self.resolution}, days={self.days})"
                )

            return data

        except (requests.RequestException, ValueError) as e:
            # Fallback to Yahoo Finance if Finnhub fails (e.g., 401/403 for candles)
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Finnhub failed ({e}). Falling back to yfinance for {symbol}...")
            
            try:
                import yfinance as yf
                import pandas as pd
                
                # Map resolution
                yf_res_map = {"D": "1d", "W": "1wk", "M": "1mo"}
                yf_res = yf_res_map.get(self.resolution, "1d")
                
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=f"{self.days}d", interval=yf_res)
                
                if df.empty:
                    raise ValueError(f"yfinance returned empty data for {symbol}")
                    
                # Convert yfinance dataframe to finnhub format
                return {
                    "c": df["Close"].tolist(),
                    "h": df["High"].tolist(),
                    "l": df["Low"].tolist(),
                    "o": df["Open"].tolist(),
                    "v": df["Volume"].tolist(),
                    "t": [int(ts.timestamp()) for ts in df.index],
                    "s": "ok"
                }
            except Exception as yf_e:
                raise ConnectionError(f"Both Finnhub and yfinance failed: Finnhub({e}), yfinance({yf_e})") from yf_e

    def fetch_prices(self) -> List[Price]:
        """Fetch historical prices.

        If symbol_b is set, aligns both series by timestamp and returns
        Price objects with both asset values (legacy pair-trading compat).
        Otherwise, asset_a = close price and asset_b = 0.
        """
        if self._prices is not None:
            return self._prices

        data_a = self._fetch_candles(self.symbol)

        if self.symbol_b:
            time.sleep(0.05)  # Respect rate limit
            data_b = self._fetch_candles(self.symbol_b)

            # Align by timestamps
            timestamps_a = set(data_a["t"])
            timestamps_b = set(data_b["t"])
            common_ts = sorted(timestamps_a & timestamps_b)

            idx_a = {t: i for i, t in enumerate(data_a["t"])}
            idx_b = {t: i for i, t in enumerate(data_b["t"])}

            self._prices = [
                Price(
                    asset_a=float(data_a["c"][idx_a[ts]]),
                    asset_b=float(data_b["c"][idx_b[ts]]),
                    timestamp=datetime.fromtimestamp(ts),
                )
                for ts in common_ts
            ]
        else:
            self._prices = [
                Price(
                    asset_a=float(close),
                    asset_b=0.0,
                    timestamp=datetime.fromtimestamp(ts),
                )
                for close, ts in zip(data_a["c"], data_a["t"])
            ]

        return self._prices

    def get_current_price(self) -> Price:
        """Get real-time quote from Finnhub."""
        try:
            response = requests.get(
                f"{self.BASE_URL}/quote",
                params={"symbol": self.symbol, "token": self.api_key},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            price_a = float(data.get("c", 0))  # Current price

            price_b = 0.0
            if self.symbol_b:
                time.sleep(0.05)
                resp_b = requests.get(
                    f"{self.BASE_URL}/quote",
                    params={"symbol": self.symbol_b, "token": self.api_key},
                    timeout=10,
                )
                resp_b.raise_for_status()
                price_b = float(resp_b.json().get("c", 0))

            return Price(
                asset_a=price_a,
                asset_b=price_b,
                timestamp=datetime.now(),
            )

        except requests.RequestException as e:
            raise ConnectionError(f"Failed to get current price from Finnhub: {e}") from e

    def get_company_profile(self, symbol: Optional[str] = None) -> dict:
        """Fetch company profile data from Finnhub."""
        sym = symbol or self.symbol
        try:
            response = requests.get(
                f"{self.BASE_URL}/stock/profile2",
                params={"symbol": sym, "token": self.api_key},
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to fetch company profile: {e}") from e

    def save_to_csv(self, output_path: str) -> None:
        """Save fetched prices to CSV for offline backtesting."""
        import pandas as pd

        prices = self.fetch_prices()
        data = {
            "epoch": range(len(prices)),
            "Asset A": [p.asset_a for p in prices],
            "Asset B": [p.asset_b for p in prices],
            "timestamp": [p.timestamp.isoformat() if p.timestamp else "" for p in prices],
        }
        df = pd.DataFrame(data)
        df.set_index("epoch", inplace=True)
        df.to_csv(output_path)
