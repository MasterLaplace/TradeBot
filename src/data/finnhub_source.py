"""
Historical Candle Source.

Fetches historical daily candles for a single stock symbol.

Primary source is **yfinance** (free, no account, no ID — SPEC.md §7).
If a Finnhub API key is configured it is tried first, with automatic
fallback to yfinance on any failure. Either way the caller gets a clean
list of `Candle` objects.

Usage:
    source = HistoricalSource(symbol="AAPL", days=90)
    candles = source.fetch()
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

import requests

from ..core.models import Candle

logger = logging.getLogger(__name__)


class HistoricalSource:
    """Fetch historical daily candles for one symbol (yfinance / Finnhub)."""

    FINNHUB_URL = "https://finnhub.io/api/v1/stock/candle"

    def __init__(
        self,
        symbol: str,
        api_key: str = "",
        resolution: str = "D",
        days: int = 90,
    ):
        """
        Args:
            symbol: Ticker symbol (e.g. "AAPL").
            api_key: Optional Finnhub API key. If empty, yfinance is used directly.
            resolution: Candle resolution (D, W, M).
            days: Days of history to fetch.
        """
        self.symbol = symbol.upper()
        self.api_key = api_key
        self.resolution = resolution
        self.days = days
        self._candles: Optional[List[Candle]] = None

    def fetch(self) -> List[Candle]:
        """Return historical candles (cached after first call)."""
        if self._candles is not None:
            return self._candles

        candles: List[Candle] = []
        if self.api_key:
            try:
                candles = self._fetch_finnhub()
            except Exception as e:
                logger.warning(
                    f"Finnhub failed for {self.symbol} ({e}). Falling back to yfinance."
                )
                candles = []

        if not candles:
            candles = self._fetch_yfinance()

        self._candles = candles
        return candles

    # ------------------------------------------------------------------ Finnhub
    def _fetch_finnhub(self) -> List[Candle]:
        now = int(datetime.now().timestamp())
        start = int((datetime.now() - timedelta(days=self.days)).timestamp())
        params = {
            "symbol": self.symbol,
            "resolution": self.resolution,
            "from": start,
            "to": now,
            "token": self.api_key,
        }
        response = requests.get(self.FINNHUB_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if data.get("s") != "ok":
            raise ValueError(f"Finnhub returned status={data.get('s')} for {self.symbol}")

        return [
            Candle(
                close=float(c),
                open=float(o),
                high=float(h),
                low=float(low),
                volume=float(v),
                timestamp=datetime.fromtimestamp(ts),
            )
            for c, o, h, low, v, ts in zip(
                data["c"], data["o"], data["h"], data["l"], data["v"], data["t"]
            )
        ]

    # ----------------------------------------------------------------- yfinance
    def _fetch_yfinance(self) -> List[Candle]:
        import yfinance as yf

        yf_res = {"D": "1d", "W": "1wk", "M": "1mo"}.get(self.resolution, "1d")
        df = yf.Ticker(self.symbol).history(period=f"{self.days}d", interval=yf_res)
        if df.empty:
            raise ConnectionError(f"yfinance returned no data for {self.symbol}")

        return [
            Candle(
                close=float(row["Close"]),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                volume=float(row["Volume"]),
                timestamp=ts.to_pydatetime(),
            )
            for ts, row in df.iterrows()
        ]
