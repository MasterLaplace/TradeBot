"""
Data Sources Module.

Provides various data sources following the DataSource protocol.
Implements Dependency Inversion: high-level modules depend on abstractions.

Data Sources:
- CSVDataSource: Load from local CSV files
- BinanceRESTSource: Fetch from Binance REST API
- BinanceWSSource: Stream from Binance WebSocket
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, List, Optional
import time

import pandas as pd
import requests

from ..core.models import Price


# =============================================================================
# BASE DATA SOURCE
# =============================================================================

class BaseDataSource(ABC):
    """Abstract base for data sources."""

    @abstractmethod
    def fetch_prices(self) -> List[Price]:
        """Fetch all available price data."""
        pass

    @abstractmethod
    def get_current_price(self) -> Price:
        """Get the most recent price."""
        pass

    def __iter__(self) -> Iterator[Price]:
        """Iterate over prices."""
        return iter(self.fetch_prices())


# =============================================================================
# CSV DATA SOURCE
# =============================================================================

class CSVDataSource(BaseDataSource):
    """
    Load price data from CSV files.

    Expected format:
    - Index column: 'epoch' or numeric index
    - Required columns: 'Asset A', 'Asset B' (or configurable names)
    """

    def __init__(
        self,
        file_path: str,
        asset_a_col: str = "Asset A",
        asset_b_col: str = "Asset B",
    ):
        self.file_path = Path(file_path)
        self.asset_a_col = asset_a_col
        self.asset_b_col = asset_b_col
        self._data: Optional[pd.DataFrame] = None
        self._prices: Optional[List[Price]] = None

    def _load(self) -> None:
        """Lazy load data from CSV.

        The method reads CSV into a pandas DataFrame, sets the 'epoch'
        column as index if present, validates that required asset columns
        are present, and converts rows into immutable Price objects.
        """
        if self._data is None:
            if not self.file_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.file_path}")

            self._data = pd.read_csv(self.file_path)

            
            if 'epoch' in self._data.columns:
                self._data.set_index('epoch', inplace=True)

            
            if self.asset_a_col not in self._data.columns:
                raise ValueError(
                    "Column '{col}' not found in {file}".format(
                        col=self.asset_a_col, file=self.file_path
                    )
                )
            if self.asset_b_col not in self._data.columns:
                raise ValueError(
                    "Column '{col}' not found in {file}".format(
                        col=self.asset_b_col, file=self.file_path
                    )
                )

            
            self._prices = [
                Price(
                    asset_a=row[self.asset_a_col],
                    asset_b=row[self.asset_b_col],
                )
                for _, row in self._data.iterrows()
            ]

    def fetch_prices(self) -> List[Price]:
        """Return all prices from CSV."""
        self._load()
        return self._prices or []

    def get_current_price(self) -> Price:
        """Return the last price in the dataset."""
        self._load()
        if not self._prices:
            raise ValueError("No price data available")
        return self._prices[-1]

    @property
    def dataframe(self) -> pd.DataFrame:
        """Access underlying DataFrame."""
        self._load()
        return self._data


# =============================================================================
# BINANCE REST DATA SOURCE
# =============================================================================

class BinanceRESTSource(BaseDataSource):
    """
    Fetch price data from Binance REST API.

    No API key required for public endpoints.
    """

    BASE_URL = "https://api.binance.com/api/v3"

    # Interval mappings (to milliseconds)
    INTERVALS = {
        '1m': 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '30m': 30 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000,
    }

    def __init__(
        self,
        symbol_a: str = "BTCUSDT",
        symbol_b: str = "ETHUSDT",
        interval: str = "1h",
        days: int = 30,
    ):
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        self.interval = interval
        self.days = days
        self._prices: Optional[List[Price]] = None

    def _fetch_klines(self, symbol: str) -> List[dict]:
        """Fetch candlestick data from Binance.

        The function iteratively requests the maximum number of klines per
        request (1000), sleeping briefly between calls to obey rate limits
        and concatenating results until the requested date range is covered.
        """
        end_time = int(datetime.now().timestamp() * 1000)
        now = datetime.now()
        start_time = int((now - timedelta(days=self.days)).timestamp() * 1000)

        all_klines = []
        current_start = start_time

        while current_start < end_time:
            params = {
                'symbol': symbol,
                'interval': self.interval,
                'startTime': current_start,
                'endTime': end_time,
                'limit': 1000,
            }

            try:
                response = requests.get(
                    f"{self.BASE_URL}/klines",
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                klines = response.json()

                if not klines:
                    break

                all_klines.extend(klines)
                current_start = (
                    klines[-1][0]
                    + self.INTERVALS.get(self.interval, 3600000)
                )

                time.sleep(0.1)

            except requests.RequestException as e:
                raise ConnectionError("Failed to fetch from Binance") from e

        return all_klines

    def fetch_prices(self) -> List[Price]:
        """Fetch historical prices for both symbols and align them by timestamp.

        The method fetches klines for both symbols, converts them to pandas
        DataFrames, aligns them by the common timestamp using an inner join,
        and returns a list of Price objects for the aligned timestamps.
        """
        if self._prices is not None:
            return self._prices

        
        klines_a = self._fetch_klines(self.symbol_a)
        klines_b = self._fetch_klines(self.symbol_b)

        
        df_a = pd.DataFrame(klines_a, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        df_b = pd.DataFrame(klines_b, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])

        
        df_a['timestamp'] = pd.to_datetime(df_a['open_time'], unit='ms')
        df_b['timestamp'] = pd.to_datetime(df_b['open_time'], unit='ms')

        merged = pd.merge(
            df_a[['timestamp', 'close']].rename(columns={'close': 'price_a'}),
            df_b[['timestamp', 'close']].rename(columns={'close': 'price_b'}),
            on='timestamp',
            how='inner'
        )

        self._prices = [
            Price(
                asset_a=float(row['price_a']),
                asset_b=float(row['price_b']),
                timestamp=row['timestamp'].to_pydatetime()
            )
            for _, row in merged.iterrows()
        ]

        return self._prices

    def get_current_price(self) -> Price:
        """Get current spot price from Binance REST endpoints.

        The method queries the ticker price endpoints for both symbols and
        returns a Price value object with the current timestamp.
        """
        try:
            resp_a = requests.get(
                f"{self.BASE_URL}/ticker/price",
                params={'symbol': self.symbol_a},
                timeout=10
            )
            resp_b = requests.get(
                f"{self.BASE_URL}/ticker/price",
                params={'symbol': self.symbol_b},
                timeout=10
            )

            resp_a.raise_for_status()
            resp_b.raise_for_status()

            return Price(
                asset_a=float(resp_a.json()['price']),
                asset_b=float(resp_b.json()['price']),
                timestamp=datetime.now()
            )
        except requests.RequestException as e:
            raise ConnectionError("Failed to get current price") from e

    def save_to_csv(self, output_path: str) -> None:
        """Save fetched historical prices to a CSV file for offline use.

        The CSV includes the 'epoch' index and provides 'Asset A' and 'Asset B'
        columns suitable for backtesting inputs.
        """
        prices = self.fetch_prices()

        data = {
            'epoch': range(len(prices)),
            'Asset A': [p.asset_a for p in prices],
            'Asset B': [p.asset_b for p in prices],
            'Cash': [1.0] * len(prices),
        }

        df = pd.DataFrame(data)
        df.set_index('epoch', inplace=True)
        df.to_csv(output_path)


# =============================================================================
# DATA SOURCE FACTORY
# =============================================================================

class DataSourceFactory:
    """Factory for creating data source instances."""

    @staticmethod
    def from_csv(file_path: str, **kwargs) -> CSVDataSource:
        """Create CSV data source."""
        return CSVDataSource(file_path, **kwargs)

    @staticmethod
    def from_binance(
        symbol_a: str = "BTCUSDT",
        symbol_b: str = "ETHUSDT",
        interval: str = "1h",
        days: int = 30,
    ) -> BinanceRESTSource:
        """Create Binance REST data source."""
        return BinanceRESTSource(symbol_a, symbol_b, interval, days)
