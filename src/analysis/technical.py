"""
Advanced Technical Analysis Module.

Wraps pandas-ta indicators with a clean interface for use in the trading pipeline.
Provides EMA, RSI, MACD, Bollinger Bands, support/resistance levels,
and RSI divergence detection.

Usage:
    analyzer = AdvancedTechnicalAnalyzer()
    df = analyzer.compute_indicators(ohlc_dataframe)
    levels = analyzer.find_support_resistance(df)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# pandas-ta is an optional dependency
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False


class AdvancedTechnicalAnalyzer:
    """
    Technical indicator calculator using pandas-ta.

    Gracefully falls back to basic calculations if pandas-ta is not installed.
    """

    def __init__(
        self,
        ema_short: int = 9,
        ema_long: int = 21,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_length: int = 20,
        bb_std: float = 2.0,
    ):
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_length = bb_length
        self.bb_std = bb_std

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical indicators on a DataFrame.

        Expects columns: 'close' (required), 'high', 'low', 'open', 'volume' (optional).
        Returns the DataFrame enriched with indicator columns.
        """
        result = df.copy()

        if "close" not in result.columns:
            raise ValueError("DataFrame must have a 'close' column")

        close = result["close"].astype(float)

        if HAS_PANDAS_TA:
            result = self._compute_with_pandas_ta(result, close)
        else:
            result = self._compute_fallback(result, close)

        return result

    def _compute_with_pandas_ta(self, df: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        """Compute indicators using pandas-ta library."""
        # EMA
        df[f"ema_{self.ema_short}"] = ta.ema(close, length=self.ema_short)
        df[f"ema_{self.ema_long}"] = ta.ema(close, length=self.ema_long)

        # RSI
        df["rsi"] = ta.rsi(close, length=self.rsi_period)

        # MACD
        macd = ta.macd(close, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        if macd is not None:
            df = pd.concat([df, macd], axis=1)

        # Bollinger Bands
        bb = ta.bbands(close, length=self.bb_length, std=self.bb_std)
        if bb is not None:
            df = pd.concat([df, bb], axis=1)

        return df

    def _compute_fallback(self, df: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        """Fallback indicator computation without pandas-ta."""
        # Simple EMA
        df[f"ema_{self.ema_short}"] = close.ewm(span=self.ema_short, adjust=False).mean()
        df[f"ema_{self.ema_long}"] = close.ewm(span=self.ema_long, adjust=False).mean()

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        return df

    def get_signal_score(self, df: pd.DataFrame) -> float:
        """Compute a technical signal score from -1.0 (bearish) to +1.0 (bullish).

        Combines EMA crossover, RSI levels, and MACD histogram direction
        into a single normalized score.
        """
        if df.empty or len(df) < 2:
            return 0.0

        latest = df.iloc[-1]
        score = 0.0
        n_signals = 0

        # EMA crossover signal
        ema_short_col = f"ema_{self.ema_short}"
        ema_long_col = f"ema_{self.ema_long}"
        if ema_short_col in df.columns and ema_long_col in df.columns:
            ema_s = latest.get(ema_short_col)
            ema_l = latest.get(ema_long_col)
            if pd.notna(ema_s) and pd.notna(ema_l) and ema_l != 0:
                cross = (ema_s - ema_l) / ema_l
                score += max(-1.0, min(1.0, cross * 20))  # Amplify small crossover
                n_signals += 1

        # RSI signal
        rsi = latest.get("rsi")
        if pd.notna(rsi):
            if rsi > 70:
                score += -0.5  # Overbought → bearish
            elif rsi < 30:
                score += 0.5  # Oversold → bullish
            else:
                score += (50 - rsi) / -50  # Linear scale
            n_signals += 1

        # MACD histogram signal
        macd_hist_col = [c for c in df.columns if "MACDh" in c or "macd_hist" in c.lower()]
        if macd_hist_col:
            hist = latest.get(macd_hist_col[0])
            if pd.notna(hist):
                score += max(-1.0, min(1.0, hist * 10))
                n_signals += 1

        return score / n_signals if n_signals > 0 else 0.0

    def find_support_resistance(
        self,
        df: pd.DataFrame,
        window: int = 20,
        num_levels: int = 5,
    ) -> Dict[str, List[float]]:
        """Find key support and resistance levels using local extrema.

        Args:
            df: DataFrame with 'close' column.
            window: Rolling window size for extrema detection.
            num_levels: Maximum number of levels to return.

        Returns:
            Dict with 'support' and 'resistance' lists of price levels.
        """
        close = df["close"].astype(float).values

        if len(close) < window * 2:
            return {"support": [], "resistance": []}

        # Find local minima and maxima
        supports = []
        resistances = []

        for i in range(window, len(close) - window):
            local_slice = close[i - window : i + window + 1]
            if close[i] == local_slice.min():
                supports.append(float(close[i]))
            elif close[i] == local_slice.max():
                resistances.append(float(close[i]))

        # Cluster nearby levels (within 1% of each other)
        supports = self._cluster_levels(supports)[:num_levels]
        resistances = self._cluster_levels(resistances)[:num_levels]

        return {"support": sorted(supports), "resistance": sorted(resistances, reverse=True)}

    @staticmethod
    def _cluster_levels(levels: List[float], tolerance: float = 0.01) -> List[float]:
        """Cluster price levels that are within tolerance of each other.

        Groups nearby levels and returns their averages, producing cleaner
        support/resistance zones.
        """
        if not levels:
            return []

        sorted_levels = sorted(levels)
        clusters: List[List[float]] = [[sorted_levels[0]]]

        for level in sorted_levels[1:]:
            cluster_mean = sum(clusters[-1]) / len(clusters[-1])
            if abs(level - cluster_mean) / cluster_mean <= tolerance:
                clusters[-1].append(level)
            else:
                clusters.append([level])

        # Return average of each cluster, sorted by frequency (most touches first)
        result = [(sum(c) / len(c), len(c)) for c in clusters]
        result.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in result]

    def detect_rsi_divergence(
        self,
        df: pd.DataFrame,
        lookback: int = 30,
    ) -> Optional[str]:
        """Detect bullish or bearish RSI divergence.

        - Bullish divergence: price makes lower lows but RSI makes higher lows
        - Bearish divergence: price makes higher highs but RSI makes lower highs

        Returns "bullish", "bearish", or None.
        """
        if "rsi" not in df.columns or len(df) < lookback:
            return None

        recent = df.tail(lookback)
        close = recent["close"].astype(float).values
        rsi = recent["rsi"].astype(float).values

        # Remove NaN
        valid = ~np.isnan(rsi)
        if valid.sum() < lookback // 2:
            return None

        close = close[valid]
        rsi = rsi[valid]

        mid = len(close) // 2

        # Check first half vs second half
        price_low1, price_low2 = close[:mid].min(), close[mid:].min()
        rsi_low1, rsi_low2 = rsi[:mid].min(), rsi[mid:].min()

        price_high1, price_high2 = close[:mid].max(), close[mid:].max()
        rsi_high1, rsi_high2 = rsi[:mid].max(), rsi[mid:].max()

        # Bullish: price lower low + RSI higher low
        if price_low2 < price_low1 and rsi_low2 > rsi_low1:
            return "bullish"

        # Bearish: price higher high + RSI lower high
        if price_high2 > price_high1 and rsi_high2 < rsi_high1:
            return "bearish"

        return None
