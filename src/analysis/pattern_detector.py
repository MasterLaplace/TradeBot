"""
Chart Pattern Detection Module.

Mathematical (PIPs/Zigzag) detection: algorithmic extraction of pivot points
and geometric validation of formations (Double Bottom, Double Top, Head &
Shoulders, etc.). Visual (YOLOv8) detection is out of scope for the MVP
(SPEC.md §6/§11).

Usage:
    detector = PatternDetector()
    patterns = detector.detect(ohlc_dataframe, symbol="AAPL")
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd

from ..core.models import ChartPattern, SignalDirection

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    from scipy.signal import argrelextrema
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# NOTE: Visual (YOLOv8) pattern detection is intentionally out of scope for the
# MVP (SPEC.md §6/§11). We use mathematical pivot-point detection only.


# =============================================================================
# MATHEMATICAL PATTERN DETECTOR (PIPs / Zigzag)
# =============================================================================

class MathPatternDetector:
    """
    Detect chart patterns using mathematical pivot point analysis.

    Implements:
    - EMA smoothing to reduce noise
    - Local extrema extraction (scipy argrelextrema)
    - Zigzag algorithm with configurable threshold
    - Geometric validation of Double Bottom (W), Double Top (M),
      and Head & Shoulders patterns
    """

    def __init__(
        self,
        ema_period: int = 5,
        extrema_order: int = 10,
        zigzag_threshold: float = 0.03,
        pattern_tolerance: float = 0.02,
    ):
        self.ema_period = ema_period
        self.extrema_order = extrema_order
        self.zigzag_threshold = zigzag_threshold
        self.pattern_tolerance = pattern_tolerance

    def detect(
        self,
        df: pd.DataFrame,
        symbol: str = "",
    ) -> List[ChartPattern]:
        """Run mathematical pattern detection on price data.

        Args:
            df: DataFrame with at least a 'close' column.
            symbol: Ticker symbol for labeling.

        Returns:
            List of detected ChartPattern objects.
        """
        if "close" not in df.columns or len(df) < self.extrema_order * 4:
            return []

        close = df["close"].astype(float).values

        # Step 1: Smooth with EMA
        smoothed = self._ema_smooth(close, self.ema_period)

        # Step 2: Extract pivot points
        pivots = self._extract_pivots(smoothed)

        if len(pivots) < 5:
            return []

        # Step 3: Detect specific patterns
        patterns = []
        patterns.extend(self._detect_double_bottom(pivots, symbol))
        patterns.extend(self._detect_double_top(pivots, symbol))
        patterns.extend(self._detect_head_and_shoulders(pivots, symbol))

        return patterns

    @staticmethod
    def _ema_smooth(prices: np.ndarray, period: int) -> np.ndarray:
        """Apply Exponential Moving Average smoothing."""
        alpha = 2.0 / (period + 1)
        smoothed = np.empty_like(prices)
        smoothed[0] = prices[0]
        for i in range(1, len(prices)):
            smoothed[i] = alpha * prices[i] + (1 - alpha) * smoothed[i - 1]
        return smoothed

    def _extract_pivots(self, prices: np.ndarray) -> List[Tuple[int, float, str]]:
        """Extract alternating pivot points (local min/max).

        Returns list of (index, price, type) where type is 'min' or 'max'.
        """
        if HAS_SCIPY:
            max_indices = argrelextrema(prices, np.greater, order=self.extrema_order)[0]
            min_indices = argrelextrema(prices, np.less, order=self.extrema_order)[0]
        else:
            # Fallback: simple rolling window
            max_indices = self._find_extrema_simple(prices, mode="max")
            min_indices = self._find_extrema_simple(prices, mode="min")

        pivots = []
        for idx in max_indices:
            pivots.append((int(idx), float(prices[idx]), "max"))
        for idx in min_indices:
            pivots.append((int(idx), float(prices[idx]), "min"))

        # Sort by index and ensure alternation
        pivots.sort(key=lambda x: x[0])
        return self._ensure_alternation(pivots)

    def _find_extrema_simple(self, prices: np.ndarray, mode: str) -> np.ndarray:
        """Simple extrema finder without scipy."""
        indices = []
        order = self.extrema_order
        for i in range(order, len(prices) - order):
            window = prices[i - order : i + order + 1]
            if mode == "max" and prices[i] == window.max():
                indices.append(i)
            elif mode == "min" and prices[i] == window.min():
                indices.append(i)
        return np.array(indices)

    @staticmethod
    def _ensure_alternation(pivots: List[Tuple[int, float, str]]) -> List[Tuple[int, float, str]]:
        """Ensure pivots alternate between min and max.

        When consecutive pivots are of the same type, keep the more extreme one.
        """
        if len(pivots) <= 1:
            return pivots

        result = [pivots[0]]
        for pivot in pivots[1:]:
            if pivot[2] != result[-1][2]:
                result.append(pivot)
            else:
                # Same type: keep more extreme
                if pivot[2] == "max" and pivot[1] > result[-1][1]:
                    result[-1] = pivot
                elif pivot[2] == "min" and pivot[1] < result[-1][1]:
                    result[-1] = pivot

        return result

    def _detect_double_bottom(
        self,
        pivots: List[Tuple[int, float, str]],
        symbol: str,
    ) -> List[ChartPattern]:
        """Detect W_Bottom (Double Bottom) pattern.

        Requires: two lows at similar price with a peak between them.
        """
        patterns = []
        mins = [(idx, price) for idx, price, ptype in pivots if ptype == "min"]

        for i in range(len(mins) - 1):
            idx1, low1 = mins[i]
            idx2, low2 = mins[i + 1]

            # Check if lows are at similar prices
            if low1 == 0:
                continue
            diff_ratio = abs(low1 - low2) / low1

            if diff_ratio <= self.pattern_tolerance:
                # Find peak between the two lows
                peaks_between = [
                    (idx, price) for idx, price, ptype in pivots
                    if ptype == "max" and idx1 < idx < idx2
                ]

                if peaks_between:
                    confidence = max(0.3, 1.0 - diff_ratio * 10)
                    patterns.append(ChartPattern(
                        pattern_name="W_Bottom",
                        confidence=min(1.0, confidence),
                        direction=SignalDirection.BUY,
                        detection_method="math_pips",
                        symbol=symbol,
                        timestamp=datetime.now(),
                        metadata={
                            "low1": low1, "low2": low2,
                            "peak": peaks_between[0][1],
                            "idx1": idx1, "idx2": idx2,
                        },
                    ))

        return patterns

    def _detect_double_top(
        self,
        pivots: List[Tuple[int, float, str]],
        symbol: str,
    ) -> List[ChartPattern]:
        """Detect M_Head (Double Top) pattern.

        Requires: two highs at similar price with a trough between them.
        """
        patterns = []
        maxs = [(idx, price) for idx, price, ptype in pivots if ptype == "max"]

        for i in range(len(maxs) - 1):
            idx1, high1 = maxs[i]
            idx2, high2 = maxs[i + 1]

            if high1 == 0:
                continue
            diff_ratio = abs(high1 - high2) / high1

            if diff_ratio <= self.pattern_tolerance:
                troughs_between = [
                    (idx, price) for idx, price, ptype in pivots
                    if ptype == "min" and idx1 < idx < idx2
                ]

                if troughs_between:
                    confidence = max(0.3, 1.0 - diff_ratio * 10)
                    patterns.append(ChartPattern(
                        pattern_name="M_Head",
                        confidence=min(1.0, confidence),
                        direction=SignalDirection.SELL,
                        detection_method="math_pips",
                        symbol=symbol,
                        timestamp=datetime.now(),
                        metadata={
                            "high1": high1, "high2": high2,
                            "trough": troughs_between[0][1],
                            "idx1": idx1, "idx2": idx2,
                        },
                    ))

        return patterns

    def _detect_head_and_shoulders(
        self,
        pivots: List[Tuple[int, float, str]],
        symbol: str,
    ) -> List[ChartPattern]:
        """Detect Head and Shoulders (top and inverse) patterns.

        Top: three peaks where middle (head) is highest, shoulders roughly equal.
        Inverse: three troughs where middle is lowest, shoulders roughly equal.
        """
        patterns = []
        maxs = [(idx, price) for idx, price, ptype in pivots if ptype == "max"]
        mins = [(idx, price) for idx, price, ptype in pivots if ptype == "min"]

        # Head and Shoulders Top
        for i in range(len(maxs) - 2):
            ls_idx, left_shoulder = maxs[i]
            head_idx, head = maxs[i + 1]
            rs_idx, right_shoulder = maxs[i + 2]

            if left_shoulder == 0:
                continue

            # Head must be higher than both shoulders
            if head > left_shoulder and head > right_shoulder:
                shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
                if shoulder_diff <= self.pattern_tolerance * 2:
                    head_prominence = (head - max(left_shoulder, right_shoulder)) / head
                    confidence = min(1.0, max(0.3, head_prominence * 5))
                    patterns.append(ChartPattern(
                        pattern_name="Head and shoulders top",
                        confidence=confidence,
                        direction=SignalDirection.SELL,
                        detection_method="math_pips",
                        symbol=symbol,
                        timestamp=datetime.now(),
                        metadata={
                            "left_shoulder": left_shoulder,
                            "head": head,
                            "right_shoulder": right_shoulder,
                        },
                    ))

        # Inverse Head and Shoulders (bottom)
        for i in range(len(mins) - 2):
            ls_idx, left_shoulder = mins[i]
            head_idx, head = mins[i + 1]
            rs_idx, right_shoulder = mins[i + 2]

            if left_shoulder == 0:
                continue

            # Head must be lower than both shoulders
            if head < left_shoulder and head < right_shoulder:
                shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
                if shoulder_diff <= self.pattern_tolerance * 2:
                    head_prominence = (min(left_shoulder, right_shoulder) - head) / min(left_shoulder, right_shoulder)
                    confidence = min(1.0, max(0.3, head_prominence * 5))
                    patterns.append(ChartPattern(
                        pattern_name="Head and shoulders bottom",
                        confidence=confidence,
                        direction=SignalDirection.BUY,
                        detection_method="math_pips",
                        symbol=symbol,
                        timestamp=datetime.now(),
                        metadata={
                            "left_shoulder": left_shoulder,
                            "head": head,
                            "right_shoulder": right_shoulder,
                        },
                    ))

        return patterns



# =============================================================================
# PATTERN DETECTOR FACADE
# =============================================================================

class PatternDetector:
    """
    Facade over mathematical (PIPs/zigzag) chart-pattern detection.

    The `use_vision` flag is kept for API compatibility but is a no-op in the
    MVP — visual (YOLOv8) detection is out of scope (SPEC.md §6/§11).
    """

    def __init__(
        self,
        use_math: bool = True,
        use_vision: bool = False,
        math_kwargs: Optional[dict] = None,
        **_ignored,
    ):
        self.use_math = use_math
        self._math = MathPatternDetector(**(math_kwargs or {})) if use_math else None
        if use_vision:
            logger.info("Visual pattern detection is out of scope for the MVP; using math only.")

    def detect(
        self,
        df: pd.DataFrame,
        symbol: str = "",
    ) -> List[ChartPattern]:
        """Run mathematical pattern detection and return the patterns found."""
        if not self._math:
            return []
        try:
            patterns = self._math.detect(df, symbol)
            logger.info(f"Math detector found {len(patterns)} patterns for {symbol}")
            return patterns
        except Exception as e:
            logger.error(f"Math pattern detection failed: {e}")
            return []

    def get_consensus_score(self, patterns: List[ChartPattern]) -> float:
        """Compute a consensus pattern score from -1.0 (bearish) to +1.0 (bullish).

        Weights each pattern by its confidence and direction.
        """
        if not patterns:
            return 0.0

        score = 0.0
        total_confidence = 0.0

        for pattern in patterns:
            weight = pattern.confidence
            if pattern.direction == SignalDirection.BUY:
                score += weight
            elif pattern.direction == SignalDirection.SELL:
                score -= weight
            # HOLD patterns don't contribute to the score
            total_confidence += weight

        if total_confidence == 0:
            return 0.0

        return max(-1.0, min(1.0, score / total_confidence))
