"""
Chart Pattern Detection Module.

Implements two complementary approaches for detecting chart patterns:
1. Mathematical (PIPs/Zigzag): Algorithmic extraction of pivot points and
   geometric validation of formations (Double Bottom, Head & Shoulders, etc.)
2. Visual (YOLOv8): Computer vision detection on candlestick chart images
   using the foduucom/stockmarket-pattern-detection-yolov8 model.

A PatternDetector facade combines both approaches for consensus scoring.

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

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

try:
    import mplfinance as mpf
    HAS_MPLFINANCE = True
except ImportError:
    HAS_MPLFINANCE = False


# =============================================================================
# YOLOV8 CLASS MAPPING
# =============================================================================

YOLO_CLASS_MAP: Dict[int, Tuple[str, SignalDirection]] = {
    0: ("Head and shoulders bottom", SignalDirection.BUY),   # Inverse H&S → bullish
    1: ("Head and shoulders top", SignalDirection.SELL),      # H&S → bearish
    2: ("M_Head (Double Top)", SignalDirection.SELL),         # Double top → bearish
    3: ("StockLine (Trendline)", SignalDirection.HOLD),       # Neutral
    4: ("Triangle", SignalDirection.HOLD),                     # Consolidation
    5: ("W_Bottom (Double Bottom)", SignalDirection.BUY),     # Double bottom → bullish
}


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
# VISUAL PATTERN DETECTOR (YOLOv8)
# =============================================================================

class VisualPatternDetector:
    """
    Detect chart patterns using YOLOv8 computer vision.

    Generates candlestick chart images from OHLC data and runs inference
    with the foduucom/stockmarket-pattern-detection-yolov8 model.
    """

    MODEL_NAME = "foduucom/stockmarket-pattern-detection-yolov8"

    def __init__(
        self,
        confidence_threshold: float = 0.30,
        max_time_gap_minutes: int = 10,
    ):
        self.confidence_threshold = confidence_threshold
        self.max_time_gap_minutes = max_time_gap_minutes
        self._model: Optional[object] = None

    def _load_model(self) -> None:
        """Lazy-load the YOLOv8 model."""
        if self._model is None:
            if not HAS_YOLO:
                raise ImportError(
                    "ultralytics is required for visual pattern detection. "
                    "Install with: pip install 'tradebot[analysis]'"
                )
            self._model = YOLO(self.MODEL_NAME)
            logger.info(f"Loaded YOLOv8 model: {self.MODEL_NAME}")

    def detect(
        self,
        df: pd.DataFrame,
        symbol: str = "",
    ) -> List[ChartPattern]:
        """Run visual pattern detection on OHLC data.

        Args:
            df: DataFrame with 'open', 'high', 'low', 'close' columns.
                Optionally 'volume' and a datetime index.
            symbol: Ticker symbol for labeling.

        Returns:
            List of detected ChartPattern objects.
        """
        required_cols = {"open", "high", "low", "close"}
        if not required_cols.issubset(df.columns):
            logger.warning(f"Visual detection requires OHLC columns. Got: {list(df.columns)}")
            return []

        if len(df) < 20:
            return []

        # Check for time gaps
        if self._has_time_gap(df):
            logger.warning("Time gap detected in data, skipping visual detection")
            return []

        # Generate chart image
        image_path = self._generate_chart_image(df)
        if image_path is None:
            return []

        # Run inference
        return self._run_inference(image_path, symbol)

    def _has_time_gap(self, df: pd.DataFrame) -> bool:
        """Check for time gaps exceeding max_time_gap_minutes."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return False

        if len(df.index) < 2:
            return False

        time_diffs = df.index.to_series().diff().dropna()
        max_gap = pd.Timedelta(minutes=self.max_time_gap_minutes)
        return (time_diffs > max_gap).any()

    def _generate_chart_image(self, df: pd.DataFrame) -> Optional[str]:
        """Generate a candlestick chart image from OHLC data."""
        if not HAS_MPLFINANCE:
            logger.warning("mplfinance not installed, cannot generate chart images")
            return None

        import tempfile
        import os

        try:
            chart_df = df[["open", "high", "low", "close"]].copy()
            if "volume" in df.columns:
                chart_df["volume"] = df["volume"]

            # Ensure datetime index
            if not isinstance(chart_df.index, pd.DatetimeIndex):
                chart_df.index = pd.to_datetime(chart_df.index)

            # Save to temp file
            temp_path = os.path.join(tempfile.gettempdir(), f"tradebot_chart_{id(df)}.png")
            mpf.plot(
                chart_df,
                type="candle",
                style="charles",
                savefig=dict(fname=temp_path, dpi=150, bbox_inches="tight"),
                volume="volume" in chart_df.columns,
            )
            return temp_path

        except Exception as e:
            logger.error(f"Failed to generate chart image: {e}")
            return None

    def _run_inference(self, image_path: str, symbol: str) -> List[ChartPattern]:
        """Run YOLOv8 inference on a chart image."""
        self._load_model()

        try:
            results = self._model.predict(
                source=image_path,
                conf=self.confidence_threshold,
                save=False,
                verbose=False,
            )

            patterns = []
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    if class_id in YOLO_CLASS_MAP:
                        name, direction = YOLO_CLASS_MAP[class_id]
                        bbox = box.xyxy[0].tolist()

                        patterns.append(ChartPattern(
                            pattern_name=name,
                            confidence=confidence,
                            direction=direction,
                            detection_method="yolov8",
                            symbol=symbol,
                            timestamp=datetime.now(),
                            metadata={
                                "class_id": class_id,
                                "bbox": bbox,
                                "image_path": image_path,
                            },
                        ))

            return patterns

        except Exception as e:
            logger.error(f"YOLOv8 inference failed: {e}")
            return []


# =============================================================================
# PATTERN DETECTOR FACADE
# =============================================================================

class PatternDetector:
    """
    Facade combining mathematical and visual pattern detection.

    Uses both approaches and produces a consensus score.
    Falls back gracefully if YOLOv8 dependencies are not available.
    """

    def __init__(
        self,
        use_math: bool = True,
        use_vision: bool = True,
        math_kwargs: Optional[dict] = None,
        vision_kwargs: Optional[dict] = None,
    ):
        self.use_math = use_math
        self.use_vision = use_vision and HAS_YOLO and HAS_MPLFINANCE

        self._math = MathPatternDetector(**(math_kwargs or {})) if use_math else None
        self._vision = VisualPatternDetector(**(vision_kwargs or {})) if self.use_vision else None

        if use_vision and not self.use_vision:
            logger.info(
                "Visual pattern detection disabled (missing ultralytics or mplfinance). "
                "Install with: pip install 'tradebot[analysis]'"
            )

    def detect(
        self,
        df: pd.DataFrame,
        symbol: str = "",
    ) -> List[ChartPattern]:
        """Run all available detection methods and return combined results.

        Args:
            df: DataFrame with at least 'close' column. OHLC preferred for vision.
            symbol: Ticker symbol.

        Returns:
            Combined list of ChartPattern objects from all detectors.
        """
        all_patterns: List[ChartPattern] = []

        if self._math:
            try:
                math_patterns = self._math.detect(df, symbol)
                all_patterns.extend(math_patterns)
                logger.info(f"Math detector found {len(math_patterns)} patterns for {symbol}")
            except Exception as e:
                logger.error(f"Math pattern detection failed: {e}")

        if self._vision:
            try:
                vision_patterns = self._vision.detect(df, symbol)
                all_patterns.extend(vision_patterns)
                logger.info(f"Vision detector found {len(vision_patterns)} patterns for {symbol}")
            except Exception as e:
                logger.error(f"Visual pattern detection failed: {e}")

        return all_patterns

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
