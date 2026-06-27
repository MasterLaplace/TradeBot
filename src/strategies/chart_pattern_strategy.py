"""
Chart Pattern Strategy Module.

A trading strategy that combines mathematical pattern detection (PIPs/Zigzag)
with advanced technical indicators to make allocation decisions.

This strategy integrates with the existing BaseStrategy architecture,
allowing it to be used seamlessly with the backtesting engine.

Usage:
    strategy = ChartPatternStrategy()
    allocation = strategy.decide(epoch, price_history)
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from ..core.models import Allocation, Price
from ..analysis.pattern_detector import MathPatternDetector, SignalDirection
from ..analysis.technical import AdvancedTechnicalAnalyzer
from .base import BaseStrategy


class ChartPatternStrategy(BaseStrategy):
    """
    Strategy combining chart pattern detection with technical analysis.

    Decision logic:
    1. Compute technical indicators (EMA crossover, RSI)
    2. Run mathematical pattern detection (PIPs)
    3. Combine signals with weighted scoring
    4. Conservative allocation based on confidence

    This strategy does NOT use YOLOv8 (vision detection) because:
    - It would be too slow for per-tick backtesting
    - The math detector covers the same patterns more efficiently
    - YOLOv8 is better suited for the real-time monitor pipeline
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)

        # Pattern detector config
        self._pattern_detector = MathPatternDetector(
            ema_period=self.params.get("ema_period", 5),
            extrema_order=self.params.get("extrema_order", 10),
            pattern_tolerance=self.params.get("pattern_tolerance", 0.02),
        )

        # Technical analyzer config
        self._tech_analyzer = AdvancedTechnicalAnalyzer(
            ema_short=self.params.get("ema_short", 9),
            ema_long=self.params.get("ema_long", 21),
            rsi_period=self.params.get("rsi_period", 14),
        )

        # Signal weights
        self._tech_weight = self.params.get("tech_weight", 0.6)
        self._pattern_weight = self.params.get("pattern_weight", 0.4)

        # Exposure limits
        self._max_exposure = self.params.get("max_exposure", 0.7)
        self._min_exposure = self.params.get("min_exposure", 0.05)
        self._base_exposure = self.params.get("base_exposure", 0.25)

    @property
    def name(self) -> str:
        return "chart_pattern"

    def _compute_weight(self, prices: List[float]) -> float:
        """Compute allocation weight using pattern detection + technical analysis.

        Overrides BaseStrategy._compute_weight to use the new analysis modules.
        """
        # Need enough data for meaningful analysis
        min_data = max(30, self._tech_analyzer.ema_long + 5)
        if len(prices) < min_data:
            return self._base_exposure

        # Build a DataFrame for the analyzers
        df = pd.DataFrame({"close": prices})

        # 1. Technical analysis score
        try:
            enriched_df = self._tech_analyzer.compute_indicators(df)
            tech_score = self._tech_analyzer.get_signal_score(enriched_df)
        except Exception:
            tech_score = 0.0

        # 2. Pattern detection score
        try:
            patterns = self._pattern_detector.detect(df)
            pattern_score = self._compute_pattern_score(patterns)
        except Exception:
            pattern_score = 0.0

        # 3. Weighted combination
        combined = (
            self._tech_weight * tech_score
            + self._pattern_weight * pattern_score
        )

        # 4. Map score to exposure level
        #    combined ranges from -1.0 to 1.0
        #    map to [min_exposure, max_exposure]
        exposure_range = self._max_exposure - self._min_exposure
        exposure = self._min_exposure + (combined + 1.0) / 2.0 * exposure_range

        return max(self._min_exposure, min(self._max_exposure, exposure))

    @staticmethod
    def _compute_pattern_score(patterns) -> float:
        """Compute score from detected patterns, -1.0 to 1.0."""
        if not patterns:
            return 0.0

        score = 0.0
        total_weight = 0.0

        for pattern in patterns:
            w = pattern.confidence
            if pattern.direction == SignalDirection.BUY:
                score += w
            elif pattern.direction == SignalDirection.SELL:
                score -= w
            total_weight += w

        if total_weight == 0:
            return 0.0

        return max(-1.0, min(1.0, score / total_weight))
