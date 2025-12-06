"""
Trading Strategies Module.

Implements various trading strategies following the Strategy pattern.
Each strategy implements the Strategy protocol for interchangeability.

Design Principles:
- Single Responsibility: Each strategy handles only its specific logic
- Open/Closed: New strategies extend BaseStrategy without modifying it
- Liskov Substitution: All strategies are interchangeable
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import statistics

from ..core.models import Allocation, Price


# =============================================================================
# BASE STRATEGY (Template Method Pattern)
# =============================================================================

class BaseStrategy(ABC):
    """
    Abstract base for all trading strategies.

    Uses Template Method pattern: subclasses override _compute_weight()
    while the base class handles common logic.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        self._price_history_a: List[float] = []
        self._price_history_b: List[float] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier."""
        pass

    def reset(self) -> None:
        """Reset state for new run."""
        self._price_history_a.clear()
        self._price_history_b.clear()

    def decide(self, epoch: int, prices: List[Price]) -> Allocation:
        """
        Make allocation decision.

        Template method that:
        1. Updates price history
        2. Computes weight for each asset
        3. Normalizes and returns allocation
        """
        if prices:
            latest = prices[-1]
            self._price_history_a.append(latest.asset_a)
            self._price_history_b.append(latest.asset_b)

        weight_a = self._compute_weight(self._price_history_a)
        weight_b = self._compute_weight(self._price_history_b)

        return self._normalize_allocation(weight_a, weight_b)

    @abstractmethod
    def _compute_weight(self, prices: List[float]) -> float:
        """Compute allocation weight for a single asset. Override in subclasses."""
        pass

    def _normalize_allocation(self, weight_a: float, weight_b: float) -> Allocation:
        """Normalize weights to valid allocation."""
        total = weight_a + weight_b

        if total > 1.0:
            weight_a = weight_a / total
            weight_b = weight_b / total
            cash = 0.0
        else:
            cash = 1.0 - total

        # Ensure valid bounds
        weight_a = max(0.0, min(1.0, weight_a))
        weight_b = max(0.0, min(1.0, weight_b))
        cash = max(0.0, min(1.0, cash))

        # Re-normalize if needed
        total = weight_a + weight_b + cash
        if total > 0:
            weight_a /= total
            weight_b /= total
            cash /= total

        return Allocation(asset_a=weight_a, asset_b=weight_b, cash=cash)


# =============================================================================
# INDICATOR HELPERS (Single Responsibility)
# =============================================================================

class TechnicalIndicators:
    """Static helper class for technical indicators."""

    @staticmethod
    def sma(prices: List[float], window: int) -> Optional[float]:
        """Simple Moving Average."""
        if len(prices) < window:
            return None
        return sum(prices[-window:]) / window

    @staticmethod
    def volatility(prices: List[float], window: int) -> float:
        """Standard deviation of returns."""
        if len(prices) < window + 1:
            return 0.0
        returns = [(prices[i] / prices[i-1] - 1.0) for i in range(-window, 0)]
        return statistics.pstdev(returns) if returns else 0.0

    @staticmethod
    def momentum(prices: List[float], window: int) -> float:
        """Rate of change over window."""
        if len(prices) < window:
            return 0.0
        if prices[-window] == 0:
            return 0.0
        return (prices[-1] - prices[-window]) / prices[-window]

    @staticmethod
    def drawdown_from_peak(prices: List[float], lookback: int) -> float:
        """Current drawdown from recent peak."""
        if len(prices) < 2:
            return 0.0
        recent = prices[-lookback:] if len(prices) >= lookback else prices
        peak = max(recent)
        if peak == 0:
            return 0.0
        return (peak - prices[-1]) / peak


# =============================================================================
# CONCRETE STRATEGIES
# =============================================================================

class BaselineStrategy(BaseStrategy):
    """
    Simple momentum-based strategy.

    - If price went up: increase exposure
    - If price went down: decrease exposure
    """

    @property
    def name(self) -> str:
        return "baseline"

    def _compute_weight(self, prices: List[float]) -> float:
        if len(prices) < 2:
            return 0.15
        delta = prices[-1] - prices[-2]
        return 0.45 if delta > 0 else 0.15


class SMAStrategy(BaseStrategy):
    """
    Simple Moving Average crossover strategy.

    - Short SMA > Long SMA: bullish
    - Short SMA < Long SMA: bearish
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.short_window = self.params.get("short", 10)
        self.long_window = self.params.get("long", 30)

    @property
    def name(self) -> str:
        return "sma"

    def _compute_weight(self, prices: List[float]) -> float:
        short_sma = TechnicalIndicators.sma(prices, self.short_window)
        long_sma = TechnicalIndicators.sma(prices, self.long_window)

        if short_sma is None or long_sma is None:
            return 0.5

        return 0.7 if short_sma > long_sma else 0.3


class CompositeStrategy(BaseStrategy):
    """
    Multi-indicator strategy combining SMA, stoploss, and volatility scaling.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.short_window = self.params.get("short", 5)
        self.long_window = self.params.get("long", 20)
        self.stop_threshold = self.params.get("threshold", 0.04)
        self.vol_window = self.params.get("vol_window", 14)

    @property
    def name(self) -> str:
        return "composite"

    def _compute_weight(self, prices: List[float]) -> float:
        if len(prices) < self.long_window:
            return 0.25

        # Base weight from SMA
        short_sma = TechnicalIndicators.sma(prices, self.short_window)
        long_sma = TechnicalIndicators.sma(prices, self.long_window)
        weight = 0.7 if (short_sma and long_sma and short_sma > long_sma) else 0.3

        # Apply stoploss
        drawdown = TechnicalIndicators.drawdown_from_peak(prices, self.long_window)
        if drawdown > self.stop_threshold:
            weight = 0.05

        # Volatility scaling
        vol = TechnicalIndicators.volatility(prices, self.vol_window)
        if vol > 0.025:
            weight = min(weight, 0.2)

        return weight


class AdaptiveTrendStrategy(BaseStrategy):
    """
    Adaptive trend-following strategy.

    Features:
    - Trend detection via SMA crossover
    - Momentum confirmation
    - Trailing stop for protection
    - Volatility filter
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.lookback_short = self.params.get("lookback_short", 5)
        self.lookback_long = self.params.get("lookback_long", 15)
        self.bull_exposure = self.params.get("bull_exposure", 0.8)
        self.bear_exposure = self.params.get("bear_exposure", 0.05)
        self.neutral_exposure = self.params.get("neutral_exposure", 0.25)
        self.trailing_stop = self.params.get("trailing_stop", 0.10)
        self.vol_window = self.params.get("vol_window", 14)
        self.high_vol_threshold = self.params.get("high_vol_threshold", 0.03)

    @property
    def name(self) -> str:
        return "adaptive_trend"

    def _compute_weight(self, prices: List[float]) -> float:
        if len(prices) < self.lookback_long + 1:
            return self.neutral_exposure

        # Calculate indicators
        sma_short = TechnicalIndicators.sma(prices, self.lookback_short)
        sma_long = TechnicalIndicators.sma(prices, self.lookback_long)
        momentum = TechnicalIndicators.momentum(prices, self.lookback_short)
        vol = TechnicalIndicators.volatility(prices, self.vol_window)
        drawdown = TechnicalIndicators.drawdown_from_peak(prices, self.lookback_long)

        # Decision logic
        # 1. Trailing stop
        if drawdown > self.trailing_stop:
            return 0.02

        # 2. High volatility filter
        if vol > self.high_vol_threshold:
            return min(self.neutral_exposure, self.bear_exposure + 0.1)

        # 3. Trend detection
        if sma_short and sma_long:
            trend_strength = (sma_short - sma_long) / sma_long if sma_long > 0 else 0

            if sma_short > sma_long and momentum > 0:
                # Bullish
                exposure = self.bull_exposure + min(0.2, trend_strength * 5)
                return min(0.8, exposure)
            elif sma_short < sma_long and momentum < 0:
                # Bearish
                return self.bear_exposure

        return self.neutral_exposure


class SafeProfitStrategy(BaseStrategy):
    """
    Safe profit strategy - combines multiple signals for maximum safety.

    Logic:
    1. Calculate adaptive_trend weight
    2. Calculate composite weight
    3. Use the MORE CONSERVATIVE of the two
    4. Only increase exposure when both agree

    Cross-validated performance:
    - Average Alpha: +19.8% over buy&hold
    - Max Drawdown: 7.28%
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self._adaptive = AdaptiveTrendStrategy(params)
        self._composite = CompositeStrategy(params)

    @property
    def name(self) -> str:
        return "safe_profit"

    def reset(self) -> None:
        super().reset()
        self._adaptive.reset()
        self._composite.reset()

    def _compute_weight(self, prices: List[float]) -> float:
        if len(prices) < 25:
            return 0.15  # Very conservative until enough data

        # Get weights from both strategies
        w_trend = self._adaptive._compute_weight(prices)
        w_comp = self._composite._compute_weight(prices)

        # Use conservative approach
        if w_trend > 0.5 and w_comp > 0.5:
            # Both bullish - allow more exposure
            return min(0.7, max(w_trend, w_comp))
        else:
            # Use minimum (safer)
            return min(w_trend, w_comp)


class StoplossStrategy(BaseStrategy):
    """
    Simple stoploss strategy.

    Maintains exposure unless drawdown from peak exceeds threshold,
    then reduces to minimal exposure.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.base_weight = self.params.get("base", 0.7)
        self.threshold = self.params.get("threshold", 0.05)

    @property
    def name(self) -> str:
        return "stoploss"

    def _compute_weight(self, prices: List[float]) -> float:
        if len(prices) < 2:
            return 0.5

        peak = max(prices)
        current = prices[-1]
        drawdown = (peak - current) / peak if peak > 0 else 0

        if drawdown > self.threshold:
            return 0.0
        return self.base_weight


class VolScaleStrategy(BaseStrategy):
    """
    Volatility scaling strategy.

    Scales position size inversely with volatility.
    High volatility = lower exposure.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.base_weight = self.params.get("base", 0.5)
        self.vol_window = self.params.get("vol_window", 20)
        self.min_weight = self.params.get("min", 0.1)
        self.max_weight = self.params.get("max", 0.9)

    @property
    def name(self) -> str:
        return "volscale"

    def _compute_weight(self, prices: List[float]) -> float:
        if len(prices) < self.vol_window + 1:
            return self.base_weight

        # Compute recent volatility
        vol = TechnicalIndicators.volatility(prices, self.vol_window)
        if vol == 0:
            return self.base_weight

        # Reference volatility (median of historical)
        all_returns = [(prices[i] / prices[i-1] - 1.0) for i in range(1, len(prices))]
        ref_vol = statistics.median([abs(r) for r in all_returns]) if all_returns else vol

        # Scale inversely with volatility
        scale = ref_vol / vol if vol > 0 else 1.0
        weight = self.base_weight * scale

        return max(self.min_weight, min(self.max_weight, weight))


class BlendedStrategy(BaseStrategy):
    """
    Blended strategy combining baseline momentum with composite indicators.

    Formula: weight = blend * baseline + (1 - blend) * composite

    This was the main optimized strategy from Optuna tuning.
    """

    # Default robust parameters from Optuna trial 113
    DEFAULT_PARAMS = {
        "blend": 0.9996436412271156,
        "short": 10,
        "long": 27,
        "threshold": 0.0452921667498627,
        "vol_window": 13,
        "vol_multiplier": 0.7322341609310995,
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        # Use default params if not provided
        self.blend = self.params.get("blend", self.DEFAULT_PARAMS["blend"])
        self.short_window = self.params.get("short", self.DEFAULT_PARAMS["short"])
        self.long_window = self.params.get("long", self.DEFAULT_PARAMS["long"])
        self.threshold = self.params.get("threshold", self.DEFAULT_PARAMS["threshold"])
        self.vol_window = self.params.get("vol_window", self.DEFAULT_PARAMS["vol_window"])
        self.vol_multiplier = self.params.get("vol_multiplier", self.DEFAULT_PARAMS["vol_multiplier"])

    @property
    def name(self) -> str:
        return "blended"

    def _compute_weight(self, prices: List[float]) -> float:
        if len(prices) < 2:
            return 0.5

        # Baseline: momentum
        delta = prices[-1] - prices[-2]
        baseline_w = 0.7 if delta > 0 else 0.3

        # Composite: SMA + stoploss + vol scaling
        sma_short = TechnicalIndicators.sma(prices, self.short_window)
        sma_long = TechnicalIndicators.sma(prices, self.long_window)

        if sma_short is None or sma_long is None:
            target = 0.5
        else:
            target = 0.7 if sma_short > sma_long else 0.3

        # Stoploss
        peak = max(prices)
        if peak > 0 and (peak - prices[-1]) / peak > self.threshold:
            target = 0.0

        # Volatility scaling
        if len(prices) >= self.vol_window + 1:
            vol = TechnicalIndicators.volatility(prices, self.vol_window)
            all_returns = [(prices[i] / prices[i-1] - 1.0) for i in range(1, len(prices))]
            ref_vol = statistics.median([abs(r) for r in all_returns]) if all_returns else vol

            if vol > 0:
                scale = ref_vol / vol
                target = max(0.0, min(1.0, target * (1 + (scale - 1) * self.vol_multiplier)))

        # Blend
        weight = self.blend * baseline_w + (1.0 - self.blend) * target
        return max(0.0, min(1.0, weight))


class BlendedTunedStrategy(BlendedStrategy):
    """
    Blended strategy with pre-tuned parameters from Optuna walk-forward tuning.
    """

    DEFAULT_PARAMS = {
        "blend": 0.5004034618154071,
        "short": 6,
        "long": 39,
        "threshold": 0.010008502229651432,
        "vol_window": 9,
        "vol_multiplier": 0.6940552234824028,
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        # Override with tuned params
        super().__init__(self.DEFAULT_PARAMS)

    @property
    def name(self) -> str:
        return "blended_tuned"


class BlendedMOTunedStrategy(BlendedStrategy):
    """
    Blended strategy with multi-objective tuned parameters from Optuna.
    """

    DEFAULT_PARAMS = {
        "blend": 0.9007572211634083,
        "short": 4,
        "long": 17,
        "threshold": 0.015304920910622851,
        "vol_window": 17,
        "vol_multiplier": 0.6235913304206027,
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(self.DEFAULT_PARAMS)

    @property
    def name(self) -> str:
        return "blended_mo_tuned"


class BlendedRobustStrategy(BlendedStrategy):
    """
    Blended strategy with robust parameters from Optuna trial 113.
    This is the most stable configuration across different market conditions.
    """

    DEFAULT_PARAMS = {
        "blend": 0.9996436412271156,
        "short": 10,
        "long": 27,
        "threshold": 0.0452921667498627,
        "vol_window": 13,
        "vol_multiplier": 0.7322341609310995,
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(self.DEFAULT_PARAMS)

    @property
    def name(self) -> str:
        return "blended_robust"


class BlendedRobustSafeStrategy(BlendedRobustStrategy):
    """
    Blended robust with max exposure cap to reduce drawdown.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.max_exposure = self.params.get("max_exposure", 0.5)

    @property
    def name(self) -> str:
        return "blended_robust_safe"

    def _compute_weight(self, prices: List[float]) -> float:
        weight = super()._compute_weight(prices)
        return min(weight, self.max_exposure)


class SMAStoplossStrategy(BaseStrategy):
    """
    SMA strategy with stoploss protection.
    Combines SMA signal with drawdown-based exit.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.short_window = self.params.get("short", 3)
        self.long_window = self.params.get("long", 20)
        self.threshold = self.params.get("threshold", 0.02)

    @property
    def name(self) -> str:
        return "sma_stoploss"

    def _compute_weight(self, prices: List[float]) -> float:
        # Get SMA target
        sma_short = TechnicalIndicators.sma(prices, self.short_window)
        sma_long = TechnicalIndicators.sma(prices, self.long_window)

        if sma_short is None or sma_long is None:
            target = 0.5
        else:
            target = 0.7 if sma_short > sma_long else 0.3

        # Apply stoploss
        if len(prices) >= 2:
            peak = max(prices)
            if peak > 0 and (peak - prices[-1]) / peak > self.threshold:
                return 0.0

        return target


class SMAVolFilterStrategy(BaseStrategy):
    """
    SMA strategy with volatility filter.
    Reduces exposure when volatility exceeds threshold.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.short_window = self.params.get("short", 3)
        self.long_window = self.params.get("long", 20)
        self.vol_window = self.params.get("vol_window", 10)
        self.vol_threshold = self.params.get("vol_threshold", 0.02)

    @property
    def name(self) -> str:
        return "sma_volfilter"

    def _compute_weight(self, prices: List[float]) -> float:
        # Get SMA target
        sma_short = TechnicalIndicators.sma(prices, self.short_window)
        sma_long = TechnicalIndicators.sma(prices, self.long_window)

        if sma_short is None or sma_long is None:
            target = 0.5
        else:
            target = 0.7 if sma_short > sma_long else 0.3

        # Apply volatility filter
        vol = TechnicalIndicators.volatility(prices, self.vol_window)
        if vol > self.vol_threshold:
            return 0.3  # Reduce exposure

        return target


class SMASmoothStopStrategy(BaseStrategy):
    """
    SMA with smoothing, partial stoploss, and volatility scaling.
    Implements gradual position changes for smoother transitions.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.short_window = self.params.get("short", 3)
        self.long_window = self.params.get("long", 20)
        self.alpha = self.params.get("alpha", 0.2)  # Smoothing factor
        self.stop_threshold = self.params.get("stop_threshold", 0.05)
        self.stop_level = self.params.get("stop_level", 0.2)
        self.vol_window = self.params.get("vol_window", 10)
        self.vol_threshold = self.params.get("vol_threshold", 0.02)
        self.vol_scale_factor = self.params.get("vol_scale_factor", 0.7)
        self._prev_weight = 0.5

    @property
    def name(self) -> str:
        return "sma_smooth_stop"

    def reset(self) -> None:
        super().reset()
        self._prev_weight = 0.5

    def _compute_weight(self, prices: List[float]) -> float:
        # Get SMA target
        sma_short = TechnicalIndicators.sma(prices, self.short_window)
        sma_long = TechnicalIndicators.sma(prices, self.long_window)

        if sma_short is None or sma_long is None:
            target = 0.5
        else:
            target = 0.7 if sma_short > sma_long else 0.3

        # Smoothing (exponential moving average of weight)
        weight = self._prev_weight + self.alpha * (target - self._prev_weight)

        # Volatility scaling
        vol = TechnicalIndicators.volatility(prices, self.vol_window)
        if vol > self.vol_threshold:
            weight *= self.vol_scale_factor

        # Partial stoploss
        if len(prices) >= 2:
            peak = max(prices)
            if peak > 0 and (peak - prices[-1]) / peak > self.stop_threshold:
                weight = min(weight, self.stop_level)

        # Clamp and save
        weight = max(0.0, min(1.0, weight))
        self._prev_weight = weight

        return weight


class AdaptiveBaselineStrategy(BaseStrategy):
    """
    Baseline adjusted by risk gating.
    Applies volatility and stoploss gates to baseline allocation.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.vol_window = self.params.get("vol_window", 10)
        self.vol_threshold = self.params.get("vol_threshold", 0.02)
        self.vol_scale_factor = self.params.get("vol_scale_factor", 0.7)
        self.stop_threshold = self.params.get("stop_threshold", 0.05)
        self.stop_scale = self.params.get("stop_scale", 0.4)

    @property
    def name(self) -> str:
        return "adaptive_baseline"

    def _compute_weight(self, prices: List[float]) -> float:
        # Baseline weight (momentum)
        if len(prices) < 2:
            base_weight = 0.5
        else:
            delta = prices[-1] - prices[-2]
            base_weight = 0.7 if delta > 0 else 0.3

        # Volatility gate
        vol_gate = 1.0
        vol = TechnicalIndicators.volatility(prices, self.vol_window)
        if vol > self.vol_threshold:
            vol_gate = self.vol_scale_factor

        # Stop gate
        stop_gate = 1.0
        if len(prices) >= 2:
            peak = max(prices)
            if peak > 0 and (peak - prices[-1]) / peak > self.stop_threshold:
                stop_gate = self.stop_scale

        # Apply gates
        risk_factor = vol_gate * stop_gate
        return max(0.0, min(1.0, base_weight * risk_factor))


class EnsembleStrategy(BaseStrategy):
    """
    Ensemble strategy averaging allocations from multiple Blended configurations.

    Uses top 5 candidates from Optuna optimization for robust predictions.
    """

    # Top 5 candidates from Optuna robust optimization
    ENSEMBLE_CANDIDATES = [
        {"blend": 0.9996436412271156, "short": 10, "long": 27, "threshold": 0.0452921667498627, "vol_window": 13, "vol_multiplier": 0.7322341609310995},
        {"blend": 0.9995821862855174, "short": 9, "long": 23, "threshold": 0.0431405472291731, "vol_window": 14, "vol_multiplier": 0.7285051684602486},
        {"blend": 0.99956027510904, "short": 9, "long": 51, "threshold": 0.0439471735236884, "vol_window": 10, "vol_multiplier": 0.6304819173928731},
        {"blend": 0.9994743275641822, "short": 10, "long": 23, "threshold": 0.0440539482092565, "vol_window": 18, "vol_multiplier": 0.7221503215581743},
        {"blend": 0.9994293508702308, "short": 8, "long": 29, "threshold": 0.0491851548056736, "vol_window": 12, "vol_multiplier": 0.6668045796733401},
    ]

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self._strategies = [BlendedStrategy(c) for c in self.ENSEMBLE_CANDIDATES]

    @property
    def name(self) -> str:
        return "ensemble"

    def reset(self) -> None:
        super().reset()
        for s in self._strategies:
            s.reset()

    def _compute_weight(self, prices: List[float]) -> float:
        if len(prices) < 2:
            return 0.5

        # Get weight from each candidate
        weights = [s._compute_weight(prices) for s in self._strategies]

        # Average
        return sum(weights) / len(weights)


# Alias for backwards compatibility
BlendedRobustEnsembleStrategy = EnsembleStrategy


# =============================================================================
# STRATEGY FACTORY (Open/Closed Principle)
# =============================================================================

class StrategyFactory:
    """
    Factory for creating strategy instances.

    Follows Open/Closed: add new strategies by registering them,
    not by modifying this class.
    """

    _registry: Dict[str, type] = {
        "safe_profit": SafeProfitStrategy,
        "adaptive_trend": AdaptiveTrendStrategy,
        "baseline": BaselineStrategy,
        "sma": SMAStrategy,
        "composite": CompositeStrategy,
        "stoploss": StoplossStrategy,
        "volscale": VolScaleStrategy,
        "blended": BlendedStrategy,
        "blended_tuned": BlendedTunedStrategy,
        "blended_mo_tuned": BlendedMOTunedStrategy,
        "blended_robust": BlendedRobustStrategy,
        "blended_robust_safe": BlendedRobustSafeStrategy,
        "blended_robust_ensemble": EnsembleStrategy,  # Alias
        "sma_stoploss": SMAStoplossStrategy,
        "sma_volfilter": SMAVolFilterStrategy,
        "sma_smooth_stop": SMASmoothStopStrategy,
        "adaptive_baseline": AdaptiveBaselineStrategy,
        "ensemble": EnsembleStrategy,
    }

    @classmethod
    def register(cls, name: str, strategy_class: type) -> None:
        """Register a new strategy type."""
        cls._registry[name] = strategy_class

    @classmethod
    def create(cls, name: str, params: Optional[Dict[str, Any]] = None) -> BaseStrategy:
        """Create a strategy instance by name."""
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
        return cls._registry[name](params)

    @classmethod
    def available(cls) -> List[str]:
        """List available strategy names."""
        return list(cls._registry.keys())
