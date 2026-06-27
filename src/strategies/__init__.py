"""Strategies module - Trading strategy implementations."""

from .base import (
    BaseStrategy,
    TechnicalIndicators,
    BaselineStrategy,
    SMAStrategy,
    CompositeStrategy,
    AdaptiveTrendStrategy,
    SafeProfitStrategy,
    StoplossStrategy,
    VolScaleStrategy,
    BlendedStrategy,
    BlendedTunedStrategy,
    BlendedMOTunedStrategy,
    BlendedRobustStrategy,
    BlendedRobustSafeStrategy,
    SMAStoplossStrategy,
    SMAVolFilterStrategy,
    SMASmoothStopStrategy,
    AdaptiveBaselineStrategy,
    EnsembleStrategy,
    BlendedRobustEnsembleStrategy,
    StrategyFactory,
)
from .chart_pattern_strategy import ChartPatternStrategy

# Register the new strategy in the factory
StrategyFactory.register("chart_pattern", ChartPatternStrategy)

__all__ = [
    "BaseStrategy",
    "TechnicalIndicators",
    "BaselineStrategy",
    "SMAStrategy",
    "CompositeStrategy",
    "AdaptiveTrendStrategy",
    "SafeProfitStrategy",
    "StoplossStrategy",
    "VolScaleStrategy",
    "BlendedStrategy",
    "BlendedTunedStrategy",
    "BlendedMOTunedStrategy",
    "BlendedRobustStrategy",
    "BlendedRobustSafeStrategy",
    "SMAStoplossStrategy",
    "SMAVolFilterStrategy",
    "SMASmoothStopStrategy",
    "AdaptiveBaselineStrategy",
    "EnsembleStrategy",
    "BlendedRobustEnsembleStrategy",
    "ChartPatternStrategy",
    "StrategyFactory",
]
