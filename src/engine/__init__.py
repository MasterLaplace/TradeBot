"""Engine module - Backtesting and trading engines."""

from .backtest import (
    MetricsCalculator,
    BacktestConfig,
    BacktestEngine,
    StrategyComparator,
)
from .paper_trading import (
    Trade,
    PaperTradingState,
    PaperTradingEngine,
    DashboardDisplay,
)

__all__ = [
    "MetricsCalculator",
    "BacktestConfig", 
    "BacktestEngine",
    "StrategyComparator",
    "Trade",
    "PaperTradingState",
    "PaperTradingEngine",
    "DashboardDisplay",
]
