"""
Backtesting Engine Module.

Provides backtesting functionality with proper separation of concerns.

Components:
- BacktestEngine: Core simulation logic
- MetricsCalculator: Performance metrics
- ResultAnalyzer: Analysis and comparison
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import statistics
import math

from ..core.models import Allocation, BacktestResult, Portfolio, Price
from ..strategies.base import BaseStrategy, StrategyFactory
from ..data.sources import BaseDataSource


# =============================================================================
# METRICS CALCULATOR (Single Responsibility)
# =============================================================================

class MetricsCalculator:
    """
    Calculate trading performance metrics.

    Implements common financial metrics:
    - Total return
    - Sharpe ratio (annualized)
    - Sortino ratio
    - Maximum drawdown
    - Win rate
    """

    @staticmethod
    def total_return(initial: float, final: float) -> float:
        """Calculate total return percentage."""
        if initial == 0:
            return 0.0
        return (final - initial) / initial

    @staticmethod
    def sharpe_ratio(
        returns: List[float],
        risk_free_rate: float = 0.0,
        periods_per_year: float = 252 * 6,  # 4h candles
    ) -> float:
        """
        Calculate annualized Sharpe ratio.

        Sharpe = (mean_return - risk_free) / std_return * sqrt(periods)
        """
        if not returns or len(returns) < 2:
            return 0.0

        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)

        if std_return == 0:
            return 0.0

        excess_return = mean_return - (risk_free_rate / periods_per_year)
        return (excess_return / std_return) * math.sqrt(periods_per_year)

    @staticmethod
    def sortino_ratio(
        returns: List[float],
        risk_free_rate: float = 0.0,
        periods_per_year: float = 252 * 6,
    ) -> float:
        """
        Calculate Sortino ratio (downside deviation only).
        """
        if not returns or len(returns) < 2:
            return 0.0

        mean_return = statistics.mean(returns)
        downside_returns = [r for r in returns if r < 0]

        if not downside_returns:
            return float('inf') if mean_return > 0 else 0.0

        downside_std = (
            statistics.stdev(downside_returns)
            if len(downside_returns) > 1
            else abs(downside_returns[0])
        )

        if downside_std == 0:
            return 0.0

        excess_return = mean_return - (risk_free_rate / periods_per_year)
        return (excess_return / downside_std) * math.sqrt(periods_per_year)

    @staticmethod
    def max_drawdown(portfolio_values: List[float]) -> float:
        """
        Calculate maximum drawdown.

        Max DD = max((peak - trough) / peak)
        """
        if len(portfolio_values) < 2:
            return 0.0

        peak = portfolio_values[0]
        max_dd = 0.0

        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, drawdown)

        return max_dd

    @staticmethod
    def win_rate(returns: List[float]) -> float:
        """Calculate percentage of positive returns."""
        if not returns:
            return 0.0
        positive = sum(1 for r in returns if r > 0)
        return positive / len(returns)

    @staticmethod
    def calculate_returns(portfolio_values: List[float]) -> List[float]:
        """Calculate period-over-period returns."""
        if len(portfolio_values) < 2:
            return []
        return [
            (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
            for i in range(1, len(portfolio_values))
            if portfolio_values[i-1] != 0
        ]


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

@dataclass
class BacktestConfig:
    """Configuration for backtest runs."""
    initial_capital: float = 10000.0
    fee_rate: float = 0.001  # 0.1% per trade
    slippage: float = 0.0


class BacktestEngine:
    """
    Core backtesting engine.

    Simulates trading strategy on historical data.

    Usage:
        engine = BacktestEngine(config)
        result = engine.run(strategy, data_source)
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self._metrics = MetricsCalculator()

    def run(
        self,
        strategy: BaseStrategy,
        data_source: BaseDataSource,
    ) -> BacktestResult:
        """
        Run backtest simulation.

        Args:
            strategy: Trading strategy to test
            data_source: Source of price data

        Returns:
            BacktestResult with performance metrics
        """
        # Initialize
        strategy.reset()
        portfolio = Portfolio(cash=self.config.initial_capital)
        prices = data_source.fetch_prices()

        if not prices:
            raise ValueError("No price data available for backtest")

        # Track results
        portfolio_values: List[float] = []
        allocations: List[Allocation] = []
        total_fees = 0.0
        num_trades = 0

        # Simulation loop
        price_history: List[Price] = []

        for epoch, price in enumerate(prices):
            price_history.append(price)

            # Get strategy decision
            allocation = strategy.decide(epoch, price_history)
            allocations.append(allocation)

            # Record value before rebalance
            current_value = portfolio.value(price)
            portfolio_values.append(current_value)

            # Rebalance portfolio
            fees = portfolio.rebalance(allocation, price, self.config.fee_rate)
            if fees > 0:
                num_trades += 1
                total_fees += fees

        # Calculate final metrics
        final_value = (
            portfolio_values[-1]
            if portfolio_values
            else self.config.initial_capital
        )
        returns = self._metrics.calculate_returns(portfolio_values)

        return BacktestResult(
            strategy_name=strategy.name,
            initial_capital=self.config.initial_capital,
            final_value=final_value,
            total_return=self._metrics.total_return(
                self.config.initial_capital, final_value
            ),
            sharpe_ratio=self._metrics.sharpe_ratio(returns),
            max_drawdown=self._metrics.max_drawdown(portfolio_values),
            win_rate=self._metrics.win_rate(returns),
            num_trades=num_trades,
            portfolio_values=portfolio_values,
            allocations=allocations,
        )


# =============================================================================
# STRATEGY COMPARATOR
# =============================================================================

class StrategyComparator:
    """
    Compare multiple strategies on the same data.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.engine = BacktestEngine(config)

    def compare(
        self,
        strategy_names: List[str],
        data_source: BaseDataSource,
    ) -> Dict[str, BacktestResult]:
        """
        Run multiple strategies and compare results.

        Returns dict mapping strategy name to BacktestResult.
        """
        results = {}

        for name in strategy_names:
            try:
                strategy = StrategyFactory.create(name)
                result = self.engine.run(strategy, data_source)
                results[name] = result
            except Exception as e:
                print(f"Warning: Strategy '{name}' failed: {e}")

        return results

    def rank_by_return(
        self,
        results: Dict[str, BacktestResult],
    ) -> List[Tuple[str, BacktestResult]]:
        """Rank strategies by total return."""
        return sorted(
            results.items(),
            key=lambda x: x[1].total_return,
            reverse=True
        )

    def calculate_benchmark(self, data_source: BaseDataSource) -> Dict[str, float]:
        """
        Calculate buy & hold benchmarks.

        Returns:
            Dict with 'asset_a', 'asset_b', '50_50' returns
        """
        prices = data_source.fetch_prices()
        if len(prices) < 2:
            return {'asset_a': 0.0, 'asset_b': 0.0, '50_50': 0.0}

        first, last = prices[0], prices[-1]

        ret_a = (
            (last.asset_a - first.asset_a) / first.asset_a if first.asset_a > 0 else 0
        )
        ret_b = (
            (last.asset_b - first.asset_b) / first.asset_b if first.asset_b > 0 else 0
        )

        return {
            'asset_a': ret_a,
            'asset_b': ret_b,
            '50_50': 0.5 * ret_a + 0.5 * ret_b,
        }
