"""
Core domain models and interfaces.

This module defines the core abstractions following SOLID principles:
- Single Responsibility: Each class has one job
- Open/Closed: Extend via inheritance, not modification
- Liskov Substitution: Subtypes are substitutable
- Interface Segregation: Small, focused interfaces
- Dependency Inversion: Depend on abstractions
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Protocol
from datetime import datetime


# =============================================================================
# VALUE OBJECTS
# =============================================================================

@dataclass(frozen=True)
class Price:
    """Immutable price value object."""
    asset_a: float
    asset_b: float
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.asset_a < 0 or self.asset_b < 0:
            raise ValueError("Prices must be non-negative")


@dataclass(frozen=True)
class Allocation:
    """Immutable portfolio allocation value object."""
    asset_a: float
    asset_b: float
    cash: float

    def __post_init__(self):
        total = self.asset_a + self.asset_b + self.cash
        if not (0.99 <= total <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Allocation must sum to 1.0, got {total}")

    def to_dict(self) -> Dict[str, float]:
        return {"Asset A": self.asset_a, "Asset B": self.asset_b, "Cash": self.cash}

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "Allocation":
        return cls(
            asset_a=d.get("Asset A", 0.0),
            asset_b=d.get("Asset B", 0.0),
            cash=d.get("Cash", 1.0)
        )

    @classmethod
    def default(cls) -> "Allocation":
        return cls(asset_a=0.15, asset_b=0.15, cash=0.70)


@dataclass
class Portfolio:
    """Mutable portfolio state."""
    cash: float
    asset_a_qty: float = 0.0
    asset_b_qty: float = 0.0

    def value(self, price: Price) -> float:
        """Calculate total portfolio value."""
        return (
            self.cash +
            self.asset_a_qty * price.asset_a +
            self.asset_b_qty * price.asset_b
        )

    def rebalance(
        self,
        allocation: Allocation,
        price: Price,
        fee_rate: float = 0.001,
    ) -> float:
        """Rebalance portfolio to target allocation. Returns fees paid."""
        current_value = self.value(price)

        target_a_value = current_value * allocation.asset_a
        target_b_value = current_value * allocation.asset_b
        # target_cash = current_value * allocation.cash  # not used directly

        # Calculate trades needed
        current_a_value = self.asset_a_qty * price.asset_a
        current_b_value = self.asset_b_qty * price.asset_b

        trade_a = abs(target_a_value - current_a_value)
        trade_b = abs(target_b_value - current_b_value)

        fees = (trade_a + trade_b) * fee_rate

        # Apply new allocation (after fees)
        net_value = current_value - fees
        self.cash = net_value * allocation.cash
        self.asset_a_qty = (
            (net_value * allocation.asset_a) / price.asset_a if price.asset_a > 0 else 0
        )
        self.asset_b_qty = (
            (net_value * allocation.asset_b) / price.asset_b if price.asset_b > 0 else 0
        )

        return fees


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    strategy_name: str
    initial_capital: float
    final_value: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    portfolio_values: List[float] = field(default_factory=list)
    allocations: List[Allocation] = field(default_factory=list)

    @property
    def alpha(self) -> float:
        """Calculate alpha over benchmark (stored externally)."""
        return self.total_return


# =============================================================================
# INTERFACES (Protocols)
# =============================================================================

class Strategy(Protocol):
    """Interface for trading strategies (Interface Segregation)."""

    @property
    def name(self) -> str:
        """Strategy identifier."""
        ...

    def decide(self, epoch: int, prices: List[Price]) -> Allocation:
        """Make allocation decision based on price history."""
        ...

    def reset(self) -> None:
        """Reset strategy state for new run."""
        ...


class DataSource(Protocol):
    """Interface for data providers (Dependency Inversion)."""

    def fetch_prices(self) -> List[Price]:
        """Fetch price data."""
        ...

    def get_current_price(self) -> Price:
        """Get most recent price."""
        ...


class Reporter(Protocol):
    """Interface for generating reports."""

    def generate(self, result: BacktestResult, output_path: str) -> None:
        """Generate report from backtest result."""
        ...


# =============================================================================
# ENUMS
# =============================================================================

class StrategyType(Enum):
    """Available strategy types."""
    SAFE_PROFIT = auto()
    ADAPTIVE_TREND = auto()
    BASELINE = auto()
    SMA = auto()
    COMPOSITE = auto()
    BLENDED = auto()
    BLENDED_ROBUST = auto()
    BLENDED_ENSEMBLE = auto()


class DataSourceType(Enum):
    """Available data sources."""
    CSV = auto()
    BINANCE_REST = auto()
    BINANCE_WS = auto()


class Command(Enum):
    """Available CLI commands."""
    BACKTEST = "backtest"
    PAPER_TRADE = "paper"
    LIVE = "live"
    FETCH_DATA = "fetch"
    OPTIMIZE = "optimize"
    COMPARE = "compare"
    REPORT = "report"
