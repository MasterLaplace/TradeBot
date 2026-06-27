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
class Position:
    """A specific asset position held in the portfolio."""
    symbol: str
    quantity: float
    average_entry_price: float
    last_updated: datetime

    @property
    def value(self) -> float:
        """Current value assuming entry price (updated dynamically in runner)."""
        return self.quantity * self.average_entry_price


@dataclass
class Portfolio:
    """Mutable portfolio state."""
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    
    # Legacy compat (will be phased out for multi-asset positions)
    asset_a_qty: float = 0.0
    asset_b_qty: float = 0.0

    def get_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol.upper())

    def value(self, price: Price) -> float:
        """Calculate total portfolio value (legacy compat)."""
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
        """Rebalance portfolio to target allocation and return fees paid.

        The method calculates the target quantities for each asset based on
        the current portfolio value and the target allocation, computes the
        fees for the trades, updates the portfolio quantities and cash, and
        returns the total fees charged.
        """
        current_value = self.value(price)
        target_a_value = current_value * allocation.asset_a
        target_b_value = current_value * allocation.asset_b

        current_a_value = self.asset_a_qty * price.asset_a
        current_b_value = self.asset_b_qty * price.asset_b

        trade_a = abs(target_a_value - current_a_value)
        trade_b = abs(target_b_value - current_b_value)

        fees = (trade_a + trade_b) * fee_rate

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
    CHART_PATTERN = auto()


class DataSourceType(Enum):
    """Available data sources."""
    CSV = auto()
    BINANCE_REST = auto()
    BINANCE_WS = auto()
    FINNHUB_REST = auto()
    FINNHUB_WS = auto()


class Command(Enum):
    """Available CLI commands."""
    BACKTEST = "backtest"
    PAPER_TRADE = "paper"
    LIVE = "live"
    FETCH_DATA = "fetch"
    OPTIMIZE = "optimize"
    COMPARE = "compare"
    REPORT = "report"
    MONITOR = "monitor"
    ANALYZE = "analyze"


class SignalDirection(Enum):
    """Trading signal direction."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


# =============================================================================
# NEW DOMAIN MODELS (Quantitative Suite)
# =============================================================================

@dataclass(frozen=True)
class NewsArticle:
    """Immutable news article value object."""
    title: str
    summary: str
    source: str
    url: str
    timestamp: datetime
    related_symbols: List[str] = field(default_factory=list)

    @property
    def age_seconds(self) -> float:
        """Time elapsed since article publication."""
        return (datetime.now() - self.timestamp).total_seconds()


@dataclass(frozen=True)
class SentimentReport:
    """Result of AI-powered sentiment analysis on a news article."""
    company_name: str
    ticker: str
    key_indicators: List[str]
    identified_risks: List[str]
    sentiment_polarity: float  # -1.0 (bearish) to +1.0 (bullish)
    confidence: float  # 0.0 to 1.0
    portfolio_advice: str = "No advice"
    source_url: str = ""
    analyzed_at: Optional[datetime] = None

    def __post_init__(self):
        if not (-1.0 <= self.sentiment_polarity <= 1.0):
            raise ValueError(
                f"Sentiment polarity must be in [-1.0, 1.0], got {self.sentiment_polarity}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Confidence must be in [0.0, 1.0], got {self.confidence}"
            )

    @property
    def is_bullish(self) -> bool:
        return self.sentiment_polarity > 0.2

    @property
    def is_bearish(self) -> bool:
        return self.sentiment_polarity < -0.2


@dataclass(frozen=True)
class ChartPattern:
    """Detected chart pattern (from math PIPs or YOLOv8 vision)."""
    pattern_name: str  # e.g. "W_Bottom", "Head and shoulders top"
    confidence: float  # 0.0 to 1.0
    direction: SignalDirection  # Expected price direction
    detection_method: str  # "math_pips" or "yolov8"
    symbol: str = ""
    timestamp: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)  # Bounding box, pivot points, etc.


@dataclass
class TradingSignal:
    """Consolidated trading signal combining technical, pattern, and sentiment analysis."""
    symbol: str
    direction: SignalDirection
    confidence: float  # 0.0 to 1.0
    timestamp: datetime

    # Component scores
    technical_score: float = 0.0  # -1.0 to 1.0
    pattern_score: float = 0.0  # -1.0 to 1.0
    sentiment_score: float = 0.0  # -1.0 to 1.0

    # Supporting data
    patterns_detected: List[ChartPattern] = field(default_factory=list)
    sentiment_reports: List[SentimentReport] = field(default_factory=list)
    reasoning: str = ""

    @property
    def composite_score(self) -> float:
        """Weighted composite score from all signal sources."""
        weights = {"technical": 0.4, "pattern": 0.35, "sentiment": 0.25}
        return (
            weights["technical"] * self.technical_score
            + weights["pattern"] * self.pattern_score
            + weights["sentiment"] * self.sentiment_score
        )

