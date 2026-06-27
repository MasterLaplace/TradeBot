"""
Core domain models.

Clean, stock-oriented domain objects shared across the app:
market data (Candle), portfolio state (Position, Portfolio), and the
analysis outputs (NewsArticle, SentimentReport, ChartPattern, TradingSignal).

Signal weights and thresholds follow SPEC.md §4.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime


# =============================================================================
# SIGNAL WEIGHTS & THRESHOLDS (SPEC.md §4)
# =============================================================================

SIGNAL_WEIGHTS = {"technical": 0.40, "pattern": 0.35, "sentiment": 0.25}
BUY_THRESHOLD = 0.20
SELL_THRESHOLD = -0.20


# =============================================================================
# MARKET DATA
# =============================================================================

@dataclass(frozen=True)
class Candle:
    """A single OHLCV candle for one symbol at one point in time."""
    close: float
    timestamp: datetime
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    volume: float = 0.0

    def __post_init__(self):
        if self.close < 0:
            raise ValueError("Close price must be non-negative")


# =============================================================================
# PORTFOLIO
# =============================================================================

@dataclass
class Position:
    """A specific asset position held in a portfolio."""
    symbol: str
    quantity: float
    average_entry_price: float
    last_updated: datetime

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.average_entry_price


@dataclass
class Portfolio:
    """Mutable portfolio state (cash + positions)."""
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)

    def get_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol.upper())


# =============================================================================
# ENUMS
# =============================================================================

class SignalDirection(Enum):
    """Trading signal direction."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


# =============================================================================
# ANALYSIS OUTPUTS
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
    """Result of sentiment analysis (Ollama or rule-based) on news."""
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
    """Detected chart pattern (math/PIPs detection)."""
    pattern_name: str  # e.g. "W_Bottom", "Head and shoulders top"
    confidence: float  # 0.0 to 1.0
    direction: SignalDirection  # Expected price direction
    detection_method: str = "math_pips"
    symbol: str = ""
    timestamp: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)  # pivot points, etc.


@dataclass
class TradingSignal:
    """Consolidated signal combining technical, pattern, and sentiment analysis."""
    symbol: str
    direction: SignalDirection
    confidence: float  # 0.0 to 1.0
    timestamp: datetime

    # Component scores (each in -1.0 .. 1.0)
    technical_score: float = 0.0
    pattern_score: float = 0.0
    sentiment_score: float = 0.0

    # Supporting data
    patterns_detected: List[ChartPattern] = field(default_factory=list)
    sentiment_reports: List[SentimentReport] = field(default_factory=list)
    reasoning: str = ""

    @property
    def composite_score(self) -> float:
        """Weighted composite score from all signal sources (SPEC.md §4)."""
        return (
            SIGNAL_WEIGHTS["technical"] * self.technical_score
            + SIGNAL_WEIGHTS["pattern"] * self.pattern_score
            + SIGNAL_WEIGHTS["sentiment"] * self.sentiment_score
        )
