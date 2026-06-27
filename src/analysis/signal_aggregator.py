"""
Signal Aggregator Module.

Consolidates signals from three independent analysis sources:
1. Technical indicators (EMA, RSI, MACD) → technical_score
2. Chart pattern detection (PIPs + YOLOv8) → pattern_score
3. News sentiment analysis (Ollama) → sentiment_score

Produces a unified TradingSignal with direction (BUY/SELL/HOLD),
confidence level, and supporting evidence.

Usage:
    aggregator = SignalAggregator()
    signal = aggregator.aggregate(
        symbol="AAPL",
        technical_score=0.6,
        patterns=[...],
        sentiment_reports=[...],
    )
"""

from datetime import datetime
from typing import Dict, List, Optional
import logging

from ..core.models import (
    ChartPattern,
    SentimentReport,
    SignalDirection,
    TradingSignal,
)

logger = logging.getLogger(__name__)


class SignalAggregator:
    """
    Combine multiple analysis signals into a single trading decision.

    Uses configurable weights for each signal source and applies
    conflict resolution logic when signals disagree.
    """

    DEFAULT_WEIGHTS = {
        "technical": 0.40,
        "pattern": 0.35,
        "sentiment": 0.25,
    }

    # Thresholds for signal direction
    BUY_THRESHOLD = 0.20
    SELL_THRESHOLD = -0.20

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        buy_threshold: float = 0.20,
        sell_threshold: float = -0.20,
        min_confidence: float = 0.15,
    ):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.min_confidence = min_confidence

        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def aggregate(
        self,
        symbol: str,
        technical_score: float = 0.0,
        patterns: Optional[List[ChartPattern]] = None,
        sentiment_reports: Optional[List[SentimentReport]] = None,
        pattern_score_override: Optional[float] = None,
        sentiment_score_override: Optional[float] = None,
    ) -> TradingSignal:
        """Aggregate all signal sources into a single TradingSignal.

        Args:
            symbol: Ticker symbol.
            technical_score: Score from technical analysis (-1.0 to 1.0).
            patterns: List of detected chart patterns.
            sentiment_reports: List of sentiment analysis reports.
            pattern_score_override: Override computed pattern score.
            sentiment_score_override: Override computed sentiment score.

        Returns:
            Consolidated TradingSignal.
        """
        patterns = patterns or []
        sentiment_reports = sentiment_reports or []

        # Compute component scores
        tech_score = max(-1.0, min(1.0, technical_score))

        if pattern_score_override is not None:
            pat_score = pattern_score_override
        else:
            pat_score = self._compute_pattern_score(patterns)

        if sentiment_score_override is not None:
            sent_score = sentiment_score_override
        else:
            sent_score = self._compute_sentiment_score(sentiment_reports)

        # Weighted composite
        composite = (
            self.weights["technical"] * tech_score
            + self.weights["pattern"] * pat_score
            + self.weights["sentiment"] * sent_score
        )

        # Determine direction with conflict resolution
        direction, confidence, reasoning = self._resolve_direction(
            composite, tech_score, pat_score, sent_score, patterns, sentiment_reports
        )

        return TradingSignal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            timestamp=datetime.now(),
            technical_score=tech_score,
            pattern_score=pat_score,
            sentiment_score=sent_score,
            patterns_detected=patterns,
            sentiment_reports=sentiment_reports,
            reasoning=reasoning,
        )

    def _compute_pattern_score(self, patterns: List[ChartPattern]) -> float:
        """Compute weighted pattern score from detected patterns."""
        if not patterns:
            return 0.0

        score = 0.0
        total_weight = 0.0

        for pattern in patterns:
            weight = pattern.confidence
            if pattern.direction == SignalDirection.BUY:
                score += weight
            elif pattern.direction == SignalDirection.SELL:
                score -= weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return max(-1.0, min(1.0, score / total_weight))

    def _compute_sentiment_score(self, reports: List[SentimentReport]) -> float:
        """Compute weighted sentiment score from analysis reports."""
        if not reports:
            return 0.0

        score = 0.0
        total_weight = 0.0

        for report in reports:
            weight = report.confidence
            score += report.sentiment_polarity * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return max(-1.0, min(1.0, score / total_weight))

    def _resolve_direction(
        self,
        composite: float,
        tech: float,
        pattern: float,
        sentiment: float,
        patterns: List[ChartPattern],
        reports: List[SentimentReport],
    ) -> tuple:
        """Determine signal direction with conflict resolution.

        Returns (direction, confidence, reasoning).

        Conflict rules:
        - If pattern says BUY but sentiment says SELL → HOLD (conflicting signals)
        - If composite is below min_confidence → HOLD
        - If all sources agree → highest confidence
        """
        reasoning_parts = []

        # Count agreeing/disagreeing signals
        signals = {"tech": tech, "pattern": pattern, "sentiment": sentiment}
        bullish = sum(1 for v in signals.values() if v > self.buy_threshold)
        bearish = sum(1 for v in signals.values() if v < self.sell_threshold)

        # Conflict: strong bullish + strong bearish from different sources
        if bullish > 0 and bearish > 0:
            reasoning_parts.append(
                f"⚠️ Conflict: {bullish} bullish vs {bearish} bearish signals"
            )

            # If composite is still meaningful, follow it cautiously
            if abs(composite) > self.min_confidence:
                direction = SignalDirection.BUY if composite > 0 else SignalDirection.SELL
                confidence = abs(composite) * 0.5  # Reduce confidence due to conflict
                reasoning_parts.append(
                    f"Following composite ({composite:+.2f}) with reduced confidence"
                )
            else:
                direction = SignalDirection.HOLD
                confidence = 0.0
                reasoning_parts.append("Holding due to insufficient consensus")

        elif composite > self.buy_threshold:
            direction = SignalDirection.BUY
            confidence = min(1.0, abs(composite))
            reasoning_parts.append(f"✅ BUY signal (composite={composite:+.2f})")

        elif composite < self.sell_threshold:
            direction = SignalDirection.SELL
            confidence = min(1.0, abs(composite))
            reasoning_parts.append(f"🔴 SELL signal (composite={composite:+.2f})")

        else:
            direction = SignalDirection.HOLD
            confidence = 0.0
            reasoning_parts.append(f"⏸️ HOLD (composite={composite:+.2f} below thresholds)")

        # Add detail about each component
        reasoning_parts.append(
            f"Components: tech={tech:+.2f}, pattern={pattern:+.2f}, sentiment={sentiment:+.2f}"
        )

        if patterns:
            pattern_names = [p.pattern_name for p in patterns[:3]]
            reasoning_parts.append(f"Patterns: {', '.join(pattern_names)}")

        if reports:
            avg_sentiment = sum(r.sentiment_polarity for r in reports) / len(reports)
            reasoning_parts.append(f"Avg news sentiment: {avg_sentiment:+.2f} ({len(reports)} articles)")
            
            # Extract portfolio advice if any
            advices = [r.portfolio_advice for r in reports if r.portfolio_advice and r.portfolio_advice != "No advice"]
            if advices:
                reasoning_parts.append(f"🤖 AI Advice: {advices[0]}")

        reasoning = " | ".join(reasoning_parts)
        return direction, confidence, reasoning
