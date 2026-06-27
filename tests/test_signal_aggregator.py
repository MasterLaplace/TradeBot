"""Tests for signal aggregation: weighting and BUY/SELL/HOLD thresholds."""

from src.analysis.signal_aggregator import SignalAggregator
from src.core.models import SignalDirection, SIGNAL_WEIGHTS, BUY_THRESHOLD, SELL_THRESHOLD


def test_weights_match_spec():
    agg = SignalAggregator()
    assert abs(agg.weights["technical"] - SIGNAL_WEIGHTS["technical"]) < 1e-9
    assert abs(agg.weights["pattern"] - SIGNAL_WEIGHTS["pattern"]) < 1e-9
    assert abs(agg.weights["sentiment"] - SIGNAL_WEIGHTS["sentiment"]) < 1e-9


def test_strong_technical_triggers_buy():
    agg = SignalAggregator()
    sig = agg.aggregate("AAPL", technical_score=1.0,
                        pattern_score_override=0.5, sentiment_score_override=0.5)
    assert sig.composite_score > BUY_THRESHOLD
    assert sig.direction == SignalDirection.BUY


def test_strong_negative_triggers_sell():
    agg = SignalAggregator()
    sig = agg.aggregate("AAPL", technical_score=-1.0,
                        pattern_score_override=-0.5, sentiment_score_override=-0.5)
    assert sig.composite_score < SELL_THRESHOLD
    assert sig.direction == SignalDirection.SELL


def test_neutral_is_hold():
    agg = SignalAggregator()
    sig = agg.aggregate("AAPL", technical_score=0.0,
                        pattern_score_override=0.0, sentiment_score_override=0.0)
    assert sig.direction == SignalDirection.HOLD


def test_composite_uses_weighting():
    agg = SignalAggregator()
    sig = agg.aggregate("AAPL", technical_score=0.4,
                        pattern_score_override=0.2, sentiment_score_override=-0.4)
    expected = 0.40 * 0.4 + 0.35 * 0.2 + 0.25 * -0.4
    assert abs(sig.composite_score - expected) < 1e-9
