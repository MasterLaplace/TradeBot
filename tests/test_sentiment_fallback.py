"""Sentiment must degrade gracefully to rule-based when Ollama is down."""

from src.analysis.sentiment import OllamaSentimentAnalyzer, RuleBasedSentiment
from src.core.models import SentimentReport


def test_analyze_falls_back_when_ollama_unreachable():
    # Point at an unroutable port so the Ollama call fails fast.
    analyzer = OllamaSentimentAnalyzer(model="none", base_url="http://127.0.0.1:1")
    report = analyzer.analyze("Company beats earnings, raises guidance, strong growth.")
    assert isinstance(report, SentimentReport)
    assert -1.0 <= report.sentiment_polarity <= 1.0
    assert 0.0 <= report.confidence <= 1.0


def test_rule_based_detects_bullish_language():
    rb = RuleBasedSentiment()
    report = rb.analyze("Record profit and surge in revenue beat expectations.")
    assert report.sentiment_polarity > 0


def test_rule_based_detects_bearish_language():
    rb = RuleBasedSentiment()
    report = rb.analyze("Company faces lawsuit, profit plunge and layoffs amid decline.")
    assert report.sentiment_polarity < 0
