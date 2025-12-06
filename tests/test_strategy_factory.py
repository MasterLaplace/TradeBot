import pytest

from src.strategies import StrategyFactory


def test_available_strategies():
    # Ensure StrategyFactory has at least the expected strategies
    avail = StrategyFactory.available()
    required = {"safe_profit","adaptive_trend","blended","ensemble","sma"}
    assert required.issubset(set(avail))


def test_create_strategy_instances():
    # Create a few strategies to ensure they instantiate without errors
    for name in ["safe_profit","adaptive_trend","blended","ensemble","sma"]:
        s = StrategyFactory.create(name)
        assert s is not None
        assert hasattr(s, "decide")

