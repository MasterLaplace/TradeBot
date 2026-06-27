"""Tests for the persisted watchlist."""

from src.portfolio.watchlist import Watchlist


def test_seed_and_dedup(tmp_path):
    wl = Watchlist(str(tmp_path), seed=["aapl", "MSFT", "aapl"])
    assert wl.symbols() == ["AAPL", "MSFT"]


def test_add_remove(tmp_path):
    wl = Watchlist(str(tmp_path), seed=[])
    assert wl.add("nvda") is True
    assert "NVDA" in wl
    assert wl.add("NVDA") is False  # duplicate
    assert wl.remove("nvda") is True
    assert wl.remove("NVDA") is False  # already gone


def test_persistence(tmp_path):
    wl = Watchlist(str(tmp_path), seed=["AAPL"])
    wl.add("TSLA")
    reloaded = Watchlist(str(tmp_path), seed=["IGNORED"])
    assert set(reloaded.symbols()) == {"AAPL", "TSLA"}
