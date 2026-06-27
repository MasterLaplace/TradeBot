"""Tests for the dual-portfolio manager: cash mechanics and P&L."""

from src.portfolio.manager import PortfolioManager


def test_simulated_portfolio_tracks_cash(tmp_path):
    pm = PortfolioManager(str(tmp_path), "sim.json", starting_cash=1000.0,
                          label="Sim", track_cash=True)
    assert pm.add_position("AAPL", 2, 100.0)
    assert pm.portfolio.cash == 800.0  # 1000 - 200
    pm.remove_position("AAPL", 1, 120.0)
    assert pm.portfolio.cash == 920.0  # 800 + 120


def test_real_portfolio_ignores_cash(tmp_path):
    pm = PortfolioManager(str(tmp_path), "real.json", starting_cash=0.0,
                          label="Real", track_cash=False)
    pm.add_position("AAPL", 2, 100.0)
    assert pm.portfolio.cash == 0.0  # unchanged
    mtm = pm.mark_to_market({"AAPL": 150.0})
    assert mtm["track_cash"] is False
    assert mtm["holdings_value"] == 300.0
    assert mtm["cost_basis"] == 200.0
    assert mtm["total"] == 300.0  # cash excluded
    assert mtm["unrealized_pnl"] == 100.0


def test_mark_to_market_pnl(tmp_path):
    pm = PortfolioManager(str(tmp_path), "p.json", starting_cash=500.0)
    pm.add_position("MSFT", 5, 100.0)  # cost 500
    mtm = pm.mark_to_market({"MSFT": 110.0})
    assert mtm["unrealized_pnl"] == 50.0
    assert mtm["positions"]["MSFT"]["pnl_pct"] == 10.0


def test_average_entry_price_on_add(tmp_path):
    pm = PortfolioManager(str(tmp_path), "p.json", starting_cash=10000.0)
    pm.add_position("NVDA", 1, 100.0)
    pm.add_position("NVDA", 1, 200.0)
    assert pm.get_position("NVDA").average_entry_price == 150.0


def test_persistence_round_trip(tmp_path):
    pm = PortfolioManager(str(tmp_path), "p.json", starting_cash=1000.0)
    pm.add_position("AAPL", 1, 100.0)
    reloaded = PortfolioManager(str(tmp_path), "p.json", starting_cash=1000.0)
    assert reloaded.get_position("AAPL").quantity == 1
