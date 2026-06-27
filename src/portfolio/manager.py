"""
Portfolio Manager Module.

Handles tracking and persisting the user's local simulated (or real) portfolio.
Persists data to `data/portfolio.json`.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

from ..core.models import Position, Portfolio

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Manages reading and writing portfolio state to local storage."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.portfolio_file = self.data_dir / "portfolio.json"
        self.portfolio = self._load()

    def _load(self) -> Portfolio:
        """Load portfolio from JSON file."""
        if not self.portfolio_file.exists():
            # Create a default empty portfolio
            p = Portfolio(cash=10000.0)
            self._save(p)
            return p

        try:
            with open(self.portfolio_file, "r") as f:
                data = json.load(f)

            cash = data.get("cash", 0.0)
            positions = {}

            for sym, p_data in data.get("positions", {}).items():
                positions[sym] = Position(
                    symbol=p_data["symbol"],
                    quantity=p_data["quantity"],
                    average_entry_price=p_data["average_entry_price"],
                    last_updated=datetime.fromisoformat(p_data["last_updated"])
                )

            return Portfolio(cash=cash, positions=positions)
        except Exception as e:
            logger.error(f"Failed to load portfolio: {e}")
            return Portfolio(cash=10000.0)

    def _save(self, portfolio: Portfolio = None) -> None:
        """Save portfolio to JSON file."""
        p = portfolio or self.portfolio
        try:
            data = {
                "cash": p.cash,
                "positions": {
                    sym: {
                        "symbol": pos.symbol,
                        "quantity": pos.quantity,
                        "average_entry_price": pos.average_entry_price,
                        "last_updated": pos.last_updated.isoformat()
                    }
                    for sym, pos in p.positions.items()
                }
            }
            with open(self.portfolio_file, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save portfolio: {e}")

    def add_position(self, symbol: str, quantity: float, price: float) -> bool:
        """Add or update a position in the portfolio (BUY)."""
        symbol = symbol.upper()
        if quantity <= 0 or price <= 0:
            return False

        cost = quantity * price
        if self.portfolio.cash < cost:
            logger.warning(f"Insufficient funds to buy {quantity} {symbol}")
            # We still allow it for simulation tracking, but warn.
            # You can comment out the next line to strictly enforce cash limits
            pass

        self.portfolio.cash -= cost

        if symbol in self.portfolio.positions:
            pos = self.portfolio.positions[symbol]
            new_qty = pos.quantity + quantity
            new_avg = ((pos.quantity * pos.average_entry_price) + cost) / new_qty
            self.portfolio.positions[symbol] = Position(
                symbol=symbol,
                quantity=new_qty,
                average_entry_price=new_avg,
                last_updated=datetime.now()
            )
        else:
            self.portfolio.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                average_entry_price=price,
                last_updated=datetime.now()
            )

        self._save()
        return True

    def remove_position(self, symbol: str, quantity: float, price: float) -> bool:
        """Remove or reduce a position (SELL)."""
        symbol = symbol.upper()
        if symbol not in self.portfolio.positions:
            return False

        pos = self.portfolio.positions[symbol]
        if quantity > pos.quantity:
            logger.warning(f"Cannot sell {quantity} {symbol}, only have {pos.quantity}")
            return False

        proceeds = quantity * price
        self.portfolio.cash += proceeds

        new_qty = pos.quantity - quantity
        if new_qty <= 0:
            del self.portfolio.positions[symbol]
        else:
            self.portfolio.positions[symbol] = Position(
                symbol=symbol,
                quantity=new_qty,
                average_entry_price=pos.average_entry_price,
                last_updated=datetime.now()
            )

        self._save()
        return True

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        return self.portfolio.positions.get(symbol.upper())

    def held_symbols(self) -> list:
        """Return the list of currently held symbols."""
        return list(self.portfolio.positions.keys())

    def mark_to_market(self, prices: Dict[str, float]) -> dict:
        """Value the portfolio at current prices.

        Args:
            prices: Mapping of symbol -> current price.

        Returns:
            Dict with cash, holdings_value, total, unrealized P&L, and a
            per-position breakdown (qty, entry, price, value, pnl, pnl_pct).
        """
        breakdown = {}
        holdings_value = 0.0
        cost_basis = 0.0

        for sym, pos in self.portfolio.positions.items():
            price = prices.get(sym, pos.average_entry_price)
            value = pos.quantity * price
            cost = pos.quantity * pos.average_entry_price
            pnl = value - cost
            holdings_value += value
            cost_basis += cost
            breakdown[sym] = {
                "quantity": round(pos.quantity, 6),
                "entry": round(pos.average_entry_price, 4),
                "price": round(price, 4),
                "value": round(value, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round((pnl / cost * 100) if cost > 0 else 0.0, 2),
            }

        total = self.portfolio.cash + holdings_value
        unrealized_pnl = holdings_value - cost_basis

        return {
            "cash": round(self.portfolio.cash, 2),
            "holdings_value": round(holdings_value, 2),
            "total": round(total, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "positions": breakdown,
        }

    def get_valued_summary(self, prices: Dict[str, float]) -> str:
        """Human-readable summary including live P&L."""
        mtm = self.mark_to_market(prices)
        lines = [
            "💼 *Portfolio (live)*",
            f"💵 Cash: `${mtm['cash']:,.2f}`",
            f"📦 Holdings: `${mtm['holdings_value']:,.2f}`",
            f"💰 Total: `${mtm['total']:,.2f}`",
            f"📈 Unrealized P&L: `${mtm['unrealized_pnl']:+,.2f}`",
            "",
        ]
        if not mtm["positions"]:
            lines.append("No active positions.")
            return "\n".join(lines)
        lines.append("*Positions:*")
        for sym, p in mtm["positions"].items():
            lines.append(
                f"• {sym}: {p['quantity']} @ ${p['entry']:.2f} → ${p['price']:.2f} "
                f"({p['pnl']:+.2f}, {p['pnl_pct']:+.1f}%)"
            )
        return "\n".join(lines)

    def get_summary(self) -> str:
        """Get a human-readable summary of the portfolio."""
        lines = [
            f"💼 *Portfolio Summary*",
            f"💵 Cash: `${self.portfolio.cash:,.2f}`",
            ""
        ]
        
        if not self.portfolio.positions:
            lines.append("No active positions.")
            return "\n".join(lines)
            
        lines.append("*Open Positions:*")
        for sym, pos in self.portfolio.positions.items():
            lines.append(
                f"• {sym}: {pos.quantity} shares @ ${pos.average_entry_price:,.2f}"
            )
            
        return "\n".join(lines)
