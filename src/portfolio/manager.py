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
