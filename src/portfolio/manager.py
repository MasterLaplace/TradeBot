"""
Portfolio Manager Module.

Tracks and persists a portfolio (cash + positions) to a local JSON file.

The same class backs BOTH portfolios (SPEC.md §3.6):
- the REAL one (positions I enter by hand after buying on Trade Republic),
- the SIMULATED one (auto-managed by the bot to prove it works),
each in its own file via the `filename` argument.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

from ..core.models import Position, Portfolio
from ..config import get_settings

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Manages reading and writing portfolio state to local storage."""

    def __init__(
        self,
        data_dir: str = "data",
        filename: str = "portfolio.json",
        starting_cash: float = 10000.0,
        label: str = "Portfolio",
        track_cash: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.portfolio_file = self.data_dir / filename
        self.starting_cash = starting_cash
        self.label = label
        # The real portfolio only records positions bought with external money,
        # so it does not model a cash balance (track_cash=False).
        self.track_cash = track_cash
        self.cur = get_settings().currency_symbol
        self.portfolio = self._load()

    def _load(self) -> Portfolio:
        """Load portfolio from JSON file."""
        if not self.portfolio_file.exists():
            # Create a default empty portfolio
            p = Portfolio(cash=self.starting_cash)
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
            return Portfolio(cash=self.starting_cash)

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
        if self.track_cash:
            if self.portfolio.cash < cost:
                logger.warning(f"Insufficient funds to buy {quantity} {symbol}")
                # We still allow it for simulation tracking, but warn.
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

        if self.track_cash:
            self.portfolio.cash += quantity * price

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

        cash = self.portfolio.cash if self.track_cash else 0.0
        total = cash + holdings_value
        unrealized_pnl = holdings_value - cost_basis

        return {
            "cash": round(cash, 2),
            "holdings_value": round(holdings_value, 2),
            "cost_basis": round(cost_basis, 2),
            "total": round(total, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "track_cash": self.track_cash,
            "positions": breakdown,
        }

    def get_valued_summary(self, prices: Dict[str, float]) -> str:
        """Human-readable summary including live P&L."""
        mtm = self.mark_to_market(prices)
        c = self.cur
        lines = [f"💼 *{self.label} (live)*"]
        if mtm["track_cash"]:
            lines.append(f"💵 Cash: `{c}{mtm['cash']:,.2f}`")
        else:
            lines.append(f"💵 Invested: `{c}{mtm['cost_basis']:,.2f}`")
        lines += [
            f"📦 Holdings: `{c}{mtm['holdings_value']:,.2f}`",
            f"💰 Total: `{c}{mtm['total']:,.2f}`",
            f"📈 Unrealized P&L: `{c}{mtm['unrealized_pnl']:+,.2f}`",
            "",
        ]
        if not mtm["positions"]:
            lines.append("No active positions.")
            return "\n".join(lines)
        lines.append("*Positions:*")
        for sym, p in mtm["positions"].items():
            lines.append(
                f"• {sym}: {p['quantity']} @ {c}{p['entry']:.2f} → {c}{p['price']:.2f} "
                f"({p['pnl']:+.2f}, {p['pnl_pct']:+.1f}%)"
            )
        return "\n".join(lines)

    def get_summary(self) -> str:
        """Get a human-readable summary of the portfolio."""
        lines = [
            f"💼 *{self.label} Summary*",
            f"💵 Cash: `{self.cur}{self.portfolio.cash:,.2f}`",
            ""
        ]
        
        if not self.portfolio.positions:
            lines.append("No active positions.")
            return "\n".join(lines)
            
        lines.append("*Open Positions:*")
        for sym, pos in self.portfolio.positions.items():
            lines.append(
                f"• {sym}: {pos.quantity} shares @ {self.cur}{pos.average_entry_price:,.2f}"
            )
            
        return "\n".join(lines)
