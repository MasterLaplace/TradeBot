"""
Trade Journal.

Append-only log of everything the bot decides and does, persisted to
`data/journal.jsonl` (one JSON object per line). This is what makes
"let it run for months and see what comes out" reviewable after the fact.

Event types:
- "signal": a generated TradingSignal for a symbol
- "trade":  a simulated BUY/SELL executed by the bot
- "equity": a periodic mark-to-market snapshot of the whole portfolio
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class TradeJournal:
    """Append-only JSONL event log for the simulation."""

    def __init__(self, data_dir: str = "data", filename: str = "journal.jsonl"):
        self.path = Path(data_dir) / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, event: dict) -> None:
        event = {"ts": datetime.now().isoformat(), **event}
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.error(f"Failed to write journal event: {e}")

    def log_signal(self, signal) -> None:
        self._append({
            "type": "signal",
            "symbol": signal.symbol,
            "direction": signal.direction.value,
            "confidence": round(signal.confidence, 3),
            "composite": round(signal.composite_score, 3),
            "technical": round(signal.technical_score, 3),
            "pattern": round(signal.pattern_score, 3),
            "sentiment": round(signal.sentiment_score, 3),
            "reasoning": signal.reasoning,
        })

    def log_trade(self, action: str, symbol: str, quantity: float,
                  price: float, reason: str = "") -> None:
        self._append({
            "type": "trade",
            "action": action,
            "symbol": symbol,
            "quantity": round(quantity, 6),
            "price": round(price, 4),
            "value": round(quantity * price, 2),
            "reason": reason,
        })

    def log_equity(self, cash: float, holdings_value: float,
                   total: float, pnl: float, detail: Optional[Dict] = None) -> None:
        self._append({
            "type": "equity",
            "cash": round(cash, 2),
            "holdings_value": round(holdings_value, 2),
            "total": round(total, 2),
            "unrealized_pnl": round(pnl, 2),
            "positions": detail or {},
        })

    def tail(self, n: int = 20) -> List[dict]:
        """Return the last n events (for quick review)."""
        if not self.path.exists():
            return []
        try:
            with open(self.path, "r") as f:
                lines = f.readlines()
            return [json.loads(line) for line in lines[-n:]]
        except Exception as e:
            logger.error(f"Failed to read journal: {e}")
            return []
