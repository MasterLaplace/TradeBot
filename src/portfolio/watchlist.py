"""
Watchlist — persisted list of followed symbols.

Managed by hand from the console (`add`/`remove`/`list`) and persisted to
`data/watchlist.json`. Seeded from the default watchlist in settings on
first run.
"""

import json
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class Watchlist:
    """A simple, deduplicated, persisted set of ticker symbols."""

    def __init__(self, data_dir: str = "data", seed: List[str] | None = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.file = self.data_dir / "watchlist.json"
        self._symbols: List[str] = self._load(seed or [])

    def _load(self, seed: List[str]) -> List[str]:
        if not self.file.exists():
            symbols = [s.upper() for s in seed]
            self._save(symbols)
            return symbols
        try:
            with open(self.file) as f:
                data = json.load(f)
            return [s.upper() for s in data.get("symbols", [])]
        except Exception as e:
            logger.error(f"Failed to load watchlist: {e}")
            return [s.upper() for s in seed]

    def _save(self, symbols: List[str] | None = None) -> None:
        syms = symbols if symbols is not None else self._symbols
        try:
            with open(self.file, "w") as f:
                json.dump({"symbols": syms}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save watchlist: {e}")

    def add(self, symbol: str) -> bool:
        """Add a symbol. Returns False if already present."""
        symbol = symbol.upper().strip()
        if not symbol or symbol in self._symbols:
            return False
        self._symbols.append(symbol)
        self._save()
        return True

    def remove(self, symbol: str) -> bool:
        """Remove a symbol. Returns False if not present."""
        symbol = symbol.upper().strip()
        if symbol not in self._symbols:
            return False
        self._symbols.remove(symbol)
        self._save()
        return True

    def symbols(self) -> List[str]:
        return list(self._symbols)

    def __contains__(self, symbol: str) -> bool:
        return symbol.upper() in self._symbols

    def __len__(self) -> int:
        return len(self._symbols)
