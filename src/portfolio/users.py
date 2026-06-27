"""
User registry — multi-user support (one set of portfolios per person).

Each user owns an isolated state directory `data/users/<slug>/` holding their
real portfolio, simulated portfolio, watchlist and journal. This lets a few
people (e.g. a family) share one TradeBot instance without mixing positions.

The registry persists to `data/users.json` and, on first run, migrates any
pre-existing single-user state (`data/portfolio_real.json`, etc.) into the
default user so nothing is lost.
"""

import json
import logging
import re
import shutil
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

DEFAULT_USER = "default"
_LEGACY_FILES = [
    "portfolio_real.json",
    "portfolio_sim.json",
    "watchlist.json",
    "journal.jsonl",
]


def slug(name: str) -> str:
    """Filesystem-safe slug for a user name."""
    s = re.sub(r"[^a-z0-9_-]+", "_", name.strip().lower()).strip("_")
    return s or "user"


class UserRegistry:
    """Tracks known users, the active one, and their state directories."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.users_root = self.data_dir / "users"
        self.users_root.mkdir(parents=True, exist_ok=True)
        self.file = self.data_dir / "users.json"
        self._current = DEFAULT_USER
        self._users: List[str] = [DEFAULT_USER]
        self._load()

    # ------------------------------------------------------------------ state
    def _load(self) -> None:
        if not self.file.exists():
            self._users = [DEFAULT_USER]
            self._current = DEFAULT_USER
            self._ensure_dir(DEFAULT_USER)
            self._migrate_legacy(DEFAULT_USER)
            self._save()
            return
        try:
            data = json.loads(self.file.read_text())
            self._users = data.get("users", [DEFAULT_USER]) or [DEFAULT_USER]
            self._current = data.get("current", self._users[0])
            if self._current not in self._users:
                self._current = self._users[0]
        except Exception as e:
            logger.error(f"Failed to load users registry: {e}")

    def _save(self) -> None:
        try:
            self.file.write_text(
                json.dumps({"current": self._current, "users": self._users}, indent=2)
            )
        except Exception as e:
            logger.error(f"Failed to save users registry: {e}")

    def _ensure_dir(self, name: str) -> Path:
        d = self.users_root / slug(name)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _migrate_legacy(self, into: str) -> None:
        """Move any pre-existing single-user files into `into`'s directory."""
        target = self._ensure_dir(into)
        for fname in _LEGACY_FILES:
            legacy = self.data_dir / fname
            if legacy.exists() and not (target / fname).exists():
                try:
                    shutil.move(str(legacy), str(target / fname))
                    logger.info(f"Migrated {fname} -> users/{slug(into)}/")
                except Exception as e:
                    logger.warning(f"Could not migrate {fname}: {e}")

    # ------------------------------------------------------------------- API
    def current(self) -> str:
        return self._current

    def users(self) -> List[str]:
        return list(self._users)

    def dir_for(self, name: str) -> str:
        """Return (creating if needed) the state directory for a user."""
        return str(self._ensure_dir(name))

    def add(self, name: str) -> bool:
        """Register a new user. Returns False if the name already exists."""
        name = name.strip()
        if not name or name in self._users:
            return False
        self._users.append(name)
        self._ensure_dir(name)
        self._save()
        return True

    def switch(self, name: str) -> bool:
        """Set the active user. Returns False if unknown."""
        if name not in self._users:
            return False
        self._current = name
        self._save()
        return True

    def remove(self, name: str) -> bool:
        """Remove a user from the registry (keeps their files on disk)."""
        if name not in self._users or len(self._users) == 1:
            return False
        self._users.remove(name)
        if self._current == name:
            self._current = self._users[0]
        self._save()
        return True
