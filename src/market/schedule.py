"""
Market Schedule — Active Window (energy sobriety, SPEC.md §5b).

Instead of modelling every exchange calendar, we use a single configurable
"active window" matching the hours when I can actually act on Trade Republic
(weekdays ~8h–23h Europe/Paris by default). Outside that window the
background tasks sleep until the next open — no point analysing what I can't
trade.

The schedule is pure date logic (no I/O), driven by Settings.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from ..config import Settings, get_settings


class MarketSchedule:
    """Configurable weekly active window in a given timezone."""

    def __init__(self, settings: Settings | None = None):
        self._s = settings or get_settings()
        self._tz = ZoneInfo(self._s.timezone)

    # ------------------------------------------------------------------ helpers
    def now(self) -> datetime:
        """Current time in the configured timezone."""
        return datetime.now(self._tz)

    def _is_active_day(self, dt: datetime) -> bool:
        return dt.weekday() in self._s.active_weekday_set

    def is_open(self, at: datetime | None = None) -> bool:
        """True if `at` (default: now) falls inside the active window."""
        dt = at or self.now()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self._tz)
        if not self._is_active_day(dt):
            return False
        return self._s.active_start_hour <= dt.hour < self._s.active_end_hour

    def next_open(self, at: datetime | None = None) -> datetime:
        """Return the next datetime the window opens (now if already open)."""
        dt = at or self.now()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self._tz)
        if self.is_open(dt):
            return dt

        start_h = self._s.active_start_hour
        candidate = dt.replace(hour=start_h, minute=0, second=0, microsecond=0)
        # If today's open already passed (or today is not active), roll forward.
        if candidate <= dt or not self._is_active_day(candidate):
            candidate += timedelta(days=1)
        for _ in range(8):  # at most a week ahead
            if self._is_active_day(candidate):
                return candidate
            candidate += timedelta(days=1)
        return candidate  # should never reach (unless no active days)

    def seconds_until_open(self, at: datetime | None = None) -> float:
        """Seconds to sleep until the next open (0 if currently open)."""
        dt = at or self.now()
        if self.is_open(dt):
            return 0.0
        return max(0.0, (self.next_open(dt) - dt).total_seconds())

    def status_line(self) -> str:
        """Human-readable one-liner for the `status` console command."""
        if self.is_open():
            return f"🟢 Market window OPEN (until {self._s.active_end_hour:02d}h {self._s.timezone})"
        nxt = self.next_open()
        return (
            f"🔴 Market window CLOSED — next open "
            f"{nxt.strftime('%a %d %b %H:%M')} {self._s.timezone}"
        )
