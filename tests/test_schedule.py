"""Tests for the active-window market schedule (energy sobriety)."""

from datetime import datetime
from zoneinfo import ZoneInfo

from src.config import Settings
from src.market.schedule import MarketSchedule

TZ = ZoneInfo("Europe/Paris")


def _schedule():
    # Default window: weekdays (Mon-Fri) 8h-23h Europe/Paris.
    return MarketSchedule(Settings(_env_file=None))


def test_open_during_weekday_window():
    s = _schedule()
    assert s.is_open(datetime(2026, 6, 24, 10, 0, tzinfo=TZ))  # Wednesday 10:00


def test_closed_before_open_hour():
    s = _schedule()
    assert not s.is_open(datetime(2026, 6, 24, 2, 0, tzinfo=TZ))  # Wednesday 02:00


def test_closed_after_close_hour():
    s = _schedule()
    assert not s.is_open(datetime(2026, 6, 24, 23, 30, tzinfo=TZ))


def test_closed_on_weekend():
    s = _schedule()
    assert not s.is_open(datetime(2026, 6, 27, 12, 0, tzinfo=TZ))  # Saturday


def test_next_open_from_weekend_is_monday():
    s = _schedule()
    nxt = s.next_open(datetime(2026, 6, 27, 12, 0, tzinfo=TZ))  # Saturday
    assert nxt.weekday() == 0  # Monday
    assert nxt.hour == 8


def test_next_open_same_day_before_hours():
    s = _schedule()
    nxt = s.next_open(datetime(2026, 6, 24, 2, 0, tzinfo=TZ))  # Wed 02:00
    assert nxt.day == 24 and nxt.hour == 8


def test_seconds_until_open_zero_when_open():
    s = _schedule()
    assert s.seconds_until_open(datetime(2026, 6, 24, 10, 0, tzinfo=TZ)) == 0.0
