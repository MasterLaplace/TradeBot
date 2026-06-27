"""
Background Scheduler — energy-sober task loop (SPEC.md §5b).

Runs the bot's autonomous work in the background while the console stays
responsive:
- in the active window: periodically analyze the watchlist, journal signals,
  auto-trade the simulated portfolio, alert on actionable signals;
- outside the window: sleep until the next open (no wasted cycles);
- once per active session: run a market scan;
- weekly: post the newsletter to Discord.

Driven by `start()` / `stop()` from the console.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from .engine import TradeBotEngine
from .core.models import SignalDirection
from .reporting.report import build_weekly_newsletter

logger = logging.getLogger(__name__)

_ALERT_MIN_CONFIDENCE = 0.25
_NEWSLETTER_PERIOD_DAYS = 7


class Scheduler:
    """Owns the background asyncio task that drives autonomous analysis."""

    def __init__(self, engine: TradeBotEngine):
        self.engine = engine
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._scanned_this_session = False
        self._last_newsletter: Optional[datetime] = None

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> bool:
        """Launch the background loop. Returns False if already running."""
        if self._running:
            return False
        self._running = True
        self._task = asyncio.create_task(self._run())
        return True

    async def stop(self) -> None:
        """Stop the background loop cleanly."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    # ------------------------------------------------------------------ loop
    async def _run(self) -> None:
        interval = self.engine.settings.analysis_interval
        try:
            while self._running:
                if not self.engine.schedule.is_open():
                    self._scanned_this_session = False
                    wait = self.engine.schedule.seconds_until_open()
                    logger.info(f"💤 Out of active window — sleeping {wait/3600:.1f}h until open.")
                    await self._sleep(min(wait, 3600))  # re-check at most hourly
                    continue

                await self._active_cycle()
                await self._maybe_newsletter()
                await self._sleep(interval)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Scheduler crashed: {e}")
            self._running = False

    async def _active_cycle(self) -> None:
        """One in-window pass over the watchlist + held symbols."""
        eng = self.engine

        if not self._scanned_this_session:
            try:
                eng.scan()  # results consumed via journal/console; cheap to run once
            except Exception as e:
                logger.warning(f"Scan failed: {e}")
            self._scanned_this_session = True

        symbols = list(dict.fromkeys(
            eng.watchlist.symbols() + eng.sim.held_symbols() + eng.real.held_symbols()
        ))
        for symbol in symbols:
            if not self._running:
                return
            try:
                signal = await eng.analyze_symbol(symbol)
                eng.journal.log_signal(signal)

                trade_line = eng.maybe_paper_trade(signal)
                if trade_line:
                    logger.info(trade_line)
                    await eng.notify(trade_line)

                if (signal.direction != SignalDirection.HOLD
                        and signal.confidence >= _ALERT_MIN_CONFIDENCE):
                    await eng.notify("📣 " + eng.format_signal(signal))
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")

        eng.snapshot_sim_equity()

    async def _maybe_newsletter(self) -> None:
        now = datetime.now()
        if (self._last_newsletter is None
                or (now - self._last_newsletter).days >= _NEWSLETTER_PERIOD_DAYS):
            # Skip the very first pass right after start to avoid an empty letter.
            if self._last_newsletter is None:
                self._last_newsletter = now
                return
            try:
                await self.engine.notify(build_weekly_newsletter(self.engine))
                self._last_newsletter = now
            except Exception as e:
                logger.warning(f"Newsletter failed: {e}")

    async def _sleep(self, seconds: float) -> None:
        """Interruptible sleep that respects stop()."""
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            raise
