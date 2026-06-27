"""
Interactive Console (REPL) — the main entry point (SPEC.md §9).

`tradebot` (no arguments) opens this console. It stays up 24/7: you type
commands to drive the bot, while background tasks (surveillance, scan,
weekly newsletter) run in parallel and sleep outside the active window.

This is deliberately NOT an argument-based CLI.
"""

import asyncio
import logging

from .config import get_settings
from .engine import TradeBotEngine
from .scheduler import Scheduler
from .reporting.report import build_report

logger = logging.getLogger(__name__)

BANNER = r"""
╔══════════════════════════════════════════════════════════╗
║   TradeBot — personal stock-analysis copilot (console)    ║
║   Type 'help' for commands, 'quit' to exit.               ║
╚══════════════════════════════════════════════════════════╝
"""

HELP = """
Commands:
  help                  show this help
  add SYMBOL            add a stock to the watchlist
  remove SYMBOL         remove a stock from the watchlist
  list                  show the watchlist
  analyze SYMBOL [days] one-shot analysis + signal (default 90 days)
  scan                  scan the market for opportunities now
  buy SYMBOL QTY PRICE  record a REAL buy (your Trade Republic order)
  sell SYMBOL QTY PRICE record a REAL sell
  portfolio             real portfolio + live P&L
  sim                   simulated portfolio (bot-managed) + live P&L
  report [n]            summary of recent journal activity (default 30)
  status                market window, background tasks
  start | stop          start/stop background surveillance
  quit                  exit cleanly
"""


class Console:
    def __init__(self):
        self.settings = get_settings()
        self.engine = TradeBotEngine(notifier=self._build_notifier())
        self.scheduler = Scheduler(self.engine)
        self._running = True

    def _build_notifier(self):
        if self.settings.has_discord:
            from .reporting.discord import DiscordNotifier
            return DiscordNotifier(self.settings.discord_webhook_url)
        if self.settings.has_telegram:
            from .reporting.telegram_bot import TelegramNotifier
            return TelegramNotifier(
                token=self.settings.telegram_bot_token,
                chat_id=self.settings.telegram_chat_id,
            )
        return None

    # ------------------------------------------------------------------ loop
    async def run(self) -> None:
        print(BANNER)
        print(self.settings.summary())
        print(f"\n{self.engine.schedule.status_line()}")
        notif = ("Discord" if self.settings.has_discord
                 else "Telegram" if self.settings.has_telegram else "console only")
        print(f"Notifications: {notif}\n")

        loop = asyncio.get_event_loop()
        while self._running:
            try:
                line = await loop.run_in_executor(None, input, "tradebot> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            line = line.strip()
            if not line:
                continue
            try:
                await self._dispatch(line)
            except Exception as e:
                print(f"⚠️  {e}")

        await self._shutdown()

    async def _shutdown(self) -> None:
        if self.scheduler.running:
            print("Stopping background tasks...")
            await self.scheduler.stop()
        print("👋 Bye.")

    # -------------------------------------------------------------- dispatch
    async def _dispatch(self, line: str) -> None:
        parts = line.split()
        cmd, args = parts[0].lower(), parts[1:]
        handler = getattr(self, f"_cmd_{cmd}", None)
        if handler is None:
            print(f"Unknown command: {cmd}. Type 'help'.")
            return
        await handler(args)

    # -------------------------------------------------------------- commands
    async def _cmd_help(self, args):
        print(HELP)

    async def _cmd_quit(self, args):
        self._running = False

    _cmd_exit = _cmd_quit

    async def _cmd_add(self, args):
        if not args:
            print("Usage: add SYMBOL")
            return
        sym = args[0].upper()
        print(f"✅ {sym} added." if self.engine.watchlist.add(sym)
              else f"{sym} already in watchlist.")

    async def _cmd_remove(self, args):
        if not args:
            print("Usage: remove SYMBOL")
            return
        sym = args[0].upper()
        print(f"🗑️  {sym} removed." if self.engine.watchlist.remove(sym)
              else f"{sym} not in watchlist.")

    async def _cmd_list(self, args):
        syms = self.engine.watchlist.symbols()
        print("Watchlist: " + (", ".join(syms) if syms else "(empty)"))

    async def _cmd_analyze(self, args):
        if not args:
            print("Usage: analyze SYMBOL [days]")
            return
        sym = args[0].upper()
        days = int(args[1]) if len(args) > 1 else 90
        print(f"Analyzing {sym} ({days}d)...")
        signal = await self.engine.analyze_symbol(sym, days=days)
        self.engine.journal.log_signal(signal)
        print(self.engine.format_signal(signal))

    async def _cmd_scan(self, args):
        print("Scanning market...")
        results = self.engine.scan()
        anomalies = results.get("anomalies", [])
        steady = results.get("steady_growth", [])
        if anomalies:
            print("Volatility anomalies:")
            for a in anomalies[:5]:
                print(f"  • {a['symbol']}: {a['type']} (Z {a['z_score']:+.2f}, "
                      f"ret {a['recent_return']:.1%}) @ ${a['close_price']:.2f}")
        if steady:
            print("Steady growth:")
            for s in steady[:5]:
                print(f"  • {s['symbol']}: +{s['total_growth_pct']:.1%} "
                      f"(R² {s['r_squared']:.2f}) @ ${s['close_price']:.2f}")
        if not anomalies and not steady:
            print("No actionable opportunities found.")

    async def _cmd_buy(self, args):
        sym, qty, price = self._parse_trade(args, "buy")
        if sym is None:
            return
        print(f"✅ Recorded REAL BUY {qty} {sym} @ ${price:.2f}"
              if self.engine.real_buy(sym, qty, price) else "❌ Buy rejected.")

    async def _cmd_sell(self, args):
        sym, qty, price = self._parse_trade(args, "sell")
        if sym is None:
            return
        print(f"✅ Recorded REAL SELL {qty} {sym} @ ${price:.2f}"
              if self.engine.real_sell(sym, qty, price)
              else "❌ Sell rejected (position too small or missing).")

    async def _cmd_portfolio(self, args):
        pm = self.engine.real
        prices = self.engine.live_prices(pm.held_symbols())
        print(pm.get_valued_summary(prices))

    async def _cmd_sim(self, args):
        pm = self.engine.sim
        prices = self.engine.live_prices(pm.held_symbols())
        print(pm.get_valued_summary(prices))

    async def _cmd_report(self, args):
        n = int(args[0]) if args else 30
        print(build_report(self.engine, last_n=n))

    async def _cmd_status(self, args):
        print(self.engine.schedule.status_line())
        print(f"Background surveillance: {'🟢 running' if self.scheduler.running else '🔴 stopped'}")
        print(f"Watchlist: {len(self.engine.watchlist)} symbols")

    async def _cmd_start(self, args):
        print("🟢 Background surveillance started."
              if self.scheduler.start() else "Already running.")

    async def _cmd_stop(self, args):
        if self.scheduler.running:
            await self.scheduler.stop()
            print("🔴 Background surveillance stopped.")
        else:
            print("Not running.")

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _parse_trade(args, verb):
        if len(args) < 3:
            print(f"Usage: {verb} SYMBOL QTY PRICE")
            return None, None, None
        try:
            return args[0].upper(), float(args[1]), float(args[2])
        except ValueError:
            print("QTY and PRICE must be numbers.")
            return None, None, None


def main() -> None:
    """Entry point for the `tradebot` command."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(Console().run())
    except KeyboardInterrupt:
        print("\n👋 Bye.")


if __name__ == "__main__":
    main()
