"""
TradeBot Engine — application core.

The single "brain" called by the interactive console and the background
scheduler. It owns the analysis pipeline, the two portfolios (real +
simulated), the watchlist, the journal, the market schedule, and the
notifier. It has no event loop of its own — the console drives it on
demand and the scheduler drives it in the background.

Pipeline (SPEC.md §4): technical (0.40) + pattern (0.35) + sentiment (0.25)
→ composite score → BUY/SELL/HOLD.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from .config import get_settings
from .core.models import TradingSignal, SignalDirection
from .data.finnhub_source import HistoricalSource
from .data.news_fetcher import NewsFetcher
from .data.quotes import get_current_prices
from .analysis.technical import AdvancedTechnicalAnalyzer
from .analysis.pattern_detector import PatternDetector
from .analysis.sentiment import OllamaSentimentAnalyzer
from .analysis.signal_aggregator import SignalAggregator
from .analysis.scanner import MarketScanner
from .portfolio.manager import PortfolioManager
from .portfolio.journal import TradeJournal
from .portfolio.watchlist import Watchlist
from .market.schedule import MarketSchedule

logger = logging.getLogger(__name__)


class TradeBotEngine:
    """Application core: analysis + dual portfolios + actions."""

    def __init__(self, notifier=None):
        self.settings = get_settings()
        data_dir = self.settings.data_dir

        # Analysis pipeline
        self._tech = AdvancedTechnicalAnalyzer()
        self._patterns = PatternDetector(use_vision=False)  # math-only (SPEC §6)
        self._sentiment = OllamaSentimentAnalyzer(
            model=self.settings.ollama_model,
            base_url=self.settings.ollama_base_url,
        )
        self._aggregator = SignalAggregator()
        self._scanner = MarketScanner()
        self._news = NewsFetcher(
            api_key=self.settings.finnhub_api_key if self.settings.has_finnhub else ""
        )

        # State
        self.watchlist = Watchlist(data_dir, seed=self.settings.watchlist_symbols)
        self.real = PortfolioManager(
            data_dir, "portfolio_real.json",
            starting_cash=self.settings.real_starting_cash, label="Real portfolio",
            track_cash=False,  # records hand-entered positions, not a cash balance
        )
        self.sim = PortfolioManager(
            data_dir, "portfolio_sim.json",
            starting_cash=self.settings.sim_starting_cash, label="Simulated portfolio",
        )
        self.journal = TradeJournal()
        self.schedule = MarketSchedule(self.settings)

        # Notifications (Discord/Telegram) — optional, injected
        self.notifier = notifier

    # ====================================================================
    # ANALYSIS
    # ====================================================================
    async def analyze_symbol(self, symbol: str, days: int = 90) -> TradingSignal:
        """Run the full one-shot pipeline on a single symbol."""
        symbol = symbol.upper()
        logger.info(f"📊 Analyzing {symbol}...")

        try:
            candles = HistoricalSource(
                symbol=symbol,
                api_key=self.settings.finnhub_api_key,
                resolution="D",
                days=days,
            ).fetch()
        except Exception as e:
            logger.error(f"Failed to fetch price data for {symbol}: {e}")
            return self._empty_signal(symbol)

        if not candles:
            return self._empty_signal(symbol)

        df = pd.DataFrame({
            "close": [c.close for c in candles],
            "timestamp": [c.timestamp for c in candles],
        })

        # Technical
        try:
            enriched = self._tech.compute_indicators(df)
            tech_score = self._tech.get_signal_score(enriched)
        except Exception as e:
            logger.error(f"Technical analysis failed: {e}")
            tech_score = 0.0

        # Patterns
        try:
            patterns = self._patterns.detect(df, symbol)
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            patterns = []

        # News sentiment
        sentiment_reports = []
        try:
            articles = await self._news.fetch_company_news(symbol, days=7)
            for article in articles[:5]:
                text = f"{article.title}\n\n{article.summary}"
                pos = self.real.get_position(symbol) or self.sim.get_position(symbol)
                sentiment_reports.append(
                    self._sentiment.analyze(text, source_url=article.url, current_position=pos)
                )
        except Exception as e:
            logger.error(f"News/sentiment analysis failed: {e}")

        signal = self._aggregator.aggregate(
            symbol=symbol,
            technical_score=tech_score,
            patterns=patterns,
            sentiment_reports=sentiment_reports,
        )
        logger.info(
            f"  ⚖️ {signal.direction.value} (conf={signal.confidence:.0%}, "
            f"composite={signal.composite_score:+.2f})"
        )
        return signal

    def scan(self) -> Dict[str, List[dict]]:
        """Run the market scanner for opportunities outside the watchlist."""
        return self._scanner.scan()

    # ====================================================================
    # SIMULATION (auto-managed paper portfolio)
    # ====================================================================
    def maybe_paper_trade(self, signal: TradingSignal) -> Optional[str]:
        """Open/close a paper position on a strong signal. Returns a log line."""
        s = self.settings
        symbol = signal.symbol
        price = get_current_prices([symbol]).get(symbol)
        if not price:
            return None

        held = self.sim.get_position(symbol)

        if signal.direction == SignalDirection.BUY and signal.confidence >= s.sim_buy_confidence:
            budget = min(s.sim_trade_budget, self.sim.portfolio.cash)
            if budget < price * 0.001:
                return None
            qty = budget / price
            if self.sim.add_position(symbol, qty, price):
                reason = f"conf={signal.confidence:.0%}, composite={signal.composite_score:+.2f}"
                self.journal.log_trade("BUY", symbol, qty, price, reason)
                return f"🟢 SIM BUY {qty:.4f} {symbol} @ {self.settings.currency_symbol}{price:.2f} ({reason})"

        elif signal.direction == SignalDirection.SELL and signal.confidence >= s.sim_sell_confidence:
            if not held:
                return None
            qty = held.quantity
            if self.sim.remove_position(symbol, qty, price):
                pnl = (price - held.average_entry_price) * qty
                reason = f"conf={signal.confidence:.0%}, P&L={self.settings.currency_symbol}{pnl:+.2f}"
                self.journal.log_trade("SELL", symbol, qty, price, reason)
                return f"🔴 SIM SELL {qty:.4f} {symbol} @ {self.settings.currency_symbol}{price:.2f} ({reason})"
        return None

    def snapshot_sim_equity(self) -> dict:
        """Mark the simulated portfolio to market and journal it."""
        held = self.sim.held_symbols()
        prices = get_current_prices(held) if held else {}
        mtm = self.sim.mark_to_market(prices)
        self.journal.log_equity(
            cash=mtm["cash"], holdings_value=mtm["holdings_value"],
            total=mtm["total"], pnl=mtm["unrealized_pnl"], detail=mtm["positions"],
        )
        return mtm

    # ====================================================================
    # REAL portfolio (manual entry — SPEC.md §3.2/§5)
    # ====================================================================
    def real_buy(self, symbol: str, qty: float, price: float) -> bool:
        ok = self.real.add_position(symbol, qty, price)
        if ok:
            self.journal.log_trade("REAL_BUY", symbol.upper(), qty, price, "manual entry")
        return ok

    def real_sell(self, symbol: str, qty: float, price: float) -> bool:
        ok = self.real.remove_position(symbol, qty, price)
        if ok:
            self.journal.log_trade("REAL_SELL", symbol.upper(), qty, price, "manual entry")
        return ok

    def live_prices(self, symbols: List[str]) -> Dict[str, float]:
        return get_current_prices(symbols) if symbols else {}

    # ====================================================================
    # Helpers
    # ====================================================================
    @staticmethod
    def format_signal(signal: TradingSignal) -> str:
        emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⏸️"}
        e = emoji.get(signal.direction.value, "❓")
        lines = [
            f"{e} {signal.symbol}: {signal.direction.value} "
            f"(confidence {signal.confidence:.0%}, composite {signal.composite_score:+.2f})",
            f"   tech={signal.technical_score:+.2f}  "
            f"pattern={signal.pattern_score:+.2f}  sentiment={signal.sentiment_score:+.2f}",
        ]
        if signal.reasoning:
            lines.append(f"   {signal.reasoning}")
        return "\n".join(lines)

    @staticmethod
    def _empty_signal(symbol: str) -> TradingSignal:
        return TradingSignal(
            symbol=symbol,
            direction=SignalDirection.HOLD,
            confidence=0.0,
            timestamp=datetime.now(),
            reasoning="Insufficient data for analysis",
        )

    async def notify(self, message: str) -> None:
        """Best-effort plain-text notification via the configured notifier."""
        if self.notifier:
            try:
                await self.notifier.send_message(message)
            except Exception as e:
                logger.warning(f"Notification failed: {e}")

    async def notify_signal(self, signal: TradingSignal) -> None:
        """Send a signal as a rich Discord embed, or formatted text otherwise."""
        if not self.notifier:
            return
        try:
            if hasattr(self.notifier, "send_embed"):
                from .reporting.discord import build_signal_embed
                price = self.live_prices([signal.symbol]).get(signal.symbol)
                embed = build_signal_embed(signal, price, self.settings.currency_symbol)
                await self.notifier.send_embed(embed)
            else:
                await self.notifier.send_message(self.format_signal(signal))
        except Exception as e:
            logger.warning(f"Signal notification failed: {e}")

    async def notify_report(self, title: str, body: str) -> None:
        """Send a report/newsletter as a Discord embed, or plain text otherwise."""
        if not self.notifier:
            return
        try:
            if hasattr(self.notifier, "send_embed"):
                from .reporting.discord import build_text_embed
                await self.notifier.send_embed(build_text_embed(title, body))
            else:
                await self.notifier.send_message(f"{title}\n\n{body}")
        except Exception as e:
            logger.warning(f"Report notification failed: {e}")
