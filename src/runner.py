"""
TradeBot Runner — Asynchronous Orchestrator.

Main event loop that coordinates the full analysis pipeline:
1. Fetch prices (Finnhub REST/WS)
2. Run technical analysis + pattern detection
3. Fetch news + run sentiment analysis (Ollama)
4. Aggregate signals
5. Send Telegram alerts

Modes:
- monitor: Continuous real-time surveillance
- analyze: One-shot analysis of a single ticker

Usage:
    runner = TradeBotRunner()
    await runner.run_monitor()
    # or
    signal = await runner.analyze_symbol("AAPL")
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional
import logging

import pandas as pd

from .config import get_settings
from .core.models import TradingSignal, SignalDirection
from .data.finnhub_source import FinnhubRESTSource
from .data.news_fetcher import NewsFetcher
from .analysis.technical import AdvancedTechnicalAnalyzer
from .analysis.pattern_detector import PatternDetector
from .analysis.sentiment import OllamaSentimentAnalyzer
from .analysis.signal_aggregator import SignalAggregator
from .analysis.scanner import MarketScanner
from .portfolio.manager import PortfolioManager
from .portfolio.journal import TradeJournal
from .data.quotes import get_current_prices
from .reporting.telegram_bot import TelegramNotifier

logger = logging.getLogger(__name__)


class TradeBotRunner:
    """
    Asynchronous orchestrator for the full trading analysis pipeline.

    Designed to run continuously in monitor mode or as a one-shot analyzer.
    All components are initialized lazily and gracefully handle missing
    configuration (e.g., no Telegram token → skip notifications).
    """

    def __init__(self):
        self.settings = get_settings()

        # Analysis components
        self._tech_analyzer = AdvancedTechnicalAnalyzer()
        self._pattern_detector = PatternDetector(use_vision=False)  # Math-only for speed
        self._sentiment_analyzer = OllamaSentimentAnalyzer(
            model=self.settings.ollama_model,
            base_url=self.settings.ollama_base_url,
        )
        self._signal_aggregator = SignalAggregator()
        self._portfolio_manager = PortfolioManager()
        self._scanner = MarketScanner()
        self._journal = TradeJournal()

        # Data components — news always available (yfinance fallback when no Finnhub key)
        self._news_fetcher: NewsFetcher = NewsFetcher(
            api_key=self.settings.finnhub_api_key if self.settings.has_finnhub else ""
        )

        # Notification
        self._notifier: Optional[TelegramNotifier] = None
        if self.settings.has_telegram:
            self._notifier = TelegramNotifier(
                token=self.settings.telegram_bot_token,
                chat_id=self.settings.telegram_chat_id,
            )

        self._running = False

    async def analyze_symbol(self, symbol: str, days: int = 90) -> TradingSignal:
        """Run a full one-shot analysis on a single symbol.

        Args:
            symbol: Ticker symbol (e.g., "AAPL").
            days: Number of days of historical data to fetch.

        Returns:
            A consolidated TradingSignal.
        """
        logger.info(f"📊 Analyzing {symbol}...")

        # 1. Fetch price data (Finnhub if a key is set, otherwise yfinance fallback)
        try:
            source = FinnhubRESTSource(
                symbol=symbol,
                api_key=self.settings.finnhub_api_key,
                resolution="D",
                days=days,
            )
            prices = source.fetch_prices()
        except Exception as e:
            logger.error(f"Failed to fetch price data for {symbol}: {e}")
            return self._empty_signal(symbol)

        if not prices:
            logger.warning(f"No price data for {symbol}")
            return self._empty_signal(symbol)

        # Build DataFrame
        df = pd.DataFrame({
            "close": [p.asset_a for p in prices],
            "timestamp": [p.timestamp for p in prices],
        })

        # 2. Technical analysis
        try:
            enriched_df = self._tech_analyzer.compute_indicators(df)
            tech_score = self._tech_analyzer.get_signal_score(enriched_df)
            logger.info(f"  📈 Technical score: {tech_score:+.2f}")
        except Exception as e:
            logger.error(f"Technical analysis failed: {e}")
            tech_score = 0.0

        # 3. Pattern detection
        try:
            patterns = self._pattern_detector.detect(df, symbol)
            logger.info(f"  📐 Patterns found: {len(patterns)}")
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            patterns = []

        # 4. News sentiment
        sentiment_reports = []
        if self._news_fetcher:
            try:
                articles = await self._news_fetcher.fetch_company_news(symbol, days=7)
                logger.info(f"  📰 News articles fetched: {len(articles)}")

                for article in articles[:5]:  # Limit to 5 most recent
                    text = f"{article.title}\n\n{article.summary}"
                    pos = self._portfolio_manager.get_position(symbol)
                    report = self._sentiment_analyzer.analyze(
                        text, source_url=article.url, current_position=pos
                    )
                    sentiment_reports.append(report)

                if sentiment_reports:
                    avg = sum(r.sentiment_polarity for r in sentiment_reports) / len(sentiment_reports)
                    logger.info(f"  🧠 Avg sentiment: {avg:+.2f} ({len(sentiment_reports)} articles)")
            except Exception as e:
                logger.error(f"News/sentiment analysis failed: {e}")

        # 5. Aggregate
        signal = self._signal_aggregator.aggregate(
            symbol=symbol,
            technical_score=tech_score,
            patterns=patterns,
            sentiment_reports=sentiment_reports,
        )

        logger.info(
            f"  ⚖️ Signal: {signal.direction.value} "
            f"(confidence={signal.confidence:.0%}, composite={signal.composite_score:+.2f})"
        )

        return signal

    async def run_monitor(self) -> None:
        """Run continuous monitoring loop for all watchlist symbols.

        Analyzes each symbol at the configured interval and sends
        Telegram alerts for actionable signals.
        """
        symbols = self.settings.watchlist_symbols
        interval = self.settings.analysis_interval

        print(self.settings.summary())
        print(f"\n🔄 Monitoring {', '.join(symbols)} every {interval}s")
        print("Press Ctrl+C to stop\n")

        self._running = True

        # Start interactive Telegram bot if enabled
        if self._notifier:
            asyncio.create_task(self._notifier.start_polling(self._portfolio_manager, self))

        while self._running:
            cycle_start = datetime.now()

            for symbol in symbols:
                try:
                    signal = await self.analyze_symbol(symbol)

                    # Send alert for actionable signals (BUY or SELL)
                    if signal.direction != SignalDirection.HOLD and signal.confidence > 0.15:
                        await self._send_alert(signal)

                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")

            # Wait for next cycle
            elapsed = (datetime.now() - cycle_start).total_seconds()
            wait_time = max(0, interval - elapsed)

            if wait_time > 0:
                logger.debug(f"Waiting {wait_time:.0f}s until next analysis cycle")
                try:
                    await asyncio.sleep(wait_time)
                except asyncio.CancelledError:
                    break

    async def run_simulation(
        self,
        auto_trade: bool = True,
        trade_budget: float = 50.0,
        buy_confidence: float = 0.25,
        sell_confidence: float = 0.25,
        cycles: Optional[int] = None,
    ) -> None:
        """Run a continuous paper-trading simulation and log everything.

        Each cycle: analyze watchlist ∪ held symbols, optionally execute paper
        trades on strong signals, mark the portfolio to market, and append a
        full record to data/journal.jsonl.

        Args:
            auto_trade: If True, the bot opens/closes paper positions itself.
            trade_budget: Cash (in account currency) committed per BUY.
            buy_confidence: Min confidence to act on a BUY signal.
            sell_confidence: Min confidence to act on a SELL signal.
            cycles: Stop after N cycles (None = run forever).
        """
        interval = self.settings.analysis_interval
        self._running = True
        cycle = 0

        print(self.settings.summary())
        print(f"\n🧪 Simulation — auto_trade={auto_trade}, budget=${trade_budget:.0f}/trade, "
              f"every {interval}s. Journal: data/journal.jsonl")
        print("Press Ctrl+C to stop\n")

        if self._notifier:
            asyncio.create_task(self._notifier.start_polling(self._portfolio_manager, self))

        while self._running:
            cycle += 1
            cycle_start = datetime.now()

            # Analyze the watchlist plus anything we currently hold
            symbols = list(dict.fromkeys(
                self.settings.watchlist_symbols + self._portfolio_manager.held_symbols()
            ))

            for symbol in symbols:
                try:
                    signal = await self.analyze_symbol(symbol)
                    self._journal.log_signal(signal)
                    if auto_trade:
                        self._maybe_trade(signal, trade_budget, buy_confidence, sell_confidence)
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")

            # Mark to market and snapshot equity
            self._snapshot_equity(symbols)

            if cycles is not None and cycle >= cycles:
                break

            elapsed = (datetime.now() - cycle_start).total_seconds()
            wait_time = max(0, interval - elapsed)
            if wait_time > 0:
                try:
                    await asyncio.sleep(wait_time)
                except asyncio.CancelledError:
                    break

    def _maybe_trade(self, signal: TradingSignal, trade_budget: float,
                     buy_confidence: float, sell_confidence: float) -> None:
        """Execute a paper trade if the signal is strong enough."""
        symbol = signal.symbol
        price = get_current_prices([symbol]).get(symbol)
        if not price:
            return

        pm = self._portfolio_manager
        held = pm.get_position(symbol)

        if signal.direction == SignalDirection.BUY and signal.confidence >= buy_confidence:
            budget = min(trade_budget, pm.portfolio.cash)
            if budget < price * 0.001:  # not enough cash for even a sliver
                logger.info(f"Skip BUY {symbol}: insufficient cash (${pm.portfolio.cash:.2f})")
                return
            qty = budget / price
            if pm.add_position(symbol, qty, price):
                reason = f"conf={signal.confidence:.0%}, composite={signal.composite_score:+.2f}"
                self._journal.log_trade("BUY", symbol, qty, price, reason)
                logger.info(f"🟢 PAPER BUY {qty:.4f} {symbol} @ ${price:.2f} ({reason})")

        elif signal.direction == SignalDirection.SELL and signal.confidence >= sell_confidence:
            if not held:
                return  # nothing to sell; no shorting in this simulation
            qty = held.quantity
            if pm.remove_position(symbol, qty, price):
                pnl = (price - held.average_entry_price) * qty
                reason = f"conf={signal.confidence:.0%}, P&L=${pnl:+.2f}"
                self._journal.log_trade("SELL", symbol, qty, price, reason)
                logger.info(f"🔴 PAPER SELL {qty:.4f} {symbol} @ ${price:.2f} ({reason})")

    def _snapshot_equity(self, symbols: List[str]) -> None:
        """Mark portfolio to market, log it, and print a one-line summary."""
        pm = self._portfolio_manager
        held = pm.held_symbols()
        prices = get_current_prices(held) if held else {}
        mtm = pm.mark_to_market(prices)
        self._journal.log_equity(
            cash=mtm["cash"],
            holdings_value=mtm["holdings_value"],
            total=mtm["total"],
            pnl=mtm["unrealized_pnl"],
            detail=mtm["positions"],
        )
        print(
            f"\n💰 Equity: ${mtm['total']:,.2f}  "
            f"(cash ${mtm['cash']:,.2f} + holdings ${mtm['holdings_value']:,.2f})  "
            f"P&L ${mtm['unrealized_pnl']:+,.2f}"
        )
        for sym, p in mtm["positions"].items():
            print(f"   {sym}: {p['quantity']} @ ${p['entry']:.2f} → ${p['price']:.2f} "
                  f"({p['pnl']:+.2f}, {p['pnl_pct']:+.1f}%)")

    def stop(self) -> None:
        """Stop the monitoring loop."""
        self._running = False

    async def _send_alert(self, signal: TradingSignal) -> None:
        """Send a trading signal alert via Telegram."""
        if self._notifier:
            try:
                success = await self._notifier.send_signal_alert(signal)
                if success:
                    logger.info(f"📱 Alert sent for {signal.symbol}: {signal.direction.value}")
                else:
                    logger.warning(f"Failed to send alert for {signal.symbol}")
            except Exception as e:
                logger.error(f"Telegram notification error: {e}")
        else:
            # Print to console if no Telegram
            self._print_signal(signal)

    @staticmethod
    def _print_signal(signal: TradingSignal) -> None:
        """Print signal to console (fallback when Telegram not configured)."""
        emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⏸️"}
        e = emoji.get(signal.direction.value, "❓")
        print(
            f"\n{e} {signal.symbol}: {signal.direction.value} "
            f"(confidence={signal.confidence:.0%})"
        )
        print(f"   Tech={signal.technical_score:+.2f}  "
              f"Pattern={signal.pattern_score:+.2f}  "
              f"Sentiment={signal.sentiment_score:+.2f}")
        if signal.reasoning:
            print(f"   {signal.reasoning}")

    @staticmethod
    def _empty_signal(symbol: str) -> TradingSignal:
        """Return an empty HOLD signal."""
        return TradingSignal(
            symbol=symbol,
            direction=SignalDirection.HOLD,
            confidence=0.0,
            timestamp=datetime.now(),
            reasoning="Insufficient data for analysis",
        )

    async def run_scan(self, manual: bool = False) -> None:
        """Run the market scanner and send results."""
        results = self._scanner.scan()
        if not self._notifier:
            logger.info("Scan complete, no notifier configured.")
            return

        lines = ["*Market Scan Results* 🔍\n"]
        
        anomalies = results.get("anomalies", [])
        if anomalies:
            lines.append("*Top Volatility Anomalies:*")
            for a in anomalies[:3]:
                lines.append(f"• {a['symbol']}: {a['type']} (Z: {a['z_score']:+.2f}, Return: {a['recent_return']:.1%}) @ ${a['close_price']:.2f}")
            lines.append("")

        steady = results.get("steady_growth", [])
        if steady:
            lines.append("*Top Steady Growth (90d):*")
            for s in steady[:3]:
                lines.append(f"• {s['symbol']}: +{s['total_growth_pct']:.1%} (R²: {s['r_squared']:.2f}) @ ${s['close_price']:.2f}")

        if not anomalies and not steady:
            lines.append("No actionable opportunities found.")

        msg = "\n".join(lines)
        await self._notifier.send_message(msg)
