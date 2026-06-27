"""
Telegram Notification Bot Module.

Asynchronous Telegram alert system for trading signals.
Sends formatted messages and chart images to a Telegram chat.

Features:
- Rate limiting (30 req/s max per Telegram API)
- Automatic message splitting for messages > 4096 chars
- Adaptive retry on 429 (Too Many Requests)
- Image sending support (annotated charts)
- Async-first design

Usage:
    notifier = TelegramNotifier(token="...", chat_id="...")
    await notifier.send_signal_alert(trading_signal)
"""

import asyncio
from datetime import datetime
from typing import List, Optional
import logging

import aiohttp

from ..core.models import TradingSignal, SignalDirection

logger = logging.getLogger(__name__)

TELEGRAM_MAX_MESSAGE_LENGTH = 4096


class TelegramNotifier:
    """
    Asynchronous Telegram notification engine.

    Respects Telegram API constraints:
    - Max 30 messages/second (enforced via TCPConnector limit)
    - Max 4096 characters per message (auto-split)
    - Retry on 429 with adaptive delay
    """

    def __init__(
        self,
        token: str,
        chat_id: str,
        max_concurrent: int = 30,
    ):
        self.token = token
        self.chat_id = chat_id
        self.max_concurrent = max_concurrent
        self._base_url = f"https://api.telegram.org/bot{token}"

    async def send_message(
        self,
        text: str,
        parse_mode: str = "Markdown",
        session: Optional[aiohttp.ClientSession] = None,
    ) -> bool:
        """Send a text message to Telegram, splitting if needed.

        Args:
            text: Message text.
            parse_mode: Telegram parse mode ("Markdown" or "HTML").
            session: Optional shared aiohttp session.

        Returns:
            True if all parts were sent successfully.
        """
        own_session = session is None
        if own_session:
            connector = aiohttp.TCPConnector(limit=self.max_concurrent)
            session = aiohttp.ClientSession(connector=connector)

        try:
            parts = self._split_message(text)
            success = True

            for part in parts:
                result = await self._send_single_message(session, part, parse_mode)
                if not result:
                    success = False

            return success

        finally:
            if own_session:
                await session.close()

    async def send_image(
        self,
        image_path: str,
        caption: str = "",
        session: Optional[aiohttp.ClientSession] = None,
    ) -> bool:
        """Send an image to Telegram.

        Args:
            image_path: Absolute path to the image file.
            caption: Optional caption (max 1024 chars).
            session: Optional shared aiohttp session.

        Returns:
            True if the image was sent successfully.
        """
        own_session = session is None
        if own_session:
            connector = aiohttp.TCPConnector(limit=self.max_concurrent)
            session = aiohttp.ClientSession(connector=connector)

        try:
            url = f"{self._base_url}/sendPhoto"
            data = aiohttp.FormData()
            data.add_field("chat_id", self.chat_id)
            if caption:
                data.add_field("caption", caption[:1024])
                data.add_field("parse_mode", "Markdown")

            data.add_field(
                "photo",
                open(image_path, "rb"),
                filename="chart.png",
                content_type="image/png",
            )

            async with session.post(url, data=data, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    return True
                elif resp.status == 429:
                    retry_data = await resp.json()
                    delay = retry_data.get("parameters", {}).get("retry_after", 1)
                    logger.warning(f"Telegram rate limit, retrying in {delay}s")
                    await asyncio.sleep(delay)
                    async with session.post(url, data=data) as retry_resp:
                        return retry_resp.status == 200
                else:
                    body = await resp.text()
                    logger.error(f"Telegram sendPhoto failed ({resp.status}): {body}")
                    return False

        except Exception as e:
            logger.error(f"Failed to send image to Telegram: {e}")
            return False
        finally:
            if own_session:
                await session.close()

    async def send_signal_alert(
        self,
        signal: TradingSignal,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> bool:
        """Format and send a TradingSignal as a rich Telegram alert.

        Args:
            signal: The trading signal to report.
            session: Optional shared aiohttp session.

        Returns:
            True if the alert was sent successfully.
        """
        message = self._format_signal(signal)
        return await self.send_message(message, session=session)

    async def send_batch_alerts(
        self,
        signals: List[TradingSignal],
    ) -> int:
        """Send multiple signal alerts concurrently.

        Args:
            signals: List of trading signals.

        Returns:
            Number of successfully sent alerts.
        """
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                self.send_signal_alert(signal, session=session)
                for signal in signals
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for r in results if r is True)
        logger.info(f"Sent {success_count}/{len(signals)} alerts successfully")
        return success_count

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    async def _send_single_message(
        self,
        session: aiohttp.ClientSession,
        text: str,
        parse_mode: str,
    ) -> bool:
        """Send a single message with retry on rate limit."""
        url = f"{self._base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }

        try:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    return True
                elif resp.status == 429:
                    data = await resp.json()
                    delay = data.get("parameters", {}).get("retry_after", 1)
                    logger.warning(f"Telegram rate limit, retrying in {delay}s")
                    await asyncio.sleep(delay)
                    async with session.post(url, json=payload) as retry_resp:
                        return retry_resp.status == 200
                else:
                    body = await resp.text()
                    logger.error(f"Telegram sendMessage failed ({resp.status}): {body}")
                    return False

        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    @staticmethod
    def _split_message(text: str) -> List[str]:
        """Split a message into chunks of max TELEGRAM_MAX_MESSAGE_LENGTH."""
        if len(text) <= TELEGRAM_MAX_MESSAGE_LENGTH:
            return [text]

        parts = []
        while text:
            if len(text) <= TELEGRAM_MAX_MESSAGE_LENGTH:
                parts.append(text)
                break

            # Try to split at a newline
            split_pos = text.rfind("\n", 0, TELEGRAM_MAX_MESSAGE_LENGTH)
            if split_pos == -1:
                split_pos = TELEGRAM_MAX_MESSAGE_LENGTH

            parts.append(text[:split_pos])
            text = text[split_pos:].lstrip("\n")

        return parts

    @staticmethod
    def _format_signal(signal: TradingSignal) -> str:
        """Format a TradingSignal into a rich Telegram message."""
        # Direction emoji
        emoji_map = {
            SignalDirection.BUY: "🟢 BUY",
            SignalDirection.SELL: "🔴 SELL",
            SignalDirection.HOLD: "⏸️ HOLD",
        }
        direction_str = emoji_map.get(signal.direction, str(signal.direction.value))

        # Confidence bar
        conf_bars = int(signal.confidence * 10)
        conf_visual = "█" * conf_bars + "░" * (10 - conf_bars)

        lines = [
            f"📊 *TradeBot Signal — {signal.symbol}*",
            f"",
            f"*Direction:* {direction_str}",
            f"*Confidence:* `[{conf_visual}]` {signal.confidence:.0%}",
            f"",
            f"*Component Scores:*",
            f"  📈 Technical: `{signal.technical_score:+.2f}`",
            f"  📐 Patterns:  `{signal.pattern_score:+.2f}`",
            f"  📰 Sentiment: `{signal.sentiment_score:+.2f}`",
            f"  ⚖️ Composite: `{signal.composite_score:+.2f}`",
        ]

        if signal.patterns_detected:
            lines.append(f"")
            lines.append(f"*Patterns Detected:*")
            for p in signal.patterns_detected[:5]:
                lines.append(f"  • {p.pattern_name} ({p.confidence:.0%}, {p.detection_method})")

        if signal.reasoning:
            lines.append(f"")
            lines.append(f"*Reasoning:*")
            lines.append(f"_{signal.reasoning}_")

        lines.append(f"")
        lines.append(f"🕐 {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Interactive Bot (Polling)
    # -------------------------------------------------------------------------

    async def start_polling(self, portfolio_manager=None, runner=None):
        """Start long-polling for incoming messages/commands."""
        logger.info("Starting Telegram interactive polling...")
        offset = 0
        connector = aiohttp.TCPConnector(limit=10)
        async with aiohttp.ClientSession(connector=connector) as session:
            while True:
                try:
                    url = f"{self._base_url}/getUpdates"
                    params = {"offset": offset, "timeout": 30}
                    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=40)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data.get("ok"):
                                for update in data.get("result", []):
                                    offset = update["update_id"] + 1
                                    await self._handle_update(update, session, portfolio_manager, runner)
                        else:
                            await asyncio.sleep(5)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Telegram polling error: {e}")
                    await asyncio.sleep(5)

    async def _handle_update(self, update: dict, session: aiohttp.ClientSession, portfolio_manager, runner):
        """Process incoming Telegram updates."""
        message = update.get("message")
        if not message:
            return

        chat_id = str(message.get("chat", {}).get("id"))
        if chat_id != self.chat_id:
            return  # Ignore messages from unauthorized chats

        text = message.get("text", "").strip()
        if not text.startswith("/"):
            return

        parts = text.split()
        command = parts[0].lower()
        args = parts[1:]

        try:
            if command == "/portfolio":
                if portfolio_manager:
                    reply = portfolio_manager.get_summary()
                else:
                    reply = "Portfolio manager not configured."
                await self.send_message(reply, session=session)

            elif command == "/buy" and len(args) == 3:
                # /buy AAPL 10 150.5
                symbol, qty, price = args[0], float(args[1]), float(args[2])
                if portfolio_manager and portfolio_manager.add_position(symbol, qty, price):
                    await self.send_message(f"✅ Bought {qty} of {symbol.upper()} at ${price}", session=session)
                else:
                    await self.send_message("❌ Failed to add position (check funds/args)", session=session)

            elif command == "/sell" and len(args) == 3:
                # /sell AAPL 5 160.0
                symbol, qty, price = args[0], float(args[1]), float(args[2])
                if portfolio_manager and portfolio_manager.remove_position(symbol, qty, price):
                    await self.send_message(f"✅ Sold {qty} of {symbol.upper()} at ${price}", session=session)
                else:
                    await self.send_message("❌ Failed to remove position (check qty/args)", session=session)

            elif command == "/scan":
                await self.send_message("🔍 Starting market scan... This may take a minute.", session=session)
                if runner and hasattr(runner, 'run_scan'):
                    # Trigger scan asynchronously so we don't block polling
                    asyncio.create_task(runner.run_scan(manual=True))
                else:
                    await self.send_message("Scanner not configured.", session=session)

            elif command == "/status":
                await self.send_message("🤖 TradeBot v3 is running smoothly in the background.", session=session)

            elif command == "/help":
                help_text = (
                    "*TradeBot Commands*\n"
                    "/portfolio - View portfolio\n"
                    "/buy SYMBOL QTY PRICE - Record a simulated buy\n"
                    "/sell SYMBOL QTY PRICE - Record a simulated sell\n"
                    "/scan - Trigger a market scan for anomalies\n"
                    "/status - Check bot status"
                )
                await self.send_message(help_text, session=session)

            else:
                await self.send_message("Unknown command or invalid arguments. Try /help", session=session)

        except ValueError:
            await self.send_message("❌ Invalid numbers provided. Use format: /buy AAPL 10 150.5", session=session)
        except Exception as e:
            logger.error(f"Error handling command {command}: {e}")
            await self.send_message(f"❌ Error executing command: {e}", session=session)
