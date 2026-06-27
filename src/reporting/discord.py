"""
Discord notifications via webhook (SPEC.md §7).

Primary notification channel: a webhook URL pointing at a channel in my
personal/family Discord server. No bot token, no OAuth — just POST JSON.
Used for real-time signal alerts and the weekly newsletter.

Gracefully degrades: if no webhook is configured, sends are no-ops.
"""

import logging
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

# Discord hard-limits a single message to 2000 characters.
_MAX_LEN = 1990


class DiscordNotifier:
    """Send messages to a Discord channel through an incoming webhook."""

    def __init__(self, webhook_url: str, username: str = "TradeBot"):
        self.webhook_url = webhook_url
        self.username = username

    @property
    def enabled(self) -> bool:
        return self.webhook_url.startswith("https://")

    async def send_message(self, content: str) -> bool:
        """Post a message. Long messages are split into chunks."""
        if not self.enabled:
            logger.debug("Discord webhook not configured — skipping send.")
            return False

        chunks = self._split(content)
        ok = True
        try:
            async with aiohttp.ClientSession() as session:
                for chunk in chunks:
                    payload = {"username": self.username, "content": chunk}
                    async with session.post(
                        self.webhook_url, json=payload,
                        timeout=aiohttp.ClientTimeout(total=15),
                    ) as resp:
                        if resp.status not in (200, 204):
                            body = await resp.text()
                            logger.warning(f"Discord webhook returned {resp.status}: {body}")
                            ok = False
        except Exception as e:
            logger.warning(f"Discord send failed: {e}")
            return False
        return ok

    @staticmethod
    def _split(content: str) -> list:
        """Split content into Discord-sized chunks on line boundaries."""
        if len(content) <= _MAX_LEN:
            return [content]
        chunks, current = [], ""
        for line in content.splitlines(keepends=True):
            if len(current) + len(line) > _MAX_LEN:
                chunks.append(current)
                current = ""
            current += line
        if current:
            chunks.append(current)
        return chunks
