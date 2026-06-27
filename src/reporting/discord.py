"""
Discord notifications via webhook (SPEC.md §7).

Primary notification channel: a webhook URL pointing at a channel in my
personal/family Discord server. No bot token, no OAuth — just POST JSON.
Used for real-time signal alerts and the weekly newsletter.

Gracefully degrades: if no webhook is configured, sends are no-ops.
"""

import logging
from datetime import datetime
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

# Discord hard-limits a single message to 2000 characters.
_MAX_LEN = 1990

# Embed colors (decimal RGB).
_COLOR_BUY = 3066993    # green
_COLOR_SELL = 15158332  # red
_COLOR_HOLD = 9807270   # grey
_COLOR_INFO = 3447003   # blue


class DiscordNotifier:
    """Send messages to a Discord channel through an incoming webhook."""

    def __init__(self, webhook_url: str, username: str = "TradeBot"):
        self.webhook_url = webhook_url
        self.username = username

    @property
    def enabled(self) -> bool:
        return self.webhook_url.startswith("https://")

    async def send_embed(self, embed: dict) -> bool:
        """Post a single rich embed."""
        return await self._post({"username": self.username, "embeds": [embed]})

    async def send_message(self, content: str) -> bool:
        """Post a message. Long messages are split into chunks."""
        if not self.enabled:
            logger.debug("Discord webhook not configured — skipping send.")
            return False
        ok = True
        for chunk in self._split(content):
            ok = await self._post({"username": self.username, "content": chunk}) and ok
        return ok

    async def _post(self, payload: dict) -> bool:
        """POST a payload to the webhook. Returns True on success."""
        if not self.enabled:
            logger.debug("Discord webhook not configured — skipping send.")
            return False
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url, json=payload,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status not in (200, 204):
                        body = await resp.text()
                        logger.warning(f"Discord webhook returned {resp.status}: {body}")
                        return False
        except Exception as e:
            logger.warning(f"Discord send failed: {e}")
            return False
        return True

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


# =============================================================================
# EMBED BUILDERS
# =============================================================================

def _footer(user: Optional[str]) -> dict:
    who = f" · {user}" if user else ""
    return {"text": f"TradeBot{who} • {datetime.now().strftime('%d/%m/%Y %H:%M')}"}


def build_signal_embed(signal, price: Optional[float] = None, currency: str = "€",
                       user: Optional[str] = None, company: Optional[str] = None,
                       url: Optional[str] = None) -> dict:
    """Build a rich Discord embed for a trading signal."""
    direction = signal.direction.value
    emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⏸️"}.get(direction, "❓")
    action = {"BUY": "ACHAT", "SELL": "VENTE", "HOLD": "CONSERVER"}.get(direction, direction)
    color = {"BUY": _COLOR_BUY, "SELL": _COLOR_SELL}.get(direction, _COLOR_HOLD)
    # "AAPL — Apple Inc." if we know the name, else just the ticker.
    label = f"{signal.symbol} — {company}" if company and company != signal.symbol else signal.symbol

    fields = [
        {"name": "🎯 Confiance", "value": f"{signal.confidence:.0%}", "inline": True},
        {"name": "⚖️ Score", "value": f"{signal.composite_score:+.2f}", "inline": True},
    ]
    if price is not None:
        fields.append({"name": "💰 Prix", "value": f"{currency}{price:,.2f}", "inline": True})
    fields += [
        {"name": "📈 Technique", "value": f"{signal.technical_score:+.2f}", "inline": True},
        {"name": "📐 Figures", "value": f"{signal.pattern_score:+.2f}", "inline": True},
        {"name": "🧠 Sentiment", "value": f"{signal.sentiment_score:+.2f}", "inline": True},
    ]

    embed = {
        "title": f"{emoji} Signal {action} — {label}",
        "color": color,
        "fields": fields,
        "footer": _footer(user),
        "timestamp": datetime.now().astimezone().isoformat(),
    }
    if url:
        embed["url"] = url  # makes the title a clickable link
    if signal.reasoning:
        embed["description"] = signal.reasoning[:4000]
    return embed


def build_text_embed(title: str, body: str, color: int = _COLOR_INFO,
                     user: Optional[str] = None) -> dict:
    """Build a simple embed with a markdown body (e.g. report, newsletter)."""
    return {
        "title": title,
        "description": body[:4000],
        "color": color,
        "footer": _footer(user),
    }
