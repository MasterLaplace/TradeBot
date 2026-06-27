"""
Centralized Configuration Module.

Loads settings from environment variables and .env file using pydantic-settings.
Provides typed, validated access to all configuration values.

Usage:
    from src.config import get_settings

    settings = get_settings()
    print(settings.finnhub_api_key)
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Finnhub
    # -------------------------------------------------------------------------
    finnhub_api_key: str = Field(
        default="",
        description="Finnhub API key for market data and news",
    )

    # -------------------------------------------------------------------------
    # Telegram
    # -------------------------------------------------------------------------
    telegram_bot_token: str = Field(
        default="",
        description="Telegram bot token from @BotFather",
    )
    telegram_chat_id: str = Field(
        default="",
        description="Telegram chat ID for notifications",
    )

    # -------------------------------------------------------------------------
    # Ollama (Local AI)
    # -------------------------------------------------------------------------
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL",
    )
    ollama_model: str = Field(
        default="qwen3",
        description="Ollama model name for sentiment analysis",
    )

    # -------------------------------------------------------------------------
    # Trading
    # -------------------------------------------------------------------------
    watchlist: str = Field(
        default="AAPL,MSFT,NVDA",
        description="Comma-separated list of symbols to monitor",
    )
    analysis_interval: int = Field(
        default=300,
        description="Seconds between analysis cycles",
    )

    # -------------------------------------------------------------------------
    # Computed properties
    # -------------------------------------------------------------------------
    @property
    def watchlist_symbols(self) -> List[str]:
        """Parse watchlist string into a list of symbols."""
        return [s.strip().upper() for s in self.watchlist.split(",") if s.strip()]

    @property
    def has_finnhub(self) -> bool:
        """Check if Finnhub API key is configured."""
        return bool(self.finnhub_api_key) and self.finnhub_api_key != "your_finnhub_api_key_here"

    @property
    def has_telegram(self) -> bool:
        """Check if Telegram notifications are configured."""
        return (
            bool(self.telegram_bot_token)
            and self.telegram_bot_token != "your_telegram_bot_token_here"
            and bool(self.telegram_chat_id)
            and self.telegram_chat_id != "your_telegram_chat_id_here"
        )

    @property
    def has_ollama(self) -> bool:
        """Check if Ollama is configured (assumes local, always available if URL set)."""
        return bool(self.ollama_base_url)

    def summary(self) -> str:
        """Return a human-readable summary of active integrations."""
        lines = [
            "╔══════════════════════════════════════╗",
            "║      TradeBot v3.0 — Configuration    ║",
            "╚══════════════════════════════════════╝",
            f"  Finnhub API:   {'✅ configured' if self.has_finnhub else '❌ not set'}",
            f"  Telegram:      {'✅ configured' if self.has_telegram else '❌ not set'}",
            f"  Ollama:        {'✅ ' + self.ollama_model + ' @ ' + self.ollama_base_url if self.has_ollama else '❌ not set'}",
            f"  Watchlist:     {', '.join(self.watchlist_symbols)}",
            f"  Interval:      {self.analysis_interval}s",
        ]
        return "\n".join(lines)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached application settings (singleton)."""
    return Settings()
