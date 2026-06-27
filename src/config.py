"""
Centralized Configuration Module.

Loads settings from environment variables and a `.env` file using
pydantic-settings. Provides typed, validated access to every configuration
value the bot needs (data sources, notifications, the active trading window
for energy sobriety, and the dual-portfolio / simulation parameters).

Usage:
    from src.config import get_settings
    settings = get_settings()
"""

from functools import lru_cache
from typing import List

from pydantic import Field
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
    # Data sources
    # -------------------------------------------------------------------------
    finnhub_api_key: str = Field(
        default="",
        description="Optional Finnhub API key. yfinance is used when empty.",
    )

    # -------------------------------------------------------------------------
    # Notifications — Discord (primary) / Telegram (optional, legacy)
    # -------------------------------------------------------------------------
    discord_webhook_url: str = Field(
        default="",
        description="Discord webhook URL for alerts and the weekly newsletter",
    )
    telegram_bot_token: str = Field(default="", description="Telegram bot token")
    telegram_chat_id: str = Field(default="", description="Telegram chat ID")

    # -------------------------------------------------------------------------
    # Ollama (local AI sentiment, rule-based fallback)
    # -------------------------------------------------------------------------
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="qwen3")

    # -------------------------------------------------------------------------
    # Trading / analysis
    # -------------------------------------------------------------------------
    watchlist: str = Field(
        default="AAPL,MSFT,NVDA",
        description="Default watchlist (used to seed the persisted watchlist)",
    )
    analysis_interval: int = Field(
        default=300, description="Seconds between analysis cycles (in-window)"
    )
    data_dir: str = Field(default="data", description="Local state directory")

    # -------------------------------------------------------------------------
    # Active window (energy sobriety — SPEC.md §5b)
    # Times are local to `timezone`. Outside the window the background tasks
    # sleep until the next open.
    # -------------------------------------------------------------------------
    timezone: str = Field(default="Europe/Paris")
    active_start_hour: int = Field(default=8, ge=0, le=23)
    active_end_hour: int = Field(default=23, ge=1, le=24)
    active_weekdays: str = Field(
        default="0,1,2,3,4",
        description="Comma-separated active weekdays (Mon=0 .. Sun=6)",
    )

    # -------------------------------------------------------------------------
    # Simulation (auto-managed paper portfolio)
    # -------------------------------------------------------------------------
    sim_starting_cash: float = Field(default=10000.0)
    sim_trade_budget: float = Field(default=50.0, description="Cash per paper BUY")
    sim_buy_confidence: float = Field(default=0.25)
    sim_sell_confidence: float = Field(default=0.25)
    real_starting_cash: float = Field(default=0.0)

    # -------------------------------------------------------------------------
    # Computed properties
    # -------------------------------------------------------------------------
    @property
    def watchlist_symbols(self) -> List[str]:
        return [s.strip().upper() for s in self.watchlist.split(",") if s.strip()]

    @property
    def active_weekday_set(self) -> set:
        return {int(d) for d in self.active_weekdays.split(",") if d.strip() != ""}

    @property
    def has_finnhub(self) -> bool:
        return bool(self.finnhub_api_key) and self.finnhub_api_key != "your_finnhub_api_key_here"

    @property
    def has_discord(self) -> bool:
        return self.discord_webhook_url.startswith("https://")

    @property
    def has_telegram(self) -> bool:
        return (
            bool(self.telegram_bot_token)
            and self.telegram_bot_token != "your_telegram_bot_token_here"
            and bool(self.telegram_chat_id)
            and self.telegram_chat_id != "your_telegram_chat_id_here"
        )

    @property
    def has_ollama(self) -> bool:
        return bool(self.ollama_base_url)

    def summary(self) -> str:
        """Human-readable summary of active integrations."""
        return "\n".join([
            "╔══════════════════════════════════════╗",
            "║      TradeBot v3.0 — Configuration    ║",
            "╚══════════════════════════════════════╝",
            f"  Data:      yfinance (default){' + Finnhub key' if self.has_finnhub else ''}",
            f"  Discord:   {'✅ webhook set' if self.has_discord else '❌ not set'}",
            f"  Telegram:  {'✅ configured' if self.has_telegram else '❌ not set'}",
            f"  Ollama:    {'✅ ' + self.ollama_model + ' @ ' + self.ollama_base_url if self.has_ollama else '❌ not set'}",
            f"  Watchlist: {', '.join(self.watchlist_symbols)}",
            f"  Window:    {self.active_start_hour:02d}h–{self.active_end_hour:02d}h "
            f"{self.timezone}, days={sorted(self.active_weekday_set)}",
            f"  Interval:  {self.analysis_interval}s",
        ])


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached application settings (singleton)."""
    return Settings()
