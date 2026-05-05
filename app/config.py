"""Centralized application configuration loaded from environment / .env."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    All values can be overridden via environment variables (prefixed `QT_`)
    or a `.env` file in the project root.
    """

    model_config = SettingsConfigDict(
        env_prefix="QT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: str = "dev"
    log_level: str = "INFO"

    data_dir: Path = Field(default=Path("./data_cache"))
    artifacts_dir: Path = Field(default=Path("./artifacts"))
    db_url: str = "sqlite:///./data_cache/quanttrade.sqlite3"

    exchange: str = "binance"
    default_symbol: str = "BTC/USDT"
    default_timeframe: str = "1h"

    initial_capital: float = 10_000.0
    commission_bps: float = 5.0
    slippage_bps: float = 2.0

    risk_per_trade: float = 0.01
    max_portfolio_risk: float = 0.05

    api_host: str = "0.0.0.0"  # noqa: S104 - dev binds to all
    api_port: int = 8000

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the cached singleton Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings
