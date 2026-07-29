"""
config.py
---------
Carregamento centralizado de variáveis de ambiente e configurações do bot CTEV.

Todas as credenciais e parâmetros sensíveis devem residir em um arquivo `.env`
na raiz do projeto (NUNCA versionado - ver .gitignore).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    # Se python-dotenv não estiver instalado em ambiente de inspeção,
    # apenas continuamos lendo direto do os.environ.
    load_dotenv = None


def _load_env() -> None:
    """Carrega o arquivo .env se existir e se python-dotenv estiver disponível."""
    if load_dotenv is not None:
        load_dotenv()


@dataclass(frozen=True)
class TelegramConfig:
    """Configurações do bot do Telegram."""
    token: str
    chat_id: str


@dataclass(frozen=True)
class BinanceConfig:
    """Configurações de acesso à Binance via ccxt."""
    api_key: Optional[str]
    api_secret: Optional[str]
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"


@dataclass(frozen=True)
class Settings:
    """Agrega todas as configurações do bot CTEV."""
    telegram: TelegramConfig
    binance: BinanceConfig
    loop_interval_seconds: int = 60  # Verifica a cada 1 minuto
    log_level: str = "INFO"


def _require_env(key: str) -> str:
    """Lê uma variável de ambiente obrigatória ou levanta erro explicativo."""
    value = os.getenv(key)
    if not value:
        raise RuntimeError(
            f"Variável de ambiente obrigatória ausente: '{key}'. "
            "Copie .env.example para .env e preencha os valores."
        )
    return value


def load_settings() -> Settings:
    """
    Carrega e valida todas as configurações a partir do ambiente.
    Levanta RuntimeError se alguma credencial essencial estiver ausente.
    """
    _load_env()

    telegram = TelegramConfig(
        token=_require_env("TELEGRAM_BOT_TOKEN"),
        chat_id=_require_env("TELEGRAM_CHAT_ID"),
    )

    binance = BinanceConfig(
        api_key=os.getenv("BINANCE_API_KEY") or None,
        api_secret=os.getenv("BINANCE_API_SECRET") or None,
        symbol=os.getenv("BINANCE_SYMBOL", "BTC/USDT"),
        timeframe=os.getenv("BINANCE_TIMEFRAME", "1h"),
    )

    return Settings(
        telegram=telegram,
        binance=binance,
        loop_interval_seconds=int(os.getenv("LOOP_INTERVAL_SECONDS", "60")),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )
