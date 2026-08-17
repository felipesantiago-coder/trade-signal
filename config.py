"""
config.py
---------
Carregamento centralizado de variáveis de ambiente e configuracoes do bot CTEV.

Todas as credenciais e parametros sensíveis devem residir em um arquivo `.env`
na raiz do projeto (NUNCA versionado - ver .gitignore).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("ctev.config")

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


def _load_env() -> None:
    """Carrega o arquivo .env se existir e se python-dotenv estiver disponível."""
    if load_dotenv is not None:
        load_dotenv()


@dataclass(frozen=True)
class TelegramConfig:
    """Configuracoes do bot do Telegram."""
    token: str
    chat_id: str


SUPPORTED_EXCHANGES = (
    "coinbase", "kraken",
    "binance", "bybit", "kucoin", "okx", "gate", "bitget",
)


@dataclass(frozen=True)
class BinanceConfig:
    """Configuracoes de acesso a exchange via ccxt.

    exchange_id: ID da exchange ccxt.
        Coinbase funciona dos EUA (sem geo-bloqueio) com dados de 730+ dias.
        Kraken funciona dos EUA mas com dados limitados (~30 dias).
        Binance/Bybit podem ser geo-blocados em servidores US/BR.
    """
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    exchange_id: str = "coinbase"
    symbol: str = "BTC/USD"
    timeframe: str = "1h"


@dataclass(frozen=True)
class RiskConfig:
    """Configuracoes de gerenciamento de risco.

    Valores default = V13-ROBUSTA (validada via WFO 17 janelas,
    Sharpe 1.30, MaxDD 33.6%, Consistency 65%).
    """
    max_daily_loss_pct: float = 5.0       # Max drawdown diario (%)
    max_weekly_loss_pct: float = 10.0     # Max drawdown semanal (%)
    max_consecutive_losses: int = 5       # Max perdas consecutivas antes de pausa
    circuit_breaker_pct: float = 3.0      # Movimento % em 1 candle que aciona circuit breaker
    cooldown_candles: int = 3             # Minimo de candles entre sinais
    cooldown_hours: int = 12              # Horas de pausa apos consecutive losses
    atr_pct_min: float = 0.08            # V13: filtro ATR amplo (0.08-0.92)
    atr_pct_max: float = 0.92


@dataclass(frozen=True)
class PositionConfig:
    """Configuracoes de position sizing e gestao de posicoes.

    Valores default = V13-ROBUSTA (validada via WFO 17 janelas).
    Risco 0.5% por trade (metade do padrao) = menor MaxDD.
    Max 3 posicoes simultaneas = diversificacao.
    """
    account_balance: float = 10000.0      # Saldo da conta em USD
    risk_per_trade_pct: float = 0.005    # V13: 0.5% do balance por trade (half-risk)
    min_position_usd: float = 10.0       # Tamanho minimo em USD
    max_position_pct: float = 0.10       # Max 10% do balance em 1 trade
    be_trigger_atr_mult: float = 1.0      # ATR mult para ativar break-even
    trailing_atr_mult: float = 0.6        # V13: 0.6x ATR trailing
    partial_tp_pct: float = 0.50          # V13: 50% no TP1
    post_tp1_sl_buffer: float = 0.1      # v15: 0.1 ATR buffer (otimizado de 0.2)
    max_open_positions: int = 3           # V13: 3 posicoes simultaneas (diversificacao)


@dataclass(frozen=True)
class ExchangeConfig:
    """Configuracoes de execucao de ordens na exchange."""
    dry_run: bool = True                  # True = simulacao (padrao seguro)
    order_type: str = "market"           # "market" ou "limit"
    slippage_pct: float = 0.05           # 0.05% slippage estimado


@dataclass(frozen=True)
class MultiTFConfig:
    """Configuracoes do filtro multi-timeframe (H4/D1)."""
    enabled: bool = True                  # Filtro MTF ativo
    cache_ttl_seconds: int = 900          # 15 min cache (H4 = 4h candle)


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuracoes do otimizador de parametros."""
    max_combos: int = 500                 # Max combinacoes para avaliar
    default_days: int = 730               # Dias de dados para otimizacao


@dataclass(frozen=True)
class Settings:
    """Agrega todas as configuracoes do bot CTEV."""
    telegram: TelegramConfig
    binance: BinanceConfig
    risk: RiskConfig
    position: PositionConfig
    exchange: ExchangeConfig
    multitf: MultiTFConfig
    optimizer: OptimizerConfig
    active_version: str                   # ID da versao ativa (ex: V13)
    loop_interval_seconds: int = 60
    log_level: str = "INFO"


def _require_env(key: str) -> str:
    """Le uma variavel de ambiente obrigatoria ou levanta erro explicativo."""
    value = os.getenv(key)
    if not value:
        raise RuntimeError(
            f"Variavel de ambiente obrigatoria ausente: '{key}'. "
            "Copie .env.example para .env e preencha os valores."
        )
    return value


def load_settings() -> Settings:
    """
    Carrega e valida todas as configuracoes a partir do ambiente.
    Levanta RuntimeError se alguma credencial essencial estiver ausente.
    """
    _load_env()

    telegram = TelegramConfig(
        token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
    )

    exchange_id = os.getenv("EXCHANGE_ID", "coinbase").lower()
    if exchange_id not in SUPPORTED_EXCHANGES:
        logger.warning(
            "EXCHANGE_ID='%s' nao suportado. Opcoes: %s. Usando 'coinbase'.",
            exchange_id, SUPPORTED_EXCHANGES,
        )
        exchange_id = "coinbase"

    binance = BinanceConfig(
        exchange_id=exchange_id,
        api_key=os.getenv("EXCHANGE_API_KEY") or os.getenv("BINANCE_API_KEY") or None,
        api_secret=os.getenv("EXCHANGE_API_SECRET") or os.getenv("BINANCE_API_SECRET") or None,
        symbol=os.getenv("EXCHANGE_SYMBOL") or os.getenv("BINANCE_SYMBOL", "BTC/USD"),
        timeframe=os.getenv("EXCHANGE_TIMEFRAME") or os.getenv("BINANCE_TIMEFRAME", "1h"),
    )

    risk = RiskConfig(
        max_daily_loss_pct=float(os.getenv("MAX_DAILY_LOSS_PCT", "5.0")),
        max_weekly_loss_pct=float(os.getenv("MAX_WEEKLY_LOSS_PCT", "10.0")),
        max_consecutive_losses=int(os.getenv("MAX_CONSECUTIVE_LOSSES", "5")),
        circuit_breaker_pct=float(os.getenv("CIRCUIT_BREAKER_PCT", "3.0")),
        cooldown_candles=int(os.getenv("COOLDOWN_CANDLES", "3")),
        cooldown_hours=int(os.getenv("COOLDOWN_HOURS", "12")),
        atr_pct_min=float(os.getenv("ATR_PCT_MIN", "0.08")),
        atr_pct_max=float(os.getenv("ATR_PCT_MAX", "0.92")),
    )

    position = PositionConfig(
        account_balance=float(os.getenv("ACCOUNT_BALANCE", "10000.0")),
        risk_per_trade_pct=float(os.getenv("RISK_PER_TRADE_PCT", "0.005")),
        min_position_usd=float(os.getenv("MIN_POSITION_USD", "10.0")),
        max_position_pct=float(os.getenv("MAX_POSITION_PCT", "0.10")),
        be_trigger_atr_mult=float(os.getenv("BE_TRIGGER_ATR_MULT", "1.0")),
        trailing_atr_mult=float(os.getenv("TRAILING_ATR_MULT", "0.6")),
        partial_tp_pct=float(os.getenv("PARTIAL_TP_PCT", "0.50")),
        post_tp1_sl_buffer=float(os.getenv("POST_TP1_SL_BUFFER", "0.1")),
        max_open_positions=int(os.getenv("MAX_OPEN_POSITIONS", "3")),
    )

    exchange = ExchangeConfig(
        dry_run=os.getenv("EXCHANGE_DRY_RUN", "true").lower() in ("true", "1", "yes"),
        order_type=os.getenv("EXCHANGE_ORDER_TYPE", "market"),
        slippage_pct=float(os.getenv("EXCHANGE_SLIPPAGE_PCT", "0.05")),
    )

    multitf = MultiTFConfig(
        enabled=os.getenv("MULTITF_ENABLED", "true").lower() in ("true", "1", "yes"),
        cache_ttl_seconds=int(os.getenv("MULTITF_CACHE_TTL", "900")),
    )

    optimizer = OptimizerConfig(
        max_combos=int(os.getenv("OPTIMIZER_MAX_COMBOS", "500")),
        default_days=int(os.getenv("OPTIMIZER_DEFAULT_DAYS", "730")),
    )

    return Settings(
        telegram=telegram,
        binance=binance,
        risk=risk,
        position=position,
        exchange=exchange,
        multitf=multitf,
        optimizer=optimizer,
        active_version=os.getenv("ACTIVE_VERSION", "V13"),
        loop_interval_seconds=int(os.getenv("LOOP_INTERVAL_SECONDS", "60")),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )
