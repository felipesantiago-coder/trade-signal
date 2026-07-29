"""
exchange_loader.py
------------------
Inicializacao automatica de exchange com fallback geografico.

Quando o bot roda em servidores americanos (ex: Render), exchanges como Binance
e Bybit bloqueiam o acesso com HTTP 403/451. Este modulo resolve isso com:

1. Tenta conectar na exchange preferida (EXCHANGE_ID env var)
2. Se falhar com erro de geo-bloqueio, tenta a proxima na cadeia
3. Inclui exchanges US-friendly no final (Kraken, Coinbase)
4. Mapeia pares automaticamente (BTC/USDT -> BTC/USD quando necessario)

Cadeia de fallback padrao:
    binance -> bybit -> kucoin -> okx -> gate -> bitget -> kraken -> coinbase

Para exchanges US-friendly, BTC/USDT e mapeado para BTC/USD automaticamente.

Referencias:
    - ccxt docs: "Unified API for 100+ cryptocurrency exchanges"
    - Render (2026): "Free web services run in Oregon, US data centers"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import ccxt.async_support as ccxt

logger = logging.getLogger("ctev.exchange_loader")

# Codes que indicam geo-bloqueio
GEOBLOCK_CODES = {403, 451}
GEOBLOCK_MESSAGES = (
    "restricted location",
    "block access from your country",
    "service unavailable from a restricted",
    "access denied",
    "forbidden",
)

# Mapeamento de pares: USDT -> USD para exchanges que nao suportam USDT
PAIR_MAP = {
    "BTC/USDT": "BTC/USD",
    "ETH/USDT": "ETH/USD",
    "BNB/USDT": "BNB/USD",
    "SOL/USDT": "SOL/USD",
    "XRP/USDT": "XRP/USD",
    "ADA/USDT": "ADA/USD",
    "DOGE/USDT": "DOGE/USD",
    "AVAX/USDT": "AVAX/USD",
    "DOT/USDT": "DOT/USD",
    "MATIC/USDT": "MATIC/USD",
}

# Exchanges que nao suportam pairs USDT (precisam de USD)
USD_ONLY_EXCHANGES = {"coinbase", "kraken", "gemini"}


@dataclass
class ExchangeInfo:
    """Informacoes da exchange conectada."""
    exchange_id: str
    symbol: str           # Par efetivamente usado (pode ser BTC/USD ao inves de BTC/USDT)
    original_symbol: str  # Par original solicitado (ex: BTC/USDT)
    is_us_friendly: bool
    connected_at: str

    def to_dict(self) -> dict:
        return {
            "exchange_id": self.exchange_id,
            "symbol": self.symbol,
            "original_symbol": self.original_symbol,
            "is_us_friendly": self.is_us_friendly,
            "connected_at": self.connected_at,
        }


class ExchangeLoader:
    """
    Loader com fallback automatico para exchanges bloqueadas por geo-restricao.

    Uso:
        loader = ExchangeLoader()
        exchange, info = await loader.connect(
            preferred_id="binance",
            symbol="BTC/USDT",
            api_key=None,
            api_secret=None,
        )
        # exchange = instancia ccxt conectada e funcionando
        # info = ExchangeInfo com detalhes
    """

    def __init__(self, fallback_chain: Optional[List[str]] = None) -> None:
        self._chain = fallback_chain or [
            "coinbase", "kraken",
            "binance", "bybit", "kucoin", "okx",
            "gate", "bitget",
        ]
        self._tried: List[str] = []

    async def connect(
        self,
        preferred_id: str = "coinbase",
        symbol: str = "BTC/USDT",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ) -> Tuple:
        """
        Tenta conectar na exchange preferida e faz fallback automatico.

        Returns:
            (exchange_instance, ExchangeInfo)
        """
        # Coloca a preferida primeiro (se nao estiver na cadeia)
        chain = list(self._chain)
        if preferred_id not in chain:
            chain.insert(0, preferred_id)
        else:
            chain.remove(preferred_id)
            chain.insert(0, preferred_id)

        last_error = None
        for ex_id in chain:
            self._tried.append(ex_id)
            try:
                exchange, info = await self._try_exchange(
                    ex_id, symbol, api_key, api_secret,
                )
                logger.info(
                    "Exchange conectada: %s (symbol=%s, us_friendly=%s) "
                    "apos tentar: %s",
                    ex_id, info.symbol, info.is_us_friendly,
                    " -> ".join(self._tried),
                )
                return exchange, info
            except _GeoBlockError as e:
                last_error = e
                logger.warning(
                    "Exchange %s bloqueada geograficamente: %s. Tentando proxima...",
                    ex_id, e,
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    "Exchange %s falhou: %s. Tentando proxima...",
                    ex_id, e,
                )

        # Todas falharam
        tried_str = ", ".join(self._tried)
        raise RuntimeError(
            f"Impossivel conectar em nenhuma exchange. "
            f"Tentadas: {tried_str}. "
            f"Ultimo erro: {last_error}"
        )

    async def _try_exchange(
        self,
        ex_id: str,
        symbol: str,
        api_key: Optional[str],
        api_secret: Optional[str],
    ) -> Tuple:
        """
        Tenta inicializar e testar uma exchange.

        Raises:
            _GeoBlockError: se bloqueada geograficamente
            Exception: outros erros de conexao
        """
        ex_class = getattr(ccxt, ex_id)
        if ex_class is None:
            raise RuntimeError(f"Exchange '{ex_id}' nao encontrada no ccxt.")

        config = {
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }
        if api_key:
            config["apiKey"] = api_key
        if api_secret:
            config["secret"] = api_secret

        exchange = ex_class(config)

        # Resolve o par (USDT -> USD se necessario)
        effective_symbol = symbol
        if ex_id in USD_ONLY_EXCHANGES:
            effective_symbol = PAIR_MAP.get(symbol, symbol.replace("/USDT", "/USD"))

        try:
            # Testa com fetch_ticker (leve, publico, sem autenticacao)
            ticker = await exchange.fetch_ticker(effective_symbol)

            # Verifica se o ticker tem dados validos
            if not ticker or "last" not in ticker or ticker["last"] is None:
                raise RuntimeError(f"Ticker invalido recebido de {ex_id}")

            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            info = ExchangeInfo(
                exchange_id=ex_id,
                symbol=effective_symbol,
                original_symbol=symbol,
                is_us_friendly=ex_id in USD_ONLY_EXCHANGES,
                connected_at=now,
            )

            # Fecha a conexao de teste e retorna (o worker vai criar a propria)
            await exchange.close()

            return ex_class(config), info

        except ccxt.NetworkError as e:
            await exchange.close()
            error_str = str(e).lower()
            status = getattr(e, "status", 0)

            if status in GEOBLOCK_CODES or any(
                msg in error_str for msg in GEOBLOCK_MESSAGES
            ):
                raise _GeoBlockError(
                    f"{ex_id}: HTTP {status} — geo-bloqueado ({e})"
                )
            raise
        except ccxt.ExchangeError as e:
            await exchange.close()
            error_str = str(e).lower()
            status = getattr(e, "status", 0)

            if status in GEOBLOCK_CODES or any(
                msg in error_str for msg in GEOBLOCK_MESSAGES
            ):
                raise _GeoBlockError(
                    f"{ex_id}: HTTP {status} — geo-bloqueado ({e})"
                )
            raise
        except Exception as e:
            await exchange.close()
            error_str = str(e).lower()
            status = getattr(getattr(e, "response", None), "status_code", 0)

            if status in GEOBLOCK_CODES or any(
                msg in error_str for msg in GEOBLOCK_MESSAGES
            ):
                raise _GeoBlockError(
                    f"{ex_id}: HTTP {status} — geo-bloqueado ({e})"
                )
            raise


class _GeoBlockError(Exception):
    """Erro especifico para geo-bloqueio."""
    pass
