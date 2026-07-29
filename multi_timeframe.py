"""
multi_timeframe.py
------------------
Filtro de confirmacao multi-timeframe para a estrategia CTEV.

Verifica a tendencia macro em H4 e D1 antes de confirmar sinais
gerados no timeframe principal (1H). Isso reduz falsos sinais
em contra-tendencia e aumenta a taxa de acerto.

Regras:
    - Para LONG:  H4 close > H4 EMA200  E  D1 close > D1 EMA200
    - Para SHORT: H4 close < H4 EMA200  E  D1 close < D1 EMA200
    - Se H4 e D1 discordam, o sinal e BLOQUEADO (filtro ativo)

Funcionalidades:
    - Fetch assincrono de candles H4 e D1 via ccxt
    - Calculo de EMA200 para cada timeframe superior
    - Cache de resultado para evitar fetch redundante
    - Status de tendencia macro disponivel via API

Referencias:
    - Investopedia (2025): "Multiple timeframe analysis involves using
      different timeframes to confirm trends"
    - LuxAlgo: "Using higher timeframe confluence is one of the most
      powerful ways to filter trades"
    - BabyPips: "Always align lower timeframes with the higher
      timeframe trend to improve win rate"
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger("ctev.multitf")


@dataclass(frozen=True)
class MTFResult:
    """Resultado da analise multi-timeframe."""
    h4_trend: str          # "bullish" | "bearish" | "neutral"
    d1_trend: str          # "bullish" | "bearish" | "neutral"
    confluence: str        # "bullish" | "bearish" | "mixed" | "insufficient"
    h4_close: float
    h4_ema200: float
    d1_close: float
    d1_ema200: float
    confirms_long: bool
    confirms_short: bool
    fetched_at: str

    def to_dict(self) -> dict:
        return {
            "h4_trend": self.h4_trend,
            "d1_trend": self.d1_trend,
            "confluence": self.confluence,
            "h4_close": round(self.h4_close, 2),
            "h4_ema200": round(self.h4_ema200, 2),
            "d1_close": round(self.d1_close, 2),
            "d1_ema200": round(self.d1_ema200, 2),
            "confirms_long": self.confirms_long,
            "confirms_short": self.confirms_short,
            "fetched_at": self.fetched_at,
        }


class MultiTimeframeFilter:
    """
    Filtro de confirmacao multi-timeframe. Singleton.

    Usa EMA(200) nos timeframes H4 e D1 para confirmar a tendencia
    macro antes de permitir sinais de entrada no timeframe 1H.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._enabled: bool = True
        self._h4_cache: Optional[MTFResult] = None
        self._cache_ttl_seconds: int = 900  # 15 min cache (H4 candle = 4h)
        self._cache_timestamp: float = 0.0

    def configure(self, kwargs: dict) -> None:
        """Configura o filtro."""
        with self._lock:
            if "enabled" in kwargs:
                self._enabled = bool(kwargs["enabled"])
            if "cache_ttl" in kwargs:
                self._cache_ttl_seconds = int(kwargs["cache_ttl"])
        logger.info(
            "MultiTimeframe configurado: enabled=%s cache_ttl=%ds",
            self._enabled, self._cache_ttl_seconds,
        )

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        with self._lock:
            self._enabled = value
        logger.info("MultiTimeframe: %s", "ATIVO" if value else "DESATIVADO")

    def check_cache_valid(self) -> bool:
        """Verifica se o cache ainda e valido."""
        import time
        return (time.time() - self._cache_timestamp) < self._cache_ttl_seconds

    async def analyze(self, exchange, symbol: str) -> MTFResult:
        """
        Busca candles H4 e D1, calcula EMA200 e retorna resultado.

        Parameters:
            exchange: instancia ccxt assincrona conectada
            symbol: par de trading (ex: BTC/USDT)

        Returns:
            MTFResult com tendencia e confluencia
        """
        if not self._enabled:
            return self._build_disabled_result()

        # Verifica cache
        if self._h4_cache is not None and self.check_cache_valid():
            return self._h4_cache

        try:
            # Fetch H4 candles (precisamos ~250 para EMA200)
            h4_ohlcv = await exchange.fetch_ohlcv(
                symbol=symbol, timeframe="4h", limit=250,
            )
            # Fetch D1 candles (precisamos ~250 para EMA200)
            d1_ohlcv = await exchange.fetch_ohlcv(
                symbol=symbol, timeframe="1d", limit=250,
            )

            h4_df = self._ohlcv_to_df(h4_ohlcv)
            d1_df = self._ohlcv_to_df(d1_ohlcv)

            h4_ema200 = self._calc_ema200(h4_df)
            d1_ema200 = self._calc_ema200(d1_df)

            if h4_ema200 is None or d1_ema200 is None:
                logger.warning("MultiTimeframe: dados insuficientes para EMA200.")
                return self._build_insufficient_result(
                    h4_df, d1_df, h4_ema200, d1_ema200,
                )

            h4_close = float(h4_df["close"].iloc[-1])
            d1_close = float(d1_df["close"].iloc[-1])

            # Determina tendencia
            h4_bullish = h4_close > h4_ema200
            d1_bullish = d1_close > d1_ema200

            h4_trend = "bullish" if h4_bullish else "bearish"
            d1_trend = "bullish" if d1_bullish else "bearish"

            # Confluencia
            confirms_long = h4_bullish and d1_bullish
            confirms_short = (not h4_bullish) and (not d1_bullish)

            if confirms_long:
                confluence = "bullish"
            elif confirms_short:
                confluence = "bearish"
            else:
                confluence = "mixed"

            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            result = MTFResult(
                h4_trend=h4_trend,
                d1_trend=d1_trend,
                confluence=confluence,
                h4_close=h4_close,
                h4_ema200=h4_ema200,
                d1_close=d1_close,
                d1_ema200=d1_ema200,
                confirms_long=confirms_long,
                confirms_short=confirms_short,
                fetched_at=now,
            )

            with self._lock:
                self._h4_cache = result
                self._cache_timestamp = __import__("time").time()

            logger.info(
                "MultiTimeframe: H4=%s D1=%s confluencia=%s | "
                "H4: close=%.2f ema200=%.2f | D1: close=%.2f ema200=%.2f",
                h4_trend, d1_trend, confluence,
                h4_close, h4_ema200, d1_close, d1_ema200,
            )
            return result

        except Exception as exc:
            logger.error("MultiTimeframe: erro ao buscar dados: %s", exc)
            return self._build_insufficient_result(None, None, None, None)

    def allows_signal(self, signal_type: str, mtf: Optional[MTFResult] = None) -> Tuple[bool, str]:
        """
        Verifica se o sinal e permitido pelo filtro multi-timeframe.

        Parameters:
            signal_type: "LONG" ou "SHORT"
            mtf: resultado da analise (se None, busca do cache)

        Returns:
            (allowed: bool, reason: str)
        """
        if not self._enabled:
            return True, "Filtro multi-timeframe desativado"

        if mtf is None:
            mtf = self._h4_cache

        if mtf is None:
            return True, "Sem dados MTF — permitindo sinal (cache vazio)"

        if signal_type.upper() == "LONG":
            if not mtf.confirms_long:
                reason = (
                    f"MTF BLOQUEOU LONG: H4={mtf.h4_trend} D1={mtf.d1_trend} "
                    f"(necessario bullish em ambos)"
                )
                logger.warning(reason)
                return False, reason
            return True, f"LONG confirmado: H4/D1 bullish"

        elif signal_type.upper() == "SHORT":
            if not mtf.confirms_short:
                reason = (
                    f"MTF BLOQUEOU SHORT: H4={mtf.h4_trend} D1={mtf.d1_trend} "
                    f"(necessario bearish em ambos)"
                )
                logger.warning(reason)
                return False, reason
            return True, f"SHORT confirmado: H4/D1 bearish"

        return True, "Tipo de sinal desconhecido — permitindo"

    def snapshot(self) -> dict:
        with self._lock:
            if self._h4_cache:
                return {
                    "enabled": self._enabled,
                    "cache_valid": self.check_cache_valid(),
                    **self._h4_cache.to_dict(),
                }
            return {
                "enabled": self._enabled,
                "cache_valid": False,
                "h4_trend": "unknown",
                "d1_trend": "unknown",
                "confluence": "insufficient",
                "confirms_long": True,
                "confirms_short": True,
                "h4_close": 0,
                "h4_ema200": 0,
                "d1_close": 0,
                "d1_ema200": 0,
                "fetched_at": "",
            }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ohlcv_to_df(ohlcv: list) -> pd.DataFrame:
        """Converte lista OHLCV do ccxt para DataFrame."""
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("datetime", inplace=True)
        df.drop(columns=["timestamp"], inplace=True)
        df.sort_index(inplace=True)
        return df

    @staticmethod
    def _calc_ema200(df: pd.DataFrame) -> Optional[float]:
        """Calcula EMA(200) e retorna o ultimo valor (ou None se insuficiente)."""
        if len(df) < 200:
            return None
        try:
            ema = df["close"].ewm(span=200, adjust=False).mean()
            val = float(ema.iloc[-1])
            return val if not pd.isna(val) else None
        except Exception:
            return None

    def _build_disabled_result(self) -> MTFResult:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return MTFResult(
            h4_trend="neutral", d1_trend="neutral",
            confluence="neutral",
            h4_close=0, h4_ema200=0, d1_close=0, d1_ema200=0,
            confirms_long=True, confirms_short=True,
            fetched_at=now,
        )

    def _build_insufficient_result(
        self, h4_df, d1_df, h4_ema200, d1_ema200,
    ) -> MTFResult:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        h4_c = float(h4_df["close"].iloc[-1]) if h4_df is not None and not h4_df.empty else 0
        d1_c = float(d1_df["close"].iloc[-1]) if d1_df is not None and not d1_df.empty else 0
        return MTFResult(
            h4_trend="neutral", d1_trend="neutral",
            confluence="insufficient",
            h4_close=h4_c, h4_ema200=h4_ema200 or 0,
            d1_close=d1_c, d1_ema200=d1_ema200 or 0,
            confirms_long=True, confirms_short=True,
            fetched_at=now,
        )


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------
_instance: Optional[MultiTimeframeFilter] = None
_lock = threading.Lock()


def get_mtf_filter() -> MultiTimeframeFilter:
    """Retorna a instancia unica de MultiTimeframeFilter."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MultiTimeframeFilter()
    return _instance
