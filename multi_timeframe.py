"""
multi_timeframe.py
------------------
Filtro de confirmacao multi-timeframe para a estrategia CTEV v4.

Verifica a tendencia macro em H4 e D1 antes de confirmar sinais
gerados no timeframe principal (1H). Isso reduz falsos sinais
em contra-tendencia e aumenta a taxa de acerto.

Regras (v4 — baseado no PDF "Framework Multi-Timeframe e de Regimes"):
    - Para LONG:  H4 close > H4 EMA50 E slope(EMA50) > 0  E  D1 close > D1 EMA200
    - Para SHORT: H4 close < H4 EMA50 E slope(EMA50) < 0  E  D1 close < D1 EMA200
    - Se H4 e D1 discordam, o sinal e BLOQUEADO (filtro ativo)

v4: Adicionado EMA(50) + slope no H4 (conforme PDF usa slope(EMA_4H_50, 20)),
    e EMA(200) no D1 para contexto macro.

Referencias:
    - PDF: "O Framework Multi-Timeframe e de Regimes" — slope-based trend
    - Investopedia (2025): "Multiple timeframe analysis involves using
      different timeframes to confirm trends"
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
    h4_ema50: float        # v4: EMA50 no H4 (slope-based)
    h4_ema200: float
    h4_slope: float        # v4: Slope da EMA50 no H4
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
            "h4_ema50": round(self.h4_ema50, 2),
            "h4_ema200": round(self.h4_ema200, 2),
            "h4_slope": round(self.h4_slope, 6),
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
        Busca candles H4 e D1, calcula EMA50+slope(H4) e EMA200(D1), retorna resultado.

        v4: Usa EMA(50) com slope no H4 (conforme PDF) e EMA(200) no D1.
        """
        if not self._enabled:
            return self._build_disabled_result()

        # Verifica cache
        if self._h4_cache is not None and self.check_cache_valid():
            return self._h4_cache

        try:
            # Fetch H4 candles (precisamos ~100 para EMA50 + 20 para slope)
            h4_ohlcv = await exchange.fetch_ohlcv(
                symbol=symbol, timeframe="4h", limit=250,
            )
            # Fetch D1 candles (precisamos ~250 para EMA200)
            d1_ohlcv = await exchange.fetch_ohlcv(
                symbol=symbol, timeframe="1d", limit=250,
            )

            h4_df = self._ohlcv_to_df(h4_ohlcv)
            d1_df = self._ohlcv_to_df(d1_ohlcv)

            # v4: EMA50 + slope no H4 (conforme PDF: slope(EMA_4H_50, 20))
            h4_ema50 = self._calc_ema50(h4_df)
            h4_slope = self._calc_ema_slope(h4_df)
            h4_ema200 = self._calc_ema200(h4_df)
            d1_ema200 = self._calc_ema200(d1_df)

            if h4_ema50 is None or d1_ema200 is None:
                logger.warning("MultiTimeframe: dados insuficientes.")
                return self._build_insufficient_result(
                    h4_df, d1_df, 0, 0,
                )

            h4_close = float(h4_df["close"].iloc[-1])
            d1_close = float(d1_df["close"].iloc[-1])

            # v4: Determina tendencia usando EMA50 + slope (do PDF)
            # slope > 0.001 = uptrend, slope < -0.001 = downtrend
            slope = h4_slope if h4_slope else 0.0
            h4_bullish = h4_close > h4_ema50 and slope > 0.001
            h4_bearish = h4_close < h4_ema50 and slope < -0.001
            d1_bullish = d1_close > d1_ema200

            if h4_bullish:
                h4_trend = "bullish"
            elif h4_bearish:
                h4_trend = "bearish"
            else:
                h4_trend = "neutral"

            d1_trend = "bullish" if d1_bullish else "bearish"

            # Confluencia (v4: H4 com slope + D1 com EMA200)
            confirms_long = h4_bullish and d1_bullish
            confirms_short = h4_bearish and (not d1_bullish)

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
                h4_ema50=h4_ema50,
                h4_ema200=h4_ema200 or 0,
                h4_slope=slope,
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
                "MultiTimeframe v4: H4=%s(slope=%.4f) D1=%s conf=%s",
                h4_trend, slope, d1_trend, confluence,
            )
            return result

        except Exception as exc:
            logger.error("MultiTimeframe: erro ao buscar dados: %s", exc)
            return self._build_insufficient_result(None, None, 0, 0)

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
                "h4_ema50": 0,
                "h4_ema200": 0, "h4_slope": 0.0,
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

    @staticmethod
    def _calc_ema50(df: pd.DataFrame) -> Optional[float]:
        """Calcula EMA(50) e retorna o ultimo valor (v4: usado no H4)."""
        if len(df) < 50:
            return None
        try:
            ema = df["close"].ewm(span=50, adjust=False).mean()
            val = float(ema.iloc[-1])
            return val if not pd.isna(val) else None
        except Exception:
            return None

    @staticmethod
    def _calc_ema_slope(df: pd.DataFrame, period: int = 20) -> Optional[float]:
        """
        Calcula o slope da EMA(50) no H4.

        slope = (EMA_atual - EMA_ha_20_periodos) / EMA_ha_20_periodos
        Conforme PDF: slope > 0.001 = uptrend, < -0.001 = downtrend.
        """
        if len(df) < 70:  # 50 para EMA + 20 para slope
            return None
        try:
            ema = df["close"].ewm(span=50, adjust=False).mean()
            slope = (float(ema.iloc[-1]) - float(ema.iloc[-1 - period])) / float(ema.iloc[-1 - period]) * 100
            return slope
        except Exception:
            return None

    def _build_disabled_result(self) -> MTFResult:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return MTFResult(
            h4_trend="neutral", d1_trend="neutral",
            confluence="neutral",
            h4_close=0, h4_ema50=0, h4_ema200=0, h4_slope=0.0,
            d1_close=0, d1_ema200=0,
            confirms_long=True, confirms_short=True,
            fetched_at=now,
        )

    def _build_insufficient_result(
        self, h4_df, d1_df, h4_ema50, d1_ema200,
    ) -> MTFResult:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        h4_c = float(h4_df["close"].iloc[-1]) if h4_df is not None and not h4_df.empty else 0
        d1_c = float(d1_df["close"].iloc[-1]) if d1_df is not None and not d1_df.empty else 0
        return MTFResult(
            h4_trend="neutral", d1_trend="neutral",
            confluence="insufficient",
            h4_close=h4_c, h4_ema50=h4_ema50 or 0,
            h4_ema200=h4_ema200 or 0, h4_slope=0.0,
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
