"""
multi_timeframe.py
------------------
Filtro de confirmacao multi-timeframe ADAPTATIVO.

Verifica a tendencia em timeframes superiores ao ativo antes de
confirmar sinais. Os timeframes de confirmacao se adaptam automaticamente:

  TF ativo  |  Confirmacao 1  |  Confirmacao 2
  ----------+----------------+----------------
  15m       |  1h             |  4h
  30m       |  1h             |  4h
  1h        |  4h             |  1d
  2h        |  4h             |  1d
  4h        |  1d             |  —
  1d        |  (desativado)   |  —

Regras (v4+ adaptativo):
    - Para LONG:  TF1 close > TF1 EMA50 E slope(EMA50) > 0  E  TF2 close > TF2 EMA200
    - Para SHORT: TF1 close < TF1 EMA50 E slope(EMA50) < 0  E  TF2 close < TF2 EMA200
    - Se TF1 e TF2 discordam, o sinal e BLOQUEADO

v5: Adicionado adaptacao automatica dos timeframes de confirmacao
    baseado no timeframe ativo (via strategy_router.get_mtf_timeframes).

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
    h4_ema50: float        # v4: EMA50 no TF1 (slope-based)
    h4_ema200: float
    h4_slope: float        # v4: Slope da EMA50 no TF1
    d1_close: float
    d1_ema200: float
    confirms_long: bool
    confirms_short: bool
    fetched_at: str
    # v5: timeframes usados na analise
    tf_active: str = "1h"   # timeframe ativo
    tf_confirm1: str = "4h" # primeiro TF de confirmacao
    tf_confirm2: str = "1d" # segundo TF de confirmacao (pode ser None)

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
            "tf_active": self.tf_active,
            "tf_confirm1": self.tf_confirm1,
            "tf_confirm2": self.tf_confirm2,
        }


class MultiTimeframeFilter:
    """
    Filtro de confirmacao multi-timeframe ADAPTATIVO. Singleton.

    v5: Os timeframes de confirmacao se adaptam ao timeframe ativo.
    15m/30m -> confirma em 1h + 4h
    1h      -> confirma em 4h + 1d (original)
    4h+     -> confirma em 1d apenas
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._enabled: bool = True
        self._h4_cache: Optional[MTFResult] = None
        self._cache_ttl_seconds: int = 900  # 15 min cache
        self._cache_timestamp: float = 0.0
        self._active_timeframe: str = "1h"  # v5: TF ativo para adaptacao MTF

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

    def set_active_timeframe(self, tf: str) -> None:
        """v5: Atualiza o timeframe ativo para adaptar MTF."""
        with self._lock:
            if tf != self._active_timeframe:
                self._active_timeframe = tf
                self._h4_cache = None  # Invalida cache
                logger.info("MTF: timeframe ativo -> %s", tf)

    @property
    def active_timeframe(self) -> str:
        with self._lock:
            return self._active_timeframe

    async def analyze(self, exchange, symbol: str, active_tf: str = None) -> MTFResult:
        """
        Busca candles nos TFs de confirmacao, calcula EMA50+slope + EMA200.

        v5: Os TFs de confirmacao se adaptam ao timeframe ativo.
        """
        if not self._enabled:
            return self._build_disabled_result()

        # v5: Atualiza TF ativo
        if active_tf:
            self.set_active_timeframe(active_tf)

        # Verifica cache
        if self._h4_cache is not None and self.check_cache_valid():
            return self._h4_cache

        try:
            # v5: Resolve TFs de confirmacao baseado no TF ativo
            from strategy_router import get_mtf_timeframes
            tf_confs = get_mtf_timeframes(self._active_timeframe)
            tf1 = tf_confs[0] if tf_confs[0] else "4h"
            tf2 = tf_confs[1] if len(tf_confs) > 1 and tf_confs[1] else None

            # Fetch TF1 candles (EMA50 + slope)
            tf1_ohlcv = await exchange.fetch_ohlcv(
                symbol=symbol, timeframe=tf1, limit=250,
            )
            # Fetch TF2 candles (EMA200 macro) se disponivel
            tf2_ohlcv = None
            if tf2:
                tf2_ohlcv = await exchange.fetch_ohlcv(
                    symbol=symbol, timeframe=tf2, limit=250,
                )

            tf1_df = self._ohlcv_to_df(tf1_ohlcv)
            tf2_df = self._ohlcv_to_df(tf2_ohlcv) if tf2_ohlcv else None

            # v4: EMA50 + slope no TF1 (conforme PDF: slope(EMA_50, 20))
            tf1_ema50 = self._calc_ema50(tf1_df)
            tf1_slope = self._calc_ema_slope(tf1_df)
            tf1_ema200 = self._calc_ema200(tf1_df)
            tf2_ema200 = self._calc_ema200(tf2_df) if tf2_df is not None else None

            if tf1_ema50 is None:
                logger.warning("MultiTimeframe: dados insuficientes para %s.", tf1)
                return self._build_insufficient_result(
                    tf1_df, tf2_df, 0, 0,
                )

            tf1_close = float(tf1_df["close"].iloc[-1])
            tf2_close = float(tf2_df["close"].iloc[-1]) if tf2_df is not None else 0.0

            # v4: Determina tendencia usando EMA50 + slope
            slope = tf1_slope if tf1_slope else 0.0
            tf1_bullish = tf1_close > tf1_ema50 and slope > 0.001
            tf1_bearish = tf1_close < tf1_ema50 and slope < -0.001
            tf2_bullish = tf2_close > tf2_ema200 if tf2_ema200 and tf2_close > 0 else True  # Se nao ha TF2, permite

            if tf1_bullish:
                tf1_trend = "bullish"
            elif tf1_bearish:
                tf1_trend = "bearish"
            else:
                tf1_trend = "neutral"

            tf2_trend = "bullish" if tf2_bullish else "bearish"

            # Confluencia
            confirms_long = tf1_bullish and tf2_bullish
            confirms_short = tf1_bearish and (not tf2_bullish)

            if confirms_long:
                confluence = "bullish"
            elif confirms_short:
                confluence = "bearish"
            else:
                confluence = "mixed"

            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            result = MTFResult(
                h4_trend=tf1_trend,
                d1_trend=tf2_trend,
                confluence=confluence,
                h4_close=tf1_close,
                h4_ema50=tf1_ema50,
                h4_ema200=tf1_ema200 or 0,
                h4_slope=slope,
                d1_close=tf2_close,
                d1_ema200=tf2_ema200 or 0,
                confirms_long=confirms_long,
                confirms_short=confirms_short,
                fetched_at=now,
                tf_active=self._active_timeframe,
                tf_confirm1=tf1,
                tf_confirm2=tf2 or "",
            )

            with self._lock:
                self._h4_cache = result
                self._cache_timestamp = __import__("time").time()

            logger.info(
                "MultiTimeframe v5 [%s]: TF1=%s(%s) TF2=%s(%s) conf=%s",
                self._active_timeframe, tf1, tf1_trend, tf2 or "-", tf2_trend, confluence,
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
