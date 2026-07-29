"""
indicators.py
-------------
Calculo de indicadores tecnicos usando pandas/numpy (sem dependencias externas).

Indicadores suportados:
- EMA(200)
- Bollinger Bands(20, 2)
- RSI(14)
- Volume SMA(20)
- ATR(14)
- ATR Percentile (rank dos ultimos 100 candles — filtro de volatilidade)
- Bollinger Bandwidth (deteccao de squeeze)

Todos os calculos operam sobre um DataFrame pandas com colunas
OCHLV padrao (Open, High, Low, Close, Volume).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Calcula EMA usando ewm do pandas."""
    return series.ewm(span=span, adjust=False).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    """Calcula SMA usando rolling do pandas."""
    return series.rolling(window=window).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calcula RSI (Relative Strength Index) usando metodo Wilder's smoothing.
    Equivalente ao RSI(14) do TradingView/pandas-ta.
    """
    delta = close.diff()
    gain = delta.where(delta > 0.0, 0.0)
    loss = (-delta).where(delta < 0.0, 0.0)

    # Wilder's smoothing: media exponencial com alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calcula ATR (Average True Range) usando metodo Wilder's smoothing.
    Equivalente ao ATR(14) do TradingView/pandas-ta.
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe um DataFrame com colunas ['open','high','low','close','volume']
    e devolve uma copia enriquecida com as colunas de indicadores usadas
    pela estrategia CTEV:
        - ema200
        - bb_lower, bb_middle, bb_upper
        - rsi
        - volume_sma20
        - atr

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame OCHLV ordenado por tempo ascendente.

    Returns
    -------
    pd.DataFrame
        DataFrame com indicadores anexados. As ultimas N linhas podem conter
        NaN dependendo do periodo do indicador (ex.: EMA200 precisa de 200+ candles).
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame esta sem colunas obrigatorias: {missing}")

    if len(df) < 210:
        logger.warning(
            "DataFrame tem apenas %d linhas; EMA(200) e ATR(14) podem conter NaN nas primeiras linhas.",
            len(df),
        )

    out = df.copy()

    # EMA 200
    out["ema200"] = _ema(out["close"], span=200)

    # Bollinger Bands 20, 2 desvios
    bb_middle = _sma(out["close"], window=20)
    bb_std = out["close"].rolling(window=20).std(ddof=0)
    out["bb_lower"] = bb_middle - 2.0 * bb_std
    out["bb_middle"] = bb_middle
    out["bb_upper"] = bb_middle + 2.0 * bb_std

    # RSI 14
    out["rsi"] = _rsi(out["close"], period=14)

    # Volume SMA 20
    out["volume_sma20"] = _sma(out["volume"], window=20)

    # ATR 14
    out["atr"] = _atr(out["high"], out["low"], out["close"], period=14)

    # ATR Percentile — ranking do ATR atual entre os ultimos 100 candles
    # Valor entre 0.0 e 1.0. Usado como filtro de volatilidade:
    #   - Baixo (< 0.20): mercado lateral/morto → evitar
    #   - Normal (0.20 - 0.80): condicoes favoraveis → operar
    #   - Alto (> 0.80): caos/evento extremo → evitar
    ATR_LOOKBACK = 100
    out["atr_percentile"] = out["atr"].rolling(ATR_LOOKBACK).apply(
        lambda x: (x.iloc[-1] > x).sum() / max(len(x) - 1, 1), raw=False
    )

    # Bollinger Bandwidth — largura relativa das bandas (squeeze detection)
    # BW = (upper - lower) / middle * 100
    out["bb_width"] = ((out["bb_upper"] - out["bb_lower"]) / out["bb_middle"]) * 100

    return out


def get_latest_signal_row(df_ind: pd.DataFrame) -> Optional[pd.Series]:
    """
    Retorna a ultima linha (candle mais recente) ja com indicadores.
    Se houver NaN em qualquer coluna critica da ultima linha, retorna None
    e emite warning (nao ha dados suficientes para sinalizar).
    """
    if df_ind.empty:
        return None

    last = df_ind.iloc[-1]
    critical = ["ema200", "bb_lower", "bb_upper", "rsi", "volume_sma20", "atr", "atr_percentile"]
    missing = [c for c in critical if pd.isna(last.get(c))]
    if missing:
        logger.warning(
            "Ultimo candle possui NaN nos indicadores: %s. Aguardando mais dados.",
            missing,
        )
        return None
    return last
