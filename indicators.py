"""
indicators.py
-------------
Cálculo de indicadores técnicos usando pandas-ta.

Indicadores suportados:
- EMA(200)
- Bollinger Bands(20, 2)
- RSI(14)
- Volume SMA(20)
- ATR(14)

Todos os cálculos operam sobre um DataFrame pandas com colunas
OCHLV padrão (Open, High, Low, Close, Volume).
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe um DataFrame com colunas ['open','high','low','close','volume']
    e devolve uma cópia enriquecida com as colunas de indicadores usadas
    pela estratégia CTEV:
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
        DataFrame com indicadores anexados. As últimas N linhas podem conter
        NaN dependendo do período do indicador (ex.: EMA200 precisa de 200+ candles).
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame está sem colunas obrigatórias: {missing}")

    if len(df) < 210:
        logger.warning(
            "DataFrame tem apenas %d linhas; EMA(200) e ATR(14) podem conter NaN nas primeiras linhas.",
            len(df),
        )

    out = df.copy()

    # EMA 200
    out["ema200"] = ta.ema(out["close"], length=200)

    # Bollinger Bands 20, 2 desvios
    bb = ta.bbands(out["close"], length=20, std=2)
    if bb is None or bb.empty:
        raise RuntimeError("Falha ao calcular Bollinger Bands.")
    # pandas-ta retorna colunas: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
    out["bb_lower"] = bb.iloc[:, 0]
    out["bb_middle"] = bb.iloc[:, 1]
    out["bb_upper"] = bb.iloc[:, 2]

    # RSI 14
    out["rsi"] = ta.rsi(out["close"], length=14)

    # Volume SMA 20
    out["volume_sma20"] = ta.sma(out["volume"], length=20)

    # ATR 14
    out["atr"] = ta.atr(out["high"], out["low"], out["close"], length=14)

    return out


def get_latest_signal_row(df_ind: pd.DataFrame) -> Optional[pd.Series]:
    """
    Retorna a última linha (candle mais recente) já com indicadores.
    Se houver NaN em qualquer coluna crítica da última linha, retorna None
    e emite warning (não há dados suficientes para sinalizar).
    """
    if df_ind.empty:
        return None

    last = df_ind.iloc[-1]
    critical = ["ema200", "bb_lower", "bb_upper", "rsi", "volume_sma20", "atr"]
    missing = [c for c in critical if pd.isna(last.get(c))]
    if missing:
        logger.warning(
            "Último candle possui NaN nos indicadores: %s. Aguardando mais dados.",
            missing,
        )
        return None
    return last
