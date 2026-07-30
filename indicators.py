"""
indicators.py
-------------
Calculo de indicadores tecnicos usando pandas/numpy (sem dependencias externas).

Indicadores suportados:
- EMA(50) e EMA(200) — filtro duplo de tendencia
- Bollinger Bands(20, 2.5) — mais largas para BTC
- RSI(14) — com derivacao de momentum (delta RSI)
- Volume SMA(20) — confirmacao de volume
- ATR(14) — volatilidade para SL/TP
- ATR Percentile — filtro de volatilidade
- Bollinger Bandwidth — squeeze detection
- Fibonacci Retracement Levels — 0.236, 0.382, 0.500, 0.618, 0.786
- Swing High/Low Detection — para calculo de Fibonacci

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


def _detect_swing_points(
    high: pd.Series, low: pd.Series, lookback: int = 20
) -> tuple:
    """
    Detecta swing highs e swing lows usando uma janela de lookback.
    
    Um swing high e o ponto mais alto dentro de uma janela centrada.
    Um swing low e o ponto mais baixo dentro de uma janela centrada.
    
    Returns:
        Tuple de (swing_highs, swing_lows) como Series booleanas.
    """
    swing_high = (
        (high == high.rolling(window=lookback, center=True).max()) &
        (high.shift(1) < high) &
        (high.shift(-1) < high)
    )
    swing_low = (
        (low == low.rolling(window=lookback, center=True).min()) &
        (low.shift(1) > low) &
        (low.shift(-1) > low)
    )
    return swing_high, swing_low


def _compute_fibonacci_levels(
    high: pd.Series, low: pd.Series, swing_highs: pd.Series, swing_lows: pd.Series,
    lookback: int = 50
) -> pd.DataFrame:
    """
    Calcula niveis de Fibonacci retracement baseados no ultimo swing relevante.
    
    Para cada candle, identifica o swing high e swing low mais recente
    dentro do lookback e calcula os niveis de Fibonacci entre eles.
    
    Retorna DataFrame com colunas:
        - fib_0236, fib_0382, fib_0500, fib_0618, fib_0786
        - fib_swing_high, fib_swing_low (para referencia)
        - fib_direction (1 = uptrend, -1 = downtrend)
    """
    n = len(high)
    fib_data = {
        "fib_0236": np.full(n, np.nan),
        "fib_0382": np.full(n, np.nan),
        "fib_0500": np.full(n, np.nan),
        "fib_0618": np.full(n, np.nan),
        "fib_0786": np.full(n, np.nan),
        "fib_swing_high": np.full(n, np.nan),
        "fib_swing_low": np.full(n, np.nan),
        "fib_direction": np.full(n, 0),
    }

    # Indices onde ocorrem swing highs e swing lows
    sh_indices = np.where(swing_highs.values)[0]
    sl_indices = np.where(swing_lows.values)[0]

    for i in range(n):
        # Busca o ultimo swing high e swing low dentro do lookback
        window_start = max(0, i - lookback)
        
        relevant_sh = sh_indices[(sh_indices >= window_start) & (sh_indices < i)]
        relevant_sl = sl_indices[(sl_indices >= window_start) & (sl_indices < i)]

        if len(relevant_sh) == 0 or len(relevant_sl) == 0:
            continue

        # Pega o swing high e low mais recentes
        last_sh_idx = relevant_sh[-1]
        last_sl_idx = relevant_sl[-1]

        sh_price = float(high.iloc[last_sh_idx])
        sl_price = float(low.iloc[last_sl_idx])

        # Determina direcao: se swing high veio DEPOIS do swing low = uptrend
        if last_sh_idx > last_sl_idx:
            # Uptrend: Fibonacci calculado do low para o high
            swing_low_price = sl_price
            swing_high_price = sh_price
            direction = 1
        else:
            # Downtrend: Fibonacci calculado do high para o low
            swing_low_price = sh_price
            swing_high_price = sl_price
            direction = -1

        diff = swing_high_price - swing_low_price
        if diff == 0:
            continue

        # Calcula niveis (para uptrend, 0.786 e o mais proximo do low; para downtrend, invertido)
        fib_data["fib_0236"][i] = swing_high_price - 0.236 * diff
        fib_data["fib_0382"][i] = swing_high_price - 0.382 * diff
        fib_data["fib_0500"][i] = swing_high_price - 0.500 * diff
        fib_data["fib_0618"][i] = swing_high_price - 0.618 * diff
        fib_data["fib_0786"][i] = swing_high_price - 0.786 * diff
        fib_data["fib_swing_high"][i] = sh_price if direction == 1 else sl_price
        fib_data["fib_swing_low"][i] = sl_price if direction == 1 else sh_price
        fib_data["fib_direction"][i] = direction

    return pd.DataFrame(fib_data, index=high.index)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe um DataFrame com colunas ['open','high','low','close','volume']
    e devolve uma copia enriquecida com os indicadores da estrategia CTEV v2.

    Indicadores calculados:
        - ema50: EMA de 50 periodos (filtro medio de tendencia)
        - ema200: EMA de 200 periodos (filtro lento de tendencia)
        - bb_lower, bb_middle, bb_upper: Bollinger Bands(20, 2.5)
        - rsi: RSI(14)
        - rsi_delta: RSI atual - RSI anterior (momentum)
        - volume_sma20: Media de volume 20 periodos
        - atr: ATR(14)
        - atr_percentile: Ranking do ATR nos ultimos 100 candles
        - bb_width: Largura das bandas Bollinger
        - fib_0236, fib_0382, fib_0500, fib_0618, fib_0786: Fibonacci levels
        - fib_direction: Direcao do Fibonacci (1=uptrend, -1=downtrend)
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame esta sem colunas obrigatorias: {missing}")

    if len(df) < 210:
        logger.warning(
            "DataFrame tem apenas %d linhas; indicadores podem conter NaN.",
            len(df),
        )

    out = df.copy()

    # ── EMA 50 (filtro medio de tendencia) ──
    out["ema50"] = _ema(out["close"], span=50)

    # ── EMA 200 (filtro lento de tendencia) ──
    out["ema200"] = _ema(out["close"], span=200)

    # ── Bollinger Bands 20, 2.5 desvios (mais largas para BTC) ──
    bb_middle = _sma(out["close"], window=20)
    bb_std = out["close"].rolling(window=20).std(ddof=0)
    out["bb_lower"] = bb_middle - 2.5 * bb_std
    out["bb_middle"] = bb_middle
    out["bb_upper"] = bb_middle + 2.5 * bb_std

    # ── RSI 14 ──
    out["rsi"] = _rsi(out["close"], period=14)

    # ── RSI Delta (momentum: RSI atual - RSI anterior) ──
    out["rsi_delta"] = out["rsi"] - out["rsi"].shift(1)

    # ── Volume SMA 20 ──
    out["volume_sma20"] = _sma(out["volume"], window=20)

    # ── ATR 14 ──
    out["atr"] = _atr(out["high"], out["low"], out["close"], period=14)

    # ── ATR Percentile ──
    ATR_LOOKBACK = 100
    out["atr_percentile"] = out["atr"].rolling(ATR_LOOKBACK).apply(
        lambda x: (x.iloc[-1] > x).sum() / max(len(x) - 1, 1), raw=False
    )

    # ── Bollinger Bandwidth ──
    out["bb_width"] = ((out["bb_upper"] - out["bb_lower"]) / out["bb_middle"]) * 100

    # ── Fibonacci Retracement Levels ──
    swing_highs, swing_lows = _detect_swing_points(out["high"], out["low"], lookback=20)
    fib_df = _compute_fibonacci_levels(
        out["high"], out["low"], swing_highs, swing_lows, lookback=50
    )
    out = pd.concat([out, fib_df], axis=1)

    return out


def get_latest_signal_row(df_ind: pd.DataFrame) -> Optional[pd.Series]:
    """
    Retorna a ultima linha (candle mais recente) ja com indicadores.
    Se houver NaN em qualquer coluna critica da ultima linha, retorna None.
    """
    if df_ind.empty:
        return None

    last = df_ind.iloc[-1]
    critical = [
        "ema50", "ema200", "bb_lower", "bb_upper", "rsi", "rsi_delta",
        "volume_sma20", "atr", "atr_percentile",
        "fib_0382", "fib_0500",
    ]
    missing = [c for c in critical if pd.isna(last.get(c))]
    if missing:
        logger.warning(
            "Ultimo candle possui NaN nos indicadores: %s. Aguardando mais dados.",
            missing,
        )
        return None
    return last
