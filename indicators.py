"""
indicators.py
-------------
Calculo de indicadores tecnicos usando pandas/numpy (sem dependencias externas).

Indicadores suportados:
- EMA(50) e EMA(200) — filtro duplo de tendencia
- Bollinger Bands(20, 2.0) — volatilidade padrao
- RSI(14) — com derivacao de momentum (delta RSI)
- Volume SMA(20) — confirmacao de volume
- ATR(14) — volatilidade para SL/TP
- ATR Percentile — filtro de volatilidade
- Bollinger Bandwidth — squeeze detection
- MACD(12, 26, 9) — confirmacao de momentum
- Fibonacci Retracement Levels — 0.236, 0.382, 0.500, 0.618, 0.786
- Swing High/Low Detection — para calculo de Fibonacci
- EMA(50) touch detection — pullback alternativo

Todos os calculos operam sobre um DataFrame pandas com colunas
OCHLV padrao (Open, High, Low, Close, Volume).

v3: Adicionado MACD, melhoria no Fibonacci (swing lookback menor,
lookback maior), EMA touch detection.
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


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """
    Calcula MACD (Moving Average Convergence Divergence).

    Returns:
        Tuple de (macd_line, signal_line, histogram)
    """
    ema_fast = _ema(close, span=fast)
    ema_slow = _ema(close, span=slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, span=signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _detect_swing_points(
    high: pd.Series, low: pd.Series, lookback: int = 10
) -> tuple:
    """
    Detecta swing highs e swing lows usando uma janela de lookback.

    Um swing high e o ponto mais alto dentro de uma janela centrada.
    Um swing low e o ponto mais baixo dentro de uma janela centrada.

    v3: lookback reduzido de 20 para 10 para detectar swings mais frequentes
    em timeframe 1H (10h = meio dia de dados).
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
    high: pd.Series, low: pd.Series, close: pd.Series,
    swing_highs: pd.Series, swing_lows: pd.Series,
    lookback: int = 120
) -> pd.DataFrame:
    """
    Calcula niveis de Fibonacci retracement baseados no ultimo swing relevante.

    v3: lookback aumentado de 50 para 120 para capturar movimentos maiores
    (5 dias em 1H = 120 candles).

    Retorna DataFrame com colunas:
        - fib_0236, fib_0382, fib_0500, fib_0618, fib_0786
        - fib_swing_high, fib_swing_low (para referencia)
        - fib_direction (1 = uptrend, -1 = downtrend)
        - fib_proximity: distancia % do preco ao nivel Fibonacci mais proximo
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
        "fib_proximity": np.full(n, np.nan),
    }

    # Indices onde ocorrem swing highs e swing lows
    sh_indices = np.where(swing_highs.values)[0]
    sl_indices = np.where(swing_lows.values)[0]

    fib_levels = [0.236, 0.382, 0.500, 0.618, 0.786]
    fib_keys = ["fib_0236", "fib_0382", "fib_0500", "fib_0618", "fib_0786"]

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

        # Ignora swings com diff menor que 1% (ruido)
        price_now = float(close.iloc[i])
        if abs(sh_price - sl_price) < price_now * 0.01:
            continue

        # Determina direcao: se swing high veio DEPOIS do swing low = uptrend
        if last_sh_idx > last_sl_idx:
            swing_low_price = sl_price
            swing_high_price = sh_price
            direction = 1
        else:
            swing_low_price = sh_price
            swing_high_price = sl_price
            direction = -1

        diff = swing_high_price - swing_low_price
        if diff == 0:
            continue

        # Calcula niveis
        for level, key in zip(fib_levels, fib_keys):
            fib_data[key][i] = swing_high_price - level * diff

        fib_data["fib_swing_high"][i] = swing_high_price
        fib_data["fib_swing_low"][i] = swing_low_price
        fib_data["fib_direction"][i] = direction

        # Calcula distancia ao nivel Fibonacci mais proximo
        min_dist = float("inf")
        for level, key in zip(fib_levels, fib_keys):
            fib_val = fib_data[key][i]
            if not np.isnan(fib_val) and fib_val > 0:
                dist_pct = abs(price_now - fib_val) / price_now * 100
                if dist_pct < min_dist:
                    min_dist = dist_pct
        if min_dist < float("inf"):
            fib_data["fib_proximity"][i] = min_dist

    return pd.DataFrame(fib_data, index=high.index)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe um DataFrame com colunas ['open','high','low','close','volume']
    e devolve uma copia enriquecida com os indicadores da estrategia CTEV v3.

    Indicadores calculados:
        - ema50: EMA de 50 periodos (filtro medio de tendencia)
        - ema200: EMA de 200 periodos (filtro lento de tendencia)
        - bb_lower, bb_middle, bb_upper: Bollinger Bands(20, 2.0)
        - rsi: RSI(14)
        - rsi_delta: RSI atual - RSI anterior (momentum)
        - volume_sma20: Media de volume 20 periodos
        - atr: ATR(14)
        - atr_percentile: Ranking do ATR nos ultimos 100 candles
        - bb_width: Largura das bandas Bollinger
        - macd, macd_signal, macd_hist: MACD(12, 26, 9)
        - fib_0236, fib_0382, fib_0500, fib_0618, fib_0786: Fibonacci levels
        - fib_direction: Direcao do Fibonacci (1=uptrend, -1=downtrend)
        - fib_proximity: Distancia % ao nivel Fibonacci mais proximo
        - ema50_touched: True se low <= EMA50 (pullback para LONG)
        - ema50_touched_up: True se high >= EMA50 (pullback para SHORT)
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

    # ── Bollinger Bands 20, 2.0 desvios (padrao) ──
    bb_middle = _sma(out["close"], window=20)
    bb_std = out["close"].rolling(window=20).std(ddof=0)
    out["bb_lower"] = bb_middle - 2.0 * bb_std
    out["bb_middle"] = bb_middle
    out["bb_upper"] = bb_middle + 2.0 * bb_std

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

    # ── MACD (12, 26, 9) ──
    out["macd"], out["macd_signal"], out["macd_hist"] = _macd(out["close"])

    # ── Fibonacci Retracement Levels (v3: melhorado) ──
    swing_highs, swing_lows = _detect_swing_points(out["high"], out["low"], lookback=10)
    fib_df = _compute_fibonacci_levels(
        out["high"], out["low"], out["close"],
        swing_highs, swing_lows, lookback=120
    )
    out = pd.concat([out, fib_df], axis=1)

    # ── EMA(50) Touch Detection (pullback alternativo) ──
    # Para LONG: low tocou EMA50 (pullback de alta)
    out["ema50_touched"] = out["low"] <= out["ema50"]
    # Para SHORT: high tocou EMA50 (pullback de baixa)
    out["ema50_touched_up"] = out["high"] >= out["ema50"]

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
        "ema50", "ema200", "rsi", "rsi_delta",
        "volume_sma20", "atr", "atr_percentile",
        "macd", "macd_signal", "macd_hist",
    ]
    missing = [c for c in critical if pd.isna(last.get(c))]
    if missing:
        logger.warning(
            "Ultimo candle possui NaN nos indicadores: %s. Aguardando mais dados.",
            missing,
        )
        return None
    return last
