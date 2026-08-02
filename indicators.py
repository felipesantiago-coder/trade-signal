"""
indicators.py
-------------
Calculo de indicadores tecnicos usando pandas/numpy (sem dependencias externas).

Indicadores suportados:
- EMA(20), EMA(50) e EMA(200) — filtro de tendencia multi-camada
- Bollinger Bands(20, 2.0) — volatilidade padrao + squeeze detection
- RSI(14) — com derivacao de momentum (delta RSI)
- Volume SMA(20) e SMA(50) — confirmacao de volume (soft + strict)
- ATR(14) — volatilidade para SL/TP
- ATR Percentile — filtro de volatilidade
- Bollinger Bandwidth + BB Squeeze Percentile — detectar expansao de volatilidade
- MACD(12, 26, 9) — confirmacao de momentum
- ADX(14) — forca de tendencia + DI+/DI- para regime detection
- Fibonacci Retracement Levels — 0.236, 0.382, 0.500, 0.618, 0.786
- Swing High/Low Detection — para calculo de Fibonacci
- EMA(20/50) touch detection — pullback
- Regime classification — trending_up, trending_down, ranging, volatile
- EMA Slope — inclinacao da EMA para confirmacao de tendencia

Todos os calculos operam sobre um DataFrame pandas com colunas
OCHLV padrao (Open, High, Low, Close, Volume).

v4: ADX + regime detection, BB squeeze percentile, EMA slope,
     volume SMA(50), EMA(20), cost-aware indicators.
     Baseado no estudo "Framework Multi-Timeframe e de Regimes"
     para BTC/USDT no timeframe 1H.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _get_timeframe_multiplier(timeframe: str) -> float:
    """
    Retorna o multiplicador de candles em relacao ao timeframe de 1h.
    Ex: 15m -> 4.0 (4 candles de 15m por candle de 1h)
        1h  -> 1.0 (base)
        4h  -> 0.25

    Usado para escalar lookbacks de indicadores que dependem de tempo real
    (Fibonacci, swing detection, ATR percentile, BB squeeze percentile).
    """
    _tf_minutes = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "6h": 360,
        "8h": 480, "12h": 720, "1d": 1440,
    }
    if timeframe not in _tf_minutes:
        raise ValueError(
            f"Timeframe '{timeframe}' nao suportado. "
            f"Opcoes: {list(_tf_minutes.keys())}"
        )
    return 60.0 / _tf_minutes[timeframe]


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


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple:
    """
    Calcula ADX (Average Directional Index) com +DI e -DI.

    ADX > 25 = tendencia forte (operar trend-following)
    ADX < 20 = mercado lateral (evitar trend-following, ou usar mean-reversion)
    ADX 20-25 = transicao (cuidado)

    Returns:
        Tuple de (adx, plus_di, minus_di)
    """
    # True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = pd.Series(0.0, index=high.index, dtype=float)
    minus_dm = pd.Series(0.0, index=high.index, dtype=float)

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # Wilder's smoothing
    atr_smooth = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    plus_di_smooth = plus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    minus_di_smooth = minus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    # DI values
    plus_di = 100.0 * (plus_di_smooth / atr_smooth)
    minus_di = 100.0 * (minus_di_smooth / atr_smooth)

    # DX and ADX
    di_sum = plus_di + minus_di
    di_sum = di_sum.replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / di_sum) * 100.0
    adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    return adx, plus_di, minus_di


def _ema_slope(series: pd.Series, slope_period: int = 20) -> pd.Series:
    """
    Calcula o slope (declive) de uma serie, medindo a inclinacao
    ao longo de N periodos.

    slope > 0 = tendencia de alta
    slope < 0 = tendencia de baixa
    slope ~ 0 = lateralizacao (sem tendencia)

    O slope e calculado como a taxa de variacao percentual da EMA
    ao longo do periodo de slope, normalizado pelo preco.
    """
    ema = series.ewm(span=slope_period, adjust=False).mean()
    slope = (ema - ema.shift(slope_period)) / ema.shift(slope_period) * 100
    return slope


def _classify_regime(
    adx: pd.Series, plus_di: pd.Series, minus_di: pd.Series,
    close: pd.Series, ema50: pd.Series, bb_width: pd.Series,
    bb_width_pct: pd.Series,
) -> pd.Series:
    """
    Classifica o regime de mercado para cada candle.

    Regimes:
        - "trending_up": ADX > 25, +DI > -DI, close > EMA50
        - "trending_down": ADX > 25, -DI > +DI, close < EMA50
        - "ranging": ADX < 20 (mercado lateral)
        - "transition": ADX 20-25 (transicao)
        - "volatile": ADX > 25 e BB width no percentil alto (> 80)

    Baseado no estudo de Adaptive Regime-Based Trading (ref. 48 do PDF).
    """
    regime = pd.Series("ranging", index=adx.index, dtype=str)

    # Trending UP: ADX forte, DI+ dominante, preco acima EMA50
    trending_up = (adx > 25) & (plus_di > minus_di) & (close > ema50)
    regime = regime.where(~trending_up, "trending_up")

    # Trending DOWN: ADX forte, DI- dominante, preco abaixo EMA50
    trending_down = (adx > 25) & (minus_di > plus_di) & (close < ema50)
    regime = regime.where(~trending_down, "trending_down")

    # Volatile: tendencia com BB width muito alta
    volatile = (regime.isin(["trending_up", "trending_down"])) & (bb_width_pct > 0.80)
    regime = regime.where(~volatile, "volatile")

    # Transition: ADX entre 20-25
    transition = (adx >= 20) & (adx <= 25) & (regime == "ranging")
    regime = regime.where(~transition, "transition")

    return regime


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

    O lookback deve ser passado pelo chamador de acordo com o timeframe:
      - 1h  -> 10  (meio dia)
      - 15m -> 40  (10 horas = 40 candles de 15min)
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

    O lookback deve ser passado pelo chamador de acordo com o timeframe:
      - 1h  -> 120  (5 dias = 120 candles)
      - 15m -> 480  (5 dias = 480 candles)

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


def compute_indicators(df: pd.DataFrame, timeframe: str = "1h") -> pd.DataFrame:
    """
    Recebe um DataFrame com colunas ['open','high','low','close','volume']
    e devolve uma copia enriquecida com os indicadores da estrategia CTEV v4.

    Parameters:
        df: DataFrame OHLCV
        timeframe: string do timeframe ('1m','5m','15m','30m','1h','2h','4h','1d')
            Usado para escalar lookbacks de Fibonacci, swing detection,
            ATR percentile e BB squeeze percentile.

    Indicadores calculados:
        - ema20: EMA de 20 periodos (pullback reference)
        - ema50: EMA de 50 periodos (filtro medio de tendencia)
        - ema200: EMA de 200 periodos (filtro lento de tendencia)
        - ema50_slope: Inclinacao da EMA50 (regime detection)
        - bb_lower, bb_middle, bb_upper: Bollinger Bands(20, 2.0)
        - bb_width: Largura das bandas Bollinger (%)
        - bb_squeeze_pct: Percentil da BB width (squeeze detection)
        - rsi: RSI(14)
        - rsi_delta: RSI atual - RSI anterior (momentum)
        - volume_sma20: Media de volume 20 periodos
        - volume_sma50: Media de volume 50 periodos (filtro soft)
        - atr: ATR(14)
        - atr_percentile: Ranking do ATR nos ultimos N candles
        - macd, macd_signal, macd_hist: MACD(12, 26, 9)
        - adx, plus_di, minus_di: ADX(14) + directional indicators
        - regime: Classificacao do mercado (trending_up/down, ranging, etc.)
        - fib_0236, fib_0382, fib_0500, fib_0618, fib_0786: Fibonacci levels
        - fib_direction: Direcao do Fibonacci (1=uptrend, -1=downtrend)
        - fib_proximity: Distancia % ao nivel Fibonacci mais proximo
        - ema20_touched: True se low <= EMA20 (pullback profundo)
        - ema50_touched: True se low <= EMA50 (pullback para LONG)
        - ema50_touched_up: True se high >= EMA50 (pullback para SHORT)
    """
    # Lookback scaling: base reference is 1h
    # Em 1h: swing_lookback=10, fib_lookback=120, atr/bb_lookback=100
    # These represent ~10h, ~5 days, ~4 days respectively
    _tf_multiplier = _get_timeframe_multiplier(timeframe)

    _swing_lookback = int(10 * _tf_multiplier)    # 10h em candles
    _fib_lookback = int(120 * _tf_multiplier)     # 5 dias em candles
    _atr_lookback = int(100 * _tf_multiplier)      # ~4 dias em candles
    _bb_width_lookback = int(100 * _tf_multiplier) # ~4 dias em candles

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

    # ── EMA 20 (pullback reference — recomendado pelo PDF estudo) ──
    out["ema20"] = _ema(out["close"], span=20)

    # ── EMA 50 (filtro medio de tendencia) ──
    out["ema50"] = _ema(out["close"], span=50)

    # ── EMA 200 (filtro lento de tendencia) ──
    out["ema200"] = _ema(out["close"], span=200)

    # ── EMA 50 Slope (regime detection — baseado no PDF) ──
    out["ema50_slope"] = _ema_slope(out["close"], slope_period=20)

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

    # ── Volume SMA 50 (filtro soft — recomendado pelo PDF: SMA(Volume,50)) ──
    out["volume_sma50"] = _sma(out["volume"], window=50)

    # ── ATR 14 ──
    out["atr"] = _atr(out["high"], out["low"], out["close"], period=14)

    # ── ATR Percentile (lookback adaptativo ao timeframe) ──
    out["atr_percentile"] = out["atr"].rolling(_atr_lookback).apply(
        lambda x: (x.iloc[-1] > x).sum() / max(len(x) - 1, 1), raw=False
    )

    # ── Bollinger Bandwidth ──
    out["bb_width"] = ((out["bb_upper"] - out["bb_lower"]) / out["bb_middle"]) * 100

    # ── BB Squeeze Percentile (lookback adaptativo ao timeframe) ──
    out["bb_squeeze_pct"] = out["bb_width"].rolling(_bb_width_lookback).apply(
        lambda x: (x.iloc[-1] > x).sum() / max(len(x) - 1, 1), raw=False
    )

    # ── MACD (12, 26, 9) ──
    out["macd"], out["macd_signal"], out["macd_hist"] = _macd(out["close"])

    # ── ADX(14) + DI+/DI- (v4: regime detection — CRITICO segundo o PDF) ──
    out["adx"], out["plus_di"], out["minus_di"] = _adx(
        out["high"], out["low"], out["close"], period=14
    )

    # ── Regime Classification (v4: trending_up, trending_down, ranging, etc.) ──
    out["regime"] = _classify_regime(
        out["adx"], out["plus_di"], out["minus_di"],
        out["close"], out["ema50"], out["bb_width"], out["bb_squeeze_pct"],
    )

    # ── Fibonacci Retracement Levels (lookback adaptativo ao timeframe) ──
    swing_highs, swing_lows = _detect_swing_points(out["high"], out["low"], lookback=_swing_lookback)
    fib_df = _compute_fibonacci_levels(
        out["high"], out["low"], out["close"],
        swing_highs, swing_lows, lookback=_fib_lookback
    )
    out = pd.concat([out, fib_df], axis=1)

    # ── On-Balance Volume (v10: OBV direction filter per PDF recommendation) ──
    obv_values = np.zeros(len(out))
    for i in range(1, len(out)):
        if out["close"].iloc[i] > out["close"].iloc[i-1]:
            obv_values[i] = obv_values[i-1] + out["volume"].iloc[i]
        elif out["close"].iloc[i] < out["close"].iloc[i-1]:
            obv_values[i] = obv_values[i-1] - out["volume"].iloc[i]
        else:
            obv_values[i] = obv_values[i-1]
    out["obv"] = obv_values
    out["obv_sma20"] = out["obv"].rolling(20).mean()
    out["obv_trend"] = 0
    out.loc[out["obv"] > out["obv_sma20"], "obv_trend"] = 1
    out.loc[out["obv"] < out["obv_sma20"], "obv_trend"] = -1

    # ── EMA(20) Touch Detection (pullback profundo) ──
    out["ema20_touched"] = out["low"] <= out["ema20"]

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
        "ema20", "ema50", "ema200", "rsi", "rsi_delta",
        "volume_sma20", "volume_sma50", "atr", "atr_percentile",
        "macd", "macd_signal", "macd_hist",
        "adx", "plus_di", "minus_di", "regime",
    ]
    missing = [c for c in critical if pd.isna(last.get(c))]
    if missing:
        logger.warning(
            "Ultimo candle possui NaN nos indicadores: %s. Aguardando mais dados.",
            missing,
        )
        return None
    return last
