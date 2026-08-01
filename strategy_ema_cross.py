"""
strategy_ema_cross.py
---------------------
Estrategia EMA Cross v8 para timeframes intraday (15m/30m).

Validada via optimize_15m_v8.py: 150 trades, WR 50%, PF 1.13, PnL +5.97%, DD 3.53%.
Supera Buy & Hold (-5.36%) em +11.33pp.

Diferenca fundamental do CTEV trend-following:
  - CTEV: pullback em tendencia (segue a tendencia, entra no pullback)
  - EMA Cross: cruza EMA20/50 como sinal primario (captura viradas)

Logica de entrada:
  LONG:  EMA20 cruza ACIMA da EMA50 + RSI delta > 0 + ADX > 15 + RSI em [30,80]
  SHORT: EMA20 cruza ABAIXO da EMA50 + RSI delta < 0 + ADX > 15 + RSI em [20,70]

Filtros:
  - NAO usa EMA200 (otimizacao mostrou que piora resultados)
  - Cooldown de 12 candles apos cada sinal (evita overtrading)
  - Regime trending_down bloqueia LONGs, trending_up bloqueia SHORTs

Gestao de risco (do profile INTRADAY):
  - SL = 2.0x ATR, TP = 2.5x ATR (R:R 1.25:1)
  - Max bars held: 48 (12h em 15m)

CRITICO: Lucrativo SOMENTE com limit orders (maker fee ~0.016%).
  Com custos de market order (0.325%/lado), a estrategia nao e lucrativa.

Integration:
  Esta estrategia e usada pelo strategy_router.py quando o timeframe
  ativo pertence ao perfil INTRADAY (15m, 30m).
"""
from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from strategy import Signal, SignalType

if TYPE_CHECKING:
    from strategy_profiles import StrategyProfile

logger = logging.getLogger(__name__)


# ==================================================================
# PARAMETROS OTIMIZADOS v8 (do winner_15m_v8.json)
# ==================================================================

EMA_CROSS_PARAMS = {
    "sl_atr_mult": 2.0,       # 2.0x ATR
    "tp_atr_mult": 2.5,       # 2.5x ATR (R:R 1.25:1)
    "adx_min": 15.0,          # ADX relaxado — foco em momentum
    "use_ema200": False,      # NAO usa EMA200 (piora resultados)
    "rsi_long_min": 30.0,     # RSI min para longs
    "rsi_long_max": 80.0,     # RSI max para longs (largo)
    "rsi_short_min": 20.0,    # RSI min para shorts (largo)
    "rsi_short_max": 70.0,    # RSI max para shorts
    "cooldown": 12,           # 12 candles sem entrar apos sinal
    "max_bars_held": 48,      # 48 candles = 12h (15m)
}


# ==================================================================
# COOLDOWN STATE (thread-safe via module variable)
# ==================================================================

_last_signal_bar: int = -999  # Global cooldown tracker (index of last signal)


def reset_cooldown() -> None:
    """Reseta o cooldown tracker (usado em testes/backtests)."""
    global _last_signal_bar
    _last_signal_bar = -999


def _check_cooldown(current_idx: int, cooldown: int) -> bool:
    """Verifica se cooldown foi respeitado."""
    global _last_signal_bar
    return (current_idx - _last_signal_bar) >= cooldown


def _register_signal(current_idx: int) -> None:
    """Registra um sinal para cooldown."""
    global _last_signal_bar
    _last_signal_bar = current_idx


# ==================================================================
# EMA CROSS SIGNAL GENERATION
# ==================================================================

def evaluate_ema_cross(
    df: pd.DataFrame,
    profile: Optional[StrategyProfile] = None,
) -> Optional[Signal]:
    """
    Avalia o ultimo candle do DataFrame para sinal EMA Cross.

    Esta funcao e compativel com a interface do bot_worker e do backtest.
    Recebe o DataFrame completo e avalia apenas a ultima linha.

    Parameters:
        df: DataFrame com indicadores calculados (compute_indicators)
        profile: StrategyProfile INTRADAY (para SL/TP)

    Returns:
        Signal ou None
    """
    if len(df) < 2:
        return None

    # Parametros
    if profile:
        sl_mult = profile.sl_atr_mult
        tp_mult = profile.tp_atr_mult
    else:
        sl_mult = EMA_CROSS_PARAMS["sl_atr_mult"]
        tp_mult = EMA_CROSS_PARAMS["tp_atr_mult"]

    adx_min = EMA_CROSS_PARAMS["adx_min"]
    cooldown = EMA_CROSS_PARAMS["cooldown"]

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    idx = len(df) - 1

    # Cooldown check
    if not _check_cooldown(idx, cooldown):
        return None

    # Extrair valores
    close = float(curr["close"])
    ema20 = float(curr["ema20"])
    ema50 = float(curr["ema50"])
    ema200 = float(curr.get("ema200", 0.0))
    ema20_prev = float(prev["ema20"])
    ema50_prev = float(prev["ema50"])
    rsi = float(curr["rsi"])
    rsi_delta = float(curr.get("rsi_delta", 0.0))
    adx = float(curr.get("adx", 0.0))
    atr = float(curr["atr"])
    atr_pct = float(curr.get("atr_percentile", 0.5))
    regime = str(curr.get("regime", ""))
    regime_v2 = str(curr.get("regime_v2", regime))
    bb_lower = float(curr.get("bb_lower", 0.0))
    bb_upper = float(curr.get("bb_upper", 0.0))
    bb_width = float(curr.get("bb_width", 0.0))
    bb_squeeze_pct = float(curr.get("bb_squeeze_pct", 0.5))
    volume = float(curr["volume"])
    volume_sma20 = float(curr.get("volume_sma20", 0.0))
    volume_sma50 = float(curr.get("volume_sma50", 0.0))
    macd_hist = float(curr.get("macd_hist", 0.0))
    plus_di = float(curr.get("plus_di", 0.0))
    minus_di = float(curr.get("minus_di", 0.0))
    ema50_slope = float(curr.get("ema50_slope", 0.0))
    fib_0382 = float(curr.get("fib_0382", float("nan")))
    fib_0500 = float(curr.get("fib_0500", float("nan")))
    fib_0618 = float(curr.get("fib_0618", float("nan")))
    fib_dir = int(curr.get("fib_direction", 0))
    fib_prox = float(curr.get("fib_proximity", float("nan")))
    ts = curr.name

    # Validacoes basicas
    if atr <= 0 or close <= 0:
        return None

    # ---- LONG: EMA20 cruza ACIMA da EMA50 ----
    if (ema20 > ema50 and ema20_prev <= ema50_prev):
        # Filtro: bloquear em trending_down
        if "down" in regime_v2.lower() or "down" in regime.lower():
            return None

        # Filtro: ADX minimo
        if adx_min > 0 and adx < adx_min:
            return None

        # Filtro: EMA200 (desativado por padrao)
        if EMA_CROSS_PARAMS["use_ema200"] and close < ema200:
            return None

        # Filtro: RSI delta positivo (momentum)
        if rsi_delta <= 0:
            return None

        # Filtro: RSI range
        if not (EMA_CROSS_PARAMS["rsi_long_min"] <= rsi <= EMA_CROSS_PARAMS["rsi_long_max"]):
            return None

        # Calcular SL/TP
        stop_loss = close - sl_mult * atr
        take_profit = close + tp_mult * atr

        if stop_loss <= 0:
            return None

        # Registrar cooldown
        _register_signal(idx)

        logger.info(
            "SIGNAL EMA CROSS LONG | entry=%.2f SL=%.2f TP=%.2f ATR=%.2f "
            "RSI=%.1f ADX=%.1f RSI_delta=%.2f regime=%s",
            close, stop_loss, take_profit, atr, rsi, adx, rsi_delta, regime_v2,
        )

        return Signal(
            type=SignalType.LONG,
            entry_price=close, stop_loss=stop_loss, take_profit=take_profit,
            atr=atr, rsi=rsi, rsi_delta=rsi_delta, macd_hist=macd_hist,
            ema20=ema20, ema50=ema50, ema200=ema200,
            adx=adx, plus_di=plus_di, minus_di=minus_di, regime=regime_v2,
            bb_lower=bb_lower, bb_upper=bb_upper, bb_width=bb_width,
            bb_squeeze_pct=bb_squeeze_pct,
            volume=volume, volume_sma20=volume_sma20, volume_sma50=volume_sma50,
            atr_percentile=atr_pct,
            fib_0382=fib_0382 if not np.isnan(fib_0382) else 0.0,
            fib_0500=fib_0500 if not np.isnan(fib_0500) else 0.0,
            fib_0618=fib_0618 if not np.isnan(fib_0618) else 0.0,
            fib_direction=fib_dir, fib_proximity=fib_prox,
            pullback_type="ema_cross",
            ema50_slope=ema50_slope, timestamp=ts,
        )

    # ---- SHORT: EMA20 cruza ABAIXO da EMA50 ----
    if (ema20 < ema50 and ema20_prev >= ema50_prev):
        # Filtro: bloquear em trending_up
        if "up" in regime_v2.lower() or "up" in regime.lower():
            return None

        # Filtro: ADX minimo
        if adx_min > 0 and adx < adx_min:
            return None

        # Filtro: EMA200 (desativado por padrao)
        if EMA_CROSS_PARAMS["use_ema200"] and close > ema200:
            return None

        # Filtro: RSI delta negativo (momentum)
        if rsi_delta >= 0:
            return None

        # Filtro: RSI range
        if not (EMA_CROSS_PARAMS["rsi_short_min"] <= rsi <= EMA_CROSS_PARAMS["rsi_short_max"]):
            return None

        # Calcular SL/TP
        stop_loss = close + sl_mult * atr
        take_profit = close - tp_mult * atr

        # Registrar cooldown
        _register_signal(idx)

        logger.info(
            "SIGNAL EMA CROSS SHORT | entry=%.2f SL=%.2f TP=%.2f ATR=%.2f "
            "RSI=%.1f ADX=%.1f RSI_delta=%.2f regime=%s",
            close, stop_loss, take_profit, atr, rsi, adx, rsi_delta, regime_v2,
        )

        return Signal(
            type=SignalType.SHORT,
            entry_price=close, stop_loss=stop_loss, take_profit=take_profit,
            atr=atr, rsi=rsi, rsi_delta=rsi_delta, macd_hist=macd_hist,
            ema20=ema20, ema50=ema50, ema200=ema200,
            adx=adx, plus_di=plus_di, minus_di=minus_di, regime=regime_v2,
            bb_lower=bb_lower, bb_upper=bb_upper, bb_width=bb_width,
            bb_squeeze_pct=bb_squeeze_pct,
            volume=volume, volume_sma20=volume_sma20, volume_sma50=volume_sma50,
            atr_percentile=atr_pct,
            fib_0382=fib_0382 if not np.isnan(fib_0382) else 0.0,
            fib_0500=fib_0500 if not np.isnan(fib_0500) else 0.0,
            fib_0618=fib_0618 if not np.isnan(fib_0618) else 0.0,
            fib_direction=fib_dir, fib_proximity=fib_prox,
            pullback_type="ema_cross",
            ema50_slope=ema50_slope, timestamp=ts,
        )

    return None


def evaluate_ema_cross_row(
    row: pd.Series,
    prev_row: pd.Series,
    bar_index: int,
    profile: Optional[StrategyProfile] = None,
) -> Optional[Signal]:
    """
    Avalia uma linha individual para sinal EMA Cross.

    Versao para uso no backtest (loop candle-a-candle).

    Parameters:
        row: linha atual do DataFrame
        prev_row: linha anterior (para detectar cruzamento)
        bar_index: indice absoluto no DataFrame (para cooldown)
        profile: StrategyProfile INTRADAY (para SL/TP)

    Returns:
        Signal ou None
    """
    # Parametros
    if profile:
        sl_mult = profile.sl_atr_mult
        tp_mult = profile.tp_atr_mult
    else:
        sl_mult = EMA_CROSS_PARAMS["sl_atr_mult"]
        tp_mult = EMA_CROSS_PARAMS["tp_atr_mult"]

    adx_min = EMA_CROSS_PARAMS["adx_min"]
    cooldown = EMA_CROSS_PARAMS["cooldown"]

    # Cooldown check
    if not _check_cooldown(bar_index, cooldown):
        return None

    # Extrair valores
    close = float(row["close"])
    ema20 = float(row["ema20"])
    ema50 = float(row["ema50"])
    ema200 = float(row.get("ema200", 0.0))
    ema20_prev = float(prev_row["ema20"])
    ema50_prev = float(prev_row["ema50"])
    rsi = float(row["rsi"])
    rsi_delta = float(row.get("rsi_delta", 0.0))
    adx = float(row.get("adx", 0.0))
    atr = float(row["atr"])
    atr_pct = float(row.get("atr_percentile", 0.5))
    regime = str(row.get("regime", ""))
    regime_v2 = str(row.get("regime_v2", regime))
    bb_lower = float(row.get("bb_lower", 0.0))
    bb_upper = float(row.get("bb_upper", 0.0))
    bb_width = float(row.get("bb_width", 0.0))
    bb_squeeze_pct = float(row.get("bb_squeeze_pct", 0.5))
    volume = float(row["volume"])
    volume_sma20 = float(row.get("volume_sma20", 0.0))
    volume_sma50 = float(row.get("volume_sma50", 0.0))
    macd_hist = float(row.get("macd_hist", 0.0))
    plus_di = float(row.get("plus_di", 0.0))
    minus_di = float(row.get("minus_di", 0.0))
    ema50_slope = float(row.get("ema50_slope", 0.0))
    fib_0382 = float(row.get("fib_0382", float("nan")))
    fib_0500 = float(row.get("fib_0500", float("nan")))
    fib_0618 = float(row.get("fib_0618", float("nan")))
    fib_dir = int(row.get("fib_direction", 0))
    fib_prox = float(row.get("fib_proximity", float("nan")))
    ts = row.name

    if atr <= 0 or close <= 0:
        return None

    # ---- LONG ----
    if (ema20 > ema50 and ema20_prev <= ema50_prev):
        if "down" in regime_v2.lower() or "down" in regime.lower():
            return None
        if adx_min > 0 and adx < adx_min:
            return None
        if EMA_CROSS_PARAMS["use_ema200"] and close < ema200:
            return None
        if rsi_delta <= 0:
            return None
        if not (EMA_CROSS_PARAMS["rsi_long_min"] <= rsi <= EMA_CROSS_PARAMS["rsi_long_max"]):
            return None

        stop_loss = close - sl_mult * atr
        take_profit = close + tp_mult * atr
        if stop_loss <= 0:
            return None

        _register_signal(bar_index)

        return Signal(
            type=SignalType.LONG,
            entry_price=close, stop_loss=stop_loss, take_profit=take_profit,
            atr=atr, rsi=rsi, rsi_delta=rsi_delta, macd_hist=macd_hist,
            ema20=ema20, ema50=ema50, ema200=ema200,
            adx=adx, plus_di=plus_di, minus_di=minus_di, regime=regime_v2,
            bb_lower=bb_lower, bb_upper=bb_upper, bb_width=bb_width,
            bb_squeeze_pct=bb_squeeze_pct,
            volume=volume, volume_sma20=volume_sma20, volume_sma50=volume_sma50,
            atr_percentile=atr_pct,
            fib_0382=fib_0382 if not np.isnan(fib_0382) else 0.0,
            fib_0500=fib_0500 if not np.isnan(fib_0500) else 0.0,
            fib_0618=fib_0618 if not np.isnan(fib_0618) else 0.0,
            fib_direction=fib_dir, fib_proximity=fib_prox,
            pullback_type="ema_cross", ema50_slope=ema50_slope, timestamp=ts,
        )

    # ---- SHORT ----
    if (ema20 < ema50 and ema20_prev >= ema50_prev):
        if "up" in regime_v2.lower() or "up" in regime.lower():
            return None
        if adx_min > 0 and adx < adx_min:
            return None
        if EMA_CROSS_PARAMS["use_ema200"] and close > ema200:
            return None
        if rsi_delta >= 0:
            return None
        if not (EMA_CROSS_PARAMS["rsi_short_min"] <= rsi <= EMA_CROSS_PARAMS["rsi_short_max"]):
            return None

        stop_loss = close + sl_mult * atr
        take_profit = close - tp_mult * atr

        _register_signal(bar_index)

        return Signal(
            type=SignalType.SHORT,
            entry_price=close, stop_loss=stop_loss, take_profit=take_profit,
            atr=atr, rsi=rsi, rsi_delta=rsi_delta, macd_hist=macd_hist,
            ema20=ema20, ema50=ema50, ema200=ema200,
            adx=adx, plus_di=plus_di, minus_di=minus_di, regime=regime_v2,
            bb_lower=bb_lower, bb_upper=bb_upper, bb_width=bb_width,
            bb_squeeze_pct=bb_squeeze_pct,
            volume=volume, volume_sma20=volume_sma20, volume_sma50=volume_sma50,
            atr_percentile=atr_pct,
            fib_0382=fib_0382 if not np.isnan(fib_0382) else 0.0,
            fib_0500=fib_0500 if not np.isnan(fib_0500) else 0.0,
            fib_0618=fib_0618 if not np.isnan(fib_0618) else 0.0,
            fib_direction=fib_dir, fib_proximity=fib_prox,
            pullback_type="ema_cross", ema50_slope=ema50_slope, timestamp=ts,
        )

    return None
