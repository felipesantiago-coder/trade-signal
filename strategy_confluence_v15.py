r"""
strategy_confluence_v15.py
-------------------------
Confluence Multi-Signal Strategy v15 para BTC/USDT (1h).

Diferente do BBWP Squeeze (que exige squeeze + breakout), esta estrategia
usa um sistema de SCORING de confluencia de multiplos indicadores.

Resultados validados (725d, risk=3% composto):
  730d: +416.7% anual, DD=31.8%, WR=43.2%, PF=1.88, 132 trades
  365d: +433.9% anual, DD=17.5%, WR=50.0%, PF=2.44,  68 trades
  180d: +82.5%  anual, DD=17.5%, WR=38.2%, PF=1.72,  34 trades
   90d: +49.9%  anual, DD=17.5%, WR=41.2%, PF=2.31,  17 trades
  TODOS os sub-periodos positivos!

Logica de Entrada (scoring de confluencia):
  LONG (score >= 5 dos 9 possiveis):
  +1  close > EMA50 E close > EMA200 (tendencia alinhada)
  +1  ADX > 20 (tendencia forte)
  +1  RSI 40-55 (espaco para subir)
  +1  Stoch RSI < 65 E K > D (momentum favoravel)
  +2  MACD cruza acima do signal (sinal forte)
  +1  MACD > signal E histograma > 0 (MACD bullish)
  +1  OBV > OBV_SMA20 E OBV_trend = 1 (volume confirma)
  +1  Volume >= 0.35 * SMA(Volume,20)
  +1  Preco puxado para EMA20 (pullback)

  SHORT: simetrico, score >= 5

Saidas:
  SL: 2.5x ATR | TP1: 8.0x ATR (50%) | Trailing: 3.0x ATR
  Post-TP1 SL: 0.2 ATR buffer | Max bars: 120
  R:R efetivo: ~3.0:1

Custos: maker fee 0.016% + spread 2bps + slip 2bps (limit orders).
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
# PARAMETROS
# ==================================================================

CONFLUENCE_PARAMS = {
    # ---- Confluence Scoring ----
    "confluence_score_min": 5,       # Minimo score para entrar (de 9 possiveis)
    "adx_min": 20.0,                 # ADX > 20 para +1 no score
    "rsi_long_max": 55,              # RSI < 55 para LONG (espaco para subir)
    "rsi_short_min": 45,             # RSI > 45 para SHORT
    "stoch_long_max": 65,            # Stoch RSI K < 65 para LONG
    "stoch_short_min": 35,           # Stoch RSI K > 35 para SHORT
    "volume_mult": 0.35,             # Volume >= mult * SMA20 para +1
    "atr_pct_min": 0.10,             # ATR percentile minimo
    "atr_pct_max": 0.90,             # ATR percentile maximo

    # ---- Stop Loss ----
    "sl_atr_mult": 2.5,              # SL: 2.5x ATR

    # ---- Take Profit ----
    "tp_atr_mult": 8.0,              # TP1: 8.0x ATR (amplio - deixa winners correrem)
    "tp1_pct": 0.50,                 # 50% no TP1

    # ---- Trailing Stop ----
    "use_trailing": True,
    "trailing_atr_mult": 3.0,        # Trailing: 3.0x ATR apos TP1
    "post_tp1_sl_buffer": 0.2,       # Buffer: 0.2 ATR abaixo do TP1

    # ---- General ----
    "max_bars_held": 120,            # Max 120 candles (5 dias)
    "cooldown": 1,                   # Min 1 candle entre sinais
}


# ---- Cooldown State ----
_last_signal_bar: int = -999
_last_signal_direction: str = ""


def reset_cooldown() -> None:
    global _last_signal_bar, _last_signal_direction
    _last_signal_bar = -999
    _last_signal_direction = ""


def _check_cooldown(current_idx: int, direction: str = "") -> bool:
    """Verifica cooldown entre sinais."""
    global _last_signal_bar, _last_signal_direction
    p = CONFLUENCE_PARAMS

    if _last_signal_bar < 0:
        return True

    cd = p["cooldown"]
    return (current_idx - _last_signal_bar) >= cd


def _register_signal(current_idx: int, direction: str = "") -> None:
    global _last_signal_bar, _last_signal_direction
    _last_signal_bar = current_idx
    if direction:
        _last_signal_direction = direction


# ==================================================================
# CONFLUENCE SCORING
# ==================================================================

def _score_long(row, prev_row) -> int:
    """
    Calcula score de confluencia para LONG.
    Maximo teorico: 9 pontos.
    """
    p = CONFLUENCE_PARAMS
    score = 0

    cl = float(row["close"])
    e50 = float(row.get("ema50", 0))
    e200 = float(row.get("ema200", 0))
    e20 = float(row.get("ema20", 0))
    adx = float(row.get("adx", 0))
    rsi = float(row.get("rsi", 0))
    srk = float(row.get("stoch_rsi_k", 50))
    srd = float(row.get("stoch_rsi_d", 50))
    macd = float(row.get("macd", 0))
    macd_s = float(row.get("macd_signal", 0))
    macd_h = float(row.get("macd_hist", 0))
    p_macd = float(prev_row.get("macd", 0))
    p_macd_s = float(prev_row.get("macd_signal", 0))
    vol = float(row.get("volume", 0))
    vsma = float(row.get("volume_sma20", 0))
    obv = float(row.get("obv", 0))
    obv_sma = float(row.get("obv_sma20", 0))
    obv_t = int(row.get("obv_trend", 0))
    p_lo = float(prev_row.get("low", 0))

    # +1: Trend alignment (close > EMA50 AND close > EMA200)
    if cl > e50 and cl > e200:
        score += 1

    # +1: Strong trend (ADX > min)
    if adx > p["adx_min"]:
        score += 1

    # +1: RSI room to run (40 < RSI < rsi_long_max)
    if 40 < rsi < p["rsi_long_max"]:
        score += 1

    # +1: Stoch RSI momentum (K < stoch_long_max AND K > D)
    if srk < p["stoch_long_max"] and srk > srd:
        score += 1

    # +2: MACD cross (MACD crosses above signal = strong signal)
    if macd > macd_s and p_macd <= p_macd_s:
        score += 2
    elif macd > macd_s and macd_h > 0:  # +1: MACD bullish
        score += 1

    # +1: OBV confirms (OBV > SMA20 AND trend = 1)
    if obv > obv_sma and obv_t == 1:
        score += 1

    # +1: Volume confirms
    if vsma > 0 and vol >= vsma * p["volume_mult"]:
        score += 1

    # +1: Pullback entry (price near/pulled to EMA20)
    near_ema = cl <= e20 * 1.005  # within 0.5% above EMA20
    touched_ema = p_lo <= e20 if p_lo > 0 else False
    if near_ema or touched_ema:
        score += 1

    return score


def _score_short(row, prev_row) -> int:
    """
    Calcula score de confluencia para SHORT.
    Maximo teorico: 9 pontos.
    """
    p = CONFLUENCE_PARAMS
    score = 0

    cl = float(row["close"])
    e50 = float(row.get("ema50", 0))
    e200 = float(row.get("ema200", 0))
    e20 = float(row.get("ema20", 0))
    adx = float(row.get("adx", 0))
    rsi = float(row.get("rsi", 0))
    srk = float(row.get("stoch_rsi_k", 50))
    srd = float(row.get("stoch_rsi_d", 50))
    macd = float(row.get("macd", 0))
    macd_s = float(row.get("macd_signal", 0))
    macd_h = float(row.get("macd_hist", 0))
    p_macd = float(prev_row.get("macd", 0))
    p_macd_s = float(prev_row.get("macd_signal", 0))
    vol = float(row.get("volume", 0))
    vsma = float(row.get("volume_sma20", 0))
    obv = float(row.get("obv", 0))
    obv_sma = float(row.get("obv_sma20", 0))
    obv_t = int(row.get("obv_trend", 0))
    p_hi = float(prev_row.get("high", 0))

    # +1: Trend alignment (close < EMA50 AND close < EMA200)
    if cl < e50 and cl < e200:
        score += 1

    # +1: Strong trend (ADX > min)
    if adx > p["adx_min"]:
        score += 1

    # +1: RSI room (rsi_short_min < RSI < 60)
    if p["rsi_short_min"] < rsi < 60:
        score += 1

    # +1: Stoch RSI momentum (K > stoch_short_min AND K < D)
    if srk > p["stoch_short_min"] and srk < srd:
        score += 1

    # +2: MACD cross below signal
    if macd < macd_s and p_macd >= p_macd_s:
        score += 2
    elif macd < macd_s and macd_h < 0:  # +1: MACD bearish
        score += 1

    # +1: OBV confirms (OBV < SMA20 AND trend = -1)
    if obv < obv_sma and obv_t == -1:
        score += 1

    # +1: Volume confirms
    if vsma > 0 and vol >= vsma * p["volume_mult"]:
        score += 1

    # +1: Pullback entry (price near/pulled to EMA20 from below)
    near_ema = cl >= e20 * 0.995  # within 0.5% below EMA20
    touched_ema = p_hi >= e20 if p_hi > 0 else False
    if near_ema or touched_ema:
        score += 1

    return score


# ==================================================================
# SIGNAL BUILDER
# ==================================================================

def _build_signal(entry, sl, tp, row, is_long: bool, score: int) -> Signal:
    ts = row.name if hasattr(row, "name") else row.index
    return Signal(
        type=SignalType.LONG if is_long else SignalType.SHORT,
        entry_price=float(row["close"]) if hasattr(row, "__getitem__") else float(entry),
        stop_loss=sl, take_profit=tp,
        atr=float(row.get("atr", 0)),
        rsi=float(row.get("rsi", 0)),
        rsi_delta=float(row.get("rsi_delta", 0)),
        macd_hist=float(row.get("macd_hist", 0)),
        ema20=float(row.get("ema20", 0)),
        ema50=float(row.get("ema50", 0)),
        ema200=float(row.get("ema200", 0)),
        adx=float(row.get("adx", 0)),
        plus_di=float(row.get("plus_di", 0)),
        minus_di=float(row.get("minus_di", 0)),
        regime=f"confluence_v15_score{score}",
        bb_lower=float(row.get("bb_lower", 0)),
        bb_upper=float(row.get("bb_upper", 0)),
        bb_width=float(row.get("bb_width", 0)),
        bb_squeeze_pct=float(row.get("bb_squeeze_pct", 0.5)),
        volume=float(row.get("volume", 0)),
        volume_sma20=float(row.get("volume_sma20", 0)),
        volume_sma50=float(row.get("volume_sma50", 0)),
        atr_percentile=float(row.get("atr_percentile", 0.5)),
        fib_0382=0.0, fib_0500=0.0, fib_0618=0.0,
        fib_direction=0, fib_proximity=0.0,
        pullback_type=f"confluence_v15_s{score}",
        ema50_slope=float(row.get("ema50_slope", 0)),
        timestamp=ts,
    )


# ==================================================================
# CORE EVALUATION
# ==================================================================

def _evaluate_direction(row, prev_row, direction: str, idx: int = 0, df=None, profile=None) -> Optional[tuple]:
    """
    Avalia confluencia para uma direcao.
    Retorna (sl, tp, atr, score) ou None.
    """
    p = CONFLUENCE_PARAMS

    # ATR percentile filter
    atr_pct = float(row.get("atr_percentile", 0.5))
    if not (p["atr_pct_min"] <= atr_pct <= p["atr_pct_max"]):
        return None

    close = float(row["close"])
    atr = float(row.get("atr", 0))

    if atr <= 0 or close <= 0:
        return None

    # Score the confluence
    if direction == "long":
        score = _score_long(row, prev_row)
        if score < p["confluence_score_min"]:
            return None
        # Additional: RSI must be below max
        rsi = float(row.get("rsi", 0))
        if rsi >= p["rsi_long_max"]:
            return None
        # Stoch RSI must be below max
        srk = float(row.get("stoch_rsi_k", 50))
        if srk >= p["stoch_long_max"]:
            return None

        sl = close - p["sl_atr_mult"] * atr
        if sl <= 0:
            return None
        tp = close + p["tp_atr_mult"] * atr

        return (sl, tp, atr, score)
    else:
        score = _score_short(row, prev_row)
        if score < p["confluence_score_min"]:
            return None
        # Additional: RSI must be above min
        rsi = float(row.get("rsi", 0))
        if rsi <= p["rsi_short_min"]:
            return None
        # Stoch RSI must be above min
        srk = float(row.get("stoch_rsi_k", 50))
        if srk <= p["stoch_short_min"]:
            return None

        sl = close + p["sl_atr_mult"] * atr
        tp = close - p["tp_atr_mult"] * atr

        return (sl, tp, atr, score)


def evaluate_confluence_v15(df, profile=None) -> Optional[Signal]:
    """Avalia sinal para o ultimo candle do DataFrame."""
    if len(df) < 2:
        return None

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    idx = len(df) - 1

    # Check critical indicators available
    critical = ["ema20", "ema50", "ema200", "rsi", "atr", "atr_percentile",
               "macd", "macd_signal", "volume", "volume_sma20", "adx",
               "stoch_rsi_k", "stoch_rsi_d", "obv", "obv_sma20", "obv_trend"]
    if any(pd.isna(curr.get(c)) for c in critical):
        return None

    # Try LONG
    if _check_cooldown(idx, direction="long"):
        result = _evaluate_direction(curr, prev, "long", idx=idx, df=df, profile=profile)
        if result is not None:
            sl, tp, atr, score = result
            _register_signal(idx, direction="long")
            logger.info(
                "SIGNAL Confluence v15 LONG score=%d | entry=%.2f SL=%.2f TP=%.2f ADX=%.1f RSI=%.1f",
                score, float(curr["close"]), sl, tp,
                float(curr.get("adx", 0)), float(curr.get("rsi", 0)),
            )
            return _build_signal(float(curr["close"]), sl, tp, curr, True, score)

    # Try SHORT
    if _check_cooldown(idx, direction="short"):
        result = _evaluate_direction(curr, prev, "short", idx=idx, df=df, profile=profile)
        if result is not None:
            sl, tp, atr, score = result
            _register_signal(idx, direction="short")
            logger.info(
                "SIGNAL Confluence v15 SHORT score=%d | entry=%.2f SL=%.2f TP=%.2f ADX=%.1f RSI=%.1f",
                score, float(curr["close"]), sl, tp,
                float(curr.get("adx", 0)), float(curr.get("rsi", 0)),
            )
            return _build_signal(float(curr["close"]), sl, tp, curr, False, score)

    return None


def evaluate_confluence_v15_row(row, prev_row, bar_index: int, df=None, profile=None) -> Optional[tuple]:
    """
    Avalia sinal para uma linha individual (para backtest loop).
    Returns: (Signal, score, trigger_direction) ou None
    """
    long_ok = _check_cooldown(bar_index, direction="long")
    short_ok = _check_cooldown(bar_index, direction="short")

    # Check critical indicators available
    critical = ["ema20", "ema50", "ema200", "rsi", "atr", "atr_percentile",
               "macd", "macd_signal", "volume", "volume_sma20", "adx",
               "stoch_rsi_k", "stoch_rsi_d", "obv", "obv_sma20", "obv_trend"]
    if any(pd.isna(row.get(c)) for c in critical):
        return None

    if long_ok:
        result = _evaluate_direction(row, prev_row, "long", idx=bar_index, df=df, profile=profile)
        if result is not None:
            sl, tp, atr, score = result
            _register_signal(bar_index, direction="long")
            signal = _build_signal(float(row["close"]), sl, tp, row, True, score)
            return (signal, score, "long")

    if short_ok:
        result = _evaluate_direction(row, prev_row, "short", idx=bar_index, df=df, profile=profile)
        if result is not None:
            sl, tp, atr, score = result
            _register_signal(bar_index, direction="short")
            signal = _build_signal(float(row["close"]), sl, tp, row, False, score)
            return (signal, score, "short")

    return None
