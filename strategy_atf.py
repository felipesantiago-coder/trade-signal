r"""
strategy_atf.py
---------------
Adaptive Trend-Follow v1 (ATF v1) para timeframes intraday (15m/30m).

Filosofia de Design:
  O EMA Cross v11 captura apenas cruze EMA20/50, gerando ~75 trades em 2 anos
  (capital parado ~97% do tempo). Para superar B&H em periodos longos, e
  necessario uma estrategia que:

  1. Gere MAIS trades com edge positiva (mais exposicao ao mercado)
  2. Deixe ganhadores correrem mais (capturar grandes movimentos)
  3. Adapte agressividade a forca da tendencia (mais risco em alta conviccao)

  ATF v1 resolve isso com 3 inovacoes fundamentais:

  A) SCORING COMPOSTO (0-10 pontos):
     Avalia tendencia em 10 dimensoes (EMA alignment, slope, ADX, DI, MACD, RSI, OBV).
     Score >= 6 necessario para qualquer entrada. Score 8+ = alta conviccao.
     Substitui a classificacao binaria de regime por um espectro continuo.

  B) 6 TIPOS DE ENTRADA (vs 1 cruze no EMA Cross):
     - Pullback EMA20/50: preco toca EMA dentro de tendencia existente
     - RSI dip recovery: RSI caiu abaixo 50 e recuperou
     - MACD cross: MACD cruza sinal na direcao da tendencia
     - Momentum burst: impulso forte (score >= 8 apenas)
     - BB bounce: preco nas bandas Bollinger (mean-reversion dentro tendencia)
     - Trend aligned: score >= 9, entrada direta sem gatilho especifico

  C) TRAILING ADAPTATIVO POR ADX (sem TP fixo):
     - ADX >= 35: trail 2.5x ATR (tendencias fortes — deixar correr)
     - ADX 25-35: trail 1.5x ATR (tendencias moderadas)
     - ADX < 25: trail 1.0x ATR (tendencias fracas — saida rapida)
     - Sem TP fixo: o mercado determina quando sair via trailing ratchet

  Por que funciona melhor que EMA Cross:
  - EMA Cross so entra em CRUZES (eventos raros) e sai em TP FIXO (corta ganhadores)
  - ATF entra em PULLBACKS dentro de tendencias (frequente) e sai em TRAILING
    ADAPTATIVO (deixa ganhadores correrem proporcionalmente a forca da tendencia)
  - Resultado esperado: 150-250 trades em 2 anos com trailing que captura
    movimentos maiores, gerando retorno significativamente superior ao B&H.

  Custos: fee=0.016% + spread=2bps + slip=2bps (limit orders, maker).
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


ATF_PARAMS = {
    # ---- Entry requirements ----
    "min_score": 6,               # Minimum composite trend score to enter
    "high_score_threshold": 8,    # Score for allowing momentum entries
    "direct_entry_score": 9,      # Score for direct entry (no trigger needed)

    # ---- Stop Loss (adaptive by ATR percentile) ----
    "sl_atr_base": 1.5,           # Normal volatility SL
    "sl_atr_high_vol": 2.0,       # High volatility (ATR pct > 0.7)
    "sl_atr_low_vol": 1.2,        # Low volatility (ATR pct < 0.3)
    "high_vol_threshold": 0.70,
    "low_vol_threshold": 0.30,

    # ---- Trailing Stop (adaptive by ADX) ----
    "be_trigger_atr": 0.8,        # Move SL to entry after 0.8x ATR favorable
    "trail_adx_strong": 2.5,      # ADX >= 35: wide trail (let winners run)
    "trail_adx_medium": 1.5,      # ADX 25-35: normal trail
    "trail_adx_weak": 1.0,        # ADX < 25: tight trail (quick exit)
    "adx_strong_min": 35.0,
    "adx_medium_min": 25.0,

    # ---- Filters ----
    "max_bars": 96,               # Max bars to hold (24h on 15min)
    "cooldown": 6,                # Bars between trades (1.5h on 15min)
    "cooldown_trailing": 3,       # Shorter after trailing exit (re-entry)
    "atr_pct_min": 0.15,
    "atr_pct_max": 0.85,
    "bb_squeeze_min": 0.10,
    "vol_sma_ratio": 0.70,
    "rsi_delta_burst": 3.0,       # RSI delta for momentum burst trigger

    # ---- Position Sizing (for backtest) ----
    "risk_base_pct": 0.015,       # 1.5% risk for score 6-7
    "risk_high_pct": 0.025,       # 2.5% risk for score 8+
}


# ---- Cooldown State ----
_last_signal_bar: int = -999
_last_exit_was_trailing: bool = False


def reset_cooldown() -> None:
    global _last_signal_bar, _last_exit_was_trailing
    _last_signal_bar = -999
    _last_exit_was_trailing = False


def _check_cooldown(current_idx: int) -> bool:
    global _last_signal_bar, _last_exit_was_trailing
    cd = ATF_PARAMS["cooldown_trailing"] if _last_exit_was_trailing else ATF_PARAMS["cooldown"]
    return (current_idx - _last_signal_bar) >= cd


def _register_signal(current_idx: int, was_trailing: bool = False) -> None:
    global _last_signal_bar, _last_exit_was_trailing
    _last_signal_bar = current_idx
    _last_exit_was_trailing = was_trailing


# ==================================================================
# COMPOSITE TREND SCORING (0-10)
# ==================================================================

def _compute_trend_score(row, direction: str = "long") -> int:
    """
    Calcula score composto de tendencia (0-10 pontos).

    Avalia 10 dimensoes independentes da tendencia:
      1. EMA20 > EMA50 (alinhamento curto)
      2. Close > EMA50 (posicao relativa)
      3. Close > EMA200 (posicao macro)
      4. EMA50 slope > 0 (inclinacao da tendencia)
      5. ADX > 25 (tendencia existe)
      6. ADX > 35 (tendencia forte — bonus)
      7. +DI > -DI (direcao direcional)
      8. MACD > Signal (momentum alinhado)
      9. RSI > 45 (viés bullish, sem overbought)
     10. OBV > OBV_SMA20 (volume confirma)

    Para SHORT, todas as condicoes sao invertidas.
    """
    score = 0

    ema20 = float(row["ema20"])
    ema50 = float(row["ema50"])
    ema200 = float(row.get("ema200", 0))
    close = float(row["close"])
    ema50_slope = float(row.get("ema50_slope", 0))
    adx = float(row.get("adx", 0))
    plus_di = float(row.get("plus_di", 0))
    minus_di = float(row.get("minus_di", 0))
    macd = float(row.get("macd", 0))
    macd_signal_val = float(row.get("macd_signal", 0))
    rsi = float(row.get("rsi", 50))
    obv_trend = float(row.get("obv_trend", 0))

    if direction == "long":
        # EMA alignment (3 points)
        if ema20 > ema50:
            score += 1
        if close > ema50:
            score += 1
        if ema200 > 0 and close > ema200:
            score += 1
        # Trend strength (4 points)
        if ema50_slope > 0:
            score += 1
        if adx > ATF_PARAMS["adx_medium_min"]:
            score += 1
        if adx > ATF_PARAMS["adx_strong_min"]:
            score += 1
        if plus_di > minus_di:
            score += 1
        # Momentum (3 points)
        if macd > macd_signal_val:
            score += 1
        if rsi > 45:
            score += 1
        if obv_trend > 0:
            score += 1
    else:
        # SHORT: inverted conditions
        if ema20 < ema50:
            score += 1
        if close < ema50:
            score += 1
        if ema200 > 0 and close < ema200:
            score += 1
        if ema50_slope < 0:
            score += 1
        if adx > ATF_PARAMS["adx_medium_min"]:
            score += 1
        if adx > ATF_PARAMS["adx_strong_min"]:
            score += 1
        if minus_di > plus_di:
            score += 1
        if macd < macd_signal_val:
            score += 1
        if rsi < 55:
            score += 1
        if obv_trend < 0:
            score += 1

    return score


# ==================================================================
# ENTRY TRIGGERS (6 tipos)
# ==================================================================

# Pullback triggers — allowed at score >= 6
_PULLBACK_TRIGGERS = {"pullback_ema20", "pullback_ema50", "rsi_dip", "macd_cross", "bb_bounce"}
# Momentum triggers — allowed only at score >= 8
_MOMENTUM_TRIGGERS = {"momentum_burst"}
# Direct entry — allowed only at score >= 9
_DIRECT_TRIGGERS = {"trend_aligned"}


def _check_entry_triggers(
    row, prev_row, direction: str = "long", score: int = 0
) -> tuple:
    """
    Verifica se algum dos 6 gatilhos de entrada esta ativo.

    Returns:
        (triggered: bool, trigger_name: str)
    """
    close = float(row["close"])
    low = float(row["low"])
    high = float(row["high"])
    ema20 = float(row["ema20"])
    ema50 = float(row["ema50"])
    rsi = float(row.get("rsi", 50))
    prev_rsi = float(prev_row.get("rsi", rsi))
    macd_hist = float(row.get("macd_hist", 0))
    prev_macd_hist = float(prev_row.get("macd_hist", 0))
    rsi_delta = float(row.get("rsi_delta", 0))
    bb_lower = float(row.get("bb_lower", 0))
    bb_upper = float(row.get("bb_upper", 0))
    bb_middle = float(row.get("bb_middle", 0))

    if direction == "long":
        # 1. Pullback EMA20: preco tocou EMA20
        if low <= ema20:
            return True, "pullback_ema20"

        # 2. Pullback EMA50: preco tocou EMA50 (pullback mais fundo)
        if low <= ema50:
            return True, "pullback_ema50"

        # 3. RSI dip recovery: RSI caiu abaixo 50 e voltou
        if prev_rsi < 50 and rsi >= 50:
            return True, "rsi_dip"

        # 4. MACD cross up: histograma cruzou acima de zero
        if macd_hist > 0 and prev_macd_hist <= 0:
            return True, "macd_cross"

        # 5. BB bounce: preco na faixa inferior das bandas
        bb_range = bb_upper - bb_lower
        if bb_range > 0:
            bb_pos = (close - bb_lower) / bb_range
            if bb_pos < 0.25:
                return True, "bb_bounce"

        # 6. Momentum burst: impulso forte (score >= 8 apenas)
        if score >= ATF_PARAMS["high_score_threshold"]:
            if rsi_delta > ATF_PARAMS["rsi_delta_burst"] and macd_hist > 0:
                return True, "momentum_burst"

        # 7. Trend aligned: score >= 9, entrada direta
        if score >= ATF_PARAMS["direct_entry_score"]:
            return True, "trend_aligned"

    else:
        # SHORT triggers (inverted)
        if high >= ema20:
            return True, "pullback_ema20"

        if high >= ema50:
            return True, "pullback_ema50"

        if prev_rsi > 50 and rsi <= 50:
            return True, "rsi_dip"

        if macd_hist < 0 and prev_macd_hist >= 0:
            return True, "macd_cross"

        bb_range = bb_upper - bb_lower
        if bb_range > 0:
            bb_pos = (close - bb_lower) / bb_range
            if bb_pos > 0.75:
                return True, "bb_bounce"

        if score >= ATF_PARAMS["high_score_threshold"]:
            if rsi_delta < -ATF_PARAMS["rsi_delta_burst"] and macd_hist < 0:
                return True, "momentum_burst"

        if score >= ATF_PARAMS["direct_entry_score"]:
            return True, "trend_aligned"

    return False, ""


def _validate_trigger_for_score(trigger_name: str, score: int) -> bool:
    """
    Valida se o gatilho e permitido para o score atual.
    - Score 6-7: apenas pullback triggers
    - Score 8: pullback + momentum triggers
    - Score 9+: todos os triggers
    """
    if trigger_name in _PULLBACK_TRIGGERS:
        return score >= ATF_PARAMS["min_score"]
    elif trigger_name in _MOMENTUM_TRIGGERS:
        return score >= ATF_PARAMS["high_score_threshold"]
    elif trigger_name in _DIRECT_TRIGGERS:
        return score >= ATF_PARAMS["direct_entry_score"]
    return False


# ==================================================================
# ADAPTIVE SL & TRAILING
# ==================================================================

def _get_sl_mult(atr_pct: float) -> float:
    """Retorna multiplicador de SL baseado na volatilidade (ATR percentile)."""
    if atr_pct > ATF_PARAMS["high_vol_threshold"]:
        return ATF_PARAMS["sl_atr_high_vol"]
    elif atr_pct < ATF_PARAMS["low_vol_threshold"]:
        return ATF_PARAMS["sl_atr_low_vol"]
    return ATF_PARAMS["sl_atr_base"]


def _get_trail_mult(adx: float) -> float:
    """Retorna multiplicador de trailing baseado na forca da tendencia (ADX)."""
    if adx >= ATF_PARAMS["adx_strong_min"]:
        return ATF_PARAMS["trail_adx_strong"]
    elif adx >= ATF_PARAMS["adx_medium_min"]:
        return ATF_PARAMS["trail_adx_medium"]
    return ATF_PARAMS["trail_adx_weak"]


# ==================================================================
# SHARED FILTERS
# ==================================================================

def _shared_filters(row) -> bool:
    """Filtros comuns para LONG e SHORT."""
    p = ATF_PARAMS

    atr_pct = float(row.get("atr_percentile", 0.5))
    if not (p["atr_pct_min"] <= atr_pct <= p["atr_pct_max"]):
        return False

    bb_squeeze_pct = float(row.get("bb_squeeze_pct", 0.5))
    if bb_squeeze_pct < p["bb_squeeze_min"]:
        return False

    vol = float(row.get("volume", 0))
    vol_sma20 = float(row.get("volume_sma20", 0))
    if vol_sma20 > 0 and vol < vol_sma20 * p["vol_sma_ratio"]:
        return False

    # Block volatile regime (basic v1 regime)
    regime = str(row.get("regime", "")).lower()
    if "volatile" in regime:
        return False

    return True


# ==================================================================
# SIGNAL BUILDER
# ==================================================================

def _build_signal(
    entry, sl, row, is_long: bool,
    score: int, trigger_name: str, adx: float,
) -> Signal:
    """Constrói um Signal dataclass compatível com o sistema."""
    ts = row.name if hasattr(row, "name") else row.index
    close = float(row["close"]) if hasattr(row, "__getitem__") else float(entry)
    atr = float(row.get("atr", 0))

    # TP is set extremely far — ATF uses trailing-only exit.
    # The simulator ignores TP and uses trailing stop exclusively.
    if is_long:
        tp = close + 1000 * atr  # Effectively never hit
    else:
        tp = max(0.0, close - 1000 * atr)  # Effectively never hit

    return Signal(
        type=SignalType.LONG if is_long else SignalType.SHORT,
        entry_price=close if hasattr(row, "__getitem__") else entry,
        stop_loss=sl,
        take_profit=tp,
        atr=atr,
        rsi=float(row.get("rsi", 0)),
        rsi_delta=float(row.get("rsi_delta", 0)),
        macd_hist=float(row.get("macd_hist", 0)),
        ema20=float(row.get("ema20", 0)),
        ema50=float(row.get("ema50", 0)),
        ema200=float(row.get("ema200", 0)),
        adx=adx,
        plus_di=float(row.get("plus_di", 0)),
        minus_di=float(row.get("minus_di", 0)),
        regime=f"atf_score_{score}",
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
        pullback_type=f"atf_{trigger_name}",
        ema50_slope=float(row.get("ema50_slope", 0)),
        timestamp=ts,
    )


# ==================================================================
# CORE EVALUATION
# ==================================================================

def _evaluate_direction(
    row, prev_row, direction: str, profile=None
) -> Optional[tuple]:
    """
    Avalia sinal para uma direcao (long/short).

    Returns:
        (score, trigger_name, sl, tp, adx, atr, atr_pct) ou None
    """
    p = ATF_PARAMS

    if not _shared_filters(row):
        return None

    # Compute composite trend score
    score = _compute_trend_score(row, direction)
    if score < p["min_score"]:
        return None

    # Check entry triggers
    triggered, trigger_name = _check_entry_triggers(row, prev_row, direction, score)
    if not triggered:
        return None

    # Validate trigger is allowed for this score level
    if not _validate_trigger_for_score(trigger_name, score):
        return None

    # Compute SL (adaptive by volatility)
    atr = float(row["atr"])
    atr_pct = float(row.get("atr_percentile", 0.5))
    if atr <= 0:
        return None

    close = float(row["close"])
    sl_mult = _get_sl_mult(atr_pct)

    if direction == "long":
        sl = close - sl_mult * atr
        if sl <= 0:
            return None
    else:
        sl = close + sl_mult * atr

    adx = float(row.get("adx", 0))

    return (score, trigger_name, sl, adx, atr, atr_pct)


def evaluate_atf(df, profile=None) -> Optional[Signal]:
    """
    Avalia sinal ATF v1 usando as ultimas 2 linhas do DataFrame.
    Para uso em live trading (bot_worker).
    """
    if len(df) < 2:
        return None

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    idx = len(df) - 1

    if not _check_cooldown(idx):
        return None

    # Try LONG first
    result = _evaluate_direction(curr, prev, "long", profile)
    if result is not None:
        score, trigger_name, sl, adx, atr, atr_pct = result
        _register_signal(idx)
        logger.info(
            "SIGNAL ATF v1 LONG score=%d trigger=%s | entry=%.2f SL=%.2f "
            "ATR=%.2f RSI=%.1f ADX=%.1f trail=%.1fx",
            score, trigger_name, curr["close"], sl, atr,
            curr.get("rsi", 0), adx, _get_trail_mult(adx),
        )
        return _build_signal(curr["close"], sl, curr, True, score, trigger_name, adx)

    # Try SHORT
    result = _evaluate_direction(curr, prev, "short", profile)
    if result is not None:
        score, trigger_name, sl, adx, atr, atr_pct = result
        _register_signal(idx)
        logger.info(
            "SIGNAL ATF v1 SHORT score=%d trigger=%s | entry=%.2f SL=%.2f "
            "ATR=%.2f RSI=%.1f ADX=%.1f trail=%.1fx",
            score, trigger_name, curr["close"], sl, atr,
            curr.get("rsi", 0), adx, _get_trail_mult(adx),
        )
        return _build_signal(curr["close"], sl, curr, False, score, trigger_name, adx)

    return None


def evaluate_atf_row(
    row, prev_row, bar_index: int, profile=None
) -> Optional[tuple]:
    """
    Avalia sinal ATF v1 para uma linha individual.
    Para uso no backtest loop.

    Returns:
        (Signal, score, trigger_name, adx_at_entry, trail_mult, sl_mult) ou None
    """
    if not _check_cooldown(bar_index):
        return None

    # Try LONG
    result = _evaluate_direction(row, prev_row, "long", profile)
    if result is not None:
        score, trigger_name, sl, adx, atr, atr_pct = result
        _register_signal(bar_index)
        signal = _build_signal(row["close"], sl, row, True, score, trigger_name, adx)
        trail_mult = _get_trail_mult(adx)
        sl_mult = _get_sl_mult(atr_pct)
        return (signal, score, trigger_name, adx, trail_mult, sl_mult)

    # Try SHORT
    result = _evaluate_direction(row, prev_row, "short", profile)
    if result is not None:
        score, trigger_name, sl, adx, atr, atr_pct = result
        _register_signal(bar_index)
        signal = _build_signal(row["close"], sl, row, False, score, trigger_name, adx)
        trail_mult = _get_trail_mult(adx)
        sl_mult = _get_sl_mult(atr_pct)
        return (signal, score, trigger_name, adx, trail_mult, sl_mult)

    return None
