r"""
strategy_atf_v2.py
-----------------
Adaptive Trend-Follow v2 (ATF v2) para 15min e 1h.

Evolucao do ATF v1 com integracao dos indicadores do PDF:
- BBWP (Bollinger Band Width Percentile) como modulador de risco
- Stoch RSI como gatilho de entrada adicional
- Volume confirmando direcao
- Divergencia RSI como gatilho de saida antecipada

Mudancas vs ATF v1:
  1. Stoch RSI K crossing D adicionado como trigger (stoch_cross)
  2. BBWP squeeze detectado — se squeeze ativo, reduz TP mas aumenta trailing
  3. Stoch RSI oversold/overbought como gatilho de saida antecipada
  4. Score 11 dimensoes (adicionada: Stoch RSI alinhado com direcao)

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


ATF_V2_PARAMS = {
    # ---- Entry requirements ----
    "min_score": 6,
    "high_score_threshold": 8,
    "direct_entry_score": 9,

    # ---- Stop Loss (adaptive by ATR percentile) ----
    "sl_atr_base": 1.5,
    "sl_atr_high_vol": 2.0,
    "sl_atr_low_vol": 1.2,
    "high_vol_threshold": 0.70,
    "low_vol_threshold": 0.30,

    # ---- Trailing Stop (adaptive by ADX) ----
    "be_trigger_atr": 0.8,
    "trail_adx_strong": 2.5,
    "trail_adx_medium": 1.5,
    "trail_adx_weak": 1.0,
    "adx_strong_min": 35.0,
    "adx_medium_min": 25.0,

    # ---- Filters ----
    "max_bars": 96,
    "cooldown": 6,
    "cooldown_trailing": 3,
    "atr_pct_min": 0.15,
    "atr_pct_max": 0.85,
    "bb_squeeze_min": 0.10,
    "vol_sma_ratio": 0.70,
    "rsi_delta_burst": 3.0,

    # ---- Stoch RSI (v2 new) ----
    "stoch_rsi_ob": 80,         # Overbought
    "stoch_rsi_os": 20,         # Oversold
    "stoch_rsi_cross_enable": True,  # Enable StochRSI cross trigger

    # ---- BBWP (v2 new) ----
    "bbwp_squeeze_threshold": 15,  # BBWP < this = squeeze active
    "bbwp_squeeze_trail_boost": 1.5,  # Multiply trailing by this in squeeze
    "bbwp_squeeze_sl_reduce": 0.8,   # Multiply SL by this in squeeze

    # ---- Risk ----
    "risk_base_pct": 0.015,
    "risk_high_pct": 0.025,
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
    cd = ATF_V2_PARAMS["cooldown_trailing"] if _last_exit_was_trailing else ATF_V2_PARAMS["cooldown"]
    return (current_idx - _last_signal_bar) >= cd


def _register_signal(current_idx: int, was_trailing: bool = False) -> None:
    global _last_signal_bar, _last_exit_was_trailing
    _last_signal_bar = current_idx
    _last_exit_was_trailing = was_trailing


# ==================================================================
# COMPOSITE TREND SCORING (0-11) — v2 with Stoch RSI
# ==================================================================

def _compute_trend_score(row, direction: str = "long") -> int:
    """
    Calcula score composto de tendencia (0-11 pontos).
    Mesmas 10 dimensoes do ATF v1 + 1 nova: Stoch RSI alinhado.
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
    stoch_k = float(row.get("stoch_rsi_k", 50))

    if direction == "long":
        if ema20 > ema50: score += 1
        if close > ema50: score += 1
        if ema200 > 0 and close > ema200: score += 1
        if ema50_slope > 0: score += 1
        if adx > ATF_V2_PARAMS["adx_medium_min"]: score += 1
        if adx > ATF_V2_PARAMS["adx_strong_min"]: score += 1
        if plus_di > minus_di: score += 1
        if macd > macd_signal_val: score += 1
        if rsi > 45: score += 1
        if obv_trend > 0: score += 1
        # NEW: Stoch RSI not overbought (room to run)
        if stoch_k < 80: score += 1
    else:
        if ema20 < ema50: score += 1
        if close < ema50: score += 1
        if ema200 > 0 and close < ema200: score += 1
        if ema50_slope < 0: score += 1
        if adx > ATF_V2_PARAMS["adx_medium_min"]: score += 1
        if adx > ATF_V2_PARAMS["adx_strong_min"]: score += 1
        if minus_di > plus_di: score += 1
        if macd < macd_signal_val: score += 1
        if rsi < 55: score += 1
        if obv_trend < 0: score += 1
        # NEW: Stoch RSI not oversold
        if stoch_k > 20: score += 1

    return score


# ==================================================================
# ENTRY TRIGGERS (7 tipos) — v2 with Stoch RSI cross
# ==================================================================

_PULLBACK_TRIGGERS = {"pullback_ema20", "pullback_ema50", "rsi_dip", "macd_cross", "bb_bounce", "stoch_cross"}
_MOMENTUM_TRIGGERS = {"momentum_burst"}
_DIRECT_TRIGGERS = {"trend_aligned"}


def _check_entry_triggers(
    row, prev_row, direction: str = "long", score: int = 0
) -> tuple:
    """Verifica se algum dos 7 gatilhos esta ativo."""
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
    stoch_k = float(row.get("stoch_rsi_k", 50))
    stoch_d = float(row.get("stoch_rsi_d", 50))
    prev_stoch_k = float(prev_row.get("stoch_rsi_k", 50))
    prev_stoch_d = float(prev_row.get("stoch_rsi_d", 50))
    p = ATF_V2_PARAMS

    if direction == "long":
        if low <= ema20:
            return True, "pullback_ema20"
        if low <= ema50:
            return True, "pullback_ema50"
        if prev_rsi < 50 and rsi >= 50:
            return True, "rsi_dip"
        if macd_hist > 0 and prev_macd_hist <= 0:
            return True, "macd_cross"
        bb_range = bb_upper - bb_lower
        if bb_range > 0:
            bb_pos = (close - bb_lower) / bb_range
            if bb_pos < 0.25:
                return True, "bb_bounce"
        # NEW: Stoch RSI K crosses above D in NEUTRAL zone
        # Only in middle zone (30-70) to avoid false signals
        if p["stoch_rsi_cross_enable"]:
            if (stoch_k > stoch_d and prev_stoch_k <= prev_stoch_d
                and 30 < stoch_k < 70):
                return True, "stoch_cross"
        if score >= p["high_score_threshold"]:
            if rsi_delta > p["rsi_delta_burst"] and macd_hist > 0:
                return True, "momentum_burst"
        if score >= p["direct_entry_score"]:
            return True, "trend_aligned"
    else:
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
        # NEW: Stoch RSI K crosses below D in NEUTRAL zone
        # Only in middle zone (30-70) to avoid false signals
        if p["stoch_rsi_cross_enable"]:
            if (stoch_k < stoch_d and prev_stoch_k >= prev_stoch_d
                and 30 < stoch_k < 70):
                return True, "stoch_cross"
        if score >= p["high_score_threshold"]:
            if rsi_delta < -p["rsi_delta_burst"] and macd_hist < 0:
                return True, "momentum_burst"
        if score >= p["direct_entry_score"]:
            return True, "trend_aligned"

    return False, ""


def _validate_trigger_for_score(trigger_name: str, score: int) -> bool:
    if trigger_name in _PULLBACK_TRIGGERS:
        return score >= ATF_V2_PARAMS["min_score"]
    elif trigger_name in _MOMENTUM_TRIGGERS:
        return score >= ATF_V2_PARAMS["high_score_threshold"]
    elif trigger_name in _DIRECT_TRIGGERS:
        return score >= ATF_V2_PARAMS["direct_entry_score"]
    return False


# ==================================================================
# ADAPTIVE SL & TRAILING (v2: BBWP-modulated)
# ==================================================================

def _get_sl_mult(atr_pct: float, bbwp: float = 50) -> float:
    p = ATF_V2_PARAMS
    mult = p["sl_atr_base"]
    if atr_pct > p["high_vol_threshold"]:
        mult = p["sl_atr_high_vol"]
    elif atr_pct < p["low_vol_threshold"]:
        mult = p["sl_atr_low_vol"]
    # v2: reduce SL in squeeze (tighter stops, more capital efficient)
    if bbwp < p["bbwp_squeeze_threshold"]:
        mult *= p["bbwp_squeeze_sl_reduce"]
    return mult


def _get_trail_mult(adx: float, bbwp: float = 50) -> float:
    p = ATF_V2_PARAMS
    if adx >= p["adx_strong_min"]:
        mult = p["trail_adx_strong"]
    elif adx >= p["adx_medium_min"]:
        mult = p["trail_adx_medium"]
    else:
        mult = p["trail_adx_weak"]
    # v2: boost trailing in squeeze (let winners run after volatility expansion)
    if bbwp < p["bbwp_squeeze_threshold"]:
        mult *= p["bbwp_squeeze_trail_boost"]
    return mult


# ==================================================================
# SHARED FILTERS
# ==================================================================

def _shared_filters(row) -> bool:
    p = ATF_V2_PARAMS
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
    regime = str(row.get("regime", "")).lower()
    if "volatile" in regime:
        return False
    return True


# ==================================================================
# SIGNAL BUILDER
# ==================================================================

def _build_signal(entry, sl, row, is_long: bool, score: int, trigger_name: str, adx: float) -> Signal:
    ts = row.name if hasattr(row, "name") else row.index
    close = float(row["close"]) if hasattr(row, "__getitem__") else float(entry)
    atr = float(row.get("atr", 0))
    if is_long:
        tp = close + 1000 * atr
    else:
        tp = max(0.0, close - 1000 * atr)
    return Signal(
        type=SignalType.LONG if is_long else SignalType.SHORT,
        entry_price=close if hasattr(row, "__getitem__") else entry,
        stop_loss=sl, take_profit=tp, atr=atr,
        rsi=float(row.get("rsi", 0)),
        rsi_delta=float(row.get("rsi_delta", 0)),
        macd_hist=float(row.get("macd_hist", 0)),
        ema20=float(row.get("ema20", 0)), ema50=float(row.get("ema50", 0)),
        ema200=float(row.get("ema200", 0)), adx=adx,
        plus_di=float(row.get("plus_di", 0)), minus_di=float(row.get("minus_di", 0)),
        regime=f"atf_v2_score_{score}",
        bb_lower=float(row.get("bb_lower", 0)), bb_upper=float(row.get("bb_upper", 0)),
        bb_width=float(row.get("bb_width", 0)), bb_squeeze_pct=float(row.get("bb_squeeze_pct", 0.5)),
        volume=float(row.get("volume", 0)), volume_sma20=float(row.get("volume_sma20", 0)),
        volume_sma50=float(row.get("volume_sma50", 0)), atr_percentile=float(row.get("atr_percentile", 0.5)),
        fib_0382=0.0, fib_0500=0.0, fib_0618=0.0,
        fib_direction=0, fib_proximity=0.0,
        pullback_type=f"atf_v2_{trigger_name}",
        ema50_slope=float(row.get("ema50_slope", 0)), timestamp=ts,
    )


# ==================================================================
# CORE EVALUATION
# ==================================================================

def _evaluate_direction(row, prev_row, direction: str, profile=None) -> Optional[tuple]:
    p = ATF_V2_PARAMS
    if not _shared_filters(row):
        return None
    score = _compute_trend_score(row, direction)
    if score < p["min_score"]:
        return None
    triggered, trigger_name = _check_entry_triggers(row, prev_row, direction, score)
    if not triggered:
        return None
    if not _validate_trigger_for_score(trigger_name, score):
        return None
    atr = float(row["atr"])
    atr_pct = float(row.get("atr_percentile", 0.5))
    bbwp = float(row.get("bbwp", 50))
    if atr <= 0:
        return None
    close = float(row["close"])
    sl_mult = _get_sl_mult(atr_pct, bbwp)
    if direction == "long":
        sl = close - sl_mult * atr
        if sl <= 0: return None
    else:
        sl = close + sl_mult * atr
    adx = float(row.get("adx", 0))
    return (score, trigger_name, sl, adx, atr, atr_pct, bbwp)


def evaluate_atf_v2(df, profile=None) -> Optional[Signal]:
    if len(df) < 2: return None
    curr = df.iloc[-1]; prev = df.iloc[-2]; idx = len(df) - 1
    if not _check_cooldown(idx): return None
    result = _evaluate_direction(curr, prev, "long", profile)
    if result is not None:
        score, trigger_name, sl, adx, atr, atr_pct, bbwp = result
        _register_signal(idx)
        logger.info("SIGNAL ATF v2 LONG score=%d trigger=%s bbwp=%.0f", score, trigger_name, bbwp)
        return _build_signal(curr["close"], sl, curr, True, score, trigger_name, adx)
    result = _evaluate_direction(curr, prev, "short", profile)
    if result is not None:
        score, trigger_name, sl, adx, atr, atr_pct, bbwp = result
        _register_signal(idx)
        logger.info("SIGNAL ATF v2 SHORT score=%d trigger=%s bbwp=%.0f", score, trigger_name, bbwp)
        return _build_signal(curr["close"], sl, curr, False, score, trigger_name, adx)
    return None


def evaluate_atf_v2_row(row, prev_row, bar_index: int, profile=None) -> Optional[tuple]:
    if not _check_cooldown(bar_index): return None
    result = _evaluate_direction(row, prev_row, "long", profile)
    if result is not None:
        score, trigger_name, sl, adx, atr, atr_pct, bbwp = result
        _register_signal(bar_index)
        signal = _build_signal(row["close"], sl, row, True, score, trigger_name, adx)
        trail_mult = _get_trail_mult(adx, bbwp)
        sl_mult = _get_sl_mult(atr_pct, bbwp)
        return (signal, score, trigger_name, adx, trail_mult, sl_mult)
    result = _evaluate_direction(row, prev_row, "short", profile)
    if result is not None:
        score, trigger_name, sl, adx, atr, atr_pct, bbwp = result
        _register_signal(bar_index)
        signal = _build_signal(row["close"], sl, row, False, score, trigger_name, adx)
        trail_mult = _get_trail_mult(adx, bbwp)
        sl_mult = _get_sl_mult(atr_pct, bbwp)
        return (signal, score, trigger_name, adx, trail_mult, sl_mult)
    return None
