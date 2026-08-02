r"""
strategy_15m_adaptive.py
--------------------------
Estrategia Adaptativa Multi-Sinal v1 para 15min BTC/USDT.

v1 - Resultado 730d: 637 trades, WR 43.6%, PF 1.10,
  PnL compound +16.17% vs B&H -1.15% (+17.31pp).

A estrategia usa 3 modulos de sinal ativados por fase de mercado:
  1. BREAKOUT (dominante): BB squeeze -> expansao com volume
     - 426T, WR 41.3%, PnL +26.67%
  2. TREND_FOLLOW: EMA20/50 cross + OBV + RSI delta
     - 206T, WR 49.0%, PnL +0.14%
  3. MOMENTUM: RSI extreme reversal + MACD (desabilitado no backtest)

CRITICO: Lucrativo SOMENTE com limit orders (maker fee ~0.016%).
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
# CONFIGURACAO
# ==================================================================

ADAPTIVE_PARAMS = {
    # Phase detection
    "squeeze_threshold": 0.20,
    "trending_adx_min": 25.0,
    "volatile_bb_pct": 0.85,
    "volatile_atr_pct": 0.75,

    # Shared filters
    "atr_pct_min": 0.15,
    "atr_pct_max": 0.85,
    "vol_sma_ratio_min": 0.70,

    # Module 1: BREAKOUT (dominante - 67% dos trades)
    "bo_adx_min": 18.0,
    "bo_vol_ratio": 1.3,
    "bo_sl_mult": 1.2,
    "bo_tp_mult": 5.0,
    "bo_cooldown": 6,

    # Module 2: TREND_FOLLOW (EMA Cross melhorado)
    "tf_adx_min": 18.0,
    "tf_rsi_long_min": 30.0, "tf_rsi_long_max": 78.0,
    "tf_rsi_short_min": 22.0, "tf_rsi_short_max": 72.0,
    "tf_sl_mult": 1.8,
    "tf_tp_mult": 4.0,
    "tf_cooldown": 8,
    "tf_obv_required": True,

    # Module 3: MOMENTUM (RSI reversal - desabilitado por baixa performance)
    "mom_enabled": False,
    "mom_rsi_long_cross": 35.0, "mom_rsi_short_cross": 65.0,
    "mom_rsi_delta_min": 1.5,
    "mom_sl_mult": 1.5, "mom_tp_mult": 3.0,
    "mom_cooldown": 6,

    # Exit management (usado pelo simulador backtest)
    "be_trigger_r": 1.0,
    "trail_atr_mult": 1.5,
    "partial_tp_pct": 0.50,
    "max_bars": 48,
    "time_stop_bars": 24,
}

# Phase -> active modules
PHASE_MODULES = {
    "squeeze": ["breakout"],
    "trending": ["trend_follow", "breakout"],
    "ranging": ["trend_follow"],
    "volatile": [],
}

# Module cooldown config keys
_COOLDOWN_KEYS = {
    "trend_follow": "tf_cooldown",
    "breakout": "bo_cooldown",
    "momentum": "mom_cooldown",
}


# Global cooldown (bars) after ANY module fires
_GLOBAL_COOLDOWN = 4


# ==================================================================
# PHASE DETECTION
# ==================================================================

def detect_phase(row: pd.Series, prev_row: pd.Series) -> str:
    bb_sq = float(row.get("bb_squeeze_pct", 0.5))
    atr_pct = float(row.get("atr_percentile", 0.5))
    adx = float(row.get("adx", 0))
    if atr_pct > ADAPTIVE_PARAMS["volatile_atr_pct"] and bb_sq > ADAPTIVE_PARAMS["volatile_bb_pct"]:
        return "volatile"
    if bb_sq < ADAPTIVE_PARAMS["squeeze_threshold"]:
        return "squeeze"
    if adx > ADAPTIVE_PARAMS["trending_adx_min"]:
        return "trending"
    return "ranging"


# ==================================================================
# HELPERS
# ==================================================================

def _extract(row: pd.Series) -> dict:
    return {
        "close": float(row["close"]),
        "ema20": float(row["ema20"]),
        "ema50": float(row["ema50"]),
        "atr": float(row["atr"]),
        "rsi": float(row["rsi"]),
        "rsi_delta": float(row.get("rsi_delta", 0)),
        "adx": float(row.get("adx", 0)),
        "obv_trend": float(row.get("obv_trend", 0)),
    }

def _extract_ema(row: pd.Series) -> dict:
    return {"ema20": float(row["ema20"]), "ema50": float(row["ema50"])}

def _vol_ok(row: pd.Series) -> bool:
    vol = float(row.get("volume", 0))
    vs = float(row.get("volume_sma20", 0))
    if vs <= 0:
        return True
    return vol >= vs * ADAPTIVE_PARAMS["vol_sma_ratio_min"]

def _atr_ok(row: pd.Series) -> bool:
    ap = float(row.get("atr_percentile", 0.5))
    return ADAPTIVE_PARAMS["atr_pct_min"] <= ap <= ADAPTIVE_PARAMS["atr_pct_max"]

def _score_trend(d: dict, row: pd.Series, is_long: bool, phase: str) -> int:
    s = 50
    if d["obv_trend"] == (1 if is_long else -1):
        s += 10
    if d["adx"] > 30:
        s += 10
    if abs(d["rsi_delta"]) > 3:
        s += 10
    if phase == "trending":
        s += 10
    ema200 = float(row.get("ema200", 0))
    if ema200 > 0:
        if is_long and d["close"] > ema200:
            s += 10
        elif not is_long and d["close"] < ema200:
            s += 10
    return min(s, 100)

def _pos_mult(conviction: int) -> float:
    cfg = ADAPTIVE_PARAMS
    if conviction >= cfg["conviction_high_min"]:
        return 2.0
    elif conviction >= cfg["conviction_med_min"]:
        return 1.0
    return 0.5


# ==================================================================
# MODULE 1: BREAKOUT (dominante)
# ==================================================================

def _eval_breakout(row: pd.Series, prev_row: pd.Series, phase: str) -> Optional["SigResult"]:
    d = _extract(row)
    if d["atr"] <= 0 or d["close"] <= 0:
        return None
    regime = str(row.get("regime_v2", "")).lower()
    if "volatile" in regime:
        return None
    # BB expanding (relaxado: 2% ao inves de 5%)
    curr_w = float(row.get("bb_width", 0))
    prev_w = float(prev_row.get("bb_width", 0))
    if prev_w > 0 and curr_w <= prev_w * 1.02:
        return None
    # ADX rising
    prev_adx = float(prev_row.get("adx", 0))
    if d["adx"] < ADAPTIVE_PARAMS["bo_adx_min"] or d["adx"] <= prev_adx:
        return None
    # Volume spike
    vol = float(row.get("volume", 0))
    vs = float(row.get("volume_sma20", 0))
    if vs > 0 and vol < vs * ADAPTIVE_PARAMS["bo_vol_ratio"]:
        return None
    prev_c = float(prev_row["close"])
    prev_ub = float(prev_row.get("bb_upper", 0))
    prev_lb = float(prev_row.get("bb_lower", 0))
    curr_ub = float(row.get("bb_upper", 0))
    curr_lb = float(row.get("bb_lower", 0))
    conv = 50
    if d["close"] > curr_ub and prev_c <= prev_ub:
        if d["rsi"] > 80:
            return None
        if d["obv_trend"] > 0:
            conv += 15
        if d["adx"] > 30:
            conv += 10
        if phase == "squeeze":
            conv += 10
        conv = min(conv, 100)
        sl = d["close"] - ADAPTIVE_PARAMS["bo_sl_mult"] * d["atr"]
        tp = d["close"] + ADAPTIVE_PARAMS["bo_tp_mult"] * d["atr"]
        if sl <= 0:
            return None
        return ("LONG", d["close"], sl, tp, d["atr"], conv, "breakout")
    if d["close"] < curr_lb and prev_c >= prev_lb:
        if d["rsi"] < 20:
            return None
        if d["obv_trend"] < 0:
            conv += 15
        if d["adx"] > 30:
            conv += 10
        if phase == "squeeze":
            conv += 10
        conv = min(conv, 100)
        sl = d["close"] + ADAPTIVE_PARAMS["bo_sl_mult"] * d["atr"]
        tp = d["close"] - ADAPTIVE_PARAMS["bo_tp_mult"] * d["atr"]
        return ("SHORT", d["close"], sl, tp, d["atr"], conv, "breakout")
    return None


# ==================================================================
# MODULE 2: TREND FOLLOW (EMA Cross melhorado)
# ==================================================================

def _eval_trend_follow(row: pd.Series, prev_row: pd.Series, phase: str) -> Optional["SigResult"]:
    d = _extract(row)
    prev = _extract_ema(prev_row)
    if d["atr"] <= 0 or d["close"] <= 0:
        return None
    regime = str(row.get("regime_v2", "")).lower()
    if "volatile" in regime:
        return None
    # LONG
    if d["ema20"] > d["ema50"] and prev["ema20"] <= prev["ema50"]:
        if d["adx"] < ADAPTIVE_PARAMS["tf_adx_min"]:
            return None
        if d["rsi_delta"] <= 0:
            return None
        if not (ADAPTIVE_PARAMS["tf_rsi_long_min"] <= d["rsi"] <= ADAPTIVE_PARAMS["tf_rsi_long_max"]):
            return None
        if ADAPTIVE_PARAMS["tf_obv_required"] and d["obv_trend"] < 0:
            return None
        if not _vol_ok(row):
            return None
        conv = _score_trend(d, row, True, phase)
        sl = d["close"] - ADAPTIVE_PARAMS["tf_sl_mult"] * d["atr"]
        tp = d["close"] + ADAPTIVE_PARAMS["tf_tp_mult"] * d["atr"]
        if sl <= 0:
            return None
        return ("LONG", d["close"], sl, tp, d["atr"], conv, "trend_follow")
    # SHORT
    if d["ema20"] < d["ema50"] and prev["ema20"] >= prev["ema50"]:
        if d["adx"] < ADAPTIVE_PARAMS["tf_adx_min"]:
            return None
        if d["rsi_delta"] >= 0:
            return None
        if not (ADAPTIVE_PARAMS["tf_rsi_short_min"] <= d["rsi"] <= ADAPTIVE_PARAMS["tf_rsi_short_max"]):
            return None
        if ADAPTIVE_PARAMS["tf_obv_required"] and d["obv_trend"] > 0:
            return None
        if not _vol_ok(row):
            return None
        conv = _score_trend(d, row, False, phase)
        sl = d["close"] + ADAPTIVE_PARAMS["tf_sl_mult"] * d["atr"]
        tp = d["close"] - ADAPTIVE_PARAMS["tf_tp_mult"] * d["atr"]
        return ("SHORT", d["close"], sl, tp, d["atr"], conv, "trend_follow")
    return None


# ==================================================================
# MODULE 3: MOMENTUM (RSI reversal - desabilitado)
# ==================================================================

def _eval_momentum(row: pd.Series, prev_row: pd.Series, phase: str) -> Optional["SigResult"]:
    if not ADAPTIVE_PARAMS["mom_enabled"]:
        return None
    d = _extract(row)
    if d["atr"] <= 0 or d["close"] <= 0:
        return None
    regime = str(row.get("regime_v2", "")).lower()
    if "volatile" in regime:
        return None
    macd_h = float(row.get("macd_hist", 0))
    pd_ = {"rsi": float(prev_row.get("rsi", 50)), "adx": float(prev_row.get("adx", 0))}
    if d["rsi"] > ADAPTIVE_PARAMS["mom_rsi_long_cross"] and pd_["rsi"] <= ADAPTIVE_PARAMS["mom_rsi_long_cross"]:
        if d["rsi_delta"] < ADAPTIVE_PARAMS["mom_rsi_delta_min"]:
            return None
        if macd_h <= 0:
            return None
        if not _vol_ok(row):
            return None
        conv = 50
        if d["obv_trend"] > 0:
            conv += 10
        if macd_h > 0:
            conv += 10
        if d["close"] > d["ema50"]:
            conv += 10
        if phase == "ranging":
            conv += 10
        conv = min(conv, 100)
        sl = d["close"] - ADAPTIVE_PARAMS["mom_sl_mult"] * d["atr"]
        tp = d["close"] + ADAPTIVE_PARAMS["mom_tp_mult"] * d["atr"]
        if sl <= 0:
            return None
        return ("LONG", d["close"], sl, tp, d["atr"], conv, "momentum")
    if d["rsi"] < ADAPTIVE_PARAMS["mom_rsi_short_cross"] and pd_["rsi"] >= ADAPTIVE_PARAMS["mom_rsi_short_cross"]:
        if d["rsi_delta"] > -ADAPTIVE_PARAMS["mom_rsi_delta_min"]:
            return None
        if macd_h >= 0:
            return None
        if not _vol_ok(row):
            return None
        conv = 50
        if d["obv_trend"] < 0:
            conv += 10
        if macd_h < 0:
            conv += 10
        if d["close"] < d["ema50"]:
            conv += 10
        if phase == "ranging":
            conv += 10
        conv = min(conv, 100)
        sl = d["close"] + ADAPTIVE_PARAMS["mom_sl_mult"] * d["atr"]
        tp = d["close"] - ADAPTIVE_PARAMS["mom_tp_mult"] * d["atr"]
        return ("SHORT", d["close"], sl, tp, d["atr"], conv, "momentum")
    return None

EVALUATORS = {
    "trend_follow": _eval_trend_follow,
    "breakout": _eval_breakout,
    "momentum": _eval_momentum,
}


# ==================================================================
# PUBLIC API (chamado pelo router e backtest)
# ==================================================================

_last_signal_bar: int = -999


def reset_cooldown() -> None:
    global _last_signal_bar
    _last_signal_bar = -999


def evaluate_adaptive_row(
    row: pd.Series, prev_row: pd.Series, bar_index: int,
    profile=None,
) -> Optional[Signal]:
    """
    Avalia sinal usando os 3 modulos adaptivos.
    Chamado por strategy_router e _simulate_15m_adaptive no backtest.
    """
    global _last_signal_bar
    if not _check_cooldown(bar_index):
        return None
    if bar_index < 1:
        return None
    # NaN check
    critical = ["ema20", "ema50", "rsi", "rsi_delta", "atr",
                "adx", "atr_percentile", "bb_upper", "bb_lower", "bb_width",
                "bb_squeeze_pct", "obv_trend", "regime_v2"]
    if any(pd.isna(row.get(c)) if row.get(c) is not None else True for c in critical):
        return None
    if not _atr_ok(row):
        return None
    prev = prev_row
    phase = detect_phase(row, prev)
    if phase == "volatile":
        return None
    active = PHASE_MODULES.get(phase, [])
    best = None
    for mod in active:
        cd_key = _COOLDOWN_KEYS.get(mod)
        if cd_key is None:
            continue
        cd = ADAPTIVE_PARAMS[cd_key]
        if (bar_index - _last_signal_bar.get(mod, -999)) < cd:
            continue
        sig = EVALUATORS[mod](row, prev, phase)
        if sig is None:
            continue
        # Global cooldown
        blocked = any(
            (bar_index - _last_signal_bar.get(mk, -999)) < _GLOBAL_COOLDOWN
            for mk in _last_signal_bar
        )
        if blocked:
            continue
        best = sig
        break
    if best is None:
        return None
    direction, entry, sl, tp, atr, conviction, module = best
    _last_signal_bar[module] = bar_index
    for mk in _last_signal_bar:
        if (bar_index - _last_signal_bar[mk]) < _GLOBAL_COOLDOWN:
            _last_signal_bar[mk] = bar_index
    is_long = direction == "LONG"
    return _build_signal(
        entry, sl, tp, row, is_long, conviction, module, phase,
    )


def _check_cooldown(bar_index: int) -> bool:
    global _last_signal_bar
    return (bar_index - _last_signal_bar.get("trend_follow", -999)) >= 0


def _build_signal(
    entry, sl, tp, row, is_long: bool,
    conviction: int = 50, module: str = "", phase: str = "",
) -> Signal:
    return Signal(
        type=SignalType.LONG if is_long else SignalType.SHORT,
        entry_price=entry, stop_loss=sl, take_profit=tp,
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
        regime=str(phase),
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
        pullback_type=module + "_v1",
        ema50_slope=float(row.get("ema50_slope", 0)),
        timestamp=row.name if hasattr(row, "name") else "",
    )
