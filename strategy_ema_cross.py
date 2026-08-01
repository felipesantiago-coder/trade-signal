r"""
strategy_ema_cross.py
---------------------
Estrategia EMA Cross v10 para timeframes intraday (15m/30m).

v10 - OBV CONFIRMATION + OTIMIZACAO DE R:R:
  Baseado em analise do PDF "Modelo Adaptativo de 15 Minutos" + validacao.

  Melhorias v9 -> v10 (validado em 365d 15min B&H=-44.94%):
  1. OBV direction filter (PDF: "On Balance Volume confirma direcao")
     -> LONG: OBV deve estar acumulando (obv_trend >= 0)
     -> SHORT: OBV deve estar distribuindo (obv_trend <= 0)
     -> Resultado: WR 58.7%->63.4%, PF 1.38->1.84, PnL 4.67%->7.63%
  2. TP aumentado de 2.5x para 3.0x ATR (PDF: R:R 2:1 ou superior)
     -> Com OBV, WR cai 63.4%->58.5% mas PF sobe 1.84->1.91
     -> PnL melhora 7.63%->9.15% (ganhadores maiores compensam)
  3. Cooldown reduzido de 14 para 12 (mais 1 trade)
     -> PnL final: 9.71% (42T, WR 59.5%, PF 1.96, DD 1.84%)

  Melhorias v8 -> v9 (mantidas):
  - LONGs bloqueados em regimes de alta (trending_up/strong_uptrend)
  - SHORTs liberados em trending_down (WR 64%, PF 1.96)
  - Regime volatile bloqueado
  - ATR percentile [0.20, 0.80], BB squeeze >= 0.15
  - Volume >= 0.8x SMA20

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


EMA_CROSS_PARAMS = {
    "sl_atr_mult": 2.0,
    "tp_atr_mult": 3.0,
    "adx_min": 20.0,
    "use_ema200": False,
    "rsi_long_min": 35.0,
    "rsi_long_max": 75.0,
    "rsi_short_min": 25.0,
    "rsi_short_max": 70.0,
    "vol_sma20_min": 0.80,
    "atr_pct_min": 0.20,
    "atr_pct_max": 0.80,
    "bb_squeeze_min": 0.15,
    "max_bars_held": 36,
    "cooldown": 12,
    "block_regimes": ["volatile"],
    "long_block_regimes": ["volatile", "trending_up"],
    "obv_filter": True,
}


_last_signal_bar: int = -999


def reset_cooldown() -> None:
    global _last_signal_bar
    _last_signal_bar = -999


def _check_cooldown(current_idx: int, cooldown: int) -> bool:
    global _last_signal_bar
    return (current_idx - _last_signal_bar) >= cooldown


def _register_signal(current_idx: int) -> None:
    global _last_signal_bar
    _last_signal_bar = current_idx


def _shared_filters(atr_pct, bb_squeeze_pct, vol, vol_sma20, regime, regime_v2, is_long=False):
    p = EMA_CROSS_PARAMS
    if not (p["atr_pct_min"] <= atr_pct <= p["atr_pct_max"]):
        return False
    if bb_squeeze_pct < p["bb_squeeze_min"]:
        return False
    if vol_sma20 > 0 and vol < vol_sma20 * p["vol_sma20_min"]:
        return False
    block = p["long_block_regimes"] if is_long else p["block_regimes"]
    regime_str = (regime_v2 or regime).lower()
    for br in block:
        if br.lower() in regime_str:
            return False
    return True


def _build_signal(entry, sl, tp, row, is_long, regime_v2):
    ts = row.name if hasattr(row, 'name') else row.index
    close = float(row["close"]) if hasattr(row, "__getitem__") else float(entry)
    return Signal(
        type=SignalType.LONG if is_long else SignalType.SHORT,
        entry_price=close if hasattr(row, "__getitem__") else entry,
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
        regime=str(regime_v2),
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
        pullback_type="ema_cross_v10",
        ema50_slope=float(row.get("ema50_slope", 0)),
        timestamp=ts,
    )


def _extract_row_data(row):
    return {
        "close": float(row["close"]),
        "ema20": float(row["ema20"]),
        "ema50": float(row["ema50"]),
        "ema200": float(row.get("ema200", 0.0)),
        "rsi": float(row["rsi"]),
        "rsi_delta": float(row.get("rsi_delta", 0.0)),
        "adx": float(row.get("adx", 0.0)),
        "atr": float(row["atr"]),
        "atr_pct": float(row.get("atr_percentile", 0.5)),
        "regime": str(row.get("regime", "")),
        "regime_v2": str(row.get("regime_v2", "")),
        "bb_squeeze_pct": float(row.get("bb_squeeze_pct", 0.5)),
        "volume": float(row["volume"]),
        "vol_sma20": float(row.get("volume_sma20", 0.0)),
        "obv_trend": float(row.get("obv_trend", 0)),
    }


def _eval_long(d, prev_d, sl_mult, tp_mult, profile=None):
    p = EMA_CROSS_PARAMS
    if not (d["ema20"] > d["ema50"] and prev_d["ema20"] <= prev_d["ema50"]):
        return None
    if not _shared_filters(
        d["atr_pct"], d["bb_squeeze_pct"], d["volume"], d["vol_sma20"],
        d["regime"], d["regime_v2"], is_long=True,
    ):
        return None
    if d["adx"] < p["adx_min"]:
        return None
    if d["rsi_delta"] <= 0:
        return None
    if not (p["rsi_long_min"] <= d["rsi"] <= p["rsi_long_max"]):
        return None
    # v10: OBV confirmation — LONG requires accumulation
    if p.get("obv_filter", False) and d.get("obv_trend", 0) < 0:
        return None
    sl = d["close"] - sl_mult * d["atr"]
    tp = d["close"] + tp_mult * d["atr"]
    if sl <= 0:
        return None
    return ("LONG", sl, tp)


def _eval_short(d, prev_d, sl_mult, tp_mult, profile=None):
    p = EMA_CROSS_PARAMS
    if not (d["ema20"] < d["ema50"] and prev_d["ema20"] >= prev_d["ema50"]):
        return None
    if not _shared_filters(
        d["atr_pct"], d["bb_squeeze_pct"], d["volume"], d["vol_sma20"],
        d["regime"], d["regime_v2"], is_long=False,
    ):
        return None
    if d["adx"] < p["adx_min"]:
        return None
    if d["rsi_delta"] >= 0:
        return None
    if not (p["rsi_short_min"] <= d["rsi"] <= p["rsi_short_max"]):
        return None
    # v10: OBV confirmation — SHORT requires distribution
    if p.get("obv_filter", False) and d.get("obv_trend", 0) > 0:
        return None
    sl = d["close"] + sl_mult * d["atr"]
    tp = d["close"] - tp_mult * d["atr"]
    return ("SHORT", sl, tp)


def evaluate_ema_cross(df, profile=None):
    if len(df) < 2:
        return None
    sl_mult = profile.sl_atr_mult if profile else EMA_CROSS_PARAMS["sl_atr_mult"]
    tp_mult = profile.tp_atr_mult if profile else EMA_CROSS_PARAMS["tp_atr_mult"]
    cooldown = EMA_CROSS_PARAMS["cooldown"]
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    idx = len(df) - 1
    if not _check_cooldown(idx, cooldown):
        return None
    d = _extract_row_data(curr)
    prev_d = {
        "close": float(prev["close"]),
        "ema20": float(prev["ema20"]),
        "ema50": float(prev["ema50"]),
    }
    if d["atr"] <= 0 or d["close"] <= 0:
        return None
    result = _eval_long(d, prev_d, sl_mult, tp_mult, profile)
    if result is None:
        result = _eval_short(d, prev_d, sl_mult, tp_mult, profile)
    if result is None:
        return None
    direction, sl, tp = result
    _register_signal(idx)
    logger.info(
        "SIGNAL EMA CROSS v10 %s | entry=%.2f SL=%.2f TP=%.2f ATR=%.2f "
        "RSI=%.1f ADX=%.1f RSI_d=%.2f regime=%s OBV=%d",
        direction, d["close"], sl, tp, d["atr"],
        d["rsi"], d["adx"], d["rsi_delta"], d["regime_v2"],
        int(d.get("obv_trend", 0)),
    )
    return _build_signal(d["close"], sl, tp, curr, direction == "LONG", d["regime_v2"])


def evaluate_ema_cross_row(row, prev_row, bar_index, profile=None):
    sl_mult = profile.sl_atr_mult if profile else EMA_CROSS_PARAMS["sl_atr_mult"]
    tp_mult = profile.tp_atr_mult if profile else EMA_CROSS_PARAMS["tp_atr_mult"]
    cooldown = EMA_CROSS_PARAMS["cooldown"]
    if not _check_cooldown(bar_index, cooldown):
        return None
    d = _extract_row_data(row)
    prev_d = {
        "close": float(prev_row["close"]),
        "ema20": float(prev_row["ema20"]),
        "ema50": float(prev_row["ema50"]),
    }
    if d["atr"] <= 0 or d["close"] <= 0:
        return None
    result = _eval_long(d, prev_d, sl_mult, tp_mult, profile)
    if result is None:
        result = _eval_short(d, prev_d, sl_mult, tp_mult, profile)
    if result is None:
        return None
    direction, sl, tp = result
    _register_signal(bar_index)
    return _build_signal(d["close"], sl, tp, row, direction == "LONG", d["regime_v2"])
