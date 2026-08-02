r"""
strategy_ema_cross.py
---------------------
Estrategia EMA Cross v11 para timeframes intraday (15m/30m).

v11 - MACRO TREND FILTER + TP ADAPTATIVO:
  Objetivo: superar Buy & Hold em periodos longos (2+ anos).

  Problema v10: Em 365d B&H=-44.94%, estrategia=+9.71%. Mas em 730d,
  B&H se recupera e supera a estrategia porque esta:
  1. Entra LONG em tendencias de baixa estruturais (whipsaw)
  2. Sai cedo demais com TP fixo (3.0x ATR), perdendo grandes movimentos
  3. Nao usa trailing stop no backtest (só SL/TP fixos)

  Melhorias v10 -> v11:
  1. EMA200 MACRO FILTER
     -> LONG: close > EMA200 (so opera long acima da macro tendencia)
     -> SHORT: close < EMA200 (so opera short abaixo da macro tendencia)
     -> Evita entrar contra a tendencia estrutural de 2 anos
  2. TP ADAPTATIVO POR REGIME
     -> SHORT em trending_down: TP = 4.5x ATR (era 3.0x)
        Tendencias de baixa tem mais room para correr
     -> LONG em ranging/transition: TP = 3.0x ATR (mantido)
     -> R:R efetivo: 4.5:2.0 = 2.25:1 em trending_down shorts
  3. TRAILING STOP NO BACKTEST (em _simulate_ema_cross)
     -> BE trigger: 1.0x ATR -> move SL para entry
     -> Trailing distance: 1.5x ATR do high water mark
     -> Partial TP: 50% no TP1, trailing no resto
     -> Deixa ganhadores correrem, capturando grandes movimentos

  Melhorias v9 -> v10 (mantidas):
  - OBV direction filter (acumulacao/distribuicao)
  - TP 3.0x ATR base, SL 2.0x ATR
  - Cooldown 12 bars
  - LONGs bloqueados em trending_up, SHORTs liberados em trending_down
  - ATR percentile [0.20, 0.80], BB squeeze >= 0.15

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
    "tp_trending_down_mult": 4.5,   # v11: TP mais largo em trending_down
    "adx_min": 20.0,
    "use_ema200": True,               # v11: filtro macro tendencia
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
    # v11: trailing stop params (usados pelo simulador backtest)
    "be_trigger_atr_mult": 1.0,       # move SL para entry apos 1.0x ATR
    "trailing_atr_mult": 1.5,         # trailing a 1.5x ATR do high water mark
    "partial_tp_pct": 0.50,           # tira 50% no TP1
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
        "ema200": float(row.get("ema200", 0.0)),
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
    # v11: EMA200 macro filter — LONG only above structural trend
    if p.get("use_ema200", False) and d["ema200"] > 0 and d["close"] <= d["ema200"]:
        return None
    # v11: TP adaptativo — regime-based (LONGs kept at base tp_mult)
    regime_str = d["regime_v2"].lower() if d["regime_v2"] else d["regime"].lower()
    effective_tp = tp_mult
    sl = d["close"] - sl_mult * d["atr"]
    tp = d["close"] + effective_tp * d["atr"]
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
    # v11: EMA200 macro filter — SHORT only below structural trend
    if p.get("use_ema200", False) and d["ema200"] > 0 and d["close"] >= d["ema200"]:
        return None
    # v11: TP adaptativo por regime
    regime_str = d["regime_v2"].lower() if d["regime_v2"] else d["regime"].lower()
    if "trending_down" in regime_str:
        effective_tp = p["tp_trending_down_mult"]
    else:
        effective_tp = tp_mult
    sl = d["close"] + sl_mult * d["atr"]
    tp = d["close"] - effective_tp * d["atr"]
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
        "SIGNAL EMA CROSS v11 %s | entry=%.2f SL=%.2f TP=%.2f ATR=%.2f "
        "RSI=%.1f ADX=%.1f RSI_d=%.2f regime=%s OBV=%d EMA200=%.0f",
        direction, d["close"], sl, tp, d["atr"],
        d["rsi"], d["adx"], d["rsi_delta"], d["regime_v2"],
        int(d.get("obv_trend", 0)), d["ema200"],
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
