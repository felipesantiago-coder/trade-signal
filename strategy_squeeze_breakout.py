r"""
strategy_squeeze_breakout.py
---------------------------
Squeeze Breakout Strategy v3 (SBS) para 15min e 1h.

Mudanca fundamental v3: REVERSAO apos squeeze, nao breakout.
Breakouts de BB geram muitos falsos positivos. A abordagem correta e:
  1. Esperar squeeze (compressao)
  2. Esperar a expansao inicial (breakout)
  3. Entrar na RETRACAO de volta a EMA/Banda oposta

Indicadores: BBWP, Bollinger Bands, Stoch RSI, Volume, RSI + EMA.

Custos: fee=0.016% + spread=2bps + slip=2bps (limit orders).
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


SBS_PARAMS = {
    # ---- Squeeze Detection (BBWP) ----
    "bbwp_lookback": 100,
    "bbwp_squeeze_threshold": 15,   # BBWP < this = squeeze
    "bbwp_expansion_min": 40,       # BBWP > this = expansion started
    "bbwp_was_squeezed_bars": 20,   # Bars to look back for squeeze

    # ---- Bollinger Bands ----
    "bb_period": 20,
    "bb_std": 2.0,
    "bb_retrace_zone": 0.30,        # Price must retrace to within 30% of BB from band

    # ---- Stoch RSI ----
    "stoch_rsi_period": 14,
    "stoch_rsi_k_smooth": 3,
    "stoch_rsi_d_smooth": 3,
    "stoch_rsi_ob": 80,
    "stoch_rsi_os": 20,

    # ---- Volume ----
    "vol_sma_period": 20,
    "vol_ratio_min": 0.7,           # Volume must not be dead

    # ---- Trend Filter (EMA) ----
    "use_ema200_filter": True,
    "use_adx_filter": True,
    "adx_min": 18,

    # ---- RSI ----
    "rsi_long_min": 30,             # Oversold zone for long
    "rsi_long_max": 55,             # Not yet overbought
    "rsi_short_min": 45,            # Not yet oversold
    "rsi_short_max": 70,            # Overbought zone for short

    # ---- Stop Loss ----
    "sl_atr_mult": 1.5,

    # ---- Take Profit ----
    "tp_atr_mult": 2.5,             # Conservative TP
    "tp_atr_mult_trending": 4.0,   # Wider in strong trends

    # ---- Trailing Stop ----
    "trailing_enabled": True,
    "be_trigger_atr": 1.0,
    "trail_atr_mult": 1.2,
    "partial_tp_pct": 0.50,

    # ---- Filters ----
    "atr_pct_min": 0.15,
    "atr_pct_max": 0.85,
    "max_bars": 48,
    "cooldown": 6,

    # ---- Reversal (Divergence) ----
    "reversal_enabled": False,
    "div_lookback": 30,
    "div_min_slope": 0.5,
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
    cd = SBS_PARAMS["cooldown"]
    return (current_idx - _last_signal_bar) >= cd


def _register_signal(current_idx: int, was_trailing: bool = False) -> None:
    global _last_signal_bar, _last_exit_was_trailing
    _last_signal_bar = current_idx
    _last_exit_was_trailing = was_trailing


# ==================================================================
# STOCH RSI COMPUTATION
# ==================================================================

def compute_stoch_rsi(
    close: pd.Series, rsi_vals: pd.Series,
    period: int = 14, k_smooth: int = 3, d_smooth: int = 3,
) -> tuple:
    """Calcula Stochastic RSI. Returns (K, D)."""
    rsi_min = rsi_vals.rolling(window=period).min()
    rsi_max = rsi_vals.rolling(window=period).max()
    rsi_range = rsi_max - rsi_min
    stoch_rsi = pd.Series(0.0, index=close.index)
    mask = rsi_range > 0
    stoch_rsi[mask] = ((rsi_vals[mask] - rsi_min[mask]) / rsi_range[mask]) * 100.0
    stoch_k = stoch_rsi.rolling(window=k_smooth).mean()
    stoch_d = stoch_k.rolling(window=d_smooth).mean()
    return stoch_k, stoch_d


def compute_bbwp(bb_width: pd.Series, lookback: int = 100) -> pd.Series:
    """Calcula BBWP (Bollinger Band Width Percentile)."""
    bbwp = bb_width.rolling(window=lookback).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )
    return bbwp


def detect_rsi_divergence(
    close: pd.Series, rsi: pd.Series, idx: int,
    lookback: int = 20, min_slope_diff: float = 0.3,
) -> str:
    if idx < lookback + 2:
        return None
    price_window = close.iloc[idx - lookback:idx + 1].values
    rsi_window = rsi.iloc[idx - lookback:idx + 1].values
    if len(price_window) < lookback:
        return None
    peaks_p, peaks_r, troughs_p, troughs_r = [], [], [], []
    for j in range(1, len(price_window) - 1):
        if price_window[j] > price_window[j-1] and price_window[j] > price_window[j+1]:
            peaks_p.append((j, price_window[j])); peaks_r.append((j, rsi_window[j]))
        if price_window[j] < price_window[j-1] and price_window[j] < price_window[j+1]:
            troughs_p.append((j, price_window[j])); troughs_r.append((j, rsi_window[j]))
    if len(peaks_p) >= 2:
        lp, pp = peaks_p[-1][1], peaks_p[-2][1]
        lr, pr = peaks_r[-1][1], peaks_r[-2][1]
        if lp > pp and lr < pr and (pr - lr) >= min_slope_diff:
            return "bearish"
    if len(troughs_p) >= 2:
        lt, pt = troughs_p[-1][1], troughs_p[-2][1]
        lr, pr = troughs_r[-1][1], troughs_r[-2][1]
        if lt < pt and lr > pr and (lr - pr) >= min_slope_diff:
            return "bullish"
    return None


# ==================================================================
# SIGNAL BUILDER
# ==================================================================

def _build_signal(
    entry: float, sl: float, tp: float, row, is_long: bool,
    mode: str, conviction: str,
) -> Signal:
    ts = row.name if hasattr(row, "name") else row.index
    close = float(row["close"])
    atr = float(row.get("atr", 0))
    return Signal(
        type=SignalType.LONG if is_long else SignalType.SHORT,
        entry_price=close,
        stop_loss=sl,
        take_profit=tp,
        atr=atr,
        rsi=float(row.get("rsi", 0)),
        rsi_delta=float(row.get("rsi_delta", 0)),
        macd_hist=float(row.get("macd_hist", 0)),
        ema20=float(row.get("ema20", 0)),
        ema50=float(row.get("ema50", 0)),
        ema200=float(row.get("ema200", 0)),
        adx=float(row.get("adx", 0)),
        plus_di=float(row.get("plus_di", 0)),
        minus_di=float(row.get("minus_di", 0)),
        regime=f"sbs_{mode}_{conviction}",
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
        pullback_type=f"sbs_{mode}",
        ema50_slope=float(row.get("ema50_slope", 0)),
        timestamp=ts,
    )


# ==================================================================
# SHARED FILTERS
# ==================================================================

def _shared_filters(row) -> bool:
    p = SBS_PARAMS
    atr_pct = float(row.get("atr_percentile", 0.5))
    if not (p["atr_pct_min"] <= atr_pct <= p["atr_pct_max"]):
        return False
    regime = str(row.get("regime", "")).lower()
    if "volatile" in regime:
        return False
    if p["use_adx_filter"]:
        adx = float(row.get("adx", 0))
        if adx < p["adx_min"]:
            return False
    return True


# ==================================================================
# RETRACEMENT ENTRY LOGIC (v3: mean reversion after squeeze expansion)
# ==================================================================

def _eval_retrace_long(
    row, prev_row, bbwp_current: float, stoch_k: float, stoch_d: float,
    vol_ratio: float, was_squeezed: bool,
) -> Optional[tuple]:
    """
    Entrada LONG na retracao apos squeeze + expansao.

    Logica: apos squeeze, houve expansao (preco subiu). Agora o preco
    esta recuando de volta para a zona media das BB. Entramos no
    recuo com Stoch RSI saindo de sobrevenda.

    Condicoes:
    1. Squeeze recente (BBWP < threshold)
    2. BBWP em expansao (expansion confirmada)
    3. Preco na metade inferior das BB (retracao)
    4. Stoch RSI K cruzando D para cima na zona < 50 (reversao momentum)
    5. Close > EMA50 (tendencia de alta)
    6. Close > EMA200 (macro trend)
    7. RSI na zona de compra (30-55)
    """
    p = SBS_PARAMS
    close = float(row["close"])
    ema50 = float(row["ema50"])
    ema200 = float(row.get("ema200", 0))
    atr = float(row["atr"])
    bb_lower = float(row["bb_lower"])
    bb_upper = float(row["bb_upper"])
    bb_middle = float(row["bb_middle"])
    rsi = float(row.get("rsi", 50))
    adx = float(row.get("adx", 0))
    ema50_slope = float(row.get("ema50_slope", 0))

    if atr <= 0:
        return None

    # 1. Squeeze history
    if not was_squeezed:
        return None

    # 2. BBWP expansion
    if bbwp_current < p["bbwp_expansion_min"]:
        return None

    # 3. Price in lower half of BB (retracement zone)
    bb_range = bb_upper - bb_lower
    if bb_range <= 0:
        return None
    bb_pos = (close - bb_lower) / bb_range
    if bb_pos > 0.55:  # Not yet retraced enough
        return None

    # 4. Trend: close > EMA50 (uptrend)
    if close <= ema50:
        return None

    # 5. Macro: close > EMA200
    if p["use_ema200_filter"] and ema200 > 0 and close <= ema200:
        return None

    # 6. EMA50 slope positive (trend direction)
    if ema50_slope <= 0:
        return None

    # 7. RSI zone: oversold-ish but not extreme
    if rsi < p["rsi_long_min"] or rsi > p["rsi_long_max"]:
        return None

    # 8. Stoch RSI reversal: K crossing D upward in lower zone
    stoch_ok = False
    if stoch_k < 50 and stoch_d < 50:
        # K crosses above D
        if stoch_k > stoch_d and prev_row is not None:
            prev_k = float(prev_row.get("stoch_rsi_k", 0))
            prev_d = float(prev_row.get("stoch_rsi_d", 0))
            if prev_k <= prev_d:
                stoch_ok = True
        # Or K simply > D and both rising
        elif stoch_k > stoch_d and prev_row is not None:
            prev_k = float(prev_row.get("stoch_rsi_k", 0))
            if stoch_k > prev_k:  # K is rising
                stoch_ok = True
    if not stoch_ok:
        return None

    # 9. Volume: not dead
    if vol_ratio < p["vol_ratio_min"]:
        return None

    # Conviction
    conviction = "medium"
    if adx >= 25 and bb_pos < 0.35:
        conviction = "high"
    if stoch_k < 30:  # Deep oversold Stoch RSI reversal
        conviction = "high"

    # SL: below recent swing or ATR-based
    sl_mult = p["sl_atr_mult"]
    sl = close - sl_mult * atr
    if sl <= 0:
        return None
    # Don't place SL below lower BB (too far)
    if sl < bb_lower:
        sl = bb_lower - 0.1 * atr  # Just below lower BB

    # TP: BB middle or upper band
    tp_mult = p["tp_atr_mult_trending"] if conviction == "high" else p["tp_atr_mult"]
    tp = close + tp_mult * atr

    return (sl, tp, "retrace", conviction)


def _eval_retrace_short(
    row, prev_row, bbwp_current: float, stoch_k: float, stoch_d: float,
    vol_ratio: float, was_squeezed: bool,
) -> Optional[tuple]:
    """Mirror de _eval_retrace_long para SHORT."""
    p = SBS_PARAMS
    close = float(row["close"])
    ema50 = float(row["ema50"])
    ema200 = float(row.get("ema200", 0))
    atr = float(row["atr"])
    bb_lower = float(row["bb_lower"])
    bb_upper = float(row["bb_upper"])
    bb_middle = float(row["bb_middle"])
    rsi = float(row.get("rsi", 50))
    adx = float(row.get("adx", 0))
    ema50_slope = float(row.get("ema50_slope", 0))

    if atr <= 0:
        return None

    # 1. Squeeze history
    if not was_squeezed:
        return None

    # 2. BBWP expansion
    if bbwp_current < p["bbwp_expansion_min"]:
        return None

    # 3. Price in upper half of BB (retracement from high)
    bb_range = bb_upper - bb_lower
    if bb_range <= 0:
        return None
    bb_pos = (close - bb_lower) / bb_range
    if bb_pos < 0.45:  # Not yet retraced enough from high
        return None

    # 4. Trend: close < EMA50 (downtrend)
    if close >= ema50:
        return None

    # 5. Macro: close < EMA200
    if p["use_ema200_filter"] and ema200 > 0 and close >= ema200:
        return None

    # 6. EMA50 slope negative
    if ema50_slope >= 0:
        return None

    # 7. RSI zone
    if rsi < p["rsi_short_min"] or rsi > p["rsi_short_max"]:
        return None

    # 8. Stoch RSI reversal: K crossing D downward in upper zone
    stoch_ok = False
    if stoch_k > 50 and stoch_d > 50:
        if stoch_k < stoch_d and prev_row is not None:
            prev_k = float(prev_row.get("stoch_rsi_k", 0))
            prev_d = float(prev_row.get("stoch_rsi_d", 0))
            if prev_k >= prev_d:
                stoch_ok = True
        elif stoch_k < stoch_d and prev_row is not None:
            prev_k = float(prev_row.get("stoch_rsi_k", 0))
            if stoch_k < prev_k:
                stoch_ok = True
    if not stoch_ok:
        return None

    # 9. Volume
    if vol_ratio < p["vol_ratio_min"]:
        return None

    conviction = "medium"
    if adx >= 25 and bb_pos > 0.65:
        conviction = "high"
    if stoch_k > 70:
        conviction = "high"

    sl_mult = p["sl_atr_mult"]
    sl = close + sl_mult * atr

    tp_mult = p["tp_atr_mult_trending"] if conviction == "high" else p["tp_atr_mult"]
    tp = close - tp_mult * atr

    return (sl, tp, "retrace", conviction)


# ==================================================================
# BREAKOUT ENTRY (v3: only in strong trends after squeeze)
# ==================================================================

def _eval_breakout_long(
    row, prev_row, bbwp_current: float, stoch_k: float, stoch_d: float,
    vol_ratio: float, was_squeezed: bool,
) -> Optional[tuple]:
    """Breakout LONG: muito seletivo, apenas em fortes tendencias."""
    p = SBS_PARAMS
    close = float(row["close"])
    ema50 = float(row["ema50"])
    ema200 = float(row.get("ema200", 0))
    atr = float(row["atr"])
    bb_upper = float(row["bb_upper"])
    rsi = float(row.get("rsi", 50))
    adx = float(row.get("adx", 0))

    if atr <= 0:
        return None

    if not was_squeezed:
        return None
    if close <= bb_upper:
        return None
    if close <= ema50:
        return None
    if p["use_ema200_filter"] and ema200 > 0 and close <= ema200:
        return None
    if adx < 30:  # Strong trend only
        return None
    if rsi > p["rsi_long_max"]:
        return None
    if stoch_k < p["stoch_rsi_ob"]:
        return None
    if vol_ratio < 1.3:
        return None

    sl = close - p["sl_atr_mult"] * atr
    if sl <= 0:
        return None
    tp = close + p["tp_atr_mult_trending"] * atr
    return (sl, tp, "breakout", "high")


def _eval_breakout_short(
    row, prev_row, bbwp_current: float, stoch_k: float, stoch_d: float,
    vol_ratio: float, was_squeezed: bool,
) -> Optional[tuple]:
    """Breakout SHORT: muito seletivo."""
    p = SBS_PARAMS
    close = float(row["close"])
    ema50 = float(row["ema50"])
    ema200 = float(row.get("ema200", 0))
    atr = float(row["atr"])
    bb_lower = float(row["bb_lower"])
    rsi = float(row.get("rsi", 50))
    adx = float(row.get("adx", 0))

    if atr <= 0:
        return None

    if not was_squeezed:
        return None
    if close >= bb_lower:
        return None
    if close >= ema50:
        return None
    if p["use_ema200_filter"] and ema200 > 0 and close >= ema200:
        return None
    if adx < 30:
        return None
    if rsi < p["rsi_short_min"]:
        return None
    if stoch_k > p["stoch_rsi_os"]:
        return None
    if vol_ratio < 1.3:
        return None

    sl = close + p["sl_atr_mult"] * atr
    tp = close - p["tp_atr_mult_trending"] * atr
    return (sl, tp, "breakout", "high")


# ==================================================================
# REVERSAL ENTRY LOGIC
# ==================================================================

def _eval_reversal_long(row, divergence_type: str) -> Optional[tuple]:
    p = SBS_PARAMS
    if divergence_type != "bullish":
        return None
    close = float(row["close"])
    atr = float(row["atr"])
    bb_lower = float(row["bb_lower"])
    bb_upper = float(row["bb_upper"])
    if atr <= 0:
        return None
    bb_range = bb_upper - bb_lower
    if bb_range <= 0:
        return None
    bb_pos = (close - bb_lower) / bb_range
    if bb_pos > 0.20:
        return None
    sl = close - p["sl_atr_mult"] * atr
    if sl <= 0:
        return None
    tp = close + p["tp_atr_mult"] * atr
    return (sl, tp, "reversal", "medium")


def _eval_reversal_short(row, divergence_type: str) -> Optional[tuple]:
    p = SBS_PARAMS
    if divergence_type != "bearish":
        return None
    close = float(row["close"])
    atr = float(row["atr"])
    bb_lower = float(row["bb_lower"])
    bb_upper = float(row["bb_upper"])
    if atr <= 0:
        return None
    bb_range = bb_upper - bb_lower
    if bb_range <= 0:
        return None
    bb_pos = (close - bb_lower) / bb_range
    if bb_pos < 0.80:
        return None
    sl = close + p["sl_atr_mult"] * atr
    tp = close - p["tp_atr_mult"] * atr
    return (sl, tp, "reversal", "medium")


# ==================================================================
# CORE EVALUATION (BACKTEST)
# ==================================================================

def evaluate_sbs_row(
    row, prev_row, bar_index: int,
    stoch_k: float, stoch_d: float,
    bbwp_current: float, vol_ratio: float, was_squeezed: bool,
    divergence: Optional[str],
    profile=None,
) -> Optional[tuple]:
    if not _check_cooldown(bar_index):
        return None
    if not _shared_filters(row):
        return None

    p = SBS_PARAMS

    # Try RETRACE LONG (primary mean-reversion signal)
    result = _eval_retrace_long(
        row, prev_row, bbwp_current, stoch_k, stoch_d, vol_ratio, was_squeezed
    )
    if result is not None:
        sl, tp, mode, conviction = result
        signal = _build_signal(float(row["close"]), sl, tp, row, True, mode, conviction)
        _register_signal(bar_index)
        return (signal, mode, conviction, p["trail_atr_mult"], p["sl_atr_mult"])

    # Try RETRACE SHORT
    result = _eval_retrace_short(
        row, prev_row, bbwp_current, stoch_k, stoch_d, vol_ratio, was_squeezed
    )
    if result is not None:
        sl, tp, mode, conviction = result
        signal = _build_signal(float(row["close"]), sl, tp, row, False, mode, conviction)
        _register_signal(bar_index)
        return (signal, mode, conviction, p["trail_atr_mult"], p["sl_atr_mult"])

    # Try BREAKOUT (secondary, very selective)
    result = _eval_breakout_long(
        row, prev_row, bbwp_current, stoch_k, stoch_d, vol_ratio, was_squeezed
    )
    if result is not None:
        sl, tp, mode, conviction = result
        signal = _build_signal(float(row["close"]), sl, tp, row, True, mode, conviction)
        _register_signal(bar_index)
        return (signal, mode, conviction, p["trail_atr_mult"], p["sl_atr_mult"])

    result = _eval_breakout_short(
        row, prev_row, bbwp_current, stoch_k, stoch_d, vol_ratio, was_squeezed
    )
    if result is not None:
        sl, tp, mode, conviction = result
        signal = _build_signal(float(row["close"]), sl, tp, row, False, mode, conviction)
        _register_signal(bar_index)
        return (signal, mode, conviction, p["trail_atr_mult"], p["sl_atr_mult"])

    # Try REVERSAL (divergence-based)
    if p["reversal_enabled"] and divergence is not None:
        result = _eval_reversal_long(row, divergence)
        if result is not None:
            sl, tp, mode, conviction = result
            signal = _build_signal(float(row["close"]), sl, tp, row, True, mode, conviction)
            _register_signal(bar_index)
            return (signal, mode, conviction, p["trail_atr_mult"], p["sl_atr_mult"])

        result = _eval_reversal_short(row, divergence)
        if result is not None:
            sl, tp, mode, conviction = result
            signal = _build_signal(float(row["close"]), sl, tp, row, False, mode, conviction)
            _register_signal(bar_index)
            return (signal, mode, conviction, p["trail_atr_mult"], p["sl_atr_mult"])

    return None


# ==================================================================
# LIVE TRADING ENTRY
# ==================================================================

def evaluate_sbs(df, profile=None) -> Optional[Signal]:
    if len(df) < 2:
        return None
    stoch_k, stoch_d = compute_stoch_rsi(df["close"], df["rsi"])
    bbwp = compute_bbwp(df["bb_width"])
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    idx = len(df) - 1
    curr_k = float(stoch_k.iloc[-1]) if not pd.isna(stoch_k.iloc[-1]) else 0
    curr_d = float(stoch_d.iloc[-1]) if not pd.isna(stoch_d.iloc[-1]) else 0
    curr_bbwp = float(bbwp.iloc[-1]) if not pd.isna(bbwp.iloc[-1]) else 50
    curr_vol_ratio = float(curr["volume"]) / float(curr["volume_sma20"]) if curr["volume_sma20"] > 0 else 0
    was_squeezed = curr_bbwp < SBS_PARAMS["bbwp_squeeze_threshold"] * 2
    divergence = detect_rsi_divergence(df["close"], df["rsi"], idx)
    result = evaluate_sbs_row(
        curr, prev, idx, curr_k, curr_d, curr_bbwp,
        curr_vol_ratio, was_squeezed, divergence, profile,
    )
    if result is not None:
        signal, mode, conviction, _, _ = result
        logger.info(
            "SIGNAL SBS v3 %s mode=%s conv=%s | entry=%.2f SL=%.2f TP=%.2f",
            signal.type.value, mode, conviction, signal.entry_price,
            signal.stop_loss, signal.take_profit,
        )
        return signal
    return None
