"""
strategy.py
-----------
Logica de validacao das condicoes de entrada CTEV v22.0 Multi-Strategy.

v22.0 -- AGGRESSIVE HIGH-FREQUENCY OVERHAUL

  7 estrategias ativas (todas reativadas com SL mais largo):
    1. CTEV Trend (pullback + momentum) -- tendencia com ADX > 18
    2. Squeeze Breakout (BBWP squeeze + breakout) -- BBWP < 50
    3. RSI Reversal (RSI extremo em tendencia) -- RSI < 42 /> 58
    4. Range Trader (BB bounce em lateral) -- ADX < 30, SL 2.5x
    5. RSI Extremes (RSI extremo + virada) -- SL 2.0x
    6. Scalp (quick intraday) -- SL 1.5x, TP 3.0x
    7. EMA Bounce (EMA20 touch + recover) -- SL 2.0x, TP 4.5x

  v22.0 Changes vs v21.0:
  - MACD_HIST_FILTER OFF -- era o maior bloqueador de sinais
  - DI_DIRECTION_FILTER OFF -- EMA stack ja confirma direcao
  - ADX_MIN 18 (de 25) -- mais oportunidades
  - RSI LONG 38-72 (de 46-64) -- zona muito mais ampla
  - RSI SHORT 28-62 (de 36-54) -- zona muito mais ampla
  - SL 2.0x ATR (de 3.5x) -- perdas menores por trade
  - TP 5.5x ATR (de 7.5x) -- TP mais alcançavel, R:R real ~2.5:1
  - Squeeze BBWP < 50 (de 35) -- mais sinais de squeeze
  - Range Trader REATIVADO com SL 2.5x (de 1.5x)
  - RSI Extremes REATIVADO com SL 2.0x (de 1.5x)
  - Scalp REATIVADO com SL 1.5x, TP 3.0x
  - EMA Bounce REATIVADO com SL 2.0x, TP 4.5x
  - Transition regime ON -- mais sinais em mudancas de tendencia
  - EMA proximity REATIVADO -- mais pullbacks detectados
  - BB touch REATIVADO -- pullbacks profundos
  - MAX_BARS aumentados para dar tempo de atingir TP
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from strategy_profiles import StrategyProfile

logger = logging.getLogger(__name__)


class SignalType(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass(frozen=True)
class Signal:
    """Representa um sinal de trade gerado pela estrategia CTEV v22.0."""
    type: SignalType
    entry_price: float
    stop_loss: float
    take_profit: float
    atr: float
    rsi: float
    rsi_delta: float
    macd_hist: float
    ema20: float
    ema50: float
    ema200: float
    adx: float
    plus_di: float
    minus_di: float
    regime: str
    bb_lower: float
    bb_upper: float
    bb_width: float
    bb_squeeze_pct: float
    volume: float
    volume_sma20: float
    volume_sma50: float
    atr_percentile: float
    fib_0382: float
    fib_0500: float
    fib_0618: float
    fib_direction: int
    fib_proximity: float
    pullback_type: str
    ema50_slope: float
    timestamp: pd.Timestamp
    entry_type: str = "ctev_pullback"
    max_bars: int = 168

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "atr": self.atr,
            "rsi": self.rsi,
            "rsi_delta": round(self.rsi_delta, 2),
            "macd_hist": round(self.macd_hist, 4),
            "ema20": self.ema20,
            "ema50": self.ema50,
            "ema200": self.ema200,
            "adx": round(self.adx, 2),
            "plus_di": round(self.plus_di, 2),
            "minus_di": round(self.minus_di, 2),
            "regime": self.regime,
            "bb_width": round(self.bb_width, 4),
            "bb_squeeze_pct": round(self.bb_squeeze_pct, 4),
            "atr_percentile": self.atr_percentile,
            "fib_0382": self.fib_0382,
            "fib_0500": self.fib_0500,
            "fib_0618": self.fib_0618,
            "fib_direction": self.fib_direction,
            "fib_proximity": round(self.fib_proximity, 3) if not pd.isna(self.fib_proximity) else None,
            "pullback_type": self.pullback_type,
            "entry_type": self.entry_type,
            "max_bars": self.max_bars,
            "ema50_slope": round(self.ema50_slope, 6),
            "timestamp": str(self.timestamp),
        }


# ═══════════════════════════════════════════════════════════════════
# v22.0: AGGRESSIVE MULTI-STRATEGY ENGINE
# Alvo: 40%/30d, 80%/90d, 120%/180d, 160%/365d, 230%/730d
# ═══════════════════════════════════════════════════════════════════

# REGIME FILTER
ADX_MIN = 18.0
ALLOW_TRANSITION = True

# RSI zonas (v22.0: muito mais amplas)
RSI_LONG_MIN = 38.0
RSI_LONG_MAX = 72.0
RSI_SHORT_MIN = 28.0
RSI_SHORT_MAX = 62.0

# RSI Delta (desabilitado)
RSI_DELTA_LONG_MIN = -5.0
RSI_DELTA_SHORT_MAX = 5.0

# Volume (DESABILITADO)
VOLUME_CONFIRM = False
VOLUME_SMA_RATIO = 0.30

# Fibonacci tolerancia
FIB_TOLERANCE_PCT = 0.035

# ATR Percentile filter
ATR_PCT_MIN = 0.08
ATR_PCT_MAX = 0.95

# Bollinger Bandwidth
BB_WIDTH_MIN = 0.0
BB_WIDTH_MAX = 999.0

# EMA proximity (v22.0: REATIVADO)
EMA20_PROXIMITY_PCT = 0.015
EMA50_PROXIMITY_PCT = 0.020

# EMA Slope
EMA50_SLOPE_MIN = -1.0

# CTEV Pullback SL/TP (v22.0: R:R 2.75:1)
SL_ATR_MULT = 2.00
TP_ATR_MULT = 5.50

# FILTROS DE CONFLUENCIA (v22.0: TODOS OFF exceto BBWP squeeze bonus)
DI_DIRECTION_FILTER = False
MACD_HIST_FILTER = False
OBV_TREND_FILTER = False
STOCH_RSI_FILTER = False
BBWP_SQUEEZE_BONUS = True

# BB TOUCH (v22.0: REATIVADO)
BB_TOUCH_PCT = 0.015

# Momentum
MOMENTUM_ADX_MIN = 18.0
MOMENTUM_SL_ATR_MULT = 1.80
MOMENTUM_TP_ATR_MULT = 4.50
MOMENTUM_MAX_BARS = 120

# EMA Bounce (v22.0: REATIVADO)
EMA_BOUNCE_SL_ATR_MULT = 2.00
EMA_BOUNCE_TP_ATR_MULT = 4.50
EMA_BOUNCE_MAX_BARS = 96

# Squeeze Breakout
SQUEEZE_SL_ATR_MULT = 2.00
SQUEEZE_TP_ATR_MULT = 5.00
SQUEEZE_MAX_BARS = 120
SQUEEZE_BBWP_THRESHOLD = 50.0

# RSI Reversal
RSI_REV_SL_ATR_MULT = 2.00
RSI_REV_TP_ATR_MULT = 5.50
RSI_REV_MAX_BARS = 120

# Range Trader (v22.0: REATIVADO com SL mais largo)
RANGE_SL_ATR_MULT = 2.50
RANGE_TP_ATR_MULT = 5.00
RANGE_MAX_BARS = 72
RANGE_BB_TOUCH_THRESHOLD = 0.020
RANGE_ADX_MAX = 30.0
RANGE_RSI_OVERSOLD = 45.0
RANGE_RSI_OVERBOUGHT = 55.0

# Scalp (v22.0: REATIVADO)
SCALP_SL_ATR_MULT = 1.50
SCALP_TP_ATR_MULT = 3.00
SCALP_MAX_BARS = 24
SCALP_RSI_LONG_MIN = 35.0
SCALP_RSI_SHORT_MAX = 65.0
SCALP_RSI_DELTA_TRIGGER = 0.5

# RSI Extremes (v22.0: REATIVADO com SL mais largo)
RSI_EXT_SL_ATR_MULT = 2.00
RSI_EXT_TP_ATR_MULT = 4.50
RSI_EXT_MAX_BARS = 48
RSI_EXT_OVERSOLD = 32.0
RSI_EXT_OVERBOUGHT = 68.0
RSI_EXT_RSI_DELTA_MIN = 0.2


def _price_near_fib(
    price: float, fib_level: float,
    tolerance_pct: float = FIB_TOLERANCE_PCT,
) -> bool:
    if pd.isna(fib_level) or fib_level <= 0:
        return False
    tolerance = price * tolerance_pct
    return abs(price - fib_level) <= tolerance


def _in_fib_zone(price: float, fib_0382: float, fib_0618: float, direction: int) -> bool:
    if pd.isna(fib_0382) or pd.isna(fib_0618):
        return False
    if direction == 1:
        return fib_0618 <= price <= fib_0382
    elif direction == -1:
        return fib_0382 <= price <= fib_0618
    return False


def _make_signal(
    sig_type: SignalType, entry: float, sl: float, tp: float,
    atr: float, row: pd.Series, pullback_type: str,
    entry_type: str, max_bars: int,
) -> Optional[Signal]:
    if sl <= 0 and sig_type == SignalType.LONG:
        return None
    return Signal(
        type=sig_type, entry_price=entry, stop_loss=sl,
        take_profit=tp, atr=atr,
        rsi=float(row["rsi"]),
        rsi_delta=float(row.get("rsi_delta", 0.0)),
        macd_hist=float(row.get("macd_hist", 0.0)),
        ema20=float(row["ema20"]),
        ema50=float(row["ema50"]),
        ema200=float(row["ema200"]),
        adx=float(row.get("adx", 0.0)),
        plus_di=float(row.get("plus_di", 0.0)),
        minus_di=float(row.get("minus_di", 0.0)),
        regime=str(row.get("regime", "")),
        bb_lower=float(row["bb_lower"]),
        bb_upper=float(row["bb_upper"]),
        bb_width=float(row.get("bb_width", 0.0)),
        bb_squeeze_pct=float(row.get("bb_squeeze_pct", 0.5)),
        volume=float(row["volume"]),
        volume_sma20=float(row["volume_sma20"]),
        volume_sma50=float(row.get("volume_sma50", 0.0)),
        atr_percentile=float(row.get("atr_percentile", 0.5)),
        fib_0382=0.0, fib_0500=0.0, fib_0618=0.0,
        fib_direction=0, fib_proximity=0.0,
        pullback_type=pullback_type,
        entry_type=entry_type,
        max_bars=max_bars,
        ema50_slope=float(row.get("ema50_slope", 0.0)),
        timestamp=row.name,
    )


def evaluate_long(
    row: pd.Series, profile: Optional[StrategyProfile] = None
) -> Optional[Signal]:
    close = float(row["close"])
    low = float(row["low"])
    high = float(row["high"])
    ema20 = float(row["ema20"])
    ema50 = float(row["ema50"])
    ema200 = float(row["ema200"])
    bb_lower = float(row["bb_lower"])
    bb_upper = float(row["bb_upper"])
    rsi = float(row["rsi"])
    rsi_delta = float(row.get("rsi_delta", 0.0))
    macd_val = float(row.get("macd", 0.0))
    macd_sig = float(row.get("macd_signal", 0.0))
    macd_hist = float(row.get("macd_hist", 0.0))
    volume = float(row["volume"])
    volume_sma20 = float(row["volume_sma20"])
    volume_sma50 = float(row.get("volume_sma50", 0.0))
    atr = float(row["atr"])
    atr_pct = float(row.get("atr_percentile", 0.5))
    bb_w = float(row.get("bb_width", 0.0))
    bb_sq_pct = float(row.get("bb_squeeze_pct", 0.5))
    adx = float(row.get("adx", 0.0))
    plus_di = float(row.get("plus_di", 0.0))
    minus_di = float(row.get("minus_di", 0.0))
    regime = str(row.get("regime", ""))
    ema50_slope = float(row.get("ema50_slope", 0.0))
    fib_0382 = float(row.get("fib_0382", float("nan")))
    fib_0500 = float(row.get("fib_0500", float("nan")))
    fib_0618 = float(row.get("fib_0618", float("nan")))
    fib_dir = int(row.get("fib_direction", 0))
    fib_prox = float(row.get("fib_proximity", float("nan")))
    ts = row.name

    _adx_min = profile.adx_min if profile else ADX_MIN
    _allow_trans = profile.allow_transition if profile else ALLOW_TRANSITION
    _rsi_l_min = profile.rsi_long_min if profile else RSI_LONG_MIN
    _rsi_l_max = profile.rsi_long_max if profile else RSI_LONG_MAX
    _fib_tol = profile.fib_tolerance_pct if profile else FIB_TOLERANCE_PCT
    _slope_min = profile.ema50_slope_min if profile else EMA50_SLOPE_MIN
    _vol_confirm = profile.volume_confirm if profile else VOLUME_CONFIRM
    _vol_ratio = profile.volume_sma_ratio if profile else VOLUME_SMA_RATIO
    _atr_pct_min = profile.atr_pct_min if profile else ATR_PCT_MIN
    _atr_pct_max = profile.atr_pct_max if profile else ATR_PCT_MAX
    _sl_mult = profile.sl_atr_mult if profile else SL_ATR_MULT
    _tp_mult = profile.tp_atr_mult if profile else TP_ATR_MULT

    _bbwp = float(row.get("bbwp", 50.0))
    if BBWP_SQUEEZE_BONUS and not pd.isna(_bbwp) and _bbwp < 20.0:
        _rsi_l_min = max(0, _rsi_l_min - 5)
        _rsi_l_max = _rsi_l_max + 5

    # 1. REGIME (v22.0: transition ON)
    if regime == "trending_up":
        if pd.isna(adx) or adx < _adx_min:
            return None
    elif regime == "transition":
        if not _allow_trans:
            return None
    else:
        return None

    # 2. TENDENCIA: Dual EMA
    if not (close > ema50 and ema50 > ema200):
        return None

    # 3. SLOPE (v22.0: permissivo)
    if pd.isna(ema50_slope) or ema50_slope <= _slope_min:
        return None

    # v22.0: DI filter OFF, MACD filter OFF

    # 4. PULLBACK: Fibonacci + EMA touch + BB touch
    pullback_type = None

    in_fib = _in_fib_zone(close, fib_0382, fib_0618, fib_dir)
    if in_fib and fib_dir == 1:
        pullback_type = "fibonacci"
    else:
        if fib_dir == 1:
            if (_price_near_fib(low, fib_0382, _fib_tol) or
                    _price_near_fib(low, fib_0500, _fib_tol) or
                    _price_near_fib(low, fib_0618, _fib_tol)):
                pullback_type = "fibonacci"

    _ema20_prox = profile.ema20_proximity_pct if profile else EMA20_PROXIMITY_PCT
    if pullback_type is None and _ema20_prox > 0:
        if close <= ema20 and close >= ema20 * (1 - _ema20_prox):
            pullback_type = "ema20_proximity"

    _ema50_prox = profile.ema50_proximity_pct if profile else EMA50_PROXIMITY_PCT
    if pullback_type is None and _ema50_prox > 0:
        if close <= ema50 and close >= ema50 * (1 - _ema50_prox):
            pullback_type = "ema50_proximity"

    if pullback_type is None and BB_TOUCH_PCT > 0:
        if close <= bb_lower * (1 + BB_TOUCH_PCT):
            pullback_type = "bb_lower_touch"

    if pullback_type is not None:
        _entry_type = "ctev_pullback"
        _max_bars_use = 168
        _sl_use = _sl_mult
        _tp_use = _tp_mult
    else:
        pullback_type = "none"
        _entry_type = "ctev_momentum"
        _max_bars_use = MOMENTUM_MAX_BARS
        _sl_use = MOMENTUM_SL_ATR_MULT
        _tp_use = MOMENTUM_TP_ATR_MULT

    # 5. RSI
    if not (_rsi_l_min <= rsi <= _rsi_l_max):
        return None

    if pullback_type == "none" and rsi < 45.0:
        return None

    # 6. VOLUME: OFF

    # 7. ATR
    if not (_atr_pct_min <= atr_pct <= _atr_pct_max):
        return None

    entry = close
    stop_loss = entry - (_sl_use * atr)
    take_profit = entry + (_tp_use * atr)

    if stop_loss <= 0:
        return None

    logger.info(
        "SINAL LONG v22 | entry=%.2f SL=%.2f TP=%.2f ATR=%.2f "
        "RSI=%.1f ADX=%.1f regime=%s pullback=%s",
        entry, stop_loss, take_profit, atr, rsi, adx, regime, pullback_type,
    )

    return Signal(
        type=SignalType.LONG,
        entry_price=entry, stop_loss=stop_loss, take_profit=take_profit,
        atr=atr, rsi=rsi, rsi_delta=rsi_delta, macd_hist=macd_hist,
        ema20=ema20, ema50=ema50, ema200=ema200,
        adx=adx, plus_di=plus_di, minus_di=minus_di, regime=regime,
        bb_lower=bb_lower, bb_upper=bb_upper, bb_width=bb_w,
        bb_squeeze_pct=bb_sq_pct, volume=volume,
        volume_sma20=volume_sma20, volume_sma50=volume_sma50,
        atr_percentile=atr_pct,
        fib_0382=fib_0382 if not pd.isna(fib_0382) else 0.0,
        fib_0500=fib_0500 if not pd.isna(fib_0500) else 0.0,
        fib_0618=fib_0618 if not pd.isna(fib_0618) else 0.0,
        fib_direction=fib_dir, fib_proximity=fib_prox,
        pullback_type=pullback_type, entry_type=_entry_type,
        max_bars=_max_bars_use, ema50_slope=ema50_slope, timestamp=ts,
    )


def evaluate_short(
    row: pd.Series, profile: Optional[StrategyProfile] = None
) -> Optional[Signal]:
    close = float(row["close"])
    low = float(row["low"])
    high = float(row["high"])
    ema20 = float(row["ema20"])
    ema50 = float(row["ema50"])
    ema200 = float(row["ema200"])
    bb_lower = float(row["bb_lower"])
    bb_upper = float(row["bb_upper"])
    rsi = float(row["rsi"])
    rsi_delta = float(row.get("rsi_delta", 0.0))
    macd_val = float(row.get("macd", 0.0))
    macd_sig = float(row.get("macd_signal", 0.0))
    macd_hist = float(row.get("macd_hist", 0.0))
    volume = float(row["volume"])
    volume_sma20 = float(row["volume_sma20"])
    volume_sma50 = float(row.get("volume_sma50", 0.0))
    atr = float(row["atr"])
    atr_pct = float(row.get("atr_percentile", 0.5))
    bb_w = float(row.get("bb_width", 0.0))
    bb_sq_pct = float(row.get("bb_squeeze_pct", 0.5))
    adx = float(row.get("adx", 0.0))
    plus_di = float(row.get("plus_di", 0.0))
    minus_di = float(row.get("minus_di", 0.0))
    regime = str(row.get("regime", ""))
    ema50_slope = float(row.get("ema50_slope", 0.0))
    fib_0382 = float(row.get("fib_0382", float("nan")))
    fib_0500 = float(row.get("fib_0500", float("nan")))
    fib_0618 = float(row.get("fib_0618", float("nan")))
    fib_dir = int(row.get("fib_direction", 0))
    fib_prox = float(row.get("fib_proximity", float("nan")))
    ts = row.name

    _adx_min = profile.adx_min if profile else ADX_MIN
    _allow_trans = profile.allow_transition if profile else ALLOW_TRANSITION
    _rsi_s_min = profile.rsi_short_min if profile else RSI_SHORT_MIN
    _rsi_s_max = profile.rsi_short_max if profile else RSI_SHORT_MAX
    _fib_tol = profile.fib_tolerance_pct if profile else FIB_TOLERANCE_PCT
    _slope_min = profile.ema50_slope_min if profile else EMA50_SLOPE_MIN
    _vol_confirm = profile.volume_confirm if profile else VOLUME_CONFIRM
    _vol_ratio = profile.volume_sma_ratio if profile else VOLUME_SMA_RATIO
    _atr_pct_min = profile.atr_pct_min if profile else ATR_PCT_MIN
    _atr_pct_max = profile.atr_pct_max if profile else ATR_PCT_MAX
    _sl_mult = profile.sl_atr_mult if profile else SL_ATR_MULT
    _tp_mult = profile.tp_atr_mult if profile else TP_ATR_MULT

    _bbwp = float(row.get("bbwp", 50.0))
    if BBWP_SQUEEZE_BONUS and not pd.isna(_bbwp) and _bbwp < 20.0:
        _rsi_s_min = max(0, _rsi_s_min - 5)
        _rsi_s_max = _rsi_s_max + 5

    if regime == "trending_down":
        if pd.isna(adx) or adx < _adx_min:
            return None
    elif regime == "transition":
        if not _allow_trans:
            return None
    else:
        return None

    if not (close < ema50 and ema50 < ema200):
        return None

    if pd.isna(ema50_slope) or ema50_slope >= -_slope_min:
        return None

    pullback_type = None

    in_fib = _in_fib_zone(close, fib_0382, fib_0618, fib_dir)
    if in_fib and fib_dir == -1:
        pullback_type = "fibonacci"
    else:
        if fib_dir == -1:
            if (_price_near_fib(high, fib_0382, _fib_tol) or
                    _price_near_fib(high, fib_0500, _fib_tol) or
                    _price_near_fib(high, fib_0618, _fib_tol)):
                pullback_type = "fibonacci"

    _ema20_prox = profile.ema20_proximity_pct if profile else EMA20_PROXIMITY_PCT
    if pullback_type is None and _ema20_prox > 0:
        if close >= ema20 and close <= ema20 * (1 + _ema20_prox):
            pullback_type = "ema20_proximity"

    _ema50_prox = profile.ema50_proximity_pct if profile else EMA50_PROXIMITY_PCT
    if pullback_type is None and _ema50_prox > 0:
        if close >= ema50 and close <= ema50 * (1 + _ema50_prox):
            pullback_type = "ema50_proximity"

    if pullback_type is None and BB_TOUCH_PCT > 0:
        if close >= bb_upper * (1 - BB_TOUCH_PCT):
            pullback_type = "bb_upper_touch"

    if pullback_type is not None:
        _entry_type_s = "ctev_pullback"
        _max_bars_use_s = 168
        _sl_use_s = _sl_mult
        _tp_use_s = _tp_mult
    else:
        pullback_type = "none"
        _entry_type_s = "ctev_momentum"
        _max_bars_use_s = MOMENTUM_MAX_BARS
        _sl_use_s = MOMENTUM_SL_ATR_MULT
        _tp_use_s = MOMENTUM_TP_ATR_MULT

    if not (_rsi_s_min <= rsi <= _rsi_s_max):
        return None

    if pullback_type == "none" and rsi > 55.0:
        return None

    if _vol_confirm and (not pd.isna(volume_sma50) and volume_sma50 > 0):
        if volume < volume_sma50 * _vol_ratio:
            return None

    if not (_atr_pct_min <= atr_pct <= _atr_pct_max):
        return None

    entry = close
    stop_loss = entry + (_sl_use_s * atr)
    take_profit = entry - (_tp_use_s * atr)

    logger.info(
        "SINAL SHORT v22 | entry=%.2f SL=%.2f TP=%.2f ATR=%.2f "
        "RSI=%.1f ADX=%.1f regime=%s pullback=%s",
        entry, stop_loss, take_profit, atr, rsi, adx, regime, pullback_type,
    )

    return Signal(
        type=SignalType.SHORT,
        entry_price=entry, stop_loss=stop_loss, take_profit=take_profit,
        atr=atr, rsi=rsi, rsi_delta=rsi_delta, macd_hist=macd_hist,
        ema20=ema20, ema50=ema50, ema200=ema200,
        adx=adx, plus_di=plus_di, minus_di=minus_di, regime=regime,
        bb_lower=bb_lower, bb_upper=bb_upper, bb_width=bb_w,
        bb_squeeze_pct=bb_sq_pct, volume=volume,
        volume_sma20=volume_sma20, volume_sma50=volume_sma50,
        atr_percentile=atr_pct,
        fib_0382=fib_0382 if not pd.isna(fib_0382) else 0.0,
        fib_0500=fib_0500 if not pd.isna(fib_0500) else 0.0,
        fib_0618=fib_0618 if not pd.isna(fib_0618) else 0.0,
        fib_direction=fib_dir, fib_proximity=fib_prox,
        pullback_type=pullback_type, entry_type=_entry_type_s,
        max_bars=_max_bars_use_s, ema50_slope=ema50_slope, timestamp=ts,
    )


def evaluate_signal(
    df_ind: pd.DataFrame, profile: Optional[StrategyProfile] = None
) -> Optional[Signal]:
    if df_ind.empty:
        return None
    return evaluate_row_signals(df_ind.iloc[-1], profile=profile)


def evaluate_row_signals(
    row: pd.Series, profile: Optional[StrategyProfile] = None
) -> Optional[Signal]:
    """v22.0: Multi-strategy chain -- 7 estrategias ativas.

    Prioridade (mais restritivas primeiro):
      1. CTEV Pullback/Momentum (trend-following)
      2. Squeeze Breakout (BBWP < 50)
      3. RSI Reversal (RSI extremo em tendencia)
      4. EMA Bounce (EMA20 touch + recover)
      5. Range Trader (BB bounce em lateral)
      6. RSI Extremes (RSI extremo + virada)
      7. Scalp (quick intraday)
    """
    # 1. CTEV
    signal = evaluate_long(row, profile=profile)
    if signal is not None:
        return signal
    signal = evaluate_short(row, profile=profile)
    if signal is not None:
        return signal

    # 2. Squeeze Breakout
    signal = evaluate_squeeze_breakout_long(row)
    if signal is not None:
        return signal
    signal = evaluate_squeeze_breakout_short(row)
    if signal is not None:
        return signal

    # 3. RSI Reversal
    signal = evaluate_rsi_reversal_long(row)
    if signal is not None:
        return signal
    signal = evaluate_rsi_reversal_short(row)
    if signal is not None:
        return signal

    # 4. EMA Bounce (v22.0: REATIVADO)
    signal = evaluate_ema_bounce_long(row)
    if signal is not None:
        return signal
    signal = evaluate_ema_bounce_short(row)
    if signal is not None:
        return signal

    # 5. Range Trader (v22.0: REATIVADO)
    signal = evaluate_range_long(row)
    if signal is not None:
        return signal
    signal = evaluate_range_short(row)
    if signal is not None:
        return signal

    # 6. RSI Extremes (v22.0: REATIVADO)
    signal = evaluate_rsi_extremes_long(row)
    if signal is not None:
        return signal
    signal = evaluate_rsi_extremes_short(row)
    if signal is not None:
        return signal

    # 7. Scalp (v22.0: REATIVADO)
    signal = evaluate_scalp_long(row)
    if signal is not None:
        return signal
    signal = evaluate_scalp_short(row)
    if signal is not None:
        return signal

    return None


def evaluate_squeeze_breakout_long(row: pd.Series) -> Optional[Signal]:
    """Squeeze Breakout LONG -- BBWP squeeze + breakout superior.
    v22.0: BBWP < 50, SL 2.0x, TP 5.0x, max 120 bars.
    """
    close = float(row["close"])
    bb_upper = float(row["bb_upper"])
    bbwp = float(row.get("bbwp", 50.0))
    rsi = float(row["rsi"])
    atr = float(row["atr"])
    atr_pct = float(row.get("atr_percentile", 0.5))

    if pd.isna(bbwp) or bbwp >= SQUEEZE_BBWP_THRESHOLD:
        return None
    if close <= bb_upper:
        return None
    if rsi <= 40.0:
        return None
    if not (0.08 <= atr_pct <= 0.95):
        return None
    if atr <= 0:
        return None

    entry = close
    sl = entry - (SQUEEZE_SL_ATR_MULT * atr)
    tp = entry + (SQUEEZE_TP_ATR_MULT * atr)
    if sl <= 0:
        return None

    return _make_signal(
        SignalType.LONG, entry, sl, tp, atr, row,
        pullback_type="squeeze_breakout", entry_type="squeeze_breakout",
        max_bars=SQUEEZE_MAX_BARS,
    )


def evaluate_squeeze_breakout_short(row: pd.Series) -> Optional[Signal]:
    """Squeeze Breakout SHORT -- BBWP squeeze + breakout inferior.
    v22.0: BBWP < 50, SL 2.0x, TP 5.0x, max 120 bars.
    """
    close = float(row["close"])
    bb_lower = float(row["bb_lower"])
    bbwp = float(row.get("bbwp", 50.0))
    rsi = float(row["rsi"])
    atr = float(row["atr"])
    atr_pct = float(row.get("atr_percentile", 0.5))

    if pd.isna(bbwp) or bbwp >= SQUEEZE_BBWP_THRESHOLD:
        return None
    if close >= bb_lower:
        return None
    if rsi >= 60.0:
        return None
    if not (0.08 <= atr_pct <= 0.95):
        return None
    if atr <= 0:
        return None

    entry = close
    sl = entry + (SQUEEZE_SL_ATR_MULT * atr)
    tp = entry - (SQUEEZE_TP_ATR_MULT * atr)

    return _make_signal(
        SignalType.SHORT, entry, sl, tp, atr, row,
        pullback_type="squeeze_breakout", entry_type="squeeze_breakout",
        max_bars=SQUEEZE_MAX_BARS,
    )


def evaluate_rsi_reversal_long(row: pd.Series) -> Optional[Signal]:
    """RSI Reversal LONG -- RSI sobrevendido em tendencia de alta.
    v22.0: RSI < 42 (de 38), SL 2.0x, TP 5.5x, max 120 bars.
    """
    close = float(row["close"])
    ema50 = float(row["ema50"])
    ema200 = float(row["ema200"])
    rsi = float(row["rsi"])
    rsi_delta = float(row.get("rsi_delta", 0.0))
    atr = float(row["atr"])
    atr_pct = float(row.get("atr_percentile", 0.5))

    if not (ema50 > ema200):
        return None
    if rsi >= 42.0:
        return None
    if rsi_delta <= 0:
        return None
    if close <= ema50:
        return None
    if not (0.08 <= atr_pct <= 0.95):
        return None
    if atr <= 0:
        return None

    entry = close
    sl = entry - (RSI_REV_SL_ATR_MULT * atr)
    tp = entry + (RSI_REV_TP_ATR_MULT * atr)
    if sl <= 0:
        return None

    return _make_signal(
        SignalType.LONG, entry, sl, tp, atr, row,
        pullback_type="rsi_reversal", entry_type="rsi_reversal",
        max_bars=RSI_REV_MAX_BARS,
    )


def evaluate_rsi_reversal_short(row: pd.Series) -> Optional[Signal]:
    """RSI Reversal SHORT -- RSI sobrecomprado em tendencia de baixa.
    v22.0: RSI > 58 (de 62), SL 2.0x, TP 5.5x, max 120 bars.
    """
    close = float(row["close"])
    ema50 = float(row["ema50"])
    ema200 = float(row["ema200"])
    rsi = float(row["rsi"])
    rsi_delta = float(row.get("rsi_delta", 0.0))
    atr = float(row["atr"])
    atr_pct = float(row.get("atr_percentile", 0.5))

    if not (ema50 < ema200):
        return None
    if rsi <= 58.0:
        return None
    if rsi_delta >= 0:
        return None
    if close >= ema50:
        return None
    if not (0.08 <= atr_pct <= 0.95):
        return None
    if atr <= 0:
        return None

    entry = close
    sl = entry + (RSI_REV_SL_ATR_MULT * atr)
    tp = entry - (RSI_REV_TP_ATR_MULT * atr)

    return _make_signal(
        SignalType.SHORT, entry, sl, tp, atr, row,
        pullback_type="rsi_reversal", entry_type="rsi_reversal",
        max_bars=RSI_REV_MAX_BARS,
    )


def evaluate_ema_bounce_long(row: pd.Series) -> Optional[Signal]:
    """EMA Bounce LONG -- preco toca EMA20 e recupera em tendencia.
    v22.0: SL 2.0x, TP 4.5x, max 96 bars, RSI 35-70.
    """
    close = float(row["close"])
    low = float(row["low"])
    ema20 = float(row["ema20"])
    ema50 = float(row["ema50"])
    ema200 = float(row["ema200"])
    rsi = float(row["rsi"])
    atr = float(row["atr"])
    atr_pct = float(row.get("atr_percentile", 0.5))
    macd_hist = float(row.get("macd_hist", 0.0))
    macd_val = float(row.get("macd", 0.0))
    macd_sig = float(row.get("macd_signal", 0.0))

    if not (close > ema50 > ema200):
        return None
    if low > ema20:
        return None
    if close <= ema20:
        return None
    if not (35.0 <= rsi <= 70.0):
        return None
    # v22.0: MACD nao e mais filtro obrigatorio (soft confirm)
    if not (0.08 <= atr_pct <= 0.95):
        return None
    if atr <= 0:
        return None

    entry = close
    sl = entry - (EMA_BOUNCE_SL_ATR_MULT * atr)
    tp = entry + (EMA_BOUNCE_TP_ATR_MULT * atr)
    if sl <= 0:
        return None

    return _make_signal(
        SignalType.LONG, entry, sl, tp, atr, row,
        pullback_type="ema20_bounce", entry_type="ema_bounce",
        max_bars=EMA_BOUNCE_MAX_BARS,
    )


def evaluate_ema_bounce_short(row: pd.Series) -> Optional[Signal]:
    """EMA Bounce SHORT -- preco toca EMA20 e rejeita em tendencia.
    v22.0: SL 2.0x, TP 4.5x, max 96 bars, RSI 30-65.
    """
    close = float(row["close"])
    high = float(row["high"])
    ema20 = float(row["ema20"])
    ema50 = float(row["ema50"])
    ema200 = float(row["ema200"])
    rsi = float(row["rsi"])
    atr = float(row["atr"])
    atr_pct = float(row.get("atr_percentile", 0.5))
    macd_hist = float(row.get("macd_hist", 0.0))
    macd_val = float(row.get("macd", 0.0))
    macd_sig = float(row.get("macd_signal", 0.0))

    if not (close < ema50 < ema200):
        return None
    if high < ema20:
        return None
    if close >= ema20:
        return None
    if not (30.0 <= rsi <= 65.0):
        return None
    if not (0.08 <= atr_pct <= 0.95):
        return None
    if atr <= 0:
        return None

    entry = close
    sl = entry + (EMA_BOUNCE_SL_ATR_MULT * atr)
    tp = entry - (EMA_BOUNCE_TP_ATR_MULT * atr)

    return _make_signal(
        SignalType.SHORT, entry, sl, tp, atr, row,
        pullback_type="ema20_bounce", entry_type="ema_bounce",
        max_bars=EMA_BOUNCE_MAX_BARS,
    )


def evaluate_range_long(row: pd.Series) -> Optional[Signal]:
    """v22.0: Range Trader LONG -- BB lower band bounce em mercado lateral.
    SL 2.5x, TP 5.0x, ADX < 30, max 72 bars.
    """
    close = float(row["close"])
    low = float(row["low"])
    bb_lower = float(row["bb_lower"])
    bb_upper = float(row["bb_upper"])
    bb_w = float(row.get("bb_width", 0.0))
    rsi = float(row["rsi"])
    rsi_delta = float(row.get("rsi_delta", 0.0))
    atr = float(row["atr"])
    atr_pct = float(row.get("atr_percentile", 0.5))
    adx = float(row.get("adx", 0.0))

    if pd.isna(adx) or adx >= RANGE_ADX_MAX:
        return None
    if close > bb_lower * (1 + RANGE_BB_TOUCH_THRESHOLD):
        return None
    if close < bb_lower * (1 - 0.03):
        return None
    if rsi >= RANGE_RSI_OVERSOLD:
        return None
    if not (0.08 <= atr_pct <= 0.95):
        return None
    if atr <= 0:
        return None

    entry = close
    sl = entry - (RANGE_SL_ATR_MULT * atr)
    tp = entry + (RANGE_TP_ATR_MULT * atr)
    if sl <= 0:
        return None

    return _make_signal(
        SignalType.LONG, entry, sl, tp, atr, row,
        pullback_type="range_bb_lower", entry_type="range_trader",
        max_bars=RANGE_MAX_BARS,
    )


def evaluate_range_short(row: pd.Series) -> Optional[Signal]:
    """v22.0: Range Trader SHORT -- BB upper band bounce em mercado lateral.
    SL 2.5x, TP 5.0x, ADX < 30, max 72 bars.
    """
    close = float(row["close"])
    high = float(row["high"])
    bb_lower = float(row["bb_lower"])
    bb_upper = float(row["bb_upper"])
    bb_w = float(row.get("bb_width", 0.0))
    rsi = float(row["rsi"])
    rsi_delta = float(row.get("rsi_delta", 0.0))
    atr = float(row["atr"])
    atr_pct = float(row.get("atr_percentile", 0.5))
    adx = float(row.get("adx", 0.0))

    if pd.isna(adx) or adx >= RANGE_ADX_MAX:
        return None
    if close < bb_upper * (1 - RANGE_BB_TOUCH_THRESHOLD):
        return None
    if close > bb_upper * (1 + 0.03):
        return None
    if rsi <= RANGE_RSI_OVERBOUGHT:
        return None
    if not (0.08 <= atr_pct <= 0.95):
        return None
    if atr <= 0:
        return None

    entry = close
    sl = entry + (RANGE_SL_ATR_MULT * atr)
    tp = entry - (RANGE_TP_ATR_MULT * atr)

    return _make_signal(
        SignalType.SHORT, entry, sl, tp, atr, row,
        pullback_type="range_bb_upper", entry_type="range_trader",
        max_bars=RANGE_MAX_BARS,
    )


def evaluate_rsi_extremes_long(row: pd.Series) -> Optional[Signal]:
    """v22.0: RSI Extremes LONG -- RSI < 32 com virada confirmada.
    SL 2.0x, TP 4.5x, max 48 bars.
    """
    close = float(row["close"])
    rsi = float(row["rsi"])
    rsi_delta = float(row.get("rsi_delta", 0.0))
    atr = float(row["atr"])
    atr_pct = float(row.get("atr_percentile", 0.5))

    if rsi >= RSI_EXT_OVERSOLD:
        return None
    if rsi_delta <= RSI_EXT_RSI_DELTA_MIN:
        return None
    if not (0.08 <= atr_pct <= 0.95):
        return None
    if atr <= 0:
        return None

    entry = close
    sl = entry - (RSI_EXT_SL_ATR_MULT * atr)
    tp = entry + (RSI_EXT_TP_ATR_MULT * atr)
    if sl <= 0:
        return None

    return _make_signal(
        SignalType.LONG, entry, sl, tp, atr, row,
        pullback_type="rsi_extreme_oversold", entry_type="rsi_extremes",
        max_bars=RSI_EXT_MAX_BARS,
    )


def evaluate_rsi_extremes_short(row: pd.Series) -> Optional[Signal]:
    """v22.0: RSI Extremes SHORT -- RSI > 68 com virada confirmada.
    SL 2.0x, TP 4.5x, max 48 bars.
    """
    close = float(row["close"])
    rsi = float(row["rsi"])
    rsi_delta = float(row.get("rsi_delta", 0.0))
    atr = float(row["atr"])
    atr_pct = float(row.get("atr_percentile", 0.5))

    if rsi <= RSI_EXT_OVERBOUGHT:
        return None
    if rsi_delta >= -RSI_EXT_RSI_DELTA_MIN:
        return None
    if not (0.08 <= atr_pct <= 0.95):
        return None
    if atr <= 0:
        return None

    entry = close
    sl = entry + (RSI_EXT_SL_ATR_MULT * atr)
    tp = entry - (RSI_EXT_TP_ATR_MULT * atr)

    return _make_signal(
        SignalType.SHORT, entry, sl, tp, atr, row,
        pullback_type="rsi_extreme_overbought", entry_type="rsi_extremes",
        max_bars=RSI_EXT_MAX_BARS,
    )


def evaluate_scalp_long(row: pd.Series) -> Optional[Signal]:
    """v22.0: Scalp LONG -- Quick intraday trade.
    SL 1.5x, TP 3.0x, max 24 bars, RSI > 35, delta > 0.5.
    """
    close = float(row["close"])
    ema20 = float(row["ema20"])
    rsi = float(row["rsi"])
    rsi_delta = float(row.get("rsi_delta", 0.0))
    macd_hist = float(row.get("macd_hist", 0.0))
    macd_val = float(row.get("macd", 0.0))
    macd_sig = float(row.get("macd_signal", 0.0))
    atr = float(row["atr"])
    atr_pct = float(row.get("atr_percentile", 0.5))
    adx = float(row.get("adx", 0.0))

    if close <= ema20:
        return None
    if rsi <= SCALP_RSI_LONG_MIN:
        return None
    if rsi_delta <= SCALP_RSI_DELTA_TRIGGER:
        return None
    if not (macd_hist > 0 or macd_val > macd_sig):
        return None
    if adx > 65:
        return None
    if not (0.08 <= atr_pct <= 0.95):
        return None
    if atr <= 0:
        return None

    entry = close
    sl = entry - (SCALP_SL_ATR_MULT * atr)
    tp = entry + (SCALP_TP_ATR_MULT * atr)
    if sl <= 0:
        return None

    return _make_signal(
        SignalType.LONG, entry, sl, tp, atr, row,
        pullback_type="scalp", entry_type="scalp",
        max_bars=SCALP_MAX_BARS,
    )


def evaluate_scalp_short(row: pd.Series) -> Optional[Signal]:
    """v22.0: Scalp SHORT -- Quick intraday trade.
    SL 1.5x, TP 3.0x, max 24 bars, RSI < 65, delta < -0.5.
    """
    close = float(row["close"])
    ema20 = float(row["ema20"])
    rsi = float(row["rsi"])
    rsi_delta = float(row.get("rsi_delta", 0.0))
    macd_hist = float(row.get("macd_hist", 0.0))
    macd_val = float(row.get("macd", 0.0))
    macd_sig = float(row.get("macd_signal", 0.0))
    atr = float(row["atr"])
    atr_pct = float(row.get("atr_percentile", 0.5))
    adx = float(row.get("adx", 0.0))

    if close >= ema20:
        return None
    if rsi >= SCALP_RSI_SHORT_MAX:
        return None
    if rsi_delta >= -SCALP_RSI_DELTA_TRIGGER:
        return None
    if not (macd_hist < 0 or macd_val < macd_sig):
        return None
    if adx > 65:
        return None
    if not (0.08 <= atr_pct <= 0.95):
        return None
    if atr <= 0:
        return None

    entry = close
    sl = entry + (SCALP_SL_ATR_MULT * atr)
    tp = entry - (SCALP_TP_ATR_MULT * atr)

    return _make_signal(
        SignalType.SHORT, entry, sl, tp, atr, row,
        pullback_type="scalp", entry_type="scalp",
        max_bars=SCALP_MAX_BARS,
    )
