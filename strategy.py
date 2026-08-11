"""
strategy.py
-----------
Logica de validacao das condicoes de entrada CTEV v19.1 Multi-Strategy.

v21.0 — Ultra-Selective Quality Overhaul

  3 estrategias ativas (down from 6 — toxicas removidas):
    1. CTEV Trend (pullback + momentum) — tendencia com ADX > 25
    2. Squeeze Breakout (BBWP squeeze + breakout) — BBWP < 35
    3. RSI Reversal (RSI extremo em tendencia) — RSI < 38 /> 62

  DESATIVADAS (v21.0 — SL muito justo para 1h, geravam ruído):
    4. Range Trader (SL 1.5x ATR = ruído em 1h)
    5. RSI Extremes (SL 1.5x ATR = ruído em 1h)
    6. Scalp (SL 1.0x ATR = ruído puro)

  v21.0 Changes vs v20.0:
  - SL 3.5x ATR (de 2.8x) — sobrevive ao ruido de 1h
  - TP 7.5x ATR (de 5.5x) — R:R real ~2.1:1
  - MACD Histogram Filter ON — confirma momentum
  - ADX min 25 (mantido, mas com +filtros de qualidade)
  - RSI zones: LONG 46-64, SHORT 36-54 (mais estreitas)
  - Squeeze: BBWP < 35, SL 3.0x, TP 7.0x
  - RSI Reversal: SL 3.0x, TP 6.5x, mais strict
  - CTEV Momentum: SL 3.0x, TP 6.0x (de 2.0/4.5)
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
    """Representa um sinal de trade gerado pela estrategia CTEV v4."""
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
    pullback_type: str  # "fibonacci", "ema20_touch", "ema50_touch", "none"
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


# ── Parametros da estrategia CTEV v13.0 (ACTIVE TRADER MULTI-STRATEGY) ──
# v13.0: CTEV relaxado + Momentum + Mean-Reversion = 4+ trades/semana.

# REGIME FILTER — v21.0: ADX 25 (seletivo — so tendencias reais)
ADX_MIN = 25.0                # v21.0: 25 (de 22) — mais seletivo
ALLOW_TRANSITION = False      # OFF — transition nao adiciona com filtros strict

# RSI como zona de pullback (v21.0: mais estreito — 46-64/36-54)
RSI_LONG_MIN = 46.0           # v21.0: 46 (de 45)
RSI_LONG_MAX = 64.0           # v21.0: 64 (de 65)
RSI_SHORT_MIN = 36.0          # v21.0: 36 (de 35)
RSI_SHORT_MAX = 54.0          # v21.0: 54 (de 55)

# RSI Delta — desabilitado
RSI_DELTA_LONG_MIN = -5.0    # effectively disabled
RSI_DELTA_SHORT_MAX = 5.0    # effectively disabled

# Volume (DESABILITADO)
VOLUME_CONFIRM = False
VOLUME_SMA_RATIO = 0.30

# Fibonacci tolerancia
FIB_TOLERANCE_PCT = 0.025     # v16.2: RESTAURADO 2.5% (grid-optimized)

# ATR Percentile filter
ATR_PCT_MIN = 0.10
ATR_PCT_MAX = 0.90

# Bollinger Bandwidth — desabilitado
BB_WIDTH_MIN = 0.0
BB_WIDTH_MAX = 999.0

# EMA proximity — v14.3: DESATIVADO (v14.2 provou que adiciona trades ruins)
EMA20_PROXIMITY_PCT = 0.000  # v16.2: RESTAURADO OFF
EMA50_PROXIMITY_PCT = 0.000  # v16.2: RESTAURADO OFF

# EMA Slope
EMA50_SLOPE_MIN = -0.5

# Gestao de risco — R:R ~2.1:1 (v21.0: SL mais largo)
SL_ATR_MULT = 3.50          # v21.0: 3.5x ATR (de 2.8x — sobrevive ruido 1h)
TP_ATR_MULT = 7.50          # v21.0: 7.5x ATR (de 5.5x — maiores ganhadores)

# FILTROS DE CONFLUENCIA
# v21.0: MACD Histogram Filter ON — confirma direcao do momentum
DI_DIRECTION_FILTER = True   # +DI > -DI para LONG
MACD_HIST_FILTER = True    # v21.0: ON (de False) — exclui entradas sem momentum
OBV_TREND_FILTER = False
STOCH_RSI_FILTER = False
BBWP_SQUEEZE_BONUS = True

# v14.3: BB TOUCH desabilitado (v14.2 provou que adiciona trades ruins)
BB_TOUCH_PCT = 0.000         # v14.3: OFF

# v21.0: Momentum com SL/TP mais largos
MOMENTUM_ADX_MIN = 25.0
MOMENTUM_SL_ATR_MULT = 3.00   # v21.0: 3.0x (de 2.8x)
MOMENTUM_TP_ATR_MULT = 6.00   # v21.0: 6.0x (de 5.5x)
MOMENTUM_MAX_BARS = 96        # v21.0: 96 (de 168)

# v14.1: MEAN-REVERSION DESATIVADO (WR < 25% — sem edge)
# Mantido apenas para referencia futura se necessario.


# ═══════════════════════════════════════════════════════════════════
# v19.0: MULTI-STRATEGY ENGINE — Regime-Aware Adaptive System
# Objetivo: detectar mercado lateral e operar com edge em qualquer cenario.
# Metas: 30d/90d/180d >= 30%, 365d >= 70%, 730d >= 120%
# ═══════════════════════════════════════════════════════════════════

# v18.1: EMA Bounce DESATIVADO — adicionou 219 trades sem edge em 730d
EMA_BOUNCE_SL_ATR_MULT = 2.00
EMA_BOUNCE_TP_ATR_MULT = 4.50
EMA_BOUNCE_MAX_BARS = 72

# Squeeze Breakout: BBWP squeeze + breakout de banda
SQUEEZE_SL_ATR_MULT = 3.00     # v21.0: 3.0x (de 2.0x)
SQUEEZE_TP_ATR_MULT = 7.00     # v21.0: 7.0x (de 6.0x)
SQUEEZE_MAX_BARS = 96           # v21.0: 96 (de 72)
SQUEEZE_BBWP_THRESHOLD = 35.0  # v21.0: 35 (de 45) -- mais seletivo

# RSI Reversal: RSI extremo em tendencia = oportunidade de reversao
RSI_REV_SL_ATR_MULT = 3.00    # v21.0: 3.0x (de 2.5x)
RSI_REV_TP_ATR_MULT = 6.50    # v21.0: 6.5x (de 5.0x)
RSI_REV_MAX_BARS = 96

# ── v21.0: RANGE TRADER -- DESATIVADO (SL 1.5x ATR = ruido em 1h) ──
RANGE_SL_ATR_MULT = 1.50
RANGE_TP_ATR_MULT = 3.50
RANGE_MAX_BARS = 48
RANGE_BB_TOUCH_THRESHOLD = 0.015
RANGE_ADX_MAX = 25.0
RANGE_RSI_OVERSOLD = 42.0
RANGE_RSI_OVERBOUGHT = 58.0

# ── v21.0: SCALP -- DESATIVADO (SL 1.0x ATR = ruido puro) ──
SCALP_SL_ATR_MULT = 1.00
SCALP_TP_ATR_MULT = 1.50
SCALP_MAX_BARS = 18
SCALP_RSI_LONG_MIN = 40.0
SCALP_RSI_SHORT_MAX = 60.0
SCALP_RSI_DELTA_TRIGGER = 0.8

# ── v21.0: RSI EXTREMES -- DESATIVADO (SL 1.5x ATR = ruido em 1h) ──
RSI_EXT_SL_ATR_MULT = 1.50
RSI_EXT_TP_ATR_MULT = 3.00
RSI_EXT_MAX_BARS = 36
RSI_EXT_OVERSOLD = 30.0
RSI_EXT_OVERBOUGHT = 70.0
RSI_EXT_RSI_DELTA_MIN = 0.3


def _price_near_fib(
    price: float, fib_level: float,
    tolerance_pct: float = FIB_TOLERANCE_PCT,
) -> bool:
    """Verifica se o preco esta dentro da tolerancia de um nivel Fibonacci."""
    if pd.isna(fib_level) or fib_level <= 0:
        return False
    tolerance = price * tolerance_pct
    return abs(price - fib_level) <= tolerance


def _in_fib_zone(price: float, fib_0382: float, fib_0618: float, direction: int) -> bool:
    """
    Verifica se o preco esta na zona de pullback Fibonacci (0.382 - 0.618).

    Para uptrend (direction=1): fib_0618 < fib_0382, preco deve estar entre eles.
    Para downtrend (direction=-1): fib_0382 < fib_0618, preco deve estar entre eles.
    """
    if pd.isna(fib_0382) or pd.isna(fib_0618):
        return False

    if direction == 1:
        return fib_0618 <= price <= fib_0382
    elif direction == -1:
        return fib_0618 <= price <= fib_0382
    return False


def _macd_bullish(macd_hist: float, macd_line: float, macd_signal: float) -> bool:
    """Verifica se MACD indica momentum de alta."""
    return macd_hist > 0 or macd_line > macd_signal


def _macd_bearish(macd_hist: float, macd_line: float, macd_signal: float) -> bool:
    """Verifica se MACD indica momentum de baixa."""
    return macd_hist < 0 or macd_line < macd_signal


def evaluate_long(
    row: pd.Series, profile: Optional[StrategyProfile] = None
) -> Optional[Signal]:
    """
    Avalia condicoes LONG (regime-based trend-following com pullback):

    Requisitos (CTEV v5.0 — 6 filtros, adaptados por timeframe via profile):
      1. REGIME: trending_up (ADX>=profile.adx_min) OU transition
      2. TENDENCIA: close > EMA(50) E EMA(50) > EMA(200)
      3. SLOPE: ema50_slope > profile.ema50_slope_min
      4. PULLBACK: Fibonacci (tol profile.fib_tolerance_pct) OU EMA(20/50) touch
      5. RSI: LONG profile.rsi_long_min - profile.rsi_long_max
      6. ATR: Percentile profile.atr_pct_min - profile.atr_pct_max
    """
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

    # Resolve profile parameters (fallback para constantes v7.0)
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

    # v7.0: BBWP squeeze bonus — relaxa RSI em 5 pontos durante squeeze
    _bbwp = float(row.get("bbwp", 50.0))
    if BBWP_SQUEEZE_BONUS and not pd.isna(_bbwp) and _bbwp < 20.0:
        _rsi_l_min = max(0, _rsi_l_min - 5)
        _rsi_l_max = _rsi_l_max + 5

    # 1. REGIME FILTER (v4.4: ALLOW_TRANSITION toggle)
    if regime == "trending_up":
        if pd.isna(adx) or adx < _adx_min:
            return None
    elif regime == "transition":
        if not _allow_trans:
            return None
    else:
        return None

    # 2. TENDENCIA: Dual EMA — uptrend confirmado
    if not (close > ema50 and ema50 > ema200):
        return None

    # 3. SLOPE: EMA50 deve estar subindo
    if pd.isna(ema50_slope) or ema50_slope <= _slope_min:
        return None

    # 3b. DI DIRECTION FILTER (v6.0): +DI deve estar acima de -DI para LONG
    if DI_DIRECTION_FILTER:
        if pd.isna(plus_di) or pd.isna(minus_di):
            return None
        if plus_di <= minus_di:
            return None

    # 3c. v7.0: MACD HISTOGRAM FILTER — momentum alinhado com direcao
    if MACD_HIST_FILTER:
        if pd.isna(macd_hist):
            return None
        if macd_hist <= 0:
            return None

    # 3d. v7.0: OBV TREND FILTER — fluxo de volume confirmando alta
    if OBV_TREND_FILTER:
        _obv_trend = int(row.get("obv_trend", 0))
        if _obv_trend < 1:
            return None

    # 3e. v7.0: STOCH RSI FILTER — momentum de curto prazo alinhado
    if STOCH_RSI_FILTER:
        _stoch_k = float(row.get("stoch_rsi_k", 50.0))
        _stoch_d = float(row.get("stoch_rsi_d", 50.0))
        if pd.isna(_stoch_k) or pd.isna(_stoch_d):
            return None
        if _stoch_k <= _stoch_d:
            return None

    # 4. PULLBACK: v18.2 — apenas Fibonacci (EMA touch removido — baixa qualidade)
    # EMA touch gerava muitos pullbacks falsos que nao tinham edge real.
    # Apenas Fibonacci levels + ADX > 25 = pullback de alta qualidade.
    pullback_type = None

    # 4a. Fibonacci check (tolerancia 2.5% — grid-optimized)
    in_fib = _in_fib_zone(close, fib_0382, fib_0618, fib_dir)
    if in_fib and fib_dir == 1:
        pullback_type = "fibonacci"
    else:
        if fib_dir == 1:
            if (_price_near_fib(low, fib_0382, _fib_tol) or
                    _price_near_fib(low, fib_0500, _fib_tol) or
                    _price_near_fib(low, fib_0618, _fib_tol)):
                pullback_type = "fibonacci"

    # v18.2: EMA touch REMOVIDO — entradas sem edge comprovado
    # Os EMA touch geravam ~40% dos pullbacks mas com WR similar ao momentum.
    # Apenas Fibonacci + ADX > 25 = pullback de qualidade superior.

    # 4d. EMA(20) PROXIMITY — 1.2% (v14.2: corrigido — so abaixo da EMA para LONG)
    _ema20_prox = profile.ema20_proximity_pct if profile else EMA20_PROXIMITY_PCT
    if pullback_type is None and _ema20_prox > 0:
        # v14.2: pullback = preco puxou de volta PARA a EMA (close <= ema20)
        if close <= ema20 and close >= ema20 * (1 - _ema20_prox):
            pullback_type = "ema20_proximity"

    # 4e. EMA(50) PROXIMITY — 1.5% (v14.2: corrigido — so abaixo da EMA para LONG)
    _ema50_prox = profile.ema50_proximity_pct if profile else EMA50_PROXIMITY_PCT
    if pullback_type is None and _ema50_prox > 0:
        if close <= ema50 and close >= ema50 * (1 - _ema50_prox):
            pullback_type = "ema50_proximity"

    # 4f. v14.2: BB LOWER TOUCH — preco perto da Banda Inferior (pullback profundo)
    if pullback_type is None:
        if close <= bb_lower * (1 + BB_TOUCH_PCT):
            pullback_type = "bb_lower_touch"

    # v21.0: Pullback requer ADX > 25
    _PULLBACK_ADX_MIN = 25.0
    if pullback_type is not None:
        if pd.isna(adx) or adx < _PULLBACK_ADX_MIN:
            pullback_type = None  # downgraded to momentum
    
    if pullback_type is not None:
        _entry_type = "ctev_pullback"
        _max_bars_use = 120  # v21.0: 120 (de 168)
        _sl_use = _sl_mult  # 3.5x/7.5x
        _tp_use = _tp_mult
    else:
        pullback_type = "none"
        _entry_type = "ctev_momentum"
        _max_bars_use = 96   # v21.0: 96 (de 72)
        # v21.0: Momentum com SL/TP mais largos
        _sl_use = MOMENTUM_SL_ATR_MULT  # 3.0x
        _tp_use = MOMENTUM_TP_ATR_MULT  # 6.0x

    # 5. RSI: Zona de pullback (adaptada ao profile)
    if not (_rsi_l_min <= rsi <= _rsi_l_max):
        return None

    # v18.2: Momentum RSI — RSI > 50 para LONG momentum (evita zona ambigua)
    # Momentum sem pullback precisa de confirmacao direcional clara.
    if pullback_type == "none" and rsi < 50.0:
        return None

    # 6. VOLUME: Soft confirmation (adaptada ao profile)
    if _vol_confirm and (not pd.isna(volume_sma50) and volume_sma50 > 0):
        if volume < volume_sma50 * _vol_ratio:
            return None

    # 7. ATR: Volatilidade na faixa normal (adaptada ao profile)
    if not (_atr_pct_min <= atr_pct <= _atr_pct_max):
        return None

    # NOTA v4.2: Filtros MACD, RSI Delta e BB Width REMOVIDOS
    # ── Gestao de risco LONG — SL/TP por entry type ──
    entry = close
    stop_loss = entry - (_sl_use * atr)
    take_profit = entry + (_tp_use * atr)

    if stop_loss <= 0:
        return None

    _profile_name = profile.name if profile else "v5.0-default"
    logger.info(
        "SINAL LONG %s | entry=%.2f SL=%.2f TP=%.2f ATR=%.2f "
        "RSI=%.1f ADX=%.1f regime=%s pullback=%s",
        _profile_name, entry, stop_loss, take_profit, atr, rsi, adx, regime, pullback_type,
    )

    return Signal(
        type=SignalType.LONG,
        entry_price=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        atr=atr,
        rsi=rsi,
        rsi_delta=rsi_delta,
        macd_hist=macd_hist,
        ema20=ema20,
        ema50=ema50,
        ema200=ema200,
        adx=adx,
        plus_di=plus_di,
        minus_di=minus_di,
        regime=regime,
        bb_lower=bb_lower,
        bb_upper=bb_upper,
        bb_width=bb_w,
        bb_squeeze_pct=bb_sq_pct,
        volume=volume,
        volume_sma20=volume_sma20,
        volume_sma50=volume_sma50,
        atr_percentile=atr_pct,
        fib_0382=fib_0382 if not pd.isna(fib_0382) else 0.0,
        fib_0500=fib_0500 if not pd.isna(fib_0500) else 0.0,
        fib_0618=fib_0618 if not pd.isna(fib_0618) else 0.0,
        fib_direction=fib_dir,
        fib_proximity=fib_prox,
        pullback_type=pullback_type,
        entry_type=_entry_type,
        max_bars=_max_bars_use,
        ema50_slope=ema50_slope,
        timestamp=ts,
    )


def evaluate_short(
    row: pd.Series, profile: Optional[StrategyProfile] = None
) -> Optional[Signal]:
    """
    Avalia condicoes SHORT (regime-based trend-following com pullback):

    Requisitos (CTEV v5.0 — 6 filtros, adaptados por timeframe via profile):
      1. REGIME: trending_down (ADX>=profile.adx_min) OU transition
      2. TENDENCIA: close < EMA(50) EMA(50) < EMA(200)
      3. SLOPE: ema50_slope < -profile.ema50_slope_min
      4. PULLBACK: Fibonacci (tol profile.fib_tolerance_pct) OU EMA(20/50) touch
      5. RSI: SHORT profile.rsi_short_min - profile.rsi_short_max
      6. ATR: Percentile profile.atr_pct_min - profile.atr_pct_max
    """
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

    # Resolve profile parameters (fallback para constantes v7.0)
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

    # v7.0: BBWP squeeze bonus — relaxa RSI em 5 pontos durante squeeze
    _bbwp = float(row.get("bbwp", 50.0))
    if BBWP_SQUEEZE_BONUS and not pd.isna(_bbwp) and _bbwp < 20.0:
        _rsi_s_min = max(0, _rsi_s_min - 5)
        _rsi_s_max = _rsi_s_max + 5

    # 1. REGIME FILTER (v4.4: ALLOW_TRANSITION toggle)
    if regime == "trending_down":
        if pd.isna(adx) or adx < _adx_min:
            return None
    elif regime == "transition":
        if not _allow_trans:
            return None
    else:
        return None

    # 2. TENDENCIA: Dual EMA — downtrend confirmado
    if not (close < ema50 and ema50 < ema200):
        return None

    # 3. SLOPE: EMA50 deve estar descendo
    if pd.isna(ema50_slope) or ema50_slope >= -_slope_min:
        return None

    # 3b. DI DIRECTION FILTER (v6.0): -DI deve estar acima de +DI para SHORT
    if DI_DIRECTION_FILTER:
        if pd.isna(plus_di) or pd.isna(minus_di):
            return None
        if minus_di <= plus_di:
                return None

    # 3c. v7.0: MACD HISTOGRAM FILTER — momentum alinhado com direcao
    if MACD_HIST_FILTER:
        if pd.isna(macd_hist):
            return None
        if macd_hist >= 0:
            return None

    # 3d. v7.0: OBV TREND FILTER — fluxo de volume confirmando baixa
    if OBV_TREND_FILTER:
        _obv_trend = int(row.get("obv_trend", 0))
        if _obv_trend > -1:
            return None

    # 3e. v7.0: STOCH RSI FILTER — momentum de curto prazo alinhado
    if STOCH_RSI_FILTER:
        _stoch_k = float(row.get("stoch_rsi_k", 50.0))
        _stoch_d = float(row.get("stoch_rsi_d", 50.0))
        if pd.isna(_stoch_k) or pd.isna(_stoch_d):
            return None
        if _stoch_k >= _stoch_d:
            return None

    # 4. PULLBACK: v18.2 — apenas Fibonacci (EMA touch removido — baixa qualidade)
    pullback_type = None

    # 4a. Fibonacci check (tolerancia 2.5% — grid-optimized)
    in_fib = _in_fib_zone(close, fib_0382, fib_0618, fib_dir)
    if in_fib and fib_dir == -1:
        pullback_type = "fibonacci"
    else:
        if fib_dir == -1:
            if (_price_near_fib(high, fib_0382, _fib_tol) or
                    _price_near_fib(high, fib_0500, _fib_tol) or
                    _price_near_fib(high, fib_0618, _fib_tol)):
                pullback_type = "fibonacci"

    # v18.2: EMA touch REMOVIDO — entradas sem edge comprovado

    # 4d. EMA(20) PROXIMITY — 1.2% (v14.2: corrigido — so acima da EMA para SHORT)
    _ema20_prox = profile.ema20_proximity_pct if profile else EMA20_PROXIMITY_PCT
    if pullback_type is None and _ema20_prox > 0:
        # v14.2: pullback = preco puxou de volta PARA a EMA (close >= ema20)
        if close >= ema20 and close <= ema20 * (1 + _ema20_prox):
            pullback_type = "ema20_proximity"

    # 4e. EMA(50) PROXIMITY — 1.5% (v14.2: corrigido — so acima da EMA para SHORT)
    _ema50_prox = profile.ema50_proximity_pct if profile else EMA50_PROXIMITY_PCT
    if pullback_type is None and _ema50_prox > 0:
        if close >= ema50 and close <= ema50 * (1 + _ema50_prox):
            pullback_type = "ema50_proximity"

    # 4f. v14.2: BB UPPER TOUCH — preco perto da Banda Superior (pullback profundo)
    if pullback_type is None:
        if close >= bb_upper * (1 - BB_TOUCH_PCT):
            pullback_type = "bb_upper_touch"

    # v21.0: Pullback requer ADX > 25
    _PULLBACK_ADX_MIN_S = 25.0
    if pullback_type is not None:
        if pd.isna(adx) or adx < _PULLBACK_ADX_MIN_S:
            pullback_type = None  # downgraded to momentum

    if pullback_type is not None:
        _entry_type_s = "ctev_pullback"
        _max_bars_use_s = 120  # v21.0: 120 (de 168)
        _sl_use_s = _sl_mult  # 3.5x/7.5x
        _tp_use_s = _tp_mult
    else:
        pullback_type = "none"
        _entry_type_s = "ctev_momentum"
        _max_bars_use_s = 96   # v21.0: 96 (de 72)
        # v21.0: Momentum com SL/TP mais largos
        _sl_use_s = MOMENTUM_SL_ATR_MULT  # 3.0x
        _tp_use_s = MOMENTUM_TP_ATR_MULT  # 6.0x

    # 5. RSI: Zona de rally (adaptada ao profile)
    if not (_rsi_s_min <= rsi <= _rsi_s_max):
        return None

    # v18.2: Momentum RSI — RSI < 50 para SHORT momentum (evita zona ambigua)
    if pullback_type == "none" and rsi > 50.0:
        return None

    # 6. VOLUME: Soft confirmation (adaptada ao profile)
    if _vol_confirm and (not pd.isna(volume_sma50) and volume_sma50 > 0):
        if volume < volume_sma50 * _vol_ratio:
            return None

    # 7. ATR: Volatilidade na faixa normal (adaptada ao profile)
    if not (_atr_pct_min <= atr_pct <= _atr_pct_max):
        return None

    # NOTA v4.2: Filtros MACD, RSI Delta e BB Width REMOVIDOS
    # ── Gestao de risco SHORT — SL/TP por entry type ──
    entry = close
    stop_loss = entry + (_sl_use_s * atr)
    take_profit = entry - (_tp_use_s * atr)

    _profile_name = profile.name if profile else "v5.0-default"
    logger.info(
        "SINAL SHORT %s | entry=%.2f SL=%.2f TP=%.2f ATR=%.2f "
        "RSI=%.1f ADX=%.1f regime=%s pullback=%s",
        _profile_name, entry, stop_loss, take_profit, atr, rsi, adx, regime, pullback_type,
    )

    return Signal(
        type=SignalType.SHORT,
        entry_price=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        atr=atr,
        rsi=rsi,
        rsi_delta=rsi_delta,
        macd_hist=macd_hist,
        ema20=ema20,
        ema50=ema50,
        ema200=ema200,
        adx=adx,
        plus_di=plus_di,
        minus_di=minus_di,
        regime=regime,
        bb_lower=bb_lower,
        bb_upper=bb_upper,
        bb_width=bb_w,
        bb_squeeze_pct=bb_sq_pct,
        volume=volume,
        volume_sma20=volume_sma20,
        volume_sma50=volume_sma50,
        atr_percentile=atr_pct,
        fib_0382=fib_0382 if not pd.isna(fib_0382) else 0.0,
        fib_0500=fib_0500 if not pd.isna(fib_0500) else 0.0,
        fib_0618=fib_0618 if not pd.isna(fib_0618) else 0.0,
        fib_direction=fib_dir,
        fib_proximity=fib_prox,
        pullback_type=pullback_type,
        entry_type=_entry_type_s,
        max_bars=_max_bars_use_s,
        ema50_slope=ema50_slope,
        timestamp=ts,
    )


def evaluate_signal(
    df_ind: pd.DataFrame, profile: Optional[StrategyProfile] = None
) -> Optional[Signal]:
    """
    Ponto de entrada principal da estrategia v17.0.
    Recebe o DataFrame com indicadores e avalia apenas a ultima linha.
    Tenta todos os tipos de entrada via evaluate_row_signals().

    v17.0: Multi-Strategy Engine — CTEV + Momentum + EMA Bounce +
    Squeeze Breakout + RSI Reversal + Mean-Reversion.
    """
    if df_ind.empty:
        return None
    return evaluate_row_signals(df_ind.iloc[-1], profile=profile)


def evaluate_row_signals(
    row: pd.Series, profile: Optional[StrategyProfile] = None
) -> Optional[Signal]:
    """v19.0: Multi-strategy priority chain — tenta todos os tipos de entrada.

    v21.0: Ultra-Selective Quality -- apenas 3 estrategias ativas
    Prioridade (maior qualidade primeiro):
      1. CTEV Pullback/Momentum (trend-following strict, SL 3.5x/3.0x)
      2. Squeeze Breakout (BBWP < 35, SL 3.0x, TP 7.0x)
      3. RSI Reversal (RSI extremo em tendencia, SL 3.0x, TP 6.5x)

    DESATIVADAS v21.0 (SL muito justo para 1h, geravam ruido):
      4. Range Trader -- DESATIVADO
      5. RSI Extremes -- DESATIVADO
      6. Scalp -- DESATIVADO
    """
    # 1. CTEV (highest quality -- strict trend-following)
    signal = evaluate_long(row, profile=profile)
    if signal is not None:
        return signal
    signal = evaluate_short(row, profile=profile)
    if signal is not None:
        return signal

    # 2. Squeeze Breakout (volatility expansion from BB squeeze)
    signal = evaluate_squeeze_breakout_long(row)
    if signal is not None:
        return signal
    signal = evaluate_squeeze_breakout_short(row)
    if signal is not None:
        return signal

    # 3. RSI Reversal (extreme RSI in established trend)
    signal = evaluate_rsi_reversal_long(row)
    if signal is not None:
        return signal
    signal = evaluate_rsi_reversal_short(row)
    if signal is not None:
        return signal

    # v21.0: Range Trader, RSI Extremes, Scalp -- DESATIVADOS
    # Estas estrategias tinham SL 1.0-1.5x ATR que e ruido em 1h BTC.
    # Resultado: 86% SL hit rate. Desativadas para focar em qualidade.

    return None


def evaluate_momentum_long(row: pd.Series) -> Optional[Signal]:
    """Momentum continuation LONG — no pullback required.
    Requisitos:
      1. Regime trending_up, ADX >= 22
      2. +DI > -DI (direcional)
      3. close > EMA20 (acima MA curta)
      4. RSI 50-72 (zona de momentum)
      5. MACD hist > 0 (momentum positivo)
      6. SL 2.0x, TP 3.5x ATR
    """
    close = float(row["close"])
    ema20 = float(row["ema20"])
    ema50 = float(row["ema50"])
    ema200 = float(row["ema200"])
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
    bb_lower = float(row["bb_lower"])
    bb_upper = float(row["bb_upper"])
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

    # 1. Regime trending_up + ADX >= 22
    if regime != "trending_up":
        return None
    if pd.isna(adx) or adx < MOMENTUM_ADX_MIN:
        return None

    # 2. DI direction
    if pd.isna(plus_di) or pd.isna(minus_di):
        return None
    if plus_di <= minus_di:
        return None

    # 3. close > EMA20
    if close <= ema20:
        return None

    # 4. RSI momentum zone
    if not (50.0 <= rsi <= 72.0):
        return None

    # 5. MACD hist > 0
    if pd.isna(macd_hist) or macd_hist <= 0:
        return None

    # 6. ATR percentile
    if not (0.10 <= atr_pct <= 0.90):
        return None

    entry = close
    sl = entry - (MOMENTUM_SL_ATR_MULT * atr)
    tp = entry + (MOMENTUM_TP_ATR_MULT * atr)
    if sl <= 0:
        return None

    return Signal(
        type=SignalType.LONG, entry_price=entry, stop_loss=sl,
        take_profit=tp, atr=atr, rsi=rsi, rsi_delta=rsi_delta,
        macd_hist=macd_hist, ema20=ema20, ema50=ema50, ema200=ema200,
        adx=adx, plus_di=plus_di, minus_di=minus_di, regime=regime,
        bb_lower=bb_lower, bb_upper=bb_upper, bb_width=bb_w,
        bb_squeeze_pct=bb_sq_pct, volume=volume, volume_sma20=volume_sma20,
        volume_sma50=volume_sma50, atr_percentile=atr_pct,
        fib_0382=0.0, fib_0500=0.0, fib_0618=0.0, fib_direction=0,
        fib_proximity=0.0, pullback_type="momentum", ema50_slope=ema50_slope,
        entry_type="momentum", max_bars=72,
        timestamp=ts,
    )


def evaluate_momentum_short(row: pd.Series) -> Optional[Signal]:
    """Momentum continuation SHORT — no pullback required.
    Requisitos:
      1. Regime trending_down, ADX >= 22
      2. -DI > +DI (direcional)
      3. close < EMA20 (abaixo MA curta)
      4. RSI 28-50 (zona de momentum baixa)
      5. MACD hist < 0 (momentum negativo)
      6. SL 2.0x, TP 3.5x ATR
    """
    close = float(row["close"])
    ema20 = float(row["ema20"])
    ema50 = float(row["ema50"])
    ema200 = float(row["ema200"])
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
    bb_lower = float(row["bb_lower"])
    bb_upper = float(row["bb_upper"])
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

    # 1. Regime trending_down + ADX >= 22
    if regime != "trending_down":
        return None
    if pd.isna(adx) or adx < MOMENTUM_ADX_MIN:
        return None

    # 2. DI direction
    if pd.isna(plus_di) or pd.isna(minus_di):
        return None
    if minus_di <= plus_di:
        return None

    # 3. close < EMA20
    if close >= ema20:
        return None

    # 4. RSI momentum zone
    if not (28.0 <= rsi <= 50.0):
        return None

    # 5. MACD hist < 0
    if pd.isna(macd_hist) or macd_hist >= 0:
        return None

    # 6. ATR percentile
    if not (0.10 <= atr_pct <= 0.90):
        return None

    entry = close
    sl = entry + (MOMENTUM_SL_ATR_MULT * atr)
    tp = entry - (MOMENTUM_TP_ATR_MULT * atr)

    return Signal(
        type=SignalType.SHORT, entry_price=entry, stop_loss=sl,
        take_profit=tp, atr=atr, rsi=rsi, rsi_delta=rsi_delta,
        macd_hist=macd_hist, ema20=ema20, ema50=ema50, ema200=ema200,
        adx=adx, plus_di=plus_di, minus_di=minus_di, regime=regime,
        bb_lower=bb_lower, bb_upper=bb_upper, bb_width=bb_w,
        bb_squeeze_pct=bb_sq_pct, volume=volume, volume_sma20=volume_sma20,
        volume_sma50=volume_sma50, atr_percentile=atr_pct,
        fib_0382=0.0, fib_0500=0.0, fib_0618=0.0, fib_direction=0,
        fib_proximity=0.0, pullback_type="momentum", ema50_slope=ema50_slope,
        entry_type="momentum", max_bars=72,
        timestamp=ts,
    )


def evaluate_mean_reversion_long(row: pd.Series) -> Optional[Signal]:
    """Mean-Reversion LONG — BB bounce em mercados laterais/tendencia fraca.
    Requisitos:
      1. close <= bb_lower * 1.01 (preco perto/at BB inferior)
      2. RSI < 42 (sobre-venda relativa)
      3. close > EMA200 (tendencia de longo prazo ancora de alta)
      4. ADX < 30 (sem tendencia forte — ranging)
      5. SL 1.5x, TP 2.5x ATR
    """
    close = float(row["close"])
    ema20 = float(row["ema20"])
    ema50 = float(row["ema50"])
    ema200 = float(row["ema200"])
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
    bb_lower = float(row["bb_lower"])
    bb_upper = float(row["bb_upper"])
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

    # 1. Price at/near lower BB
    if close > bb_lower * 1.01:
        return None

    # 2. RSI oversold
    if rsi >= 42.0:
        return None

    # 3. Above EMA200 (long-term uptrend anchor)
    if close <= ema200:
        return None

    # 4. Low ADX (ranging market)
    if pd.isna(adx) or adx >= 30.0:
        return None

    # 5. ATR percentile
    if not (0.10 <= atr_pct <= 0.90):
        return None

    entry = close
    sl = entry - (1.5 * atr)
    tp = entry + (2.5 * atr)
    if sl <= 0:
        return None

    return Signal(
        type=SignalType.LONG, entry_price=entry, stop_loss=sl,
        take_profit=tp, atr=atr, rsi=rsi, rsi_delta=rsi_delta,
        macd_hist=macd_hist, ema20=ema20, ema50=ema50, ema200=ema200,
        adx=adx, plus_di=plus_di, minus_di=minus_di, regime=regime,
        bb_lower=bb_lower, bb_upper=bb_upper, bb_width=bb_w,
        bb_squeeze_pct=bb_sq_pct, volume=volume, volume_sma20=volume_sma20,
        volume_sma50=volume_sma50, atr_percentile=atr_pct,
        fib_0382=0.0, fib_0500=0.0, fib_0618=0.0, fib_direction=0,
        fib_proximity=0.0, pullback_type="mean_reversion", ema50_slope=ema50_slope,
        entry_type="ranging_mr", max_bars=48,
        timestamp=ts,
    )


def evaluate_mean_reversion_short(row: pd.Series) -> Optional[Signal]:
    """Mean-Reversion SHORT — BB bounce em mercados laterais/tendencia fraca.
    Requisitos:
      1. close >= bb_upper * 0.99 (preco perto/at BB superior)
      2. RSI > 58 (sobre-compra relativa)
      3. close < EMA200 (tendencia de longo prazo ancora de baixa)
      4. ADX < 30 (sem tendencia forte — ranging)
      5. SL 1.5x, TP 2.5x ATR
    """
    close = float(row["close"])
    ema20 = float(row["ema20"])
    ema50 = float(row["ema50"])
    ema200 = float(row["ema200"])
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
    bb_lower = float(row["bb_lower"])
    bb_upper = float(row["bb_upper"])
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

    # 1. Price at/near upper BB
    if close < bb_upper * 0.99:
        return None

    # 2. RSI overbought
    if rsi <= 58.0:
        return None

    # 3. Below EMA200 (long-term downtrend anchor)
    if close >= ema200:
        return None

    # 4. Low ADX (ranging market)
    if pd.isna(adx) or adx >= 30.0:
        return None

    # 5. ATR percentile
    if not (0.10 <= atr_pct <= 0.90):
        return None

    entry = close
    sl = entry + (1.5 * atr)
    tp = entry - (2.5 * atr)

    return Signal(
        type=SignalType.SHORT, entry_price=entry, stop_loss=sl,
        take_profit=tp, atr=atr, rsi=rsi, rsi_delta=rsi_delta,
        macd_hist=macd_hist, ema20=ema20, ema50=ema50, ema200=ema200,
        adx=adx, plus_di=plus_di, minus_di=minus_di, regime=regime,
        bb_lower=bb_lower, bb_upper=bb_upper, bb_width=bb_w,
        bb_squeeze_pct=bb_sq_pct, volume=volume, volume_sma20=volume_sma20,
        volume_sma50=volume_sma50, atr_percentile=atr_pct,
        fib_0382=0.0, fib_0500=0.0, fib_0618=0.0, fib_direction=0,
        fib_proximity=0.0, pullback_type="mean_reversion", ema50_slope=ema50_slope,
        entry_type="ranging_mr", max_bars=48,
        timestamp=ts,
    )


# ═══════════════════════════════════════════════════════════════════
# v17.0: NOVOS TIPOS DE ENTRADA
# ═══════════════════════════════════════════════════════════════════

def _make_signal(
    sig_type: SignalType, entry: float, sl: float, tp: float,
    atr: float, row: pd.Series, pullback_type: str,
    entry_type: str, max_bars: int,
) -> Optional[Signal]:
    """Helper para criar Signal com todos os campos obrigatorios."""
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


def evaluate_ema_bounce_long(row: pd.Series) -> Optional[Signal]:
    """EMA Bounce LONG — preco toca EMA20 e recupera em tendencia.

    v18.0: Requer EMA stack + MACD alinhado para edge real.
    SL mais justo (2.0x) e TP moderado (4.5x) para R:R ~2.25:1.

    Requisitos:
      1. close > EMA50 E EMA50 > EMA200 (EMA stack — tendencia real)
      2. low <= EMA20 (preco tocou EMA20 — pullback)
      3. close > EMA20 (recuperou — bounce confirmado)
      4. RSI 38-62 (zona moderada — mais ampla que v17.1)
      5. MACD hist > 0 OU macd > macd_signal (momentum alinhado)
      6. ATR percentile 10-90
      7. SL 2.0x, TP 4.5x ATR, max 72 bars
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

    # 1. EMA stack — tendencia REAL
    if not (close > ema50 > ema200):
        return None
    # 2. Price touched EMA20 (pullback)
    if low > ema20:
        return None
    # 3. Price recovered above EMA20 (bounce)
    if close <= ema20:
        return None
    # 4. RSI zone (not extreme) — v18.0: 38-62 (mais amplo)
    if not (38.0 <= rsi <= 62.0):
        return None
    # 5. v18.0: MACD alinhado — pelo menos um indicador de momentum positivo
    if not (macd_hist > 0 or macd_val > macd_sig):
        return None
    # 6. ATR percentile
    if not (0.10 <= atr_pct <= 0.90):
        return None
    # 7. Minimum ATR
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
    """EMA Bounce SHORT — preco toca EMA20 e rejeita em tendencia.

    v18.0: Requer EMA stack + MACD alinhado para edge real.
    SL mais justo (2.0x) e TP moderado (4.5x) para R:R ~2.25:1.

    Requisitos:
      1. close < EMA50 E EMA50 < EMA200 (EMA stack — tendencia real)
      2. high >= EMA20 (preco tocou EMA20 — rally)
      3. close < EMA20 (rejeitou — bounce baixista)
      4. RSI 38-62 (zona moderada)
      5. MACD hist < 0 OU macd < macd_signal (momentum alinhado)
      6. ATR percentile 10-90
      7. SL 2.0x, TP 4.5x ATR, max 72 bars
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

    # 1. EMA stack — tendencia REAL
    if not (close < ema50 < ema200):
        return None
    # 2. Price touched EMA20 (rally)
    if high < ema20:
        return None
    # 3. Price rejected below EMA20
    if close >= ema20:
        return None
    # 4. RSI zone — v18.0: 38-62 (mais amplo)
    if not (38.0 <= rsi <= 62.0):
        return None
    # 5. v18.0: MACD alinhado — pelo menos um indicador de momentum negativo
    if not (macd_hist < 0 or macd_val < macd_sig):
        return None
    # 6. ATR percentile
    if not (0.10 <= atr_pct <= 0.90):
        return None
    # 7. Minimum ATR
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


def evaluate_squeeze_breakout_long(row: pd.Series) -> Optional[Signal]:
    """Squeeze Breakout LONG — BBWP squeeze seguido de breakout superior.

    v18.2: SL 2.0x, TP 6.0x ATR, BBWP < 25, max 72 bars.

    Requisitos:
      1. BBWP < 25 (squeeze — volatilidade comprimida)
      2. close > bb_upper (breakout da banda superior)
      3. RSI > 45 (momentum de alta)
      4. ATR percentile 10-90
      5. SL 2.0x, TP 6.0x ATR, max 72 bars
    """
    close = float(row["close"])
    bb_upper = float(row["bb_upper"])
    bbwp = float(row.get("bbwp", 50.0))
    rsi = float(row["rsi"])
    atr = float(row["atr"])
    atr_pct = float(row.get("atr_percentile", 0.5))

    # 1. BB Squeeze
    if pd.isna(bbwp) or bbwp >= SQUEEZE_BBWP_THRESHOLD:
        return None
    # 2. Breakout above upper BB
    if close <= bb_upper:
        return None
    # 3. RSI momentum
    if rsi <= 45.0:
        return None
    # 4. ATR percentile
    if not (0.10 <= atr_pct <= 0.90):
        return None
    # 5. Minimum ATR
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
    """Squeeze Breakout SHORT — BBWP squeeze seguido de breakout inferior.

    v18.2: SL 2.0x, TP 6.0x ATR, BBWP < 25, max 72 bars.

    Requisitos:
      1. BBWP < 25 (squeeze)
      2. close < bb_lower (breakout da banda inferior)
      3. RSI < 55 (momentum de baixa)
      4. ATR percentile 10-90
      5. SL 2.0x, TP 6.0x ATR, max 72 bars
    """
    close = float(row["close"])
    bb_lower = float(row["bb_lower"])
    bbwp = float(row.get("bbwp", 50.0))
    rsi = float(row["rsi"])
    atr = float(row["atr"])
    atr_pct = float(row.get("atr_percentile", 0.5))

    # 1. BB Squeeze
    if pd.isna(bbwp) or bbwp >= SQUEEZE_BBWP_THRESHOLD:
        return None
    # 2. Breakout below lower BB
    if close >= bb_lower:
        return None
    # 3. RSI momentum
    if rsi >= 55.0:
        return None
    # 4. ATR percentile
    if not (0.10 <= atr_pct <= 0.90):
        return None
    # 5. Minimum ATR
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


def evaluate_range_long(row: pd.Series) -> Optional[Signal]:
    """v19.0: Range Trader LONG — BB lower band bounce em mercado lateral.

    Detecta mercados lateralizados (ADX < 20) e opera o bounce
    da banda inferior com SL justo e TP rapido.

    Requisitos:
      1. ADX < 20 (sem tendencia — mercado lateral confirmado)
      2. close <= bb_lower * (1 + RANGE_BB_TOUCH_THRESHOLD) — perto da banda inf.
      3. RSI < RANGE_RSI_OVERSOLD (38) — zona de sobrevenda relativa
      4. close > bb_lower * (1 - 0.02) — nao fora da banda (evita breakdown)
      5. ATR percentile 10-90
      6. SL 1.5x, TP 3.5x ATR, max 36 bars
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

    # 1. Lateral market (no trend)
    if pd.isna(adx) or adx >= RANGE_ADX_MAX:
        return None

    # 2. Price near/at lower BB band
    if close > bb_lower * (1 + RANGE_BB_TOUCH_THRESHOLD):
        return None

    # 3. Not too far below BB (avoid breakdown)
    if close < bb_lower * (1 - 0.02):
        return None

    # 4. RSI oversold relative
    if rsi >= RANGE_RSI_OVERSOLD:
        return None

    # 5. ATR percentile
    if not (0.10 <= atr_pct <= 0.90):
        return None

    # 6. Minimum ATR
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
    """v19.0: Range Trader SHORT — BB upper band bounce em mercado lateral.

    Requisitos:
      1. ADX < 20 (sem tendencia)
      2. close >= bb_upper * (1 - RANGE_BB_TOUCH_THRESHOLD)
      3. RSI > RANGE_RSI_OVERBOUGHT (62)
      4. close < bb_upper * (1 + 0.02) — nao fora da banda
      5. ATR percentile 10-90
      6. SL 1.5x, TP 3.5x ATR, max 36 bars
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

    # 1. Lateral market
    if pd.isna(adx) or adx >= RANGE_ADX_MAX:
        return None

    # 2. Price near/at upper BB band
    if close < bb_upper * (1 - RANGE_BB_TOUCH_THRESHOLD):
        return None

    # 3. Not too far above BB (avoid breakout)
    if close > bb_upper * (1 + 0.02):
        return None

    # 4. RSI overbought relative
    if rsi <= RANGE_RSI_OVERBOUGHT:
        return None

    # 5. ATR percentile
    if not (0.10 <= atr_pct <= 0.90):
        return None

    # 6. Minimum ATR
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


def evaluate_scalp_long(row: pd.Series) -> Optional[Signal]:
    """v19.1: Scalp LONG — Quick intraday trade em qualquer regime.

    Usa RSI turning up + MACD alignment para entradas rapidas.
    Alta frequencia: max 18 bars, SL justo 1.0x ATR.

    Requisitos:
      1. close > EMA20 (preco acima media curta)
      2. RSI > 40 e RSI delta > 0.8 (virada para cima)
      3. MACD hist > 0 OU macd > macd_signal
      4. ATR percentile 10-90
      5. SL 1.0x, TP 1.5x ATR, max 18 bars
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

    # 1. Above EMA20
    if close <= ema20:
        return None

    # 2. RSI turning up
    if rsi <= SCALP_RSI_LONG_MIN:
        return None
    if rsi_delta <= SCALP_RSI_DELTA_TRIGGER:
        return None

    # 3. MACD aligned
    if not (macd_hist > 0 or macd_val > macd_sig):
        return None

    # 4. Not in volatile regime
    if adx > 60:
        return None

    # 5. ATR percentile
    if not (0.10 <= atr_pct <= 0.90):
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
    """v19.1: Scalp SHORT — Quick intraday trade em qualquer regime.

    Requisitos:
      1. close < EMA20
      2. RSI < 60 e RSI delta < -0.8 (virada para baixo)
      3. MACD hist < 0 OU macd < macd_signal
      4. ATR percentile 10-90
      5. SL 1.0x, TP 1.5x ATR, max 18 bars
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

    # 1. Below EMA20
    if close >= ema20:
        return None

    # 2. RSI turning down
    if rsi >= SCALP_RSI_SHORT_MAX:
        return None
    if rsi_delta >= -SCALP_RSI_DELTA_TRIGGER:
        return None

    # 3. MACD aligned
    if not (macd_hist < 0 or macd_val < macd_sig):
        return None

    # 4. Not in volatile regime
    if adx > 60:
        return None

    # 5. ATR percentile
    if not (0.10 <= atr_pct <= 0.90):
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


def evaluate_rsi_extremes_long(row: pd.Series) -> Optional[Signal]:
    """v19.0: RSI Extremes LONG — Reversao por RSI extremo.

    Detecta sobrevenda extrema (RSI < 25) com virada confirmada
    (rsi_delta > 0.5). Funciona em QUALQUER regime (ranging + trending).

    Requisitos:
      1. RSI < 25 (sobrevenda extrema)
      2. rsi_delta > 0.5 (virada para cima confirmada)
      3. ATR percentile 10-90
      4. SL 1.5x, TP 3.0x ATR, max 36 bars
    """
    close = float(row["close"])
    rsi = float(row["rsi"])
    rsi_delta = float(row.get("rsi_delta", 0.0))
    atr = float(row["atr"])
    atr_pct = float(row.get("atr_percentile", 0.5))

    # 1. Extreme oversold
    if rsi >= RSI_EXT_OVERSOLD:
        return None

    # 2. Turning up (reversal confirmation)
    if rsi_delta <= RSI_EXT_RSI_DELTA_MIN:
        return None

    # 3. ATR percentile
    if not (0.10 <= atr_pct <= 0.90):
        return None

    # 4. Minimum ATR
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
    """v19.0: RSI Extremes SHORT — Reversao por RSI extremo.

    Detecta sobrecompra extrema (RSI > 75) com virada confirmada.
    Funciona em QUALQUER regime.

    Requisitos:
      1. RSI > 75 (sobrecompra extrema)
      2. rsi_delta < -0.5 (virada para baixo confirmada)
      3. ATR percentile 10-90
      4. SL 1.5x, TP 3.0x ATR, max 36 bars
    """
    close = float(row["close"])
    rsi = float(row["rsi"])
    rsi_delta = float(row.get("rsi_delta", 0.0))
    atr = float(row["atr"])
    atr_pct = float(row.get("atr_percentile", 0.5))

    # 1. Extreme overbought
    if rsi <= RSI_EXT_OVERBOUGHT:
        return None

    # 2. Turning down (reversal confirmation)
    if rsi_delta >= -RSI_EXT_RSI_DELTA_MIN:
        return None

    # 3. ATR percentile
    if not (0.10 <= atr_pct <= 0.90):
        return None

    # 4. Minimum ATR
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


def evaluate_rsi_reversal_long(row: pd.Series) -> Optional[Signal]:
    """RSI Reversal LONG -- RSI sobrevendido em tendencia de alta = compra.

    v21.0: RSI < 38 (de 40) mais seletivo. SL 3.0x, TP 6.5x.

    Requisitos:
      1. ema50 > ema200 (tendencia de alta estabelecida)
      2. RSI < 38 (sobrevenda relativa na tendencia)
      3. rsi_delta > 0 (RSI virando para cima -- reversao)
      4. close > ema50 (ainda acima da MA media)
      5. ATR percentile 10-90
      6. SL 3.0x, TP 6.5x ATR, max 96 bars
    """
    close = float(row["close"])
    ema50 = float(row["ema50"])
    ema200 = float(row["ema200"])
    rsi = float(row["rsi"])
    rsi_delta = float(row.get("rsi_delta", 0.0))
    atr = float(row["atr"])
    atr_pct = float(row.get("atr_percentile", 0.5))

    # 1. Established uptrend (EMA stack)
    if not (ema50 > ema200):
        return None
    # 2. RSI oversold in uptrend -- v21.0: 38 (de 40)
    if rsi >= 38.0:
        return None
    # 3. RSI turning up (reversal confirmation)
    if rsi_delta <= 0:
        return None
    # 4. Still above medium-term MA
    if close <= ema50:
        return None
    # 5. ATR percentile
    if not (0.10 <= atr_pct <= 0.90):
        return None
    # 6. Minimum ATR
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
    """RSI Reversal SHORT -- RSI sobrecomprado em tendencia de baixa = venda.

    v21.0: RSI > 62 (de 60) mais seletivo. SL 3.0x, TP 6.5x.

    Requisitos:
      1. ema50 < ema200 (tendencia de baixa estabelecida)
      2. RSI > 62 (sobrecompra relativa na tendencia)
      3. rsi_delta < 0 (RSI virando para baixo -- reversao)
      4. close < ema50 (ainda abaixo da MA media)
      5. ATR percentile 10-90
      6. SL 3.0x, TP 6.5x ATR, max 96 bars
    """
    close = float(row["close"])
    ema50 = float(row["ema50"])
    ema200 = float(row["ema200"])
    rsi = float(row["rsi"])
    rsi_delta = float(row.get("rsi_delta", 0.0))
    atr = float(row["atr"])
    atr_pct = float(row.get("atr_percentile", 0.5))

    # 1. Established downtrend (EMA stack)
    if not (ema50 < ema200):
        return None
    # 2. RSI overbought in downtrend -- v21.0: 62 (de 60)
    if rsi <= 62.0:
        return None
    # 3. RSI turning down (reversal confirmation)
    if rsi_delta >= 0:
        return None
    # 4. Still below medium-term MA
    if close >= ema50:
        return None
    # 5. ATR percentile
    if not (0.10 <= atr_pct <= 0.90):
        return None
    # 6. Minimum ATR
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

