"""
strategy.py
-----------
Logica de validacao das condicoes de entrada CTEV v4.4 para LONG e SHORT.

Estrategia CTEV v4.4 = Regime-Based Trend-Following com Pullback

v4.4 — PARAMETROS OTIMIZADOS VIA GRID SEARCH MASSIVO (15.110 combinacoes):
  Phase 1: 14.400 filtros (RSI, ADX, Vol, Fib, Trans, VC)
  Phase 2: 700 SL/TP refinamentos
  Phase 3: 23 SL/TP com validacao avancada
  Total: 15.123 combinacoes em ~22s (numpy-accelerated)

  Melhor resultado (simulacao basica):
    13 trades, WR 69.2%, PF 4.48, PnL +18.64%, DD 1.92%
    (vs Buy&Hold +9.88% = alpha +8.76pp)

  Parametros vencedores:
    - RSI LONG: 28-48 (era 20-65 em v4.2)
    - RSI SHORT: 55-75 (era 35-80 em v4.2)
    - ADX_MIN: 30 (era 25 em v4.3 — tendencia mais forte)
    - VOLUME_CONFIRM: False (era True — removido, degradava)
    - VOLUME_SMA_RATIO: 0.30 (era 0.50)
    - FIB_TOLERANCE: 2.5% (mantido de v4.3)
    - ALLOW_TRANSITION: False (mantido de v4.3)
    - SL/TP: 1.75x / 4.0x ATR (R:R 2.3:1 — otimizado)
    - Trailing/BE: desabilitado (degradava pullback entries)
    - EMA proximity: desabilitado (mantido de v4.3)

Chave: RSI estreito + ADX>=30 + sem transition + R:R 2.3:1
Referencias:
    - Grid search massivo com 17.398 candles BTC/USDT 1H (2 anos)
    - Buy&Hold do periodo: +9.88%
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd

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
    pullback_type: str  # "fibonacci", "ema20_touch", "ema50_touch", "fib_ema_combo"
    ema50_slope: float
    timestamp: pd.Timestamp

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
            "ema50_slope": round(self.ema50_slope, 6),
            "timestamp": str(self.timestamp),
        }


# ── Parametros da estrategia CTEV v4.4 ──
# v4.4: Otimizado via grid search massivo — 15.123 combinacoes em 22s
# Melhor resultado (basico): 13 trades, WR 69.2%, PF 4.48, PnL +18.64%, DD 1.92%

# REGIME FILTER (core — apenas trending, sem transition)
ADX_MIN = 30.0                # ADX minimo para trending (era 25 em v4.3)
ALLOW_TRANSITION = False      # True = aceita regime 'transition', False = apenas trending

# RSI como zona de pullback (v4.4: otimizado)
RSI_LONG_MIN = 28.0           # RSI 28-48 para LONG
RSI_LONG_MAX = 48.0
RSI_SHORT_MIN = 55.0          # RSI 55-75 para SHORT
RSI_SHORT_MAX = 75.0

# RSI Delta — desabilitado
RSI_DELTA_LONG_MIN = -5.0    # effectively disabled
RSI_DELTA_SHORT_MAX = 5.0    # effectively disabled

# Volume (v4.4: DESABILITADO — removido pelo grid search massivo)
VOLUME_CONFIRM = False
VOLUME_SMA_RATIO = 0.30       # (mantido como referencia, nao usado quando VC=False)

# Fibonacci tolerancia (v4.4: 2.5% — confirmado otimo)
FIB_TOLERANCE_PCT = 0.025     # 2.5%

# ATR Percentile filter
ATR_PCT_MIN = 0.10
ATR_PCT_MAX = 0.90

# Bollinger Bandwidth — desabilitado
BB_WIDTH_MIN = 0.0
BB_WIDTH_MAX = 999.0

# EMA proximity — DESABILITADO
EMA20_PROXIMITY_PCT = 0.0
EMA50_PROXIMITY_PCT = 0.0

# EMA Slope (confirmacao de tendencia — mantido)
EMA50_SLOPE_MIN = 0.0

# Gestao de risco — R:R 2.3:1 (otimizado via grid search SL/TP)
SL_ATR_MULT = 1.75
TP_ATR_MULT = 4.0


def _price_near_fib(price: float, fib_level: float, tolerance_pct: float = FIB_TOLERANCE_PCT) -> bool:
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


def evaluate_long(row: pd.Series) -> Optional[Signal]:
    """
    Avalia condicoes LONG (regime-based trend-following com pullback):

    Requisitos (CTEV v4.3 — 6 filtros, otimizados via grid search):
      1. REGIME: trending_up (ADX>25) — sem transition
      2. TENDENCIA: close > EMA(50) E EMA(50) > EMA(200)
      3. SLOPE: ema50_slope > 0
      4. PULLBACK: Fibonacci (tol 2.5%) OU EMA(20/50) touch
      5. RSI: LONG 28-48, SHORT 55-75
      6. VOLUME: volume > 50% SMA(50)
      7. ATR: Percentile 10%-90%
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

    # 1. REGIME FILTER (v4.4: ALLOW_TRANSITION toggle)
    if regime == "trending_up":
        if pd.isna(adx) or adx < ADX_MIN:
            return None
    elif regime == "transition":
        if not ALLOW_TRANSITION:
            return None
    else:
        return None

    # 2. TENDENCIA: Dual EMA — uptrend confirmado
    if not (close > ema50 and ema50 > ema200):
        return None

    # 3. SLOPE: EMA50 deve estar subindo
    if pd.isna(ema50_slope) or ema50_slope <= EMA50_SLOPE_MIN:
        return None

    # 4. PULLBACK: Fibonacci zone OU EMA touch (v4.4: sem EMA proximity)
    # Nota: EMA proximity desabilitado (EMA20/50_PROXIMITY_PCT = 0)
    # Fibonacci check e EMA touch mantidos como em v4.3

    # 4. PULLBACK: Fibonacci zone OU EMA touch (v4.4)
    pullback_type = None

    # Fibonacci check (tolerancia 2.5% v4.4)
    in_fib = _in_fib_zone(close, fib_0382, fib_0618, fib_dir)
    if in_fib and fib_dir == 1:
        pullback_type = "fibonacci"
    else:
        if fib_dir == 1:
            if (_price_near_fib(low, fib_0382) or
                    _price_near_fib(low, fib_0500) or
                    _price_near_fib(low, fib_0618)):
                pullback_type = "fibonacci"

    # EMA(20) touch (low cruzou EMA20 e close recuperou)
    if pullback_type is None:
        if bool(row.get("ema20_touched", False)) and close > ema20:
            pullback_type = "ema20_touch"

    # EMA(50) touch
    if pullback_type is None:
        if bool(row.get("ema50_touched", False)) and close > ema50:
            pullback_type = "ema50_touch"

    if pullback_type is None:
        return None

    # 5. RSI: Zona de pullback alargada (20-65)
    if not (RSI_LONG_MIN <= rsi <= RSI_LONG_MAX):
        return None

    # 6. VOLUME: Soft confirmation v4.2 (40% da SMA50)
    if VOLUME_CONFIRM and (not pd.isna(volume_sma50) and volume_sma50 > 0):
        if volume < volume_sma50 * VOLUME_SMA_RATIO:
            return None

    # 7. ATR: Volatilidade na faixa normal
    if not (ATR_PCT_MIN <= atr_pct <= ATR_PCT_MAX):
        return None

    # NOTA v4.2: Filtros MACD, RSI Delta e BB Width REMOVIDOS
    # (eram redundantes/restritivos demais — ver docstring do modulo)

    # ── Gestao de risco LONG — R:R 2:1 ──
    entry = close
    stop_loss = entry - (SL_ATR_MULT * atr)
    take_profit = entry + (TP_ATR_MULT * atr)

    if stop_loss <= 0:
        return None

    logger.info(
        "SINAL LONG v4.4 | entry=%.2f SL=%.2f TP=%.2f ATR=%.2f "
        "RSI=%.1f ADX=%.1f regime=%s pullback=%s",
        entry, stop_loss, take_profit, atr, rsi, adx, regime, pullback_type,
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
        ema50_slope=ema50_slope,
        timestamp=ts,
    )


def evaluate_short(row: pd.Series) -> Optional[Signal]:
    """
    Avalia condicoes SHORT (regime-based trend-following com pullback):

    Requisitos (CTEV v4.3 — 6 filtros, otimizados via grid search):
      1. REGIME: trending_down (ADX>25) — sem transition
      2. TENDENCIA: close < EMA(50) EMA(50) < EMA(200)
      3. SLOPE: ema50_slope < 0
      4. PULLBACK: Fibonacci (tol 2.5%) OU EMA(20/50) touch
      5. RSI: LONG 28-48, SHORT 55-75
      6. VOLUME: volume > 50% SMA(50)
      7. ATR: Percentile 10%-90%
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

    # 1. REGIME FILTER (v4.4: ALLOW_TRANSITION toggle)
    if regime == "trending_down":
        if pd.isna(adx) or adx < ADX_MIN:
            return None
    elif regime == "transition":
        if not ALLOW_TRANSITION:
            return None
    else:
        return None

    # 2. TENDENCIA: Dual EMA — downtrend confirmado
    if not (close < ema50 and ema50 < ema200):
        return None

    # 3. SLOPE: EMA50 deve estar descendo
    if pd.isna(ema50_slope) or ema50_slope >= -EMA50_SLOPE_MIN:
        return None

    # 4. PULLBACK: Fibonacci OU EMA touch (v4.4)
    pullback_type = None

    # Fibonacci check
    in_fib = _in_fib_zone(close, fib_0382, fib_0618, fib_dir)
    if in_fib and fib_dir == -1:
        pullback_type = "fibonacci"
    else:
        if fib_dir == -1:
            if (_price_near_fib(high, fib_0382) or
                    _price_near_fib(high, fib_0500) or
                    _price_near_fib(high, fib_0618)):
                pullback_type = "fibonacci"

    # EMA(20) touch (high cruzou EMA20 e close rejeitou abaixo)
    if pullback_type is None:
        if bool(row.get("ema20_touched", False)) and close < ema20:
            if high >= ema20:
                pullback_type = "ema20_touch"

    # EMA(50) touch
    if pullback_type is None:
        if bool(row.get("ema50_touched_up", False)) and close < ema50:
            if high >= ema50:
                pullback_type = "ema50_touch"

    if pullback_type is None:
        return None

    # 5. RSI: Zona de rally alargada (35-80)
    if not (RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX):
        return None

    # 6. VOLUME: Soft confirmation v4.2 (40% da SMA50)
    if VOLUME_CONFIRM and (not pd.isna(volume_sma50) and volume_sma50 > 0):
        if volume < volume_sma50 * VOLUME_SMA_RATIO:
            return None

    # 7. ATR: Volatilidade na faixa normal
    if not (ATR_PCT_MIN <= atr_pct <= ATR_PCT_MAX):
        return None

    # NOTA v4.2: Filtros MACD, RSI Delta e BB Width REMOVIDOS

    # ── Gestao de risco SHORT — R:R 2:1 ──
    entry = close
    stop_loss = entry + (SL_ATR_MULT * atr)
    take_profit = entry - (TP_ATR_MULT * atr)

    logger.info(
        "SINAL SHORT v4.4 | entry=%.2f SL=%.2f TP=%.2f ATR=%.2f "
        "RSI=%.1f ADX=%.1f regime=%s pullback=%s",
        entry, stop_loss, take_profit, atr, rsi, adx, regime, pullback_type,
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
        ema50_slope=ema50_slope,
        timestamp=ts,
    )


def evaluate_signal(df_ind: pd.DataFrame) -> Optional[Signal]:
    """
    Ponto de entrada principal da estrategia v4.
    Recebe o DataFrame com indicadores e avalia apenas a ultima linha.
    Retorna um Signal LONG, SHORT ou None.
    """
    if df_ind.empty:
        return None

    last = df_ind.iloc[-1]
    signal = evaluate_long(last)
    if signal is not None:
        return signal
    return evaluate_short(last)
