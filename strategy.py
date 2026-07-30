"""
strategy.py
-----------
Logica de validacao das condicoes de entrada CTEV v4.1 para LONG e SHORT.

Estrategia CTEV v4.1 = Regime-Based Trend-Following com Fibonacci Pullback

Mudancas v4.1 vs v4 (relaxamento controlado apos diagnostico de 0 sinais em 17.4K candles):
    - REGIME: transition agora gera sinais com ADX > 20 (antes era bloqueado)
    - RSI: faixas alargadas LONG 25-55 (era 30-50), SHORT 45-75 (era 50-70)
    - RSI DELTA: tolerancia de -1/+1 (era 0 exato)
    - VOLUME: agora requer volume > 60% da SMA(50) (era 100%)
    - FIBONACCI: tolerancia 2.5% (era 1.5%)
    - ATR PERCENTILE: 10-90% (era 15-85%)
    - BB WIDTH: 0.5-20% (era 1-15%)

Mudancas v4 vs v3:
    - REGIME FILTER: ADX > 25 para trending, ADX > 20 para transition
    - VOLUME FILTER re-ativado (soft): volume > 60% SMA(50)
    - R:R OTIMIZADO: SL = 1.5 ATR, TP = 3.0 ATR (R:R 2:1)
    - EMA(20) como pullback primario, EMA(50) SLOPE como confirmacao

Por que essas mudancas (baseado no PDF "Framework Multi-Timeframe e de Regimes"):
    1. O estudo de Adaptive Regime-Based Trading (ref. 48) alcancou CAGR 70.94%
       com max DD de -20.42% usando regime classification — vs buy-hold 33.48%/-77.22%
    2. ADX > 25 e o padrao-ouro para filtrar mercados laterais (Quantpedia, 2025)
    3. O sistema mestre do PDF usa EMA(50) slope no 4H para definir tendencia,
       e EMA(20) no 1H como zona de pullback
    4. Volume > SMA(50) e menos restritivo que SMA(20) mas filtra entrada sem
       confirmacao (versao equilibrada do PDF)
    5. SL de 1.5 ATR reduz perda media por trade sem sacrificar o R:R

Referencias:
    - PDF: "O Framework Multi-Timeframe e de Regimes como Chave para
      Estrategias Robustas em BTC/USDT no Timeframe de 1H"
    - Adaptive Regime-Based Trading on Bitcoin (ref. 48): CAGR 70.94%, DD -20.42%
    - Quantpedia (2025): 355 estrategias — mediana deterioracao Sharpe 43.90%
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


# ── Parametros da estrategia CTEV v4.1 ──
# v4.1: Relaxamento controlado de filtros para gerar sinais reais
# Diagnostico mostrou 0 sinais em 17.4K candles — filtros excessivamente restritivos

# REGIME FILTER (CRITICO — baseado no PDF)
ADX_MIN = 25.0                # ADX minimo para trending (mantido para trending)
ADX_MIN_TRANSITION = 20.0     # ADX minimo para transition (mais permissivo)
# Nota: v4.1 permite operar em transition com ADX > 20 (antes era bloqueado)

# RSI como zona de pullback (v4.1: faixas alargadas)
RSI_LONG_MIN = 25.0           # RSI > 25 para LONG (era 30 — mais permissivo)
RSI_LONG_MAX = 55.0           # RSI < 55 para LONG (era 50 — captura mais pullbacks)
RSI_SHORT_MIN = 45.0          # RSI > 45 para SHORT (era 50)
RSI_SHORT_MAX = 75.0          # RSI < 75 para SHORT (era 70)

# RSI Delta (momentum turning)
RSI_DELTA_LONG_MIN = -1.0     # RSI pode estar levemente descendo para LONG (era 0)
RSI_DELTA_SHORT_MAX = 1.0     # RSI pode estar levemente subindo para SHORT (era 0)

# Volume (soft filter — v4.1: mais suave)
VOLUME_CONFIRM = True         # Ativado
VOLUME_SMA_RATIO = 0.60       # volume > 60% da SMA(50) (era 100% — muito restritivo)

# Fibonacci tolerancia (v4.1: alargada)
FIB_TOLERANCE_PCT = 0.025     # 2.5% (era 1.5% — mais leeway no pullback)

# ATR Percentile filter
ATR_PCT_MIN = 0.10            # 10% (era 15% — permite mais volatilidade baixa)
ATR_PCT_MAX = 0.90            # 90% (era 85% — permite mais volatilidade alta)

# Bollinger Bandwidth (v4.1: alargado)
BB_WIDTH_MIN = 0.5            # 0.5% (era 1.0%)
BB_WIDTH_MAX = 20.0           # 20% (era 15%)

# EMA Slope (confirmacao de tendencia — do PDF)
EMA50_SLOPE_MIN = 0.0         # Slope > 0 para uptrend, < 0 para downtrend

# Gestao de risco — R:R 2:1 (SL mais apertado que v3)
SL_ATR_MULT = 1.5             # Stop = Entry - 1.5 * ATR (vs 2.0 da v3)
TP_ATR_MULT = 3.0             # TP = Entry + 3.0 * ATR (vs 4.0 da v3, R:R mantido em 2:1)


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

    Requisitos (CTEV v4.1 — 10 filtros com relaxamento controlado):
      1. REGIME: trending_up (ADX>25) OU transition (ADX>20)
      2. TENDENCIA: close > EMA(50) E EMA(50) > EMA(200)
      3. SLOPE: ema50_slope > 0
      4. PULLBACK: Fibonacci (tol 2.5%) OU EMA(20/50) touch
      5. RSI: 25 < RSI < 55
      6. RSI DELTA: rsi_delta > -1
      7. MACD: Histograma > 0 OU MACD > Signal
      8. VOLUME: volume > 60% SMA(50)
      9. ATR: Percentile 10%-90%
      10. BB: Bandwidth 0.5%-20%
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

    # 1. REGIME FILTER (v4.1: permite transition com ADX > 20)
    if pd.isna(adx) or adx < ADX_MIN_TRANSITION:
        return None

    # 1b. Regime deve ser trending_up OU transition (v4.1)
    #    transition agora permite sinais com ADX > 20 (antes era bloqueado)
    if regime == "trending_up":
        # Regime forte: mantem ADX > 25
        if adx < ADX_MIN:
            return None
    elif regime == "transition":
        # Regime transicao: permite com ADX > 20 (mais permissivo)
        pass
    else:
        return None

    # 2. TENDENCIA: Dual EMA — uptrend confirmado
    if not (close > ema50 and ema50 > ema200):
        return None

    # 3. SLOPE: EMA50 deve estar subindo (do PDF: slope > 0)
    if pd.isna(ema50_slope) or ema50_slope <= EMA50_SLOPE_MIN:
        return None

    # 4. PULLBACK: Fibonacci zone OU EMA(20) touch OU EMA(50) touch
    pullback_type = None

    # Fibonacci check (prioridade maxima)
    in_fib = _in_fib_zone(close, fib_0382, fib_0618, fib_dir)
    if in_fib and fib_dir == 1:
        pullback_type = "fibonacci"
    else:
        # Tolerancia: low tocou um nivel Fibonacci
        if fib_dir == 1:
            if (_price_near_fib(low, fib_0382, 0.02) or
                    _price_near_fib(low, fib_0500, 0.02) or
                    _price_near_fib(low, fib_0618, 0.02)):
                pullback_type = "fibonacci"

    # EMA(20) touch (pullback primario — recomendado pelo PDF)
    if pullback_type is None:
        if bool(row.get("ema20_touched", False)) and close > ema20:
            pullback_type = "ema20_touch"

    # EMA(50) touch (pullback secundario)
    if pullback_type is None:
        if bool(row.get("ema50_touched", False)) and close > ema50:
            pullback_type = "ema50_touch"

    # Combo: fib + ema20
    if pullback_type == "fibonacci" and bool(row.get("ema20_touched", False)):
        pullback_type = "fib_ema_combo"

    if pullback_type is None:
        return None

    # 5. RSI: Zona de pullback (30-50)
    if not (RSI_LONG_MIN <= rsi <= RSI_LONG_MAX):
        return None

    # 6. RSI DELTA: Momentum virando para cima (v4.1: tolerancia de -1)
    if rsi_delta < RSI_DELTA_LONG_MIN:
        return None

    # 7. MACD: Momentum virando para cima
    if not _macd_bullish(macd_hist, macd_val, macd_sig):
        return None

    # 8. VOLUME: Soft confirmation v4.1 (60% da SMA50 em vez de 100%)
    if VOLUME_CONFIRM and (not pd.isna(volume_sma50) and volume_sma50 > 0):
        if volume < volume_sma50 * VOLUME_SMA_RATIO:
            return None

    # 9. ATR: Volatilidade na faixa normal
    if not (ATR_PCT_MIN <= atr_pct <= ATR_PCT_MAX):
        return None

    # 10. BOLLINGER: Evitar squeeze extremo
    if bb_w < BB_WIDTH_MIN or bb_w > BB_WIDTH_MAX:
        return None

    # ── Gestao de risco LONG — R:R 2:1 (SL mais apertado) ──
    entry = close
    stop_loss = entry - (SL_ATR_MULT * atr)
    take_profit = entry + (TP_ATR_MULT * atr)

    # Sanity: SL nao pode ser negativo
    if stop_loss <= 0:
        return None

    logger.info(
        "SINAL LONG v4 | entry=%.2f SL=%.2f TP=%.2f R:R=1:2 ATR=%.2f "
        "RSI=%.1f dRSI=%.1f ADX=%.1f regime=%s MACD_hist=%.4f pullback=%s",
        entry, stop_loss, take_profit, atr, rsi,
        rsi_delta, adx, regime, macd_hist, pullback_type,
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

    Requisitos (CTEV v4.1):
      1. REGIME: trending_down (ADX>25) OU transition (ADX>20)
      2. TENDENCIA: close < EMA(50) E EMA(50) < EMA(200)
      3. SLOPE: ema50_slope < 0
      4. PULLBACK: Fibonacci (tol 2.5%) OU EMA(20/50) touch
      5. RSI: 45 < RSI < 75
      6. RSI DELTA: rsi_delta < 1
      7. MACD: Histograma < 0 OU MACD < Signal
      8. VOLUME: volume > 60% SMA(50)
      9. ATR: Percentile 10%-90%
      10. BB: Bandwidth 0.5%-20%
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

    # 1. REGIME FILTER (v4.1: permite transition com ADX > 20)
    if pd.isna(adx) or adx < ADX_MIN_TRANSITION:
        return None

    # 1b. Regime deve ser trending_down OU transition (v4.1)
    if regime == "trending_down":
        # Regime forte: mantem ADX > 25
        if adx < ADX_MIN:
            return None
    elif regime == "transition":
        # Regime transicao: permite com ADX > 20
        pass
    else:
        return None

    # 2. TENDENCIA: Dual EMA — downtrend confirmado
    if not (close < ema50 and ema50 < ema200):
        return None

    # 3. SLOPE: EMA50 deve estar descendo
    if pd.isna(ema50_slope) or ema50_slope >= -EMA50_SLOPE_MIN:
        return None

    # 4. PULLBACK: Fibonacci zone OU EMA(20) touch OU EMA(50) touch
    pullback_type = None

    # Fibonacci check
    in_fib = _in_fib_zone(close, fib_0382, fib_0618, fib_dir)
    if in_fib and fib_dir == -1:
        pullback_type = "fibonacci"
    else:
        if fib_dir == -1:
            if (_price_near_fib(high, fib_0382, 0.02) or
                    _price_near_fib(high, fib_0500, 0.02) or
                    _price_near_fib(high, fib_0618, 0.02)):
                pullback_type = "fibonacci"

    # EMA(20) touch (pullback primario)
    if pullback_type is None:
        if bool(row.get("ema20_touched", False)) and close < ema20:
            # Para short: close < EMA20 (rally toca EMA20 e rejeita)
            # Mais preciso: high tocou EMA20 e close rejeitou abaixo
            if high >= ema20:
                pullback_type = "ema20_touch"

    # EMA(50) touch (pullback secundario)
    if pullback_type is None:
        if bool(row.get("ema50_touched_up", False)) and close < ema50:
            if high >= ema50:
                pullback_type = "ema50_touch"

    if pullback_type is None:
        return None

    # 5. RSI: Zona de rally em downtrend (50-70)
    if not (RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX):
        return None

    # 6. RSI DELTA: Momentum virando para baixo (v4.1: tolerancia de +1)
    if rsi_delta > RSI_DELTA_SHORT_MAX:
        return None

    # 7. MACD: Momentum virando para baixo
    if not _macd_bearish(macd_hist, macd_val, macd_sig):
        return None

    # 8. VOLUME: Soft confirmation v4.1 (60% da SMA50)
    if VOLUME_CONFIRM and (not pd.isna(volume_sma50) and volume_sma50 > 0):
        if volume < volume_sma50 * VOLUME_SMA_RATIO:
            return None

    # 9. ATR: Volatilidade na faixa normal
    if not (ATR_PCT_MIN <= atr_pct <= ATR_PCT_MAX):
        return None

    # 10. BOLLINGER: Evitar squeeze extremo
    if bb_w < BB_WIDTH_MIN or bb_w > BB_WIDTH_MAX:
        return None

    # ── Gestao de risco SHORT — R:R 2:1 ──
    entry = close
    stop_loss = entry + (SL_ATR_MULT * atr)
    take_profit = entry - (TP_ATR_MULT * atr)

    logger.info(
        "SINAL SHORT v4 | entry=%.2f SL=%.2f TP=%.2f R:R=1:2 ATR=%.2f "
        "RSI=%.1f dRSI=%.1f ADX=%.1f regime=%s MACD_hist=%.4f pullback=%s",
        entry, stop_loss, take_profit, atr, rsi,
        rsi_delta, adx, regime, macd_hist, pullback_type,
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
