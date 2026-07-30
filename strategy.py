"""
strategy.py
-----------
Logica de validacao das condicoes de entrada CTEV v3 para LONG e SHORT.

Estrategia CTEV v3 = Fibonacci Pullback Trend-Following com MACD Confirmation

Principais mudancas vs v2:
    - RSI invertido: agora usado como ZONA DE PULLBACK (RSI 30-50 para LONG,
      nao como confirmacao de momentum > 55)
    - MACD(12,26,9) adicionado como confirmacao de virada de momentum
    - EMA(50) touch como pullback alternativo (alem de Fibonacci)
    - SL mais largo: 2.0 ATR (vs 1.5) — reduz stop hunting
    - TP mais realista: 4.0 ATR (vs 4.5) — R:R 2:1 (vs 3:1)
    - Volume removido como filtro obrigatorio (era muito restritivo em 1H)
    - BB squeeze minimo reduzido para 1.0% (mais permisivo)
    - Fibonacci swing lookback aumentado (120 vs 50) para capturar movimentos maiores
    - Min diff de 1% entre swing high/low para filtrar ruido

Referencias:
    - Quantpedia (2025): "Multi-Timeframe Trend Strategy on Bitcoin"
    - prorsi.com (2026): "Best Combination of Indicators" — 68% WR backtest
    - QuantifiedStrategies: "RSI Trading Strategy" — pullback approach
    - Phemex Academy: "Fibonacci Retracement in Crypto Trading"
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
    """Representa um sinal de trade gerado pela estrategia CTEV v3."""
    type: SignalType
    entry_price: float
    stop_loss: float
    take_profit: float
    atr: float
    rsi: float
    rsi_delta: float
    macd_hist: float
    ema50: float
    ema200: float
    bb_lower: float
    bb_upper: float
    volume: float
    volume_sma20: float
    atr_percentile: float
    bb_width: float
    fib_0382: float
    fib_0500: float
    fib_0618: float
    fib_direction: int
    fib_proximity: float
    pullback_type: str  # "fibonacci", "ema50_touch", "fib_ema_combo"
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
            "ema50": self.ema50,
            "ema200": self.ema200,
            "atr_percentile": self.atr_percentile,
            "bb_width": self.bb_width,
            "fib_0382": self.fib_0382,
            "fib_0500": self.fib_0500,
            "fib_0618": self.fib_0618,
            "fib_direction": self.fib_direction,
            "fib_proximity": round(self.fib_proximity, 3) if not pd.isna(self.fib_proximity) else None,
            "pullback_type": self.pullback_type,
            "timestamp": str(self.timestamp),
        }


# ── Parametros da estrategia CTEV v3 ──

# RSI como zona de pullback (NAO como momentum!)
RSI_LONG_MIN = 30.0           # RSI > 30 para LONG (pullback saudavel, nao colapso)
RSI_LONG_MAX = 50.0           # RSI < 50 para LONG (ainda em zona de pullback)
RSI_SHORT_MIN = 50.0          # RSI > 50 para SHORT (rally em downtrend)
RSI_SHORT_MAX = 70.0          # RSI < 70 para SHORT (nao sobrecomprado extremo)

# Fibonacci tolerancia (distancia maxima % do preco ao nivel Fib)
FIB_TOLERANCE_PCT = 0.015     # 1.5% de tolerancia

# ATR Percentile filter (evitar volatilidade extrema)
ATR_PCT_MIN = 0.15
ATR_PCT_MAX = 0.85

# Bollinger Bandwidth (evitar squeeze extremo)
BB_WIDTH_MIN = 1.0             # Minimo reduzido (vs 2.0 da v2)
BB_WIDTH_MAX = 15.0

# Gestao de risco — R:R 2:1
SL_ATR_MULT = 2.0             # Stop = Entry - 2.0 * ATR (mais largo que v2)
TP_ATR_MULT = 4.0             # TP = Entry + 4.0 * ATR  (R:R = 2.0)


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
        # Uptrend: pullback de alta, preco entre fib_0618 (mais baixo) e fib_0382 (mais alto)
        return fib_0618 <= price <= fib_0382
    elif direction == -1:
        # Downtrend: pullback de baixa, preco entre fib_0382 (mais alto) e fib_0618 (mais baixo)
        return fib_0618 <= price <= fib_0382
    return False


def _macd_bullish(macd_hist: float, macd_line: float, macd_signal: float) -> bool:
    """Verifica se MACD indica momentum de alta."""
    # Histograma positivo (momentum ja virou para cima)
    # OU linha MACD acima da linha de sinal (tendencia de alta ativa)
    return macd_hist > 0 or macd_line > macd_signal


def _macd_bearish(macd_hist: float, macd_line: float, macd_signal: float) -> bool:
    """Verifica se MACD indica momentum de baixa."""
    return macd_hist < 0 or macd_line < macd_signal


def evaluate_long(row: pd.Series) -> Optional[Signal]:
    """
    Avalia condicoes LONG (trend-following com pullback Fibonacci/EMA):

    Requisitos:
      1. TENDENCIA: close > EMA(50) E EMA(50) > EMA(200) (uptrend confirmado)
      2. PULLBACK: Preco na zona Fibonacci (0.382-0.618)
         OU low tocou EMA(50) e close recuperou acima da EMA(50)
      3. RSI: 30 < RSI < 50 (zona de pullback saudavel)
      4. MACD: Histograma > 0 OU MACD > Signal (momentum virando para cima)
      5. ATR: Percentile entre 15%-85% (volatilidade normal)
      6. BB: Bandwidth > 1% (nao em squeeze extremo)
    """
    close = float(row["close"])
    low = float(row["low"])
    high = float(row["high"])
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
    atr = float(row["atr"])
    atr_pct = float(row.get("atr_percentile", 0.5))
    bb_w = float(row.get("bb_width", 0.0))
    fib_0382 = float(row.get("fib_0382", float("nan")))
    fib_0500 = float(row.get("fib_0500", float("nan")))
    fib_0618 = float(row.get("fib_0618", float("nan")))
    fib_dir = int(row.get("fib_direction", 0))
    fib_prox = float(row.get("fib_proximity", float("nan")))
    ts = row.name

    # 1. TENDENCIA: Dual EMA — uptrend confirmado
    if not (close > ema50 and ema50 > ema200):
        return None

    # 2. PULLBACK: Fibonacci zone OU EMA(50) touch
    pullback_type = None

    # Fibonacci check (prioridade)
    in_fib = _in_fib_zone(close, fib_0382, fib_0618, fib_dir)
    if in_fib and fib_dir == 1:
        pullback_type = "fibonacci"
    else:
        # Tolerancia: se o low tocou um nivel Fibonacci mesmo que close esteja acima
        if fib_dir == 1:
            if _price_near_fib(low, fib_0382, 0.02) or _price_near_fib(low, fib_0500, 0.02) or _price_near_fib(low, fib_0618, 0.02):
                pullback_type = "fibonacci"

    # EMA(50) touch fallback: low tocou EMA50 e close recuperou acima
    if pullback_type is None:
        ema50_touched = bool(row.get("ema50_touched", False))
        if ema50_touched and close > ema50:
            pullback_type = "ema50_touch"

    if pullback_type is None:
        return None

    # 3. RSI: Zona de pullback (30-50)
    if not (RSI_LONG_MIN <= rsi <= RSI_LONG_MAX):
        return None

    # 4. MACD: Momentum virando para cima
    if not _macd_bullish(macd_hist, macd_val, macd_sig):
        return None

    # 5. ATR: Volatilidade na faixa normal
    if not (ATR_PCT_MIN <= atr_pct <= ATR_PCT_MAX):
        return None

    # 6. BOLLINGER: Evitar squeeze extremo
    if bb_w < BB_WIDTH_MIN or bb_w > BB_WIDTH_MAX:
        return None

    # ── Gestao de risco LONG — R:R 2:1 ──
    entry = close
    stop_loss = entry - (SL_ATR_MULT * atr)
    take_profit = entry + (TP_ATR_MULT * atr)

    # Sanity: SL nao pode ser negativo
    if stop_loss <= 0:
        return None

    logger.info(
        "SINAL LONG v3 | entry=%.2f SL=%.2f TP=%.2f R:R=1:2 ATR=%.2f RSI=%.1f "
        "MACD_hist=%.4f pullback=%s",
        entry, stop_loss, take_profit, atr, rsi,
        macd_hist, pullback_type,
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
        ema50=ema50,
        ema200=ema200,
        bb_lower=bb_lower,
        bb_upper=bb_upper,
        volume=volume,
        volume_sma20=volume_sma20,
        atr_percentile=atr_pct,
        bb_width=bb_w,
        fib_0382=fib_0382 if not pd.isna(fib_0382) else 0.0,
        fib_0500=fib_0500 if not pd.isna(fib_0500) else 0.0,
        fib_0618=fib_0618 if not pd.isna(fib_0618) else 0.0,
        fib_direction=fib_dir,
        fib_proximity=fib_prox,
        pullback_type=pullback_type,
        timestamp=ts,
    )


def evaluate_short(row: pd.Series) -> Optional[Signal]:
    """
    Avalia condicoes SHORT (trend-following com pullback Fibonacci/EMA):

    Requisitos:
      1. TENDENCIA: close < EMA(50) E EMA(50) < EMA(200) (downtrend confirmado)
      2. PULLBACK: Preco na zona Fibonacci (0.382-0.618)
         OU high tocou EMA(50) e close caiu abaixo da EMA(50)
      3. RSI: 50 < RSI < 70 (zona de rally em downtrend)
      4. MACD: Histograma < 0 OU MACD < Signal (momentum virando para baixo)
      5. ATR: Percentile entre 15%-85%
      6. BB: Bandwidth > 1%
    """
    close = float(row["close"])
    low = float(row["low"])
    high = float(row["high"])
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
    atr = float(row["atr"])
    atr_pct = float(row.get("atr_percentile", 0.5))
    bb_w = float(row.get("bb_width", 0.0))
    fib_0382 = float(row.get("fib_0382", float("nan")))
    fib_0500 = float(row.get("fib_0500", float("nan")))
    fib_0618 = float(row.get("fib_0618", float("nan")))
    fib_dir = int(row.get("fib_direction", 0))
    fib_prox = float(row.get("fib_proximity", float("nan")))
    ts = row.name

    # 1. TENDENCIA: Dual EMA — downtrend confirmado
    if not (close < ema50 and ema50 < ema200):
        return None

    # 2. PULLBACK: Fibonacci zone OU EMA(50) touch
    pullback_type = None

    # Fibonacci check (prioridade)
    in_fib = _in_fib_zone(close, fib_0382, fib_0618, fib_dir)
    if in_fib and fib_dir == -1:
        pullback_type = "fibonacci"
    else:
        if fib_dir == -1:
            if _price_near_fib(high, fib_0382, 0.02) or _price_near_fib(high, fib_0500, 0.02) or _price_near_fib(high, fib_0618, 0.02):
                pullback_type = "fibonacci"

    # EMA(50) touch fallback: high tocou EMA50 e close caiu abaixo
    if pullback_type is None:
        ema50_touched_up = bool(row.get("ema50_touched_up", False))
        if ema50_touched_up and close < ema50:
            pullback_type = "ema50_touch"

    if pullback_type is None:
        return None

    # 3. RSI: Zona de rally em downtrend (50-70)
    if not (RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX):
        return None

    # 4. MACD: Momentum virando para baixo
    if not _macd_bearish(macd_hist, macd_val, macd_sig):
        return None

    # 5. ATR: Volatilidade na faixa normal
    if not (ATR_PCT_MIN <= atr_pct <= ATR_PCT_MAX):
        return None

    # 6. BOLLINGER: Evitar squeeze extremo
    if bb_w < BB_WIDTH_MIN or bb_w > BB_WIDTH_MAX:
        return None

    # ── Gestao de risco SHORT — R:R 2:1 ──
    entry = close
    stop_loss = entry + (SL_ATR_MULT * atr)
    take_profit = entry - (TP_ATR_MULT * atr)

    logger.info(
        "SINAL SHORT v3 | entry=%.2f SL=%.2f TP=%.2f R:R=1:2 ATR=%.2f RSI=%.1f "
        "MACD_hist=%.4f pullback=%s",
        entry, stop_loss, take_profit, atr, rsi,
        macd_hist, pullback_type,
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
        ema50=ema50,
        ema200=ema200,
        bb_lower=bb_lower,
        bb_upper=bb_upper,
        volume=volume,
        volume_sma20=volume_sma20,
        atr_percentile=atr_pct,
        bb_width=bb_w,
        fib_0382=fib_0382 if not pd.isna(fib_0382) else 0.0,
        fib_0500=fib_0500 if not pd.isna(fib_0500) else 0.0,
        fib_0618=fib_0618 if not pd.isna(fib_0618) else 0.0,
        fib_direction=fib_dir,
        fib_proximity=fib_prox,
        pullback_type=pullback_type,
        timestamp=ts,
    )


def evaluate_signal(df_ind: pd.DataFrame) -> Optional[Signal]:
    """
    Ponto de entrada principal da estrategia v3.
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
