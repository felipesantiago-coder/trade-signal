"""
strategy.py
-----------
Logica de validacao das condicoes de entrada CTEV v2 para LONG e SHORT.

Estrategia CTEV v2 = Confluencia de Tendencia e Exaustao Volumetrica
(now TREND-FOLLOWING with Fibonacci pullback entries).

Principais mudancas vs v1:
    - Flip de mean-reversion para trend-following (comportamento natural do BTC)
    - RSI usado como CONFIRMACAO de momentum (nao como trigger de reversao)
    - Fibonacci retracement (0.382/0.500) como zona de entrada preferencial
    - Dual EMA (50 + 200) como filtro de tendencia
    - BB(20, 2.5) mais largas para volatilidade de BTC
    - R:R minimo 1:3.0 (TP = 4.5 * ATR)
    - Volume mais flexivel (1.2x SMA20)

Referencias:
    - Quantpedia (2024): "Revisiting Trend-following and Mean-reversion in Bitcoin"
    - QuantifiedStrategies: "Bitcoin RSI Trading Strategy" (1.95 PF, 57.69% WR)
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
    """Representa um sinal de trade gerado pela estrategia CTEV v2."""
    type: SignalType
    entry_price: float
    stop_loss: float
    take_profit: float
    atr: float
    rsi: float
    rsi_delta: float
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
            "ema50": self.ema50,
            "ema200": self.ema200,
            "atr_percentile": self.atr_percentile,
            "bb_width": self.bb_width,
            "fib_0382": self.fib_0382,
            "fib_0500": self.fib_0500,
            "fib_0618": self.fib_0618,
            "fib_direction": self.fib_direction,
            "timestamp": str(self.timestamp),
        }


# ── Parametros da estrategia CTEV v2 ──
# RSI como momentum (nao mean-reversion!)
RSI_LONG_MIN = 55.0          # RSI > 55 indica momentum de alta
RSI_SHORT_MAX = 40.0         # RSI < 40 indica momentum de baixa
RSI_DELTA_MIN = 0.0          # RSI deve estar subindo (delta > 0) para LONG

# Volume confirmacao (mais flexivel que v1)
VOLUME_MULTIPLIER = 1.2       # Volume > 1.2x media 20 (mais realista para 1H)

# Bollinger Bands — usadas como referencia de volatilidade, nao como trigger
BB_WIDTH_MIN = 2.0            # BW minimo para evitar squeeze total
BB_WIDTH_MAX = 15.0           # BW maximo para evitar caos

# Fibonacci zones (pullback de alta probabilidade)
FIB_ZONE_LOW = 0.30           # Aceita entrada se preco > fib_0382 * (1 - 0.08)  (~8% tolerancia)
FIB_ZONE_HIGH = 0.70          # Aceita entrada se preco < fib_0618 * (1 + 0.08)

# Gestao de risco — R:R minimo 1:3.0
SL_ATR_MULT = 1.5             # Stop = Entry - 1.5 * ATR
TP_ATR_MULT = 4.5             # TP = Entry + 4.5 * ATR  (R:R = 3.0)

# Tolerancia para preco na zona de Fibonacci (em % do preco)
FIB_TOLERANCE_PCT = 0.012     # 1.2% de tolerancia acima/abaixo do nivel


def _price_in_fib_zone(price: float, fib_level: float) -> bool:
    """Verifica se o preco esta dentro da zona de tolerancia de um nivel Fibonacci."""
    if pd.isna(fib_level) or fib_level <= 0:
        return False
    tolerance = price * FIB_TOLERANCE_PCT
    return abs(price - fib_level) <= tolerance


def evaluate_long(row: pd.Series) -> Optional[Signal]:
    """
    Avalia condicoes LONG (trend-following com pullback Fibonacci):
    
    Requisitos:
      1. TENDENCIA: close > EMA(50) E EMA(50) > EMA(200) (uptrend confirmado)
      2. MOMENTUM: RSI(14) > 55 E RSI esta subindo (delta > 0)
      3. PULLBACK: Preco esta na zona Fibonacci (0.382 - 0.618)
         OU preco tocou BB lower (pullback tecnico)
      4. VOLUME: Volume > 1.2x SMA(20)
      5. BOLLINGER: BB nao em squeeze extremo (BW > 2%)
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
    volume = float(row["volume"])
    volume_sma20 = float(row["volume_sma20"])
    atr = float(row["atr"])
    atr_pct = float(row.get("atr_percentile", 0.5))
    bb_w = float(row.get("bb_width", 0.0))
    fib_0382 = float(row.get("fib_0382", float("nan")))
    fib_0500 = float(row.get("fib_0500", float("nan")))
    fib_0618 = float(row.get("fib_0618", float("nan")))
    fib_dir = int(row.get("fib_direction", 0))
    ts = row.name

    # 1. TENDENCIA: Dual EMA — uptrend confirmado
    if not (close > ema50 and ema50 > ema200):
        return None

    # 2. MOMENTUM: RSI > 55 e subindo (confirma que o pullback esta acabando)
    if not (rsi > RSI_LONG_MIN and rsi_delta > RSI_DELTA_MIN):
        return None

    # 3. PULLBACK: Preco na zona Fibonacci OU tocou BB lower
    in_fib_zone = False
    # Fibonacci direcao deve ser uptrend (1)
    if fib_dir == 1:
        # Zona Fibonacci 0.382 - 0.618: preco deve estar entre fib_0618 e fib_0382
        # (em uptrend, fib_0618 e mais baixo que fib_0382)
        if not pd.isna(fib_0382) and not pd.isna(fib_0618):
            if close >= fib_0618 and close <= fib_0382:
                in_fib_zone = True
        # Tolerancia: se o low tocou o nivel Fibonacci mesmo que close esteja acima
        if not in_fib_zone:
            if not pd.isna(fib_0382) and _price_in_fib_zone(close, fib_0382):
                in_fib_zone = True
            elif not pd.isna(fib_0500) and _price_in_fib_zone(close, fib_0500):
                in_fib_zone = True
    
    # Fallback BB: se nao tiver Fibonacci, aceita pullback que tocou BB lower
    if not in_fib_zone:
        if not (low <= bb_lower or close <= bb_lower):
            return None

    # 4. VOLUME: Confirmacao de interesse institucional
    if not (volume_sma20 > 0 and volume > volume_sma20 * VOLUME_MULTIPLIER):
        return None

    # 5. BOLLINGER: Evitar squeeze extremo
    if bb_w < BB_WIDTH_MIN or bb_w > BB_WIDTH_MAX:
        return None

    # Gestao de risco LONG — R:R 1:3.0
    entry = close
    stop_loss = entry - (SL_ATR_MULT * atr)
    take_profit = entry + (TP_ATR_MULT * atr)

    logger.info(
        "SINAL LONG v2 | entry=%.2f SL=%.2f TP=%.2f R:R=1:3 ATR=%.2f RSI=%.1f "
        "fib_zone=%s",
        entry, stop_loss, take_profit, atr, rsi,
        in_fib_zone,
    )

    return Signal(
        type=SignalType.LONG,
        entry_price=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        atr=atr,
        rsi=rsi,
        rsi_delta=rsi_delta,
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
        timestamp=ts,
    )


def evaluate_short(row: pd.Series) -> Optional[Signal]:
    """
    Avalia condicoes SHORT (trend-following com pullback Fibonacci):
    
    Requisitos:
      1. TENDENCIA: close < EMA(50) E EMA(50) < EMA(200) (downtrend confirmado)
      2. MOMENTUM: RSI(14) < 40 E RSI esta descendo (delta < 0)
      3. PULLBACK: Preco esta na zona Fibonacci (0.382 - 0.618)
         OU preco tocou BB upper (pullback tecnico)
      4. VOLUME: Volume > 1.2x SMA(20)
      5. BOLLINGER: BB nao em squeeze extremo
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
    volume = float(row["volume"])
    volume_sma20 = float(row["volume_sma20"])
    atr = float(row["atr"])
    atr_pct = float(row.get("atr_percentile", 0.5))
    bb_w = float(row.get("bb_width", 0.0))
    fib_0382 = float(row.get("fib_0382", float("nan")))
    fib_0500 = float(row.get("fib_0500", float("nan")))
    fib_0618 = float(row.get("fib_0618", float("nan")))
    fib_dir = int(row.get("fib_direction", 0))
    ts = row.name

    # 1. TENDENCIA: Dual EMA — downtrend confirmado
    if not (close < ema50 and ema50 < ema200):
        return None

    # 2. MOMENTUM: RSI < 40 e descendo
    if not (rsi < RSI_SHORT_MAX and rsi_delta < -RSI_DELTA_MIN):
        return None

    # 3. PULLBACK: Zona Fibonacci OU BB upper
    in_fib_zone = False
    if fib_dir == -1:
        # Downtrend: fib_0382 e mais alto que fib_0618
        if not pd.isna(fib_0382) and not pd.isna(fib_0618):
            if close <= fib_0618 and close >= fib_0382:
                in_fib_zone = True
        if not in_fib_zone:
            if not pd.isna(fib_0382) and _price_in_fib_zone(close, fib_0382):
                in_fib_zone = True
            elif not pd.isna(fib_0500) and _price_in_fib_zone(close, fib_0500):
                in_fib_zone = True

    if not in_fib_zone:
        if not (high >= bb_upper or close >= bb_upper):
            return None

    # 4. VOLUME
    if not (volume_sma20 > 0 and volume > volume_sma20 * VOLUME_MULTIPLIER):
        return None

    # 5. BOLLINGER
    if bb_w < BB_WIDTH_MIN or bb_w > BB_WIDTH_MAX:
        return None

    # Gestao de risco SHORT — R:R 1:3.0
    entry = close
    stop_loss = entry + (SL_ATR_MULT * atr)
    take_profit = entry - (TP_ATR_MULT * atr)

    logger.info(
        "SINAL SHORT v2 | entry=%.2f SL=%.2f TP=%.2f R:R=1:3 ATR=%.2f RSI=%.1f "
        "fib_zone=%s",
        entry, stop_loss, take_profit, atr, rsi,
        in_fib_zone,
    )

    return Signal(
        type=SignalType.SHORT,
        entry_price=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        atr=atr,
        rsi=rsi,
        rsi_delta=rsi_delta,
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
        timestamp=ts,
    )


def evaluate_signal(df_ind: pd.DataFrame) -> Optional[Signal]:
    """
    Ponto de entrada principal da estrategia v2.
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
