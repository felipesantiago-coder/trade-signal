"""
strategy.py
-----------
Lógica de validação das condições de entrada CTEV para LONG e SHORT,
incluindo cálculo de Stop Loss e Take Profit baseados em ATR(14).

Estratégia CTEV = Confluência de Tendência e Exaustão Volumétrica.
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
    """Representa um sinal de trade gerado pela estratégia CTEV."""
    type: SignalType
    entry_price: float
    stop_loss: float
    take_profit: float
    atr: float
    rsi: float
    ema200: float
    bb_lower: float
    bb_upper: float
    volume: float
    volume_sma20: float
    timestamp: pd.Timestamp

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "atr": self.atr,
            "rsi": self.rsi,
            "ema200": self.ema200,
            "timestamp": str(self.timestamp),
        }


# ---- Parâmetros da estratégia (podem ser ajustados via env se necessário) ----
RSI_LONG_THRESHOLD = 35.0      # RSI < 35 para exaustão baixa
RSI_SHORT_THRESHOLD = 65.0     # RSI > 65 para exaustão alta
VOLUME_MULTIPLIER = 1.5        # Volume atual > 1.5x média 20
SL_ATR_MULT = 1.5              # Stop = Entry ± 1.5 * ATR
TP_ATR_MULT = 2.0              # TP1  = Entry ∓ 2.0 * ATR


def evaluate_long(row: pd.Series) -> Optional[Signal]:
    """
    Avalia condições LONG no candle fechado:
      1. close > ema200
      2. low OU close <= bb_lower
      3. rsi < 35
      4. volume > 1.5 * volume_sma20
    Retorna Signal se todas as condições forem satisfeitas, senão None.
    """
    close = float(row["close"])
    low = float(row["low"])
    high = float(row["high"])
    ema200 = float(row["ema200"])
    bb_lower = float(row["bb_lower"])
    rsi = float(row["rsi"])
    volume = float(row["volume"])
    volume_sma20 = float(row["volume_sma20"])
    atr = float(row["atr"])
    ts = row.name

    # 1. Tendência de alta macro
    if not close > ema200:
        return None

    # 2. Pullback até a banda inferior (low ou close tocando a banda)
    if not (low <= bb_lower or close <= bb_lower):
        return None

    # 3. Exaustão baixa no RSI
    if not rsi < RSI_LONG_THRESHOLD:
        return None

    # 4. Confirmação institucional via volume
    if not (volume_sma20 > 0 and volume > volume_sma20 * VOLUME_MULTIPLIER):
        return None

    # Gestão de risco LONG
    entry = close
    stop_loss = entry - (SL_ATR_MULT * atr)
    take_profit = entry + (TP_ATR_MULT * atr)

    logger.info(
        "SINAL LONG detectado | entry=%.2f SL=%.2f TP=%.2f ATR=%.2f RSI=%.2f",
        entry, stop_loss, take_profit, atr, rsi,
    )

    return Signal(
        type=SignalType.LONG,
        entry_price=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        atr=atr,
        rsi=rsi,
        ema200=ema200,
        bb_lower=bb_lower,
        bb_upper=float(row["bb_upper"]),
        volume=volume,
        volume_sma20=volume_sma20,
        timestamp=ts,
    )


def evaluate_short(row: pd.Series) -> Optional[Signal]:
    """
    Avalia condições SHORT no candle fechado:
      1. close < ema200
      2. high OU close >= bb_upper
      3. rsi > 65
      4. volume > 1.5 * volume_sma20
    Retorna Signal se todas as condições forem satisfeitas, senão None.
    """
    close = float(row["close"])
    high = float(row["high"])
    ema200 = float(row["ema200"])
    bb_upper = float(row["bb_upper"])
    rsi = float(row["rsi"])
    volume = float(row["volume"])
    volume_sma20 = float(row["volume_sma20"])
    atr = float(row["atr"])
    ts = row.name

    # 1. Tendência de baixa macro
    if not close < ema200:
        return None

    # 2. Pullback para cima até a banda superior
    if not (high >= bb_upper or close >= bb_upper):
        return None

    # 3. Exaustão alta no RSI
    if not rsi > RSI_SHORT_THRESHOLD:
        return None

    # 4. Confirmação institucional via volume
    if not (volume_sma20 > 0 and volume > volume_sma20 * VOLUME_MULTIPLIER):
        return None

    # Gestão de risco SHORT
    entry = close
    stop_loss = entry + (SL_ATR_MULT * atr)
    take_profit = entry - (TP_ATR_MULT * atr)

    logger.info(
        "SINAL SHORT detectado | entry=%.2f SL=%.2f TP=%.2f ATR=%.2f RSI=%.2f",
        entry, stop_loss, take_profit, atr, rsi,
    )

    return Signal(
        type=SignalType.SHORT,
        entry_price=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        atr=atr,
        rsi=rsi,
        ema200=ema200,
        bb_lower=float(row["bb_lower"]),
        bb_upper=bb_upper,
        volume=volume,
        volume_sma20=volume_sma20,
        timestamp=ts,
    )


def evaluate_signal(df_ind: pd.DataFrame) -> Optional[Signal]:
    """
    Ponto de entrada principal da estratégia.
    Recebe o DataFrame já com indicadores calculados e avalia apenas a última
    linha (candle mais recente fechado).
    Retorna um Signal LONG, SHORT ou None.
    """
    if df_ind.empty:
        return None

    last = df_ind.iloc[-1]
    # Tenta LONG primeiro; se não bater, tenta SHORT.
    signal = evaluate_long(last)
    if signal is not None:
        return signal
    return evaluate_short(last)
