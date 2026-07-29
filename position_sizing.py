"""
position_sizing.py
------------------
Calculo de tamanho de posicao baseado em risco por trade e ATR.

O position sizing padroniza o risco de cada operacao, impedindo
overexposure em qualquer trade individual. Baseado nos principios
do Kelly Criterion (versao conservadora Quarter Kelly) e risk-per-trade.

Formula principal:
    pos_size = (account_balance * risk_pct) / sl_distance
    Onde sl_distance = |entry - stop_loss| = ATR * SL_ATR_MULT

Referencias:
    - QuantVPS (2026): "Position Sizing: Limit risk per trade to 1-2% of your account"
    - Reddit r/algotrading: "Full Kelly is way too aggressive for live trading.
      Most people use quarter or half Kelly"
    - Tradefundrr (2026): "Kelly Criterion: formula that estimates the fraction
      of capital to risk on any given trade based on your edge"
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("ctev.sizing")


@dataclass(frozen=True)
class PositionSize:
    """Resultado do calculo de tamanho de posicao."""
    position_size: float       # Quantidade em unidades de base (ex: BTC)
    position_usd: float        # Valor em USD
    risk_usd: float            # Risco em USD (perda se SL for atingido)
    risk_pct: float            # Percentual do balance arriscado
    sl_distance: float         # Distancia do SL em preco absoluto
    sl_distance_pct: float     # Distancia do SL em percentual
    balance: float             # Balance usado no calculo
    leverage: float            # Alavancagem efetiva (pos_size * price / balance)


class PositionSizer:
    """
    Calculadora de tamanho de posicao. Singleton (ver get_position_sizer()).

    O calculo e feito por:
        1. Distancia do SL (ATR * SL_ATR_MULT)
        2. Risco maximo por trade (ex: 1% do balance)
        3. Tamanho = risk_usd / sl_distance_pct

    Parametros configuraveis:
        - balance: saldo da conta em USD
        - risk_per_trade_pct: % do balance a arriscar por trade (default 1%)
        - min_position_usd: tamanho minimo da posicao em USD (default $10)
        - max_position_pct: tamanho maximo como % do balance (default 10%)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._balance: float = 10000.0  # Default $10,000
        self._risk_per_trade_pct: float = 0.01  # 1%
        self._min_position_usd: float = 10.0
        self._max_position_pct: float = 0.10  # 10%

    def configure(self, kwargs: dict) -> None:
        """Atualiza parametros de sizing."""
        with self._lock:
            if "balance" in kwargs:
                self._balance = max(float(kwargs["balance"]), 0.0)
            if "risk_per_trade_pct" in kwargs:
                self._risk_per_trade_pct = min(
                    max(float(kwargs["risk_per_trade_pct"]), 0.001),
                    0.05,  # Cap at 5% max
                )
            if "min_position_usd" in kwargs:
                self._min_position_usd = max(float(kwargs["min_position_usd"]), 1.0)
            if "max_position_pct" in kwargs:
                self._max_position_pct = min(
                    max(float(kwargs["max_position_pct"]), 0.01),
                    0.50,  # Cap at 50% max
                )
        logger.info(
            "PositionSizer configurado: balance=$%.2f risk=%.2f%% min=$%.2f max=%.1f%%",
            self._balance, self._risk_per_trade_pct * 100,
            self._min_position_usd, self._max_position_pct * 100,
        )

    def calculate(
        self,
        entry_price: float,
        stop_loss: float,
    ) -> PositionSize:
        """
        Calcula o tamanho da posicao para um trade.

        Parameters:
            entry_price: Preco de entrada
            stop_loss: Preco do stop loss

        Returns:
            PositionSize com todos os dados da posicao
        """
        with self._lock:
            balance = self._balance
            risk_pct = self._risk_per_trade_pct

            # Distancia do SL em valor absoluto e percentual
            sl_distance = abs(entry_price - stop_loss)
            sl_distance_pct = sl_distance / entry_price if entry_price > 0 else 0

            # Risco maximo em USD
            risk_usd = balance * risk_pct

            # Tamanho da posicao em unidades de base
            if sl_distance_pct > 0:
                position_size = risk_usd / (sl_distance_pct * entry_price)
            else:
                position_size = 0.0

            # Valor da posicao em USD
            position_usd = position_size * entry_price

            # Aplica limites
            max_usd = balance * self._max_position_pct
            if position_usd > max_usd:
                position_usd = max_usd
                position_size = max_usd / entry_price
            if position_usd < self._min_position_usd:
                position_usd = 0.0
                position_size = 0.0

            # Alavancagem efetiva
            leverage = position_usd / balance if balance > 0 else 0.0

            return PositionSize(
                position_size=round(position_size, 8),
                position_usd=round(position_usd, 2),
                risk_usd=round(risk_usd, 2),
                risk_pct=risk_pct,
                sl_distance=round(sl_distance, 2),
                sl_distance_pct=round(sl_distance_pct, 6),
                balance=balance,
                leverage=round(leverage, 4),
            )

    def update_balance(self, pnl_usd: float) -> float:
        """
        Atualiza o balance apos um trade fechado.
        Retorna o novo balance.
        """
        with self._lock:
            self._balance = max(0.0, self._balance + pnl_usd)
            logger.info(
                "Balance atualizado: $%.2f (PnL: %s$%.2f)",
                self._balance, "+" if pnl_usd >= 0 else "", pnl_usd,
            )
            return self._balance

    @property
    def balance(self) -> float:
        with self._lock:
            return self._balance

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "balance": round(self._balance, 2),
                "risk_per_trade_pct": self._risk_per_trade_pct,
                "min_position_usd": self._min_position_usd,
                "max_position_pct": self._max_position_pct,
            }


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------
_instance: Optional[PositionSizer] = None
_lock = threading.Lock()


def get_position_sizer() -> PositionSizer:
    """Retorna a instancia unica de PositionSizer."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = PositionSizer()
    return _instance
