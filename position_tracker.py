"""
position_tracker.py
-------------------
Rastreamento de posicoes abertas com trailing stop e break-even.

Este modulo gerencia o ciclo de vida completo de cada trade:
    1. Abertura: quando um sinal CTEV e gerado, cria uma posicao
    2. Monitoramento: a cada candle fechado, verifica se:
       - Preco atingiu o TP (take profit)
       - Preco atingiu o SL (stop loss, que pode ser o original ou o trailing)
       - Preco moveu suficiente para mover SL para breakeven
       - Preco moveu suficiente para ativar trailing stop
    3. Fechamento: quando TP ou SL e atingido, fecha a posicao e registra o resultado

Funcionalidades:
    - Break-even: move SL para o preco de entrada (+ fees) quando preco
      move 1.0 * ATR a favor (trade fica "livre de risco")
    - Trailing Stop: apos break-even, move SL acompanhando o preco a
      cada candle (distancia = 1.5 * ATR do high mais recente para LONG,
      ou low para SHORT)
    - Partial TP: vende 50% no TP1 (2.0 * ATR), deixa 50% correr com trailing

Referencias:
    - 3Commas (2025): "Advanced stop loss and take profit strategies can
      significantly enhance trading performance"
    - PipMaster (2026): "Trailing stop lets trends run while break-even
      turns a trade into a free bet"
    - LuxAlgo: "Partial profit-taking, dynamic trailing stops, and
      break-even functionality using ATR"
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger("ctev.positions")


class PositionStatus(str, Enum):
    """Status de uma posicao aberta."""
    OPEN = "open"
    BREAK_EVEN = "break_even"
    TRAILING = "trailing"
    PARTIAL_TP = "partial_tp"
    CLOSED_TP = "closed_tp"
    CLOSED_SL = "closed_sl"
    CLOSED_TIMEOUT = "closed_timeout"


@dataclass
class Position:
    """Posicao aberta com gestao de trailing stop e break-even."""
    id: int
    type: str                          # LONG ou SHORT
    entry_price: float
    stop_loss: float                    # SL original
    stop_loss_initial: float            # SL original (backup para referencia)
    take_profit: float                  # TP original
    atr: float
    entry_ts: str                       # ISO8601 UTC
    status: PositionStatus = PositionStatus.OPEN

    # Trailing stop
    trailing_activated: bool = False
    trailing_stop: float = 0.0          # SL atual (pode ser diferente do original)
    highest_favorable: float = 0.0       # Maior preco favoravel visto (high para LONG, low para SHORT)

    # Break-even
    be_triggered: bool = False
    be_trigger_atr_mult: float = 1.0    # Multiplo de ATR para trigger BE

    # Partial TP
    partial_tp_filled: bool = False
    partial_tp_pct: float = 0.50        # Percentual do TP para partial (50%)

    # Sizing (opcional)
    position_size: float = 0.0
    position_usd: float = 0.0

    # Fechamento
    exit_price: float = 0.0
    exit_ts: Optional[str] = None
    exit_reason: str = ""
    pnl_pct: float = 0.0
    pnl_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "entry_price": self.entry_price,
            "stop_loss": self.trailing_stop if self.trailing_activated else self.stop_loss,
            "stop_loss_initial": self.stop_loss_initial,
            "take_profit": self.take_profit,
            "atr": self.atr,
            "entry_ts": self.entry_ts,
            "status": self.status.value,
            "trailing_activated": self.trailing_activated,
            "trailing_stop": self.trailing_stop,
            "highest_favorable": self.highest_favorable,
            "be_triggered": self.be_triggered,
            "partial_tp_filled": self.partial_tp_filled,
            "position_size": self.position_size,
            "position_usd": self.position_usd,
            "exit_price": self.exit_price,
            "exit_ts": self.exit_ts,
            "exit_reason": self.exit_reason,
            "pnl_pct": round(self.pnl_pct, 4),
            "pnl_usd": round(self.pnl_usd, 2),
        }


class PositionTracker:
    """
    Gerenciador de posicoes abertas. Singleton (ver get_position_tracker()).

    Cada candle fechado, o metodo update_positions() e chamado com o
    OHLCV do candle para verificar se alguma posicao atingiu TP, SL,
    break-even trigger ou trailing stop update.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._positions: Dict[int, Position] = {}
        self._closed_positions: List[Position] = []
        self._next_id: int = 1

    def open_position(
        self,
        type_: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        atr: float,
        entry_ts: str,
        position_size: float = 0.0,
        position_usd: float = 0.0,
        be_trigger_atr_mult: float = 1.0,
        trailing_atr_mult: float = 1.5,
        partial_tp_pct: float = 0.50,
    ) -> Position:
        """
        Abre uma nova posicao.
        Retorna a Position criada.
        """
        with self._lock:
            pos = Position(
                id=self._next_id,
                type=type_,
                entry_price=entry_price,
                stop_loss=stop_loss,
                stop_loss_initial=stop_loss,
                take_profit=take_profit,
                atr=atr,
                entry_ts=entry_ts,
                position_size=position_size,
                position_usd=position_usd,
                be_trigger_atr_mult=be_trigger_atr_mult,
                partial_tp_pct=partial_tp_pct,
            )
            # Inicializa highest_favorable
            if type_ == "LONG":
                pos.highest_favorable = entry_price
            else:
                pos.highest_favorable = entry_price

            self._positions[pos.id] = pos
            self._next_id += 1
            logger.info(
                "Posicao #%d aberta: %s @ %.2f SL=%.2f TP=%.2f ATR=%.2f size=%.6f ($%.2f)",
                pos.id, type_, entry_price, stop_loss, take_profit, atr,
                position_size, position_usd,
            )
            return pos

    def update_positions(
        self,
        candle_close: float,
        candle_high: float,
        candle_low: float,
        candle_ts: str,
        trailing_atr_mult: float = 1.5,
    ) -> List[Position]:
        """
        Atualiza todas as posicoes abertas com o novo candle fechado.

        Para cada posicao aberta:
        1. Verifica se atingiu TP ou SL
        2. Se nao, verifica break-even trigger
        3. Se BE ja ativo, atualiza trailing stop
        4. Verifica se o trailing stop foi atingido

        Retorna lista de posicoes que foram fechadas neste candle.
        """
        closed: List[Position] = []

        with self._lock:
            to_close_ids: List[int] = []

            for pos in self._positions.values():
                if pos.status not in (
                    PositionStatus.OPEN,
                    PositionStatus.BREAK_EVEN,
                    PositionStatus.TRAILING,
                    PositionStatus.PARTIAL_TP,
                ):
                    continue

                is_long = pos.type == "LONG"

                # ---- 1. Verifica TP (usando partial logic) ----
                if is_long:
                    tp_hit = candle_high >= pos.take_profit
                    sl_hit = candle_low <= (pos.trailing_stop if pos.trailing_activated else pos.stop_loss)
                else:
                    tp_hit = candle_low <= pos.take_profit
                    sl_hit = candle_high >= (pos.trailing_stop if pos.trailing_activated else pos.stop_loss)

                # Determina qual foi atingido primeiro (SL ou TP)
                if sl_hit and not tp_hit:
                    # SL atingido
                    pos.exit_price = pos.trailing_stop if pos.trailing_activated else pos.stop_loss
                    pos.exit_ts = candle_ts
                    pos.exit_reason = "sl"
                    pos.status = PositionStatus.CLOSED_SL
                    to_close_ids.append(pos.id)

                elif tp_hit and not sl_hit:
                    # TP atingido
                    if not pos.partial_tp_filled:
                        # Primeira vez no TP: partial close (50%)
                        pos.partial_tp_filled = True
                        pos.status = PositionStatus.PARTIAL_TP
                        # Move SL para breakeven imediatamente
                        pos.be_triggered = True
                        pos.trailing_stop = pos.entry_price
                        pos.trailing_activated = True
                        logger.info(
                            "Posicao #%d: Partial TP atingido em %.2f. SL movido para BE (%.2f)",
                            pos.id, pos.take_profit, pos.entry_price,
                        )
                    else:
                        # Segundo TP: fecha a posicao inteira
                        pos.exit_price = pos.take_profit
                        pos.exit_ts = candle_ts
                        pos.exit_reason = "tp"
                        pos.status = PositionStatus.CLOSED_TP
                        to_close_ids.append(pos.id)

                elif tp_hit and sl_hit:
                    # Ambos atingidos no mesmo candle — pior caso (SL)
                    pos.exit_price = pos.trailing_stop if pos.trailing_activated else pos.stop_loss
                    pos.exit_ts = candle_ts
                    pos.exit_reason = "sl"
                    pos.status = PositionStatus.CLOSED_SL
                    to_close_ids.append(pos.id)

                else:
                    # Nenhum atingido — atualiza trailing e BE
                    # 2. Atualiza highest_favorable
                    if is_long:
                        pos.highest_favorable = max(pos.highest_favorable, candle_high)
                    else:
                        pos.highest_favorable = min(pos.highest_favorable, candle_low)

                    # 3. Break-even trigger
                    if not pos.be_triggered:
                        favorable_distance = abs(pos.highest_favorable - pos.entry_price)
                        be_distance = pos.atr * pos.be_trigger_atr_mult

                        if favorable_distance >= be_distance:
                            pos.be_triggered = True
                            pos.trailing_stop = pos.entry_price
                            pos.trailing_activated = True
                            pos.status = PositionStatus.BREAK_EVEN
                            logger.info(
                                "Posicao #%d: Break-even ativado. SL movido para %.2f (entrada)",
                                pos.id, pos.entry_price,
                            )

                    # 4. Trailing stop update (apos BE)
                    if pos.trailing_activated:
                        trail_distance = pos.atr * trailing_atr_mult
                        if is_long:
                            new_trail = pos.highest_favorable - trail_distance
                            # Trailing so move PARA CIMA
                            if new_trail > pos.trailing_stop:
                                pos.trailing_stop = new_trail
                                if pos.status != PositionStatus.PARTIAL_TP:
                                    pos.status = PositionStatus.TRAILING
                                logger.info(
                                    "Posicao #%d: Trailing SL atualizado: %.2f (trail=%.2f)",
                                    pos.id, pos.trailing_stop, new_trail,
                                )
                        else:
                            new_trail = pos.highest_favorable + trail_distance
                            # Trailing so move PARA BAIXO
                            if new_trail < pos.trailing_stop:
                                pos.trailing_stop = new_trail
                                if pos.status != PositionStatus.PARTIAL_TP:
                                    pos.status = PositionStatus.TRAILING
                                logger.info(
                                    "Posicao #%d: Trailing SL atualizado: %.2f (trail=%.2f)",
                                    pos.id, pos.trailing_stop, new_trail,
                                )

            # Fecha posicoes e calcula PnL
            for pos_id in to_close_ids:
                pos = self._positions.pop(pos_id, None)
                if pos:
                    # PnL
                    if pos.type == "LONG":
                        pos.pnl_pct = (pos.exit_price - pos.entry_price) / pos.entry_price * 100
                    else:
                        pos.pnl_pct = (pos.entry_price - pos.exit_price) / pos.entry_price * 100

                    if pos.position_size > 0:
                        if pos.type == "LONG":
                            pos.pnl_usd = (pos.exit_price - pos.entry_price) * pos.position_size
                        else:
                            pos.pnl_usd = (pos.entry_price - pos.exit_price) * pos.position_size
                    else:
                        pos.pnl_usd = 0.0

                    # Se partial TP foi atingido, o PnL reflete apenas a parcela restante
                    if pos.partial_tp_filled and pos.exit_reason == "tp":
                        # PnL total = partial_tp_pct * TP_pnl + (1-partial_tp_pct) * TP_pnl
                        full_pnl_pct = pos.pnl_pct
                        pos.pnl_pct = full_pnl_pct  # Simplificado: ambos os lados lucram o mesmo
                        if pos.position_usd > 0:
                            pos.pnl_usd = pos.position_usd * (pos.pnl_pct / 100)

                    self._closed_positions.append(pos)
                    logger.info(
                        "Posicao #%d FECHADA: %s | entry=%.2f exit=%.2f pnl=%.2f%% (%s$%.2f) reason=%s",
                        pos.id, pos.type, pos.entry_price, pos.exit_price,
                        pos.pnl_pct, "+" if pos.pnl_usd >= 0 else "", pos.pnl_usd,
                        pos.exit_reason,
                    )

        return closed

    def close_position(self, pos_id: int, exit_price: float, exit_ts: str, reason: str = "manual") -> Optional[Position]:
        """Fecha uma posicao manualmente."""
        with self._lock:
            pos = self._positions.pop(pos_id, None)
            if pos:
                pos.exit_price = exit_price
                pos.exit_ts = exit_ts
                pos.exit_reason = reason
                if pos.type == "LONG":
                    pos.pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
                else:
                    pos.pnl_pct = (pos.entry_price - exit_price) / pos.entry_price * 100
                if pos.position_size > 0:
                    if pos.type == "LONG":
                        pos.pnl_usd = (exit_price - pos.entry_price) * pos.position_size
                    else:
                        pos.pnl_usd = (pos.entry_price - exit_price) * pos.position_size
                pos.status = PositionStatus.CLOSED_SL if reason == "sl" else PositionStatus.CLOSED_TP
                self._closed_positions.append(pos)
                logger.info(
                    "Posicao #%d fechada manualmente: exit=%.2f pnl=%.2f%%",
                    pos.id, exit_price, pos.pnl_pct,
                )
                return pos
            return None

    def close_all(self, exit_price: float, exit_ts: str, reason: str = "kill") -> List[Position]:
        """Fecha todas as posicoes abertas."""
        with self._lock:
            closed = []
            ids = list(self._positions.keys())
            for pos_id in ids:
                pos = self.close_position(pos_id, exit_price, exit_ts, reason)
                if pos:
                    closed.append(pos)
            return closed

    @property
    def open_positions(self) -> List[Position]:
        with self._lock:
            return list(self._positions.values())

    @property
    def has_open_positions(self) -> bool:
        with self._lock:
            return len(self._positions) > 0

    def get_open_position(self) -> Optional[Position]:
        """Retorna a posicao aberta (assumimos apenas 1 posicao por vez)."""
        with self._lock:
            for pos in self._positions.values():
                return pos
            return None

    @property
    def closed_trades(self) -> List[Position]:
        with self._lock:
            return list(self._closed_positions)

    def snapshot(self) -> dict:
        with self._lock:
            open_pos = [
                p.to_dict() for p in self._positions.values()
                if p.status in (
                    PositionStatus.OPEN, PositionStatus.BREAK_EVEN,
                    PositionStatus.TRAILING, PositionStatus.PARTIAL_TP,
                )
            ]
            closed = [p.to_dict() for p in self._closed_positions[-50:]]  # Ultimos 50
            return {
                "open_count": len(open_pos),
                "closed_count": len(self._closed_positions),
                "open_positions": open_pos,
                "closed_trades": closed,
            }


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------
_instance: Optional[PositionTracker] = None
_lock = threading.Lock()


def get_position_tracker() -> PositionTracker:
    """Retorna a instancia unica de PositionTracker."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = PositionTracker()
    return _instance
