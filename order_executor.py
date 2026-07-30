"""
order_executor.py
-----------------
Execucao de ordens reais na Binance via ccxt (spot/futures).

Funcionalidades:
    - Market orders (compra/venda)
    - Limit orders (entrada com preco especifico)
    - Cancelamento de ordens abertas
    - Consulta de saldo e posicoes
    - Modo DRY-RUN (simulacao sem execucao real)
    - Rastreamento de ordens (order_id, status, timestamp)

O executor opera em dois modos:
    1. DRY_RUN=True:  simula ordens sem enviar na exchange (default seguro)
    2. DRY_RUN=False: envia ordens reais na Binance

Referencias:
    - ccxt docs: "Unified API for 100+ cryptocurrency exchanges"
    - Binance API: "Place and manage orders on the spot market"
    - QuantVPS (2026): "Always test order execution in paper trading
      before going live with real capital"
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ctev.executor")


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELED = "canceled"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class Order:
    """Ordem enviada ou simulada."""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    amount: float
    price: Optional[float]
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_amount: float = 0.0
    filled_price: float = 0.0
    fee: float = 0.0
    cost: float = 0.0
    created_at: str = ""
    updated_at: str = ""
    exchange_order_id: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "type": self.order_type.value,
            "amount": round(self.amount, 8),
            "price": self.price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "status": self.status.value,
            "filled_amount": round(self.filled_amount, 8),
            "filled_price": round(self.filled_price, 2),
            "fee": round(self.fee, 4),
            "cost": round(self.cost, 2),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "exchange_order_id": self.exchange_order_id,
            "error": self.error,
        }


@dataclass
class BalanceInfo:
    """Informacoes de saldo da conta."""
    total: float = 0.0
    free: float = 0.0
    used: float = 0.0
    currency: str = "USDT"

    def to_dict(self) -> dict:
        return {
            "total": round(self.total, 2),
            "free": round(self.free, 2),
            "used": round(self.used, 2),
            "currency": self.currency,
        }


class OrderExecutor:
    """
    Executor de ordens na Binance. Singleton (ver get_order_executor()).

    Thread-safe via Lock. Operacao em dois modos:
    - DRY_RUN: simula sem enviar na exchange
    - LIVE: envia ordens reais
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._dry_run: bool = True
        self._exchange = None
        self._orders: Dict[str, Order] = {}
        self._next_order_id: int = 1
        self._last_error: Optional[str] = None

    def configure(self, kwargs: dict) -> None:
        """Configura o executor."""
        with self._lock:
            if "dry_run" in kwargs:
                self._dry_run = bool(kwargs["dry_run"])
            if "exchange" in kwargs:
                self._exchange = kwargs["exchange"]
        logger.info(
            "OrderExecutor configurado: dry_run=%s exchange=%s",
            self._dry_run, "connected" if self._exchange else "none",
        )

    @property
    def dry_run(self) -> bool:
        with self._lock:
            return self._dry_run

    @dry_run.setter
    def dry_run(self, value: bool) -> None:
        with self._lock:
            self._dry_run = value
        logger.info("OrderExecutor modo: %s", "DRY-RUN" if value else "LIVE")

    # ------------------------------------------------------------------
    # Market Orders
    # ------------------------------------------------------------------
    def execute_market(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        price: Optional[float] = None,
    ) -> Order:
        """
        Executa uma ordem a mercado.

        Parameters:
            symbol: par de trading (ex: BTC/USDT)
            side: BUY ou SELL
            amount: quantidade em unidades base
            price: preco estimado (usado em dry-run para PnL)

        Returns:
            Order com status atualizado
        """
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        order_id = f"ord-{self._next_order_id:06d}"

        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            amount=amount,
            price=price,
            status=OrderStatus.PENDING,
            created_at=now,
        )

        if self._dry_run:
            return self._simulate_fill(order, price or 0.0, now)

        return self._execute_real_market(order)

    # ------------------------------------------------------------------
    # Limit Orders
    # ------------------------------------------------------------------
    def execute_limit(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        price: float,
    ) -> Order:
        """
        Cria uma ordem limite.

        Parameters:
            symbol: par de trading
            side: BUY ou SELL
            amount: quantidade
            price: preco limite

        Returns:
            Order criada
        """
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        order_id = f"ord-{self._next_order_id:06d}"

        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            amount=amount,
            price=price,
            status=OrderStatus.OPEN,
            created_at=now,
        )

        if self._dry_run:
            logger.info(
                "[DRY-RUN] Ordem limite criada: %s %s %.8f @ %.2f",
                side.value, symbol, amount, price,
            )
            self._orders[order_id] = order
            self._next_order_id += 1
            return order

        return self._execute_real_limit(order)

    # ------------------------------------------------------------------
    # Cancel Orders
    # ------------------------------------------------------------------
    def cancel_order(self, order_id: str) -> Optional[Order]:
        """Cancela uma ordem aberta."""
        with self._lock:
            order = self._orders.get(order_id)
            if order is None:
                logger.warning("Ordem %s nao encontrada para cancelamento.", order_id)
                return None

            if self._dry_run:
                order.status = OrderStatus.CANCELED
                order.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                logger.info("[DRY-RUN] Ordem %s cancelada.", order_id)
                return order

            return self._cancel_real_order(order)

    # ------------------------------------------------------------------
    # Balance
    # ------------------------------------------------------------------
    def get_balance(self, currency: str = "USDT") -> BalanceInfo:
        """Retorna o saldo disponivel na exchange."""
        if self._dry_run:
            return BalanceInfo(
                total=10000.0, free=10000.0, used=0.0, currency=currency,
            )

        try:
            balance = self._exchange.fetch_balance()
            if currency in balance:
                b = balance[currency]
                return BalanceInfo(
                    total=float(b.get("total", 0)),
                    free=float(b.get("free", 0)),
                    used=float(b.get("used", 0)),
                    currency=currency,
                )
        except Exception as exc:
            logger.error("Erro ao buscar saldo: %s", exc)
            self._last_error = str(exc)

        return BalanceInfo(currency=currency)

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------
    @property
    def orders(self) -> List[dict]:
        with self._lock:
            return [o.to_dict() for o in self._orders.values()]

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "dry_run": self._dry_run,
                "total_orders": len(self._orders),
                "last_error": self._last_error,
                "orders": [o.to_dict() for o in self._orders.values()][-20:],
            }

    # ------------------------------------------------------------------
    # Internal — Dry Run
    # ------------------------------------------------------------------
    def _simulate_fill(self, order: Order, price: float, now: str) -> Order:
        """Simula preenchimento da ordem em dry-run."""
        with self._lock:
            order.status = OrderStatus.FILLED
            order.filled_amount = order.amount
            order.filled_price = price if price > 0 else order.price or 0.0
            order.cost = order.filled_amount * order.filled_price
            order.fee = order.cost * 0.001  # ~0.1% Binance spot fee
            order.updated_at = now
            self._orders[order.id] = order
            self._next_order_id += 1

        logger.info(
            "[DRY-RUN] Ordem PREENCHIDA: %s %s %.8f @ %.2f (cost=$%.2f fee=$%.4f)",
            order.side.value, order.symbol, order.filled_amount,
            order.filled_price, order.cost, order.fee,
        )
        return order

    # ------------------------------------------------------------------
    # Internal — Real Execution
    # ------------------------------------------------------------------
    def _execute_real_market(self, order: Order) -> Order:
        """Envia ordem a mercado real na Binance."""
        if self._exchange is None:
            order.status = OrderStatus.FAILED
            order.error = "Exchange nao inicializada"
            self._last_error = order.error
            return order

        try:
            result = self._exchange.create_order(
                symbol=order.symbol,
                type="market",
                side=order.side.value,
                amount=order.amount,
            )

            with self._lock:
                order.exchange_order_id = str(result.get("id", ""))
                order.status = OrderStatus.FILLED
                order.filled_amount = float(result.get("filled", order.amount))
                order.filled_price = float(result.get("average", result.get("price", 0)))
                order.cost = float(result.get("cost", 0))
                order.fee = float(result.get("fee", {}).get("cost", 0)) if result.get("fee") else 0.0
                order.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                self._orders[order.id] = order
                self._next_order_id += 1

            logger.info(
                "ORDEM REAL MARKET: %s %s %.8f @ %.2f id=%s",
                order.side.value, order.symbol, order.filled_amount,
                order.filled_price, order.exchange_order_id,
            )
            return order

        except Exception as exc:
            with self._lock:
                order.status = OrderStatus.FAILED
                order.error = str(exc)
                order.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                self._orders[order.id] = order
                self._next_order_id += 1
                self._last_error = str(exc)

            logger.error("Falha na ordem real: %s", exc)
            return order

    def _execute_real_limit(self, order: Order) -> Order:
        """Envia ordem limite real na Binance."""
        if self._exchange is None:
            order.status = OrderStatus.FAILED
            order.error = "Exchange nao inicializada"
            self._last_error = order.error
            return order

        try:
            result = self._exchange.create_order(
                symbol=order.symbol,
                type="limit",
                side=order.side.value,
                amount=order.amount,
                price=order.price,
            )

            with self._lock:
                order.exchange_order_id = str(result.get("id", ""))
                order.status = OrderStatus.OPEN
                order.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                self._orders[order.id] = order
                self._next_order_id += 1

            logger.info(
                "ORDEM REAL LIMIT: %s %s %.8f @ %.2f id=%s",
                order.side.value, order.symbol, order.amount,
                order.price, order.exchange_order_id,
            )
            return order

        except Exception as exc:
            with self._lock:
                order.status = OrderStatus.FAILED
                order.error = str(exc)
                order.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                self._orders[order.id] = order
                self._next_order_id += 1
                self._last_error = str(exc)

            logger.error("Falha na ordem limite: %s", exc)
            return order

    def _cancel_real_order(self, order: Order) -> Optional[Order]:
        """Cancela ordem real na Binance."""
        if self._exchange is None or order.exchange_order_id is None:
            order.status = OrderStatus.CANCELED
            order.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            return order

        try:
            self._exchange.cancel_order(
                order.exchange_order_id,
                symbol=order.symbol,
            )
            order.status = OrderStatus.CANCELED
            order.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            logger.info("Ordem %s cancelada na exchange.", order.exchange_order_id)
            return order
        except Exception as exc:
            logger.error("Falha ao cancelar ordem %s: %s", order.exchange_order_id, exc)
            self._last_error = str(exc)
            return None


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------
_instance: Optional[OrderExecutor] = None
_lock = threading.Lock()


def get_order_executor() -> OrderExecutor:
    """Retorna a instancia unica de OrderExecutor."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = OrderExecutor()
    return _instance
