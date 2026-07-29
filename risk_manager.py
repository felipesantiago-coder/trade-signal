"""
risk_manager.py
---------------
Gerenciador de risco do bot CTEV — Kill Switch, Circuit Breaker,
Max Drawdown diário/semanal e controle de perdas consecutivas.

Este módulo é o guarda-capital do sistema. Antes de qualquer sinal ser
gerado, o RiskManager.valida() é chamado. Se retornar bloqueado,
nenhum sinal é emitido e o bot notifica o operador via Telegram e logs.

Funcionalidades:
    - Max Drawdown Diário: pausa o bot se perda > X% em 24h UTC
    - Max Drawdown Semanal: pausa o bot se perda > Y% em 7 dias
    - Consecutive Losses: pausa após N losses seguidos (com cooldown de retomada)
    - Circuit Breaker: pausa imediato se preço move > Z% em 1 candle
    - Cooldown entre sinais: mínimo de K candles entre sinais
    - Filtro de volatilidade: bloqueia se ATR percentile fora da faixa

Referências:
    - 3Commas (2025): "Drawdown limits are stop thresholds that prevent a bot
      from continuing to trade after losing a specified portion of the portfolio"
    - Reddit r/algotrading (2026): "Regime detection made a bigger difference
      than optimizing any single indicator"
    - ChangeHero (2025): "Kill switch: triggered by extreme losses, connectivity
      issues, or operational errors"
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger("ctev.risk")


class RiskBlockReason(str, Enum):
    """Motivos pelos quais o risk manager pode bloquear sinais."""
    DAILY_DRAWDOWN = "drawdown_diario"
    WEEKLY_DRAWDOWN = "drawdown_semanal"
    CONSECUTIVE_LOSSES = "perdas_consecutivas"
    CIRCUIT_BREAKER = "circuit_breaker"
    VOLATILITY_FILTER = "filtro_volatilidade"
    COOLDOWN = "cooldown_entre_sinais"
    KILLED = "kill_switch_manual"


@dataclass
class ValidationResult:
    """Resultado da validação de risco antes de um sinal."""
    allowed: bool = True
    reason: Optional[RiskBlockReason] = None
    message: str = ""


@dataclass
class RiskState:
    """Estado persistente do gerenciador de risco (serializável para JSON)."""
    is_killed: bool = False
    daily_loss_pct: float = 0.0
    weekly_loss_pct: float = 0.0
    consecutive_losses: int = 0
    total_signals: int = 0
    total_filtered: int = 0
    circuit_breaker_active: bool = False
    circuit_breaker_until: Optional[str] = None
    daily_pnl_reset_at: str = ""
    weekly_pnl_reset_at: str = ""
    last_signal_candle_ts: Optional[str] = None
    max_daily_loss_pct: float = 5.0
    max_weekly_loss_pct: float = 10.0
    max_consecutive_losses: int = 5
    circuit_breaker_pct: float = 3.0
    cooldown_candles: int = 3
    atr_pct_min: float = 0.20
    atr_pct_max: float = 0.80
    cooldown_hours: int = 12  # horas de pausa após consecutive losses


class RiskManager:
    """
    Gerenciador de risco central do bot CTEV.

    Singleton (ver `get_risk_manager()`). Thread-safe via Lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = RiskState(
            daily_pnl_reset_at=_now_iso(),
            weekly_pnl_reset_at=_now_iso(),
        )
        self._paused_until: Optional[datetime] = None
        self._consecutive_pause_until: Optional[datetime] = None

    @property
    def state(self) -> RiskState:
        with self._lock:
            return self._state

    # ------------------------------------------------------------------
    # Configuração via ambiente
    # ------------------------------------------------------------------
    def configure(self, kwargs: dict) -> None:
        """Atualiza parâmetros de risco a partir de um dicionário (geralmente do .env)."""
        with self._lock:
            if "max_daily_loss_pct" in kwargs:
                self._state.max_daily_loss_pct = float(kwargs["max_daily_loss_pct"])
            if "max_weekly_loss_pct" in kwargs:
                self._state.max_weekly_loss_pct = float(kwargs["max_weekly_loss_pct"])
            if "max_consecutive_losses" in kwargs:
                self._state.max_consecutive_losses = int(kwargs["max_consecutive_losses"])
            if "circuit_breaker_pct" in kwargs:
                self._state.circuit_breaker_pct = float(kwargs["circuit_breaker_pct"])
            if "cooldown_candles" in kwargs:
                self._state.cooldown_candles = int(kwargs["cooldown_candles"])
            if "atr_pct_min" in kwargs:
                self._state.atr_pct_min = float(kwargs["atr_pct_min"])
            if "atr_pct_max" in kwargs:
                self._state.atr_pct_max = float(kwargs["atr_pct_max"])
            if "cooldown_hours" in kwargs:
                self._state.cooldown_hours = int(kwargs["cooldown_hours"])
        logger.info(
            "RiskManager configurado: daily_max=%.1f%% weekly_max=%.1f%% "
            "max_consec=%d cb_pct=%.1f%% cooldown=%d candles atr_range=%.0f%%-%.0f%%",
            self._state.max_daily_loss_pct,
            self._state.max_weekly_loss_pct,
            self._state.max_consecutive_losses,
            self._state.circuit_breaker_pct,
            self._state.cooldown_candles,
            self._state.atr_pct_min * 100,
            self._state.atr_pct_max * 100,
        )

    # ------------------------------------------------------------------
    # Validação pré-sinal (chamada pelo bot_worker antes de evaluate_signal)
    # ------------------------------------------------------------------
    def validate(
        self,
        atr_percentile: float = 0.5,
        candle_ts: Optional[str] = None,
    ) -> ValidationResult:
        """
        Verifica se é seguro gerar um sinal. Chamado antes de evaluate_signal().

        Parâmetros:
            atr_percentile: percentil do ATR(14) nos últimos 100 candles (0.0 a 1.0)
            candle_ts: timestamp ISO do candle atual

        Retorna:
            ValidationResult com allowed=True/False e motivo se bloqueado
        """
        with self._lock:
            self._reset_periods_if_needed()
            st = self._state

            # 1. Kill switch manual
            if st.is_killed:
                st.total_filtered += 1
                logger.warning("SINAL BLOQUEADO: Kill Switch manual ativo.")
                return ValidationResult(
                    allowed=False,
                    reason=RiskBlockReason.KILLED,
                    message="Kill Switch manual ativo. Nenhum sinal será gerado.",
                )

            # 2. Pausa temporária (consecutive losses cooldown)
            if self._consecutive_pause_until and datetime.now(timezone.utc) < self._consecutive_pause_until:
                remaining = (self._consecutive_pause_until - datetime.now(timezone.utc)).total_seconds() / 60
                st.total_filtered += 1
                return ValidationResult(
                    allowed=False,
                    reason=RiskBlockReason.CONSECUTIVE_LOSSES,
                    message=f"Cooldown por perdas consecutivas. Restam {remaining:.0f} minutos.",
                )

            # 3. Max drawdown diário
            if st.daily_loss_pct >= st.max_daily_loss_pct:
                st.total_filtered += 1
                logger.warning(
                    "SINAL BLOQUEADO: Drawdown diário %.2f%% >= %.2f%%",
                    st.daily_loss_pct, st.max_daily_loss_pct,
                )
                return ValidationResult(
                    allowed=False,
                    reason=RiskBlockReason.DAILY_DRAWDOWN,
                    message=f"Drawdown diário {st.daily_loss_pct:.2f}% atingiu o limite de {st.max_daily_loss_pct}%. Bot pausado até 00:00 UTC.",
                )

            # 4. Max drawdown semanal
            if st.weekly_loss_pct >= st.max_weekly_loss_pct:
                st.total_filtered += 1
                logger.warning(
                    "SINAL BLOQUEADO: Drawdown semanal %.2f%% >= %.2f%%",
                    st.weekly_loss_pct, st.max_weekly_loss_pct,
                )
                return ValidationResult(
                    allowed=False,
                    reason=RiskBlockReason.WEEKLY_DRAWDOWN,
                    message=f"Drawdown semanal {st.weekly_loss_pct:.2f}% atingiu o limite de {st.max_weekly_loss_pct}%. Bot pausado até segunda-feira UTC.",
                )

            # 5. Circuit breaker ativo
            if st.circuit_breaker_active:
                if st.circuit_breaker_until and datetime.now(timezone.utc) < datetime.fromisoformat(st.circuit_breaker_until):
                    st.total_filtered += 1
                    return ValidationResult(
                        allowed=False,
                        reason=RiskBlockReason.CIRCUIT_BREAKER,
                        message="Circuit Breaker ativo — movimento extremo detectado. Aguardando normalização.",
                    )
                else:
                    st.circuit_breaker_active = False
                    st.circuit_breaker_until = None
                    logger.info("Circuit Breaker desativado — retomando operações.")

            # 6. Filtro de volatilidade (ATR percentile)
            if atr_percentile < st.atr_pct_min or atr_percentile > st.atr_pct_max:
                st.total_filtered += 1
                return ValidationResult(
                    allowed=False,
                    reason=RiskBlockReason.VOLATILITY_FILTER,
                    message=(
                        f"ATR percentile {atr_percentile:.2f} fora da faixa segura "
                        f"[{st.atr_pct_min:.2f}, {st.atr_pct_max:.2f}]. "
                        f"Mercado muito {'lateral' if atr_percentile < st.atr_pct_min else 'volátil'}."
                    ),
                )

            # 7. Cooldown entre sinais
            if st.last_signal_candle_ts and candle_ts:
                if candle_ts == st.last_signal_candle_ts:
                    st.total_filtered += 1
                    return ValidationResult(
                        allowed=False,
                        reason=RiskBlockReason.COOLDOWN,
                        message="Mesmo candle já gerou sinal anteriormente.",
                    )

            return ValidationResult(allowed=True)

    # ------------------------------------------------------------------
    # Registro de eventos de trading
    # ------------------------------------------------------------------
    def register_signal(self, candle_ts: str) -> None:
        """Registra que um sinal foi gerado (para controle de cooldown)."""
        with self._lock:
            self._state.total_signals += 1
            self._state.last_signal_candle_ts = candle_ts

    def register_trade_result(self, pnl_pct: float) -> None:
        """
        Registra resultado de um trade encerrado.

        Parâmetros:
            pnl_pct: lucro/prejuízo em percentual (positivo = lucro, negativo = perda)
        """
        with self._lock:
            self._reset_periods_if_needed()
            st = self._state

            # Atualiza drawdowns
            st.daily_loss_pct = max(0.0, st.daily_loss_pct + (-pnl_pct if pnl_pct < 0 else 0))
            st.weekly_loss_pct = max(0.0, st.weekly_loss_pct + (-pnl_pct if pnl_pct < 0 else 0))

            if pnl_pct < 0:
                # Loss
                st.consecutive_losses += 1
                logger.warning(
                    "Trade loss registrado: %.2f%%. Consecutivas: %d/%d",
                    pnl_pct, st.consecutive_losses, st.max_consecutive_losses,
                )
                if st.consecutive_losses >= st.max_consecutive_losses:
                    self._consecutive_pause_until = datetime.now(timezone.utc) + timedelta(
                        hours=st.cooldown_hours
                    )
                    logger.critical(
                        "KILL SWITCH automático: %d perdas consecutivas. "
                        "Pausa por %d horas.",
                        st.consecutive_losses, st.cooldown_hours,
                    )
            else:
                # Win
                st.consecutive_losses = 0
                self._consecutive_pause_until = None
                logger.info(
                    "Trade win registrado: +%.2f%%. Consecutivas resetadas.",
                    pnl_pct,
                )

    # ------------------------------------------------------------------
    # Circuit Breaker
    # ------------------------------------------------------------------
    def trigger_circuit_breaker(self, price_move_pct: float, duration_minutes: int = 60) -> None:
        """
        Aciona o circuit breaker após movimento extremo de preço.

        Parâmetros:
            price_move_pct: percentual de movimento no candle (ex: 3.5)
            duration_minutes: duração da pausa (default: 60 min)
        """
        with self._lock:
            self._state.circuit_breaker_active = True
            self._state.circuit_breaker_until = (
                datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)
            ).isoformat()
            logger.critical(
                "CIRCUIT BREAKER acionado: movimento de %.2f%%. "
                "Pausa por %d minutos até %s.",
                price_move_pct, duration_minutes, self._state.circuit_breaker_until,
            )

    # ------------------------------------------------------------------
    # Kill Switch manual
    # ------------------------------------------------------------------
    def kill(self) -> None:
        """Ativa o kill switch manual — bloqueia TODOS os sinais até reset manual."""
        with self._lock:
            self._state.is_killed = True
            logger.critical("KILL SWITCH manual ATIVADO pelo operador.")

    def resurrect(self) -> None:
        """Desativa o kill switch manual e reseta contadores de perda."""
        with self._lock:
            self._state.is_killed = False
            self._state.consecutive_losses = 0
            self._state.circuit_breaker_active = False
            self._state.circuit_breaker_until = None
            self._consecutive_pause_until = None
            logger.info("KILL SWITCH desativado. Bot retomando operações.")

    def reset_daily(self) -> None:
        """Reseta contadores diários (chamado à meia-noite UTC)."""
        with self._lock:
            self._state.daily_loss_pct = 0.0
            self._state.daily_pnl_reset_at = _now_iso()
            logger.info("Drawdown diário resetado.")

    def reset_weekly(self) -> None:
        """Reseta contadores semanais (chamado à segunda-feira UTC)."""
        with self._lock:
            self._state.weekly_loss_pct = 0.0
            self._state.weekly_pnl_reset_at = _now_iso()
            logger.info("Drawdown semanal resetado.")

    # ------------------------------------------------------------------
    # Snapshot para JSON (usado pelo painel)
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            st = self._state
            return {
                "is_killed": st.is_killed,
                "daily_loss_pct": round(st.daily_loss_pct, 4),
                "weekly_loss_pct": round(st.weekly_loss_pct, 4),
                "consecutive_losses": st.consecutive_losses,
                "total_signals": st.total_signals,
                "total_filtered": st.total_filtered,
                "circuit_breaker_active": st.circuit_breaker_active,
                "circuit_breaker_until": st.circuit_breaker_until,
                "max_daily_loss_pct": st.max_daily_loss_pct,
                "max_weekly_loss_pct": st.max_weekly_loss_pct,
                "max_consecutive_losses": st.max_consecutive_losses,
                "circuit_breaker_pct": st.circuit_breaker_pct,
                "cooldown_candles": st.cooldown_candles,
                "atr_pct_min": st.atr_pct_min,
                "atr_pct_max": st.atr_pct_max,
                "cooldown_hours": st.cooldown_hours,
                "consecutive_pause_until": (
                    self._consecutive_pause_until.isoformat()
                    if self._consecutive_pause_until
                    else None
                ),
            }

    # ------------------------------------------------------------------
    # Interno
    # ------------------------------------------------------------------
    def _reset_periods_if_needed(self) -> None:
        """Reseta drawdown diário/semanal se o período expirou."""
        now = datetime.now(timezone.utc)

        # Reset diário à meia-noite UTC
        if self._state.daily_pnl_reset_at:
            last_reset = datetime.fromisoformat(self._state.daily_pnl_reset_at)
            if now.date() > last_reset.date():
                self._state.daily_loss_pct = 0.0
                self._state.daily_pnl_reset_at = _now_iso()
                logger.info("Auto-reset: drawdown diário (nova dia UTC).")

        # Reset semanal às segundas (weekday 0)
        if self._state.weekly_pnl_reset_at:
            last_reset = datetime.fromisoformat(self._state.weekly_pnl_reset_at)
            if now.weekday() == 0 and now.date() > last_reset.date():
                self._state.weekly_loss_pct = 0.0
                self._state.weekly_pnl_reset_at = _now_iso()
                logger.info("Auto-reset: drawdown semanal (segunda-feira UTC).")


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------
_instance: Optional[RiskManager] = None
_lock = threading.Lock()


def get_risk_manager() -> RiskManager:
    """Retorna a instância única de RiskManager."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = RiskManager()
    return _instance


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
