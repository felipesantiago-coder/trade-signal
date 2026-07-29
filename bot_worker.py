"""
bot_worker.py
-------------
Background worker que executa o loop de trading CTEV de forma assincrona,
paralela ao servidor web FastAPI.

Caracteristicas:
- Roda como asyncio.Task criado no startup do servidor
- Respeita a flag `bot_state.running` (True = ativo, False = pausado)
- Quando pausado, ainda atualiza `last_check` para o painel mostrar que esta vivo
- Salva sinais em SQLite em memoria (db.py) e envia notificacao ao Telegram
- Tolerante a falhas: excecoes sao logadas e o loop continua
- Nao reprocessa o mesmo candle (controle por timestamp)
- Integra RiskManager: valida risco antes de gerar sinais
- Circuit Breaker: detecta movimentos extremos de preco
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import ccxt.async_support as ccxt
import pandas as pd

from bot_state import get_bot_state
from config import Settings
from db import insert_log, insert_signal
from indicators import compute_indicators
from notifier import TelegramNotifier
from risk_manager import RiskBlockReason, get_risk_manager
from strategy import evaluate_signal

logger = logging.getLogger("ctev.worker")

CANDLE_LIMIT = 300  # 200+ para EMA200 com folga
TIMEFRAME_MS = 60 * 60 * 1000  # 1h


class CTEVWorker:
    """Loop de trading rodando como task assincrona paralela ao servidor web."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.state = get_bot_state()
        self.risk = get_risk_manager()
        self.exchange: Optional[ccxt.binance] = None
        self.notifier: Optional[TelegramNotifier] = None
        self.last_processed_ts: Optional[pd.Timestamp] = None
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Inicializa recursos e dispara o loop em background."""
        try:
            self.exchange = ccxt.binance({
                "apiKey": self.settings.binance.api_key,
                "secret": self.settings.binance.api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            })
        except Exception as exc:
            logger.exception("Falha ao inicializar ccxt/Binance: %s", exc)
            insert_log("ERROR", f"Falha ao inicializar Binance: {exc}", "worker")
            self.state.last_error = str(exc)

        # Telegram (opcional)
        try:
            self.notifier = TelegramNotifier(
                bot_token=self.settings.telegram.token,
                chat_id=self.settings.telegram.chat_id,
            )
        except Exception as exc:
            logger.warning("Telegram nao configurado: %s", exc)
            insert_log("WARNING", f"Telegram nao configurado: {exc}", "worker")
            self.notifier = None

        # Configura RiskManager a partir das Settings
        self.risk.configure({
            "max_daily_loss_pct": self.settings.risk.max_daily_loss_pct,
            "max_weekly_loss_pct": self.settings.risk.max_weekly_loss_pct,
            "max_consecutive_losses": self.settings.risk.max_consecutive_losses,
            "circuit_breaker_pct": self.settings.risk.circuit_breaker_pct,
            "cooldown_candles": self.settings.risk.cooldown_candles,
            "cooldown_hours": self.settings.risk.cooldown_hours,
            "atr_pct_min": self.settings.risk.atr_pct_min,
            "atr_pct_max": self.settings.risk.atr_pct_max,
        })

        self.state.last_status_message = "Worker iniciado"
        insert_log(
            "INFO",
            f"Worker CTEV iniciado | symbol={self.settings.binance.symbol} "
            f"tf={self.settings.binance.timeframe} "
            f"risk=daily_max={self.settings.risk.max_daily_loss_pct}% "
            f"cb={self.settings.risk.circuit_breaker_pct}%",
            "worker",
        )

        self._task = asyncio.create_task(self._run_loop())
        logger.info("Background worker CTEV agendado.")

    async def stop(self) -> None:
        """Cancela o loop e fecha recursos."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self.exchange is not None:
            try:
                await self.exchange.close()
            except Exception as exc:
                logger.warning("Erro ao fechar exchange: %s", exc)
        insert_log("INFO", "Worker CTEV parado.", "worker")
        logger.info("Worker parado.")

    # ------------------------------------------------------------------
    # Loop principal
    # ------------------------------------------------------------------
    async def _run_loop(self) -> None:
        logger.info("Loop principal do worker iniciado.")
        while True:
            try:
                await self._cycle()
            except asyncio.CancelledError:
                logger.info("Loop cancelado.")
                raise
            except Exception as exc:
                logger.exception("Erro nao tratado no ciclo: %s", exc)
                self.state.error_count += 1
                self.state.last_error = str(exc)
                insert_log("ERROR", f"Erro no ciclo: {exc}", "worker")
            await asyncio.sleep(self.settings.loop_interval_seconds)

    async def _cycle(self) -> None:
        """Executa um ciclo de verificacao."""
        self.state.cycle_count += 1
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.state.last_check = now_iso

        # Se pausado, apenas atualiza status e retorna
        if not self.state.running:
            self.state.last_status_message = "Pausado — aguardando reativacao"
            return

        if self.exchange is None:
            self.state.last_status_message = "Exchange nao inicializada"
            return

        # Busca candles
        try:
            df = await self._fetch_candles()
        except Exception as exc:
            self.state.error_count += 1
            self.state.last_error = f"Fetch candles: {exc}"
            self.state.last_status_message = f"Erro ao buscar candles: {exc}"
            insert_log("ERROR", f"Fetch candles falhou: {exc}", "worker")
            return

        if df.empty:
            self.state.last_status_message = "Sem dados de candle"
            return

        # Verifica se o ultimo candle ja fechou
        last_open_ts_ms = int(df.index[-1].timestamp() * 1000)
        last_close_ts_ms = last_open_ts_ms + TIMEFRAME_MS
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        if now_ms < last_close_ts_ms:
            self.state.last_status_message = (
                f"Aguardando fechamento do candle {df.index[-1].isoformat()}"
            )
            return

        closed_candle_ts = df.index[-1]

        # Evita reprocessar
        if self.last_processed_ts is not None and closed_candle_ts == self.last_processed_ts:
            self.state.last_status_message = (
                f"Candle {closed_candle_ts.isoformat()} ja processado"
            )
            return

        # Calcula indicadores
        try:
            df_ind = compute_indicators(df)
        except Exception as exc:
            self.state.error_count += 1
            self.state.last_error = f"Indicadores: {exc}"
            self.state.last_status_message = f"Erro ao calcular indicadores: {exc}"
            insert_log("ERROR", f"Indicadores falharam: {exc}", "worker")
            return

        # ---- CIRCUIT BREAKER: detecta movimento extremo de preco ----
        self._check_circuit_breaker(df_ind)

        # ---- VALIDACAO DE RISCO antes de avaliar sinal ----
        last_row = df_ind.iloc[-1]
        atr_pct = float(last_row.get("atr_percentile", 0.5))

        risk_check = self.risk.validate(
            atr_percentile=atr_pct,
            candle_ts=str(closed_candle_ts),
        )

        if not risk_check.allowed:
            self.state.last_status_message = f"Risco bloqueou: {risk_check.message}"
            insert_log(
                "WARNING",
                f"Sinal bloqueado pelo RiskManager: [{risk_check.reason.value}] {risk_check.message}",
                "risk",
            )

            # Notifica Telegram em caso de bloqueio critico
            if risk_check.reason in (
                RiskBlockReason.DAILY_DRAWDOWN,
                RiskBlockReason.WEEKLY_DRAWDOWN,
                RiskBlockReason.CONSECUTIVE_LOSSES,
                RiskBlockReason.KILLED,
                RiskBlockReason.CIRCUIT_BREAKER,
            ) and self.notifier is not None:
                try:
                    await self.notifier.send_text(
                        f"⚠️ *RISK MANAGER* — Sinal bloqueado\n"
                        f"Motivo: `{risk_check.reason.value}`\n"
                        f"{risk_check.message}"
                    )
                except Exception:
                    pass

            self.last_processed_ts = closed_candle_ts
            return

        # Avalia sinal
        try:
            signal = evaluate_signal(df_ind)
        except Exception as exc:
            self.state.error_count += 1
            self.state.last_error = f"Strategy: {exc}"
            self.state.last_status_message = f"Erro ao avaliar sinal: {exc}"
            insert_log("ERROR", f"Strategy falhou: {exc}", "worker")
            return

        if signal is None:
            self.state.last_status_message = (
                f"Sem sinal no candle {closed_candle_ts.isoformat()} "
                f"(ATR pct: {atr_pct:.2f})"
            )
            self.last_processed_ts = closed_candle_ts
            return

        # ---- SINAL GERADO ----
        # Registra no RiskManager (para cooldown)
        self.risk.register_signal(str(closed_candle_ts))

        # Salva sinal no banco em memoria
        signal_dict = signal.to_dict()
        signal_dict["symbol"] = self.settings.binance.symbol
        signal_dict["notified"] = 0
        try:
            insert_signal(signal_dict)
        except Exception as exc:
            logger.exception("Erro ao inserir sinal no DB: %s", exc)
            insert_log("ERROR", f"DB insert signal: {exc}", "worker")

        # Envia ao Telegram (se configurado)
        if self.notifier is not None:
            try:
                await self.notifier.send_signal(signal, self.settings.binance.symbol)
                signal_dict["notified"] = 1
            except Exception as exc:
                logger.exception("Erro ao enviar Telegram: %s", exc)
                insert_log("ERROR", f"Telegram envio: {exc}", "worker")

        # Atualiza estado
        self.state.last_signal_ts = now_iso
        self.state.last_status_message = (
            f"Sinal {signal.type.value} gerado em {signal.entry_price:.2f} "
            f"(ATR pct: {signal.atr_percentile:.2f})"
        )
        self.last_processed_ts = closed_candle_ts

        insert_log(
            "INFO",
            f"Sinal {signal.type.value} | entry={signal.entry_price:.2f} "
            f"SL={signal.stop_loss:.2f} TP={signal.take_profit:.2f} "
            f"ATR_pct={signal.atr_percentile:.2f}",
            "worker",
        )

    # ------------------------------------------------------------------
    # Circuit Breaker
    # ------------------------------------------------------------------
    def _check_circuit_breaker(self, df_ind: pd.DataFrame) -> None:
        """
        Verifica se o ultimo candle teve movimento extremo (ex: > 3%).
        Se sim, aciona o circuit breaker automaticamente.
        """
        if len(df_ind) < 2:
            return

        prev_close = float(df_ind.iloc[-2]["close"])
        curr_close = float(df_ind.iloc[-1]["close"])
        move_pct = abs(curr_close - prev_close) / prev_close * 100

        cb_pct = self.settings.risk.circuit_breaker_pct
        if move_pct > cb_pct:
            self.risk.trigger_circuit_breaker(
                price_move_pct=round(move_pct, 2),
                duration_minutes=60,
            )
            self.state.last_status_message = (
                f"CIRCUIT BREAKER: movimento de {move_pct:.2f}% detectado!"
            )
            insert_log(
                "CRITICAL",
                f"Circuit Breaker acionado: candle moveu {move_pct:.2f}% "
                f"(limite: {cb_pct}%). Pausa por 60 minutos.",
                "risk",
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    async def _fetch_candles(self) -> pd.DataFrame:
        if self.exchange is None:
            raise RuntimeError("Exchange nao inicializada.")
        ohlcv = await self.exchange.fetch_ohlcv(
            symbol=self.settings.binance.symbol,
            timeframe=self.settings.binance.timeframe,
            limit=CANDLE_LIMIT,
        )
        if not ohlcv:
            raise RuntimeError("Resposta vazia da Binance.")
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("datetime", inplace=True)
        df.drop(columns=["timestamp"], inplace=True)
        return df
