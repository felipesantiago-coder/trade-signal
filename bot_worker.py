"""
bot_worker.py
-------------
Background worker que executa o loop de trading CTEV de forma assíncrona,
paralela ao servidor web FastAPI.

Características:
- Roda como asyncio.Task criado no startup do servidor
- Respeita a flag `bot_state.running` (True = ativo, False = pausado)
- Quando pausado, ainda atualiza `last_check` para o painel mostrar que está vivo
- Salva sinais em SQLite em memória (db.py) e envia notificação ao Telegram
- Tolerante a falhas: exceções são logadas e o loop continua
- Não reprocessa o mesmo candle (controle por timestamp)
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
from strategy import evaluate_signal

logger = logging.getLogger("ctev.worker")

CANDLE_LIMIT = 300  # 200+ para EMA200 com folga
TIMEFRAME_MS = 60 * 60 * 1000  # 1h


class CTEVWorker:
    """Loop de trading rodando como task assíncrona paralela ao servidor web."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.state = get_bot_state()
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
            # Continua tentando — a Binance pode voltar

        # Telegram é opcional — se não configurado, apenas loga
        try:
            self.notifier = TelegramNotifier(
                bot_token=self.settings.telegram.token,
                chat_id=self.settings.telegram.chat_id,
            )
        except Exception as exc:
            logger.warning("Telegram não configurado: %s", exc)
            insert_log("WARNING", f"Telegram não configurado: {exc}", "worker")
            self.notifier = None

        self.state.last_status_message = "Worker iniciado"
        insert_log(
            "INFO",
            f"Worker CTEV iniciado | symbol={self.settings.binance.symbol} "
            f"tf={self.settings.binance.timeframe}",
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
                logger.exception("Erro não tratado no ciclo: %s", exc)
                self.state.error_count += 1
                self.state.last_error = str(exc)
                insert_log("ERROR", f"Erro no ciclo: {exc}", "worker")
            await asyncio.sleep(self.settings.loop_interval_seconds)

    async def _cycle(self) -> None:
        """Executa um ciclo de verificação."""
        self.state.cycle_count += 1
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.state.last_check = now_iso

        # Se pausado, apenas atualiza status e retorna
        if not self.state.running:
            self.state.last_status_message = "Pausado — aguardando reativação"
            return

        if self.exchange is None:
            self.state.last_status_message = "Exchange não inicializada"
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

        # Verifica se o último candle já fechou
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
                f"Candle {closed_candle_ts.isoformat()} já processado"
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
                f"Sem sinal no candle {closed_candle_ts.isoformat()}"
            )
            self.last_processed_ts = closed_candle_ts
            return

        # Salva sinal no banco em memória
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
            f"Sinal {signal.type.value} gerado em {signal.entry_price:.2f}"
        )
        self.last_processed_ts = closed_candle_ts

        insert_log(
            "INFO",
            f"Sinal {signal.type.value} | entry={signal.entry_price:.2f} "
            f"SL={signal.stop_loss:.2f} TP={signal.take_profit:.2f}",
            "worker",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    async def _fetch_candles(self) -> pd.DataFrame:
        if self.exchange is None:
            raise RuntimeError("Exchange não inicializada.")
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
