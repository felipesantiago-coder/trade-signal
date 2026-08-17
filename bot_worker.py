"""
bot_worker.py
-------------
Background worker que executa o loop de trading CTEV de forma assincrona,
paralela ao servidor web FastAPI.

Caracteristicas:
- Roda como asyncio.Task criado no startup do servidor
- Respeita a flag `bot_state.running` (True = ativo, False = pausado)
- Tolerante a falhas: excecoes sao logadas e o loop continua
- Nao reprocessa o mesmo candle (controle por timestamp)
- Integra RiskManager: valida risco antes de gerar sinais
- Circuit Breaker: detecta movimentos extremos de preco
- Position Tracker: monitora posicoes abertas com trailing stop e break-even
- Position Sizing: calcula tamanho de posicao baseado em risco % e ATR
- Multi-Timeframe Filter: confirma tendencia em H4/D1 antes de sinal
- Order Executor: executa ordens reais ou simuladas (dry-run)
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
from db import insert_closed_trade, insert_log, insert_signal
from exchange_loader import ExchangeLoader, ExchangeInfo
from indicators import compute_indicators
from multi_timeframe import get_mtf_filter
from notifier import TelegramNotifier
from order_executor import OrderSide, get_order_executor
from position_sizing import get_position_sizer
from position_tracker import get_position_tracker, PositionStatus
from risk_manager import RiskBlockReason, get_risk_manager
from strategy import evaluate_signal
from strategy_profiles import get_profile
from strategy_router import (
    evaluate_signal as router_evaluate_signal,
    get_strategy_type,
    get_strategy_label,
)
from regime_engine import classify_regimes_v2
from strategy_regime import evaluate_signal_regime_aware

logger = logging.getLogger("ctev.worker")

CANDLE_LIMIT = 300

# Mapa de timeframe para milissegundos (dinamico)
_TF_MS_MAP = {
    "1m": 60 * 1000,
    "3m": 3 * 60 * 1000,
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "2h": 2 * 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "6h": 6 * 60 * 60 * 1000,
    "8h": 8 * 60 * 60 * 1000,
    "12h": 12 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}


def _get_timeframe_ms(timeframe: str) -> int:
    """Converte string de timeframe para milissegundos."""
    if timeframe in _TF_MS_MAP:
        return _TF_MS_MAP[timeframe]
    raise ValueError(
        f"Timeframe '{timeframe}' nao suportado. Opcoes: {list(_TF_MS_MAP.keys())}"
    )


class CTEVWorker:
    """Loop de trading rodando como task assincrona paralela ao servidor web."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.state = get_bot_state()
        self.risk = get_risk_manager()
        self.sizer = get_position_sizer()
        self.tracker = get_position_tracker()
        self.executor = get_order_executor()
        self.mtf = get_mtf_filter()
        self.exchange = None  # type: Optional[ccxt.Exchange]
        self.exchange_info: Optional[ExchangeInfo] = None
        self.notifier: Optional[TelegramNotifier] = None
        self.last_processed_ts: Optional[pd.Timestamp] = None
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        # Conecta exchange com fallback automatico
        try:
            loader = ExchangeLoader()
            self.exchange, self.exchange_info = await loader.connect(
                preferred_id=self.settings.binance.exchange_id,
                symbol=self.settings.binance.symbol,
                api_key=self.settings.binance.api_key,
                api_secret=self.settings.binance.api_secret,
            )
            logger.info(
                "Exchange conectada: %s (symbol=%s)",
                self.exchange_info.exchange_id,
                self.exchange_info.symbol,
            )
            insert_log(
                "INFO",
                f"Exchange conectada via fallback: {self.exchange_info.exchange_id} "
                f"symbol={self.exchange_info.symbol} "
                f"(original={self.exchange_info.original_symbol})",
                "worker",
            )
        except Exception as exc:
            logger.exception("Falha ao conectar exchange: %s", exc)
            insert_log("ERROR", f"Falha ao conectar exchange: {exc}", "worker")
            self.state.last_error = str(exc)

        try:
            self.notifier = TelegramNotifier(
                bot_token=self.settings.telegram.token,
                chat_id=self.settings.telegram.chat_id,
            )
        except Exception as exc:
            logger.warning("Telegram nao configurado: %s", exc)
            insert_log("WARNING", f"Telegram nao configurado: {exc}", "worker")
            self.notifier = None

        # Configura RiskManager
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

        # Configura PositionSizer
        self.sizer.configure({
            "balance": self.settings.position.account_balance,
            "risk_per_trade_pct": self.settings.position.risk_per_trade_pct,
            "min_position_usd": self.settings.position.min_position_usd,
            "max_position_pct": self.settings.position.max_position_pct,
        })

        # Configura OrderExecutor
        self.executor.configure({
            "dry_run": self.settings.exchange.dry_run,
            "exchange": self.exchange,
        })

        # Configura Multi-Timeframe Filter
        self.mtf.configure({
            "enabled": self.settings.multitf.enabled,
            "cache_ttl": self.settings.multitf.cache_ttl_seconds,
        })

        self.state.last_status_message = "Worker iniciado"
        ex_id = self.exchange_info.exchange_id if self.exchange_info else "?"
        ex_sym = self.exchange_info.symbol if self.exchange_info else self.settings.binance.symbol

        # Notifica via Telegram que o bot iniciou
        if self.notifier is not None:
            try:
                await self.notifier.send_welcome(ex_id, ex_sym)
                if self.exchange_info:
                    await self.notifier.send_exchange_connected(
                        self.exchange_info.exchange_id,
                        self.exchange_info.symbol,
                    )
            except Exception:
                pass
        insert_log(
            "INFO",
            f"Worker CTEV V13-ROBUSTA iniciado | exchange={ex_id} "
            f"symbol={ex_sym} "
            f"balance=${self.settings.position.account_balance:,.0f} "
            f"risk={self.settings.position.risk_per_trade_pct*100:.1f}%/trade "
            f"trailing={self.settings.position.trailing_atr_mult}xATR "
            f"mtf={'ON' if self.settings.multitf.enabled else 'OFF'} "
            f"executor={'DRY-RUN' if self.settings.exchange.dry_run else 'LIVE'}",
            "worker",
        )

        self._task = asyncio.create_task(self._run_loop())
        logger.info("Background worker CTEV agendado.")

    async def stop(self) -> None:
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

        # Resolve timeframe: override em runtime > settings > default
        active_tf = self.state.timeframe_override or self.settings.binance.timeframe

        # Verifica candle fechado (dinamico por timeframe)
        timeframe_ms = _get_timeframe_ms(active_tf)
        last_open_ts_ms = int(df.index[-1].timestamp() * 1000)
        last_close_ts_ms = last_open_ts_ms + timeframe_ms
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        if now_ms < last_close_ts_ms:
            pos_status = self._get_position_status_text()
            self.state.last_status_message = (
                f"Aguardando fechamento do candle {df.index[-1].isoformat()} | {pos_status}"
            )
            return

        closed_candle_ts = df.index[-1]

        if self.last_processed_ts is not None and closed_candle_ts == self.last_processed_ts:
            self.state.last_status_message = (
                f"Candle {closed_candle_ts.isoformat()} ja processado"
            )
            return

        # Calcula indicadores
        try:
            df_ind = compute_indicators(df, timeframe=self.settings.binance.timeframe)
        except Exception as exc:
            self.state.error_count += 1
            self.state.last_error = f"Indicadores: {exc}"
            self.state.last_status_message = f"Erro ao calcular indicadores: {exc}"
            insert_log("ERROR", f"Indicadores falharam: {exc}", "worker")
            return

        # ---- MONITORAR POSICOES ABERTAS (a cada candle fechado) ----
        await self._monitor_open_positions(df_ind, closed_candle_ts)

        # ---- CIRCUIT BREAKER ----
        self._check_circuit_breaker(df_ind)

        # ---- NAO gera novo sinal se ja tem posicao aberta ----
        if self.tracker.has_open_positions:
            pos = self.tracker.get_open_position()
            if pos:
                self.state.last_status_message = (
                    f"Posicao #{pos.id} {pos.type} aberta | "
                    f"SL={pos.trailing_stop if pos.trailing_activated else pos.stop_loss:.2f} | "
                    f"entry={pos.entry_price:.2f}"
                )
            self.last_processed_ts = closed_candle_ts
            return

        # ---- MULTI-TIMEFRAME ANALYSIS ----
        if self.mtf.enabled:
            try:
                mtf_symbol = self.exchange_info.symbol if self.exchange_info else self.settings.binance.symbol
                active_tf = self.state.timeframe_override or self.settings.binance.timeframe
                mtf_result = await self.mtf.analyze(
                    self.exchange,
                    mtf_symbol,
                    active_tf=active_tf,  # v5: passa TF ativo para MTF adaptativo
                )
            except Exception as exc:
                logger.warning("MTF analysis falhou: %s (continuando sem filtro)", exc)
                mtf_result = None

        # ---- VALIDACAO DE RISCO ----
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
                f"Sinal bloqueado: [{risk_check.reason.value}] {risk_check.message}",
                "risk",
            )

            if risk_check.reason in (
                RiskBlockReason.DAILY_DRAWDOWN,
                RiskBlockReason.WEEKLY_DRAWDOWN,
                RiskBlockReason.CONSECUTIVE_LOSSES,
                RiskBlockReason.KILLED,
                RiskBlockReason.CIRCUIT_BREAKER,
            ) and self.notifier is not None:
                try:
                    await self.notifier.send_risk_alert(
                        risk_check.reason.value,
                        risk_check.message,
                    )
                except Exception:
                    pass

            self.last_processed_ts = closed_candle_ts
            return

        # ---- AVALIA SINAL (ROUTER INTELIGENTE POR TIMEFRAME) ----
        try:
            active_tf = self.state.timeframe_override or self.settings.binance.timeframe
            _profile = get_profile(active_tf)
            _strategy_type = get_strategy_type(active_tf)
            _strategy_label = get_strategy_label(active_tf)

            # Router: seleciona estrategia automaticamente por timeframe
            # 15m/30m -> EMA Cross v8 | 1h+ -> CTEV v7.1 Regime-Switching
            signal = router_evaluate_signal(df_ind, timeframe=active_tf, profile=_profile)

            # Expose regime/strategy info in status
            last_row = df_ind.iloc[-1]
            _regime_v2 = str(last_row.get("regime_v2", last_row.get("regime", "?")))
            _regime_conf = float(last_row.get("regime_confidence", 0))
            _regime_strat = str(last_row.get("regime_strategy", _strategy_label))
            logger.info(
                "Router [%s] -> %s | regime=%s (conf=%.2f)",
                active_tf, _strategy_label, _regime_v2, _regime_conf,
            )
        except Exception as exc:
            self.state.error_count += 1
            self.state.last_error = f"Strategy: {exc}"
            self.state.last_status_message = f"Erro ao avaliar sinal: {exc}"
            insert_log("ERROR", f"Strategy falhou: {exc}", "worker")
            return

        if signal is None:
            self.state.last_status_message = (
                f"Sem sinal no candle {closed_candle_ts.isoformat()} "
                f"| regime={_regime_v2} ({_regime_strat}) "
                f"| ATR pct: {atr_pct:.2f}"
            )
            self.last_processed_ts = closed_candle_ts
            return

        # ---- MULTI-TIMEFRAME FILTER ----
        if self.mtf.enabled:
            allowed, reason = self.mtf.allows_signal(signal.type.value, mtf_result)
            if not allowed:
                self.state.last_status_message = f"MTF bloqueou: {reason}"
                insert_log("WARNING", reason, "mtf")
                if self.notifier is not None:
                    try:
                        await self.notifier.send_text(
                            f"⏳ *MULTI-TF FILTER* — Sinal {signal.type.value} bloqueado\n"
                            f"{reason}"
                        )
                    except Exception:
                        pass
                self.last_processed_ts = closed_candle_ts
                return

        # ---- SINAL GERADO — Registrar e notificar ----
        self.risk.register_signal(str(closed_candle_ts))

        # Salva sinal no DB
        signal_dict = signal.to_dict()
        signal_dict["symbol"] = self.settings.binance.symbol
        signal_dict["notified"] = 0
        try:
            insert_signal(signal_dict)
        except Exception as exc:
            logger.exception("Erro ao inserir sinal no DB: %s", exc)
            insert_log("ERROR", f"DB insert signal: {exc}", "worker")

        # === SINAL GERADO — Apenas notifica, sem executar ordem ===

        # Telegram: envia sinal detalhado com analise e regime
        if self.notifier is not None:
            try:
                await self.notifier.send_signal(
                    signal, self.settings.binance.symbol,
                    regime=_regime_v2,
                    strategy=_regime_strat,
                    confidence=_regime_conf,
                )
                signal_dict["notified"] = 1
            except Exception as exc:
                logger.exception("Erro ao enviar Telegram: %s", exc)
                insert_log("ERROR", f"Telegram envio: {exc}", "worker")

        self.state.last_signal_ts = now_iso
        self.state.last_status_message = (
            f"Sinal {signal.type.value} em {signal.entry_price:.2f} "
            f"| SL={signal.stop_loss:.2f} TP={signal.take_profit:.2f} "
            f"RSI={signal.rsi:.1f} regime={_regime_v2} ({_regime_strat})"
        )
        self.last_processed_ts = closed_candle_ts

        insert_log(
            "INFO",
            f"Sinal {signal.type.value} emitido | "
            f"entry={signal.entry_price:.2f} SL={signal.stop_loss:.2f} TP={signal.take_profit:.2f} "
            f"RSI={signal.rsi:.1f} ATR={signal.atr:.2f} "
            f"regime={_regime_v2} ({_regime_strat}) conf={_regime_conf:.2f}",
            "worker",
        )

    # ------------------------------------------------------------------
    # Monitoramento de posicoes abertas
    # ------------------------------------------------------------------
    async def _monitor_open_positions(
        self, df_ind: pd.DataFrame, candle_ts: pd.Timestamp
    ) -> None:
        """Verifica e atualiza posicoes abertas a cada candle fechado."""
        if not self.tracker.has_open_positions:
            return

        last = df_ind.iloc[-1]
        close = float(last["close"])
        high = float(last["high"])
        low = float(last["low"])

        prev_trailing = {}
        for pos in self.tracker.open_positions:
            prev_trailing[pos.id] = pos.trailing_stop if pos.trailing_activated else pos.stop_loss

        closed_positions = self.tracker.update_positions(
            candle_close=close,
            candle_high=high,
            candle_low=low,
            candle_ts=str(candle_ts),
            trailing_atr_mult=self.settings.position.trailing_atr_mult,
        )

        # Processa posicoes fechadas
        for pos in closed_positions:
            # Salva no DB
            try:
                insert_closed_trade({
                    "entry_ts": pos.entry_ts,
                    "exit_ts": pos.exit_ts,
                    "type": pos.type,
                    "symbol": self.settings.binance.symbol,
                    "entry_price": pos.entry_price,
                    "exit_price": pos.exit_price,
                    "stop_loss": pos.stop_loss_initial,
                    "take_profit": pos.take_profit,
                    "atr": pos.atr,
                    "position_size": pos.position_size,
                    "position_usd": pos.position_usd,
                    "pnl_pct": pos.pnl_pct,
                    "pnl_usd": pos.pnl_usd,
                    "exit_reason": pos.exit_reason,
                    "trailing_activated": int(pos.trailing_activated),
                    "be_triggered": int(pos.be_triggered),
                    "partial_tp_filled": int(pos.partial_tp_filled),
                })
            except Exception as exc:
                logger.exception("Erro ao salvar trade no DB: %s", exc)

            # Atualiza balance no sizer
            self.sizer.update_balance(pos.pnl_usd)

            # Registra resultado no risk manager
            self.risk.register_trade_result(pos.pnl_pct)

            # Telegram: notificacao de fechamento
            if self.notifier is not None:
                try:
                    await self.notifier.send_trade_close(
                        pos_type=pos.type,
                        entry=pos.entry_price,
                        exit_p=pos.exit_price,
                        pnl_pct=pos.pnl_pct,
                        pnl_usd=pos.pnl_usd,
                        reason=pos.exit_reason,
                        be=pos.be_triggered,
                        trailing=pos.trailing_activated,
                        partial=pos.partial_tp_filled,
                    )
                except Exception:
                    pass

            insert_log(
                "INFO" if pos.pnl_pct >= 0 else "WARNING",
                f"Trade #{pos.id} FECHADO: {pos.type} exit={pos.exit_price:.2f} "
                f"pnl={pos.pnl_pct:+.2f}% (${pos.pnl_usd:+,.2f}) reason={pos.exit_reason} "
                f"BE={pos.be_triggered} trail={pos.trailing_activated} partial={pos.partial_tp_filled}",
                "worker",
            )

            self.state.last_status_message = (
                f"Trade #{pos.id} {pos.exit_reason.upper()}: "
                f"{pos.pnl_pct:+.2f}% (${pos.pnl_usd:+,.2f})"
            )

        # Notifica atualizacoes de trailing
        if self.notifier is not None:
            for pos in self.tracker.open_positions:
                old_sl = prev_trailing.get(pos.id)
                new_sl = pos.trailing_stop if pos.trailing_activated else pos.stop_loss
                if old_sl and new_sl and abs(new_sl - old_sl) > 0.01:
                    try:
                        await self.notifier.send_trailing_update(
                            pos.type, new_sl, pos.entry_price,
                        )
                    except Exception:
                        pass

    def _get_position_status_text(self) -> str:
        pos = self.tracker.get_open_position()
        if pos is None:
            return "sem posicoes"
        sl = pos.trailing_stop if pos.trailing_activated else pos.stop_loss
        return (
            f"#{pos.id} {pos.type} entry={pos.entry_price:.2f} SL={sl:.2f} "
            f"[{pos.status.value}]"
        )

    # ------------------------------------------------------------------
    # Circuit Breaker
    # ------------------------------------------------------------------
    def _check_circuit_breaker(self, df_ind: pd.DataFrame) -> None:
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

            # Fecha todas as posicoes abertas
            closed = []
            if self.tracker.has_open_positions:
                closed = self.tracker.close_all(
                    exit_price=curr_close,
                    exit_ts=str(df_ind.index[-1]),
                    reason="circuit_breaker",
                )
                for pos in closed:
                    self.sizer.update_balance(pos.pnl_usd)
                    self.risk.register_trade_result(pos.pnl_pct)
                    try:
                        insert_closed_trade({
                            "entry_ts": pos.entry_ts,
                            "exit_ts": pos.exit_ts,
                            "type": pos.type,
                            "symbol": self.settings.binance.symbol,
                            "entry_price": pos.entry_price,
                            "exit_price": pos.exit_price,
                            "stop_loss": pos.stop_loss_initial,
                            "take_profit": pos.take_profit,
                            "atr": pos.atr,
                            "pnl_pct": pos.pnl_pct,
                            "pnl_usd": pos.pnl_usd,
                            "exit_reason": pos.exit_reason,
                        })
                    except Exception:
                        pass

            self.state.last_status_message = (
                f"CIRCUIT BREAKER: {move_pct:.2f}% detectado! {len(closed)} posicoes fechadas."
            )
            insert_log(
                "CRITICAL",
                f"Circuit Breaker: {move_pct:.2f}% (limite {cb_pct}%). "
                f"{len(closed)} posicoes fechadas.",
                "risk",
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    async def _fetch_candles(self) -> pd.DataFrame:
        if self.exchange is None:
            raise RuntimeError("Exchange nao inicializada.")
        symbol = self.exchange_info.symbol if self.exchange_info else self.settings.binance.symbol
        # Usa timeframe ativo (pode ser override em runtime)
        active_tf = self.state.timeframe_override or self.settings.binance.timeframe
        ohlcv = await self.exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=active_tf,
            limit=CANDLE_LIMIT,
        )
        if not ohlcv:
            raise RuntimeError(f"Resposta vazia de {self.exchange_info.exchange_id}.")
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("datetime", inplace=True)
        df.drop(columns=["timestamp"], inplace=True)
        return df
