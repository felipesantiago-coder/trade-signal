"""
server.py
---------
Servidor web FastAPI que expoe:
- GET  /               -> Painel de analise (HTML)
- GET  /health         -> 200 OK (para UptimeRobot / liveness probes)
- HEAD /api/status     -> Health check HEAD (UptimeRobot)
- GET  /api/status     -> Estado completo do bot + stats
- GET  /api/mode        -> Modo de operacao (sempre 'signal_only')
- GET  /api/signals    -> Lista sinais recentes
- GET  /api/logs       -> Lista logs recentes
- GET  /api/risk       -> Estado do RiskManager
- GET  /api/positions  -> Posicoes e trades
- GET  /api/trades     -> Trades fechados com summary
- GET  /api/mtf        -> Analise multi-timeframe H4/D1
- GET  /api/executor   -> Estado do executor (signal_only)
- POST /api/backtest   -> Dispara backtest assincrono
- GET  /api/backtest/status -> Status do backtest
- GET  /api/backtest/progress -> Progresso em tempo real
- GET  /api/backtest/stream  -> SSE stream de progresso

MODO: Signal-Only — apenas analise e emissao de sinais.
Nenhum endpoint de execucao de ordens esta disponivel.

O servidor inicializa o background worker (CTEVWorker) no startup e o
encerra graciosamente no shutdown.

Endpoints de chart data:
- GET  /api/chart-data -> Dados de precos + indicadores + sinais para graficos
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from bot_state import get_bot_state
from bot_worker import CTEVWorker
from config import Settings, load_settings
from db import (
    count_signals_by_type_today,
    count_signals_today,
    get_trades_summary,
    insert_log,
    list_recent_logs,
    list_recent_signals,
    list_recent_trades,
)
from indicators import compute_indicators
from multi_timeframe import get_mtf_filter
from position_tracker import get_position_tracker
from risk_manager import get_risk_manager

logger = logging.getLogger("ctev.server")

# ------------------------------------------------------------------
# App + estado global do worker
# ------------------------------------------------------------------
app = FastAPI(
    title="CTEV Signal Bot - Painel de Analise",
    description="Bot de sinais CTEV para BTC/USD 1H com painel web, "
                "RiskManager e Backtesting. Apenas analise — sem execucao de ordens.",
    version="4.0.0",
)

_worker: Optional[CTEVWorker] = None
_settings: Optional[Settings] = None
_backtest_task: Optional[asyncio.Task] = None
_backtest_result: Optional[dict] = None
_backtest_running: bool = False


def get_worker() -> CTEVWorker:
    if _worker is None:
        raise RuntimeError("Worker nao inicializado.")
    return _worker


def get_settings() -> Settings:
    if _settings is None:
        raise RuntimeError("Settings nao carregado.")
    return _settings


# ------------------------------------------------------------------
# Lifecycle
# ------------------------------------------------------------------
@app.on_event("startup")
async def _on_startup() -> None:
    """Carrega settings e dispara o background worker (nao-bloqueante)."""
    global _worker, _settings
    try:
        _settings = load_settings()
    except RuntimeError as exc:
        logger.error("Falha ao carregar settings: %s", exc)
        from config import (
            BinanceConfig, ExchangeConfig, MultiTFConfig, OptimizerConfig,
            PositionConfig, Settings as S, TelegramConfig, RiskConfig,
        )
        _settings = S(
            telegram=TelegramConfig(token="", chat_id=""),
            binance=BinanceConfig(
                exchange_id=os.getenv("EXCHANGE_ID", "coinbase"),
                symbol=os.getenv("EXCHANGE_SYMBOL", "BTC/USD"),
                timeframe=os.getenv("EXCHANGE_TIMEFRAME", "1h"),
            ),
            risk=RiskConfig(),
            position=PositionConfig(),
            exchange=ExchangeConfig(),
            multitf=MultiTFConfig(),
            optimizer=OptimizerConfig(),
            loop_interval_seconds=int(os.getenv("LOOP_INTERVAL_SECONDS", "60")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )
        insert_log(
            "WARNING",
            f"Configuracao incompleta - painel funcionara em modo observador. {exc}",
            "server",
        )

    # Worker start com timeout para nao bloquear o startup do servidor
    _worker = CTEVWorker(_settings)
    try:
        await asyncio.wait_for(_worker.start(), timeout=30.0)
    except asyncio.TimeoutError:
        logger.warning("Worker start timed out (30s) — servidor iniciando sem worker.")
        insert_log(
            "WARNING",
            "Worker start timed out — servidor iniciando sem exchange conectada.",
            "server",
        )
    except Exception as exc:
        logger.error("Worker start falhou — servidor iniciando sem worker: %s", exc)
        insert_log(
            "ERROR",
            f"Worker start falhou: {exc} — servidor iniciando sem exchange.",
            "server",
        )
    logger.info("Servidor FastAPI pronto.")


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    if _worker is not None:
        await _worker.stop()


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"


@app.get("/health", summary="Health check para UptimeRobot / liveness")
async def health() -> JSONResponse:
    return JSONResponse(
        status_code=200,
        content={"status": "ok", "service": "ctev-bot", "version": "3.0.0"},
    )


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    """Serve o painel administrativo HTML."""
    index_path = TEMPLATES_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse(
            "<h1>Painel nao encontrado</h1><p>Arquivo templates/index.html ausente.</p>",
            status_code=500,
        )
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/api/signals", summary="Lista sinais recentes")
async def api_signals(limit: int = 50) -> dict:
    limit = max(1, min(limit, 200))
    return {"signals": list_recent_signals(limit=limit)}


@app.get("/api/logs", summary="Lista logs recentes")
async def api_logs(limit: int = 50) -> dict:
    limit = max(1, min(limit, 200))
    return {"logs": list_recent_logs(limit=limit)}


@app.get("/api/risk", summary="Estado do RiskManager")
async def api_risk() -> dict:
    """Retorna o estado completo do gerenciador de risco."""
    return get_risk_manager().snapshot()


@app.get("/api/backtest/progress", summary="Progresso em tempo real do backtest")
async def api_backtest_progress() -> dict:
    """Retorna o progresso atual do backtest."""
    from backtest import get_backtest_progress
    return get_backtest_progress()


@app.get("/api/backtest/stream", summary="SSE de progresso do backtest")
async def api_backtest_stream():
    """Server-Sent Events com progresso em tempo real do backtest."""
    import json as _json
    from backtest import get_backtest_progress

    async def event_generator():
        last_json = ""
        while True:
            try:
                prog = get_backtest_progress()
                data = _json.dumps(prog, default=str)
                if data != last_json:
                    yield f"data: {data}\n\n"
                    last_json = data
                if not prog.get("running") and prog.get("phase") == "Concluido":
                    yield f"event: done\ndata: {data}\n\n"
                    break
                if _backtest_result is not None and not prog.get("running"):
                    yield f"event: done\ndata: {data}\n\n"
                    break
                await asyncio.sleep(0.25)
            except asyncio.CancelledError:
                break
            except Exception:
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/backtest", summary="Dispara backtest da estrategia CTEV")
async def api_backtest(days: int = 730, timeframe: str = None) -> dict:
    """
    Executa backtest assincrono da estrategia CTEV.

    Parametros query:
        days: dias de dados historicos (default 730)
        timeframe: timeframe dos candles (ex: '15m', '1h', '4h'). Default = config do bot.
    """
    global _backtest_task, _backtest_result, _backtest_running

    if _backtest_running and _backtest_task is not None and not _backtest_task.done():
        return {"ok": False, "error": "Backtest ja em andamento. Aguarde."}

    _backtest_running = True
    _backtest_result = None

    # Reset progress state for streaming UI
    from backtest import reset_backtest_progress
    reset_backtest_progress()

    # Resolve timeframe: usa o parametro ou fallback para config do bot
    effective_tf = timeframe or get_settings().binance.timeframe

    async def _run_backtest():
        global _backtest_result, _backtest_running
        try:
            from backtest import run_backtest
            loop = asyncio.get_event_loop()
            metrics, trades = await loop.run_in_executor(
                None, lambda: run_backtest(days=days, timeframe=effective_tf, advanced=False, regime_switching=True)
            )
            _backtest_result = {
                "ok": True,
                "metrics": metrics.to_dict(),
                "trades": [
                    {
                        "entry_ts": str(t.entry_ts),
                        "exit_ts": str(t.exit_ts),
                        "type": t.type,
                        "entry_price": t.entry_price,
                        "exit_price": t.exit_price,
                        "pnl_pct": t.pnl_pct,
                        "exit_reason": t.exit_reason,
                        "bars_held": t.bars_held,
                        "atr_percentile": t.atr_percentile,
                        "be_triggered": t.be_triggered,
                        "trailing_activated": t.trailing_activated,
                        "partial_tp_filled": t.partial_tp_filled,
                        "position_usd": t.position_usd,
                    }
                    for t in trades
                ],
            }
            insert_log(
                "INFO",
                f"Backtest concluido: {metrics.total_trades} trades, "
                f"WR={metrics.win_rate:.1f}%, PF={metrics.profit_factor:.2f}",
                "backtest",
            )
        except Exception as exc:
            logger.exception("Backtest falhou: %s", exc)
            _backtest_result = {"ok": False, "error": str(exc)}
            insert_log("ERROR", f"Backtest falhou: {exc}", "backtest")
        finally:
            _backtest_running = False

    _backtest_task = asyncio.create_task(_run_backtest())
    return {"ok": True, "message": f"Backtest iniciado ({days} dias). Consulte /api/backtest/status."}


@app.get("/api/backtest/status", summary="Status e resultado do backtest")
async def api_backtest_status() -> dict:
    """Retorna o status e resultado do backtest mais recente."""
    global _backtest_result, _backtest_running
    if _backtest_running:
        return {"status": "running", "result": None}
    if _backtest_result is not None:
        return {"status": "done", "result": _backtest_result}
    return {"status": "none", "result": None}


@app.get("/api/positions", summary="Posicoes abertas e trades fechados")
async def api_positions() -> dict:
    """Retorna posicoes abertas e historico de trades."""
    tracker = get_position_tracker()
    return tracker.snapshot()


@app.get("/api/mtf", summary="Analise multi-timeframe H4/D1")
async def api_multitf() -> dict:
    """Retorna o estado do filtro multi-timeframe."""
    return get_mtf_filter().snapshot()


@app.get("/api/mode", summary="Modo de operacao")
async def api_mode() -> dict:
    """Retorna o modo de operacao (sempre 'signal_only')."""
    return {"mode": "signal_only", "description": "Apenas analise e emissao de sinais. Nenhuma ordem e executada."}


@app.head("/api/status", summary="Health check (HEAD) para UptimeRobot")
async def api_status_head():
    """Responde ao método HEAD para compatibilidade com monitores como UptimeRobot."""
    return Response(status_code=200)


@app.get("/api/status", summary="Estado completo do bot")
async def api_status() -> dict:
    state = get_bot_state().snapshot()
    risk = get_risk_manager().snapshot()
    tracker = get_position_tracker()
    mtf = get_mtf_filter().snapshot()
    # Timeframe ativo: override > config
    active_tf = state.get("timeframe_override") or get_settings().binance.timeframe
    return {
        "bot": state,
        "risk": risk,
        "positions": tracker.snapshot(),
        "multitf": mtf,
        "executor": {"mode": "signal_only"},
        "trades_summary": get_trades_summary(),
        "symbol": get_settings().binance.symbol,
        "timeframe": active_tf,
        "timeframe_base": get_settings().binance.timeframe,
        "signals_today": count_signals_today(),
        "signals_today_by_type": count_signals_by_type_today(),
    }


@app.get("/api/trades", summary="Trades fechados com summary")
async def api_trades(limit: int = 50) -> dict:
    """Retorna trades fechados e resumo."""
    limit = max(1, min(limit, 200))
    return {
        "trades": list_recent_trades(limit=limit),
        "summary": get_trades_summary(),
    }


@app.post("/api/settings/timeframe", summary="Altera timeframe em runtime (sem restart)")
async def api_set_timeframe(timeframe: str) -> dict:
    """
    Altera o timeframe do worker e do backtest em tempo real.
    O proximo ciclo do worker ja usara o novo timeframe.
    Nao requer restart do servidor.
    """
    valid_tfs = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]
    if timeframe not in valid_tfs:
        return {"ok": False, "error": f"Timeframe invalido: '{timeframe}'. Opcoes: {valid_tfs}"}

    get_bot_state().timeframe_override = timeframe
    insert_log(
        "INFO",
        f"Timeframe alterado em runtime para {timeframe} (base: {get_settings().binance.timeframe})",
        "settings",
    )
    return {"ok": True, "timeframe": timeframe, "message": f"Timeframe alterado para {timeframe}. Proximo ciclo ja usa o novo TF."}


@app.get("/api/chart-data", summary="Dados de precos + indicadores + sinais para graficos")
async def api_chart_data(bars: int = 100) -> dict:
    """
    Retorna dados de precos com indicadores calculados e posicao dos sinais.
    Usado pelos graficos Chart.js do painel.
    """
    try:
        worker = get_worker()
        if worker.exchange is None:
            return {"ok": False, "error": "Exchange nao conectada"}
        symbol = worker.exchange_info.symbol if worker.exchange_info else get_settings().binance.symbol
        ohlcv = await worker.exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=get_settings().binance.timeframe,
            limit=max(50, min(bars, 300)),
        )
        if not ohlcv:
            return {"ok": False, "error": "Sem dados"}

        import pandas as pd
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("datetime", inplace=True)
        df.drop(columns=["timestamp"], inplace=True)

        df_ind = compute_indicators(df, timeframe=get_settings().binance.timeframe)
        signals = list_recent_signals(limit=50)

        candles = []
        for idx, row in df_ind.iterrows():
            candles.append({
                "t": idx.isoformat(),
                "o": round(float(row["open"]), 2),
                "h": round(float(row["high"]), 2),
                "l": round(float(row["low"]), 2),
                "c": round(float(row["close"]), 2),
                "v": round(float(row["volume"]), 2),
                "ema200": round(float(row["ema200"]), 2) if pd.notna(row["ema200"]) else None,
                "bb_upper": round(float(row["bb_upper"]), 2) if pd.notna(row["bb_upper"]) else None,
                "bb_lower": round(float(row["bb_lower"]), 2) if pd.notna(row["bb_lower"]) else None,
                "rsi": round(float(row["rsi"]), 2) if pd.notna(row["rsi"]) else None,
                "vol_sma20": round(float(row["volume_sma20"]), 2) if pd.notna(row["volume_sma20"]) else None,
            })

        return {"ok": True, "candles": candles, "signals": signals}
    except Exception as exc:
        logger.warning("chart-data falhou: %s", exc)
        return {"ok": False, "error": str(exc)}


# Mount de arquivos estaticos (se houver necessidade futura de CSS/JS externos)
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
