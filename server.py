"""
server.py
---------
Servidor web FastAPI que expoe:
- GET  /               -> Painel administrativo (HTML)
- GET  /health         -> 200 OK (para UptimeRobot / liveness probes)
- GET  /api/status     -> Estado completo do bot + stats
- GET  /api/signals    -> Lista sinais recentes
- GET  /api/logs       -> Lista logs recentes
- GET  /api/risk       -> Estado do RiskManager
- POST /api/start      -> Ativa o bot (running=True)
- POST /api/stop       -> Pausa o bot (running=False)
- POST /api/kill       -> Kill Switch manual (bloqueia tudo)
- POST /api/resurrect  -> Desativa Kill Switch
- POST /api/backtest   -> Dispara backtest assincrono e retorna resultados
- GET  /api/backtest/status -> Status do backtest em andamento

O servidor inicializa o background worker (CTEVWorker) no startup e o
encerra graciosamente no shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from bot_state import get_bot_state
from bot_worker import CTEVWorker
from config import Settings, load_settings
from db import (
    count_signals_by_type_today,
    count_signals_today,
    insert_log,
    list_recent_logs,
    list_recent_signals,
)
from risk_manager import get_risk_manager

logger = logging.getLogger("ctev.server")

# ------------------------------------------------------------------
# App + estado global do worker
# ------------------------------------------------------------------
app = FastAPI(
    title="CTEV Trading Bot - Painel Administrativo",
    description="Bot de sinais CTEV para BTC/USD 1H com painel web, "
                "RiskManager e Backtesting.",
    version="2.0.0",
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
    """Carrega settings e dispara o background worker."""
    global _worker, _settings
    try:
        _settings = load_settings()
    except RuntimeError as exc:
        logger.error("Falha ao carregar settings: %s", exc)
        from config import BinanceConfig, Settings as S, TelegramConfig, RiskConfig
        _settings = S(
            telegram=TelegramConfig(token="", chat_id=""),
            binance=BinanceConfig(
                api_key=None,
                api_secret=None,
                symbol=os.getenv("BINANCE_SYMBOL", "BTC/USDT"),
                timeframe=os.getenv("BINANCE_TIMEFRAME", "1h"),
            ),
            risk=RiskConfig(),
            loop_interval_seconds=int(os.getenv("LOOP_INTERVAL_SECONDS", "60")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )
        insert_log(
            "WARNING",
            f"Configuracao incompleta - painel funcionara em modo observador. {exc}",
            "server",
        )

    _worker = CTEVWorker(_settings)
    await _worker.start()
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
        content={"status": "ok", "service": "ctev-bot", "version": "2.0.0"},
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


@app.get("/api/status", summary="Estado completo do bot")
async def api_status() -> dict:
    state = get_bot_state().snapshot()
    risk = get_risk_manager().snapshot()
    return {
        "bot": state,
        "risk": risk,
        "symbol": get_settings().binance.symbol,
        "timeframe": get_settings().binance.timeframe,
        "signals_today": count_signals_today(),
        "signals_today_by_type": count_signals_by_type_today(),
    }


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


@app.post("/api/start", summary="Ativa o bot")
async def api_start() -> dict:
    state = get_bot_state()
    state.running = True
    state.last_status_message = "Reativado pelo painel"
    insert_log("INFO", "Bot reativado via painel.", "server")
    return {"ok": True, "running": True}


@app.post("/api/stop", summary="Pausa o bot")
async def api_stop() -> dict:
    state = get_bot_state()
    state.running = False
    state.last_status_message = "Pausado pelo painel"
    insert_log("INFO", "Bot pausado via painel.", "server")
    return {"ok": True, "running": False}


@app.post("/api/kill", summary="Kill Switch manual - bloqueia todos os sinais")
async def api_kill() -> dict:
    """Ativa o kill switch manual. Nenhum sinal sera gerado ate resurrect."""
    risk = get_risk_manager()
    risk.kill()
    state = get_bot_state()
    state.running = False
    state.last_status_message = "KILL SWITCH ATIVADO"
    insert_log("CRITICAL", "Kill Switch manual ativado via painel.", "server")
    return {"ok": True, "killed": True}


@app.post("/api/resurrect", summary="Desativa Kill Switch e retoma operacoes")
async def api_resurrect() -> dict:
    """Desativa o kill switch e reseta contadores de perda."""
    risk = get_risk_manager()
    risk.resurrect()
    state = get_bot_state()
    state.running = True
    state.last_status_message = "Kill Switch desativado - bot retomando"
    insert_log("INFO", "Kill Switch desativado via painel. Bot retomando.", "server")
    return {"ok": True, "killed": False}


@app.post("/api/backtest", summary="Dispara backtest da estrategia CTEV")
async def api_backtest(days: int = 365) -> dict:
    """
    Executa backtest assincrono da estrategia CTEV.

    Parametros query:
        days: dias de dados historicos (default 365)
    """
    global _backtest_task, _backtest_result, _backtest_running

    if _backtest_running and _backtest_task is not None and not _backtest_task.done():
        return {"ok": False, "error": "Backtest ja em andamento. Aguarde."}

    _backtest_running = True
    _backtest_result = None

    async def _run_backtest():
        global _backtest_result, _backtest_running
        try:
            from backtest import run_backtest
            metrics, trades = run_backtest(days=days)
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


# Mount de arquivos estaticos (se houver necessidade futura de CSS/JS externos)
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
