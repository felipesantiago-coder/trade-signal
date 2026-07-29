"""
server.py
---------
Servidor web FastAPI que expõe:
- GET  /               → Painel administrativo (HTML)
- GET  /health         → 200 OK (para UptimeRobot / liveness probes)
- GET  /api/status     → Estado completo do bot + stats
- GET  /api/signals    → Lista sinais recentes
- GET  /api/logs       → Lista logs recentes
- POST /api/start      → Ativa o bot (running=True)
- POST /api/stop       → Pausa o bot (running=False)

O servidor inicializa o background worker (CTEVWorker) no startup e o
encerra graciosamente no shutdown.
"""

from __future__ import annotations

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

logger = logging.getLogger("ctev.server")

# ------------------------------------------------------------------
# App + estado global do worker
# ------------------------------------------------------------------
app = FastAPI(
    title="CTEV Trading Bot — Painel Administrativo",
    description="Bot de sinais CTEV para BTC/USD 1H com painel web.",
    version="1.1.0",
)

_worker: Optional[CTEVWorker] = None
_settings: Optional[Settings] = None


def get_worker() -> CTEVWorker:
    if _worker is None:
        raise RuntimeError("Worker não inicializado.")
    return _worker


def get_settings() -> Settings:
    if _settings is None:
        raise RuntimeError("Settings não carregado.")
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
        # Em nuvem gratuita, mesmo sem .env completo o servidor deve subir
        # para que o usuário possa ver o painel. Criamos settings "vazios".
        from config import BinanceConfig, Settings as S, TelegramConfig
        _settings = S(
            telegram=TelegramConfig(token="", chat_id=""),
            binance=BinanceConfig(
                api_key=None,
                api_secret=None,
                symbol=os.getenv("BINANCE_SYMBOL", "BTC/USDT"),
                timeframe=os.getenv("BINANCE_TIMEFRAME", "1h"),
            ),
            loop_interval_seconds=int(os.getenv("LOOP_INTERVAL_SECONDS", "60")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )
        insert_log(
            "WARNING",
            f"Configuração incompleta — painel funcionará em modo observador. {exc}",
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
    """
    Retorna 200 OK com timestamp. Use este endpoint no UptimeRobot para
    evitar que a instância gratuita (Render/Koyeb) durma.
    """
    return JSONResponse(
        status_code=200,
        content={"status": "ok", "service": "ctev-bot", "version": "1.1.0"},
    )


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    """Serve o painel administrativo HTML."""
    index_path = TEMPLATES_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse(
            "<h1>Painel não encontrado</h1><p>Arquivo templates/index.html ausente.</p>",
            status_code=500,
        )
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/api/status", summary="Estado completo do bot")
async def api_status() -> dict:
    state = get_bot_state().snapshot()
    return {
        "bot": state,
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


# Mount de arquivos estáticos (se houver necessidade futura de CSS/JS externos)
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
