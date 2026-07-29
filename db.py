"""
db.py
------
Camada de persistência leve usando SQLite em memória (:memory:).

Em serviços de hospedagem gratuita (Render, Koyeb) o sistema de arquivos é
efêmero — arquivos locais são perdidos a cada restart. Por isso usamos um
banco SQLite em memória, mantido vivo pelo processo enquanto ele existir.
Isto é suficiente para exibir o histórico RECENTE de sinais no painel,
que é o objetivo do MVP.

Tabelas:
    - signals: sinais LONG/SHORT gerados
    - logs:    entradas de log de ações do sistema

A conexão é compartilhada entre threads via check_same_thread=False e
protegida por um asyncio.Lock criado pelo caller (servidor) para evitar
concorrência entre o worker e os endpoints HTTP.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, List, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Conexão singleton em memória
# ------------------------------------------------------------------
_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    """Retorna (ou cria) a conexão singleton com SQLite em memória."""
    global _conn
    if _conn is None:
        with _lock:
            if _conn is None:
                _conn = sqlite3.connect(
                    ":memory:",
                    check_same_thread=False,
                    isolation_level=None,  # autocommit
                )
                _conn.row_factory = sqlite3.Row
                _init_schema(_conn)
                logger.info("SQLite em memória inicializado.")
    return _conn


def _init_schema(conn: sqlite3.Connection) -> None:
    """Cria tabelas se não existirem."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS signals (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            ts            TEXT    NOT NULL,   -- ISO8601 UTC
            candle_ts     TEXT,                -- timestamp do candle que gerou o sinal
            type          TEXT    NOT NULL,    -- 'LONG' | 'SHORT'
            symbol        TEXT    NOT NULL,
            entry_price   REAL    NOT NULL,
            stop_loss     REAL    NOT NULL,
            take_profit   REAL    NOT NULL,
            atr           REAL,
            rsi           REAL,
            ema200        REAL,
            volume        REAL,
            volume_sma20  REAL,
            bb_lower      REAL,
            bb_upper      REAL,
            notified      INTEGER DEFAULT 0    -- 0=não enviado ao Telegram, 1=enviado
        );

        CREATE TABLE IF NOT EXISTS logs (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            ts      TEXT    NOT NULL,   -- ISO8601 UTC
            level   TEXT    NOT NULL,    -- INFO|WARNING|ERROR|DEBUG
            source  TEXT,                -- módulo/origem
            message TEXT    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(ts DESC);
        CREATE INDEX IF NOT EXISTS idx_logs_ts    ON logs(ts DESC);
        """
    )


# ------------------------------------------------------------------
# Helpers de conversão
# ------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {k: row[k] for k in row.keys()}


# ------------------------------------------------------------------
# API pública — Sinais
# ------------------------------------------------------------------
def insert_signal(signal_data: dict) -> int:
    """
    Insere um sinal na tabela signals.
    `signal_data` deve conter as chaves compatíveis com strategy.Signal.to_dict().
    """
    conn = _get_conn()
    with _lock:
        cur = conn.execute(
            """
            INSERT INTO signals (
                ts, candle_ts, type, symbol,
                entry_price, stop_loss, take_profit,
                atr, rsi, ema200, volume, volume_sma20, bb_lower, bb_upper, notified
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _now_iso(),
                str(signal_data.get("timestamp", "")),
                signal_data["type"],
                signal_data.get("symbol", "BTC/USDT"),
                float(signal_data["entry_price"]),
                float(signal_data["stop_loss"]),
                float(signal_data["take_profit"]),
                float(signal_data.get("atr", 0) or 0),
                float(signal_data.get("rsi", 0) or 0),
                float(signal_data.get("ema200", 0) or 0),
                float(signal_data.get("volume", 0) or 0),
                float(signal_data.get("volume_sma20", 0) or 0),
                float(signal_data.get("bb_lower", 0) or 0),
                float(signal_data.get("bb_upper", 0) or 0),
                int(signal_data.get("notified", 0)),
            ),
        )
        return cur.lastrowid


def list_recent_signals(limit: int = 50) -> List[dict]:
    """Retorna os N sinais mais recentes (default 50)."""
    conn = _get_conn()
    with _lock:
        cur = conn.execute(
            "SELECT * FROM signals ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [_row_to_dict(r) for r in cur.fetchall()]


def count_signals_today() -> int:
    """Conta quantos sinais foram gerados hoje (UTC)."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn = _get_conn()
    with _lock:
        cur = conn.execute(
            "SELECT COUNT(*) AS c FROM signals WHERE substr(ts, 1, 10) = ?",
            (today,),
        )
        return int(cur.fetchone()["c"])


def count_signals_by_type_today() -> dict:
    """Retorna {'LONG': x, 'SHORT': y} para hoje."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn = _get_conn()
    with _lock:
        cur = conn.execute(
            "SELECT type, COUNT(*) AS c FROM signals WHERE substr(ts, 1, 10) = ? GROUP BY type",
            (today,),
        )
        result = {"LONG": 0, "SHORT": 0}
        for row in cur.fetchall():
            result[row["type"]] = int(row["c"])
        return result


# ------------------------------------------------------------------
# API pública — Logs
# ------------------------------------------------------------------
def insert_log(level: str, message: str, source: Optional[str] = None) -> int:
    conn = _get_conn()
    with _lock:
        cur = conn.execute(
            "INSERT INTO logs (ts, level, source, message) VALUES (?, ?, ?, ?)",
            (_now_iso(), level.upper(), source or "", message),
        )
        return cur.lastrowid


def list_recent_logs(limit: int = 50) -> List[dict]:
    conn = _get_conn()
    with _lock:
        cur = conn.execute(
            "SELECT * FROM logs ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [_row_to_dict(r) for r in cur.fetchall()]


# ------------------------------------------------------------------
# Utilitário para reset (útil em testes)
# ------------------------------------------------------------------
def reset_db() -> None:
    """Limpa todas as tabelas. Use com cautela."""
    conn = _get_conn()
    with _lock:
        conn.execute("DELETE FROM signals")
        conn.execute("DELETE FROM logs")
    logger.info("Banco em memória resetado.")
