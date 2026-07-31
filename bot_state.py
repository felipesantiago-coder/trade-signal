"""
bot_state.py
------------
Estado compartilhado entre o background worker (loop de trading) e o
servidor web (FastAPI). Mantém em memória:

- `running`       : flag booleana (True = bot ativo, False = pausado)
- `last_check`    : timestamp ISO da última verificação do worker
- `last_signal_ts`: timestamp ISO do último sinal gerado
- `started_at`    : quando o processo começou
- `cycle_count`   : quantos ciclos já foram executados
- `error_count`   : quantas exceções ocorreram desde o start

Acesso thread-safe: a flag `running` é atômica em Python (GIL), mas usamos
um Lock explícito para as operações compostas (ex.: snapshot completo).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class BotState:
    """Estado global do bot. Singleton (ver `get_bot_state()`)."""
    running: bool = True  # Bot inicia ATIVO por padrão
    started_at: str = field(default_factory=_now_iso)
    last_check: Optional[str] = None
    last_signal_ts: Optional[str] = None
    cycle_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    last_status_message: str = "Inicializando..."
    # Runtime overrides (mutáveis sem restart)
    timeframe_override: Optional[str] = None

    def snapshot(self) -> dict:
        """Retorna um dicionário imutável com o estado atual (para JSON)."""
        return {
            "running": self.running,
            "started_at": self.started_at,
            "last_check": self.last_check,
            "last_signal_ts": self.last_signal_ts,
            "cycle_count": self.cycle_count,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "last_status_message": self.last_status_message,
            "timeframe_override": self.timeframe_override,
            "now": _now_iso(),
        }


# Singleton module-level
_instance: Optional[BotState] = None
_lock = threading.Lock()


def get_bot_state() -> BotState:
    """Retorna a instância única de BotState."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = BotState()
    return _instance
