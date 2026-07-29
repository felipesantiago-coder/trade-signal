"""
main.py
-------
Ponto de entrada da aplicação CTEV Bot (versão 1.1.0 — web + worker).

Inicia o servidor FastAPI via Uvicorn, escutando na porta definida pela
variável de ambiente `PORT` (exigência de serviços gratuitos como Render).

O FastAPI, no seu lifecycle (startup/shutdown), inicializa e encerra o
background worker (CTEVWorker) que roda o loop de trading em paralelo,
sem bloquear o servidor web.

Variáveis de ambiente relevantes:
    PORT                : porta HTTP (default 8000) — obrigatória em Render/Koyeb
    LOG_LEVEL           : DEBUG | INFO | WARNING | ERROR  (default INFO)
    + variáveis do bot  : ver .env.example

Execução local:
    python main.py
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload   # dev
"""

from __future__ import annotations

import logging
import os
import sys

import uvicorn

# Logging configurado antes de importar server (que importa db etc.)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ctev.main")

# Importa a app FastAPI (que carrega worker no startup)
from server import app  # noqa: E402


def main() -> None:
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(
        "Iniciando CTEV Bot — host=%s port=%d log_level=%s",
        host, port, LOG_LEVEL,
    )

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=LOG_LEVEL.lower(),
        access_log=False,  # Reduz ruído em produção; ative se precisar debugar
    )


if __name__ == "__main__":
    main()
