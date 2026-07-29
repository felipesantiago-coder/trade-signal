#!/bin/bash
# ============================================================
# CTEV Bot — Entrypoint
# ============================================================
# Configura variáveis de ambiente padrão e inicia o Uvicorn.
# ============================================================

set -e

# Defaults
export PORT="${PORT:-8000}"
export HOST="${HOST:-0.0.0.0}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export EXCHANGE_ID="${EXCHANGE_ID:-binance}"
export EXCHANGE_DRY_RUN="${EXCHANGE_DRY_RUN:-true}"

echo "=========================================="
echo "  CTEV Bot — Confluência de Tendência"
echo "  e Exaustão Volumétrica"
echo "=========================================="
echo "  Exchange  : ${EXCHANGE_ID}"
echo "  Dry Run   : ${EXCHANGE_DRY_RUN}"
echo "  Port      : ${PORT}"
echo "  Log Level : ${LOG_LEVEL}"
echo "=========================================="

# Inicia o servidor (que também dispara o worker no startup)
exec uvicorn main:app \
    --host "${HOST}" \
    --port "${PORT}" \
    --log-level "${LOG_LEVEL,,}" \
    --no-access-log
