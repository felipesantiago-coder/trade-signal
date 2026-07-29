# ============================================================
# CTEV Bot — Docker Image
# ============================================================
# Multi-stage build para imagem leve e segura.
# Target: Oracle Cloud Always Free ARM (Ampere A1) — Frankfurt
# ============================================================

# ---- Stage 1: Build ----
FROM python:3.11-slim AS builder

WORKDIR /build

# Instala dependências de sistema para compilar wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---- Stage 2: Runtime ----
FROM python:3.11-slim AS runtime

WORKDIR /app

# Instala apenas runtime deps (curl para healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copia pacotes Python do builder
COPY --from=builder /install /usr/local

# Copia código da aplicação
COPY . .

# Cria diretório de non-root user (segurança)
RUN useradd --create-home --shell /bin/false ctevbot && \
    chown -R ctevbot:ctevbot /app

# Porta do servidor
EXPOSE 8000

# Health check (Render/UptimeRobot compatível)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Roda como non-root
USER ctevbot

# Entrypoint
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
