#!/bin/bash
# ============================================================
# CTEV Bot — Script de Deploy (Oracle Cloud / qualquer VPS)
# ============================================================
# Execute este script após SSH na sua instância.
# Uso: bash deploy.sh
# ============================================================

set -e

echo "=============================================="
echo "  CTEV Bot — Deploy Script"
echo "=============================================="

# ---- 1. Verifica se é root ----
if [ "$EUID" -ne 0 ]; then
    echo "Execute como root: sudo bash deploy.sh"
    exit 1
fi

# ---- 2. Instala Docker (se necessário) ----
if ! command -v docker &> /dev/null; then
    echo ""
    echo "[1/5] Instalando Docker..."
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
    echo "Docker instalado com sucesso."
else
    echo ""
    echo "[1/5] Docker já instalado ($(docker --version))."
fi

# ---- 3. Instala Docker Compose (se necessário) ----
if ! docker compose version &> /dev/null; then
    echo ""
    echo "[2/5] Instalando Docker Compose..."
    apt-get update && apt-get install -y docker-compose-plugin
    echo "Docker Compose instalado."
else
    echo ""
    echo "[2/5] Docker Compose já instalado."
fi

# ---- 4. Clona o repositório (se necessário) ----
DEPLOY_DIR="/opt/ctev-bot"
echo ""
echo "[3/5] Configurando repositório em ${DEPLOY_DIR}..."

if [ -d "$DEPLOY_DIR" ]; then
    echo "Diretório existe. Atualizando..."
    cd "$DEPLOY_DIR"
    git pull origin main
else
    echo "Clonando repositório..."
    git clone https://github.com/felipesantiago-coder/trade-signal.git "$DEPLOY_DIR"
    cd "$DEPLOY_DIR"
fi

# ---- 5. Cria .env a partir do exemplo (se não existir) ----
echo ""
echo "[4/5] Configurando variáveis de ambiente..."

if [ ! -f "$DEPLOY_DIR/.env" ]; then
    cp "$DEPLOY_DIR/.env.example" "$DEPLOY_DIR/.env"
    echo ""
    echo "⚠️  ARQUIVO .env CRIADO EM: ${DEPLOY_DIR}/env"
    echo "⚠️  EDITE ESTE ARQUIVO E PREENCHA SUAS CHAVES:"
    echo ""
    echo "    nano ${DEPLOY_DIR}/env"
    echo ""
    echo "    Variáveis obrigatórias:"
    echo "      - TELEGRAM_BOT_TOKEN"
    echo "      - TELEGRAM_CHAT_ID"
    echo ""
    echo "    Variáveis opcionais (para dry-run, não precisa):"
    echo "      - BINANCE_API_KEY"
    echo "      - BINANCE_API_SECRET"
    echo ""
else
    echo ".env já existe. Pulando."
fi

# ---- 6. Build e Start ----
echo ""
echo "[5/5] Fazendo build e iniciando o container..."
cd "$DEPLOY_DIR"
docker compose build --no-cache
docker compose up -d

echo ""
echo "=============================================="
echo "  DEPLOY CONCLUÍDO COM SUCESSO!"
echo "=============================================="
echo ""
echo "  Painel: http://<IP_DA_INSTANCIA>:8000"
echo "  Health: http://<IP_DA_INSTANCIA>:8000/health"
echo "  Logs:   docker logs -f ctev-bot"
echo "  Parar:  docker compose -f ${DEPLOY_DIR}/docker-compose.yml down"
echo ""
echo "  Para editar configurações:"
echo "    nano ${DEPLOY_DIR}/env"
echo "    docker compose -f ${DEPLOY_DIR}/docker-compose.yml restart"
echo ""
