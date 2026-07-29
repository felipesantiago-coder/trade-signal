# 🤖 Bot CTEV — Confluência de Tendência e Exaustão Volumétrica

Sistema de sinalização de trading automatizado para **BTC/USD** em timeframe de **1 hora**, operando **LONG (compra)** e **SHORT (venda)** com confirmação institucional por volume, **com painel administrativo web** e pronto para deploy 24/7 em nuvem gratuita (Render/Koyeb).

> ⚠️ **Aviso de risco**: Este software é fornecido apenas para fins educacionais e de pesquisa. Trading de criptomoedas envolve risco substancial de perda. Não constitui aconselhamento financeiro. Teste exaustivamente em paper-trading antes de qualquer uso com capital real.

---

## 📐 Estratégia

A estratégia **CTEV** combina 4 confirmações independentes antes de emitir um sinal:

### Indicadores utilizados

| Indicador | Parâmetro |
|---|---|
| EMA | 200 períodos |
| Bollinger Bands | 20, 2 desvios |
| RSI | 14 períodos |
| Volume SMA | 20 períodos |
| ATR | 14 períodos |

### Condições de entrada — 🟢 LONG (compra)

1. `Close > EMA(200)` — tendência de alta macro
2. `Low` ou `Close <= Banda Inferior de Bollinger` — pullback/correção
3. `RSI(14) < 35` — exaustão baixa
4. `Volume atual > Volume SMA(20) × 1.5` — confirmação institucional

**Gestão de risco LONG**
- Stop Loss = `Entry - 1.5 × ATR(14)`
- Take Profit 1 = `Entry + 2.0 × ATR(14)`

### Condições de entrada — 🔴 SHORT (venda)

1. `Close < EMA(200)` — tendência de baixa macro
2. `High` ou `Close >= Banda Superior de Bollinger` — pullback para cima
3. `RSI(14) > 65` — exaustão alta
4. `Volume atual > Volume SMA(20) × 1.5` — confirmação institucional

**Gestão de risco SHORT**
- Stop Loss = `Entry + 1.5 × ATR(14)`
- Take Profit 1 = `Entry - 2.0 × ATR(14)`

> Razão risco:retorno padrão: **1 : 1.33** (1.5 ATR de risco para 2.0 ATR de retorno).

---

## 🏗️ Arquitetura

A partir da versão 1.1.0 o sistema roda um **servidor web (FastAPI)** e um **background worker (loop de trading)** no mesmo processo, em paralelo, sem um bloquear o outro.

```
ctev-bot/
├── main.py               # Entry point: sobe Uvicorn na porta $PORT
├── server.py             # App FastAPI: /health, /, /api/status, /api/start|stop, /api/signals, /api/logs
├── bot_worker.py         # Background task: loop CTEV (respeita flag running)
├── bot_state.py          # Estado compartilhado (running, last_check, ciclo, erros)
├── db.py                 # SQLite em memória (sinais + logs)
├── indicators.py         # EMA200, BB(20,2), RSI14, VolSMA20, ATR14 (pandas-ta)
├── strategy.py           # Validação LONG/SHORT + cálculo SL/TP
├── notifier.py           # Envio ao Telegram (python-telegram-bot)
├── config.py             # Carregamento de .env
├── templates/
│   └── index.html        # Painel admin (Tailwind via CDN, polling 5s)
├── render.yaml           # Deploy no Render (Blueprint)
├── Procfile              # Deploy alternativo (Koyeb, Heroku-like)
├── requirements.txt
├── .env.example
└── README.md
```

### Fluxo de execução

```
                  ┌────────────────────────────────────┐
                  │        main.py  (Uvicorn)          │
                  │  porta: $PORT  (default 8000)      │
                  └─────────────┬──────────────────────┘
                                │
              ┌─────────────────┴─────────────────┐
              ▼                                   ▼
   ┌────────────────────┐              ┌──────────────────────┐
   │   server.py        │              │   bot_worker.py      │
   │   (FastAPI)        │  compart.    │   (asyncio.Task)     │
   │                    │◄────────────►│                      │
   │ • /health          │  bot_state   │ Loop infinito:       │
   │ • /                │   .running   │  fetch candles 1H    │
   │ • /api/status      │              │  → indicators        │
   │ • /api/signals     │  db.py       │  → strategy          │
   │ • /api/logs        │   (SQLite    │  → notifier(Telegram)│
   │ • /api/start|stop  │    in-mem)   │  → db.insert_signal  │
   └─────────┬──────────┘              └──────────┬───────────┘
             │                                    │
             ▼                                    ▼
   Painel HTML (Tailwind)              Binance API (ccxt)
   polling 5s                           + Telegram Bot API
```

### Características operacionais

- **Loop paralelo sem bloqueio**: o servidor web responde mesmo enquanto o worker busca candles
- **Pause/Resume via painel**: altera `bot_state.running` em memória
- **Apenas candle fechado**: o worker só avalia o último candle quando `now >= close_ts`
- **Sem reprocessamento**: controle por timestamp evita duplicar sinais
- **Tolerante a falhas**: exceções no worker são logadas mas não derrubam o processo
- **SQLite em memória**: ideal para filesystem efêmero dos planos gratuitos

---

## 🚀 Instalação

### Pré-requisitos

- Python **3.10+**
- Conta na **Binance** (API key opcional para dados públicos, mas recomendada)
- Bot do **Telegram** criado via [@BotFather](https://t.me/BotFather)

### Passo a passo (local)

```bash
# 1. Clone o repositório
git clone https://github.com/felipesantiago-coder/trade-signal.git
cd trade-signal

# 2. Crie e ative o virtualenv
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure as variáveis de ambiente
cp .env.example .env
# Edite o .env com seu TOKEN do Telegram, CHAT_ID e chaves da Binance

# 5. Execute o servidor + worker
python main.py
# ou: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Abra o painel em **http://localhost:8000**

---

## 🔑 Configuração do ambiente (`.env`)

| Variável | Descrição | Obrigatório |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Token do bot criado no @BotFather | ✅ |
| `TELEGRAM_CHAT_ID` | ID do chat/canal que receberá os sinais | ✅ |
| `BINANCE_API_KEY` | API key da Binance | ⚠️ recomendado |
| `BINANCE_API_SECRET` | API secret da Binance | ⚠️ recomendado |
| `BINANCE_SYMBOL` | Par de trading (default: `BTC/USDT`) | ❌ |
| `BINANCE_TIMEFRAME` | Timeframe (default: `1h`) | ❌ |
| `LOOP_INTERVAL_SECONDS` | Intervalo entre ciclos (default: `60`) | ❌ |
| `LOG_LEVEL` | Nível de log: `DEBUG`/`INFO`/`WARNING`/`ERROR` | ❌ |
| `PORT` | Porta HTTP (default: `8000`) — obrigatória em nuvem | ❌ |
| `HOST` | Bind host (default: `0.0.0.0`) | ❌ |

### Como obter o `TELEGRAM_CHAT_ID`

1. Mande uma mensagem qualquer para o seu bot no Telegram.
2. Acesse no navegador:
   ```
   https://api.telegram.org/bot<SEU_TOKEN>/getUpdates
   ```
3. Procure por `"chat":{"id":XXXXXXXXX}` no JSON retornado. Esse número é o `chat_id`.

---

## 📱 Exemplo de mensagem no Telegram

```
🟢 SINAL DE LONG — COMPRA
━━━━━━━━━━━━━━━━━━
📊 Par: BTC/USDT
⏱ Timeframe: 1H
🕐 Candle: 2026-01-15 13:00:00+00:00

💰 Entrada: 42150.00
🛑 Stop Loss: 41820.50
🎯 Take Profit 1: 42590.50
⚖️ Risco:Retorno: 1.33 : 1

━━━━━━━━━━━━━━━━━━
📋 Confirmações:
• EMA200: 41800.00
• RSI(14): 28.45
• ATR(14): 219.67
• Volume: 1,245.00
• Vol SMA20: 712.30
• BB Lower: 41890.00
• BB Upper: 42500.00

Estratégia CTEV — Confluência de Tendência e Exaustão Volumétrica
```

---

## ☁️ Deploy em nuvem gratuita (24/7)

O servidor escuta na porta definida por `PORT` e expõe `/health` para que serviços gratuitos (Render/Koyeb) não entrem em modo sleep.

### Render (recomendado)

1. Faça push do código para o GitHub (já configurado)
2. Acesse https://render.com → **New +** → **Blueprint**
3. Selecione o repositório `felipesantiago-coder/trade-signal`
4. O Render detecta `render.yaml` automaticamente
5. Em **Environment**, adicione as variáveis sensíveis (não versionadas):
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
   - `BINANCE_API_KEY` (opcional)
   - `BINANCE_API_SECRET` (opcional)
6. Deploy → aguarde o build → acesse `https://ctev-bot.onrender.com`

### Koyeb / Heroku-like

Use o `Procfile` incluído:
```
web: uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
```

### Manter a instância acordada (UptimeRobot)

1. Crie conta gratuita em https://uptimerobot.com
2. **Add New Monitor** → tipo **HTTP(s)**
3. URL: `https://<sua-app>.onrender.com/health`
4. Intervalo: **5 minutos**

Isso evita que o Render durma a instância gratuita após 15 min de inatividade.

---

## 🛠️ Execução em produção (VPS / local)

Recomenda-se usar `systemd`, `tmux` ou `screen` para manter o bot rodando:

### tmux (rápido)

```bash
tmux new -s ctev
python main.py
# Ctrl+B, D para desanexar
# tmux attach -t ctev para voltar
```

### systemd (servidor Linux)

Crie `/etc/systemd/system/ctev-bot.service`:

```ini
[Unit]
Description=CTEV Trading Signal Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/trade-signal
ExecStart=/home/ubuntu/trade-signal/.venv/bin/python main.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ctev-bot
sudo journalctl -u ctev-bot -f   # ver logs
```

---

## 🧪 Testes rápidos (sem Telegram)

Para validar apenas o cálculo de indicadores e a lógica de estratégia, você pode usar o Python interativo:

```python
import ccxt, pandas as pd
from indicators import compute_indicators
from strategy import evaluate_signal

ex = ccxt.binance()
ohlcv = ex.fetch_ohlcv('BTC/USDT', '1h', limit=300)
df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
df['datetime'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
df.set_index('datetime', inplace=True); df.drop(columns=['ts'], inplace=True)

df_ind = compute_indicators(df)
signal = evaluate_signal(df_ind)
print(signal)
```

---

## ⚙️ Personalização

Os parâmetros da estratégia estão definidos como constantes em `strategy.py`:

```python
RSI_LONG_THRESHOLD = 35.0
RSI_SHORT_THRESHOLD = 65.0
VOLUME_MULTIPLIER = 1.5
SL_ATR_MULT = 1.5
TP_ATR_MULT = 2.0
```

Ajuste conforme seu backtest. Para um sistema mais robusto, considere mover estes parâmetros para o `.env` ou um arquivo `params.yaml`.

---

## 🖥️ Painel administrativo

Acesse `/` para ver o painel com:

- **Status do bot** (Online/Pausado) com indicador pulsante
- **KPIs**: sinais hoje, breakdown LONG/SHORT, ciclos executados, erros, última verificação
- **Tabela de sinais recentes** (data, tipo, entrada, SL, TP, RSI, status Telegram)
- **Log do sistema** em tempo real (INFO/WARNING/ERROR)
- **Botões Iniciar/Pausar** funcionais (alteram flag em memória)
- **Polling automático a cada 5 segundos**

Interface responsiva com Tailwind CSS via CDN, dark mode profissional.

Endpoints REST disponíveis:

| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | 200 OK para liveness probes (UptimeRobot) |
| GET | `/` | Painel HTML |
| GET | `/api/status` | Estado completo do bot |
| GET | `/api/signals?limit=50` | Sinais recentes |
| GET | `/api/logs?limit=50` | Logs recentes |
| POST | `/api/start` | Ativa o bot |
| POST | `/api/stop` | Pausa o bot |
| GET | `/docs` | Documentação OpenAPI interativa |

---

## 📝 Roadmap sugerido

- [ ] Backtest histórico com `vectorbt` ou `backtesting.py`
- [ ] Múltiplos pares (ETH/USDT, SOL/USDT, etc.)
- [ ] Take Profit 2 e 3 (escala parcial)
- [ ] Trailing stop baseado em ATR
- [ ] Dashboard web (FastAPI + React) para acompanhar sinais
- [ ] Integração com exchange para execução automática (paper trading)

---

## 📄 Licença

MIT License. Use livremente, por sua conta e risco.

---

## 🧭 Aviso final

Este bot **apenas sinaliza** oportunidades baseadas na estratégia CTEV. Ele **não executa ordens**. Toda decisão de trade é sua. Sempre valide sinais em ambiente de simulação antes de arriscar capital real.
