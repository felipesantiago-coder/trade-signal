# 🤖 Bot CTEV — Confluência de Tendência e Exaustão Volumétrica

Sistema de sinalização de trading automatizado para **BTC/USD** em timeframe de **1 hora**, operando **LONG (compra)** e **SHORT (venda)** com confirmação institucional por volume.

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

```
ctev-bot/
├── main.py            # Ponto de entrada, loop assíncrono e controle de tempo
├── indicators.py      # Cálculo dos indicadores (EMA, BB, RSI, Vol SMA, ATR) com pandas-ta
├── strategy.py        # Validação das condições LONG/SHORT e cálculo de SL/TP
├── notifier.py        # Envio de sinais via Telegram (python-telegram-bot)
├── config.py          # Carregamento de variáveis de ambiente (.env)
├── requirements.txt   # Dependências Python
├── .env.example       # Template de variáveis de ambiente
├── .gitignore
└── README.md
```

### Fluxo de execução

```
[Binance API (ccxt)] → fetch 300 candles 1H
        ↓
[indicators.py] → EMA200, BB(20,2), RSI14, VolSMA20, ATR14
        ↓
[strategy.py] → avalia apenas o último candle FECHADO
        ↓              (LONG / SHORT / nenhum)
[notifier.py] → envia mensagem 🟢/🔴 para o Telegram
        ↓
aguarda 60s e repete
```

O bot:
- Roda em loop infinito assíncrono
- Verifica a cada 60 segundos se um novo candle de 1H foi fechado
- Só avalia o candle já fechado (nunca o em formação)
- Não reprocessa o mesmo candle (controle por timestamp)

---

## 🚀 Instalação

### Pré-requisitos

- Python **3.10+**
- Conta na **Binance** (API key opcional para dados públicos, mas recomendada)
- Bot do **Telegram** criado via [@BotFather](https://t.me/BotFather)

### Passo a passo

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

# 5. Execute o bot
python main.py
```

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

## 🛠️ Execução em produção

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
