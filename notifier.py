"""
notifier.py
-----------
Integracao assincrona com o Telegram usando python-telegram-bot.

MODO: Signal-Only (apenas analise, sem execucao de ordens).
Mensagens projetadas para serem intuitivas, educativas e acionaveis,
com guia passo a passo DETALHADO para montar operacoes na Binance (BTC/USDT).

V13-ROBUSTA: Versao ativa validada via Walk-Forward OOS
  (17 janelas, Sharpe 1.30, MaxDD 33.6%, Consistency 65%).
"""

from __future__ import annotations

import logging
from typing import Optional

try:
    from telegram import Bot
    from telegram.constants import ParseMode
    _HAS_TELEGRAM = True
except ImportError:
    _HAS_TELEGRAM = False
    Bot = None
    ParseMode = None

from strategy import Signal, SignalType

logger = logging.getLogger(__name__)


# ==================================================================
# Mapas de traducao e formatacao
# ==================================================================

_REGIME_PT = {
    "STRONG_UPTREND":   "Tendencia de Alta Forte",
    "WEAK_UPTREND":     "Tendencia de Alta Fraca",
    "RANGING":          "Mercado Lateral",
    "SQUEEZE":          "Compressao de Volatilidade",
    "BREAKOUT_BULL":    "Expansao de Volatilidade (Alta)",
    "BREAKOUT_BEAR":    "Expansao de Volatilidade (Baixa)",
    "WEAK_DOWNTREND":   "Tendencia de Baixa Fraca",
    "STRONG_DOWNTREND": "Tendencia de Baixa Forte",
    "HIGH_VOLATILITY":  "Volatilidade Extrema",
}

_STRATEGY_LABEL = {
    "squeeze_breakout": "Squeeze Breakout",
    "rsi_reversal": "RSI Reversal",
    "trend_follow_long": "Trend-Following (Compra)",
    "trend_follow_short": "Trend-Following (Venda)",
    "mean_reversion": "Mean-Reversion (Reversao)",
    "breakout_long": "Breakout (Alta)",
    "breakout_short": "Breakout (Baixa)",
}

_EXIT_REASON_PT = {
    "tp": "Take Profit (alvo atingido)",
    "sl": "Stop Loss (protecao acionada)",
    "timeout": "Timeout (maximo de barras sem saida)",
    "be":  "Break-Even (stop movido para entrada)",
}

_STRATEGY_EXPLANATION = {
    "squeeze_breakout": {
        "what": (
            "As Bandas de Bollinger ficaram tao comprimidas que o preco "
            "esta 'acumulando energia' — como uma mola sendo pressionada. "
            "Quando a mola se solta, o preco se move com forca em uma direcao."
        ),
        "why_now": (
            "O sistema detectou que a compressao acabou e o preco comecou "
            "a se expandir para {'CIMA (alta)' if True else 'BAIXO (baixa)'}. "
            "Isso significa que a 'mola' esta se soltando agora."
        ),
        "risk_profile": (
            "Risco controlado: se o movimento for falso, o stop loss fecha "
            "a operacao com perca pequena. Se for real, o ganho alvo e "
            "3.6x maior que o risco (relacao risco:retorno de 1:3.6)."
        ),
    },
    "rsi_reversal": {
        "what": (
            "O RSI (Indice de Forca Relativa) mede se o mercado esta "
            "sobrecomprado ou sobrevendido. Quando ele atinge extremos "
            "dentro de uma tendencia forte, ha probabilidade de reversao "
            "de curto prazo de volta para a tendencia principal."
        ),
        "why_now": (
            "O RSI atingiu uma zona de {'sobrevenda' if True else 'sobrecompra'} "
            "no contexto de uma tendencia estabelecida. O sistema identificou "
            "o ponto de reversao com SL apertado para risco minimo."
        ),
        "risk_profile": (
            "Risco baixo por operacao: entrada precisa com SL justo. "
            "Relacao risco:retorno de 1:3.1 — ganho potencial 3x maior que a perda."
        ),
    },
}


# ==================================================================
# Helpers
# ==================================================================

def _rr_ratio(signal: Signal) -> str:
    """Calcula e formata a relacao risco:retorno real."""
    risk = abs(signal.entry_price - signal.stop_loss)
    reward = abs(signal.take_profit - signal.entry_price)
    rr = reward / max(risk, 1e-9)
    return f"1:{rr:.1f}"


def _confidence_bar(confidence: float) -> str:
    """Retorna barra visual de confianca (5 blocos)."""
    filled = round(confidence * 5)
    return "⚡" * filled + "⚫" * (5 - filled)


def _pct_diff(base: float, target: float) -> float:
    """Percentual absoluto de diferenca."""
    return abs(target - base) / max(base, 1e-9) * 100


def _fmt_price(v: float) -> str:
    """Formata preco com 2 casas decimais e separador de milhar."""
    return f"${v:,.2f}"


def _fmt_btc(v: float) -> str:
    """Formata quantidade BTC com 5 casas."""
    return f"{v:.5f} BTC"


def _vol_description(atr_pct: float) -> str:
    """Descricao textual do percentile de ATR."""
    if atr_pct > 0.7:
        return "alta"
    if atr_pct > 0.3:
        return "normal"
    return "baixa"


def _strategy_explanation_text(strategy: str, is_long: bool) -> str:
    """Retorna explicacao didatica da estrategia."""
    info = _STRATEGY_EXPLANATION.get(strategy)
    if not info:
        if is_long:
            return "Estrategia de tendencia identificou ponto favoravel para compra."
        return "Estrategia de tendencia identificou ponto favoravel para venda."

    what = info["what"]
    why = info["why_now"]
    risk = info["risk_profile"]

    if strategy == "squeeze_breakout":
        why = why.replace(
            "{'CIMA (alta)' if True else 'BAIXO (baixa)'}",
            "CIMA (alta)" if is_long else "BAIXO (baixa)",
        )
    elif strategy == "rsi_reversal":
        why = why.replace(
            "{'sobrevenda' if True else 'sobrecompra'}",
            "sobrevenda" if is_long else "sobrecompra",
        )

    return f"{what}\n\n{why}\n\n{risk}"


# ==================================================================
# Analise tecnica por estrategia
# ==================================================================

def _build_analysis(
    signal: Signal, regime: str, strategy: str, confidence: float,
) -> str:
    """Secao de analise tecnica explicada de forma simples."""
    regime_pt = _REGIME_PT.get(regime, regime)
    strategy_label = _STRATEGY_LABEL.get(strategy, strategy)
    conf_bar = _confidence_bar(confidence)
    is_long = signal.type == SignalType.LONG

    # --- Explicacao da estrategia em linguagem simples ---
    explanation = _strategy_explanation_text(strategy, is_long)

    # --- Regime em linguagem simples ---
    regime_explained = {
        "STRONG_UPTREND": "O mercado esta subindo com forca — tendencia clara de alta.",
        "WEAK_UPTREND": "O mercado sobe, mas sem mucha forca — tendencia fraca.",
        "RANGING": "O mercado esta lateral, sem direcao clara — oscila entre suporte e resistencia.",
        "SQUEEZE": "As Bandas de Bollinger estao muito juntas — o preco esta 'comprimido' como uma mola.",
        "BREAKOUT_BULL": "A volatilidade esta expandindo para cima — o preco acelerou.",
        "BREAKOUT_BEAR": "A volatilidade esta expandindo para baixo — o preco caiu com forca.",
        "WEAK_DOWNTREND": "O mercado desce, mas sem mucha forca — tendencia fraca de baixa.",
        "STRONG_DOWNTREND": "O mercado esta caindo com forca — tendencia clara de baixa.",
        "HIGH_VOLATILITY": "O mercado esta muito volatil — movimentos grandes e imprevisiveis.",
    }
    regime_simple = regime_explained.get(regime, f"Regime de mercado: {regime_pt}.")

    # --- Indicadores com explicacao ---
    indicators = []
    indicators.append(f"RSI: {signal.rsi:.1f}")
    indicators.append(f"ATR: {signal.atr:,.2f}")
    if signal.adx > 0:
        indicators.append(f"ADX: {signal.adx:.1f}")
    if signal.bb_width > 0:
        indicators.append(f"BB Width: {signal.bb_width:.4f}")

    lines = [
        "",
        "═══  ✍  O QUE ESTA ACONTECENDO?  ═══",
        "",
        f"*Regime do mercado:* {regime_pt}",
        f"{regime_simple}",
        "",
        f"*Estrategia:* {strategy_label}",
        f"{explanation}",
        "",
        f"*Indicadores chave:*  {'  |  '.join(indicators)}",
    ]

    # --- Explicacao dos indicadores para leigos ---
    indicator_explainers = []
    if signal.rsi < 35:
        indicator_explainers.append(
            "ℹ RSI abaixo de 35: o ativo pode estar 'barato' (sobrevenda) — revertendo a alta"
        )
    elif signal.rsi > 65:
        indicator_explainers.append(
            "ℹ RSI acima de 65: o ativo pode estar 'caro' (sobrecompra) — revertendo a baixa"
        )
    if signal.adx > 25:
        indicator_explainers.append(
            f"ℹ ADX {signal.adx:.1f} > 25: tendencia forte confirmada (quanto maior, mais forte)"
        )
    if signal.bb_width > 0 and signal.bb_width < 0.01:
        indicator_explainers.append(
            "ℹ BB Width muito baixo: compressao detectada — breakout iminente"
        )

    if indicator_explainers:
        lines.append("")
        lines.append("*Em termos simples:*")
        lines.extend(indicator_explainers)

    # --- Volatilidade e extras ---
    extras = []
    if signal.atr_percentile > 0:
        vol = _vol_description(signal.atr_percentile)
        extras.append(f"Volatilidade: {signal.atr_percentile:.0%} ({vol})")
    if signal.macd_hist != 0:
        extras.append(f"MACD Hist: {signal.macd_hist:+.4f}")

    di_spread = signal.plus_di - signal.minus_di
    if abs(di_spread) > 0.1:
        side = ("compradores dominam" if di_spread > 0
                else "vendedores dominam")
        extras.append(f"DI spread: {di_spread:+.1f} ({side})")

    if extras:
        lines.append("")
        lines.append("  |  ".join(extras))

    # Confianca no final da analise
    lines.append("")
    lines.append(f"*Nivel de confianca:* {confidence:.0%} {conf_bar}")

    return "\n".join(lines)


# ==================================================================
# Guia Binance passo a passo (DETALHADO para iniciantes)
# ==================================================================

def _build_binance_guide(signal: Signal) -> str:
    """
    Guia passo a passo EXTREMAMENTE detalhado para montar a operacao na Binance.
    Adaptado para LONG ou SHORT.
    Cada passo inclui: o que fazer, onde clicar, e por que.
    """
    is_long = signal.type == SignalType.LONG

    # Direcao
    if is_long:
        direction_word = "COMPRAR (Long)"
        direction_btn = "*Comprar/Long*  (botao verde)"
        direction_emoji = "📈"
        sl_direction = "abaixo do preco de entrada"
        tp_direction = "acima do preco de entrada"
        explain_direction = (
            "Voce vai APOSTAR que o Bitcoin vai SUBIR de preco. "
            "Se subir ate o Take Profit, voce lucra. "
            "Se cair ate o Stop Loss, a operacao fecha com prejuizo controlado."
        )
    else:
        direction_word = "VENDER (Short)"
        direction_btn = "*Vender/Short*  (botao vermelho)"
        direction_emoji = "📉"
        sl_direction = "acima do preco de entrada"
        tp_direction = "abaixo do preco de entrada"
        explain_direction = (
            "Voce vai APOSTAR que o Bitcoin vai CAIR de preco. "
            "Se cair ate o Take Profit, voce lucra. "
            "Se subir ate o Stop Loss, a operacao fecha com prejuizo controlado."
        )

    # Calculos de posicao
    risk_per_unit = abs(signal.entry_price - signal.stop_loss)
    reward_per_unit = abs(signal.take_profit - signal.entry_price)
    sl_pct = _pct_diff(signal.entry_price, signal.stop_loss)
    tp_pct = _pct_diff(signal.entry_price, signal.take_profit)
    rr = _rr_ratio(signal)

    sizing_rows = []
    for capital in [500, 1000, 5000, 10000]:
        risk_amt = capital * 0.005  # 0.5% do capital
        qty = risk_amt / max(risk_per_unit, 1e-9)
        profit = qty * reward_per_unit
        loss = qty * risk_per_unit
        sizing_rows.append(
            f"  ${capital:>7,}  |  ${risk_amt:>6,.2f}  "
            f"|  {qty:.5f}  |  +${profit:>7,.2f}  |  -${loss:>6,.2f}"
        )

    lines = [
        "",
        "═══  📱  COMO MONTAR NA BINANCE  ═══",
        "",
        f"{direction_emoji}  *Operacao: {direction_word}*\n{explain_direction}",
        "",
        "────────────────────────",
        "",
        "*PASSO 1*  —  Abrir a Binance Futures",
        "",
        "  1.1.  Abra o app da Binance no celular OU acesse",
        "       *binance.com* no navegador do computador.",
        "",
        "  1.2.  No menu inferior (app) ou no topo (site), "
        "       toque em *Trade*  >  *Futuros*.",
        "",
        "  1.3.  Confirme que esta na aba *USD-M Perpetuo* "
        "       (nao 'COIN-M').",
        "",
        "  ❗  *Primeira vez em Futuros?*",
        "       A Binance pode pedir para voce ativar o modo Futuros. "
        "       Va no perfil > Configuracoes > Modo de Trade > ative 'Futuros'. "
        "       Voce precisara aceitar os termos. Nao se preocupe: "
        "       os stops que vamos configurar protegem seu capital.",
        "",
        "────────────────────────",
        "",
        "*PASSO 2*  —  Selecionar o par BTC/USDT",
        "",
        "  2.1.  Na barra de busca no topo da tela de Futuros, "
        "       digite *BTCUSDT*.",
        "",
        "  2.2.  Selecione o resultado que mostrar "
        "       *BTCUSDT Perpetual*.",
        "",
        "  2.3.  Confirme o grafico de Bitcoin apareceu no centro da tela.",
        "",
        "────────────────────────",
        "",
        "*PASSO 3*  —  Configurar a Alavancagem",
        "",
        "  3.1.  Procure o campo *Alavancagem* no painel de ordens "
        "       (geralmente no canto superior esquerdo, abaixo do par).",
        "",
        "  3.2.  Clique nele e escolha *5x*.",
        "",
        "  3.3.  IMPORTANTE: Se aparecer a opcao *Isolada* vs *Cruzada*, "
        "       escolha *Isolada*.",
        "",
        "  ℹ  *O que e alavancagem Isolada?*",
        "       A alavancagem multiplica seu ganho (e perda). Com 5x, "
        "       um movimento de 1% no BTC vira 5% na sua posicao. "
        "       O modo *Isolada* significa que o risco e limitado apenas "
        "       ao dinheiro que voce colocar nessa operacao — se o Stop Loss "
        "       falhar (raro), voce so perde o alocado nesta trade, "
        "       nao todo seu saldo. Mais seguro.",
        "",
        "────────────────────────",
        "",
        f"*PASSO 4*  —  Criar a ordem de {direction_word}",
        "",
        "  4.1.  No painel de ordens, certifique-se de que o botao "
        f"       {direction_btn} esta selecionado.",
        "",
        "  4.2.  *Tipo de ordem:* clique no menu suspenso (geralmente "
        "       mostra 'Limite' por padrao) e selecione *Limitada*.",
        "",
        "  ℹ  *Por que Limitada?* A ordem limitada so executa "
        "       quando o preco atingir EXATAMENTE o valor que voce "
        "       definir. Isso garante que voce entra no preco certo, "
        "       sem surpresas de slippage.",
        "",
        f"  4.3.  *Preco (Price):* digite exatamente:",
        f"       📍  *{_fmt_price(signal.entry_price)}*",
        "",
        f"  4.4.  *Quantidade (Quantity):* veja a tabela la embaixo "
        f"       ⬇️ para saber quanto colocar conforme seu capital.",
        "       (Dica: se nao souber, use a linha do seu saldo total)",
        "",
        "────────────────────────",
        "",
        "*PASSO 5*  —  Configurar o STOP LOSS (protecao)",
        "",
        "  ⚠️  *Este e o passo mais importante!* O Stop Loss "
        "  e o seu seguro. Se o preco for contra voce, a Binance fecha "
        "  a operacao automaticamente, limitando a perda.",
        "",
        f"  5.1.  No painel de ordens, procure a secao *TP/SL* "
        f"       (pode estar abaixando um painel ou em uma aba).",
        "",
        f"  5.2.  Ative o *Stop Loss* (toggle/botao).",
        "",
        f"  5.3.  *Tipo de gatilho:* selecione *Preco* (Mark Price).",
        "",
        f"  5.4.  *Preco do Stop:* digite:",
        f"       📍  *{_fmt_price(signal.stop_loss)}*",
        f"       (fica {sl_pct:.2f}% {sl_direction} do preco de entrada)",
        "",
        f"  5.5.  *Preco de execucao (Trigger Price):* deixe em branco "
        f"       ou igual ao preco do Stop. Isso faz a ordem executar "
        f"       a mercado quando o Stop for atingido (mais rapido).",
        "",
        f"  5.6.  ✅  MARQUE a opcao *'Apenas reducao de posicao'* "
        f"       (somente se disponivel). Isso impede que o Stop Loss "
        f"       abra uma posicao inversa por engano.",
        "",
        f"  ℹ  *O que acontece se o Stop Loss for atingido?*",
        f"       A Binance fecha automaticamente sua posicao. "
        f"       Voce perde no maximo {sl_pct:.2f}% do capital alocado "
        f"       nesta operacao. E o custo de se proteger de perdas maiores.",
        "",
        "────────────────────────",
        "",
        "*PASSO 6*  —  Configurar o TAKE PROFIT (alvo de lucro)",
        "",
        "  ✅  O Take Profit fecha a operacao automaticamente "
        "  quando o preco atinge seu alvo de lucro. Voce nao precisa "
        "  ficar olhando o grafico 24h.",
        "",
        f"  6.1.  Na mesma secao *TP/SL*, ative o *Take Profit*.",
        "",
        f"  6.2.  *Tipo de gatilho:* selecione *Preco* (Mark Price).",
        "",
        f"  6.3.  *Preco do Take Profit:* digite:",
        f"       📍  *{_fmt_price(signal.take_profit)}*",
        f"       (fica {tp_pct:.2f}% {tp_direction} do preco de entrada)",
        "",
        f"  6.4.  *Preco de execucao:* deixe em branco (executa a mercado).",
        "",
        f"  ℹ  *Relacao Risco:Retorno = {rr}*",
        f"       Para cada $1 arriscado, o alvo de ganho e de ${float(rr.split(':')[1]):.1f}. "
        f"       Exemplo: se voce arrisca perder $10, o alvo e ganhar ${10 * float(rr.split(':')[1]):.0f}.",
        "",
        "────────────────────────",
        "",
        "*PASSO 7*  —  Revisar e Enviar a Ordem",
        "",
        "  7.1.  ANTES de clicar em confirmar, verifique os 4 valores:",
        "",
        f"       ✅ Direcao: {direction_word}",
        f"       ✅ Preco de entrada: {_fmt_price(signal.entry_price)}",
        f"       ✅ Stop Loss: {_fmt_price(signal.stop_loss)}",
        f"       ✅ Take Profit: {_fmt_price(signal.take_profit)}",
        "",
        "  7.2.  Clique em *Comprar/Long* ou *Vender/Short* "
        "       (o botao grande verde ou vermelho) para enviar.",
        "",
        "  7.3.  A ordem aparecera na aba *Ordens Abertas*. "
        "       Enquanto o preco nao atingir seu limite, a ordem fica "
        "       aguardando. Quando executar, voce vera na aba *Posicoes*.",
        "",
        "══════════════════════════════",
        "",
        "💰  *TABELA: QUANTO COLOCAR EM CADA OPERACAO*",
        "",
        "  Regra: arriscar sempre *0.5% do seu capital* por operacao.",
        "  Isso significa que, mesmo que o Stop Loss seja atingido,",
        "  voce perde no maximo 0.5% do total. Se tiver $1.000,",
        "  a perda maxima seria de $5,00.",
        "",
        "  Formula: Quantidade = (Capital x 0,5%) / (Entrada - Stop Loss)",
        "",
        "  Capital   | Risco Max  | Quantidade    | Ganho Alvo  | Perda Max",
        "  ─────────┊─────────┊─────────────┊──────────┊──────────",
        *sizing_rows,
        "",
        "  ℹ  *Como usar esta tabela:*",
        "       1. Veja quanto voce tem de capital total disponivel.",
        "       2. Encontre a linha mais proxima na tabela.",
        "       3. Copie o valor da coluna 'Quantidade' para o campo 'Quantity' na Binance (Passo 4.4).",
        "",
        "  ℹ  *Nao tem o valor exato?* Use a proporcao.",
        "       Exemplo: se tem $2.000, pegue a linha de $1.000 e multiplique\n       a quantidade por 2.",
        "",
        "══════════════════════════════",
        "",
        "📖  *GLOSSARIO RAPIDO (para consulta)*",
        "",
        "  *Entrada (Entry):*  preco onde voce quer entrar na operacao.",
        "  *Stop Loss (SL):*  preco que fecha a operacao com prejuizo. Se o preco chegar la, voce perde. Mas a perda e limitada.",
        "  *Take Profit (TP):*  preco que fecha a operacao com lucro. Se o preco chegar la, voce ganha e a operacao encerra.",
        "  *Long:*  aposta na ALTA do Bitcoin (compra barato, vende caro).",
        "  *Short:*  aposta na BAIXA do Bitcoin (vende caro, compra barato).",
        "  *Alavancagem:*  multiplica seu ganho e perda. 5x = 5x mais intensidade.",
        "  *Isolada:*  modo onde o risco e limitado ao valor alocado na operacao.",
        "  *Ordem Limitada:*  ordem que so executa no preco exato que voce definir.",
        "",
        "══════════════════════════════",
        "",
        "❓  *PERGUNTAS FREQUENTES*",
        "",
        "  *A ordem pode nao executar?*",
        "  Sim. A ordem limitada so executa se o preco alcancar o\n       valor exato. Se o preco ja passou, voce pode atualizar o\n       valor ou usar 'Mercado' (mas o preco pode ser pior).",
        "",
        "  *Posso alterar o Stop Loss depois?*",
        "  Sim! Vá em Posicoes > clique na posicao > edite o SL.\n       Voce pode mover o SL para mais perto do preco de entrada\n       (reduzindo risco) ou ativar trailing stop manualmente.",
        "",
        "  *E se eu fechar manualmente?*",
        "  Vá em Posicoes > clique em 'Fechar' ao lado da posicao.\n       O sistema do bot tambem enviara uma notificacao de fechamento.",
        "",
        "  *Quanto tempo a operacao fica aberta?*",
        "  Fica aberta ate o Stop Loss ou Take Profit ser atingido,\n       ou ate voce fechar manualmente. O sistema monitora o tempo\n       maximo (144 candles = 6 dias para Squeeze, 120 = 5 dias para RSI).",
        "",
        "  *Posso ter mais de uma operacao ao mesmo tempo?*",
        "  Sim, o sistema permite ate 3 posicoes simultaneas.\n       Mas so abra uma nova se tiver capital suficiente e o risco\n       total estiver dentro do limite.",
    ]

    return "\n".join(lines)


# ==================================================================
# Construtor principal de mensagem de sinal
# ==================================================================

def _format_signal_message(
    signal: Signal, symbol: str,
    regime: str = "", strategy: str = "", confidence: float = 0.5,
) -> str:
    """
    Constroi a mensagem completa do Telegram para um sinal.
    Inclui: cabecalho com dados, analise tecnica explicada,
    guia passo a passo Binance, tabela de sizing e glossario.
    """
    is_long = signal.type == SignalType.LONG
    direction = "COMPRA (LONG)" if is_long else "VENDA (SHORT)"
    icon = "🚀" if is_long else "🔽"

    # Compatibilidade: se regime/strategy nao informados
    if not regime:
        regime = signal.regime or "UNKNOWN"
    if not strategy:
        # V13 default: squeeze ou rsi_reversal
        strategy = signal.entry_type or "squeeze_breakout"

    strategy_label = _STRATEGY_LABEL.get(strategy, strategy)

    # Calculos do cabecalho
    sl_pct = _pct_diff(signal.entry_price, signal.stop_loss)
    tp_pct = _pct_diff(signal.entry_price, signal.take_profit)
    rr = _rr_ratio(signal)

    if is_long:
        sl_label = f"−{sl_pct:.2f}%"
        tp_label = f"+{tp_pct:.2f}%"
    else:
        sl_label = f"+{sl_pct:.2f}%"
        tp_label = f"−{tp_pct:.2f}%"

    # --- Cabecalho + dados da operacao ---
    header = [
        f"{icon}  *SINAL DE {direction}*  —  {symbol}",
        "",
        "═══  📊  DADOS DA OPERACAO  ═══",
        "",
        f"  Estrategia:  *{strategy_label}*",
        f"  Preco de Entrada:  {_fmt_price(signal.entry_price)}",
        f"  Stop Loss:  {_fmt_price(signal.stop_loss)}  ({sl_label})",
        f"  Take Profit:  {_fmt_price(signal.take_profit)}  ({tp_label})",
        f"  Risco:Retorno:  {rr}",
    ]

    # --- Analise tecnica ---
    analysis = _build_analysis(signal, regime, strategy, confidence)

    # --- Guia Binance ---
    binance = _build_binance_guide(signal)

    # --- Rodape ---
    footer = [
        "",
        "══════════════════════════════",
        "",
        "⚠️  *AVISO IMPORTANTE:*",
        "",
        "  Este e um sinal de *analise automatica*. O sistema "
        "  *NAO executa* nenhuma operacao na Binance.",
        "",
        "  Use este sinal como *referencia* para sua propria decisao. Voce e o unico responsavel por suas operacoes.",
        "",
        "  *Nunca invista mais do que pode se permitir perder.*",
        "  *Cripto e um ativo de alto risco.*",
        "",
        "  V13-ROBUSTA  |  Squeeze + RSI Reversal  |  Half-Risk 0.5%",
    ]

    return "\n".join(header + [analysis, binance] + footer)


# ==================================================================
# Mensagem de fechamento de trade
# ==================================================================

def _format_trade_close_message(
    pos_type: str,
    entry: float,
    exit_p: float,
    pnl_pct: float,
    pnl_usd: float,
    reason: str,
    be: bool = False,
    trailing: bool = False,
    partial: bool = False,
) -> str:
    """Constroi a mensagem de fechamento com orientacao didatica."""
    is_win = pnl_pct >= 0
    icon = "✅" if is_win else "❌"
    result = "LUCRO" if is_win else "PERDA"
    reason_pt = _EXIT_REASON_PT.get(reason, reason)

    # Explicacao do motivo em linguagem simples
    reason_explained = {
        "tp": (
            "O preco atingiu o alvo! O Take Profit foi executado "
            "pela Binance e a operacao fechou automaticamente com lucro."
        ),
        "sl": (
            "O preco atingiu o limite de protecao. O Stop Loss foi "
            "acionado para evitar uma perda maior. Isso e normal e faz "
            "parte da estrategia — e o custo de se proteger."
        ),
        "timeout": (
            "O preco nao atingiu nem o alvo nem o stop dentro do "
            "tempo maximo. A operacao foi fechada pelo tempo esgotado."
        ),
        "be": (
            "O Stop Loss foi movido para o preco de entrada (breakeven)."
        ),
    }
    reason_simple = reason_explained.get(
        reason, f"Operacao fechada: {reason_pt}."
    )

    # O que fazer agora
    if is_win:
        next_steps = (
            "🎉  *Parabens pela operacao!*\n"
            "  ℹ  Voce nao precisa fazer nada. A Binance ja fechou\n       a posicao e o lucro esta no seu saldo. Voce pode verificar\n       em *Carteira* > *Futuros* > *Historico de Ordens*.\n"
            "  ℹ  Aguarde o proximo sinal. O sistema continua monitorando."
        )
    else:
        next_steps = (
            "😢  *Operacao com perda — normal na estrategia.*\n"
            "  ℹ  Perdas fazem parte de qualquer estrategia. O importante\n       e que a perda foi LIMITADA pelo Stop Loss. Sem o SL, a\n       perda poderia ser muito maior.\n"
            "  ℹ  O sistema tem protecoes automaticas: apos perdas\n       consecutivas, ele entra em cooldown para evitar erros em\n       sequencia.\n"
            "  ℹ  Nao tente 'recuperar' a perda fazendo operacoes\n       maiores. Mantenha o risco de 0.5% por operacao.\n"
            "  ℹ  Aguarde o proximo sinal. O sistema continua monitorando."
        )

    lines = [
        f"{icon}  *OPERACAO FECHADA — {result}*",
        "",
        f"  Resultado:  {'+' if is_win else ''}{pnl_pct:.2f}%  "
        f"({'+' if pnl_usd >= 0 else ''}{pnl_usd:,.2f} USD)",
        f"  Entrada:  {_fmt_price(entry)}  →  Saida:  {_fmt_price(exit_p)}",
        f"  Motivo:  {reason_pt}",
        "",
        f"  {reason_simple}",
    ]

    extras = []
    if be:
        extras.append(
            "   ⚡ Break-even: SL foi movido para o preco de entrada."
        )
    if trailing:
        extras.append(
            "   🔄 Trailing stop acompanhou o preco para cima."
        )
    if partial:
        extras.append(
            "   🎯 Take Profit parcial: 50% da posicao fechada no alvo."
        )

    if extras:
        lines.append("")
        lines.append("*Gerenciamento ativo durante a operacao:*")
        lines.extend(extras)

    lines.extend([
        "",
        "══════════════════════════════",
        "",
        next_steps,
        "",
        "📱  *Para verificar manualmente na Binance:*",
        "  *App:*  Trade  >  Futuros  >  Posicoes  (ou Historico)",
        "  *Site:*  Ordens  >  Historico de Ordens  >  Futuros",
        "",
        "V13-ROBUSTA  |  Half-Risk 0.5%",
    ])
    return "\n".join(lines)


# ==================================================================
# Classe principal  (API inalterada)
# ==================================================================

class TelegramNotifier:
    """Wrapper assincrono sobre telegram.Bot para envio de sinais."""

    def __init__(self, bot_token: str, chat_id: str) -> None:
        if not bot_token or not chat_id or not _HAS_TELEGRAM:
            self.bot = None
            self.chat_id = chat_id
            logger.info("Telegram desabilitado (token ou chat_id ausente).")
            return
        self.bot = Bot(token=bot_token)
        self.chat_id = chat_id

    # ------------------------------------------------------------------
    # Chunking: dividir mensagens longas no limite do Telegram
    # ------------------------------------------------------------------
    @staticmethod
    def _chunk_message(text: str, max_len: int = 4000) -> list[str]:
        """
        Divide texto em chunks respeitando o limite do Telegram.
        Tenta quebrar em linhas em branco ou dividers (═══) para
        manter secoes inteiras.
        """
        if len(text) <= max_len:
            return [text]

        chunks = []
        # Tenta quebrar em secoes (linhas com ═══)
        lines = text.split("\n")
        current = []
        current_len = 0

        i = 0
        while i < len(lines):
            line = lines[i]
            line_len = len(line) + 1  # +1 para o \n
            # Se adicionar essa linha exceder o limite
            if current_len + line_len > max_len and current:
                # Tenta encontrar um ponto de quebra bom dentro do bloco atual
                chunk_text = "\n".join(current)
                chunks.append(chunk_text)
                current = []
                current_len = 0
                # Se a linha atual e um divisor, adiciona como inicio do proximo
                if line.strip().startswith("═══"):
                    current.append("")
                    current.append(line)
                    current_len = line_len + 1
                    i += 1
                    continue

            current.append(line)
            current_len += line_len
            i += 1

        if current:
            chunks.append("\n".join(current))

        return chunks if chunks else [text]

    async def _send_chunked(self, text: str, parse_mode=None) -> None:
        """Envia mensagem dividida em chunks se necessario."""
        chunks = self._chunk_message(text)
        for i, chunk in enumerate(chunks):
            try:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=chunk,
                    parse_mode=parse_mode,
                )
            except Exception:
                # Fallback: texto puro
                try:
                    await self.bot.send_message(
                        chat_id=self.chat_id,
                        text=chunk,
                    )
                except Exception as exc:
                    logger.exception(
                        "Falha ao enviar chunk %d/%d: %s", i + 1, len(chunks), exc
                    )

    # ------------------------------------------------------------------
    # Mensagem de boas-vindas
    # ------------------------------------------------------------------
    async def send_welcome(self, exchange_id: str, symbol: str) -> None:
        """Envia mensagem de boas-vindas ao iniciar."""
        if not self.bot:
            return
        msg = (
            "🤖  *CTEV Signal Bot — V13-ROBUSTA — Iniciado*\n"

            "📖  *O que este bot faz:*\n"
            "Monitora o Bitcoin (BTC/USDT) 24 horas por dia, 7 dias por "
            "semana. Ele analisa o mercado a cada 1 hora usando indicadores "
            "tecnicos avancados (Bollinger Bands, RSI, ADX, ATR, MACD) e "
            "identifica os melhores momentos para entrar em operacoes de "
            "compra ou venda.\n"
            "Quando uma oportunidade e detectada, voce recebe uma mensagem "
            "com TODOS os dados da operacao (preco de entrada, stop loss, "
            "take profit) e um *guia passo a passo* de como configurar "
            "cada parametro na Binance — mesmo que voce nunca tenha operado "
            "antes.\n"

            "🌐  *Versao ativa: V13-ROBUSTA*\n"
            "Esta e a unica versao que passou no teste de robustez "
            "(Walk-Forward OOS com 17 janelas de validacao):\n"
            "• Sharpe Ratio: 1.30 (bom risco-retorno)"
            "• Sortino Ratio: ~2.0 (excelente na queda)"
            "• Drawdown Maximo: 33.6% (perda maxima historica)"
            "• Consistencia: 65% (11 de 17 janelas positivas)"
            "• Overfit Score: 35.1 (< 40 = modelo robusto)\n"

            "🏈  *Estrategias ativas (validadas):*\n"
            "• *Squeeze Breakout*"
            "  Detecta quando o Bitcoin esta 'comprimido' (Bandas de Bollinger "
            "muito juntas) e entra no inicio da expansao. E como comprar "
            "uma mola sendo pressionada antes de ela saltar."
            "  Risco:Retorno de 1:3.6 | Stop: 1.8x ATR | Alvo: 6.5x ATR\n"
            "• *RSI Reversal*"
            "  Detecta reversoes de curto prazo quando o RSI atinge extremos "
            "dentro de uma tendencia. E como comprar na 'promoção' dentro "
            "de uma tendencia de alta."
            "  Risco:Retorno de 1:3.1 | Stop: 1.8x ATR | Alvo: 5.5x ATR\n"

            "🛡  *Protecoes automaticas do sistema:*\n"
            "• *Circuit Breaker:* pausa automaticamente em movimentos extremos"
            "• *Drawdown Diario:* se perder mais de 5% no dia, pausa ate as 00:00 UTC"
            "• *Drawdown Semanal:* se perder mais de 10% na semana, pausa ate segunda"
            "• *Cooldown:* 2 stops consecutivos = pausa de 3 horas"
            "• *Filtro ATR:* so opera quando a volatilidade esta entre 8% e 92%"
            "• *Max 3 posicoes:* nunca mais de 3 operacoes ao mesmo tempo"


            "📱  *Como funciona cada sinal que voce recebera:*\n"
            "• *Cabecalho* — dados da operacao (entrada, SL, TP, risco)"
            "• *Analise* — explicacao simples do que esta acontecendo no mercado"
            "• *Guia Binance* — 7 passos detalhados para montar na plataforma,"
            "  incluindo onde clicar e o que cada campo significa"
            "• *Tabela de Capital* — quanto colocar conforme seu saldo"
            "• *Glossario* — explicacao de cada termo tecnico"
            "• *FAQ* — perguntas frequentes respondidas\n"

            f"📁  *Configuracao atual:*"
            f"  Exchange: {exchange_id.upper()}"
            f"  Par: {symbol}"
            f"  Timeframe: 1H (analise a cada 1 hora)"
            f"  Risco por operacao: 0.5% do capital\n"

            "⚠️  *MODO: APENAS ANALISE*\n"
            "Este bot NAO executa nenhuma operacao automaticamente. "
            "Ele apenas analisa e envia sinais com instrucoes para voce "
            "configurar manualmente na Binance. Voce tem controle total.\n"
            "*Nunca invista mais do que pode se permitir perder.*\n"

            "🖥  Acesse o painel web para acompanhar em tempo real."
        )
        try:
            await self._send_chunked(msg, ParseMode.MARKDOWN)
        except Exception:
            try:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=(
                        "CTEV Signal Bot V13-ROBUSTA iniciado!\n\n"
                        f"Monitorando {symbol} em 1H via {exchange_id.upper()}.\n"
                        "Modo: apenas sinais (sem execucao).\n\n"
                        "Voce recebera alertas com guia passo a passo "
                        "detalhado para montar cada operacao na Binance."
                    ),
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Sinal de trade
    # ------------------------------------------------------------------
    async def send_signal(
        self,
        signal: Signal,
        symbol: str,
        regime: str = "",
        strategy: str = "",
        confidence: float = 0.5,
    ) -> None:
        """Envia o sinal formatado para o chat configurado."""
        if not self.bot:
            return
        text = _format_signal_message(
            signal, symbol, regime, strategy, confidence,
        )
        try:
            await self._send_chunked(text, ParseMode.MARKDOWN)
            logger.info(
                "Sinal %s enviado ao Telegram (chat_id=%s, chunks=%d).",
                signal.type.value,
                self.chat_id,
                len(self._chunk_message(text)),
            )
        except Exception as exc:
            logger.exception("Falha ao enviar mensagem ao Telegram: %s", exc)

    # ------------------------------------------------------------------
    # Fechamento de trade
    # ------------------------------------------------------------------
    async def send_trade_close(
        self,
        pos_type: str,
        entry: float,
        exit_p: float,
        pnl_pct: float,
        pnl_usd: float,
        reason: str,
        be: bool = False,
        trailing: bool = False,
        partial: bool = False,
    ) -> None:
        """Notifica o fechamento de uma posicao."""
        if not self.bot:
            return
        text = _format_trade_close_message(
            pos_type, entry, exit_p, pnl_pct, pnl_usd,
            reason, be, trailing, partial,
        )
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN,
            )
            logger.info(
                "Trade close %s enviado ao Telegram (PnL=%+.2f%%).",
                pos_type, pnl_pct,
            )
        except Exception as exc:
            logger.exception("Falha ao enviar trade close: %s", exc)
            try:
                await self.bot.send_message(chat_id=self.chat_id, text=text)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Texto livre
    # ------------------------------------------------------------------
    async def send_text(self, text: str) -> None:
        """Envia uma mensagem de texto livre."""
        if not self.bot:
            return
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=text)
        except Exception as exc:
            logger.exception("Falha ao enviar texto ao Telegram: %s", exc)

    # ------------------------------------------------------------------
    # Alerta de risco
    # ------------------------------------------------------------------
    async def send_risk_alert(self, reason: str, message: str) -> None:
        """Envia alerta quando o RiskManager bloqueia ou pausa o bot."""
        if not self.bot:
            return
        explanations = {
            "drawdown_diario": (
                "⚠️  *Limite de perda diaria atingido.*\n"
                "O sistema atingiu o limite de perda diaria de 5% do "
                "capital. Isso e uma protecao automatica para evitar "
                "operações em um dia desfavoravel.\n"
                "ℹ  *O que isso significa para voce?*"
                "Nenhum novo sinal sera enviado hoje. O sistema pausou para "
                "evitar perdas maiores em um dia ruim — isso e boa pratica "
                "de gestao de risco.\n"
                "🔄  *Quando volta?*"
                "Automaticamente as 00:00 UTC (21h de Brasilia)."
            ),
            "drawdown_semanal": (
                "⚠️  *Limite de perda semanal atingido.*\n"
                "O sistema atingiu o limite de perda semanal de 10% do "
                "capital. Protecao mais forte ativada.\n"
                "ℹ  *O que isso significa para voce?*"
                "Nenhum novo sinal sera enviado esta semana. O sistema "
                "pausou para evitar uma espiral de perdas.\n"
                "🔄  *Quando volta?*"
                "Automaticamente na segunda-feira as 00:00 UTC."
            ),
            "perdas_consecutivas": (
                "⏳  *Cooldown por perdas consecutivas.*\n"
                "O sistema registrou 2 Stop Loss seguidos. Quando isso "
                "acontece, o bot entra em pausa automatica para evitar "
                "sinais em uma sequencia ruim do mercado.\n"
                "ℹ  *O que isso significa para voce?*"
                "Nenhum novo sinal por algumas horas. O mercado pode estar "
                "em um periodo desfavoravel e e melhor aguardar.\n"
                "🔄  *Quando volta?*"
                "Apos o cooldown automatico (3 candles = 3 horas)."
            ),
            "circuit_breaker": (
                "🚨  *Circuit Breaker ativado.*\n"
                "Movimento extremo de preco detectado. O Bitcoin se moveu "
                "muito rapido em pouco tempo, o que indica instabilidade.\n"
                "ℹ  *O que isso significa para voce?*"
                "O sistema pausou temporariamente. Operar em momentos de "
                "extrema volatilidade e muito arriscado — os precos podem "
                "saltar em qualquer direcao.\n"
                "🔄  *Quando volta?*"
                "Tao logo a volatilidade volte ao normal. Geralmente algumas "
                "horas."
            ),
            "filtro_volatilidade": (
                "🚨  *Volatilidade fora da faixa segura.*\n"
                f"O mercado esta muito "
                f"{'lateral (sem direcao clara — ruim para tendencias)' if 'lateral' in message.lower() else 'volatil (movimentos muito grandes e imprevisiveis)'}. "
                "O sistema so opera quando a volatilidade esta entre 8% e 92% "
                "do historico — fora disso, os sinais sao menos confiaveis.\n"
                "ℹ  *O que isso significa para voce?*"
                "Nenhum sinal por enquanto. O sistema aguarda condicoes "
                "melhores para gerar sinais com mais probabilidade de acerto."
            ),
            "cooldown_entre_sinais": (
                "⏳  *Intervalo minimo entre sinais.*\n"
                "O sistema acabou de gerar um sinal e precisa aguardar "
                "algumas velas antes do proximo. Isso evita sinais muito "
                "proximo uns dos outros.\n"
                "ℹ  *O que fazer?*"
                "Apenas aguarde. Se o sinal anterior ainda esta aberto, "
                "gerencie essa posicao. Um novo sinal vira quando o cooldown "
                "acabar e uma nova oportunidade for detectada."
            ),
            "kill_switch_manual": (
                "🚑  *Bot DESLIGADO manualmente.*\n"
                "O bot foi desligado por um administrador. Nenhum sinal "
                "sera gerado enquanto estiver desligado.\n"
                "ℹ  *O que fazer?*"
                "Para reativar, acesse o painel web e clique no botao "
                "'Reativar'. Se voce nao sabe como, entre em contato "
                "com o administrador."
            ),
        }

        explanation = explanations.get(reason, f"{reason}: {message}")

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=explanation,
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception:
            try:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=explanation,
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Alerta de exchange conectada
    # ------------------------------------------------------------------
    async def send_exchange_connected(self, exchange_id: str, symbol: str) -> None:
        """Notifica qual exchange foi conectada."""
        if not self.bot:
            return
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=(
                    f"🔐  *Conectado a {exchange_id.upper()}*\n"
                    f"📁  *Monitorando:* {symbol}"
                    f"📅  *Timeframe:* 1H (analise a cada 1 hora)"
                    f"🌐  *Versao:* V13-ROBUSTA (WFO validada)\n"
                    f"📈  *Estrategias ativas:*"
                    f"• Squeeze Breakout (SL 1.8x ATR, TP 6.5x ATR)"
                    f"• RSI Reversal (SL 1.8x ATR, TP 5.5x ATR)\n"
                    f"🛡  *Protecoes:*"
                    f"• Half-risk: 0.5% por operacao"
                    f"• Max 3 posicoes simultaneas"
                    f"• Trailing stop: 0.6x ATR"
                    f"• Circuit breaker + Drawdown filter\n"
                    f"✅  O bot esta ativo e analisando o mercado."
                    f"Voce recebera sinais com *guia passo a passo detalhado*"
                    f"para montar cada operacao na Binance.\n"
                    f"ℹ  *Primeiro sinal?* Nao se preocupe — cada mensagem"
                    f"vem com todas as instrucoes, incluindo o que cada"
                    f"campo significa e onde clicar na plataforma."
                ),
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception:
            pass
