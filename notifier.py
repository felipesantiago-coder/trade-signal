"""
notifier.py
-----------
Integracao assincrona com o Telegram usando python-telegram-bot.

MODO: Signal-Only (apenas analise, sem execucao de ordens).
Mensagens projetadas para serem intuitivas, educativas e acionaveis,
detalhando o regime de mercado, a estrategia usada e o racional
detras de cada sinal.

v7.1: Reescrito para refletir regime-switching v7 com:
  - 9 regimes de mercado com histerese e confidence scoring
  - 3 estrategias adaptativas (trend-follow, mean-reversion, neutral)
  - ADX floor v7.1 para WEAK_UPTREND LONG
  - Notificacao de trade fechado (antes ausente)

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
# Mapas de tradução e formatacao
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

_STRATEGY_PT = {
    "trend_follow_long":  "Trend-Following (Compra)",
    "trend_follow_short": "Trend-Following (Venda)",
    "mean_reversion":     "Mean-Reversion (Reversao)",
    "breakout_long":      "Breakout (Alta)",
    "breakout_short":     "Breakout (Baixa)",
    "neutral":            "Neutro (Sem operacao)",
}

_PULLBACK_PT = {
    "fibonacci":      "Fibonacci (zona de retracao)",
    "ema20_touch":    "Toque na EMA(20)",
    "ema50_touch":    "Toque na EMA(50)",
    "bb_bounce_long": "Bounce na Banda Inferior (BB)",
    "bb_bounce_short":"Bounce na Banda Superior (BB)",
    "breakout_long":  "Rompimento da Banda Superior",
    "breakout_short": "Rompimento da Banda Inferior",
}

_EXIT_REASON_PT = {
    "tp": "Take Profit (alvo atingido)",
    "sl": "Stop Loss (protecao acionada)",
    "timeout": "Timeout (maximo de barras sem saida)",
    "be":  "Break-Even (stop movido para entrada)",
}


def _rr_ratio(signal: Signal) -> str:
    """Calcula e formata a relacao risco:retorno real."""
    risk = abs(signal.entry_price - signal.stop_loss)
    reward = abs(signal.take_profit - signal.entry_price)
    rr = reward / max(risk, 1e-9)
    return f"1:{rr:.1f}"


def _confidence_bar(confidence: float) -> str:
    """Retorna barra visual de confianca (5 blocos)."""
    filled = round(confidence * 5)
    return "\U0001f7e2" * filled + "\u26ab" * (5 - filled)  # verde + cinza


# ==================================================================
# Construtores de mensagem de sinal
# ==================================================================

def _build_trend_follow_long_msg(
    signal: Signal, symbol: str,
    regime: str, strategy: str, confidence: float,
) -> str:
    """Mensagem detalhada para sinal LONG de trend-following."""
    rr = _rr_ratio(signal)
    regime_pt = _REGIME_PT.get(regime, regime)
    pullback_pt = _PULLBACK_PT.get(signal.pullback_type, signal.pullback_type)
    conf_bar = _confidence_bar(confidence)
    di_spread = signal.plus_di - signal.minus_di

    # Contexto do regime
    if regime == "STRONG_UPTREND":
        regime_ctx = (
            "O mercado esta em tendencia de alta forte: ADX elevado, "
            "EMA(20) > EMA(50) > EMA(200), e o slope da EMA(50) "
            "e acelerado. Este e o melhor cenario para trend-following."
        )
    elif regime == "WEAK_UPTREND":
        regime_ctx = (
            "O mercado esta em tendencia de alta, porem enfraquecendo. "
            "A tendencia ainda esta intacta (EMA(50) > EMA(200)), "
            "mas a forca direcional diminuiu. O sinal passou pelo "
            "filtro ADX >= 22 (v7.1) para garantir qualidade."
        )
    else:
        regime_ctx = f"Regime: {regime_pt}"

    lines = [
        f"\U0001f4c8  *SINAL DE COMPRA (LONG)*  no {symbol}",
        "",
        f"\U0001f310  *Regime de mercado:* {regime_pt}",
        f"\U0001f4ca  Confianca: {confidence:.0%}  {conf_bar}",
        f"\U0001f3af  Estrategia: {_STRATEGY_PT.get(strategy, strategy)}",
        "",
        "\U0001f50d  *O que o sistema detectou:*",
        "",
        f"1. {regime_ctx}",
        "",
        f"2. Pullback identificado em zona de compra",
        f"   Tipo: {pullback_pt}",
        f"   RSI: {signal.rsi:.1f} (zona de pullback saudavel)",
        f"   MACD histograma: {signal.macd_hist:+.4f}",
        "",
        "3. Indicadores chave",
        f"   ADX: {signal.adx:.1f} (forca da tendencia)",
        f"   DI spread: {di_spread:+.1f} (compradores {'dominam' if di_spread > 0 else 'sob pressao'})",
        f"   EMA(50) slope: {signal.ema50_slope:+.3f} ({'acelerando' if signal.ema50_slope > 1.0 else 'estavel' if signal.ema50_slope > 0.3 else 'fraca'})",
        f"   ATR percentile: {signal.atr_percentile:.0%} ({'volatilidade alta' if signal.atr_percentile > 0.7 else 'volatilidade normal' if signal.atr_percentile > 0.3 else 'volatilidade baixa'})",
        "",
        f"\U0001f4b0  *Niveis de operacao:*",
        f"   Entrada:  ${signal.entry_price:,.2f}",
        f"   Stop Loss:  ${signal.stop_loss:,.2f} ({abs(signal.entry_price - signal.stop_loss) / signal.entry_price * 100:.2f}% abaixo)",
        f"   Take Profit:  ${signal.take_profit:,.2f} ({abs(signal.take_profit - signal.entry_price) / signal.entry_price * 100:.2f}% acima)",
        f"   Risco:Retorno: {rr}",
        f"   ATR: ${signal.atr:,.2f}",
        "",
        "\U0001f6a9  *Aviso:* Este e um sinal de analise automatica. "
        "O sistema NAO executa nenhuma operacao. "
        "Use como referencia para sua propria decisao.",
        "",
        "CTEV v7.1 | V13-ROBUSTA | Squeeze + RSI Reversal | Half-Risk",
    ]
    return "\n".join(lines)


def _build_trend_follow_short_msg(
    signal: Signal, symbol: str,
    regime: str, strategy: str, confidence: float,
) -> str:
    """Mensagem detalhada para sinal SHORT de trend-following."""
    rr = _rr_ratio(signal)
    regime_pt = _REGIME_PT.get(regime, regime)
    pullback_pt = _PULLBACK_PT.get(signal.pullback_type, signal.pullback_type)
    conf_bar = _confidence_bar(confidence)
    di_spread = signal.plus_di - signal.minus_di

    if regime == "STRONG_DOWNTREND":
        regime_ctx = (
            "O mercado esta em tendencia de baixa forte: ADX elevado, "
            "EMA(20) < EMA(50) < EMA(200), e o slope da EMA(50) "
            "e fortemente negativo. Tendencia de baixa bem estabelecida."
        )
    elif regime == "WEAK_DOWNTREND":
        regime_ctx = (
            "O mercado esta em tendencia de baixa, porem enfraquecendo. "
            "A estrutura de baixa permanece (EMA(50) < EMA(200)), "
            "mas a pressao vendedora diminuiu."
        )
    else:
        regime_ctx = f"Regime: {regime_pt}"

    lines = [
        f"\U0001f4c9  *SINAL DE VENDA (SHORT)*  no {symbol}",
        "",
        f"\U0001f310  *Regime de mercado:* {regime_pt}",
        f"\U0001f4ca  Confianca: {confidence:.0%}  {conf_bar}",
        f"\U0001f3af  Estrategia: {_STRATEGY_PT.get(strategy, strategy)}",
        "",
        "\U0001f50d  *O que o sistema detectou:*",
        "",
        f"1. {regime_ctx}",
        "",
        f"2. Rally de correcao em zona de venda",
        f"   Tipo: {pullback_pt}",
        f"   RSI: {signal.rsi:.1f} (rally em downtrend)",
        f"   MACD histograma: {signal.macd_hist:+.4f}",
        "",
        "3. Indicadores chave",
        f"   ADX: {signal.adx:.1f} (forca da tendencia)",
        f"   DI spread: {di_spread:+.1f} (vendedores {'dominam' if di_spread < 0 else 'sob pressao'})",
        f"   EMA(50) slope: {signal.ema50_slope:+.3f} ({'acelerando queda' if signal.ema50_slope < -1.0 else 'descendo' if signal.ema50_slope < -0.3 else 'fraca'})",
        f"   ATR percentile: {signal.atr_percentile:.0%}",
        "",
        f"\U0001f4b0  *Niveis de operacao:*",
        f"   Entrada:  ${signal.entry_price:,.2f}",
        f"   Stop Loss:  ${signal.stop_loss:,.2f} ({abs(signal.stop_loss - signal.entry_price) / signal.entry_price * 100:.2f}% acima)",
        f"   Take Profit:  ${signal.take_profit:,.2f} ({abs(signal.entry_price - signal.take_profit) / signal.entry_price * 100:.2f}% abaixo)",
        f"   Risco:Retorno: {rr}",
        f"   ATR: ${signal.atr:,.2f}",
        "",
        "\U0001f6a9  *Aviso:* Este e um sinal de analise automatica. "
        "O sistema NAO executa nenhuma operacao. "
        "Use como referencia para sua propria decisao.",
        "",
        "CTEV v7.1 | V13-ROBUSTA | Squeeze + RSI Reversal | Half-Risk",
    ]
    return "\n".join(lines)


def _build_mean_reversion_msg(
    signal: Signal, symbol: str,
    regime: str, strategy: str, confidence: float,
) -> str:
    """Mensagem detalhada para sinal de mean-reversion (RANGING)."""
    is_long = signal.type == SignalType.LONG
    rr = _rr_ratio(signal)
    regime_pt = _REGIME_PT.get(regime, regime)
    pullback_pt = _PULLBACK_PT.get(signal.pullback_type, signal.pullback_type)
    conf_bar = _confidence_bar(confidence)
    direction = "COMPRA (LONG)" if is_long else "VENDA (SHORT)"
    icon = "\U0001f4c8" if is_long else "\U0001f4c9"

    bb_ref = signal.bb_lower if is_long else signal.bb_upper
    bb_label = "Banda Inferior" if is_long else "Banda Superior"
    rsi_desc = ("sobrevendido (fundo da faixa)" if is_long
                else "sobrecomprado (topo da faixa)")

    lines = [
        f"{icon}  *SINAL DE {direction.upper()}*  no {symbol}",
        "",
        f"\U0001f310  *Regime de mercado:* {regime_pt}",
        f"\U0001f4ca  Confianca: {confidence:.0%}  {conf_bar}",
        f"\U0001f3af  Estrategia: {_STRATEGY_PT.get(strategy, strategy)}",
        "",
        "\U0001f50d  *O que o sistema detectou:*",
        "",
        "1. Mercado lateral identificado",
        "O preco oscila entre suporte e resistencia sem "
        "tendencia clara. A estrategia de mean-reversion "
        "opera os extremos das Bandas de Bollinger.",
        "",
        f"2. Preco tocou a {bb_label} das Bollinger",
        f"   {bb_label}: ${bb_ref:,.2f}",
        f"   RSI: {signal.rsi:.1f} ({rsi_desc})",
        f"   BB width: {signal.bb_width:.4f} (lateralidade)",
        f"   MACD histograma: {signal.macd_hist:+.4f}",
        "",
        "3. Indicadores complementares",
        f"   ADX: {signal.adx:.1f} (sem tendencia — confirmando lateral)",
        f"   ATR percentile: {signal.atr_percentile:.0%}",
        f"   Volume: {signal.volume / max(signal.volume_sma20, 1):.2f}x media",
        "",
        f"\U0001f4b0  *Niveis de operacao:*",
        f"   Entrada:  ${signal.entry_price:,.2f}",
        f"   Stop Loss: ${signal.stop_loss:,.2f}",
        f"   Take Profit: ${signal.take_profit:,.2f}",
        f"   Risco:Retorno: {rr}",
        "",
        "\U0001f6a9  *Aviso:* Este e um sinal de analise automatica. "
        "O sistema NAO executa nenhuma operacao. "
        "Use como referencia para sua propria decisao.",
        "",
        "CTEV v7.1 | V13-ROBUSTA | Squeeze + RSI Reversal | Half-Risk",
    ]
    return "\n".join(lines)


def _format_signal_message(
    signal: Signal, symbol: str,
    regime: str = "", strategy: str = "", confidence: float = 0.5,
) -> str:
    """
    Constroi a mensagem do Telegram para um sinal.

    Roteia para o formato especifico da estrategia usada.
    """
    is_long = signal.type == SignalType.LONG

    # Fallback para chamadas sem regime (compatibilidade)
    if not regime:
        regime = signal.regime or "UNKNOWN"
    if not strategy:
        strategy = ("trend_follow_long" if is_long else "trend_follow_short")

    # Roteia pelo tipo de estrategia
    if "mean_reversion" in strategy:
        return _build_mean_reversion_msg(
            signal, symbol, regime, strategy, confidence,
        )
    elif is_long:
        return _build_trend_follow_long_msg(
            signal, symbol, regime, strategy, confidence,
        )
    else:
        return _build_trend_follow_short_msg(
            signal, symbol, regime, strategy, confidence,
        )


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
    """Constroi a mensagem de fechamento de trade."""
    is_long = pos_type == "LONG"
    is_win = pnl_pct >= 0

    icon = "\u2705" if is_win else "\u274c"
    result = "LUCRO" if is_win else "PERDA"
    direction = "COMPRA (LONG)" if is_long else "VENDA (SHORT)"

    reason_pt = _EXIT_REASON_PT.get(reason, reason)

    lines = [
        f"{icon}  *{result} — {direction} fechada*",
        "",
        f"\U0001f4b0  Resultado:",
        f"   PnL: {'+' if is_win else ''}{pnl_pct:.2f}%  ({'+' if pnl_usd >= 0 else ''}{pnl_usd:,.2f} USD)",
        f"   Entrada: ${entry:,.2f}",
        f"   Saida:  ${exit_p:,.2f}",
        f"   Motivo: {reason_pt}",
    ]

    extras = []
    if be:
        extras.append("   \u26a1 Break-even ativado (SL movido para entrada)")
    if trailing:
        extras.append("   \U0001f504 Trailing stop ativado (SL acompanhou o preco)")
    if partial:
        extras.append("   \U0001f3af Take Profit parcial executado")

    if extras:
        lines.append("")
        lines.append("\U0001f527  Gerenciamento ativo:")
        lines.extend(extras)

    lines.extend([
        "",
        "CTEV v7.1 | V13-ROBUSTA | Half-Risk",
    ])
    return "\n".join(lines)


# ==================================================================
# Classe principal
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
    # Mensagem de boas-vindas
    # ------------------------------------------------------------------
    async def send_welcome(self, exchange_id: str, symbol: str) -> None:
        """Envia mensagem de boas-vindas ao iniciar."""
        if not self.bot:
            return
        msg = (
            "\U0001f916  *CTEV Signal Bot — V13-ROBUSTA — Iniciado*\n\n"

            "\U0001f4d6  *O que este bot faz:*\n\n"
            "Monitora o Bitcoin 24h por dia e identifica os melhores "
            "momentos para compra ou venda, usando analise tecnica "
            "automatizada com deteccao adaptativa de regime de mercado.\n\n"

            "\U0001f310  *Versao ativa: V13-ROBUSTA*\n\n"
            "Estrategia validada via Walk-Forward OOS (17 janelas):\n\n"
            "\u2022 OOS Sharpe: 1.30 | Sortino: ~2.0 | MaxDD: 33.6%\n"
            "\u2022 Consistency: 65% (11/17 janelas positivas)\n"
            "\u2022 Overfit Score: 35.1 (< 40 = robusto)\n"
            "\u2022 Risco: 0.5% por trade (half-risk) | 3 posicoes max\n"
            "\u2022 Trailing: 0.6x ATR | Partial TP: 50%\n\n"

            "\U0001f4c8  *Estrategias ativas (WFO validadas):*\n\n"
            "\u2022 *Squeeze Breakout*: BBWP squeeze + breakout das Bandas de "
            "Bollinger. SL 1.8x ATR, TP 6.5x ATR (R:R 3.61). "
            "Capital: 3.0% por trade. Max 144 bars.\n"
            "\u2022 *RSI Reversal*: RSI sobrevendido/sobrecomprado em tendencia. "
            "SL 1.8x ATR, TP 5.5x ATR (R:R 3.06). "
            "Capital: 1.5% por trade. Max 120 bars.\n\n"

            "\U0001f6e1  *Protecoes automaticas:*\n\n"
            "\u2022 Circuit Breaker: pausa em movimentos extremos de preco\n"
            "\u2022 Filtro de drawdown diario (5%) e semanal (10%)\n"
            "\u2022 Cooldown: 2 SLs consecutivos -> pausa de 3 bars\n"
            "\u2022 Filtro de volatilidade ATR (percentile 8%-92%)\n"
            "\u2022 Max 3 posicoes simultaneas\n\n"

            f"\U0001f4c1  *Configuracao:*\n"
            f"Exchange: {exchange_id.upper()}\n"
            f"Par: {symbol}\n"
            f"Timeframe: 1H (analise a cada 1 hora)\n\n"

            "\u26a0\ufe0f  *Modo: apenas analise*\n\n"
            "Este bot APENAS analisa o mercado e envia sinais. "
            "Nenhuma operacao e executada automaticamente. "
            "Use os sinais como referencia para suas proprias "
            "decisoes de investimento.\n\n"

            "\U0001f5a5  Acesse o painel web para acompanhar em tempo real."
        )
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=msg,
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception:
            try:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=(
                        "CTEV Signal Bot V13-ROBUSTA iniciado!\n\n"
                        f"Monitorando {symbol} em 1H via {exchange_id.upper()}.\n"
                        "Modo: apenas sinais (sem execucao).\n\n"
                        "Voce recebera alertas quando oportunidades forem detectadas, "
                        "com detalhamento do regime, estrategia e indicadores."
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
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN,
            )
            logger.info(
                "Sinal %s enviado ao Telegram (chat_id=%s).",
                signal.type.value,
                self.chat_id,
            )
        except Exception as exc:
            logger.exception("Falha ao enviar mensagem ao Telegram: %s", exc)
            try:
                await self.bot.send_message(chat_id=self.chat_id, text=text)
            except Exception as fallback_exc:
                logger.exception(
                    "Fallback de texto puro tambem falhou: %s", fallback_exc
                )

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
                "\u26a0\ufe0f  Limite de perda diaria atingido.\n\n"
                "O bot pausou automaticamente para proteger contra "
                "sinais em periodo desfavoravel.\n\n"
                "\U0001f504  Retoma automaticamente as 00:00 UTC (21h de Brasilia)."
            ),
            "drawdown_semanal": (
                "\u26a0\ufe0f  Limite de perda semanal atingido.\n\n"
                "O bot pausou automaticamente para proteger contra "
                "sinais em periodo desfavoravel.\n\n"
                "\U0001f504  Retoma automaticamente na segunda-feira UTC."
            ),
            "perdas_consecutivas": (
                "\u26a0\ufe0f  Muitas perdas seguidas.\n\n"
                "O bot entrou em cooldown para evitar sinais "
                "em sequencia ruim. Retomara automaticamente "
                "apos o cooldown."
            ),
            "circuit_breaker": (
                "\U0001f6a8  Movimento extremo de preco detectado.\n\n"
                "O mercado esta muito volatil. O bot pausou "
                "temporariamente para evitar sinais durante "
                "instabilidade."
            ),
            "filtro_volatilidade": (
                "\U0001f6a8  Volatilidade fora da faixa segura.\n\n"
                "O mercado esta muito {'lateral (sem direcao clara)' if 'lateral' in message.lower() else 'volatil (instavel)'}. "
                "O bot aguarda melhora nas condicoes."
            ),
            "cooldown_entre_sinais": (
                "\u23f3  Intervalo minimo entre sinais.\n\n"
                "O bot ja gerou um sinal recente e precisa "
                "esperar algumas velas antes do proximo."
            ),
            "kill_switch_manual": (
                "\U0001f6d1  Bot DESLIGADO manualmente.\n\n"
                "Nenhum sinal sera gerado. "
                "Para reativar, use o painel web (botao 'Reativar')."
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
                    f"\U0001f510  Conectado a {exchange_id.upper()}\n\n"
                    f"\U0001f4c1  Monitorando: {symbol} no timeframe 1H\n"
                    f"\U0001f310  Versao: V13-ROBUSTA (WFO validada)\n"
                    f"\u2022 Squeeze Breakout (SL 1.8x, TP 6.5x)\n"
                    f"\u2022 RSI Reversal (SL 1.8x, TP 5.5x)\n"
                    f"\u2022 Half-risk: 0.5% por trade\n"
                    f"\u2022 Protecao automatica ativa\n\n"
                    "O bot esta ativo e analisando o mercado.\n"
                    "Voce recebera sinais quando oportunidades forem detectadas."
                ),
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception:
            pass
