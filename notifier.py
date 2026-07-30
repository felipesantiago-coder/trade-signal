"""
notifier.py
-----------
Integracao assincrona com o Telegram usando python-telegram-bot.

Todas as mensagens sao projetadas para serem intuitivas para usuarios
nao familiarizados com trading, explicando o que o sistema detectou,
o que esta fazendo automaticamente e o que esperar a seguir.
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
# Helpers de formatacao
# ==================================================================

def _fmt_usd(value: float) -> str:
    """Formata valor em USD com sinal."""
    sign = "+" if value >= 0 else ""
    return f"${sign}{value:,.2f}"


def _explain_long_conditions(signal: Signal) -> str:
    """Explica em linguagem simples o que o sinal LONG significa."""
    lines = [
        "O que o sistema detectou:",
        "",
        "1. Tendencia de ALTA confirmada",
        f"   O preco (${signal.entry_price:,.2f}) esta acima da media",
        f"   dos ultimos 200 periodos (${signal.ema200:,.2f}).",
        "",
        "2. Pullback (recuo temporario)",
        "   O preco recuou e tocou a faixa inferior de Bollinger.",
        "   Isso e normal em tendencias de alta e cria ponto de entrada.",
        "",
        "3. Exaustao da venda",
        f"   O RSI esta em {signal.rsi:.1f} (abaixo de 35).",
        "   Indica que a presso vendedora esta enfraquecendo.",
        "",
        "4. Volume elevado confirma",
        f"   O volume esta {signal.volume / max(signal.volume_sma20, 1):.1f}x acima",
        "   da media. Isso sugere entrada forte de compradores.",
    ]
    return "\n".join(lines)


def _explain_short_conditions(signal: Signal) -> str:
    """Explica em linguagem simples o que o sinal SHORT significa."""
    lines = [
        "O que o sistema detectou:",
        "",
        "1. Tendencia de BAIXA confirmada",
        f"   O preco (${signal.entry_price:,.2f}) esta abaixo da media",
        f"   dos ultimos 200 periodos (${signal.ema200:,.2f}).",
        "",
        "2. Pullback para cima (recuo temporario)",
        "   O preco subiu temporariamente e tocou a faixa superior.",
        "   Isso e normal em tendencias de baixa e cria ponto de entrada.",
        "",
        "3. Exaustao da compra",
        f"   O RSI esta em {signal.rsi:.1f} (acima de 65).",
        "   Indica que a pressao compradora esta enfraquecendo.",
        "",
        "4. Volume elevado confirma",
        f"   O volume esta {signal.volume / max(signal.volume_sma20, 1):.1f}x acima",
        "   da media. Isso sugere entrada forte de vendedores.",
    ]
    return "\n".join(lines)


def _explain_risk_management() -> str:
    """Explica em linguagem simples o que o bot faz automaticamente."""
    return (
        "Protecoes automaticas ativas:\n"
        "Stop Loss — se o preco atingir o SL, a posicao fecha automaticamente.\n"
        "Break-Even — se o preco subir o suficiente, o SL e movido para o\n"
        "  preco de entrada (zero risco).\n"
        "Trailing Stop — apos o break-even, o SL acompanha o preco para\n"
        "  proteger os lucros conquistados.\n"
        "Partial TP — 50% da posicao fecha no alvo; o restante corre."
    )


def _format_signal_message(signal: Signal, symbol: str) -> str:
    """Constrói a mensagem do Telegram para um sinal, em linguagem acessivel."""
    is_long = signal.type == SignalType.LONG

    # Cabecalho
    if is_long:
        header = (
            f"COMPRA (LONG) detectada no {symbol}\n\n"
            "O sistema encontrou uma boa oportunidade de compra.\n"
            "Veja a analise detalhada abaixo."
        )
    else:
        header = (
            f"VENDA (SHORT) detectada no {symbol}\n\n"
            "O sistema encontrou uma boa oportunidade de venda.\n"
            "Veja a analise detalhada abaixo."
        )

    # Explicacao das condicoes
    explanation = (
        _explain_long_conditions(signal) if is_long
        else _explain_short_conditions(signal)
    )

    # Precos
    entry = signal.entry_price
    sl = signal.stop_loss
    tp = signal.take_profit
    risk_usd = abs(entry - sl)
    reward_usd = abs(tp - entry)
    rr = reward_usd / max(risk_usd, 1e-9)

    prices = (
        f"\nValores da operacao:\n"
        f"Entrada: ${entry:,.2f}\n"
        f"Stop Loss (protecao): ${sl:,.2f}\n"
        f"  → Se atingir, perda maxima: ${risk_usd:,.2f} por BTC\n"
        f"Take Profit (alvo): ${tp:,.2f}\n"
        f"  → Se atingir, ganho estimado: ${reward_usd:,.2f} por BTC\n"
        f"Relacao risco/ganho: 1:{rr:.2f}"
    )

    # Protecoes
    protections = f"\n{_explain_risk_management()}"

    # Footer
    footer = "\nAnalise automatica by CTEV v3.0"

    return (
        f"{header}\n\n"
        f"{explanation}\n\n"
        f"{prices}\n\n"
        f"{protections}\n\n"
        f"{footer}"
    )


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
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=(
                    "CTEV Bot foi iniciado com sucesso!\n\n"
                    "O que este bot faz:\n"
                    "Monitora o Bitcoin 24h por dia e identifica os melhores\n"
                    "momentos para entrar em operacoes de compra ou venda,\n"
                    "usando analise tecnica automatizada.\n\n"
                    f"Exchange: {exchange_id.upper()}\n"
                    f"Par: {symbol}\n"
                    f"Timeframe: 1H (analise a cada 1 hora)\n\n"
                    "Voce recebera notificacoes quando:\n"
                    "- Uma oportunidade de compra (LONG) for detectada\n"
                    "- Uma oportunidade de venda (SHORT) for detectada\n"
                    "- Uma posicao for aberta, ajustada ou fechada\n"
                    "- O sistema precisar pausar por protecao de capital\n\n"
                    "Dicas:\n"
                    "- O bot opera em modo SIMULACAO (dry-run) por padrao.\n"
                    "- Nenhuma ordem real e enviada automaticamente.\n"
                    "- Acesse o painel web para acompanhar em tempo real.\n\n"
                    "CTEV = Confluencia de Tendencia e Exaustao Volumetrica"
                ),
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception:
            try:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=(
                        "CTEV Bot foi iniciado com sucesso!\n\n"
                        "Monitorando Bitcoin 24h. Voce recebera alertas quando "
                        "oportunidades forem detectadas.\n\n"
                        f"Exchange: {exchange_id.upper()} | Par: {symbol}"
                    ),
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Sinal de trade
    # ------------------------------------------------------------------
    async def send_signal(self, signal: Signal, symbol: str) -> None:
        """Envia o sinal formatado para o chat configurado."""
        if not self.bot:
            return
        text = _format_signal_message(signal, symbol)
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
    # Posicao aberta
    # ------------------------------------------------------------------
    async def send_position_open(self, signal: Signal, position_info: dict, symbol: str) -> None:
        """Envia notificacao de abertura de posicao em linguagem acessivel."""
        if not self.bot:
            return
        is_long = signal.type == SignalType.LONG
        direction = "COMPRA (LONG)" if is_long else "VENDA (SHORT)"

        # Sizing em linguagem simples
        pos_size = position_info.get("position_size", 0)
        pos_usd = position_info.get("position_usd", 0)
        risk_usd = position_info.get("risk_usd", 0)
        risk_pct = position_info.get("risk_pct", 0)

        sizing = (
            f"Posicao aberta:\n"
            f"Quantidade: {pos_size:.8f} BTC\n"
            f"Valor total: ${pos_usd:,.2f}\n"
            f"Risco maximo: ${risk_usd:,.2f} ({risk_pct * 100:.1f}% do saldo)\n\n"
        )

        # MTF em linguagem simples
        mtf = position_info.get("mtf")
        mtf_str = ""
        if mtf:
            h4 = mtf.get("h4_trend", "?")
            d1 = mtf.get("d1_trend", "?")
            confluence = mtf.get("confluence", "?")
            h4_desc = "alta" if "alta" in str(h4).lower() else "baixa" if "baixa" in str(h4).lower() else h4
            d1_desc = "alta" if "alta" in str(d1).lower() else "baixa" if "baixa" in str(d1).lower() else d1
            mtf_str = (
                f"Confirmacao em outros tempos:\n"
                f"- 4 horas: tendencia de {h4_desc}\n"
                f"- Diario: tendencia de {d1_desc}\n"
                f"- Confluencia: {confluence}\n\n"
            )

        # Modo
        exec_mode = position_info.get("executor", "SIMULACAO")
        if "DRY" in str(exec_mode).upper():
            mode_str = (
                "Modo: SIMULACAO (dry-run)\n"
                "Nenhuma ordem real foi enviada.\n"
                "Para ordens reais, ative via painel web.\n"
            )
        else:
            mode_str = f"Modo: REAL (live)\n"

        order_id = position_info.get("order_id")
        if order_id:
            mode_str += f"ID da ordem: {order_id}\n"

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=(
                    f"Posicao de {direction} ABERTA\n\n"
                    f"{sizing}{mtf_str}{mode_str}\n"
                    f"Entrada: ${signal.entry_price:,.2f}\n"
                    f"Stop Loss: ${signal.stop_loss:,.2f}\n"
                    f"Take Profit: ${signal.take_profit:,.2f}"
                ),
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception:
            try:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=(
                        f"Posicao de {direction} ABERTA\n\n"
                        f"{sizing}{mtf_str}{mode_str}\n"
                        f"Entrada: ${signal.entry_price:,.2f}\n"
                        f"SL: ${signal.stop_loss:,.2f}\n"
                        f"TP: ${signal.take_profit:,.2f}"
                    ),
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Trailing stop update
    # ------------------------------------------------------------------
    async def send_trailing_update(self, pos_type: str, trailing_sl: float, entry_price: float) -> None:
        """Envia notificacao de atualizacao de trailing stop em linguagem acessivel."""
        if not self.bot:
            return
        is_long = pos_type == "LONG"
        emoji = "🟢" if is_long else "🔴"
        direction = "compra" if is_long else "venda"

        if is_long:
            gain = trailing_sl - entry_price
            desc = (
                f"O stop loss foi movido para ${trailing_sl:,.2f}\n"
                f"(preco de entrada: ${entry_price:,.2f})\n\n"
                f"Se o preco cair ate ${trailing_sl:,.2f},\n"
                f"a posicao fecha com lucro de ${gain:,.2f} por BTC."
            )
        else:
            gain = entry_price - trailing_sl
            desc = (
                f"O stop loss foi movido para ${trailing_sl:,.2f}\n"
                f"(preco de entrada: ${entry_price:,.2f})\n\n"
                f"Se o preco subir ate ${trailing_sl:,.2f},\n"
                f"a posicao fecha com lucro de ${gain:,.2f} por BTC."
            )

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=(
                    f"Protecao ajustada {emoji} (posicao de {direction})\n\n"
                    f"{desc}"
                ),
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Fechamento de trade
    # ------------------------------------------------------------------
    async def send_trade_close(self, pos_type: str, entry: float, exit_p: float,
                               pnl_pct: float, pnl_usd: float, reason: str,
                               be: bool = False, trailing: bool = False,
                               partial: bool = False) -> None:
        """Envia notificacao de fechamento em linguagem acessivel."""
        if not self.bot:
            return
        is_long = pos_type == "LONG"
        emoji = "🟢" if is_long else "🔴"
        direction = "compra" if is_long else "venda"

        # Resultado
        is_profit = pnl_pct >= 0
        if reason == "tp":
            result_icon = "🎯"
            result_label = "ALVO ATINGIDO (Take Profit)"
            explanation = (
                "O preco atingiu o nivel de alvo definido.\n"
                "A operacao foi fechada no melhor momento planejado."
            )
        else:
            result_icon = "🛑"
            result_label = "STOP LOSS ATINGIDO"
            explanation = (
                "O preco atingiu o nivel de protecao.\n"
                "A operacao foi fechada para limitar a perda."
            )

        # Features
        features = []
        if be:
            features.append("Break-even ativado (risco zero)")
        if trailing:
            features.append("Trailing stop protegeu lucros")
        if partial:
            features.append("50% fechado no alvo parcial")

        feat_str = ""
        if features:
            feat_str = "\nRecursos usados:\n" + "\n".join(f"• {f}" for f in features)

        # PnL em linguagem simples
        pnl_sign = "+" if pnl_pct >= 0 else ""
        if is_profit:
            pnl_desc = (
                f"Resultado: LUCRO de {pnl_sign}{pnl_pct:.2f}%\n"
                f"Em dolares: {_fmt_usd(pnl_usd)}"
            )
        else:
            pnl_desc = (
                f"Resultado: PERDA de {pnl_pct:.2f}%\n"
                f"Em dolares: {_fmt_usd(pnl_usd)}"
            )

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=(
                    f"{result_icon} {result_label} {emoji}\n\n"
                    f"Posicao de {direction} fechada\n\n"
                    f"{explanation}\n\n"
                    f"Entrada: ${entry:,.2f}\n"
                    f"Saida: ${exit_p:,.2f}\n\n"
                    f"{pnl_desc}\n"
                    f"{feat_str}"
                ),
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Alerta de risco
    # ------------------------------------------------------------------
    async def send_risk_alert(self, reason: str, message: str) -> None:
        """Envia alerta quando o RiskManager bloqueia ou pausa o bot."""
        if not self.bot:
            return
        # Explica o motivo em linguagem simples
        explanations = {
            "drawdown_diario": (
                "Limite de perda diaria atingido.\n"
                "O bot pausou automaticamente para proteger seu capital.\n"
                "Retoma automaticamente as 00:00 UTC (21h horario de Brasilia)."
            ),
            "drawdown_semanal": (
                "Limite de perda semanal atingido.\n"
                "O bot pausou automaticamente para proteger seu capital.\n"
                "Retoma automaticamente na segunda-feira UTC."
            ),
            "perdas_consecutivas": (
                "Muitas perdas seguidas.\n"
                "O bot entrou em cooldown para evitar decisoes ruins\n"
                "em sequencia. Retomara automaticamente apos o cooldown."
            ),
            "circuit_breaker": (
                "Movimento extremo de preco detectado.\n"
                "O mercado esta muito volatil. O bot pausou temporariamente\n"
                "para evitar entrar em operacoes durante instabilidade."
            ),
            "filtro_volatilidade": (
                "Volatilidade fora da faixa segura.\n"
                "O mercado esta muito {'lateral (sem direcao)' if 'lateral' in message.lower() else 'volatil (instavel)'}. "
                "O bot aguarda melhora nas condicoes."
            ),
            "cooldown_entre_sinais": (
                "Intervalo minimo entre sinais.\n"
                "O bot ja gerou um sinal recente e precisa esperar\n"
                "algumas velas antes do proximo."
            ),
            "kill_switch_manual": (
                "Bot DESLIGADO manualmente pelo operador.\n"
                "Nenhum sinal sera gerado.\n"
                "Para reativar, use o painel web (botao 'Reativar')."
            ),
        }

        explanation = explanations.get(reason, message)

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=(
                    f"Protecao de capital ativada\n\n"
                    f"{explanation}\n\n"
                    f"Detalhe tecnico: {message}"
                ),
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception:
            try:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=(
                        f"Protecao de capital ativada\n\n"
                        f"{explanation}"
                    ),
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Resumo diario
    # ------------------------------------------------------------------
    async def send_daily_summary(self, summary: dict) -> None:
        """Envia resumo do dia com linguagem acessivel."""
        if not self.bot:
            return
        total = summary.get("total_trades", 0)
        wins = summary.get("total_wins", 0)
        losses = summary.get("total_losses", 0)
        win_rate = summary.get("win_rate", 0)
        pnl_pct = summary.get("total_pnl_pct", 0)
        pnl_usd = summary.get("total_pnl_usd", 0)
        daily_loss = summary.get("daily_loss_pct", 0)

        if total == 0:
            body = (
                "Nenhuma operacao foi realizada hoje.\n\n"
                "O bot monitorou o mercado mas nao encontrou\n"
                "condicoes que atendessem todos os criterios.\n\n"
                "Isso e normal — e melhor nao operar do que\n"
                "entrar em operacoes sem boa confirmacao."
            )
        else:
            is_profit = pnl_pct >= 0
            result_word = "lucro" if is_profit else "perda"
            body = (
                f"Resumo do dia:\n\n"
                f"Operacoes: {total} ({wins} ganhos, {losses} perdas)\n"
                f"Taxa de acerto: {win_rate:.1f}%\n"
                f"Resultado: {result_word} de {pnl_pct:+.2f}%\n"
                f"Em dolares: {_fmt_usd(pnl_usd)}\n"
                f"Perda diaria acumulada: {daily_loss:.2f}%\n\n"
                f"{'Bom resultado!' if is_profit else 'Dia negativo. O sistema de protecao esta ativo para limitar perdas.'}"
            )

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=(
                    f"Relatorio diario CTEV Bot\n\n"
                    f"{body}"
                ),
                parse_mode=ParseMode.MARKDOWN,
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
                    f"Exchange conectada: {exchange_id.upper()}\n"
                    f"Monitorando: {symbol} no timeframe 1H\n\n"
                    "O bot esta ativo e monitorando o mercado."
                ),
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception:
            pass
