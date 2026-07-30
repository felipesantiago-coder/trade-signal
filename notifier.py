"""
notifier.py
-----------
Integracao assincrona com o Telegram usando python-telegram-bot.

MODO: Signal-Only (apenas analise, sem execucao de ordens).
Todas as mensagens sao projetadas para serem intuitivas e acionaveis,
explicando o que o sistema detectou e quais niveis de preco observar.
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
    """Explica em linguagem acessivel o que o sinal LONG significa."""
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
        "   Indica que a pressao vendedora esta enfraquecendo.",
        "",
        "4. Volume elevado confirma",
        f"   O volume esta {signal.volume / max(signal.volume_sma20, 1):.1f}x acima",
        "   da media. Isso sugere entrada forte de compradores.",
    ]
    return "\n".join(lines)


def _explain_short_conditions(signal: Signal) -> str:
    """Explica em linguagem acessivel o que o sinal SHORT significa."""
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


def _format_signal_message(signal: Signal, symbol: str) -> str:
    """Constroi a mensagem do Telegram para um sinal, em linguagem acessivel."""
    is_long = signal.type == SignalType.LONG

    # Cabecalho
    if is_long:
        header = (
            f"Sinal de COMPRA (LONG) detectado no {symbol}\n\n"
            f"O sistema encontrou uma oportunidade de compra apos recuo "
            f"numa tendencia de alta. Veja a analise detalhada abaixo."
        )
    else:
        header = (
            f"Sinal de VENDA (SHORT) detectado no {symbol}\n\n"
            f"O sistema encontrou uma oportunidade de venda apos alta "
            f"numa tendencia de baixa. Veja a analise detalhada abaixo."
        )

    # Explicacao das condicoes
    explanation = (
        _explain_long_conditions(signal) if is_long
        else _explain_short_conditions(signal)
    )

    # Niveis de preco para o usuario acompanhar
    entry = signal.entry_price
    sl = signal.stop_loss
    tp = signal.take_profit
    risk_usd = abs(entry - sl)
    reward_usd = abs(tp - entry)
    rr = reward_usd / max(risk_usd, 1e-9)

    prices = (
        f"\nNiveis para acompanhar:\n"
        f"  Preco atual: ${entry:,.2f}\n"
        f"  Zona de protecao (Stop Loss): ${sl:,.2f}\n"
        f"  Alvo (Take Profit): ${tp:,.2f}\n"
        f"  Relacao risco/ganho: 1:{rr:.2f}"
    )

    # Aviso claro de que e apenas analise
    disclaimer = (
        "\nEste e um sinal de analise automatica.\n"
        "O sistema NAO executa nenhuma operacao.\n"
        "Use esta informacao como referencia\n"
        "para sua propria decisao de trading."
    )

    footer = "\n\nCTEV Signal Bot v4.0 — Apenas analise"

    return (
        f"{header}\n\n"
        f"{explanation}\n\n"
        f"{prices}\n\n"
        f"{disclaimer}\n"
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
                    "CTEV Signal Bot foi iniciado!\n\n"
                    "O que este bot faz:\n"
                    "Monitora o Bitcoin 24h por dia e identifica os melhores\n"
                    "momentos para compra ou venda, usando analise tecnica\n"
                    "automatizada (CTEV — Confluencia de Tendencia e\n"
                    "Exaustao Volumetrica).\n\n"
                    "Exchange: {exchange}\n"
                    "Par: {symbol}\n"
                    "Timeframe: 1H (analise a cada 1 hora)\n\n"
                    "Voce recebera notificacoes quando:\n"
                    "- Um sinal de compra (LONG) for detectado\n"
                    "- Um sinal de venda (SHORT) for detectado\n"
                    "- O sistema precisar pausar por seguranca\n\n"
                    "Importante:\n"
                    "Este bot APENAS analisa o mercado e envia sinais.\n"
                    "Nenhuma operacao e executada automaticamente.\n"
                    "Use os sinais como referencia para suas proprias\n"
                    "decisoes de investimento.\n\n"
                    "Acesse o painel web para acompanhar em tempo real."
                ).format(
                    exchange=exchange_id.upper(),
                    symbol=symbol,
                ),
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception:
            try:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=(
                        "CTEV Signal Bot iniciado!\n\n"
                        "Monitorando Bitcoin 24h. Voce recebera alertas "
                        "quando oportunidades forem detectadas.\n\n"
                        "Modo: apenas analise (sem execucao de ordens).\n"
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
    # Alerta de risco
    # ------------------------------------------------------------------
    async def send_risk_alert(self, reason: str, message: str) -> None:
        """Envia alerta quando o RiskManager bloqueia ou pausa o bot."""
        if not self.bot:
            return
        explanations = {
            "drawdown_diario": (
                "Limite de perda diaria atingido.\n"
                "O bot pausou automaticamente para proteger contra\n"
                "sinais em periodo desfavoravel.\n"
                "Retoma automaticamente as 00:00 UTC (21h de Brasilia)."
            ),
            "drawdown_semanal": (
                "Limite de perda semanal atingido.\n"
                "O bot pausou automaticamente para proteger contra\n"
                "sinais em periodo desfavoravel.\n"
                "Retoma automaticamente na segunda-feira UTC."
            ),
            "perdas_consecutivas": (
                "Muitas perdas seguidas nos sinais recentes.\n"
                "O bot entrou em cooldown para evitar sinais\n"
                "em sequencia ruim. Retomara automaticamente\n"
                "apos o cooldown."
            ),
            "circuit_breaker": (
                "Movimento extremo de preco detectado.\n"
                "O mercado esta muito volatil. O bot pausou\n"
                "temporariamente para evitar sinais durante\n"
                "instabilidade."
            ),
            "filtro_volatilidade": (
                "Volatilidade fora da faixa segura.\n"
                "O mercado esta muito {'lateral (sem direcao clara)' if 'lateral' in message.lower() else 'volatil (instavel)'}. "
                "O bot aguarda melhora nas condicoes."
            ),
            "cooldown_entre_sinais": (
                "Intervalo minimo entre sinais.\n"
                "O bot ja gerou um sinal recente e precisa\n"
                "esperar algumas velas antes do proximo."
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
                    f"Protecao automatica ativada\n\n"
                    f"{explanation}"
                ),
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception:
            try:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=(
                        f"Protecao automatica ativada\n\n"
                        f"{explanation}"
                    ),
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
                    f"Conectado a {exchange_id.upper()}\n"
                    f"Monitorando: {symbol} no timeframe 1H\n\n"
                    "O bot esta ativo e analisando o mercado.\n"
                    "Modo: apenas sinais (sem execucao)."
                ),
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception:
            pass
