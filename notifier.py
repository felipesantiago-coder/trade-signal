"""
notifier.py
-----------
Integração assíncrona com o Telegram usando python-telegram-bot.

Envia mensagens formatadas com emojis 🟢 (LONG) / 🔴 (SHORT) contendo
preço de entrada, stop loss, take profit e métricas de confirmação.
"""

from __future__ import annotations

import logging
from typing import Optional

from telegram import Bot
from telegram.constants import ParseMode

from strategy import Signal, SignalType

logger = logging.getLogger(__name__)


def _format_signal_message(signal: Signal, symbol: str) -> str:
    """Constrói a mensagem Markdown do Telegram para um sinal."""
    if signal.type == SignalType.LONG:
        emoji = "🟢"
        label = "SINAL DE LONG"
        direction = "COMPRA"
    else:
        emoji = "🔴"
        label = "SINAL DE SHORT"
        direction = "VENDA"

    # Razão risco:retorno com base em ATR
    rr = abs(signal.take_profit - signal.entry_price) / max(
        abs(signal.entry_price - signal.stop_loss), 1e-9
    )

    msg = (
        f"{emoji} *{label}* — {direction}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📊 *Par:* `{symbol}`\n"
        f"⏱ *Timeframe:* 1H\n"
        f"🕐 *Candle:* `{signal.timestamp}`\n"
        f"\n"
        f"💰 *Entrada:* `{signal.entry_price:.2f}`\n"
        f"🛑 *Stop Loss:* `{signal.stop_loss:.2f}`\n"
        f"🎯 *Take Profit 1:* `{signal.take_profit:.2f}`\n"
        f"⚖️ *Risco:Retorno:* `{rr:.2f} : 1`\n"
        f"\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📋 *Confirmações:*\n"
        f"• EMA200: `{signal.ema200:.2f}`\n"
        f"• RSI(14): `{signal.rsi:.2f}`\n"
        f"• ATR(14): `{signal.atr:.2f}`\n"
        f"• ATR Percentile: `{signal.atr_percentile:.0%}`\n"
        f"• BB Width: `{signal.bb_width:.2f}%`\n"
        f"• Volume: `{signal.volume:,.0f}`\n"
        f"• Vol SMA20: `{signal.volume_sma20:,.0f}`\n"
        f"• BB Lower: `{signal.bb_lower:.2f}`\n"
        f"• BB Upper: `{signal.bb_upper:.2f}`\n"
        f"\n"
        f"_Estratégia CTEV v2.0 — Risk Manager ativo_"
    )
    return msg


class TelegramNotifier:
    """Wrapper assíncrono sobre telegram.Bot para envio de sinais."""

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self.bot = Bot(token=bot_token)
        self.chat_id = chat_id

    async def send_signal(self, signal: Signal, symbol: str) -> None:
        """Envia o sinal formatado para o chat configurado."""
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
            # Fallback: tentar enviar como texto puro sem Markdown
            try:
                await self.bot.send_message(chat_id=self.chat_id, text=text)
            except Exception as fallback_exc:
                logger.exception(
                    "Fallback de texto puro também falhou: %s", fallback_exc
                )

    async def send_text(self, text: str) -> None:
        """Envia uma mensagem de texto livre (útil para notificações de status/erros)."""
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=text)
        except Exception as exc:
            logger.exception("Falha ao enviar texto ao Telegram: %s", exc)
