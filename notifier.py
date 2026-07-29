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

    async def send_position_open(self, signal: Signal, position_info: dict, symbol: str) -> None:
        """Envia notificacao detalhada de abertura de posicao com sizing."""
        base_msg = _format_signal_message(signal, symbol)
        sizing = (
            f"\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📐 *Posicao:*\n"
            f"• Tamanho: `{position_info.get('position_size', 0):.8f}` BTC\n"
            f"• Valor: `$`{position_info.get('position_usd', 0):,.2f}`\n"
            f"• Risco: `$`{position_info.get('risk_usd', 0):,.2f}` ({position_info.get('risk_pct', 0) * 100:.1f}%)\n"
            f"• Alavancagem: `{position_info.get('leverage', 0):.2f}x`\n"
        )
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=base_msg + sizing,
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception:
            try:
                await self.bot.send_message(chat_id=self.chat_id, text=base_msg + sizing)
            except Exception:
                pass

    async def send_trailing_update(self, pos_type: str, trailing_sl: float, entry_price: float) -> None:
        """Envia notificacao de atualizacao de trailing stop."""
        emoji = "🟢" if pos_type == "LONG" else "🔴"
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=(
                    f"📊 *TRAILING STOP* {emoji}\n"
                    f"Posicao {pos_type} | Entrada: `{entry_price:.2f}`\n"
                    f"Novo SL: `{trailing_sl:.2f}`"
                ),
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception:
            pass

    async def send_trade_close(self, pos_type: str, entry: float, exit_p: float,
                               pnl_pct: float, pnl_usd: float, reason: str,
                               be: bool = False, trailing: bool = False,
                               partial: bool = False) -> None:
        """Envia notificacao de fechamento de trade."""
        emoji = "🟢" if pos_type == "LONG" else "🔴"
        if reason == "tp":
            icon = "🎯"
            label = "TAKE PROFIT"
        else:
            icon = "🛑"
            label = "STOP LOSS"

        features = []
        if be:
            features.append("BE ativado")
        if trailing:
            features.append("Trailing")
        if partial:
            features.append("50% partial")

        feat_str = " | ".join(features) if features else ""

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=(
                    f"{icon} *{label}* {emoji} {pos_type}\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"Entrada: `{entry:.2f}`\n"
                    f"Saida: `{exit_p:.2f}`\n"
                    f"PnL: `{'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%`\n"
                    f"PnL $: `{'+' if pnl_usd >= 0 else ''}{pnl_usd:,.2f}`\n"
                    f"{feat_str}\n"
                    f"\n_CTEV Bot v2.0 — Position Manager_"
                ),
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception:
            pass
