"""
main.py
-------
Ponto de entrada do bot CTEV.

Responsabilidades:
- Carregar configurações (.env)
- Configurar logging
- Inicializar ccxt (Binance), TelegramNotifier e estratégia
- Rodar loop assíncrono infinito verificando a cada 1 minuto se um novo
  candle de 1H foi fechado
- Avaliar apenas o candle fechado e disparar sinal via Telegram

Execução:
    python main.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timezone
from typing import Optional

import ccxt.async_support as ccxt
import pandas as pd

from config import Settings, load_settings
from indicators import compute_indicators
from notifier import TelegramNotifier
from strategy import evaluate_signal

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ctev.main")


# Quantidade de candles buscados da Binance.
# Precisamos de pelo menos 200+ para EMA200; buscamos 300 para ter folga.
CANDLE_LIMIT = 300


class CTEVBot:
    """Orquestra o loop principal do bot CTEV."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.exchange: Optional[ccxt.binance] = None
        self.notifier: Optional[TelegramNotifier] = None
        # Guarda timestamp (ms) do último candle já processado para evitar
        # disparos duplicados.
        self.last_processed_ts: Optional[int] = None

    async def __aenter__(self) -> "CTEVBot":
        # Inicializa exchange
        self.exchange = ccxt.binance({
            "apiKey": self.settings.binance.api_key,
            "secret": self.settings.binance.api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        # Inicializa notifier
        self.notifier = TelegramNotifier(
            bot_token=self.settings.telegram.token,
            chat_id=self.settings.telegram.chat_id,
        )
        logger.info(
            "CTEV Bot inicializado | symbol=%s tf=%s",
            self.settings.binance.symbol,
            self.settings.binance.timeframe,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.exchange is not None:
            await self.exchange.close()
            logger.info("Conexão com a Binance encerrada.")

    # ------------------------------------------------------------------
    # Coleta de dados
    # ------------------------------------------------------------------
    async def fetch_candles(self) -> pd.DataFrame:
        """
        Busca os últimos CANDLE_LIMIT candles de 1H da Binance e devolve
        um DataFrame OCHLV com índice datetime em UTC.
        """
        if self.exchange is None:
            raise RuntimeError("Exchange não inicializada.")

        ohlcv = await self.exchange.fetch_ohlcv(
            symbol=self.settings.binance.symbol,
            timeframe=self.settings.binance.timeframe,
            limit=CANDLE_LIMIT,
        )
        if not ohlcv:
            raise RuntimeError("Resposta vazia da Binance ao buscar candles.")

        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("datetime", inplace=True)
        df.drop(columns=["timestamp"], inplace=True)
        return df

    # ------------------------------------------------------------------
    # Loop principal
    # ------------------------------------------------------------------
    async def run(self) -> None:
        logger.info("Iniciando loop principal do CTEV Bot...")
        # Aviso inicial no Telegram
        await self.notifier.send_text(
            f"🤖 *Bot CTEV iniciado* — monitorando {self.settings.binance.symbol} "
            f"({self.settings.binance.timeframe})."
        )

        while True:
            try:
                await self._cycle()
            except Exception as exc:
                logger.exception("Erro no ciclo principal: %s", exc)
                # Evita loop apertado em caso de falha contínua
                await asyncio.sleep(self.settings.loop_interval_seconds)
            await asyncio.sleep(self.settings.loop_interval_seconds)

    async def _cycle(self) -> None:
        """Executa um ciclo de verificação de candle fechado."""
        df = await self.fetch_candles()
        if df.empty:
            logger.warning("DataFrame vazio; pulando ciclo.")
            return

        # O último candle retornado pode ainda estar em formação.
        # Binance retorna candles alinhados ao início do período: o timestamp
        # do último candle indica quando ele ABRIU. Ele "fecha" em ts + 1h.
        last_open_ts_ms = int(df.index[-1].timestamp() * 1000)
        timeframe_ms = 60 * 60 * 1000  # 1h
        last_close_ts_ms = last_open_ts_ms + timeframe_ms
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        if now_ms < last_close_ts_ms:
            # Último candle ainda não fechou
            logger.debug(
                "Candle mais recente ainda aberto (abertura=%s, fecha em %s). Aguardando.",
                df.index[-1].isoformat(),
                datetime.fromtimestamp(last_close_ts_ms / 1000, tz=timezone.utc).isoformat(),
            )
            return

        # O candle efetivamente fechado é o ÚLTIMO do DataFrame.
        closed_candle_ts = df.index[-1]

        # Evita reprocessar o mesmo candle
        last_processed = (
            self.last_processed_ts.isoformat() if self.last_processed_ts else None
        )
        if self.last_processed_ts is not None and closed_candle_ts == self.last_processed_ts:
            logger.debug(
                "Candle %s já foi processado. Aguardando próximo fechamento.",
                closed_candle_ts.isoformat(),
            )
            return

        logger.info(
            "Novo candle fechado detectado: %s (processado anterior: %s)",
            closed_candle_ts.isoformat(),
            last_processed,
        )

        # Calcula indicadores sobre todo o histórico
        df_ind = compute_indicators(df)

        # Avalia sinal na última linha (candle fechado)
        signal = evaluate_signal(df_ind)

        if signal is None:
            logger.info(
                "Nenhum sinal CTEV no candle %s. Próxima verificação em %ds.",
                closed_candle_ts.isoformat(),
                self.settings.loop_interval_seconds,
            )
        else:
            # Dispara notificação
            await self.notifier.send_signal(signal, self.settings.binance.symbol)

        # Marca como processado
        self.last_processed_ts = closed_candle_ts


async def main() -> None:
    try:
        settings = load_settings()
    except RuntimeError as exc:
        logger.error("Falha ao carregar configurações: %s", exc)
        sys.exit(1)

    # Ajusta nível de log conforme configuração
    logging.getLogger().setLevel(settings.log_level)

    async with CTEVBot(settings) as bot:
        try:
            await bot.run()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Encerrando bot CTEV por interrupção do usuário.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
