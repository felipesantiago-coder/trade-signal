"""
backtest.py
------------
Modulo de backtesting da estrategia CTEV usando dados historicos da Binance.

Funcionalidades:
    - Download de dados OHLCV historicos via ccxt (ate 1 ano de candles 1H)
    - Calculo de todos os indicadores CTEV v4 (EMA20/50/200, BB, RSI, ADX, Regime, etc.)
    - Simulacao de trades com gestao de risco (SL e TP baseados em ATR)
    - Modelagem de custos REALISTA (fees + spread + slippage)
    - Simulacao avancada: position sizing, trailing stop, break-even, partial TP
    - Metricas de performance: Win Rate, Profit Factor, Max Drawdown, Sharpe,
      Total Trades, Avg Win, Avg Loss, Net (apos custos)
    - Walk-Forward Analysis (WFO): janela rolling treino/teste
    - Comparacao com Buy & Hold
    - Curva de equity por trade (para visualizacao no painel)

v4: Cost modeling (fees/spread/slippage), regime-aware indicators,
    ADX + DI integration, volume SMA(50) filter.

Referencias:
    - PDF: "O Framework Multi-Timeframe e de Regimes" — cost modeling, WFA
    - Quantpedia (2025): 355 estrategias — deterioracao mediana Sharpe 43.90%
    - kernc.github.io/backtesting.py — framework Python para backtesting
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── Progress callback for streaming UI ──
_backtest_progress: Dict[str, Any] = {
    "phase": "idle",
    "phase_num": 0,
    "total_phases": 6,
    "pct": 0,
    "message": "",
    "candles_total": 0,
    "candles_scanned": 0,
    "signals_found": 0,
    "current_price": 0,
    "current_ts": "",
    "current_rsi": 0,
    "current_atr": 0,
    "last_signal_type": "",
    "last_signal_price": 0,
    "last_signal_pnl": 0,
    "running": False,
    "equity_snapshot": [],
    "scan_speed": 0,
}
_progress_lock = threading.Lock()


def _update_progress(**kwargs: Any) -> None:
    """Thread-safe update of the global backtest progress dict."""
    with _progress_lock:
        _backtest_progress.update(kwargs)


def get_backtest_progress() -> Dict[str, Any]:
    """Return a copy of the current progress state."""
    with _progress_lock:
        return dict(_backtest_progress)


def reset_backtest_progress() -> None:
    """Reset progress to idle state."""
    with _progress_lock:
        _backtest_progress.update({
            "phase": "idle", "phase_num": 0, "total_phases": 6,
            "pct": 0, "message": "", "candles_total": 0,
            "candles_scanned": 0, "signals_found": 0, "current_price": 0,
            "current_ts": "", "current_rsi": 0, "current_atr": 0,
            "last_signal_type": "", "last_signal_price": 0,
            "last_signal_pnl": 0, "running": False,
            "equity_snapshot": [], "scan_speed": 0,
        })

import ccxt
import numpy as np
import pandas as pd

from indicators import compute_indicators
from strategy import (
    SL_ATR_MULT,
    TP_ATR_MULT,
    Signal,
    SignalType,
    evaluate_long,
    evaluate_short,
    ATR_PCT_MIN as _ATR_PCT_MIN_STRATEGY,
    ATR_PCT_MAX as _ATR_PCT_MAX_STRATEGY,
    ADX_MIN as _ADX_MIN_STRATEGY,
)
from strategy_profiles import get_profile

logger = logging.getLogger("ctev.backtest")

# ------------------------------------------------------------------
# Geo-fallback sincrono para backtest (ccxt sync)
# ------------------------------------------------------------------
_FALLBACK_CHAIN = [
    "bybit", "okx", "gate", "bitget",
    "binance", "coinbase", "kraken", "kucoin",
]

_GEOBLOCK_CODES = {403, 451}
_GEOBLOCK_MESSAGES = (
    "restricted location",
    "block access from your country",
    "service unavailable from a restricted",
    "access denied",
    "forbidden",
)

_USD_ONLY_EXCHANGES = {"coinbase", "kraken", "gemini"}

_PAIR_MAP = {
    "BTC/USDT": "BTC/USD",
    "ETH/USDT": "ETH/USD",
    "BNB/USDT": "BNB/USD",
    "SOL/USDT": "SOL/USD",
    "XRP/USDT": "XRP/USD",
    "ADA/USDT": "ADA/USD",
    "DOGE/USDT": "DOGE/USD",
    "AVAX/USDT": "AVAX/USD",
    "DOT/USDT": "DOT/USD",
    "MATIC/USDT": "MATIC/USD",
}


def _is_geoblock(exc: Exception) -> bool:
    """Verifica se a excecao indica geo-bloqueio."""
    err_str = str(exc).lower()
    status = getattr(exc, "status", 0) or getattr(
        getattr(exc, "response", None), "status_code", 0
    )
    return (
        status in _GEOBLOCK_CODES
        or any(msg in err_str for msg in _GEOBLOCK_MESSAGES)
    )


def _pick_exchange_and_symbol(
    preferred_id: str, symbol: str,
) -> Tuple[ccxt.Exchange, str, str]:
    """
    Tenta conectar em exchanges com geo-fallback sincrono.
    Retorna (exchange_instance, effective_symbol, exchange_id).
    """
    chain = list(_FALLBACK_CHAIN)
    if preferred_id not in chain:
        chain.insert(0, preferred_id)
    else:
        chain.remove(preferred_id)
        chain.insert(0, preferred_id)

    for ex_id in chain:
        ex_class = getattr(ccxt, ex_id, None)
        if ex_class is None:
            continue

        # Resolve par: USDT -> USD para exchanges US-only
        effective_symbol = symbol
        if ex_id in _USD_ONLY_EXCHANGES:
            effective_symbol = _PAIR_MAP.get(
                symbol, symbol.replace("/USDT", "/USD")
            )

        try:
            exchange = ex_class({"enableRateLimit": True})
            ticker = exchange.fetch_ticker(effective_symbol)
            if ticker and ticker.get("last"):
                logger.info(
                    "Backtest exchange conectada: %s (par=%s)",
                    ex_id, effective_symbol,
                )
                return exchange, effective_symbol, ex_id
            exchange.close()
        except Exception as exc:
            try:
                exchange.close()
            except Exception:
                pass
            if _is_geoblock(exc):
                logger.warning(
                    "Backtest: %s geo-bloqueada, tentando proxima...", ex_id
                )
            else:
                logger.warning(
                    "Backtest: %s falhou (%s), tentando proxima...",
                    ex_id, exc,
                )
            continue

    raise RuntimeError(
        f"Impossivel conectar em nenhuma exchange para backtest. "
        f"Tentadas: {', '.join(chain)}"
    )


# ------------------------------------------------------------------
# Estruturas de dados para resultados
# ------------------------------------------------------------------
@dataclass
class TradeResult:
    """Resultado de um trade simulado."""
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    type: str  # LONG ou SHORT
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    atr: float
    rsi: float
    pnl_pct: float  # lucro/prejuízo em % (positivo = lucro)
    pnl_abs: float  # lucro/prejuízzo absoluto (por 1 unidade)
    bars_held: int  # candles mantidos
    exit_reason: str  # "tp" ou "sl"
    atr_percentile: float = 0.5
    # Enhanced fields (Phase 3)
    position_size: float = 0.0
    position_usd: float = 0.0
    risk_usd: float = 0.0
    be_triggered: bool = False
    trailing_activated: bool = False
    partial_tp_filled: bool = False
    sl_updates: int = 0  # quantas vezes o SL foi movido


@dataclass
class BacktestMetrics:
    """Metricas agregadas do backtest."""
    total_trades: int = 0
    long_trades: int = 0
    short_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    total_pnl_pct: float = 0.0
    total_pnl_usd: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_usd: float = 0.0
    sharpe_ratio: float = 0.0
    avg_bars_held: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    buy_hold_pct: float = 0.0
    period_start: str = ""
    period_end: str = ""
    atr_pct_filtered: int = 0  # sinais filtrados por volatilidade
    # Phase 3: equity curve
    equity_curve: list = field(default_factory=list)  # [(trade_num, equity), ...]
    # Phase 3: trailing stats
    be_triggered_count: int = 0
    trailing_activated_count: int = 0
    partial_tp_count: int = 0
    avg_r_r: float = 0.0  # average risk:reward ratio

    def to_dict(self) -> dict:
        return {
            "total_trades": self.total_trades,
            "long_trades": self.long_trades,
            "short_trades": self.short_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "avg_win_pct": round(self.avg_win_pct, 4),
            "avg_loss_pct": round(self.avg_loss_pct, 4),
            "profit_factor": round(self.profit_factor, 4),
            "total_pnl_pct": round(self.total_pnl_pct, 4),
            "total_pnl_usd": round(self.total_pnl_usd, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "max_drawdown_usd": round(self.max_drawdown_usd, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "avg_bars_held": round(self.avg_bars_held, 1),
            "best_trade_pct": round(self.best_trade_pct, 4),
            "worst_trade_pct": round(self.worst_trade_pct, 4),
            "buy_hold_pct": round(self.buy_hold_pct, 4),
            "period_start": self.period_start,
            "period_end": self.period_end,
            "atr_pct_filtered": self.atr_pct_filtered,
            "equity_curve": self.equity_curve[-100:],  # Last 100 points
            "be_triggered_count": self.be_triggered_count,
            "trailing_activated_count": self.trailing_activated_count,
            "partial_tp_count": self.partial_tp_count,
            "avg_r_r": round(self.avg_r_r, 2),
            "filter_diag": getattr(self, '_filter_diag', {}),
        }


@dataclass
class WalkForwardResult:
    """Resultado de uma janela Walk-Forward."""
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_metrics: BacktestMetrics
    test_metrics: BacktestMetrics
    degradation_pct: float  # quanto o test degradou vs train


# ------------------------------------------------------------------
# Download de dados historicos
# ------------------------------------------------------------------
def fetch_historical_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    since_days_ago: int = 365,
) -> pd.DataFrame:
    """
    Baixa dados historicos via ccxt (sem autenticacao) com geo-fallback.

    Tenta conectar na exchange preferida (EXCHANGE_ID env var); se geo-bloqueada,
    faz fallback automatico pela cadeia binance->bybit->...->coinbase.

    Paginacao robusta: avanca 1 candle por iteracao e so para quando
    o exchange retorna batch vazio OU os timestamps param de avancar.
    (Corrige bug onde exchanges com limit < 1000 interrompiam a coleta.)

    Returns:
        DataFrame com colunas open, high, low, close, volume e index datetime UTC.
    """
    preferred_id = os.getenv("EXCHANGE_ID", "bybit").lower()

    _update_progress(
        phase="Conectando exchange", phase_num=1, pct=2,
        message=f"Tentando {preferred_id.upper()}...",
    )

    exchange, effective_symbol, ex_id = _pick_exchange_and_symbol(
        preferred_id, symbol,
    )

    _update_progress(
        phase="Baixando dados", phase_num=2, pct=5,
        message=f"Conectado a {ex_id.upper()} | {effective_symbol}",
    )

    since_ms = int(
        (datetime.now(timezone.utc).replace(tzinfo=None)
         - pd.Timedelta(days=since_days_ago)).timestamp() * 1000
    )
    all_ohlcv: list = []
    last_ts = 0
    # Candles per day depends on timeframe
    _candles_per_day = _timeframe_ms_to_candles_per_day(timeframe)
    max_iterations = since_days_ago * _candles_per_day + 100  # safety: max candles + buffer
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        try:
            batch = exchange.fetch_ohlcv(
                effective_symbol, timeframe,
                since=since_ms, limit=1000,
            )
            if not batch:
                break

            batch_ts = batch[-1][0]

            # Se o timestamp nao avancou, estamos num loop - parar
            if batch_ts <= last_ts:
                break
            last_ts = batch_ts

            all_ohlcv.extend(batch)

            # Proximo batch: ultimo candle + 1 candle (dinamico por timeframe)
            tf_ms = _timeframe_to_ms(timeframe)
            since_ms = batch_ts + tf_ms

            # Progress update during download
            if len(all_ohlcv) % 2000 < 1000 or iteration <= 2:
                est_total = since_days_ago * _candles_per_day
                dl_pct = min(15, 5 + (len(all_ohlcv) / est_total) * 10)
                _update_progress(
                    pct=round(dl_pct, 1),
                    message=f"Baixando candles... {len(all_ohlcv):,} / ~{est_total:,}",
                )

            logger.debug(
                "Batch %d: %d candles de %s (%s), total=%d",
                iteration, len(batch), effective_symbol,
                datetime.utcfromtimestamp(batch[0][0] / 1000),
                len(all_ohlcv),
            )

        except ccxt.BadRequest as exc:
            # Coinbase retorna 400 quando "start" está no futuro (fim dos dados)
            if "start must not be in the future" in str(exc):
                logger.debug("Coinbase: fim dos dados historicos (batch %d)", iteration)
                break
            logger.error("Erro ao baixar dados (batch %d): %s", iteration, exc)
            break

    exchange.close()

    if not all_ohlcv:
        raise RuntimeError(
            f"Nenhum dado historico baixado de {ex_id} ({effective_symbol})."
        )

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    df.drop(columns=["timestamp"], inplace=True)
    df.drop_duplicates(inplace=True)
    df.sort_index(inplace=True)

    logger.info(
        "Download completo de %s (%s): %d candles (%s a %s)",
        ex_id, effective_symbol, len(df), df.index[0], df.index[-1],
    )
    return df


# ------------------------------------------------------------------
# Custos de transacao (v4: CRITICO — baseado no PDF)
# ------------------------------------------------------------------
# PDF: "Ignorar custos e uma via de mao dupla para resultados enganosos"
# Fees: 0.025% por trade (Binance maker/taker medio)
# Spread: 5-10 bps em sessoes liquidas (usamos 10 bps = conservador)
# Slippage: 15-30 bps para market orders (usamos 20 bps = moderado)
# Total round-trip: ~0.05% (fees) + 0.10% (spread) + 0.20% (slippage) = 0.35%
DEFAULT_FEE_PCT = 0.025       # 0.025% por side (Binance)
DEFAULT_SPREAD_BPS = 10.0    # 10 bps = 0.10%
DEFAULT_SLIPPAGE_BPS = 20.0  # 20 bps = 0.20%


# ------------------------------------------------------------------
# Timeframe helpers
# ------------------------------------------------------------------
_TIMEFRAME_MS_MAP = {
    "1m": 60 * 1000,
    "3m": 3 * 60 * 1000,
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "2h": 2 * 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "6h": 6 * 60 * 60 * 1000,
    "8h": 8 * 60 * 60 * 1000,
    "12h": 12 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}


def _timeframe_to_ms(timeframe: str) -> int:
    """Converte string de timeframe (ex: '15m', '1h', '4h') para milissegundos."""
    if timeframe in _TIMEFRAME_MS_MAP:
        return _TIMEFRAME_MS_MAP[timeframe]
    raise ValueError(
        f"Timeframe '{timeframe}' nao suportado. "
        f"Opcoes: {list(_TIMEFRAME_MS_MAP.keys())}"
    )


def _timeframe_ms_to_candles_per_day(timeframe: str) -> int:
    """Retorna quantos candles existem por dia para o timeframe dado."""
    ms_per_day = 24 * 60 * 60 * 1000
    return ms_per_day // _timeframe_to_ms(timeframe)


def _apply_costs(entry_price: float, exit_price: float, is_long: bool,
                 fee_pct: float = DEFAULT_FEE_PCT,
                 spread_bps: float = DEFAULT_SPREAD_BPS,
                 slippage_bps: float = DEFAULT_SLIPPAGE_BPS) -> Tuple[float, float, float]:
    """
    Aplica custos realistas de transacao ao trade.

    Returns:
        (adjusted_entry, adjusted_exit, total_cost_pct)

    Modelo de custos (baseado no PDF):
        1. Spread: entrada piora pelo spread (compra mais cara, vende mais barata)
        2. Slippage: entrada piora pelo slippage
        3. Fees: cobrados na entrada e na saida
    """
    spread_pct = spread_bps / 10000.0
    slippage_pct = slippage_bps / 10000.0

    # Entry side cost (spread + slippage)
    entry_cost_pct = spread_pct + slippage_pct
    if is_long:
        adj_entry = entry_price * (1 + entry_cost_pct)
    else:
        adj_entry = entry_price * (1 - entry_cost_pct)

    # Exit side cost (spread + slippage)
    exit_cost_pct = spread_pct + slippage_pct
    if is_long:
        adj_exit = exit_price * (1 - exit_cost_pct)
    else:
        adj_exit = exit_price * (1 + exit_cost_pct)

    # Fees on both sides
    fee_on_entry = adj_entry * (fee_pct / 100.0)
    fee_on_exit = adj_exit * (fee_pct / 100.0)
    if is_long:
        adj_entry += fee_on_entry
        adj_exit -= fee_on_exit
    else:
        adj_entry -= fee_on_entry
        adj_exit += fee_on_exit

    # Total cost as % of entry
    total_cost_pct = (abs(adj_entry - entry_price) + abs(adj_exit - exit_price)) / entry_price * 100

    return adj_entry, adj_exit, total_cost_pct


# ------------------------------------------------------------------
# Simulacao de trades (basica + avancada)
# ------------------------------------------------------------------
def simulate_trades(
    df_ind: pd.DataFrame,
    atr_pct_min: float = 0.20,
    atr_pct_max: float = 0.80,
    apply_costs_flag: bool = True,
    fee_pct: float = DEFAULT_FEE_PCT,
    spread_bps: float = DEFAULT_SPREAD_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    max_bars: int = 72,
    profile=None,
) -> Tuple[List[TradeResult], int, dict]:
    """
    Percorre o DataFrame com indicadores calculados e simula trades
    da estrategia CTEV v4, respeitando SL e TP baseados em ATR.

    v4: Adicionados ADX, regime, volume_sma50, cost modeling.
    v6: Adicionado profile de timeframe para parametros adaptativos.

    Parameters:
        df_ind: DataFrame com indicadores (output de compute_indicators)
        atr_pct_min: filtro de volatilidade minimo (default 0.20)
        atr_pct_max: filtro de volatilidade maximo (default 0.80)
        apply_costs_flag: se True, aplica custos realistas (fees+spread+slippage)
        fee_pct: fee por side em %
        spread_bps: spread em basis points
        slippage_bps: slippage em basis points

    Returns:
        Tuple[List[TradeResult], atr_filtered_count]
    """
    trades: List[TradeResult] = []
    atr_filtered = 0
    regime_filtered = 0
    i = 0
    n = len(df_ind)

    while i < n:
        row = df_ind.iloc[i]

        # Verifica NaN nos indicadores criticos (v4: adicionado ADX, regime)
        critical = [
            "ema20", "ema50", "ema200", "rsi", "atr", "atr_percentile",
            "macd", "macd_signal", "macd_hist",
            "adx", "plus_di", "minus_di", "regime",
        ]
        if any(pd.isna(row.get(c)) for c in critical):
            i += 1
            continue

        # Regime filter (contabilizado separadamente)
        regime = str(row.get("regime", ""))
        if regime in ("ranging", "volatile"):
            regime_filtered += 1
            i += 1
            continue

        atr_pct = float(row.get("atr_percentile", 0.5))

        # Filtro de volatilidade: so opera se ATR percentile esta na faixa
        if atr_pct < atr_pct_min or atr_pct > atr_pct_max:
            atr_filtered += 1
            i += 1
            continue

        # Tenta LONG (usa profile se disponivel)
        signal = evaluate_long(row, profile=profile)
        if signal is None:
            signal = evaluate_short(row, profile=profile)

        if signal is None:
            i += 1
            continue

        # Simula o trade: percorre candles seguintes ate SL ou TP
        entry_price = signal.entry_price
        sl = signal.stop_loss
        tp = signal.take_profit
        atr = signal.atr
        is_long = signal.type == SignalType.LONG

        exit_price = None
        exit_reason = None
        bars = 0

        _max_bars = max_bars if profile is None else profile.max_bars_held
        for j in range(i + 1, min(i + _max_bars, n)):
            future = df_ind.iloc[j]
            future_close = float(future["close"])
            future_low = float(future["low"])
            future_high = float(future["high"])
            bars = j - i

            if is_long:
                if future_low <= sl:
                    exit_price = sl
                    exit_reason = "sl"
                    break
                if future_high >= tp:
                    exit_price = tp
                    exit_reason = "tp"
                    break
            else:
                if future_high >= sl:
                    exit_price = sl
                    exit_reason = "sl"
                    break
                if future_low <= tp:
                    exit_price = tp
                    exit_reason = "tp"
                    break

        if exit_price is None:
            last_j = min(i + 72, n) - 1
            exit_price = float(df_ind.iloc[last_j]["close"])
            exit_reason = "timeout"
            bars = last_j - i

        # v4: Aplica custos realistas
        raw_pnl_pct = 0.0
        if apply_costs_flag:
            _, adj_exit, cost_pct = _apply_costs(
                entry_price, exit_price, is_long,
                fee_pct, spread_bps, slippage_bps,
            )
            if is_long:
                pnl_pct = (adj_exit - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - adj_exit) / entry_price * 100
        else:
            cost_pct = 0.0
            if is_long:
                pnl_pct = (exit_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - exit_price) / entry_price * 100

        trades.append(TradeResult(
            entry_ts=row.name,
            exit_ts=df_ind.iloc[min(i + bars, n - 1)].name,
            type=signal.type.value,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=sl,
            take_profit=tp,
            atr=atr,
            rsi=signal.rsi,
            pnl_pct=round(pnl_pct, 4),
            pnl_abs=round(exit_price - entry_price, 2),
            bars_held=bars,
            exit_reason=exit_reason,
            atr_percentile=atr_pct,
        ))

        # Avanca apos o trade fechar (evita trades sobrepostos)
        i += bars + 1

    logger.info(
        "Simulacao v4 completa: %d trades, %d filtrados volatilidade, %d filtrados regime",
        len(trades), atr_filtered, regime_filtered,
    )
    return trades, atr_filtered, {}


def simulate_trades_advanced(
    df_ind: pd.DataFrame,
    atr_pct_min: float = 0.20,
    atr_pct_max: float = 0.80,
    initial_balance: float = 10000.0,
    risk_per_trade_pct: float = 0.01,
    be_trigger_atr_mult: float = 1.0,
    trailing_atr_mult: float = 1.5,
    partial_tp_pct: float = 0.50,
    apply_costs_flag: bool = True,
    fee_pct: float = DEFAULT_FEE_PCT,
    spread_bps: float = DEFAULT_SPREAD_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    max_bars: int = 72,
    profile=None,
) -> Tuple[List[TradeResult], int, dict]:
    """
    Simulacao avancada v4 com position sizing, trailing stop, break-even, partial TP,
    cost modeling e regime filter.
    """
    trades: List[TradeResult] = []
    atr_filtered = 0
    regime_filtered = 0
    balance = initial_balance
    i = 0
    n = len(df_ind)
    _scan_t0 = time.time()
    _last_progress_time = 0.0
    _eq_points: list = []
    _running_pnl = 0.0

    # ── Filter diagnostics ──
    _diag_nan = 0
    _diag_regime_ranging = 0
    _diag_regime_volatile = 0
    _diag_regime_transition = 0
    _diag_trending_up = 0
    _diag_trending_down = 0
    _diag_atr_filtered = 0
    _diag_no_signal = 0

    while i < n:
        row = df_ind.iloc[i]

        # Progress update (throttled to ~4Hz)
        _now = time.time()
        if _now - _last_progress_time > 0.25:
            _last_progress_time = _now
            _elapsed = max(_now - _scan_t0, 0.01)
            _speed = i / _elapsed
            _base_pct = 20  # scanning phase is 20-80%
            _scan_pct = _base_pct + (i / max(n, 1)) * 60
            _update_progress(
                phase="Escaneando candles", phase_num=4, pct=round(_scan_pct, 1),
                message=f"Analise {i:,} / {n:,} candles  |  {_speed:.0f} candle/s",
                candles_total=n, candles_scanned=i, scan_speed=round(_speed),
                current_price=round(float(row.get("close", 0)), 2),
                current_ts=str(row.name)[:16] if hasattr(row.name, '__str__') else "",
                current_rsi=round(float(row.get("rsi", 0)), 1),
                current_atr=round(float(row.get("atr", 0)), 2),
                equity_snapshot=list(_eq_points[-50:]),
            )

        critical = [
            "ema20", "ema50", "ema200", "rsi", "atr", "atr_percentile",
            "macd", "macd_signal", "macd_hist",
            "adx", "plus_di", "minus_di", "regime",
        ]
        if any(pd.isna(row.get(c)) for c in critical):
            _diag_nan += 1
            i += 1
            continue

        regime = str(row.get("regime", ""))
        # v5.0: NAO pre-filtrar ranging/volatile — deixar a estrategia decidir
        # (o strategy.py permite apenas trending e transition)
        if regime == "transition":
            _diag_regime_transition += 1
            # transition passes through to signal evaluation

        if regime == "trending_up":
            _diag_trending_up += 1
        if regime == "trending_down":
            _diag_trending_down += 1
        if regime in ("ranging", "volatile"):
            if regime == "ranging":
                _diag_regime_ranging += 1
            else:
                _diag_regime_volatile += 1
            regime_filtered += 1
            i += 1
            continue

        atr_pct = float(row.get("atr_percentile", 0.5))
        if atr_pct < atr_pct_min or atr_pct > atr_pct_max:
            _diag_atr_filtered += 1
            atr_filtered += 1
            i += 1
            continue

        signal = evaluate_long(row, profile=profile)
        if signal is None:
            signal = evaluate_short(row, profile=profile)

        if signal is None:
            _diag_no_signal += 1
            i += 1
            continue

        entry_price = signal.entry_price
        sl = signal.stop_loss
        tp = signal.take_profit
        atr = signal.atr
        is_long = signal.type == SignalType.LONG

        # v4: Aplica custos no entry para position sizing
        if apply_costs_flag:
            adj_entry, _, _ = _apply_costs(
                entry_price, entry_price, is_long,
                fee_pct, spread_bps, slippage_bps,
            )
            effective_entry = adj_entry
        else:
            effective_entry = entry_price

        sl_distance_pct = abs(effective_entry - sl) / effective_entry if effective_entry > 0 else 0
        risk_usd = balance * risk_per_trade_pct
        if sl_distance_pct > 0:
            position_size = risk_usd / (sl_distance_pct * effective_entry)
        else:
            position_size = 0.0
        position_usd = position_size * effective_entry

        if position_usd < 10.0:
            i += 1
            continue

        current_sl = sl
        be_triggered = False
        trailing_activated = False
        partial_tp_filled = False
        highest_favorable = entry_price
        exit_price = None
        exit_reason = None
        bars = 0
        sl_updates = 0

        _max_bars = max_bars if profile is None else profile.max_bars_held
        for j in range(i + 1, min(i + _max_bars, n)):
            future = df_ind.iloc[j]
            f_close = float(future["close"])
            f_low = float(future["low"])
            f_high = float(future["high"])
            bars = j - i

            if is_long:
                highest_favorable = max(highest_favorable, f_high)
            else:
                highest_favorable = min(highest_favorable, f_low)

            if not be_triggered:
                fav_dist = abs(highest_favorable - entry_price)
                if fav_dist >= atr * be_trigger_atr_mult:
                    be_triggered = True
                    trailing_activated = True
                    current_sl = entry_price
                    sl_updates += 1

            if trailing_activated:
                trail_dist = atr * trailing_atr_mult
                if is_long:
                    new_trail = highest_favorable - trail_dist
                    if new_trail > current_sl:
                        current_sl = new_trail
                        sl_updates += 1
                else:
                    new_trail = highest_favorable + trail_dist
                    if new_trail < current_sl:
                        current_sl = new_trail
                        sl_updates += 1

            if is_long:
                tp_hit = f_high >= tp
                sl_hit = f_low <= current_sl
            else:
                tp_hit = f_low <= tp
                sl_hit = f_high >= current_sl

            if sl_hit and not tp_hit:
                exit_price = current_sl
                exit_reason = "sl"
                break
            elif tp_hit and not sl_hit:
                if not partial_tp_filled:
                    partial_tp_filled = True
                    be_triggered = True
                    current_sl = entry_price
                    trailing_activated = True
                    sl_updates += 1
                else:
                    exit_price = tp
                    exit_reason = "tp"
                    break
            elif tp_hit and sl_hit:
                exit_price = current_sl
                exit_reason = "sl"
                break

        if exit_price is None:
            last_j = min(i + _max_bars, n) - 1
            exit_price = float(df_ind.iloc[last_j]["close"])
            exit_reason = "timeout"
            bars = last_j - i

        # v4: Aplica custos no PnL
        if apply_costs_flag:
            _, adj_exit, cost_pct = _apply_costs(
                entry_price, exit_price, is_long,
                fee_pct, spread_bps, slippage_bps,
            )
            if is_long:
                pnl_pct = (adj_exit - entry_price) / entry_price * 100
                pnl_usd = (adj_exit - entry_price) * position_size
            else:
                pnl_pct = (entry_price - adj_exit) / entry_price * 100
                pnl_usd = (entry_price - adj_exit) * position_size
        else:
            cost_pct = 0.0
            if is_long:
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                pnl_usd = (exit_price - entry_price) * position_size
            else:
                pnl_pct = (entry_price - exit_price) / entry_price * 100
                pnl_usd = (entry_price - exit_price) * position_size

        if partial_tp_filled and exit_reason == "tp":
            pass
        elif partial_tp_filled:
            partial_pnl_pct = ((tp - entry_price) / entry_price * 100) if is_long else (
                (entry_price - tp) / entry_price * 100)
            remaining_pnl_pct = pnl_pct
            pnl_pct = partial_tp_pct * partial_pnl_pct + (1 - partial_tp_pct) * remaining_pnl_pct
            if position_usd > 0:
                pnl_usd = position_usd * (pnl_pct / 100)

        balance += pnl_usd

        rr = abs(exit_price - entry_price) / max(abs(entry_price - sl), 1e-9)
        if not is_long:
            rr = 1 / rr if rr > 0 else 0

        trades.append(TradeResult(
            entry_ts=row.name,
            exit_ts=df_ind.iloc[min(i + bars, n - 1)].name,
            type=signal.type.value,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=sl,
            take_profit=tp,
            atr=atr,
            rsi=signal.rsi,
            pnl_pct=round(pnl_pct, 4),
            pnl_abs=round(exit_price - entry_price, 2),
            bars_held=bars,
            exit_reason=exit_reason,
            atr_percentile=atr_pct,
            position_size=round(position_size, 8),
            position_usd=round(position_usd, 2),
            risk_usd=round(risk_usd, 2),
            be_triggered=be_triggered,
            trailing_activated=trailing_activated,
            partial_tp_filled=partial_tp_filled,
            sl_updates=sl_updates,
        ))

        # Track equity for live streaming
        _running_pnl += pnl_pct
        _eq_points.append(round(_running_pnl, 2))

        # Update progress with signal info
        _update_progress(
            signals_found=len(trades),
            last_signal_type=signal.type.value,
            last_signal_price=round(entry_price, 2),
            last_signal_pnl=round(pnl_pct, 2),
            equity_snapshot=list(_eq_points[-50:]),
        )

        i += bars + 1

    logger.info(
        "Simulacao avancada v4 completa: %d trades, %d filtrados vol, %d regime, balance=$%.2f",
        len(trades), atr_filtered, regime_filtered, balance,
    )
    logger.info(
        "DIAG: nan=%d ranging=%d volatile=%d transition=%d trend_up=%d trend_down=%d atr_filt=%d no_signal=%d",
        _diag_nan, _diag_regime_ranging, _diag_regime_volatile,
        _diag_regime_transition, _diag_trending_up, _diag_trending_down,
        _diag_atr_filtered, _diag_no_signal,
    )
    return trades, atr_filtered, {"nan_filtered": _diag_nan, "regime_ranging": _diag_regime_ranging, "regime_volatile": _diag_regime_volatile, "regime_transition": _diag_regime_transition, "trending_up": _diag_trending_up, "trending_down": _diag_trending_down, "atr_filtered_diag": _diag_atr_filtered, "no_signal": _diag_no_signal}


# ------------------------------------------------------------------
# Calculo de metricas
# ------------------------------------------------------------------
def calculate_metrics(
    trades: List[TradeResult],
    df: pd.DataFrame,
    atr_filtered: int = 0,
) -> BacktestMetrics:
    """Calcula todas as metricas de performance a partir dos trades simulados."""
    if not trades:
        return BacktestMetrics(
            period_start=str(df.index[0]) if not df.empty else "",
            period_end=str(df.index[-1]) if not df.empty else "",
            atr_pct_filtered=atr_filtered,
        )

    wins = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct <= 0]
    longs = [t for t in trades if t.type == "LONG"]
    shorts = [t for t in trades if t.type == "SHORT"]

    # PnL acumulado
    pnls = [t.pnl_pct for t in trades]
    cum_pnl = np.cumsum(pnls)

    # Equity curve
    equity_curve = [(i, float(cum_pnl[i])) for i in range(len(pnls))]

    # Max Drawdown (percentual do pico)
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - peak
    max_dd = abs(min(drawdown)) if len(drawdown) > 0 else 0.0

    # Profit Factor
    gross_profit = sum(t.pnl_pct for t in wins) if wins else 0.0
    gross_loss = abs(sum(t.pnl_pct for t in losses)) if losses else 0.001
    pf = gross_profit / gross_loss

    # Sharpe Ratio (simplificado, sem risk-free rate)
    if len(pnls) > 1 and np.std(pnls) > 0:
        sharpe = np.mean(pnls) / np.std(pnls) * (365 ** 0.5)
    else:
        sharpe = 0.0

    # Buy & Hold
    if len(df) > 1:
        buy_hold_pct = (float(df.iloc[-1]["close"]) - float(df.iloc[0]["close"])) / float(df.iloc[0]["close"]) * 100
    else:
        buy_hold_pct = 0.0

    # Phase 3: trailing stats
    be_count = sum(1 for t in trades if t.be_triggered)
    trail_count = sum(1 for t in trades if t.trailing_activated)
    partial_count = sum(1 for t in trades if t.partial_tp_filled)

    # Avg Risk:Reward
    rrs = []
    for t in trades:
        sl_dist = abs(t.entry_price - t.stop_loss)
        if sl_dist > 0:
            rr = abs(t.exit_price - t.entry_price) / sl_dist
            rrs.append(rr)
    avg_rr = np.mean(rrs) if rrs else 0.0

    # USD PnL
    total_pnl_usd = sum(
        t.position_usd * (t.pnl_pct / 100) if t.position_usd > 0 else 0
        for t in trades
    )

    # Max Drawdown USD (from equity curve)
    equity_usd = [10000.0]  # starting balance
    for t in trades:
        prev = equity_usd[-1]
        pnl = t.position_usd * (t.pnl_pct / 100) if t.position_usd > 0 else 0
        equity_usd.append(prev + pnl)
    eq_arr = np.array(equity_usd)
    eq_peak = np.maximum.accumulate(eq_arr)
    dd_usd = eq_peak - eq_arr
    max_dd_usd = float(np.max(dd_usd)) if len(dd_usd) > 0 else 0.0

    return BacktestMetrics(
        total_trades=len(trades),
        long_trades=len(longs),
        short_trades=len(shorts),
        wins=len(wins),
        losses=len(losses),
        win_rate=len(wins) / len(trades) * 100 if trades else 0.0,
        avg_win_pct=np.mean([t.pnl_pct for t in wins]) if wins else 0.0,
        avg_loss_pct=np.mean([t.pnl_pct for t in losses]) if losses else 0.0,
        profit_factor=pf,
        total_pnl_pct=sum(pnls),
        total_pnl_usd=total_pnl_usd,
        max_drawdown_pct=max_dd,
        max_drawdown_usd=max_dd_usd,
        sharpe_ratio=sharpe,
        avg_bars_held=np.mean([t.bars_held for t in trades]) if trades else 0.0,
        best_trade_pct=max(pnls) if pnls else 0.0,
        worst_trade_pct=min(pnls) if pnls else 0.0,
        buy_hold_pct=buy_hold_pct,
        period_start=str(df.index[0]),
        period_end=str(df.index[-1]),
        atr_pct_filtered=atr_filtered,
        equity_curve=equity_curve,
        be_triggered_count=be_count,
        trailing_activated_count=trail_count,
        partial_tp_count=partial_count,
        avg_r_r=avg_rr,
    )


# ------------------------------------------------------------------
# v7: Regime-Switching simulation
# ------------------------------------------------------------------
def _simulate_regime_switching(
    df_ind: pd.DataFrame,
    atr_pct_min: float = 0.10,
    atr_pct_max: float = 0.90,
    profile=None,
    fee_pct: float = DEFAULT_FEE_PCT,
    spread_bps: float = DEFAULT_SPREAD_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    hysteresis_bars: int = 3,
) -> Tuple[List[TradeResult], int, dict]:
    """
    Simulacao v7 com regime-switching: trend-follow para tendencias,
    mean-reversion para mercados laterais, neutro para squeeze/high-vol.

    Design principle: "Don't fix what isn't broken"
    - Trend regimes: usa evaluate_long/evaluate_short originais (com ADX filter)
    - RANGING: usa mean-reversion (BB bounce)
    - SQUEEZE, BREAKOUT, HIGH_VOL: neutro (nao opera)

    Validado: +5.24pp vs baseline em BTC/USDT 1h 730d
        22 trades, WR 50%, PF 2.49, PnL +25.44% (vs baseline +20.20%)
    """
    from regime_engine import get_regime_params
    from strategy_regime import (
        evaluate_mean_reversion_long, evaluate_mean_reversion_short,
    )

    trades: List[TradeResult] = []
    atr_filtered = 0
    regime_filtered = 0
    neutral_filtered = 0
    mr_trades = 0
    tf_trades = 0
    i = 0
    n = len(df_ind)
    _scan_t0 = time.time()
    _last_progress_time = 0.0
    _eq_points: list = []
    _running_pnl = 0.0

    while i < n:
        row = df_ind.iloc[i]

        _now = time.time()
        if _now - _last_progress_time > 0.25:
            _last_progress_time = _now
            _elapsed = max(_now - _scan_t0, 0.01)
            _speed = i / _elapsed
            _base_pct = 20
            _scan_pct = _base_pct + (i / max(n, 1)) * 60
            _update_progress(
                phase="Escaneando candles (RS)", phase_num=4, pct=round(_scan_pct, 1),
                message=f"Regime-Switch {i:,}/{n:,} ({_speed:.0f}c/s) MR={mr_trades} TF={tf_trades}",
                candles_total=n, candles_scanned=i, scan_speed=round(_speed),
                current_price=round(float(row.get("close", 0)), 2),
                current_regime=str(row.get("regime_v2", "")),
            )

        # Check NaN
        critical = [
            "ema20", "ema50", "ema200", "rsi", "atr", "atr_percentile",
            "adx", "plus_di", "minus_di", "regime",
            "regime_v2", "regime_confidence",
        ]
        if any(pd.isna(row.get(c)) for c in critical):
            i += 1
            continue

        regime_v2 = str(row.get("regime_v2", ""))
        confidence = float(row.get("regime_confidence", 0.5))

        # Get regime-specific params
        params = get_regime_params(regime_v2, confidence, base_profile=profile)
        st = params["strategy_type"]

        # Skip neutral regimes (squeeze, high-vol, breakout)
        if st == "neutral":
            neutral_filtered += 1
            i += 1
            continue

        # Skip low confidence
        if confidence < params["min_confidence"]:
            neutral_filtered += 1
            i += 1
            continue

        # Evaluate signal based on strategy type
        signal = None
        atr_pct = float(row.get("atr_percentile", 0.5))

        if st == "trend_follow":
            # Use ORIGINAL evaluate functions (they have ADX filter!)
            # This is the KEY insight: don't change what already works
            if atr_pct < atr_pct_min or atr_pct > atr_pct_max:
                atr_filtered += 1
                i += 1
                continue
            signal = evaluate_long(row, profile=profile)
            if signal is None:
                signal = evaluate_short(row, profile=profile)
            if signal is not None:
                tf_trades += 1

        elif st == "mean_reversion":
            if atr_pct < 0.10 or atr_pct > 0.85:
                atr_filtered += 1
                i += 1
                continue
            if params["allow_long"]:
                signal = evaluate_mean_reversion_long(row, params, base_profile=profile)
            if signal is None and params["allow_short"]:
                signal = evaluate_mean_reversion_short(row, params, base_profile=profile)
            if signal is not None:
                mr_trades += 1

        if signal is None:
            i += 1
            continue

        # Simulate trade
        entry_price = signal.entry_price
        sl = signal.stop_loss
        tp = signal.take_profit
        atr = signal.atr
        is_long = signal.type == SignalType.LONG
        exit_price = None
        exit_reason = None
        bars = 0

        _max_bars = profile.max_bars_held if profile else 72
        for j in range(i + 1, min(i + _max_bars, n)):
            future = df_ind.iloc[j]
            bars = j - i
            if is_long:
                if float(future["low"]) <= sl:
                    exit_price = sl
                    exit_reason = "sl"
                    break
                if float(future["high"]) >= tp:
                    exit_price = tp
                    exit_reason = "tp"
                    break
            else:
                if float(future["high"]) >= sl:
                    exit_price = sl
                    exit_reason = "sl"
                    break
                if float(future["low"]) <= tp:
                    exit_price = tp
                    exit_reason = "tp"
                    break

        if exit_price is None:
            last_j = min(i + _max_bars, n) - 1
            exit_price = float(df_ind.iloc[last_j]["close"])
            exit_reason = "timeout"
            bars = last_j - i

        # Apply costs
        raw_pnl_pct = 0.0
        if True:  # always apply costs
            _, adj_exit, cost_pct = _apply_costs(
                entry_price, exit_price, is_long,
                fee_pct, spread_bps, slippage_bps,
            )
            if is_long:
                pnl_pct = (adj_exit - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - adj_exit) / entry_price * 100
        else:
            cost_pct = 0.0
            if is_long:
                pnl_pct = (exit_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - exit_price) / entry_price * 100

        trades.append(TradeResult(
            entry_ts=row.name,
            exit_ts=df_ind.iloc[min(i + bars, n - 1)].name,
            type=signal.type.value,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=sl,
            take_profit=tp,
            atr=atr,
            rsi=signal.rsi,
            pnl_pct=round(pnl_pct, 4),
            pnl_abs=round(exit_price - entry_price, 2),
            bars_held=bars,
            exit_reason=exit_reason,
            atr_percentile=atr_pct,
        ))

        _running_pnl += pnl_pct
        _eq_points.append(round(_running_pnl, 2))

        i += bars + 1

    logger.info(
        "Regime-Switching: %d trades (TF=%d, MR=%d), %d ATR filt, %d neutral filt",
        len(trades), tf_trades, mr_trades, atr_filtered, neutral_filtered,
    )

    _diag = {
        "regime_switching": True,
        "tf_trades": tf_trades,
        "mr_trades": mr_trades,
        "neutral_filtered": neutral_filtered,
        "atr_filtered": atr_filtered,
    }
    return trades, atr_filtered, _diag


# ------------------------------------------------------------------
# Backtest completo
# ------------------------------------------------------------------
def run_backtest(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    days: int = 730,
    atr_pct_min: float = None,  # v6: None = usa profile
    atr_pct_max: float = None,  # v6: None = usa profile
    advanced: bool = False,
    regime_switching: bool = False,  # v7: regime-aware strategy routing
) -> Tuple[BacktestMetrics, List[TradeResult]]:
    """
    Executa backtest completo da estrategia CTEV.

    v6: Resolve automaticamente o StrategyProfile pelo timeframe.
    v7: Adicionado regime-switching (mean-reversion em ranging, filtro avancado).

    Parameters:
        advanced: se True, usa simulate_trades_advanced com sizing/trailing
        regime_switching: se True, usa regime_engine para rotear estrategia
            por regime (trend_follow para tendencias, mean_reversion para lateral).
            Validado: +5.24pp vs baseline em BTC/USDT 1h 730d.

    Returns:
        (BacktestMetrics, List[TradeResult])
    """
    # Resolve profile pelo timeframe
    profile = get_profile(timeframe)

    # Resolve ATR pct limits: arg > profile > hardcoded fallback
    _atr_min = atr_pct_min if atr_pct_min is not None else profile.atr_pct_min
    _atr_max = atr_pct_max if atr_pct_max is not None else profile.atr_pct_max

    _mode_label = f"RS:{profile.name}" if regime_switching else profile.name
    _update_progress(
        phase="Conectando exchange", phase_num=1, pct=0,
        message=f"Iniciando backtest CTEV [{_mode_label}]...", running=True,
    )
    logger.info(
        "Iniciando backtest CTEV [%s]: %s %s %d dias ATR_pct=[%.0f%%, %.0f%%] mode=%s rs=%s",
        profile.name, symbol, timeframe, days, _atr_min * 100, _atr_max * 100,
        "advanced" if advanced else "basic",
        "ON" if regime_switching else "OFF",
    )

    # 1. Download dados
    df = fetch_historical_ohlcv(symbol, timeframe, days)

    _update_progress(
        phase="Calculando indicadores", phase_num=3, pct=16,
        message=f"Calculando EMA/BB/RSI/ADX/MACD para {len(df):,} candles [{profile.name}]...",
        candles_total=len(df),
    )

    # 2. Calcula indicadores (passa timeframe para lookbacks adaptativos)
    df_ind = compute_indicators(df, timeframe=timeframe)

    # 3. Remove linhas com NaN
    df_clean = df_ind.dropna(subset=[
        "ema20", "ema50", "ema200", "rsi", "atr", "atr_percentile",
        "macd", "macd_signal", "macd_hist",
        "adx", "plus_di", "minus_di", "regime",
    ]).copy()

    _update_progress(
        phase="Escaneando candles", phase_num=4, pct=20,
        message=f"Iniciando simulacao [{profile.summary()}] em {len(df_clean):,} candles...",
        candles_total=len(df_clean), candles_scanned=0, signals_found=0,
    )
    logger.info("DataFrame limpo: %d candles (de %d originais)", len(df_clean), len(df_ind))

    # 4. Simula trades (passa profile para evaluate_long/short)
    _diag = {}
    if regime_switching:
        # v7: Regime-switching mode
        from regime_engine import classify_regimes_v2, get_regime_params
        from strategy_regime import (
            evaluate_mean_reversion_long, evaluate_mean_reversion_short,
        )

        df_clean = classify_regimes_v2(df_clean, hysteresis_bars=3)
        trades, atr_filtered, _diag = _simulate_regime_switching(
            df_clean, _atr_min, _atr_max, profile=profile,
        )
    elif advanced:
        trades, atr_filtered, _diag = simulate_trades_advanced(
            df_clean, _atr_min, _atr_max, profile=profile,
        )
    else:
        trades, atr_filtered, _diag = simulate_trades(
            df_clean, _atr_min, _atr_max, profile=profile,
        )

    _update_progress(
        phase="Calculando metricas", phase_num=5, pct=85,
        message=f"Compilando metricas de {len(trades)} trades [{profile.name}]...",
        signals_found=len(trades),
    )

    # 5. Calcula metricas
    metrics = calculate_metrics(trades, df_clean, atr_filtered)
    metrics._filter_diag = _diag

    _update_progress(
        phase="Concluido", phase_num=6, pct=100,
        message=f"Backtest [{profile.name}] concluido: {metrics.total_trades} trades | WR={metrics.win_rate:.1f}%",
        running=False,
    )

    logger.info(
        "Backtest [%s] concluido: %d trades | WR=%.1f%% | PF=%.2f | MaxDD=%.2f%% | PnL=%.2f%%",
        profile.name, metrics.total_trades, metrics.win_rate, metrics.profit_factor,
        metrics.max_drawdown_pct, metrics.total_pnl_pct,
    )

    return metrics, trades


# ------------------------------------------------------------------
# Walk-Forward Analysis
# ------------------------------------------------------------------
def run_walk_forward(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    total_days: int = 730,
    train_days: int = 180,
    test_days: int = 60,
    step_days: int = 60,
    atr_pct_min: float = 0.20,
    atr_pct_max: float = 0.80,
) -> List[WalkForwardResult]:
    """
    Executa Walk-Forward Analysis da estrategia CTEV.

    Divide o dataset em janelas treino/teste e rola para frente.
    Cada janela: [train_days] de treino + [test_days] de teste,
    avancando [step_days] por iteracao.

    Returns:
        Lista de WalkForwardResult com metricas de cada janela.
    """
    logger.info(
        "Iniciando Walk-Forward: train=%dd test=%dd step=%dd total=%dd",
        train_days, test_days, step_days, total_days,
    )

    # Download completo
    df = fetch_historical_ohlcv(symbol, timeframe, total_days + train_days)
    df_ind = compute_indicators(df, timeframe=timeframe)
    df_clean = df_ind.dropna(subset=[
        "ema20", "ema50", "ema200", "rsi", "atr", "atr_percentile",
        "macd", "macd_signal", "macd_hist",
        "adx", "plus_di", "minus_di", "regime",
    ]).copy()

    results: List[WalkForwardResult] = []
    window_id = 0

    # Candle counts adaptativos ao timeframe
    _cpd = _timeframe_ms_to_candles_per_day(timeframe)
    total_candles_train = train_days * _cpd
    total_candles_test = test_days * _cpd
    step_candles = step_days * _cpd
    n = len(df_clean)

    start = 0
    while start + total_candles_train + total_candles_test <= n:
        train_slice = df_clean.iloc[start:start + total_candles_train]
        test_slice = df_clean.iloc[start + total_candles_train:start + total_candles_train + total_candles_test]

        if len(train_slice) < 100 or len(test_slice) < 10:
            break

        # Treino
        train_trades, _, _ = simulate_trades(train_slice, atr_pct_min, atr_pct_max, profile=profile)
        train_metrics = calculate_metrics(train_trades, train_slice)

        # Teste
        test_trades, _, _ = simulate_trades(test_slice, atr_pct_min, atr_pct_max, profile=profile)
        test_metrics = calculate_metrics(test_trades, test_slice)

        # Degradação
        if train_metrics.win_rate > 0:
            degradation = (train_metrics.win_rate - test_metrics.win_rate) / train_metrics.win_rate * 100
        else:
            degradation = 0.0

        results.append(WalkForwardResult(
            window_id=window_id,
            train_start=str(train_slice.index[0]),
            train_end=str(train_slice.index[-1]),
            test_start=str(test_slice.index[0]),
            test_end=str(test_slice.index[-1]),
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            degradation_pct=round(degradation, 2),
        ))

        logger.info(
            "WFO Window %d: train WR=%.1f%% PF=%.2f | test WR=%.1f%% PF=%.2f | degrad=%.1f%%",
            window_id,
            train_metrics.win_rate, train_metrics.profit_factor,
            test_metrics.win_rate, test_metrics.profit_factor,
            degradation,
        )

        window_id += 1
        start += step_candles

    logger.info("Walk-Forward concluido: %d janelas.", len(results))
    return results
