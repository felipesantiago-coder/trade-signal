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
v9.0: Custos realistas para BTC/USD (maker fee 0.016%, spread 2bps, slippage 5bps).
       Partial TP com trailing fixo apos TP1 otimizado. Sem BE/momentum/time-decay.
v10.0: Cooldown apos 2 SL consecutivos na mesma direcao (24 bars pause).
       Pos-TP1 SL buffer 1.5x ATR (de 1.0x).
v11.0: Cooldown relaxado: 3 SL / 12 bars (DESEMPENHO PIOROU em todos os periodos).
v12.0: Cooldown revertido para 2 SL / 24 bars. Estrategia professional seletiva:
       ADX 36, DI direction filter ON, EMA proximity REMOVIDO (so fib + touch).
v13.0: Active Trader Multi-Strategy — 3 entry types (CTEV Pullback + Momentum
       Continuation + Mean-Reversion BB Bounce) to achieve 4+ trades/week.
       Regime skip removed; each strategy handles regime internally.
v14.0: CTEV Trend-Flow — pullback opcional (maior gargalo removido).
       ADX 25, transition ON, EMA proximity 2%/2.5%.
v15.0: Risk management layer — melhora periodos curtos sem alterar geracao de sinais.
       1. Trailing pos-TP1: 1.0x ATR (so apos 50% seguro no TP)
       2. Cooldown: 2 SL / 16 bars (de 24)
       3. Anti-martingale sizing: infraestrutura (requer metrica balance-based)
       4. Pos-TP1 SL buffer: 1.5x ATR (mantido — grid-otimizado)

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
    evaluate_row_signals,
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
    # v17.0: Multi-strategy tracking
    entry_type: str = "ctev_pullback"  # ctev_pullback, ctev_momentum, momentum,
                                         # ema_bounce, squeeze_breakout, rsi_reversal, ranging_mr
    # v26.0: Enhanced audit fields
    regime_at_entry: str = ""
    concurrent_count: int = 0
    equity_before: float = 0.0
    equity_after: float = 0.0
    capital_allocated: float = 0.0
    quantity: float = 0.0
    gross_pnl: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    funding: float = 0.0
    net_pnl: float = 0.0
    return_pct: float = 0.0
    r_multiple: float = 0.0
    duration: str = ""


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
    # v26.0: Advanced risk metrics
    recovery_factor: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    cagr: float = 0.0
    expectancy: float = 0.0
    var_95: float = 0.0
    expected_shortfall: float = 0.0
    omega_ratio: float = 0.0

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
            "equity_curve": self.equity_curve,
            "be_triggered_count": self.be_triggered_count,
            "trailing_activated_count": self.trailing_activated_count,
            "partial_tp_count": self.partial_tp_count,
            "avg_r_r": round(self.avg_r_r, 2),
            "recovery_factor": round(self.recovery_factor, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "cagr": round(self.cagr, 4),
            "expectancy": round(self.expectancy, 4),
            "var_95": round(self.var_95, 4),
            "expected_shortfall": round(self.expected_shortfall, 4),
            "omega_ratio": round(self.omega_ratio, 4),
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

        except (ccxt.BadRequest, ccxt.NetworkError, ccxt.ExchangeError,
                ccxt.RateLimitExceeded) as exc:
            # Coinbase retorna 400 quando "start" está no futuro (fim dos dados)
            if "start must not be in the future" in str(exc):
                logger.debug("Coinbase: fim dos dados historicos (batch %d)", iteration)
                break
            logger.error("Erro ao baixar dados (batch %d): %s", iteration, exc)
            break
        except Exception as exc:
            logger.error("Erro inesperado ao baixar dados (batch %d): %s", iteration, exc)
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
# Custos de transacao (v9.0: OTIMIZADO — realista para BTC/USD 1h)
# ------------------------------------------------------------------
# Fees: 0.016% maker (Binance BTC/USD spot — usado com limit orders)
# Spread: 2 bps (BTC/USD em sessoes liquidas — observado real)
# Slippage: 5 bps (limit orders em mercado liquido — minimo)
# Total round-trip: ~0.032% (fees) + 0.02% (spread) + 0.05% (slippage) = 0.10%
# v4 usava: 0.35% round-trip (conservador demais para BTC/USD)
# v9.0: 0.10% round-trip (realista para BTC/USD com limit orders)
DEFAULT_FEE_PCT = 0.016       # 0.016% maker fee (Binance)
DEFAULT_SPREAD_BPS = 2.0     # 2 bps (BTC/USD spread real)
DEFAULT_SLIPPAGE_BPS = 5.0   # 5 bps (limit order slippage)


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
    "1w": 7 * 24 * 60 * 60 * 1000,
}


def _timeframe_to_ms(timeframe: str) -> int:
    """Converte string de timeframe (ex: '15m', '1h', '4h') para milissegundos."""
    tf = timeframe.lower()
    if tf in _TIMEFRAME_MS_MAP:
        return _TIMEFRAME_MS_MAP[tf]
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

    v26.0: tambem exporta componentes individuais via _apply_costs_detail()
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


def _apply_costs_detail(entry_price: float, exit_price: float, is_long: bool,
                         fee_pct: float = DEFAULT_FEE_PCT,
                         spread_bps: float = DEFAULT_SPREAD_BPS,
                         slippage_bps: float = DEFAULT_SLIPPAGE_BPS) -> Tuple[float, float, float, float, float, float]:
    """
    v26.0: Versao detalhada que retorna componentes individuais de custo.

    Returns:
        (adjusted_entry, adjusted_exit, total_cost_pct, fee_component, slippage_component, spread_component)
        Todos em % relativo ao entry_price.
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

    # Componentes individuais (% do entry_price)
    fee_component = (fee_on_entry + fee_on_exit) / entry_price * 100
    slippage_component = entry_price * slippage_pct * 2 / entry_price * 100  # round-trip
    spread_component = entry_price * spread_pct * 2 / entry_price * 100  # round-trip

    # Total cost as % of entry
    total_cost_pct = (abs(adj_entry - entry_price) + abs(adj_exit - exit_price)) / entry_price * 100

    return adj_entry, adj_exit, total_cost_pct, fee_component, slippage_component, spread_component

# ------------------------------------------------------------------
# v26.0: Audit field calculator for TradeResult
# ------------------------------------------------------------------
def _compute_audit_fields(
    entry_price: float, exit_price: float, is_long: bool,
    position_size: float, position_usd: float, bars_held: int,
    entry_ts, exit_ts,
    fee_pct: float = DEFAULT_FEE_PCT,
    spread_bps: float = DEFAULT_SPREAD_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    funding_rate_bps: float = 0.01,  # ~0.01% per 8h for BTC perpetual
    apply_costs_flag: bool = True,
) -> dict:
    """v26.0: Compute detailed audit fields for a TradeResult.
    
    Returns dict with: gross_pnl, fees, slippage, funding, net_pnl, return_pct, duration
    """
    if entry_price <= 0:
        return {"gross_pnl": 0.0, "fees": 0.0, "slippage": 0.0, "funding": 0.0,
            "net_pnl": 0.0, "return_pct": 0.0, "duration": ""}

    # Raw gross PnL (no costs)
    if is_long:
        gross_pnl_pct = (exit_price - entry_price) / entry_price * 100
    else:
        gross_pnl_pct = (entry_price - exit_price) / entry_price * 100
    
    if apply_costs_flag and position_usd > 0:
        # Detailed cost decomposition
        _, _, total_cost_pct, fee_comp, slip_comp, spread_comp = _apply_costs_detail(
            entry_price, exit_price, is_long, fee_pct, spread_bps, slippage_bps,
        )
        
        fees_usd = position_usd * (fee_comp / 100)
        slippage_usd = position_usd * (slip_comp / 100)
        
        # Adjusted exit for net PnL
        _, adj_exit, _ = _apply_costs(
            entry_price, exit_price, is_long, fee_pct, spread_bps, slippage_bps,
        )
        if is_long:
            net_pnl_pct = (adj_exit - entry_price) / entry_price * 100
        else:
            net_pnl_pct = (entry_price - adj_exit) / entry_price * 100
        
        net_pnl_usd = position_usd * (net_pnl_pct / 100)
    else:
        fees_usd = 0.0
        slippage_usd = 0.0
        total_cost_pct = 0.0
        net_pnl_pct = gross_pnl_pct
        net_pnl_usd = position_usd * (net_pnl_pct / 100) if position_usd > 0 else 0.0

    # Funding rate estimation (perpetual swap: ~0.01% per 8h, 1h candles = 0.01/8 per bar)
    funding_per_bar = position_usd * (funding_rate_bps / 10000) / 8.0 if position_usd > 0 else 0.0
    funding_usd = -funding_per_bar * bars_held  # always a cost for long, offset for short
    # For shorts, funding can be positive (you receive it), but model as cost for conservatism
    if not is_long:
        funding_usd = abs(funding_usd) * 0.7  # shorts receive ~70% of time on BTC
    else:
        funding_usd = -abs(funding_usd) * 0.6  # longs pay ~60% of time
    
    # Duration string
    hours = bars_held  # 1h candles = 1h per bar
    if hours < 24:
        duration = f"{hours}h"
    elif hours < 48:
        duration = f"1d {hours - 24}h"
    elif hours < 168:
        days = hours // 24
        rem_h = hours % 24
        duration = f"{days}d {rem_h}h"
    else:
        weeks = hours // 168
        rem = hours % 168
        days = rem // 24
        rem_h = rem % 24
        duration = f"{weeks}w {days}d {rem_h}h"

    # Return % relative to capital allocated
    return_pct = net_pnl_pct  # already in %

    return {
        "gross_pnl": round(gross_pnl_pct, 4),
        "fees": round(fees_usd, 4),
        "slippage": round(slippage_usd, 4),
        "funding": round(funding_usd, 4),
        "net_pnl": round(net_pnl_usd, 4),
        "return_pct": round(return_pct, 4),
        "duration": duration,
    }




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
            last_j = min(i + _max_bars, n) - 1
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
            # v26.0: Audit fields (basic sim - no position sizing)
            **_compute_audit_fields(
                entry_price=entry_price, exit_price=exit_price, is_long=is_long,
                position_size=0.0, position_usd=0.0, bars_held=bars,
                entry_ts=row.name, exit_ts=df_ind.iloc[min(i + bars, n - 1)].name,
                fee_pct=fee_pct, spread_bps=spread_bps, slippage_bps=slippage_bps,
                apply_costs_flag=apply_costs_flag,
            ),
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
    be_trigger_atr_mult: float = 0.0,  # v8.0: BE DESATIVADO (0 = desliga)
    trailing_atr_mult: float = 1.0,  # v15.0: 1.0x ATR — ativa APOS partial TP only
    trailing_activate_atr_mult: float = 999.0,  # v8.0: trailing pre-TP DESATIVADO
    partial_tp_pct: float = 0.50,
    apply_costs_flag: bool = True,
    fee_pct: float = DEFAULT_FEE_PCT,
    spread_bps: float = DEFAULT_SPREAD_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    max_bars: int = 72,
    profile=None,
) -> Tuple[List[TradeResult], int, dict]:
    """
    Simulacao avancada v15.0 com position sizing, trailing stop, partial TP,
    cost modeling, regime filter, RSI exhaustion e anti-martingale sizing.

    v15.0 Mudancas vs v8.0 (risk management layer — sem alterar geracao de sinais):
      - Trailing pos-TP1: 1.0x ATR apos 50% parcial (antes: desativado)
      - Cooldown: 2 SL / 16 bars (de 24 bars)
      - Max_bars: 168 (mantido — grid-otimizado)
      - Pos-TP1 SL buffer: 1.5x ATR (mantido — grid-otimizado)
      - Mantido: BE off, no pre-TP trailing, RSI exhaustion, partial TP 50%
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
    _diag_cooldown_skip = 0

    # ── v12.0: Consecutive loss cooldown (conservative — professional) ──
    # Pausa entradas na mesma direcao apos SL consecutivos
    # v15.0: 2 SL / 16 bars (reduzido de 24 — periodos curtos precisam de mais rapidez)
    _consecutive_sl_long = 0
    _consecutive_sl_short = 0
    _cooldown_direction = None   # "LONG" or "SHORT" when in cooldown
    _cooldown_until_bar = 0     # bar index when cooldown ends
    _COOLDOWN_TRIGGER = 2       # v12.0: 2 consecutive SL to trigger cooldown
    _COOLDOWN_BARS = 16         # v15.0: 16 bars (~2/3 dia, de 24)

    # ── v16.2: Anti-martingale DESATIVADO — balance-based ja protege
    _base_risk = risk_per_trade_pct
    _current_risk = _base_risk
    _consecutive_losses = 0
    _RISK_REDUCTION = 0.00
    _MIN_RISK_FRACTION = 1.00

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
        if regime == "volatile":
            _diag_regime_volatile += 1
            regime_filtered += 1
            i += 1
            continue
        # v17.0: ranging NAO e mais pulado — EMA Bounce, Squeeze, RSI Rev
        # e MR podem gerar sinais em ranging.
        if regime == "ranging":
            _diag_regime_ranging += 1
            # passa adiante — evaluate_row_signals decide se ha sinal

        atr_pct = float(row.get("atr_percentile", 0.5))
        if atr_pct < atr_pct_min or atr_pct > atr_pct_max:
            _diag_atr_filtered += 1
            atr_filtered += 1
            i += 1
            continue

        # v17.0: Multi-Strategy Engine — tenta TODOS os tipos de entrada
        # CTEV > Momentum > EMA Bounce > Squeeze > RSI Rev > MR
        signal = evaluate_row_signals(row, profile=profile)

        if signal is None:
            _diag_no_signal += 1
            i += 1
            continue

        # v10.0: Cooldown check — pula sinais na direcao em cooldown
        _sig_dir = signal.type.value
        if (_cooldown_direction is not None and _sig_dir == _cooldown_direction and i < _cooldown_until_bar):
            _diag_cooldown_skip += 1
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
        # v15.0: Position sizing com anti-martingale (usa _current_risk adaptativo)
        risk_usd = balance * _current_risk
        if sl_distance_pct > 0:
            position_size = risk_usd / (sl_distance_pct * effective_entry)
        else:
            position_size = 0.0
        position_usd = position_size * effective_entry

        if position_usd < 1.0:
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
        # v8.0: entry_adx removido (momentum exit desativado)

        # v16.0: max_bars from signal (entry-type specific)
        _max_bars = getattr(signal, "max_bars", 72)
        for j in range(i + 1, min(i + _max_bars, n)):
            future = df_ind.iloc[j]
            f_close = float(future["close"])
            f_low = float(future["low"])
            f_high = float(future["high"])
            f_rsi = float(future.get("rsi", 50.0))  # v8.0: RSI para exhaustion
            # v8.0: f_adx removido (momentum exit desativado)
            bars = j - i

            if is_long:
                highest_favorable = max(highest_favorable, f_high)
            else:
                highest_favorable = min(highest_favorable, f_low)

            fav_dist = abs(highest_favorable - entry_price)

            # v8.0: Trailing — ativa apos trailing_activate_atr_mult * ATR (1.5x)
            # SEM time-decay (removido — encolhia trailing prematuramente)
            # SEM break-even (desativado — causava 54% de trades com lucro ~0%)
            if fav_dist >= atr * trailing_activate_atr_mult:
                trailing_activated = True

            if trailing_activated:
                trail_dist = atr * trailing_atr_mult  # v8.0: 3.0x ATR, sem decay
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

            # v8.0: RSI Exhaustion — saida por extrema sobrecompra/sobrevenda
            # Mantido: protecao legitima em extremos (RSI > 80 / < 20)
            # Apenas apos 24+ barras no trade e com lucro positivo
            if bars >= 24:
                if is_long and f_rsi > 80.0:
                    _current_profit = (f_close - entry_price) if is_long else (entry_price - f_close)
                    if _current_profit > 0:
                        exit_price = f_close
                        exit_reason = "rsi_exhaustion"
                        break
                elif not is_long and f_rsi < 20.0:
                    _current_profit = (f_close - entry_price) if is_long else (entry_price - f_close)
                    if _current_profit > 0:
                        exit_price = f_close
                        exit_reason = "rsi_exhaustion"
                        break

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
                    trailing_activated = True
                    # v15.1: Pos-TP1 SL com buffer 1.5x ATR (mantido — grid-otimizado)
                    if is_long:
                        current_sl = tp - atr * 1.5
                    else:
                        current_sl = tp + atr * 1.5
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
            entry_type=getattr(signal, 'entry_type', 'unknown'),
            # v26.0: Audit fields
            **_compute_audit_fields(
                entry_price=entry_price, exit_price=exit_price, is_long=is_long,
                position_size=position_size, position_usd=position_usd,
                bars_held=bars, entry_ts=row.name,
                exit_ts=df_ind.iloc[min(i + bars, n - 1)].name,
                fee_pct=fee_pct, spread_bps=spread_bps, slippage_bps=slippage_bps,
                apply_costs_flag=apply_costs_flag,
            ),
        ))

        # Track equity for live streaming
        _running_pnl += pnl_pct
        _eq_points.append(round(_running_pnl, 2))

        # v10.0: Update consecutive loss cooldown state
        if exit_reason == "sl" and pnl_pct < 0:
            if is_long:
                _consecutive_sl_long += 1
                _consecutive_sl_short = 0
                if _consecutive_sl_long >= _COOLDOWN_TRIGGER:
                    _cooldown_direction = "LONG"
                    _cooldown_until_bar = i + bars + 1 + _COOLDOWN_BARS
                    _consecutive_sl_long = 0
            else:
                _consecutive_sl_short += 1
                _consecutive_sl_long = 0
                if _consecutive_sl_short >= _COOLDOWN_TRIGGER:
                    _cooldown_direction = "SHORT"
                    _cooldown_until_bar = i + bars + 1 + _COOLDOWN_BARS
                    _consecutive_sl_short = 0
        else:
            # Win or non-SL exit resets counter for that direction
            if is_long:
                _consecutive_sl_long = 0
            else:
                _consecutive_sl_short = 0
            # If the winning direction was in cooldown, clear it
            if _cooldown_direction == _sig_dir:
                _cooldown_direction = None

        # v15.0: Anti-martingale — ajusta risco baseado no resultado
        if pnl_pct < 0:
            # Loss: reduz risco em 25%, minimo 50% do base
            _consecutive_losses += 1
            _current_risk = max(
                _base_risk * _MIN_RISK_FRACTION,
                _current_risk * (1 - _RISK_REDUCTION),
            )
        else:
            # Win: reseta risco ao base
            _consecutive_losses = 0
            _current_risk = _base_risk

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

    # PnL por trade (price-based, para Sharpe / PF / avg win-loss / best-worst)
    pnls = [t.pnl_pct for t in trades]

    # v15.0: Balance-based equity curve (composto — reflete anti-martingale sizing)
    _INITIAL_BALANCE = 10000.0
    _eq_balances = [_INITIAL_BALANCE]
    for t in trades:
        _prev = _eq_balances[-1]
        _pnl_usd = t.position_usd * (t.pnl_pct / 100) if t.position_usd > 0 else 0
        _eq_balances.append(_prev + _pnl_usd)
    _eq_arr = np.array(_eq_balances)
    _eq_peak = np.maximum.accumulate(_eq_arr)

    # Equity curve (return % from start)
    equity_curve = [(i, float((_eq_arr[i + 1] / _INITIAL_BALANCE - 1) * 100)) for i in range(len(trades))]

    # Max Drawdown % (balance-based: % drop from peak balance)
    _dd_pct = (_eq_peak - _eq_arr) / _eq_peak * 100
    max_dd = float(np.max(_dd_pct)) if len(_dd_pct) > 0 else 0.0

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

    # v15.0: Total PnL (balance-based, composto — reflete anti-martingale)
    total_pnl_pct = float((_eq_arr[-1] / _INITIAL_BALANCE - 1) * 100)
    total_pnl_usd = float(_eq_arr[-1] - _INITIAL_BALANCE)
    max_dd_usd = float(np.max(_eq_peak - _eq_arr)) if len(_eq_arr) > 0 else 0.0

    # v26.0: Advanced risk metrics
    pnls_arr = np.array(pnls) if pnls else np.array([0.0])
    wr = len(wins) / len(trades) if trades else 0.0
    avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0.0
    avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0.0

    # Recovery Factor
    recovery_factor = abs(total_pnl_pct) / max_dd if max_dd > 0 else 0.0

    # Sortino Ratio
    downside = pnls_arr[pnls_arr < 0]
    if len(downside) > 1 and np.std(downside) > 0:
        sortino = (np.mean(pnls_arr) / np.std(downside)) * (365 ** 0.5)
    else:
        sortino = 0.0

    # Calmar Ratio
    calmar = total_pnl_pct / max_dd if max_dd > 0 else 0.0

    # CAGR (1h candles proxy: len(df)/24 = days)
    _days = max(len(df) / 24, 1)
    if total_pnl_pct > -100:
        cagr = ((1 + total_pnl_pct / 100) ** (365 / _days) - 1) * 100
    else:
        cagr = -100.0

    # Expectancy
    expectancy = (wr * avg_win) + ((1 - wr) * avg_loss)

    # VaR 95 (5th percentile of PnLs, negative)
    var_95 = float(np.percentile(pnls_arr, 5)) if len(pnls_arr) > 1 else 0.0

    # Expected Shortfall (mean of PnLs below VaR)
    below_var = pnls_arr[pnls_arr <= var_95]
    expected_shortfall = float(np.mean(below_var)) if len(below_var) > 0 else 0.0

    # Omega Ratio (threshold=0)
    gains = pnls_arr[pnls_arr > 0]
    losses_abs = -pnls_arr[pnls_arr < 0]
    omega = float(np.sum(gains) / np.sum(losses_abs)) if np.sum(losses_abs) > 0 else 0.0

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
        total_pnl_pct=total_pnl_pct,
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
        recovery_factor=recovery_factor,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        cagr=cagr,
        expectancy=expectancy,
        var_95=var_95,
        expected_shortfall=expected_shortfall,
        omega_ratio=omega,
    )


# ------------------------------------------------------------------
# v8: EMA Cross simulation (for INTRADAY profiles: 15m/30m)
# ------------------------------------------------------------------
def _simulate_ema_cross(
    df_ind: pd.DataFrame,
    atr_pct_min: float = 0.10,
    atr_pct_max: float = 0.90,
    profile=None,
    fee_pct: float = DEFAULT_FEE_PCT,
    spread_bps: float = DEFAULT_SPREAD_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> Tuple[List[TradeResult], int, dict]:
    r"""
    Simulacao EMA Cross v11 para timeframes intraday (15m/30m).

    v11: Adicionado trailing stop + break-even + partial TP.
      - BE trigger: apos 1.0x ATR favoravel, SL move para entry
      - Trailing: 1.5x ATR do high water mark (ratchet-only)
      - Partial TP: 50% no TP1, resto fica em trailing

    v10: OBV filter + R:R otimizado.
    v8 base: cruze EMA20/50 + RSI delta + ADX > 15 + cooldown 12 bars.
    NAO usa regime-switching — esta estrategia tem logica propria.

    Custos: fee=0.016% + spread=2bps + slip=2bps (limit orders).
    """
    from strategy_ema_cross import evaluate_ema_cross_row, reset_cooldown, EMA_CROSS_PARAMS

    # Reset cooldown state
    reset_cooldown()

    # v11: trailing stop params from strategy
    _be_trigger = EMA_CROSS_PARAMS.get("be_trigger_atr_mult", 1.0)
    _trail_dist = EMA_CROSS_PARAMS.get("trailing_atr_mult", 1.5)
    _partial_pct = EMA_CROSS_PARAMS.get("partial_tp_pct", 0.50)

    trades: List[TradeResult] = []
    atr_filtered = 0
    i = 0
    n = len(df_ind)
    _scan_t0 = time.time()
    _last_progress_time = 0.0
    _eq_points: list = []
    _running_pnl = 0.0

    # Para INTRADAY, usar custos de limit order (maker)
    _fee = 0.016   # maker fee
    _spread = 2.0   # bps
    _slip = 2.0     # bps

    # Diagnostics
    _be_count = 0
    _trail_count = 0
    _partial_count = 0

    while i < n:
        row = df_ind.iloc[i]

        # Progress
        _now = time.time()
        if _now - _last_progress_time > 0.25:
            _last_progress_time = _now
            _elapsed = max(_now - _scan_t0, 0.01)
            _speed = i / _elapsed
            _scan_pct = 20 + (i / max(n, 1)) * 60
            _update_progress(
                phase="Escaneando candles (EMA Cross v11)", phase_num=4,
                pct=round(_scan_pct, 1),
                message=f"EMA Cross v11 {i:,}/{n:,} ({_speed:.0f}c/s) trades={len(trades)}",
                candles_total=n, candles_scanned=i, scan_speed=round(_speed),
                current_price=round(float(row.get("close", 0)), 2),
            )

        if i < 1:
            i += 1
            continue

        # Check NaN
        critical = ["ema20", "ema50", "ema200", "rsi", "atr", "rsi_delta"]
        if any(pd.isna(row.get(c)) for c in critical):
            i += 1
            continue

        # ATR filter
        atr_pct = float(row.get("atr_percentile", 0.5))
        if atr_pct < atr_pct_min or atr_pct > atr_pct_max:
            atr_filtered += 1
            i += 1
            continue

        # Evaluate EMA Cross signal
        prev = df_ind.iloc[i - 1]
        signal = evaluate_ema_cross_row(row, prev, i, profile=profile)

        if signal is None:
            i += 1
            continue

        # ── v11: Advanced trade simulation with trailing stop ──
        entry_price = signal.entry_price
        sl = signal.stop_loss
        tp = signal.take_profit
        atr = signal.atr
        is_long = signal.type == SignalType.LONG
        exit_price = None
        exit_reason = None
        bars = 0

        # Trailing stop state
        current_sl = sl
        be_triggered = False
        trailing_activated = False
        partial_tp_filled = False
        highest_favorable = entry_price
        sl_updates = 0

        _max_bars = profile.max_bars_held if profile else 48
        for j in range(i + 1, min(i + _max_bars, n)):
            future = df_ind.iloc[j]
            f_close = float(future["close"])
            f_low = float(future["low"])
            f_high = float(future["high"])
            bars = j - i

            # Track best price seen (high water mark)
            if is_long:
                highest_favorable = max(highest_favorable, f_high)
            else:
                highest_favorable = min(highest_favorable, f_low)

            # Check SL/TP hit
            sl_hit = False
            tp_hit = False
            if is_long:
                if f_low <= current_sl:
                    sl_hit = True
                if f_high >= tp:
                    tp_hit = True
            else:
                if f_high >= current_sl:
                    sl_hit = True
                if f_low <= tp:
                    tp_hit = True

            # v11: Partial TP logic (first TP hit → take partial, activate trailing)
            if tp_hit and not partial_tp_filled and not be_triggered:
                # First TP: take _partial_pct profit, move SL to entry, activate trailing
                partial_tp_filled = True
                _partial_count += 1
                be_triggered = True
                _be_count += 1
                current_sl = entry_price
                trailing_activated = True
                # Don't exit yet — let trailing manage the rest
                if not sl_hit:
                    continue

            # v11: Break-even trigger (price moved be_trigger * ATR in favor)
            if not be_triggered:
                fav_dist = abs(highest_favorable - entry_price)
                if fav_dist >= atr * _be_trigger:
                    be_triggered = True
                    _be_count += 1
                    trailing_activated = True
                    current_sl = entry_price

            # v11: Trailing stop update (ratchet-only)
            if trailing_activated:
                trail_distance = atr * _trail_dist
                if is_long:
                    new_trail = highest_favorable - trail_distance
                    if new_trail > current_sl:
                        current_sl = new_trail
                        sl_updates += 1
                        _trail_count += 1
                else:
                    new_trail = highest_favorable + trail_distance
                    if new_trail < current_sl:
                        current_sl = new_trail
                        sl_updates += 1
                        _trail_count += 1

            # Exit checks
            if sl_hit and not tp_hit:
                exit_price = current_sl
                exit_reason = "trailing_sl" if trailing_activated else "sl"
                break
            elif tp_hit and not sl_hit:
                if partial_tp_filled:
                    # Second TP hit on remaining position
                    exit_price = tp
                    exit_reason = "tp"
                    break
                # First TP (no partial yet — should not happen with v11 logic)
                exit_price = tp
                exit_reason = "tp"
                break
            elif tp_hit and sl_hit:
                # Worst case: both hit same bar → use SL (conservative)
                exit_price = current_sl
                exit_reason = "trailing_sl" if trailing_activated else "sl"
                break

        if exit_price is None:
            last_j = min(i + _max_bars, n) - 1
            exit_price = float(df_ind.iloc[last_j]["close"])
            exit_reason = "timeout"
            bars = last_j - i

        # Apply costs (limit order pricing)
        _, adj_exit, cost_pct = _apply_costs(
            entry_price, exit_price, is_long,
            _fee, _spread, _slip,
        )
        if is_long:
            pnl_pct = (adj_exit - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - adj_exit) / entry_price * 100

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
            be_triggered=be_triggered,
            trailing_activated=trailing_activated,
            partial_tp_filled=partial_tp_filled,
            sl_updates=sl_updates,
        ))

        _running_pnl += pnl_pct
        _eq_points.append(round(_running_pnl, 2))

        i += bars + 1

    logger.info(
        "EMA Cross v11: %d trades, %d ATR filt, BE=%d, trail=%d, partial=%d",
        len(trades), atr_filtered, _be_count, _trail_count, _partial_count,
    )

    _diag = {
        "strategy": "ema_cross_v11",
        "atr_filtered": atr_filtered,
        "be_triggered": _be_count,
        "trailing_activations": _trail_count,
        "partial_tp_filled": _partial_count,
        "trailing_params": {
            "be_trigger_atr": _be_trigger,
            "trailing_atr": _trail_dist,
            "partial_pct": _partial_pct,
        },
        "costs": {"fee_pct": _fee, "spread_bps": _spread, "slippage_bps": _slip},
    }
    return trades, atr_filtered, _diag


# ------------------------------------------------------------------
# ATF v1: Adaptive Trend-Follow simulation (for INTRADAY: 15m/30m)
# ------------------------------------------------------------------
def _simulate_atf(
    df_ind: pd.DataFrame,
    atr_pct_min: float = 0.15,
    atr_pct_max: float = 0.85,
    profile=None,
    fee_pct: float = DEFAULT_FEE_PCT,
    spread_bps: float = DEFAULT_SPREAD_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> Tuple[List[TradeResult], int, dict]:
    r"""
    Simulacao ATF v1 (Adaptive Trend-Follow) para 15m/30m.

    Diferencas fundamentais vs _simulate_ema_cross:
      1. Sem TP fixo — saida puramente via trailing stop adaptativo ao ADX
      2. Trailing width: 1.0x (ADX<25) / 1.5x (ADX 25-35) / 2.5x (ADX>=35) ATR
      3. BE trigger em 0.8x ATR (mais rapido que v11)
      4. Sem partial TP — posicao inteira faz trailing
      5. Max bars: 96 (24h em 15min) vs 36 do v11
      6. Cooldown: 6 bars (1.5h) vs 12 do v11, 3 bars apos trailing exit
      7. Entradas via score composto (0-10) + 6 tipos de gatilho

    Custos: fee=0.016% + spread=2bps + slip=2bps (limit orders).
    """
    from strategy_atf import evaluate_atf_row, reset_cooldown, ATF_PARAMS

    reset_cooldown()

    _be_trigger = ATF_PARAMS["be_trigger_atr"]
    _max_bars = ATF_PARAMS["max_bars"]

    trades: List[TradeResult] = []
    atr_filtered = 0
    i = 0
    n = len(df_ind)
    _scan_t0 = time.time()
    _last_progress_time = 0.0
    _eq_points: list = []
    _running_pnl = 0.0

    # Limit order costs (maker)
    _fee = 0.016
    _spread = 2.0
    _slip = 2.0

    # Diagnostics
    _be_count = 0
    _trail_count = 0
    _diag_triggers = {}
    _diag_scores = []
    _diag_trail_types = {"strong": 0, "medium": 0, "weak": 0}

    while i < n:
        row = df_ind.iloc[i]

        # Progress (throttled to ~4Hz)
        _now = time.time()
        if _now - _last_progress_time > 0.25:
            _last_progress_time = _now
            _elapsed = max(_now - _scan_t0, 0.01)
            _speed = i / _elapsed
            _scan_pct = 20 + (i / max(n, 1)) * 60
            _update_progress(
                phase="Escaneando candles (ATF v1)", phase_num=4,
                pct=round(_scan_pct, 1),
                message=f"ATF v1 {i:,}/{n:,} ({_speed:.0f}c/s) trades={len(trades)}",
                candles_total=n, candles_scanned=i, scan_speed=round(_speed),
                current_price=round(float(row.get("close", 0)), 2),
            )

        if i < 1:
            i += 1
            continue

        # Check NaN
        critical = ["ema20", "ema50", "ema200", "rsi", "atr", "rsi_delta",
                     "atr_percentile", "adx", "macd", "macd_signal", "macd_hist",
                     "obv_trend", "bb_lower", "bb_upper", "bb_middle"]
        if any(pd.isna(row.get(c)) for c in critical):
            i += 1
            continue

        # ATR filter (ATF has its own internal filter too, but we pre-filter here)
        atr_pct = float(row.get("atr_percentile", 0.5))
        if atr_pct < atr_pct_min or atr_pct > atr_pct_max:
            atr_filtered += 1
            i += 1
            continue

        # Evaluate ATF signal
        prev = df_ind.iloc[i - 1]
        result = evaluate_atf_row(row, prev, i, profile=profile)

        if result is None:
            i += 1
            continue

        signal, score, trigger_name, adx_at_entry, trail_mult, sl_mult = result

        # ── Track diagnostics ──
        _diag_triggers[trigger_name] = _diag_triggers.get(trigger_name, 0) + 1
        _diag_scores.append(score)
        if adx_at_entry >= 35:
            _diag_trail_types["strong"] += 1
        elif adx_at_entry >= 25:
            _diag_trail_types["medium"] += 1
        else:
            _diag_trail_types["weak"] += 1

        # ── Trade simulation with ADX-adaptive trailing ──
        entry_price = signal.entry_price
        sl = signal.stop_loss
        atr = signal.atr
        is_long = signal.type == SignalType.LONG
        exit_price = None
        exit_reason = None
        bars = 0

        # Trailing state
        current_sl = sl
        be_triggered = False
        trailing_activated = False
        highest_favorable = entry_price
        sl_updates = 0
        was_trailing_exit = False

        # Trail distance is FIXED at entry time based on ADX
        trail_distance = atr * trail_mult

        for j in range(i + 1, min(i + _max_bars, n)):
            future = df_ind.iloc[j]
            f_close = float(future["close"])
            f_low = float(future["low"])
            f_high = float(future["high"])
            bars = j - i

            # Track high water mark
            if is_long:
                highest_favorable = max(highest_favorable, f_high)
            else:
                highest_favorable = min(highest_favorable, f_low)

            # Check SL hit
            sl_hit = False
            if is_long:
                if f_low <= current_sl:
                    sl_hit = True
            else:
                if f_high >= current_sl:
                    sl_hit = True

            # NO TP check — ATF uses trailing-only exit

            # BE trigger: price moved be_trigger * ATR in favor
            if not be_triggered:
                fav_dist = abs(highest_favorable - entry_price)
                if fav_dist >= atr * _be_trigger:
                    be_triggered = True
                    trailing_activated = True
                    current_sl = entry_price
                    _be_count += 1

            # Trailing stop (ratchet-only — only moves in favor)
            if trailing_activated:
                if is_long:
                    new_trail = highest_favorable - trail_distance
                    if new_trail > current_sl:
                        current_sl = new_trail
                        sl_updates += 1
                        _trail_count += 1
                else:
                    new_trail = highest_favorable + trail_distance
                    if new_trail < current_sl:
                        current_sl = new_trail
                        sl_updates += 1
                        _trail_count += 1

            # Exit check
            if sl_hit:
                exit_price = current_sl
                exit_reason = "trailing_sl" if trailing_activated else "sl"
                was_trailing_exit = trailing_activated
                break

        if exit_price is None:
            last_j = min(i + _max_bars, n) - 1
            exit_price = float(df_ind.iloc[last_j]["close"])
            exit_reason = "timeout"
            bars = last_j - i

        # Apply costs (limit order pricing)
        _, adj_exit, cost_pct = _apply_costs(
            entry_price, exit_price, is_long,
            _fee, _spread, _slip,
        )
        if is_long:
            pnl_pct = (adj_exit - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - adj_exit) / entry_price * 100

        trades.append(TradeResult(
            entry_ts=row.name,
            exit_ts=df_ind.iloc[min(i + bars, n - 1)].name,
            type=signal.type.value,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=sl,
            take_profit=signal.take_profit,
            atr=atr,
            rsi=signal.rsi,
            pnl_pct=round(pnl_pct, 4),
            pnl_abs=round(exit_price - entry_price, 2),
            bars_held=bars,
            exit_reason=exit_reason,
            atr_percentile=atr_pct,
            be_triggered=be_triggered,
            trailing_activated=trailing_activated,
            partial_tp_filled=False,
            sl_updates=sl_updates,
        ))

        # Register trailing exit for shorter cooldown on re-entry
        _from_strategy_atf = __import__("strategy_atf")
        _from_strategy_atf._register_signal(i + bars, was_trailing=was_trailing_exit)

        _running_pnl += pnl_pct
        _eq_points.append(round(_running_pnl, 2))

        _update_progress(
            signals_found=len(trades),
            last_signal_type=signal.type.value,
            last_signal_price=round(entry_price, 2),
            last_signal_pnl=round(pnl_pct, 2),
            equity_snapshot=list(_eq_points[-50:]),
        )

        i += bars + 1

    # Summary
    avg_score = np.mean(_diag_scores) if _diag_scores else 0
    logger.info(
        "ATF v1: %d trades, %d ATR filt, BE=%d, trail_updates=%d, "
        "avg_score=%.1f, triggers=%s, trail_types=%s",
        len(trades), atr_filtered, _be_count, _trail_count,
        avg_score, _diag_triggers, _diag_trail_types,
    )

    _diag = {
        "strategy": "atf_v1",
        "atr_filtered": atr_filtered,
        "be_triggered": _be_count,
        "trailing_updates": _trail_count,
        "avg_score": round(avg_score, 2),
        "triggers": _diag_triggers,
        "trail_types": _diag_trail_types,
        "score_distribution": {
            "min": min(_diag_scores) if _diag_scores else 0,
            "max": max(_diag_scores) if _diag_scores else 0,
            "avg": round(avg_score, 2),
        },
        "costs": {"fee_pct": _fee, "spread_bps": _spread, "slippage_bps": _slip},
    }
    return trades, atr_filtered, _diag


# ------------------------------------------------------------------
# v13: ATF v2 Simulation (StochRSI + BBWP integration)
# ------------------------------------------------------------------
def _simulate_atf_v2(
    df_ind: pd.DataFrame,
    atr_pct_min: float = 0.15,
    atr_pct_max: float = 0.85,
    profile=None,
    fee_pct: float = DEFAULT_FEE_PCT,
    spread_bps: float = DEFAULT_SPREAD_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> Tuple[List[TradeResult], int, dict]:
    r"""
    Simulacao ATF v2 para 15m/30m.
    Identico ao _simulate_atf mas usa evaluate_atf_v2_row.
    Retorna (signal, score, trigger_name, adx, trail_mult, sl_mult).
    """
    from strategy_atf_v2 import evaluate_atf_v2_row, reset_cooldown, ATF_V2_PARAMS

    reset_cooldown()
    _be_trigger = ATF_V2_PARAMS["be_trigger_atr"]
    _max_bars = ATF_V2_PARAMS["max_bars"]
    trades: List[TradeResult] = []
    atr_filtered = 0
    i = 0; n = len(df_ind)
    _scan_t0 = time.time(); _last_progress_time = 0.0
    _eq_points: list = []; _running_pnl = 0.0
    _fee = 0.016; _spread = 2.0; _slip = 2.0
    _be_count = 0; _trail_count = 0
    _diag_triggers = {}; _diag_scores = []
    _diag_trail_types = {"strong": 0, "medium": 0, "weak": 0}
    _diag_bbwp_squeeze = 0

    while i < n:
        row = df_ind.iloc[i]
        _now = time.time()
        if _now - _last_progress_time > 0.25:
            _last_progress_time = _now
            _elapsed = max(_now - _scan_t0, 0.01)
            _speed = i / _elapsed
            _scan_pct = 20 + (i / max(n, 1)) * 60
            _update_progress(
                phase="Escaneando candles (ATF v2)", phase_num=4,
                pct=round(_scan_pct, 1),
                message=f"ATF v2 {i:,}/{n:,} ({_speed:.0f}c/s) trades={len(trades)}",
                candles_total=n, candles_scanned=i, scan_speed=round(_speed),
                current_price=round(float(row.get("close", 0)), 2),
            )
        if i < 1: i += 1; continue
        critical = ["ema20","ema50","ema200","rsi","atr","rsi_delta",
                     "atr_percentile","adx","macd","macd_signal","macd_hist",
                     "obv_trend","bb_lower","bb_upper","bb_middle","stoch_rsi_k","bbwp"]
        if any(pd.isna(row.get(c)) for c in critical): i += 1; continue
        atr_pct = float(row.get("atr_percentile", 0.5))
        if atr_pct < atr_pct_min or atr_pct > atr_pct_max:
            atr_filtered += 1; i += 1; continue
        prev = df_ind.iloc[i - 1]
        result = evaluate_atf_v2_row(row, prev, i, profile=profile)
        if result is None: i += 1; continue
        signal, score, trigger_name, adx_at_entry, trail_mult, sl_mult = result
        _diag_triggers[trigger_name] = _diag_triggers.get(trigger_name, 0) + 1
        _diag_scores.append(score)
        if adx_at_entry >= 35: _diag_trail_types["strong"] += 1
        elif adx_at_entry >= 25: _diag_trail_types["medium"] += 1
        else: _diag_trail_types["weak"] += 1
        bbwp_val = float(row.get("bbwp", 50))
        if bbwp_val < 15: _diag_bbwp_squeeze += 1

        entry_price = signal.entry_price; sl = signal.stop_loss; atr = signal.atr
        is_long = signal.type == SignalType.LONG
        exit_price = None; exit_reason = None; bars = 0
        current_sl = sl; be_triggered = False; trailing_activated = False
        highest_favorable = entry_price; sl_updates = 0; was_trailing_exit = False
        trail_distance = atr * trail_mult

        for j in range(i + 1, min(i + _max_bars, n)):
            future = df_ind.iloc[j]
            f_close = float(future["close"]); f_low = float(future["low"])
            f_high = float(future["high"]); bars = j - i
            if is_long: highest_favorable = max(highest_favorable, f_high)
            else: highest_favorable = min(highest_favorable, f_low)
            sl_hit = False
            if is_long:
                if f_low <= current_sl: sl_hit = True
            else:
                if f_high >= current_sl: sl_hit = True
            if not be_triggered:
                fav_dist = abs(highest_favorable - entry_price)
                if fav_dist >= atr * _be_trigger:
                    be_triggered = True; trailing_activated = True
                    current_sl = entry_price; _be_count += 1
            if trailing_activated:
                if is_long:
                    new_trail = highest_favorable - trail_distance
                    if new_trail > current_sl: current_sl = new_trail; sl_updates += 1; _trail_count += 1
                else:
                    new_trail = highest_favorable + trail_distance
                    if new_trail < current_sl: current_sl = new_trail; sl_updates += 1; _trail_count += 1
            if sl_hit:
                exit_price = current_sl
                exit_reason = "trailing_sl" if trailing_activated else "sl"
                was_trailing_exit = trailing_activated; break
        if exit_price is None:
            last_j = min(i + _max_bars, n) - 1
            exit_price = float(df_ind.iloc[last_j]["close"])
            exit_reason = "timeout"; bars = last_j - i
        _, adj_exit, cost_pct = _apply_costs(entry_price, exit_price, is_long, _fee, _spread, _slip)
        if is_long: pnl_pct = (adj_exit - entry_price) / entry_price * 100
        else: pnl_pct = (entry_price - adj_exit) / entry_price * 100
        trades.append(TradeResult(
            entry_ts=row.name, exit_ts=df_ind.iloc[min(i + bars, n - 1)].name,
            type=signal.type.value, entry_price=entry_price, exit_price=exit_price,
            stop_loss=sl, take_profit=signal.take_profit, atr=atr, rsi=signal.rsi,
            pnl_pct=round(pnl_pct, 4), pnl_abs=round(exit_price - entry_price, 2),
            bars_held=bars, exit_reason=exit_reason, atr_percentile=atr_pct,
            be_triggered=be_triggered, trailing_activated=trailing_activated,
            partial_tp_filled=False, sl_updates=sl_updates))
        _from_atf_v2 = __import__("strategy_atf_v2")
        _from_atf_v2._register_signal(i + bars, was_trailing=was_trailing_exit)
        _running_pnl += pnl_pct; _eq_points.append(round(_running_pnl, 2))
        _update_progress(
            signals_found=len(trades), last_signal_type=signal.type.value,
            last_signal_price=round(entry_price, 2), last_signal_pnl=round(pnl_pct, 2),
            equity_snapshot=list(_eq_points[-50:]))
        i += bars + 1

    avg_score = np.mean(_diag_scores) if _diag_scores else 0
    logger.info(
        "ATF v2: %d trades, %d ATR filt, BE=%d, trail=%d, avg_score=%.1f, "
        "triggers=%s, bbwp_squeeze=%d",
        len(trades), atr_filtered, _be_count, _trail_count, avg_score,
        _diag_triggers, _diag_bbwp_squeeze,
    )
    _diag = {
        "strategy": "atf_v2", "atr_filtered": atr_filtered,
        "be_triggered": _be_count, "trailing_updates": _trail_count,
        "avg_score": round(avg_score, 2), "triggers": _diag_triggers,
        "trail_types": _diag_trail_types, "bbwp_squeeze_trades": _diag_bbwp_squeeze,
        "score_distribution": {
            "min": min(_diag_scores) if _diag_scores else 0,
            "max": max(_diag_scores) if _diag_scores else 0,
            "avg": round(avg_score, 2),
        },
    }
    return trades, atr_filtered, _diag


# ------------------------------------------------------------------
# BBWP Squeeze v5 Simulation
# ------------------------------------------------------------------
def _simulate_bbwp_squeeze(
    df_ind: pd.DataFrame,
    atr_pct_min: float = 0.10,
    atr_pct_max: float = 0.90,
    profile=None,
    fee_pct: float = DEFAULT_FEE_PCT,
    spread_bps: float = DEFAULT_SPREAD_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> Tuple[List[TradeResult], int, dict]:
    r"""
    Simulacao BBWP Squeeze v14 para 1h.

    v14 - SAIDAS AMPLIADAS:
    - SL: 2.2x ATR (v8)
    - TP1: 6.0x ATR (v14)
    - Pos-TP1 SL: TP1 - 0.2*ATR (floor de 5.8*ATR)
    - Trailing: 2.5x ATR (v14)
    - BBWP threshold: 15 (v8)
    - Cooldown direcional: mesma dir=2, oposta=1
    - R:R floor: (0.50*6.0 + 0.50*5.8)/2.2 = 2.68

    Custos: maker fee 0.016% + spread 2bps + slip 2bps (limit orders).
    """
    from strategy_bbwp_squeeze import (
        evaluate_bbwp_squeeze_row, reset_cooldown, BBWP_SQUEEZE_PARAMS,
    )

    reset_cooldown()
    _be_trigger = BBWP_SQUEEZE_PARAMS.get("be_trigger_atr_mult", 1.0)
    _trail_dist = BBWP_SQUEEZE_PARAMS.get("trailing_atr_mult", 1.5)
    _max_bars = BBWP_SQUEEZE_PARAMS.get("max_bars_held", 120)
    _use_trailing = BBWP_SQUEEZE_PARAMS.get("use_trailing", True)
    _tp1_pct = BBWP_SQUEEZE_PARAMS.get("tp1_pct", 0.50)
    _post_tp1_sl_buf = BBWP_SQUEEZE_PARAMS.get("post_tp1_sl_buffer", 0.5)

    trades: List[TradeResult] = []
    atr_filtered = 0
    i = 0
    n = len(df_ind)
    _scan_t0 = time.time()
    _last_progress_time = 0.0
    _eq_points: list = []
    _running_pnl = 0.0

    # Limit order costs (maker)
    _fee = 0.016
    _spread = 2.0
    _slip = 2.0

    # Diagnostics
    _be_count = 0
    _trail_count = 0
    _partial_tp_count = 0
    _diag_directions = {"long": 0, "short": 0}
    _diag_bbwp_at_entry = []
    _diag_tp1_exits = 0
    _diag_trailing_exits = 0
    _diag_sl_exits = 0
    _diag_timeout_exits = 0

    while i < n:
        row = df_ind.iloc[i]

        # Progress (throttled to ~4Hz)
        _now = time.time()
        if _now - _last_progress_time > 0.25:
            _last_progress_time = _now
            _elapsed = max(_now - _scan_t0, 0.01)
            _speed = i / _elapsed
            _scan_pct = 20 + (i / max(n, 1)) * 60
            _update_progress(
                phase="Escaneando candles (BBWP Squeeze v14)", phase_num=4,
                pct=round(_scan_pct, 1),
                message=f"BBWP Squeeze v14 {i:,}/{n:,} ({_speed:.0f}c/s) trades={len(trades)}",
                candles_total=n, candles_scanned=i, scan_speed=round(_speed),
                current_price=round(float(row.get("close", 0)), 2),
            )

        if i < 1:
            i += 1
            continue

        # Check NaN (BBWP Squeeze v14 needs bbwp, stoch_rsi, bb bands, adx)
        critical = [
            "ema20", "ema50", "ema200", "rsi", "atr", "atr_percentile",
            "bbwp", "stoch_rsi_k", "stoch_rsi_d", "bb_lower", "bb_upper",
            "volume", "volume_sma20", "adx",
        ]
        if any(pd.isna(row.get(c)) for c in critical):
            i += 1
            continue

        # ATR filter (strategy has internal filter too, but pre-filter here)
        atr_pct = float(row.get("atr_percentile", 0.5))
        if atr_pct < atr_pct_min or atr_pct > atr_pct_max:
            atr_filtered += 1
            i += 1
            continue

        # Evaluate BBWP Squeeze signal
        prev = df_ind.iloc[i - 1]
        result = evaluate_bbwp_squeeze_row(
            row, prev, i, df=df_ind, profile=profile,
        )

        if result is None:
            i += 1
            continue

        signal, bbwp_val, direction = result
        _diag_directions[direction] = _diag_directions.get(direction, 0) + 1
        _diag_bbwp_at_entry.append(round(bbwp_val, 1))

        # ---- Trade simulation with v4: partial TP + trailing ----
        entry_price = signal.entry_price
        sl = signal.stop_loss
        tp = signal.take_profit  # This is TP1 (3.0x ATR)
        atr = signal.atr
        is_long = signal.type == SignalType.LONG
        bars = 0

        current_sl = sl
        be_triggered = False
        trailing_activated = False
        highest_favorable = entry_price
        sl_updates = 0
        was_trailing_exit = False

        # v4: Partial TP state
        tp1_filled = False
        tp1_fill_price = 0.0

        for j in range(i + 1, min(i + _max_bars, n)):
            future = df_ind.iloc[j]
            f_close = float(future["close"])
            f_low = float(future["low"])
            f_high = float(future["high"])
            bars = j - i

            # Track high water mark
            if is_long:
                highest_favorable = max(highest_favorable, f_high)
            else:
                highest_favorable = min(highest_favorable, f_low)

            # Check SL/TP hit
            sl_hit = False
            tp_hit = False
            if is_long:
                if f_low <= current_sl:
                    sl_hit = True
                if f_high >= tp:
                    tp_hit = True
            else:
                if f_high >= current_sl:
                    sl_hit = True
                if f_low <= tp:
                    tp_hit = True

            # --- v4: TP1 hit (partial exit) ---
            if tp_hit and not tp1_filled:
                tp1_filled = True
                tp1_fill_price = tp
                _partial_tp_count += 1
                _diag_tp1_exits += 1

                # v11: After TP1, set SL at TP1 - post_tp1_sl_buffer*ATR
                # This creates a HIGH FLOOR (2.5*ATR profit on trailing portion)
                # Trailing at 1.5x ATR then ratchets ABOVE this floor
                if _use_trailing:
                    be_triggered = True
                    trailing_activated = True
                    post_tp1_sl_dist = atr * _post_tp1_sl_buf
                    if is_long:
                        current_sl = tp - post_tp1_sl_dist  # TP1 - 0.5*ATR
                    else:
                        current_sl = tp + post_tp1_sl_dist  # TP1 + 0.5*ATR (SHORT)
                    _be_count += 1

                # Don't break — continue with trailing on remaining 50%
                if not _use_trailing:
                    # No trailing = full exit at TP1
                    break
                # Continue to trailing loop below

            # --- Both SL and TP hit same bar after TP1 filled ---
            if tp_hit and sl_hit and tp1_filled:
                # SL hit on remaining position — exit at SL
                # Calculate weighted PnL from TP1 (40%) + SL (60%)
                if is_long:
                    tp1_pnl = (tp1_fill_price - entry_price) / entry_price
                    sl_pnl = (current_sl - entry_price) / entry_price
                else:
                    tp1_pnl = (entry_price - tp1_fill_price) / entry_price
                    sl_pnl = (entry_price - current_sl) / entry_price

                # Apply costs
                _, adj_tp1, cost1 = _apply_costs(entry_price, tp1_fill_price, is_long, _fee, _spread, _slip)
                _, adj_sl, cost2 = _apply_costs(entry_price, current_sl, is_long, _fee, _spread, _slip)
                if is_long:
                    tp1_pnl = (adj_tp1 - entry_price) / entry_price * 100
                    sl_pnl = (adj_sl - entry_price) / entry_price * 100
                else:
                    tp1_pnl = (entry_price - adj_tp1) / entry_price * 100
                    sl_pnl = (entry_price - adj_sl) / entry_price * 100

                weighted_pnl = _tp1_pct * tp1_pnl + (1 - _tp1_pct) * sl_pnl

                trades.append(TradeResult(
                    entry_ts=row.name,
                    exit_ts=df_ind.iloc[j].name,
                    type=signal.type.value,
                    entry_price=entry_price,
                    exit_price=tp,  # TP1 price as reference
                    stop_loss=sl, take_profit=tp, atr=atr, rsi=signal.rsi,
                    pnl_pct=round(weighted_pnl, 4),
                    pnl_abs=round(tp - entry_price, 2),
                    bars_held=bars, exit_reason="tp1_then_sl",
                    atr_percentile=atr_pct, be_triggered=be_triggered,
                    trailing_activated=trailing_activated,
                    partial_tp_filled=True, sl_updates=sl_updates,
                ))
                _diag_sl_exits += 1
                was_trailing_exit = True
                break

            # --- SL hit (before TP1 or after TP1 on remaining) ---
            if sl_hit and not tp_hit:
                if tp1_filled:
                    # TP1 already filled — exit remaining at SL
                    # Apply costs
                    _, adj_tp1, cost1 = _apply_costs(entry_price, tp1_fill_price, is_long, _fee, _spread, _slip)
                    _, adj_sl, cost2 = _apply_costs(entry_price, current_sl, is_long, _fee, _spread, _slip)
                    if is_long:
                        tp1_pnl = (adj_tp1 - entry_price) / entry_price * 100
                        sl_pnl = (adj_sl - entry_price) / entry_price * 100
                    else:
                        tp1_pnl = (entry_price - adj_tp1) / entry_price * 100
                        sl_pnl = (entry_price - adj_sl) / entry_price * 100
                    weighted_pnl = _tp1_pct * tp1_pnl + (1 - _tp1_pct) * sl_pnl

                    trades.append(TradeResult(
                        entry_ts=row.name,
                        exit_ts=df_ind.iloc[j].name,
                        type=signal.type.value,
                        entry_price=entry_price,
                        exit_price=current_sl,
                        stop_loss=sl, take_profit=tp, atr=atr, rsi=signal.rsi,
                        pnl_pct=round(weighted_pnl, 4),
                        pnl_abs=round(current_sl - entry_price, 2),
                        bars_held=bars, exit_reason="trailing_sl",
                        atr_percentile=atr_pct, be_triggered=be_triggered,
                        trailing_activated=trailing_activated,
                        partial_tp_filled=True, sl_updates=sl_updates,
                    ))
                    _diag_trailing_exits += 1
                    was_trailing_exit = True
                else:
                    # SL hit before TP1 — full loss
                    _, adj_exit, cost_pct = _apply_costs(
                        entry_price, current_sl, is_long, _fee, _spread, _slip,
                    )
                    if is_long:
                        pnl_pct = (adj_exit - entry_price) / entry_price * 100
                    else:
                        pnl_pct = (entry_price - adj_exit) / entry_price * 100

                    trades.append(TradeResult(
                        entry_ts=row.name,
                        exit_ts=df_ind.iloc[j].name,
                        type=signal.type.value,
                        entry_price=entry_price,
                        exit_price=current_sl, stop_loss=sl, take_profit=tp,
                        atr=atr, rsi=signal.rsi,
                        pnl_pct=round(pnl_pct, 4),
                        pnl_abs=round(current_sl - entry_price, 2),
                        bars_held=bars, exit_reason="sl",
                        atr_percentile=atr_pct, be_triggered=False,
                        trailing_activated=False, partial_tp_filled=False,
                        sl_updates=0,
                    ))
                    _diag_sl_exits += 1
                break

            # --- Trailing stop (ratchet-only — only moves in favor) ---
            if trailing_activated and _use_trailing:
                trail_distance = atr * _trail_dist
                if is_long:
                    new_trail = highest_favorable - trail_distance
                    if new_trail > current_sl:
                        current_sl = new_trail
                        sl_updates += 1
                        _trail_count += 1
                else:
                    new_trail = highest_favorable + trail_distance
                    if new_trail < current_sl:
                        current_sl = new_trail
                        sl_updates += 1
                        _trail_count += 1

        # === End of bar loop — handle exit ===
        if len(trades) > 0 and trades[-1].entry_ts == row.name:
            # Trade was already appended inside the loop (SL/TP hit)
            pass
        else:
            # Timeout or no exit triggered
            last_j = min(i + _max_bars, n) - 1
            exit_price_close = float(df_ind.iloc[last_j]["close"])
            bars = last_j - i

            if tp1_filled:
                # TP1 was hit, remaining exited at close (timeout)
                _, adj_tp1, cost1 = _apply_costs(entry_price, tp1_fill_price, is_long, _fee, _spread, _slip)
                _, adj_close, cost2 = _apply_costs(entry_price, exit_price_close, is_long, _fee, _spread, _slip)
                if is_long:
                    tp1_pnl = (adj_tp1 - entry_price) / entry_price * 100
                    close_pnl = (adj_close - entry_price) / entry_price * 100
                else:
                    tp1_pnl = (entry_price - adj_tp1) / entry_price * 100
                    close_pnl = (entry_price - adj_close) / entry_price * 100
                weighted_pnl = _tp1_pct * tp1_pnl + (1 - _tp1_pct) * close_pnl

                trades.append(TradeResult(
                    entry_ts=row.name,
                    exit_ts=df_ind.iloc[last_j].name,
                    type=signal.type.value,
                    entry_price=entry_price,
                    exit_price=exit_price_close,
                    stop_loss=sl, take_profit=tp, atr=atr, rsi=signal.rsi,
                    pnl_pct=round(weighted_pnl, 4),
                    pnl_abs=round(exit_price_close - entry_price, 2),
                    bars_held=bars, exit_reason="tp1_then_timeout",
                    atr_percentile=atr_pct, be_triggered=be_triggered,
                    trailing_activated=trailing_activated,
                    partial_tp_filled=True, sl_updates=sl_updates,
                ))
                _diag_timeout_exits += 1
                was_trailing_exit = trailing_activated
            else:
                # No TP1 hit — full exit at close
                _, adj_exit, cost_pct = _apply_costs(
                    entry_price, exit_price_close, is_long, _fee, _spread, _slip,
                )
                if is_long:
                    pnl_pct = (adj_exit - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - adj_exit) / entry_price * 100

                trades.append(TradeResult(
                    entry_ts=row.name,
                    exit_ts=df_ind.iloc[last_j].name,
                    type=signal.type.value,
                    entry_price=entry_price,
                    exit_price=exit_price_close, stop_loss=sl, take_profit=tp,
                    atr=atr, rsi=signal.rsi,
                    pnl_pct=round(pnl_pct, 4),
                    pnl_abs=round(exit_price_close - entry_price, 2),
                    bars_held=bars, exit_reason="timeout",
                    atr_percentile=atr_pct, be_triggered=False,
                    trailing_activated=False, partial_tp_filled=False,
                    sl_updates=0,
                ))
                _diag_timeout_exits += 1

        # Register trailing exit for shorter cooldown on re-entry
        _from_bbwp = __import__("strategy_bbwp_squeeze")
        _from_bbwp._register_signal(i + bars, was_trailing=was_trailing_exit, direction=direction)

        _running_pnl += trades[-1].pnl_pct
        _eq_points.append(round(_running_pnl, 2))

        _update_progress(
            signals_found=len(trades),
            last_signal_type=signal.type.value,
            last_signal_price=round(entry_price, 2),
            last_signal_pnl=round(trades[-1].pnl_pct, 2),
            equity_snapshot=list(_eq_points[-50:]),
        )

        i += bars + 1

    # Summary
    avg_bbwp = np.mean(_diag_bbwp_at_entry) if _diag_bbwp_at_entry else 0
    logger.info(
        "BBWP Squeeze v14: %d trades, %d ATR filt, BE=%d, trail=%d, partial=%d, "
        "longs=%d, shorts=%d, avg_bbwp=%.1f, exits=[tp1=%d trail=%d sl=%d timeout=%d]",
        len(trades), atr_filtered, _be_count, _trail_count, _partial_tp_count,
        _diag_directions.get("long", 0), _diag_directions.get("short", 0),
        avg_bbwp, _diag_tp1_exits, _diag_trailing_exits, _diag_sl_exits, _diag_timeout_exits,
    )

    _diag = {
        "strategy": "bbwp_squeeze_v14",
        "atr_filtered": atr_filtered,
        "be_triggered": _be_count,
        "trailing_updates": _trail_count,
        "partial_tp_count": _partial_tp_count,
        "directions": _diag_directions,
        "exits": {
            "tp1": _diag_tp1_exits,
            "trailing": _diag_trailing_exits,
            "sl": _diag_sl_exits,
            "timeout": _diag_timeout_exits,
        },
        "bbwp_at_entry": {
            "min": min(_diag_bbwp_at_entry) if _diag_bbwp_at_entry else 0,
            "max": max(_diag_bbwp_at_entry) if _diag_bbwp_at_entry else 0,
            "avg": round(avg_bbwp, 1),
        },
        "costs": {"fee_pct": _fee, "spread_bps": _spread, "slippage_bps": _slip},
    }
    return trades, atr_filtered, _diag


# ------------------------------------------------------------------
# v15: Confluence Multi-Signal simulation
# ------------------------------------------------------------------
def _simulate_confluence_v15(
    df_ind: pd.DataFrame,
    atr_pct_min: float = 0.10,
    atr_pct_max: float = 0.90,
    profile=None,
    fee_pct: float = DEFAULT_FEE_PCT,
    spread_bps: float = DEFAULT_SPREAD_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> Tuple[List[TradeResult], int, dict]:
    r"""Simulacao Confluence v15 para 1h."""
    from strategy_confluence_v15 import (
        evaluate_confluence_v15_row, reset_cooldown, CONFLUENCE_PARAMS,
    )
    from strategy import SignalType

    reset_cooldown()
    _trail_dist = CONFLUENCE_PARAMS.get("trailing_atr_mult", 3.0)
    _max_bars = CONFLUENCE_PARAMS.get("max_bars_held", 120)
    _tp1_pct = CONFLUENCE_PARAMS.get("tp1_pct", 0.50)
    _post_tp1_buf = CONFLUENCE_PARAMS.get("post_tp1_sl_buffer", 0.2)

    trades: List[TradeResult] = []
    atr_filtered = 0
    i = 0
    n = len(df_ind)
    _fee = 0.016; _spread = 2.0; _slip = 2.0

    while i < n - 1:
        row = df_ind.iloc[i]
        prev = df_ind.iloc[i - 1] if i > 0 else row
        critical = ["ema20", "ema50", "ema200", "rsi", "atr", "atr_percentile",
                    "macd", "macd_signal", "volume", "volume_sma20", "adx",
                    "stoch_rsi_k", "stoch_rsi_d", "obv", "obv_sma20", "obv_trend"]
        if any(pd.isna(row.get(c)) for c in critical):
            i += 1; continue
        atr_pct = float(row.get("atr_percentile", 0.5))
        if not (atr_pct_min <= atr_pct <= atr_pct_max):
            atr_filtered += 1; i += 1; continue

        result = evaluate_confluence_v15_row(row, prev, i, df=df_ind)
        if result is None:
            i += 1; continue

        signal, score, direction = result
        is_long = signal.type == SignalType.LONG
        entry_price = signal.entry_price
        sl, tp, atr = signal.stop_loss, signal.take_profit, signal.atr
        if atr <= 0 or entry_price <= 0:
            i += 1; continue

        csl = sl; tp1f = False; tp1_price = 0.0; hwm = entry_price
        was_trailing = False; exit_done = False; bars = 0
        hit_sl = hit_tp = False

        for j in range(i + 1, min(i + _max_bars, n)):
            f = df_ind.iloc[j]
            fc, fl, fh = float(f["close"]), float(f["low"]), float(f["high"])
            bars = j - i
            hwm = max(hwm, fh) if is_long else min(hwm, fl)
            hit_sl = (fl <= csl) if is_long else (fh >= csl)
            hit_tp = (fh >= tp) if is_long else (fl <= tp)

            if hit_tp and not tp1f:
                tp1f = True; tp1_price = tp
                buf = atr * _post_tp1_buf
                csl = (tp - buf) if is_long else (tp + buf)
                was_trailing = True
            if hit_sl or (hit_tp and tp1f):
                exit_done = True; break

            if tp1f:
                td = atr * _trail_dist
                if is_long:
                    ns = hwm - td
                    if ns > csl: csl = ns
                else:
                    ns = hwm + td
                    if ns < csl: csl = ns

        last_j = min(i + max(bars, 1), n - 1)
        xp = float(df_ind.iloc[last_j]["close"])

        if tp1f:
            a1, _, _ = _apply_costs(entry_price, tp1_price, is_long, _fee, _spread, _slip)
            a2, _, _ = _apply_costs(entry_price, csl if exit_done else xp, is_long, _fee, _spread, _slip)
            if is_long:
                pnl = _tp1_pct * (a1 - entry_price) / entry_price * 100 + (1 - _tp1_pct) * (a2 - entry_price) / entry_price * 100
            else:
                pnl = _tp1_pct * (entry_price - a1) / entry_price * 100 + (1 - _tp1_pct) * (entry_price - a2) / entry_price * 100
        else:
            _, adj_exit, _ = _apply_costs(entry_price, csl if exit_done else xp, is_long, _fee, _spread, _slip)
            pnl = (adj_exit - entry_price) / entry_price * 100 if is_long else (entry_price - adj_exit) / entry_price * 100

        if exit_done and hit_sl and not hit_tp: reason = "sl"
        elif tp1f and exit_done: reason = "trailing"
        elif exit_done: reason = "tp"
        else: reason = "timeout"

        trades.append(TradeResult(
            entry_ts=df_ind.index[i], exit_ts=df_ind.index[last_j],
            type=direction.upper(), entry_price=entry_price, exit_price=xp,
            stop_loss=sl, take_profit=tp, atr=atr, rsi=float(row.get("rsi", 0)),
            pnl_pct=round(pnl, 4), pnl_abs=round(xp - entry_price, 2),
            bars_held=bars, exit_reason=reason, atr_percentile=atr_pct,
            be_triggered=tp1f, trailing_activated=was_trailing, partial_tp_filled=tp1f, sl_updates=0))

        from strategy_confluence_v15 import _register_signal as _cr
        _cr(i + max(bars, 1), direction=direction)
        i += max(bars, 1) + 1

    _diag = {"strategy": "confluence_v15", "atr_filtered": atr_filtered,
            "costs": {"fee_pct": _fee, "spread_bps": _spread, "slippage_bps": _slip}}
    return trades, atr_filtered, _diag


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
            # The adapted functions are too permissive for trend-follow.
            # CTEV needs ADX>=30 for quality trend signals.
            #
            # v7.1 FIX: WEAK_UPTREND LONG ADX floor.
            # evaluate_long uses the OLD regime column; when old regime="transition"
            # (allow_transition=True), ADX>=30 is bypassed. This causes low-ADX
            # trades (21-25) to fire in WEAK_UPTREND. Analysis showed 10/14 LONG
            # losers had ADX<30. Fix: require ADX >= ADX_TRENDING_MIN (22.0)
            # for WEAK_UPTREND LONGs, matching regime_engine's trending threshold.
            if regime_v2 == "WEAK_UPTREND":
                _weak_adx = float(row.get("adx", 0))
                if _weak_adx < 22.0:
                    neutral_filtered += 1
                    i += 1
                    continue

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
    liga_crypto: bool = False,  # Liga Crypto multi-TF methodology
) -> Tuple[BacktestMetrics, List[TradeResult]]:
    """
    Executa backtest completo da estrategia CTEV.

    v6: Resolve automaticamente o StrategyProfile pelo timeframe.
    v7: Adicionado regime-switching (mean-reversion em ranging, filtro avancado).
    Liga Crypto: Analise hierarquica multi-timeframe (1W->1D->4H->1H->15M).

    Parameters:
        advanced: se True, usa simulate_trades_advanced com sizing/trailing
        regime_switching: se True, usa regime_engine para rotear estrategia
            por regime (trend_follow para tendencias, mean_reversion para lateral).
            Validado: +5.24pp vs baseline em BTC/USDT 1h 730d.
        liga_crypto: se True, usa metodologia Liga Crypto (multi-TF).
            Requer busca de dados 5 timeframes.

    Returns:
        (BacktestMetrics, List[TradeResult])
    """
    # Resolve profile pelo timeframe
    if liga_crypto:
        from strategy_profiles import PROFILE_LIGA_CRYPTO
        profile = PROFILE_LIGA_CRYPTO
    else:
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

    # 4. Simula trades — ROUTING INTELIGENTE POR TIMEFRAME
    _diag = {}
    if profile.name == "INTRADAY":
        # v13: ATF v2 para 15m/30m (ATF v1 + StochRSI + BBWP)
        trades, atr_filtered, _diag = _simulate_atf_v2(
            df_clean, _atr_min, _atr_max, profile=profile,
        )
    elif profile.name == 'ADAPTIVE_MOM':
        from strategy_adaptive_momentum import evaluate_adaptive_momentum, reset_cooldown
        ADAPTIVE_MOM_PARAMS = {
            'adx_trend_threshold': 22,
            'adx_range_max': 28,
            'roc_fast_period': 8,
            'roc_slow_period': 16,
            'volume_mult': 1.0,
            'adx_min_trend': 12,
            'rsi_long_min': 45, 'rsi_long_max': 80,
            'rsi_short_min': 20, 'rsi_short_max': 50,
            'roc_min_pct': 0.3,
            'use_ema200_trend': False,
            'rsi_oversold': 28, 'rsi_overbought': 73,
            'bb_touch_pct': 0.08,
            'stoch_rsi_oversold': 30, 'stoch_rsi_overbought': 70,
            'sl_range_mult': 1.5, 'tp_range_mult': 3.0,
            'sl_trend_mult': 2.0, 'tp_trend_mult': 5.0,
            'cooldown': 3,
            'cooldown_opp_dir': 1, 'cooldown_same_dir': 2,
            'sl_atr_mult': 1.8, 'tp_atr_mult': 5.0,
            'trailing_atr_mult': 2.5, 'post_tp1_sl_buffer': 0.1,
            'max_bars_held': 120,
        }
        _be_trigger = ADAPTIVE_MOM_PARAMS.get('be_trigger_atr_mult', 1.0)
        _trail_dist = ADAPTIVE_MOM_PARAMS.get('trailing_atr_mult', 2.5)
        _max_bars = ADAPTIVE_MOM_PARAMS.get('max_bars_held', 120)
        trades, atr_filtered, _diag = _simulate_bbwp_squeeze(
            df_clean, _atr_min, _atr_max, profile=ADAPTIVE_MOM_PARAMS,
        )
        _diag['strategy'] = 'adaptive_momentum_v1'
    elif profile.name == 'BBWP_SQUEEZE':
        # v15: Confluence Multi-Signal para 1h
        trades, atr_filtered, _diag = _simulate_confluence_v15(
            df_clean, _atr_min, _atr_max, profile=profile,
        )
    elif regime_switching:
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
        # v17.0: Multi-Strategy Concurrent Positions
        from sim_concurrent import simulate_trades_concurrent
        trades, atr_filtered, _diag = simulate_trades_concurrent(
            df_clean, _atr_min, _atr_max,
            profile=profile,
        )
    elif profile.name == "LIGA_CRYPTO":
        # Liga Crypto: analise hierarquica multi-timeframe (1W->1D->4H->1H->15M)
        from sim_liga_crypto import (
            fetch_liga_crypto_data, prepare_liga_crypto_dfs,
            simulate_liga_crypto,
        )
        _update_progress(
            phase="Baixando MTF data", phase_num=2, pct=8,
            message="Liga Crypto: baixando 5 timeframes (1W,1D,4H,1H,15M)...",
        )
        raw_dfs = fetch_liga_crypto_data(symbol, days)
        _update_progress(
            phase="Calculando indicadores MTF", phase_num=3, pct=14,
            message="Liga Crypto: calculando indicadores para 5 timeframes...",
        )
        dfs_ind = prepare_liga_crypto_dfs(raw_dfs)
        trades, atr_filtered, _diag = simulate_liga_crypto(dfs_ind)
    elif profile.name == "STANDARD":
        # v17.0: Multi-Strategy Concurrent Positions (DEFAULT para 1h)
        from sim_concurrent import simulate_trades_concurrent
        trades, atr_filtered, _diag = simulate_trades_concurrent(
            df_clean, _atr_min, _atr_max,
            profile=profile,
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
