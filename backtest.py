"""
backtest.py
------------
Módulo de backtesting da estratégia CTEV usando dados históricos da Binance.

Funcionalidades:
    - Download de dados OHLCV históricos via ccxt (até 1 ano de candles 1H)
    - Cálculo de todos os indicadores CTEV (EMA200, BB, RSI, Volume SMA, ATR)
    - Simulação de trades com gestão de risco (SL e TP baseados em ATR)
    - Métricas de performance: Win Rate, Profit Factor, Max Drawdown, Sharpe,
      Total Trades, Avg Win, Avg Loss
    - Walk-Forward Analysis (WFO): janela rolling treino/teste
    - Comparação com Buy & Hold

Referências:
    - kernc.github.io/backtesting.py — framework Python para backtesting
    - InteractiveBrokers (2025): "Walk Forward Analysis — sophisticated technique
      to test and optimize trading strategies"
    - CoinBureau (2026): "To backtest a crypto strategy, turn the idea into exact
      trading rules, choose historical data matching the asset and timeframe"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import ccxt
import numpy as np
import pandas as pd

from indicators import compute_indicators
from strategy import (
    RSI_LONG_THRESHOLD,
    RSI_SHORT_THRESHOLD,
    SL_ATR_MULT,
    TP_ATR_MULT,
    VOLUME_MULTIPLIER,
    Signal,
    SignalType,
    evaluate_long,
    evaluate_short,
)

logger = logging.getLogger("ctev.backtest")


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


@dataclass
class BacktestMetrics:
    """Métricas agregadas do backtest."""
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
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    avg_bars_held: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    buy_hold_pct: float = 0.0
    period_start: str = ""
    period_end: str = ""
    atr_pct_filtered: int = 0  # sinais filtrados por volatilidade

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
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "avg_bars_held": round(self.avg_bars_held, 1),
            "best_trade_pct": round(self.best_trade_pct, 4),
            "worst_trade_pct": round(self.worst_trade_pct, 4),
            "buy_hold_pct": round(self.buy_hold_pct, 4),
            "period_start": self.period_start,
            "period_end": self.period_end,
            "atr_pct_filtered": self.atr_pct_filtered,
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
# Download de dados históricos
# ------------------------------------------------------------------
def fetch_historical_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    since_days_ago: int = 365,
) -> pd.DataFrame:
    """
    Baixa dados históricos da Binance via ccxt (sem autenticação).

    O endpoint publico da Binance permite até 1500 candles por request.
    Para 1 ano de candles 1H (~8760), fazemos ~6 requests paginadas.

    Returns:
        DataFrame com colunas open, high, low, close, volume e index datetime UTC.
    """
    exchange = ccxt.binance({"enableRateLimit": True})
    since_ms = int(
        (datetime.now(timezone.utc).replace(tzinfo=None)
         - pd.Timedelta(days=since_days_ago)).timestamp() * 1000
    )
    all_ohlcv: list = []

    while True:
        try:
            logger.info(
                "Baixando candles %s %s desde %s...",
                symbol, timeframe, datetime.utcfromtimestamp(since_ms / 1000),
            )
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=1000)
            if not batch:
                break
            all_ohlcv.extend(batch)
            # Próximo batch: timestamp do último candle + 1 candle
            since_ms = batch[-1][0] + (60 * 60 * 1000)  # +1h em ms
            if len(batch) < 1000:
                break
        except Exception as exc:
            logger.error("Erro ao baixar dados: %s", exc)
            break

    exchange.close()

    if not all_ohlcv:
        raise RuntimeError("Nenhum dado histórico baixado.")

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    df.drop(columns=["timestamp"], inplace=True)
    df.drop_duplicates(inplace=True)
    df.sort_index(inplace=True)

    logger.info(
        "Download completo: %d candles (%s a %s)",
        len(df), df.index[0], df.index[-1],
    )
    return df


# ------------------------------------------------------------------
# Simulação de trades
# ------------------------------------------------------------------
def simulate_trades(
    df_ind: pd.DataFrame,
    atr_pct_min: float = 0.20,
    atr_pct_max: float = 0.80,
) -> List[TradeResult]:
    """
    Percorre o DataFrame com indicadores calculados e simula trades
    da estratégia CTEV, respeitando SL e TP baseados em ATR.

    Parameters:
        df_ind: DataFrame com indicadores (output de compute_indicators)
        atr_pct_min: filtro de volatilidade mínimo (default 0.20)
        atr_pct_max: filtro de volatilidade máximo (default 0.80)

    Returns:
        Lista de TradeResult com todos os trades simulados.
    """
    trades: List[TradeResult] = []
    atr_filtered = 0
    i = 0
    n = len(df_ind)

    while i < n:
        row = df_ind.iloc[i]

        # Verifica NaN nos indicadores críticos
        critical = ["ema200", "bb_lower", "bb_upper", "rsi", "volume_sma20", "atr", "atr_percentile"]
        if any(pd.isna(row.get(c)) for c in critical):
            i += 1
            continue

        atr_pct = float(row.get("atr_percentile", 0.5))

        # Filtro de volatilidade: só opera se ATR percentile está na faixa
        if atr_pct < atr_pct_min or atr_pct > atr_pct_max:
            atr_filtered += 1
            i += 1
            continue

        # Tenta LONG
        signal = evaluate_long(row)
        if signal is None:
            signal = evaluate_short(row)

        if signal is None:
            i += 1
            continue

        # Simula o trade: percorre candles seguintes até SL ou TP
        entry_price = signal.entry_price
        sl = signal.stop_loss
        tp = signal.take_profit
        atr = signal.atr
        is_long = signal.type == SignalType.LONG

        exit_price = None
        exit_reason = None
        bars = 0

        for j in range(i + 1, min(i + 48, n)):  # Max 48 candles (48h)
            future = df_ind.iloc[j]
            future_close = float(future["close"])
            future_low = float(future["low"])
            future_high = float(future["high"])
            bars = j - i

            if is_long:
                # SL: low do candle <= stop loss
                if future_low <= sl:
                    exit_price = sl
                    exit_reason = "sl"
                    break
                # TP: high do candle >= take profit
                if future_high >= tp:
                    exit_price = tp
                    exit_reason = "tp"
                    break
            else:
                # SL: high do candle >= stop loss
                if future_high >= sl:
                    exit_price = sl
                    exit_reason = "sl"
                    break
                # TP: low do candle <= take profit
                if future_low <= tp:
                    exit_price = tp
                    exit_reason = "tp"
                    break

        if exit_price is None:
            # Timeout: sai no close do último candle avaliado
            last_j = min(i + 48, n) - 1
            exit_price = float(df_ind.iloc[last_j]["close"])
            exit_reason = "timeout"
            bars = last_j - i

        # Calcula PnL
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

        # Avança após o trade fechar (evita trades sobrepostos)
        i += bars + 1

    logger.info(
        "Simulação completa: %d trades, %d sinais filtrados por volatilidade",
        len(trades), atr_filtered,
    )
    return trades, atr_filtered


# ------------------------------------------------------------------
# Cálculo de métricas
# ------------------------------------------------------------------
def calculate_metrics(
    trades: List[TradeResult],
    df: pd.DataFrame,
    atr_filtered: int = 0,
) -> BacktestMetrics:
    """Calcula todas as métricas de performance a partir dos trades simulados."""
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
        sharpe = np.mean(pnls) / np.std(pnls) * (365 ** 0.5)  # anualizado ~365 trades max
    else:
        sharpe = 0.0

    # Buy & Hold
    if len(df) > 1:
        buy_hold_pct = (float(df.iloc[-1]["close"]) - float(df.iloc[0]["close"])) / float(df.iloc[0]["close"]) * 100
    else:
        buy_hold_pct = 0.0

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
        max_drawdown_pct=max_dd,
        sharpe_ratio=sharpe,
        avg_bars_held=np.mean([t.bars_held for t in trades]) if trades else 0.0,
        best_trade_pct=max(pnls) if pnls else 0.0,
        worst_trade_pct=min(pnls) if pnls else 0.0,
        buy_hold_pct=buy_hold_pct,
        period_start=str(df.index[0]),
        period_end=str(df.index[-1]),
        atr_pct_filtered=atr_filtered,
    )


# ------------------------------------------------------------------
# Backtest completo
# ------------------------------------------------------------------
def run_backtest(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    days: int = 365,
    atr_pct_min: float = 0.20,
    atr_pct_max: float = 0.80,
) -> Tuple[BacktestMetrics, List[TradeResult]]:
    """
    Executa backtest completo da estratégia CTEV.

    Returns:
        (BacktestMetrics, List[TradeResult])
    """
    logger.info(
        "Iniciando backtest CTEV: %s %s %d dias ATR_pct=[%.0f%%, %.0f%%]",
        symbol, timeframe, days, atr_pct_min * 100, atr_pct_max * 100,
    )

    # 1. Download dados
    df = fetch_historical_ohlcv(symbol, timeframe, days)

    # 2. Calcula indicadores
    df_ind = compute_indicators(df)

    # 3. Remove linhas com NaN (início do dataset)
    df_clean = df_ind.dropna(subset=[
        "ema200", "bb_lower", "bb_upper", "rsi", "volume_sma20", "atr", "atr_percentile"
    ]).copy()
    logger.info("DataFrame limpo: %d candles (de %d originais)", len(df_clean), len(df_ind))

    # 4. Simula trades
    trades, atr_filtered = simulate_trades(df_clean, atr_pct_min, atr_pct_max)

    # 5. Calcula métricas
    metrics = calculate_metrics(trades, df_clean, atr_filtered)

    logger.info(
        "Backtest concluído: %d trades | WR=%.1f%% | PF=%.2f | MaxDD=%.2f%% | PnL=%.2f%%",
        metrics.total_trades, metrics.win_rate, metrics.profit_factor,
        metrics.max_drawdown_pct, metrics.total_pnl_pct,
    )

    return metrics, trades


# ------------------------------------------------------------------
# Walk-Forward Analysis
# ------------------------------------------------------------------
def run_walk_forward(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    total_days: int = 365,
    train_days: int = 180,
    test_days: int = 60,
    step_days: int = 60,
    atr_pct_min: float = 0.20,
    atr_pct_max: float = 0.80,
) -> List[WalkForwardResult]:
    """
    Executa Walk-Forward Analysis da estratégia CTEV.

    Divide o dataset em janelas treino/teste e rola para frente.
    Cada janela: [train_days] de treino + [test_days] de teste,
    avançando [step_days] por iteração.

    Returns:
        Lista de WalkForwardResult com métricas de cada janela.
    """
    logger.info(
        "Iniciando Walk-Forward: train=%dd test=%dd step=%dd total=%dd",
        train_days, test_days, step_days, total_days,
    )

    # Download completo
    df = fetch_historical_ohlcv(symbol, timeframe, total_days + train_days)
    df_ind = compute_indicators(df)
    df_clean = df_ind.dropna(subset=[
        "ema200", "bb_lower", "bb_upper", "rsi", "volume_sma20", "atr", "atr_percentile"
    ]).copy()

    results: List[WalkForwardResult] = []
    window_id = 0

    total_hours_train = train_days * 24
    total_hours_test = test_days * 24
    step_hours = step_days * 24
    n = len(df_clean)

    start = 0
    while start + total_hours_train + total_hours_test <= n:
        train_slice = df_clean.iloc[start:start + total_hours_train]
        test_slice = df_clean.iloc[start + total_hours_train:start + total_hours_train + total_hours_test]

        if len(train_slice) < 100 or len(test_slice) < 10:
            break

        # Treino
        train_trades, _ = simulate_trades(train_slice, atr_pct_min, atr_pct_max)
        train_metrics = calculate_metrics(train_trades, train_slice)

        # Teste
        test_trades, _ = simulate_trades(test_slice, atr_pct_min, atr_pct_max)
        test_metrics = calculate_metrics(test_trades, test_slice)

        # Degradação (quanto o WR caiu do treino pro teste)
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
        start += step_hours

    logger.info("Walk-Forward concluído: %d janelas.", len(results))
    return results
