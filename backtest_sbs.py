#!/usr/bin/env python3
"""
backtest_sbs.py
-------------
Backtest autônomo para a Squeeze Breakout Strategy (SBS).

Roda independentemente do servidor, com loop de otimização automática.
Gera resultados detalhados para 15min e 1h.
"""
from __future__ import annotations

import sys
import os
import time
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indicators import compute_indicators
from strategy_squeeze_breakout import (
    SBS_PARAMS, evaluate_sbs_row, reset_cooldown,
    compute_stoch_rsi, compute_bbwp, detect_rsi_divergence,
    _register_signal,
)
from backtest import (
    TradeResult, BacktestMetrics, calculate_metrics,
    fetch_historical_ohlcv, DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS,
    DEFAULT_SLIPPAGE_BPS, _apply_costs,
)
from strategy import SignalType

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("sbs_backtest")


@dataclass
class SBSResult:
    """Resultado do backtest SBS."
    """
    params: dict
    metrics: BacktestMetrics
    trades: List[TradeResult]
    diag: dict


def simulate_sbs(
    df_ind: pd.DataFrame,
    params: dict = None,
    fee_pct: float = DEFAULT_FEE_PCT,
    spread_bps: float = DEFAULT_SPREAD_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> Tuple[List[TradeResult], int, dict]:
    """
    Simulacao SBS com Stoch RSI + BBWP + Volume + Divergence.

    Returns:
        (trades, atr_filtered, diagnostics)
    """
    global SBS_PARAMS
    if params:
        SBS_PARAMS.update(params)

    reset_cooldown()
    p = SBS_PARAMS

    # Pre-compute Stoch RSI and BBWP for the entire DataFrame
    stoch_k, stoch_d = compute_stoch_rsi(
        df_ind["close"], df_ind["rsi"],
        period=p["stoch_rsi_period"],
        k_smooth=p["stoch_rsi_k_smooth"],
        d_smooth=p["stoch_rsi_d_smooth"],
    )
    bbwp = compute_bbwp(df_ind["bb_width"], lookback=p["bbwp_lookback"])

    # Pre-compute was_squeezed flag
    was_squeezed_series = bbwp.rolling(
        window=p["bbwp_was_squeezed_bars"]
    ).apply(lambda x: (x < p["bbwp_squeeze_threshold"]).any(), raw=False).astype(bool)

    # Add stoch_rsi_k, stoch_rsi_d, bbwp, was_squeezed to df_ind for row access
    df_work = df_ind.copy()
    df_work["stoch_rsi_k"] = stoch_k
    df_work["stoch_rsi_d"] = stoch_d
    df_work["bbwp"] = bbwp
    df_work["was_squeezed"] = was_squeezed_series

    trades: List[TradeResult] = []
    atr_filtered = 0
    i = 0
    n = len(df_work)

    # Limit order costs
    _fee = fee_pct
    _spread = spread_bps
    _slip = slippage_bps

    # Diagnostics
    _diag_modes = {"breakout": 0, "reversal": 0}
    _diag_conviction = {"high": 0, "medium": 0, "low": 0}
    _diag_long = 0
    _diag_short = 0
    _diag_reasons = {}
    _be_count = 0
    _trail_count = 0
    _partial_count = 0

    while i < n:
        row = df_work.iloc[i]

        if i < 2:
            i += 1
            continue

        # Check NaN in critical indicators
        critical = [
            "ema20", "ema50", "rsi", "atr", "atr_percentile",
            "bb_lower", "bb_upper", "bb_middle", "bb_width",
            "volume", "volume_sma20", "close",
        ]
        if any(pd.isna(row.get(c)) for c in critical):
            i += 1
            continue

        # NaN check for pre-computed indicators
        if pd.isna(row.get("stoch_rsi_k", np.nan)) or pd.isna(row.get("bbwp", np.nan)):
            i += 1
            continue

        # ATR filter
        atr_pct = float(row.get("atr_percentile", 0.5))
        if atr_pct < p["atr_pct_min"] or atr_pct > p["atr_pct_max"]:
            atr_filtered += 1
            i += 1
            continue

        prev = df_work.iloc[i - 1]
        stoch_k_val = float(row["stoch_rsi_k"])
        stoch_d_val = float(row["stoch_rsi_d"])
        bbwp_val = float(row["bbwp"])
        vol = float(row["volume"])
        vol_sma20 = float(row["volume_sma20"])
        vol_ratio = vol / vol_sma20 if vol_sma20 > 0 else 0
        was_squeezed = bool(row.get("was_squeezed", False))
        divergence = detect_rsi_divergence(
            df_work["close"], df_work["rsi"], i,
            lookback=p["div_lookback"], min_slope_diff=p["div_min_slope"],
        )

        # Evaluate signal
        result = evaluate_sbs_row(
            row, prev, i,
            stoch_k_val, stoch_d_val, bbwp_val,
            vol_ratio, was_squeezed, divergence,
        )

        if result is None:
            i += 1
            continue

        signal, mode, conviction, trail_mult, sl_mult = result

        # ── Track diagnostics ──
        _diag_modes[mode] = _diag_modes.get(mode, 0) + 1
        _diag_conviction[conviction] = _diag_conviction.get(conviction, 0) + 1
        if signal.type == SignalType.LONG:
            _diag_long += 1
        else:
            _diag_short += 1

        # ── Trade simulation ──
        entry_price = signal.entry_price
        sl = signal.stop_loss
        tp = signal.take_profit
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
        partial_tp_hit = False
        was_trailing_exit = False

        trail_distance = atr * trail_mult
        be_trigger_dist = atr * p["be_trigger_atr"]

        for j in range(i + 1, min(i + p["max_bars"], n)):
            future = df_work.iloc[j]
            f_close = float(future["close"])
            f_low = float(future["low"])
            f_high = float(future["high"])
            bars = j - i

            # Track high water mark
            if is_long:
                highest_favorable = max(highest_favorable, f_high)
            else:
                highest_favorable = min(highest_favorable, f_low)

            # Check TP hit (if trailing not yet activated)
            if not trailing_activated:
                tp_hit = False
                if is_long and f_high >= tp:
                    tp_hit = True
                    if p["trailing_enabled"] and p["partial_tp_pct"] > 0:
                        # Partial TP: close part, activate trailing
                        partial_tp_hit = True
                        trailing_activated = True
                        be_triggered = True
                        current_sl = entry_price  # Move to BE
                        _be_count += 1
                        _partial_count += 1
                        # Continue holding for trailing
                        continue
                    else:
                        exit_price = tp
                        exit_reason = "tp"
                        break
                elif not is_long and f_low <= tp:
                    tp_hit = True
                    if p["trailing_enabled"] and p["partial_tp_pct"] > 0:
                        partial_tp_hit = True
                        trailing_activated = True
                        be_triggered = True
                        current_sl = entry_price
                        _be_count += 1
                        _partial_count += 1
                        continue
                    else:
                        exit_price = tp
                        exit_reason = "tp"
                        break

            # Check SL hit
            sl_hit = False
            if is_long:
                if f_low <= current_sl:
                    sl_hit = True
            else:
                if f_high >= current_sl:
                    sl_hit = True

            # BE trigger
            if not be_triggered and p["trailing_enabled"]:
                fav_dist = abs(highest_favorable - entry_price)
                if fav_dist >= be_trigger_dist:
                    be_triggered = True
                    trailing_activated = True
                    current_sl = entry_price
                    _be_count += 1

            # Trailing stop (ratchet-only)
            if trailing_activated and p["trailing_enabled"]:
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

            if sl_hit:
                exit_price = current_sl
                exit_reason = "trailing_sl" if trailing_activated else "sl"
                was_trailing_exit = trailing_activated
                break

        if exit_price is None:
            last_j = min(i + p["max_bars"], n) - 1
            exit_price = float(df_work.iloc[last_j]["close"])
            exit_reason = "timeout"
            bars = last_j - i

        # Apply costs
        _, adj_exit, cost_pct = _apply_costs(
            entry_price, exit_price, is_long, _fee, _spread, _slip
        )
        if is_long:
            pnl_pct = (adj_exit - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - adj_exit) / entry_price * 100

        # For partial TP trades, reduce PnL to reflect only partial position
        # (the other half continues in trailing, but for simplicity we report
        #  the final exit PnL which includes the trailing portion)
        trades.append(TradeResult(
            entry_ts=row.name,
            exit_ts=df_work.iloc[min(i + bars, n - 1)].name,
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
            partial_tp_filled=partial_tp_hit,
            sl_updates=sl_updates,
        ))

        _diag_reasons[exit_reason] = _diag_reasons.get(exit_reason, 0) + 1

        _from_sbs = __import__("strategy_squeeze_breakout")
        _from_sbs._register_signal(i + bars, was_trailing=was_trailing_exit)

        i += bars + 1

    _diag = {
        "strategy": "sbs_v1",
        "atr_filtered": atr_filtered,
        "modes": _diag_modes,
        "conviction": _diag_conviction,
        "direction": {"long": _diag_long, "short": _diag_short},
        "exit_reasons": _diag_reasons,
        "be_triggered": _be_count,
        "trailing_updates": _trail_count,
        "partial_tp": _partial_count,
    }

    return trades, atr_filtered, _diag


def run_sbs_backtest(
    timeframe: str = "15m",
    days: int = 730,
    params: dict = None,
) -> SBSResult:
    """
    Executa backtest completo SBS.
    """
    logger.info(f"=== SBS Backtest: {timeframe} / {days} dias ===")

    t0 = time.time()

    # 1. Fetch data
    logger.info("Baixando dados...")
    df = fetch_historical_ohlcv("BTC/USDT", timeframe, days)

    # 2. Compute indicators
    logger.info(f"Calculando indicadores para {len(df)} candles...")
    df_ind = compute_indicators(df, timeframe=timeframe)

    # 3. Clean NaN
    df_clean = df_ind.dropna(subset=[
        "ema20", "ema50", "rsi", "atr", "atr_percentile",
        "bb_lower", "bb_upper", "bb_middle", "bb_width",
        "volume", "volume_sma20",
    ]).copy()

    logger.info(f"DataFrame limpo: {len(df_clean)} candles")

    # 4. Simulate
    logger.info("Simulando trades...")
    trades, atr_filtered, diag = simulate_sbs(df_clean, params)

    # 5. Calculate metrics
    metrics = calculate_metrics(trades, df_clean, atr_filtered)

    elapsed = time.time() - t0

    # Print summary
    print(f"\n{'='*60}")
    print(f"SBS v1 Backtest Results — {timeframe} / {days}d ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"Total Trades:     {metrics.total_trades}")
    print(f"  Longs:          {metrics.long_trades}")
    print(f"  Shorts:         {metrics.short_trades}")
    print(f"Win Rate:         {metrics.win_rate:.1f}%")
    print(f"Profit Factor:    {metrics.profit_factor:.2f}")
    print(f"Total PnL:        {metrics.total_pnl_pct:.2f}%")
    print(f"Max Drawdown:     {metrics.max_drawdown_pct:.2f}%")
    print(f"Buy & Hold:       {metrics.buy_hold_pct:.2f}%")
    print(f"vs B&H:           {metrics.total_pnl_pct - metrics.buy_hold_pct:+.2f} pp")
    print(f"Sharpe Ratio:     {metrics.sharpe_ratio:.2f}")
    print(f"Avg Bars Held:    {metrics.avg_bars_held:.1f}")
    print(f"Avg Win:          {metrics.avg_win_pct:.2f}%")
    print(f"Avg Loss:         {metrics.avg_loss_pct:.2f}%")
    print(f"Best Trade:       {metrics.best_trade_pct:.2f}%")
    print(f"Worst Trade:      {metrics.worst_trade_pct:.2f}%")
    print(f"Avg R:R:          {metrics.avg_r_r:.2f}")
    print(f"\nDiagnostics:")
    print(f"  Modes:          {diag.get('modes', {})}")
    print(f"  Conviction:     {diag.get('conviction', {})}")
    print(f"  Exit Reasons:   {diag.get('exit_reasons', {})}")
    print(f"  BE Triggered:   {diag.get('be_triggered', 0)}")
    print(f"  Trail Updates:  {diag.get('trailing_updates', 0)}")
    print(f"  Partial TP:     {diag.get('partial_tp', 0)}")
    print(f"  ATR Filtered:   {atr_filtered}")
    print(f"{'='*60}\n")

    return SBSResult(params=params or SBS_PARAMS, metrics=metrics, trades=trades, diag=diag)


def optimize_params(
    timeframe: str = "15m",
    days: int = 730,
    param_grid: dict = None,
) -> dict:
    """
    Otimiza parâmetros via grid search.
    """
    if param_grid is None:
        param_grid = {
            "bbwp_squeeze_threshold": [5, 10, 15, 20, 25, 30],
            "bbwp_was_squeezed_bars": [4, 8, 12, 16],
            "vol_ratio_min": [1.0, 1.2, 1.3, 1.5, 1.8],
            "sl_atr_mult": [1.2, 1.5, 1.8, 2.0, 2.5],
            "sl_atr_squeeze_mult": [1.0, 1.2, 1.5, 1.8],
            "trail_atr_mult": [1.0, 1.5, 2.0, 2.5],
            "tp_max_atr_mult": [4.0, 5.0, 6.0, 8.0],
            "be_trigger_atr": [0.5, 0.8, 1.0, 1.2],
            "max_bars": [36, 48, 72, 96],
            "cooldown": [2, 4, 6, 8],
            "stoch_rsi_ob": [70, 75, 80],
            "stoch_rsi_os": [20, 25, 30],
        }

    # We'll do a smart iterative search: vary one param at a time
    best_pnl = -999
    best_params = {}
    best_wr = 0
    best_vs_bh = -999

    # Start with defaults
    base_params = dict(SBS_PARAMS)

    # Phase 1: Coarse search - vary key params
    key_params = [
        "bbwp_squeeze_threshold", "vol_ratio_min", "sl_atr_mult",
        "trail_atr_mult", "tp_max_atr_mult", "max_bars", "cooldown",
    ]

    for param_name in key_params:
        if param_name not in param_grid:
            continue
        logger.info(f"\nOptimizing {param_name}...")
        best_for_param = base_params[param_name]
        best_score = -999

        for val in param_grid[param_name]:
            test_params = dict(base_params)
            test_params[param_name] = val

            try:
                result = run_sbs_backtest(timeframe, days, test_params)
                m = result.metrics
                # Score: prioritize win_rate > 60%, then vs B&H
                score = 0
                if m.win_rate >= 60:
                    score += 100
                score += (m.total_pnl_pct - m.buy_hold_pct) * 0.5
                score += m.profit_factor * 10

                if score > best_score and m.total_trades >= 10:
                    best_score = score
                    best_for_param = val
                    logger.info(
                        f"  {param_name}={val}: WR={m.win_rate:.1f}% "
                        f"PnL={m.total_pnl_pct:.1f}% vsB&H={m.total_pnl_pct - m.buy_hold_pct:+.1f}pp "
                        f"score={score:.1f}"
                    )
            except Exception as e:
                logger.warning(f"  {param_name}={val}: ERROR {e}")

        if best_for_param != base_params[param_name]:
            logger.info(f"  >> Best {param_name} = {best_for_param}")
            base_params[param_name] = best_for_param

    # Final result with best params
    logger.info(f"\n{'='*60}")
    logger.info("Running FINAL backtest with best parameters...")
    final = run_sbs_backtest(timeframe, days, base_params)
    m = final.metrics

    return {
        "params": base_params,
        "metrics": {
            "total_trades": m.total_trades,
            "win_rate": m.win_rate,
            "profit_factor": m.profit_factor,
            "total_pnl_pct": m.total_pnl_pct,
            "buy_hold_pct": m.buy_hold_pct,
            "vs_bh_pp": m.total_pnl_pct - m.buy_hold_pct,
            "max_drawdown_pct": m.max_drawdown_pct,
            "sharpe_ratio": m.sharpe_ratio,
        },
    }


if __name__ == "__main__":
    tf = sys.argv[1] if len(sys.argv) > 1 else "15m"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 730
    mode = sys.argv[3] if len(sys.argv) > 3 else "run"  # "run" or "optimize"

    if mode == "optimize":
        result = optimize_params(tf, days)
        print(f"\nBest params for {tf}/{days}d:")
        for k, v in result["params"].items():
            print(f"  {k}: {v}")
        print(f"\nBest metrics:")
        for k, v in result["metrics"].items():
            print(f"  {k}: {v}")
    else:
        run_sbs_backtest(tf, days)
