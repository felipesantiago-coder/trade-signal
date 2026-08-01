"""
test_regime_switching.py
----------------------
Backtest comparativo: Regime-Switching vs Estrategia Unica (CTEV baseline).

Compara:
  1. BASELINE: CTEV trend-following atual (rejeita ranging/volatile)
  2. REGIME-SWITCHING: Usa estrategia adequada para cada regime

Roda em BTC/USDT 1h, 730 dias, com custos realistas.
"""
from __future__ import annotations
import sys
import os
import time
import json
import logging

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

from indicators import compute_indicators
from strategy import evaluate_long, evaluate_short, SignalType
from strategy_profiles import get_profile
from backtest import (
    fetch_historical_ohlcv, _apply_costs,
    DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS,
    calculate_metrics, BacktestMetrics,
)
from regime_engine import (
    classify_regimes_v2, get_regime_params,
    get_regime_summary, LONG_REGIMES, SHORT_REGIMES, NEUTRAL_REGIMES,
)
from strategy_regime import (
    evaluate_mean_reversion_long, evaluate_mean_reversion_short,
    evaluate_breakout_long, evaluate_breakout_short,
    _evaluate_trend_long_adapted, _evaluate_trend_short_adapted,
)


def simulate_baseline(
    df_ind: pd.DataFrame, profile, max_bars: int = 72,
    fee_pct=DEFAULT_FEE_PCT, spread_bps=DEFAULT_SPREAD_BPS, slippage_bps=DEFAULT_SLIPPAGE_BPS,
) -> list:
    """Simulacao baseline: CTEV trend-following (rejeita ranging/volatile)."""
    trades = []
    n = len(df_ind)
    i = 0

    while i < n:
        row = df_ind.iloc[i]
        critical = ["ema20", "ema50", "ema200", "rsi", "atr", "atr_percentile",
                    "adx", "plus_di", "minus_di", "regime"]
        if any(pd.isna(row.get(c)) for c in critical):
            i += 1
            continue

        regime = str(row.get("regime", ""))
        if regime in ("ranging", "volatile"):
            i += 1
            continue

        atr_pct = float(row.get("atr_percentile", 0.5))
        if atr_pct < profile.atr_pct_min or atr_pct > profile.atr_pct_max:
            i += 1
            continue

        signal = evaluate_long(row, profile=profile)
        if signal is None:
            signal = evaluate_short(row, profile=profile)
        if signal is None:
            i += 1
            continue

        # Simulate trade
        entry_price = signal.entry_price
        sl, tp = signal.stop_loss, signal.take_profit
        is_long = signal.type == SignalType.LONG
        exit_price, exit_reason, bars = None, None, 0

        for j in range(i + 1, min(i + max_bars, n)):
            f = df_ind.iloc[j]
            bars = j - i
            if is_long:
                if float(f["low"]) <= sl:
                    exit_price, exit_reason = sl, "sl"; break
                if float(f["high"]) >= tp:
                    exit_price, exit_reason = tp, "tp"; break
            else:
                if float(f["high"]) >= sl:
                    exit_price, exit_reason = sl, "sl"; break
                if float(f["low"]) <= tp:
                    exit_price, exit_reason = tp, "tp"; break

        if exit_price is None:
            last_j = min(i + max_bars, n) - 1
            exit_price = float(df_ind.iloc[last_j]["close"])
            exit_reason = "timeout"
            bars = last_j - i

        _, adj_exit, cost_pct = _apply_costs(entry_price, exit_price, is_long,
                                               fee_pct, spread_bps, slippage_bps)
        pnl_pct = ((adj_exit - entry_price) / entry_price * 100) if is_long else \
                  ((entry_price - adj_exit) / entry_price * 100)

        trades.append({
            "entry_ts": str(row.name), "type": signal.type.value,
            "entry_price": entry_price, "exit_price": exit_price,
            "pnl_pct": round(pnl_pct, 4), "bars_held": bars,
            "exit_reason": exit_reason, "regime": regime,
            "pullback_type": signal.pullback_type,
            "sl_mult": round(sl / (entry_price - signal.stop_loss + 0.01), 2) if is_long else round(sl / (signal.stop_loss - entry_price + 0.01), 2),
        })
        i += bars + 1

    return trades


def simulate_regime_switching(
    df_ind: pd.DataFrame, profile, hysteresis_bars: int = 3,
    max_bars: int = 72,
    fee_pct=DEFAULT_FEE_PCT, spread_bps=DEFAULT_SPREAD_BPS, slippage_bps=DEFAULT_SLIPPAGE_BPS,
) -> list:
    """Simulacao regime-switching: aplica estrategia adequada por regime."""
    trades = []
    n = len(df_ind)
    i = 0

    while i < n:
        row = df_ind.iloc[i]
        critical = ["ema20", "ema50", "ema200", "rsi", "atr", "atr_percentile",
                    "adx", "plus_di", "minus_di", "regime_v2", "regime_confidence"]
        if any(pd.isna(row.get(c)) for c in critical):
            i += 1
            continue

        regime = str(row.get("regime_v2", ""))
        confidence = float(row.get("regime_confidence", 0.5))
        strategy_type = str(row.get("regime_strategy", "neutral"))

        # Get regime-specific params
        params = get_regime_params(regime, confidence, base_profile=profile)

        # Skip neutral regimes
        if params["strategy_type"] == "neutral":
            i += 1
            continue

        # Skip low confidence
        if confidence < params["min_confidence"]:
            i += 1
            continue

        # Evaluate signal based on strategy type
        signal = None
        st = params["strategy_type"]

        if st == "trend_follow":
            # CRITICAL: Use ORIGINAL evaluate functions (they have ADX filter!)
            # The adapted versions don't have ADX checks and produce low-quality signals
            signal = evaluate_long(row, profile=profile)
            if signal is None:
                signal = evaluate_short(row, profile=profile)

        elif st == "mean_reversion":
            atr_pct = float(row.get("atr_percentile", 0.5))
            if atr_pct < 0.10 or atr_pct > 0.85:
                i += 1
                continue
            if params["allow_long"]:
                signal = evaluate_mean_reversion_long(row, params, base_profile=profile)
            if signal is None and params["allow_short"]:
                signal = evaluate_mean_reversion_short(row, params, base_profile=profile)

        elif st == "breakout":
            atr_pct = float(row.get("atr_percentile", 0.5))
            if atr_pct < 0.40:
                i += 1
                continue
            if params["allow_long"]:
                signal = evaluate_breakout_long(row, params, base_profile=profile)
            if signal is None and params["allow_short"]:
                signal = evaluate_breakout_short(row, params, base_profile=profile)

        if signal is None:
            i += 1
            continue

        # Simulate trade (same logic as baseline)
        entry_price = signal.entry_price
        sl, tp = signal.stop_loss, signal.take_profit
        is_long = signal.type == SignalType.LONG
        exit_price, exit_reason, bars = None, None, 0

        for j in range(i + 1, min(i + max_bars, n)):
            f = df_ind.iloc[j]
            bars = j - i
            if is_long:
                if float(f["low"]) <= sl:
                    exit_price, exit_reason = sl, "sl"; break
                if float(f["high"]) >= tp:
                    exit_price, exit_reason = tp, "tp"; break
            else:
                if float(f["high"]) >= sl:
                    exit_price, exit_reason = sl, "sl"; break
                if float(f["low"]) <= tp:
                    exit_price, exit_reason = tp, "tp"; break

        if exit_price is None:
            last_j = min(i + max_bars, n) - 1
            exit_price = float(df_ind.iloc[last_j]["close"])
            exit_reason = "timeout"
            bars = last_j - i

        _, adj_exit, cost_pct = _apply_costs(entry_price, exit_price, is_long,
                                               fee_pct, spread_bps, slippage_bps)
        pnl_pct = ((adj_exit - entry_price) / entry_price * 100) if is_long else \
                  ((entry_price - adj_exit) / entry_price * 100)

        trades.append({
            "entry_ts": str(row.name), "type": signal.type.value,
            "entry_price": entry_price, "exit_price": exit_price,
            "pnl_pct": round(pnl_pct, 4), "bars_held": bars,
            "exit_reason": exit_reason, "regime": regime,
            "strategy_type": st,
            "pullback_type": signal.pullback_type,
            "confidence": round(confidence, 3),
        })
        i += bars + 1

    return trades


def compute_stats(trades: list) -> dict:
    """Compute statistics from trade list."""
    if not trades:
        return {"total": 0, "wins": 0, "losses": 0, "wr": 0, "pnl": 0,
                "avg_win": 0, "avg_loss": 0, "pf": 0, "avg_bars": 0}

    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    total_pnl = sum(t["pnl_pct"] for t in trades)
    gross_profit = sum(t["pnl_pct"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl_pct"] for t in losses)) if losses else 0.001

    # Regime breakdown
    by_regime = {}
    by_strategy = {}
    for t in trades:
        r = t.get("regime", "unknown")
        s = t.get("strategy_type", t.get("pullback_type", "unknown"))
        by_regime[r] = by_regime.get(r, 0) + 1
        by_strategy[s] = by_strategy.get(s, 0) + 1

    return {
        "total": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "wr": 100 * len(wins) / len(trades) if trades else 0,
        "pnl": round(total_pnl, 2),
        "avg_win": round(np.mean([t["pnl_pct"] for t in wins]), 3) if wins else 0,
        "avg_loss": round(np.mean([t["pnl_pct"] for t in losses]), 3) if losses else 0,
        "pf": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
        "avg_bars": round(np.mean([t["bars_held"] for t in trades]), 1) if trades else 0,
        "by_regime": dict(sorted(by_regime.items(), key=lambda x: -x[1])),
        "by_strategy": dict(sorted(by_strategy.items(), key=lambda x: -x[1])),
    }


def main():
    symbol = "BTC/USDT"
    timeframe = "1h"
    days = 730

    print("=" * 70)
    print("  REGIME-SWITCHING vs BASELINE — BTC/USDT 1h 730d")
    print("=" * 70)

    # 1. Download data
    print("\n[1/5] Baixando dados...")
    t0 = time.time()
    df = fetch_historical_ohlcv(symbol, timeframe, days)
    print(f"  {len(df):,} candles baixados em {time.time()-t0:.1f}s")

    # 2. Compute indicators
    print("\n[2/5] Calculando indicadores...")
    t0 = time.time()
    df_ind = compute_indicators(df, timeframe=timeframe)
    print(f"  Indicadores calculados em {time.time()-t0:.1f}s")

    # 3. Classify regimes v2
    print("\n[3/5] Classificando regimes v2 (hysteresis=3)...")
    t0 = time.time()
    df_ind = classify_regimes_v2(df_ind, hysteresis_bars=3)
    print(f"  Regimes classificados em {time.time()-t0:.1f}s")

    # Print regime distribution
    print(f"\n{get_regime_summary(df_ind)}")

    # 4. Clean NaN
    critical_v1 = ["ema20", "ema50", "ema200", "rsi", "atr", "atr_percentile",
                   "adx", "plus_di", "minus_di", "regime"]
    critical_v2 = critical_v1 + ["regime_v2", "regime_confidence", "regime_strategy"]
    df_clean_v1 = df_ind.dropna(subset=critical_v1).copy()
    df_clean_v2 = df_ind.dropna(subset=critical_v2).copy()
    print(f"\n  V1 clean: {len(df_clean_v1):,} bars | V2 clean: {len(df_clean_v2):,} bars")

    # 5. Get profile
    profile = get_profile(timeframe)
    print(f"  Profile: {profile.summary()}")

    # 6. Buy & Hold
    first_close = float(df_clean_v1.iloc[0]["close"])
    last_close = float(df_clean_v1.iloc[-1]["close"])
    bh_pct = (last_close - first_close) / first_close * 100
    print(f"\n  Buy & Hold: {first_close:.2f} -> {last_close:.2f} = {bh_pct:+.2f}%")

    # 7. Run BASELINE backtest
    print("\n[4/5] Rodando BASELINE (CTEV trend-following)...")
    t0 = time.time()
    baseline_trades = simulate_baseline(df_clean_v1, profile, max_bars=profile.max_bars_held)
    baseline_stats = compute_stats(baseline_trades)
    print(f"  {len(baseline_trades)} trades em {time.time()-t0:.1f}s")

    # 8. Run REGIME-SWITCHING backtest
    print("\n[5/5] Rodando REGIME-SWITCHING...")
    t0 = time.time()
    rs_trades = simulate_regime_switching(
        df_clean_v2, profile, hysteresis_bars=3,
        max_bars=profile.max_bars_held,
    )
    rs_stats = compute_stats(rs_trades)
    print(f"  {len(rs_trades)} trades em {time.time()-t0:.1f}s")

    # ========================================
    # RESULTS COMPARISON
    # ========================================
    print("\n" + "=" * 70)
    print("  RESULTADOS COMPARATIVOS")
    print("=" * 70)

    print(f"\n{'Metrica':<25} {'BASELINE':>12} {'REGIME-SW':>12} {'Buy&Hold':>12}")
    print("-" * 65)
    print(f"{'Total Trades':<25} {baseline_stats['total']:>12} {rs_stats['total']:>12} {'N/A':>12}")
    print(f"{'Win Rate':<25} {baseline_stats['wr']:>11.1f}% {rs_stats['wr']:>11.1f}% {bh_pct:>11.2f}%")
    print(f"{'Profit Factor':<25} {baseline_stats['pf']:>12.2f} {rs_stats['pf']:>12.2f} {'N/A':>12}")
    print(f"{'PnL Total':<25} {baseline_stats['pnl']:>11.2f}% {rs_stats['pnl']:>11.2f}% {bh_pct:>11.2f}%")
    print(f"{'Avg Win':<25} {baseline_stats['avg_win']:>11.3f}% {rs_stats['avg_win']:>11.3f}% {'N/A':>12}")
    print(f"{'Avg Loss':<25} {baseline_stats['avg_loss']:>11.3f}% {rs_stats['avg_loss']:>11.3f}% {'N/A':>12}")
    print(f"{'Avg Bars Held':<25} {baseline_stats['avg_bars']:>12.1f} {rs_stats['avg_bars']:>12.1f} {'N/A':>12}")

    # Delta
    pnl_delta = rs_stats['pnl'] - baseline_stats['pnl']
    print(f"\n{'PnL Delta (RS - Base)':<25} {pnl_delta:>+11.2f}pp")

    # Regime breakdown for regime-switching
    print(f"\n--- Regime-Switching: Trades por Regime ---")
    for regime, count in rs_stats.get("by_regime", {}).items():
        regime_trades = [t for t in rs_trades if t.get("regime") == regime]
        regime_pnl = sum(t["pnl_pct"] for t in regime_trades)
        regime_wins = sum(1 for t in regime_trades if t["pnl_pct"] > 0)
        regime_wr = 100 * regime_wins / len(regime_trades) if regime_trades else 0
        print(f"  {regime:20s}: {count:4d} trades | PnL {regime_pnl:+8.2f}% | WR {regime_wr:.1f}%")

    print(f"\n--- Regime-Switching: Trades por Estrategia ---")
    for strat, count in rs_stats.get("by_strategy", {}).items():
        strat_trades = [t for t in rs_trades if t.get("strategy_type", t.get("pullback_type")) == strat]
        strat_pnl = sum(t["pnl_pct"] for t in strat_trades)
        strat_wins = sum(1 for t in strat_trades if t["pnl_pct"] > 0)
        strat_wr = 100 * strat_wins / len(strat_trades) if strat_trades else 0
        print(f"  {strat:25s}: {count:4d} trades | PnL {strat_pnl:+8.2f}% | WR {strat_wr:.1f}%")

    # Save results
    results = {
        "symbol": symbol, "timeframe": timeframe, "days": days,
        "buy_hold_pct": round(bh_pct, 2),
        "baseline": baseline_stats,
        "regime_switching": rs_stats,
        "pnl_delta_pp": round(pnl_delta, 2),
        "regime_switching_beats_baseline": pnl_delta > 0,
        "regime_switching_beats_bh": rs_stats["pnl"] > bh_pct,
        "baseline_beats_bh": baseline_stats["pnl"] > bh_pct,
    }

    out_path = "/home/z/my-project/download/regime_switching_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResultados salvos em: {out_path}")

    # Final verdict
    print("\n" + "=" * 70)
    if pnl_delta > 0:
        print(f"  RESULTADO: Regime-Switching SUPEROU o Baseline em {pnl_delta:+.2f}pp")
    else:
        print(f"  RESULTADO: Regime-Switching NAO superou o Baseline ({pnl_delta:+.2f}pp)")
        print("  Possiveis acoes: ajustar thresholds de mean-reversion/breakout")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
