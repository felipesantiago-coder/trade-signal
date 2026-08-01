"""
test_dual_optimization.py
----------------------
Testa otimizacoes para os dois paths identificados:

PATH 1 (LONG WR): O problema e que evaluate_long usa o OLD regime column.
  Quando old regime = "transition" (allow_transition=True), ADX>=30 e bypassado.
  10/14 LONG losers tem ADX < 30.
  FIX: Adicionar ADX floor no _simulate_regime_switching para WEAK_UPTREND.

PATH 2 (SHORT freq): O RSI [55,75] e muito restritivo em downtrends.
  Em downtrend, RSI naturalmente fica < 50. Pullbacks mostram RSI 40-60.
  FIX: Alargar RSI short range para downtrend regimes.

Testes:
  A) Baseline (atual)
  B) ADX floor 25 para WEAK_UPTREND LONGs
  C) RSI short [45,75] para downtrends
  D) RSI short [40,78] para downtrends
  E) Combinação B+C
  F) Combinacao B+D
  G) ADX floor 28 WEAK_UP
  H) G+D combined
  I) RSI short [42,75] + RANGING [55,80]
  J) ADX 25 + RSI [42,75] + RANGING [55,80]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import ccxt
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from copy import deepcopy

from indicators import compute_indicators
from regime_engine import (
    classify_regimes_v2, get_regime_params, REGIME_STRATEGY,
)
from strategy import evaluate_long, evaluate_short, SignalType
from strategy_regime import (
    evaluate_mean_reversion_long, evaluate_mean_reversion_short,
)
from strategy_profiles import get_profile
from backtest import (
    DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS,
    _apply_costs,
)

logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
logger = logging.getLogger("dual_opt")
logger.setLevel(logging.INFO)


def download_btc_730d():
    exchange = ccxt.binance({"enableRateLimit": True})
    since_ms = int((datetime.now(timezone.utc).timestamp() - 730 * 86400) * 1000)
    all_ohlcv = []
    last_ts = 0
    max_iter = 730 * 24 + 100
    iteration = 0
    tf_ms = 3600 * 1000

    while iteration < max_iter:
        iteration += 1
        batch = exchange.fetch_ohlcv("BTC/USDT", "1h", since=since_ms, limit=1000)
        if not batch:
            break
        batch_ts = batch[-1][0]
        if batch_ts <= last_ts:
            break
        last_ts = batch_ts
        all_ohlcv.extend(batch)
        since_ms = batch_ts + tf_ms

    df = pd.DataFrame(all_ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").astype(float)
    df = df[~df.index.duplicated(keep="first")]
    return df


def simulate_variant(
    df_ind, profile,
    adx_floor_weak_up=0,      # PATH 1: ADX floor for WEAK_UPTREND LONGs (0=disabled)
    rsi_short_dt=None,        # PATH 2: (min, max) for downtrend shorts (None=use default)
    rsi_short_dt_wide=None,   # PATH 2: even wider RSI range
    fee_pct=DEFAULT_FEE_PCT,
    spread_bps=DEFAULT_SPREAD_BPS,
    slippage_bps=DEFAULT_SLIPPAGE_BPS,
):

    trades = []
    i = 0
    n = len(df_ind)
    _max_bars = profile.max_bars_held if profile else 72

    while i < n:
        row = df_ind.iloc[i]

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
        params = get_regime_params(regime_v2, confidence, base_profile=profile)
        st = params["strategy_type"]

        if st == "neutral" or confidence < params["min_confidence"]:
            i += 1
            continue

        signal = None
        atr_pct = float(row.get("atr_percentile", 0.5))

        if st == "trend_follow":
            # PATH 1 FIX: ADX floor for WEAK_UPTREND LONGs
            if adx_floor_weak_up > 0 and regime_v2 == "WEAK_UPTREND":
                adx_val = float(row.get("adx", 0))
                if adx_val < adx_floor_weak_up:
                    i += 1
                    continue

            if atr_pct < 0.10 or atr_pct > 0.90:
                i += 1
                continue

            signal = evaluate_long(row, profile=profile)
            if signal is None:
                # PATH 2 FIX: Wider RSI for downtrend shorts
                if rsi_short_dt and regime_v2 in ("STRONG_DOWNTREND", "WEAK_DOWNTREND"):
                    rsi_val = float(row.get("rsi", 0))
                    if rsi_short_dt[0] <= rsi_val <= rsi_short_dt[1]:
                        # Temporarily override RSI range in params and use adapted short
                        mod_params = dict(params)
                        mod_params["rsi_short_range"] = rsi_short_dt
                        from strategy_regime import _evaluate_trend_short_adapted
                        signal = _evaluate_trend_short_adapted(row, mod_params, base_profile=profile)
                    else:
                        signal = evaluate_short(row, profile=profile)
                else:
                    signal = evaluate_short(row, profile=profile)

        elif st == "mean_reversion":
            if atr_pct < 0.10 or atr_pct > 0.85:
                i += 1
                continue
            # For RANGING shorts, optionally widen RSI range
            if params["allow_short"]:
                mr_params = dict(params)
                if rsi_short_dt_wide and regime_v2 == "RANGING":
                    mr_params["rsi_short_range"] = rsi_short_dt_wide
                signal = evaluate_mean_reversion_short(row, mr_params, base_profile=profile)
            if signal is None and params["allow_long"]:
                signal = evaluate_mean_reversion_long(row, params, base_profile=profile)

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

        _, adj_exit, cost_pct = _apply_costs(
            entry_price, exit_price, is_long, fee_pct, spread_bps, slippage_bps,
        )
        if is_long:
            pnl_pct = (adj_exit - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - adj_exit) / entry_price * 100

        trades.append({
            "type": "LONG" if is_long else "SHORT",
            "pnl_pct": round(pnl_pct, 4),
            "exit_reason": exit_reason,
            "regime_v2": regime_v2,
            "adx": round(float(row.get("adx", 0)), 1),
        })

        i += bars + 1

    return trades


def calc_metrics(trades):
    if not trades:
        return {"total": 0, "longs": 0, "shorts": 0, "wr": 0, "pnl": 0, "long_wr": 0, "short_wr": 0}

    longs = [t for t in trades if t["type"] == "LONG"]
    shorts = [t for t in trades if t["type"] == "SHORT"]
    wins = sum(1 for t in trades if t["pnl_pct"] > 0)
    long_wins = sum(1 for t in longs if t["pnl_pct"] > 0)
    short_wins = sum(1 for t in shorts if t["pnl_pct"] > 0)
    total_pnl = sum(t["pnl_pct"] for t in trades)
    long_pnl = sum(t["pnl_pct"] for t in longs)
    short_pnl = sum(t["pnl_pct"] for t in shorts)

    # Max DD
    running = 0
    peak = 0
    max_dd = 0
    for t in trades:
        running += t["pnl_pct"]
        peak = max(peak, running)
        dd = peak - running
        max_dd = max(max_dd, dd)

    return {
        "total": len(trades),
        "longs": len(longs),
        "shorts": len(shorts),
        "wr": round(100 * wins / len(trades), 1),
        "pnl": round(total_pnl, 2),
        "max_dd": round(max_dd, 2),
        "long_wr": round(100 * long_wins / max(len(longs), 1), 1),
        "short_wr": round(100 * short_wins / max(len(shorts), 1), 1),
        "long_pnl": round(long_pnl, 2),
        "short_pnl": round(short_pnl, 2),
    }


def main():
    logger.info("Baixando dados...")
    df = download_btc_730d()
    logger.info(f"Dados: {len(df)} candles")

    logger.info("Computando indicadores...")
    df_ind = compute_indicators(df, "1h")
    df_ind = classify_regimes_v2(df_ind, hysteresis_bars=3)

    profile = get_profile("STANDARD")

    variants = {
        "A) Baseline (atual)": {},
        "B) ADX floor 25 WEAK_UP LONG": {"adx_floor_weak_up": 25},
        "C) RSI short [45,75] downtrend": {"rsi_short_dt": (45, 75)},
        "D) RSI short [40,78] downtrend": {"rsi_short_dt": (40, 78)},
        "E) B+C combined": {"adx_floor_weak_up": 25, "rsi_short_dt": (45, 75)},
        "F) B+D combined": {"adx_floor_weak_up": 25, "rsi_short_dt": (40, 78)},
        "G) ADX floor 28 WEAK_UP": {"adx_floor_weak_up": 28},
        "H) G+D combined": {"adx_floor_weak_up": 28, "rsi_short_dt": (40, 78)},
        "I) RSI short [42,75] + RANGING [55,80]": {
            "rsi_short_dt": (42, 75),
            "rsi_short_dt_wide": (55, 80),
        },
        "J) ADX 25 + RSI [42,75] + RANGING [55,80]": {
            "adx_floor_weak_up": 25,
            "rsi_short_dt": (42, 75),
            "rsi_short_dt_wide": (55, 80),
        },
    }

    print("\n" + "="*100)
    print(f"{'Variant':45s} {'Trades':>7s} {'L':>3s} {'S':>3s} {'WR%':>6s} {'PnL%':>7s} {'DD%':>6s} {'L_WR%':>6s} {'S_WR%':>6s} {'L_PnL':>7s} {'S_PnL':>7s}")
    print("-"*100)

    results = {}
    for name, kwargs in variants.items():
        trades = simulate_variant(df_ind, profile, **kwargs)
        m = calc_metrics(trades)
        results[name] = {**m, "trades_detail": trades}
        print(
            f"{name:45s} {m['total']:7d} {m['longs']:3d} {m['shorts']:3d} "
            f"{m['wr']:6.1f} {m['pnl']:7.2f} {m['max_dd']:6.2f} "
            f"{m['long_wr']:6.1f} {m['short_wr']:6.1f} "
            f"{m['long_pnl']:7.2f} {m['short_pnl']:7.2f}"
        )

    # Detailed comparison: Best variant vs baseline
    print("\n" + "="*100)
    print("MELHOR VARIANTE vs BASELINE (detalhamento)")
    baseline = results["A) Baseline (atual)"]
    best_name = max(
        [k for k in results if k != "A) Baseline (atual)"],
        key=lambda k: results[k]["pnl"],
    )
    best = results[best_name]
    print(f"  Baseline: {baseline['total']} trades, WR={baseline['wr']}%, PnL={baseline['pnl']}%")
    print(f"  Best:     {best_name}")
    print(f"            {best['total']} trades, WR={best['wr']}%, PnL={best['pnl']}%")
    print(f"  Delta:    PnL {best['pnl']-baseline['pnl']:+.2f}pp, Trades {best['total']-baseline['total']:+d}, DD {best['max_dd']-baseline['max_dd']:+.2f}pp")

    # Per-regime breakdown for best variant
    print(f"\n  Per-regime breakdown ({best_name}):")
    for t in best["trades_detail"]:
        win = "WIN " if t["pnl_pct"] > 0 else "LOSS"
        print(f"    {t['type']:5s} {t['regime_v2']:20s} ADX={t['adx']:5.1f} PnL={t['pnl_pct']:+6.2f}% {win} exit={t['exit_reason']}")

    # Save
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "download", "dual_optimization_results.json"
    )
    save_data = {k: {kk: vv for kk, vv in v.items() if kk != "trades_detail"} for k, v in results.items()}
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResultados salvos em: {out_path}")


if __name__ == "__main__":
    main()
