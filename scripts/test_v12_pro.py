"""
test_v12_pro.py
--------------
Backtest CTEV v19.1 Multi-Strategy em todos os periodos.
6 estrategias: CTEV Trend, Squeeze Breakout, Range Trader, RSI Extremes, RSI Reversal.
Metas: 90d/180d >= 30%, 365d >= 70%, 730d >= 120%.
"""

import sys
import os
import logging
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from backtest import run_backtest

# v18.2 baseline (pre-multi-strategy)
V18_BASELINE = {
    30:  {"trades": 30,  "wr": 23.3, "pnl": -10.97},
    90:  {"trades": 104, "wr": 35.6, "pnl": 5.07},
    180: {"trades": 215, "wr": 34.4, "pnl": 1.07},
    365: {"trades": 419, "wr": 39.4, "pnl": 89.24},
    730: {"trades": 815, "wr": 36.2, "pnl": 53.27},
}

# v19.1 metas
TARGETS = {
    30:  30.0,
    90:  30.0,
    180: 30.0,
    365: 70.0,
    730: 120.0,
}

PERIODS = [30, 90, 180, 365, 730]


def main():
    print()
    print("=" * 80)
    print("  CTEV v19.1 Multi-Strategy — Backtest Multi-Periodo")
    print("  BTC/USDT | 1h | 6 Estrategias Adaptativas")
    print("  CTEV Trend + Squeeze + Range + RSI Extremes + RSI Reversal")
    print("  Position sizing por tipo de entrada")
    print("=" * 80)
    print()

    results = {}
    trade_data = {}
    t0 = time.time()

    for days in PERIODS:
        print(f"\n{'─' * 60}")
        print(f"  Periodo: {days} dias")
        print(f"{'─' * 60}")

        try:
            m, trades = run_backtest(
                symbol="BTC/USDT",
                timeframe="1h",
                days=days,
                advanced=True,
            )
            results[days] = m
            trade_data[days] = trades

            print(f"  Trades:      {m.total_trades:4d}  (L:{m.long_trades} S:{m.short_trades})")
            print(f"  Win Rate:    {m.win_rate:6.1f}%")
            print(f"  Profit Factor:{m.profit_factor:6.2f}")
            print(f"  PnL Total:   {m.total_pnl_pct:+7.2f}%")
            print(f"  Buy & Hold:  {m.buy_hold_pct:+7.2f}%")
            print(f"  Max DD:      {m.max_drawdown_pct:6.2f}%")
            print(f"  Sharpe:      {m.sharpe_ratio:6.2f}")
            print(f"  Avg R:R:     {m.avg_r_r:6.2f}")
            print(f"  Avg Bars:    {m.avg_bars_held:6.1f}")
            print(f"  Partial TP:  {m.partial_tp_count}")

            # Entry type breakdown
            if trades:
                types = {}
                for t in trades:
                    et = getattr(t, 'entry_type', 'unknown')
                    types[et] = types.get(et, 0) + 1
                print(f"  Strategies:  {types}")

            # Exit reasons
            if trades:
                reasons = {}
                for t in trades:
                    r = t.exit_reason
                    reasons[r] = reasons.get(r, 0) + 1
                print(f"  Exits:       {reasons}")

            # Target check
            target = TARGETS[days]
            passed = m.total_pnl_pct >= target
            status = "PASS" if passed else "FAIL"
            print(f"  META {target:.0f}%:  {status}")

        except Exception as e:
            print(f"  ERRO: {e}")
            results[days] = None
            trade_data[days] = []

    elapsed = time.time() - t0

    # Comparison table
    print(f"\n\n{'=' * 80}")
    print(f"  COMPARACAO: v19.1 Multi-Strategy vs v18.2 Baseline")
    print(f"  Tempo de execucao: {elapsed:.0f}s")
    print(f"{'=' * 80}")
    print()
    print(f"  {'Periodo':<10} {'Target':>7} {'v18 PnL':>9} {'v19 PnL':>9} {'v18 T':>6} {'v19 T':>6} {'v19 WR':>7} {'Status':>6}")
    print(f"  {'-' * 78}")

    pass_count = 0
    for days in PERIODS:
        b = V18_BASELINE[days]
        r = results.get(days)
        target = TARGETS[days]
        if r is None:
            print(f"  {days:<10d} {target:>6.0f}% {b['pnl']:>+8.2f}% {'ERROR':>9} {b['trades']:>6d} {'N/A':>6} {'N/A':>7} {'ERROR':>6}")
            continue
        delta_pnl = r.total_pnl_pct - b["pnl"]
        passed = r.total_pnl_pct >= target
        if passed:
            pass_count += 1
        print(
            f"  {days:<10d} {target:>6.0f}% {b['pnl']:>+8.2f}% {r.total_pnl_pct:>+8.2f}% "
            f"{b['trades']:>6d} {r.total_trades:>6d} {r.win_rate:>6.1f}% "
            f"{'PASS':>6}"
        )

    # Summary
    print(f"\n{'=' * 80}")
    print(f"  AVALIACAO v19.1 Multi-Strategy")
    print(f"{'=' * 80}")
    print(f"  Targets PASSED: {pass_count}/5")
    print()
    for days in PERIODS:
        r = results.get(days)
        if r is None:
            continue
        trades = trade_data[days]
        types = {}
        for t in trades:
            et = getattr(t, 'entry_type', 'unknown')
            types[et] = types.get(et, 0) + 1
        print(f"  {days}d: {r.total_pnl_pct:+7.2f}% | WR {r.win_rate:.1f}% | DD {r.max_drawdown_pct:.1f}% | {r.total_trades} trades | {types}")
    print()


if __name__ == "__main__":
    main()
