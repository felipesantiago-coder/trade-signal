"""
test_v13_multi.py
--------------
Backtest CTEV v13.0 Active Trader Multi-Strategy em todos os periodos.
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

PERIODS = [30, 90, 180, 365, 730]

def main():
    print()
    print("=" * 80)
    print("  CTEV v13.0 Active Trader Multi-Strategy")
    print("  BTC/USDT | 1h | CTEV Pullback + Momentum + Mean-Reversion")
    print("=" * 80)
    print()

    results = {}
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

            print(f"  Trades:      {m.total_trades:4d}  (L:{m.long_trades} S:{m.short_trades})")
            print(f"  Trades/sem:  {m.total_trades / (days/7):.1f}")
            print(f"  Win Rate:    {m.win_rate:6.1f}%")
            print(f"  Profit Factor:{m.profit_factor:6.2f}")
            print(f"  PnL Total:   {m.total_pnl_pct:+7.2f}%")
            print(f"  Buy & Hold:  {m.buy_hold_pct:+7.2f}%")
            print(f"  Max DD:      {m.max_drawdown_pct:6.2f}%")
            print(f"  Sharpe:      {m.sharpe_ratio:6.2f}")
            print(f"  Avg R:R:     {m.avg_r_r:6.2f}")
            print(f"  Avg Bars:    {m.avg_bars_held:6.1f}")
            print(f"  Trades/sem:  {m.total_trades / (days/7):.1f}")

            if trades:
                reasons = {}
                for t in trades:
                    r = t.exit_reason
                    reasons[r] = reasons.get(r, 0) + 1
                print(f"  Exits:       {reasons}")

        except Exception as e:
            print(f"  ERRO: {e}")
            import traceback; traceback.print_exc()
            results[days] = None

    elapsed = time.time() - t0

    # Summary
    print(f"\n\n{'=' * 80}")
    print(f"  RESUMO v13.0 — Tempo: {elapsed:.0f}s")
    print(f"{'=' * 80}")
    print()
    print(f"  {'Periodo':<10} {'Trades':>7} {'/sem':>6} {'WR':>7} {'PF':>6} {'PnL':>9} {'DD':>7} {'Sharpe':>7}")
    print(f"  {'-' * 62}")

    for days in PERIODS:
        r = results.get(days)
        if r is None:
            print(f"  {days:<10d} {'ERR':>7}")
            continue
        per_week = r.total_trades / (days / 7)
        print(
            f"  {days:<10d} {r.total_trades:>7d} {per_week:>6.1f} "
            f"{r.win_rate:>6.1f}% {r.profit_factor:>6.2f} "
            f"{r.total_pnl_pct:>+8.2f}% {r.max_drawdown_pct:>6.2f}% {r.sharpe_ratio:>7.2f}"
        )

    # Assessment
    print(f"\n{'=' * 80}")
    all_positive = all(results.get(d) and results[d].total_pnl_pct > 0 for d in PERIODS)
    if all_positive:
        print("  >>> TODOS os periodos positivos!")
    else:
        neg = [d for d in PERIODS if results.get(d) and results[d].total_pnl_pct <= 0]
        print(f"  >>> Periodos negativos: {neg}")
    print()


if __name__ == "__main__":
    main()