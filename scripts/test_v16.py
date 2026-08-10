"""
test_v16.py
-------------
Backtest CTEV v16.0 Multi-Strategy em todos os periodos.
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
    print("  CTEV v16.0 Multi-Strategy — Backtest Multi-Periodo")
    print("  BTC/USDT | 1h | CTEV pullback + momentum + ranging MR")
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
            results[days] = (m, trades)

            print(f"  Trades:      {m.total_trades:4d}  (L:{m.long_trades} S:{m.short_trades})")
            print(f"  Trades/day:  {m.total_trades / max(days, 1):.2f}")
            print(f"  Win Rate:    {m.win_rate:6.1f}%")
            print(f"  Profit Factor:{m.profit_factor:6.2f}")
            print(f"  PnL Total:   {m.total_pnl_pct:+7.2f}%")
            print(f"  Buy & Hold:  {m.buy_hold_pct:+7.2f}%")
            print(f"  Max DD:      {m.max_drawdown_pct:6.2f}%")
            print(f"  Sharpe:      {m.sharpe_ratio:6.2f}")
            print(f"  Avg Bars:    {m.avg_bars_held:6.1f}")
            print(f"  Partial TP:  {m.partial_tp_count}")

            # Entry type breakdown
            if trades:
                types = {}
                for t in trades:
                    # entry_type is not on TradeResult, use exit info
                    r = t.exit_reason
                    types[r] = types.get(r, 0) + 1
                print(f"  Exits:       {types}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            results[days] = None

    elapsed = time.time() - t0

    # Summary
    print(f"\n\n{'=' * 80}")
    print(f"  RESUMO v16.0  (tempo: {elapsed:.0f}s)")
    print(f"{'=' * 80}")
    print()
    print(f"  {'Periodo':<10} {'Trades':>8} {'T/day':>7} {'WR':>7} {'PF':>7} {'PnL':>9} {'DD':>7} {'Sharpe':>7}")
    print(f"  {'-' * 70}")

    all_positive = True
    for days in PERIODS:
        r = results.get(days)
        if r is None:
            print(f"  {days:<10d} {'ERROR':>8}")
            all_positive = False
            continue
        m = r[0]
        td = m.total_trades / max(days, 1)
        mark = "***" if m.total_pnl_pct < 0 else ""
        print(
            f"  {days:<10d} {m.total_trades:>8d} {td:>6.2f} "
            f"{m.win_rate:>6.1f}% {m.profit_factor:>6.2f} "
            f"{m.total_pnl_pct:>+8.2f}% {m.max_drawdown_pct:>6.2f}% "
            f"{m.sharpe_ratio:>6.2f} {mark}"
        )
        if m.total_pnl_pct < 0:
            all_positive = False

    print()
    if all_positive:
        print("  >>> TODOS os periodos POSITIVOS!")
    else:
        print("  >>> Ainda ha periodos negativos — ajustes necessarios")
    print()


if __name__ == "__main__":
    main()
