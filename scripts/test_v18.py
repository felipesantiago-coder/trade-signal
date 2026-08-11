"""
test_v18.py
-----------
Backtest CTEV v18.0 em todos os periodos.
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
    print("  CTEV v18.0 — Backtest Multi-Periodo")
    print("  BTC/USDT | 1h | EMA Bounce + Squeeze + CTEV concurrent")
    print("  ADX 22, RSI 42-62/38-58, Cooldown 8b, Anti-martingale ON")
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

            # Entry type breakdown
            entry_types = {}
            for t in trades:
                et = t.entry_type
                entry_types[et] = entry_types.get(et, 0) + 1

            # Exit reasons
            exit_reasons = {}
            for t in trades:
                r = t.exit_reason
                exit_reasons[r] = exit_reasons.get(r, 0) + 1

            trd = m.total_trades / max(days, 1)
            print(f"  Trades:      {m.total_trades:4d}  (L:{m.long_trades} S:{m.short_trades}) | {trd:.2f}/dia")
            print(f"  Win Rate:    {m.win_rate:6.1f}%")
            print(f"  Profit Factor:{m.profit_factor:6.2f}")
            print(f"  PnL Total:   {m.total_pnl_pct:+7.2f}%")
            print(f"  Buy & Hold:  {m.buy_hold_pct:+7.2f}%")
            print(f"  Max DD:      {m.max_drawdown_pct:6.2f}%")
            print(f"  Sharpe:      {m.sharpe_ratio:6.2f}")
            print(f"  Avg R:R:     {m.avg_r_r:6.2f}")
            print(f"  Avg Bars:    {m.avg_bars_held:6.1f}")
            print(f"  Partial TP:  {m.partial_tp_count}")
            print(f"  Entry Types: {entry_types}")
            print(f"  Exits:       {exit_reasons}")

        except Exception as e:
            print(f"  ERRO: {e}")
            import traceback
            traceback.print_exc()
            results[days] = None

    elapsed = time.time() - t0

    # Summary table
    print(f"\n\n{'=' * 90}")
    print(f"  RESUMO v18.0 — Tempo: {elapsed:.0f}s")
    print(f"{'=' * 90}")
    print(f"  {'Periodo':<10} {'Trades':>7} {'Tr/dia':>7} {'WR':>7} {'PF':>7} {'PnL':>9} {'MaxDD':>7} {'Sharpe':>7}")
    print(f"  {'-' * 68}")

    for days in PERIODS:
        data = results.get(days)
        if data is None:
            print(f"  {days:<10d} {'ERROR':>68}")
            continue
        m = data[0]
        trd = m.total_trades / max(days, 1)
        print(
            f"  {days:<10d} {m.total_trades:>7d} {trd:>7.2f} "
            f"{m.win_rate:>6.1f}% {m.profit_factor:>7.2f} "
            f"{m.total_pnl_pct:>+8.2f}% {m.max_drawdown_pct:>6.2f}% "
            f"{m.sharpe_ratio:>7.2f}"
        )

    # Requirements check
    print(f"\n{'=' * 90}")
    print(f"  CHECK DE REQUISITOS")
    print(f"{'=' * 90}")

    all_ok = True
    for days in PERIODS:
        data = results.get(days)
        if data is None:
            all_ok = False
            print(f"  {days}d: ERROR — nao foi possivel executar")
            continue
        m = data[0]
        trd = m.total_trades / max(days, 1)
        ok_freq = "OK" if trd >= 1.0 else "FAIL"
        ok_pnl = "OK" if m.total_pnl_pct > 0 else "FAIL"
        status = "PASS" if ok_freq == "OK" and ok_pnl == "OK" else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"  {days}d: [{status}] freq={ok_freq} ({trd:.2f}/dia) | pnl={ok_pnl} ({m.total_pnl_pct:+.2f}%)")

    print()
    if all_ok:
        print("  >>> TODOS OS PERIODOS APROVADOS!")
    else:
        print("  >>> AINDA HA PERIODOS COM PROBLEMAS")
    print()


if __name__ == "__main__":
    main()
