"""
test_v12_pro.py
--------------
Backtest CTEV v12.0 Professional Selective em todos os periodos.
Compara com baseline v10.0.
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

# Baseline v10.0
V10_BASELINE = {
    30:  {"trades": 3,   "wr": 0.0,  "pnl": -3.53},
    90:  {"trades": 16,  "wr": 37.5, "pnl": 2.52},
    180: {"trades": 31,  "wr": 35.5, "pnl": -2.01},
    365: {"trades": 67,  "wr": 46.3, "pnl": 31.17},
    730: {"trades": 134, "wr": 44.0, "pnl": 45.96},
}

PERIODS = [30, 90, 180, 365, 730]


def main():
    print()
    print("=" * 80)
    print("  CTEV v12.0 Professional Selective — Backtest Multi-Periodo")
    print("  BTC/USDT | 1h | Mudancas: ADX 36, DI ON, EMA proximity OFF")
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
            print(f"  Win Rate:    {m.win_rate:6.1f}%")
            print(f"  Profit Factor:{m.profit_factor:6.2f}")
            print(f"  PnL Total:   {m.total_pnl_pct:+7.2f}%")
            print(f"  Buy & Hold:  {m.buy_hold_pct:+7.2f}%")
            print(f"  Max DD:      {m.max_drawdown_pct:6.2f}%")
            print(f"  Sharpe:      {m.sharpe_ratio:6.2f}")
            print(f"  Avg R:R:     {m.avg_r_r:6.2f}")
            print(f"  Avg Bars:    {m.avg_bars_held:6.1f}")
            print(f"  Partial TP:  {m.partial_tp_count}")

            # Exit reasons
            if trades:
                reasons = {}
                for t in trades:
                    r = t.exit_reason
                    reasons[r] = reasons.get(r, 0) + 1
                print(f"  Exits:       {reasons}")

        except Exception as e:
            print(f"  ERRO: {e}")
            results[days] = None

    elapsed = time.time() - t0

    # Comparison table
    print(f"\n\n{'=' * 80}")
    print(f"  COMPARACAO: v12.0 Professional vs v10.0 Baseline")
    print(f"  Tempo de execucao: {elapsed:.0f}s")
    print(f"{'=' * 80}")
    print()
    print(f"  {'Periodo':<10} {'v10 Trades':>10} {'v12 Trades':>11} {'v10 WR':>8} {'v12 WR':>8} {'v10 PnL':>9} {'v12 PnL':>9} {'Delta':>8}")
    print(f"  {'-' * 78}")

    for days in PERIODS:
        b = V10_BASELINE[days]
        r = results.get(days)
        if r is None:
            print(f"  {days:<10d} {b['trades']:>10d} {'ERROR':>11} {b['wr']:>7.1f}% {'N/A':>8} {b['pnl']:>+8.2f}% {'N/A':>9} {'N/A':>8}")
            continue
        delta_pnl = r.total_pnl_pct - b["pnl"]
        delta_wr = r.win_rate - b["wr"]
        print(
            f"  {days:<10d} {b['trades']:>10d} {r.total_trades:>11d} "
            f"{b['wr']:>7.1f}% {r.win_rate:>7.1f}% "
            f"{b['pnl']:>+8.2f}% {r.total_pnl_pct:>+8.2f}% {delta_pnl:>+7.2f}%"
        )

    # Summary assessment
    print(f"\n{'=' * 80}")
    print(f"  AVALIACAO v12.0")
    print(f"{'=' * 80}")

    all_positive = True
    improved = 0
    worsened = 0
    for days in PERIODS:
        b = V10_BASELINE[days]
        r = results.get(days)
        if r is None:
            all_positive = False
            continue
        if r.total_pnl_pct > 0:
            print(f"  {days}d: POSITIVO ({r.total_pnl_pct:+.2f}%) — trades: {r.total_trades}, WR: {r.win_rate:.1f}%")
        else:
            all_positive = False
            print(f"  {days}d: NEGATIVO ({r.total_pnl_pct:+.2f}%) — trades: {r.total_trades}, WR: {r.win_rate:.1f}% ***")

        if r.total_pnl_pct > b["pnl"]:
            improved += 1
        else:
            worsened += 1

    print()
    if all_positive:
        print("  >>> RESULTADO: TODOS os periodos positivos!")
    else:
        print("  >>> RESULTADO: Ainda ha periodos negativos — ajustes necessarios")

    print(f"  Melhorou em {improved}/5 periodos vs v10.0, piorou em {worsened}/5")
    print()


if __name__ == "__main__":
    main()
