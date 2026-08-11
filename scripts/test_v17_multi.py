"""
test_v17_multi.py
---------------
Backtest CTEV v17.0 Multi-Strategy Engine em todos os periodos.
Mostra breakdown por entry type.
"""

import sys
import os
import logging
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from backtest import run_backtest

V14_BASELINE = {
    30:  {"trades": 3,   "wr": 0.0,  "pnl": -3.53},
    90:  {"trades": 16,  "wr": 37.5, "pnl": 2.48},
    180: {"trades": 30,  "wr": 36.7, "pnl": -0.21},
    365: {"trades": 64,  "wr": 46.9, "pnl": 33.55},
    730: {"trades": 128, "wr": 44.5, "pnl": 49.03},
}

PERIODS = [30, 90, 180, 365, 730]


def main():
    print()
    print("=" * 80)
    print("  CTEV v17.0 Multi-Strategy Engine — Backtest Multi-Periodo")
    print("  BTC/USDT | 1h | CTEV + Momentum + EMA Bounce + Squeeze + RSI Rev + MR")
    print("=" * 80)
    print()

    results = {}
    all_trades = {}
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
            all_trades[days] = trades

            trades_per_day = m.total_trades / days if days > 0 else 0
            print(f"  Trades:      {m.total_trades:4d}  ({trades_per_day:.2f}/dia)  L:{m.long_trades} S:{m.short_trades}")
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

                # Entry type breakdown
                type_counts = Counter(t.entry_type for t in trades)
                type_pnl = {}
                type_wr = {}
                for t in trades:
                    et = t.entry_type
                    if et not in type_pnl:
                        type_pnl[et] = []
                    type_pnl[et].append(t.pnl_pct)
                print(f"  Entry Types:")
                for et, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                    pnls = type_pnl[et]
                    wins = sum(1 for p in pnls if p > 0)
                    wr = wins / len(pnls) * 100 if pnls else 0
                    avg_pnl = sum(pnls) / len(pnls) if pnls else 0
                    print(f"    {et:<20s}: {count:3d} trades  WR={wr:.0f}%  avg_PnL={avg_pnl:+.2f}%")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERRO: {e}")
            results[days] = None

    elapsed = time.time() - t0

    # Comparison table
    print(f"\n\n{'=' * 80}")
    print(f"  COMPARACAO: v17.0 Multi-Strategy vs v14.3 Baseline")
    print(f"  Tempo de execucao: {elapsed:.0f}s")
    print(f"{'=' * 80}")
    print()
    print(f"  {'Periodo':<10} {'v14 T':>6} {'v17 T':>6} {'v14/d':>6} {'v17/d':>6} {'v14 WR':>8} {'v17 WR':>8} {'v14 PnL':>9} {'v17 PnL':>9} {'Delta':>8}")
    print(f"  {'-' * 86}")

    for days in PERIODS:
        b = V14_BASELINE[days]
        r = results.get(days)
        if r is None:
            print(f"  {days:<10d} {b['trades']:>6d} {'ERR':>6} {b['trades']/days:>5.2f} {'N/A':>6} {b['wr']:>7.1f}% {'N/A':>8} {b['pnl']:>+8.2f}% {'N/A':>9} {'N/A':>8}")
            continue
        delta_pnl = r.total_pnl_pct - b["pnl"]
        v14_tpd = b['trades'] / days
        v17_tpd = r.total_trades / days if days > 0 else 0
        print(
            f"  {days:<10d} {b['trades']:>6d} {r.total_trades:>6d} "
            f"{v14_tpd:>5.2f} {v17_tpd:>5.2f} "
            f"{b['wr']:>7.1f}% {r.win_rate:>7.1f}% "
            f"{b['pnl']:>+8.2f}% {r.total_pnl_pct:>+8.2f}% {delta_pnl:>+7.2f}%"
        )

    # Summary assessment
    print(f"\n{'=' * 80}")
    print(f"  AVALIACAO v17.0")
    print(f"{'=' * 80}")

    all_positive = True
    improved = 0
    worsened = 0
    for days in PERIODS:
        b = V14_BASELINE[days]
        r = results.get(days)
        if r is None:
            all_positive = False
            continue
        tpd = r.total_trades / days if days > 0 else 0
        if r.total_pnl_pct > 0:
            print(f"  {days}d: POSITIVO ({r.total_pnl_pct:+.2f}%) — {r.total_trades} trades ({tpd:.2f}/dia), WR: {r.win_rate:.1f}%")
        else:
            all_positive = False
            print(f"  {days}d: NEGATIVO ({r.total_pnl_pct:+.2f}%) — {r.total_trades} trades ({tpd:.2f}/dia), WR: {r.win_rate:.1f}% ***")

        if r.total_pnl_pct > b["pnl"]:
            improved += 1
        else:
            worsened += 1

    print()
    if all_positive:
        print("  >>> RESULTADO: TODOS os periodos positivos!")
    else:
        print("  >>> RESULTADO: Ainda ha periodos negativos — ajustes necessarios")

    # Check frequency target
    freq_ok = True
    for days in PERIODS:
        r = results.get(days)
        if r and r.total_trades < days * 0.8:  # 0.8 trades/day minimum
            freq_ok = False
            print(f"  >>> FREQUENCIA: {days}d tem {r.total_trades/days:.2f}/dia (meta: 1.0/dia)")
    if freq_ok:
        print("  >>> FREQUENCIA: Todos os periodos com >= 0.8 trades/dia!")

    print(f"  Melhorou em {improved}/5 periodos vs v14.3, piorou em {worsened}/5")
    print()


if __name__ == "__main__":
    main()
