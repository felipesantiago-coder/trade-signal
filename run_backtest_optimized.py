"""
run_backtest_optimized.py
---------------------------
Compara backtest basico vs advanced (com trailing/BE/partial TP).
"""

import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from backtest import run_backtest


def print_metrics(label, m):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Total Sinais:     {m.total_trades}  (LONG: {m.long_trades}, SHORT: {m.short_trades})")
    print(f"  Taxa Acerto:      {m.win_rate:.1f}%")
    print(f"  Fator Lucro:      {m.profit_factor:.2f}")
    print(f"  Max Drawdown:     {m.max_drawdown_pct:.2f}%")
    print(f"  PnL Total:        {m.total_pnl_pct:+.2f}%")
    print(f"  Buy & Hold:       {m.buy_hold_pct:+.2f}%")
    print(f"  Sharpe Ratio:     {m.sharpe_ratio:.2f}")
    print(f"  Melhor Trade:     {m.best_trade_pct:+.2f}%")
    print(f"  Pior Trade:       {m.worst_trade_pct:+.2f}%")
    print(f"  Media Bars Held:  {m.avg_bars_held:.1f}")
    print(f"  Media R:R:        {m.avg_r_r:.2f}")
    print(f"  BE Triggered:     {m.be_triggered_count}")
    print(f"  Trailing Activ.:  {m.trailing_activated_count}")
    print(f"  Partial TP:       {m.partial_tp_count}")


def main():
    print("\n" + "=" * 60)
    print("CTEV v3 — Comparacao Basico vs Advanced")
    print("=" * 60)

    # Basic
    print("\nExecutando backtest basico...")
    m_basic, t_basic = run_backtest(
        symbol="BTC/USDT", timeframe="1h", days=730,
        atr_pct_min=0.15, atr_pct_max=0.85, advanced=False,
    )
    print_metrics("MODE: BASICO (SL/TP fixos)", m_basic)

    # Advanced
    print("\nExecutando backtest avancado (trailing/BE/partial)...")
    m_adv, t_adv = run_backtest(
        symbol="BTC/USDT", timeframe="1h", days=730,
        atr_pct_min=0.15, atr_pct_max=0.85, advanced=True,
    )
    print_metrics("MODE: AVANCADO (Trailing/BE/Partial TP)", m_adv)

    # Comparison table
    print(f"\n{'='*60}")
    print("  COMPARACAO: BASICO vs AVANCADO")
    print(f"{'='*60}")
    print(f"  {'Metrica':<25} {'Basico':>10} {'Avancado':>10} {'Delta':>10}")
    print(f"  {'-'*58}")
    for key, label in [
        ("total_trades", "Total Sinais"),
        ("win_rate", "Taxa Acerto (%)"),
        ("profit_factor", "Fator Lucro"),
        ("max_drawdown_pct", "Max DD (%)"),
        ("total_pnl_pct", "PnL Total (%)"),
    ]:
        b = getattr(m_basic, key)
        a = getattr(m_adv, key)
        print(f"  {label:<25} {b:>10.2f} {a:>10.2f} {a-b:>+10.2f}")

    # v2 reference
    print(f"\n{'='*60}")
    print("  COMPARACAO COM v2 (anterior)")
    print(f"{'='*60}")
    v2 = {"total_trades": 61, "win_rate": 13.1, "profit_factor": 0.10, "max_drawdown_pct": 40.97, "total_pnl_pct": -40.77}
    best = m_adv if m_adv.profit_factor > m_basic.profit_factor else m_basic
    mode = "Avancado" if m_adv.profit_factor > m_basic.profit_factor else "Basico"
    print(f"  Melhor modo: {mode}")
    print(f"  {'Metrica':<25} {'v2':>10} {'v3 ({mode})':>14} {'Delta':>10}")
    print(f"  {'-'*62}")
    for key, label in [
        ("total_trades", "Total Sinais"),
        ("win_rate", "Taxa Acerto (%)"),
        ("profit_factor", "Fator Lucro"),
        ("max_drawdown_pct", "Max DD (%)"),
        ("total_pnl_pct", "PnL Total (%)"),
    ]:
        old = v2[key]
        new = getattr(best, key)
        print(f"  {label:<25} {old:>10.1f} {new:>14.2f} {new-old:>+10.2f}")

    print("\nBacktest concluido!\n")


if __name__ == "__main__":
    main()
