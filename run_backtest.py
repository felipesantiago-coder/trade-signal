"""
run_backtest.py
---------------
Script para executar o backtest da CTEV v3 e exibir resultados comparativos.
"""

import logging
import json
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from backtest import run_backtest

def main():
    print("=" * 60)
    print("CTEV v3 — Backtest BTC/USDT 1H (730 dias)")
    print("Estrategia: Fibonacci Pullback Trend-Following + MACD")
    print("=" * 60)
    print()

    metrics, trades = run_backtest(
        symbol="BTC/USDT",
        timeframe="1h",
        days=730,
        atr_pct_min=0.15,
        atr_pct_max=0.85,
        advanced=False,
    )

    # Print results
    print()
    print("=" * 60)
    print("RESULTADOS DO BACKTEST")
    print("=" * 60)
    print(f"Periodo: {metrics.period_start} a {metrics.period_end}")
    print()
    print(f"Total de Sinais:        {metrics.total_trades}")
    print(f"  LONG:                 {metrics.long_trades}")
    print(f"  SHORT:                {metrics.short_trades}")
    print(f"  Wins:                 {metrics.wins}")
    print(f"  Losses:               {metrics.losses}")
    print()
    print(f"Taxa de Acerto:         {metrics.win_rate:.1f}%")
    print(f"Fator de Lucro:         {metrics.profit_factor:.2f}")
    print(f"Max Drawdown:           {metrics.max_drawdown_pct:.2f}%")
    print(f"Resultado Total:         {metrics.total_pnl_pct:+.2f}%")
    print(f"Buy & Hold:              {metrics.buy_hold_pct:+.2f}%")
    print(f"Sharpe Ratio:           {metrics.sharpe_ratio:.2f}")
    print()
    print(f"Melhor Trade:           {metrics.best_trade_pct:+.2f}%")
    print(f"Pior Trade:             {metrics.worst_trade_pct:+.2f}%")
    print(f"Media Bars Held:        {metrics.avg_bars_held:.1f}")
    print(f"Media R:R:              {metrics.avg_r_r:.2f}")
    print()
    print(f"Sinais Filtrados ATR:   {metrics.atr_pct_filtered}")
    print()

    # Compare with v2
    print("=" * 60)
    print("COMPARACAO v2 vs v3")
    print("=" * 60)
    v2 = {
        "total_trades": 61,
        "win_rate": 13.1,
        "profit_factor": 0.10,
        "max_drawdown_pct": 40.97,
        "total_pnl_pct": -40.77,
        "buy_hold_pct": 6.14,
    }
    print(f"{'Metrica':<25} {'v2':>10} {'v3':>10} {'Delta':>10}")
    print("-" * 60)
    for key, label in [
        ("total_trades", "Total Sinais"),
        ("win_rate", "Taxa Acerto (%)"),
        ("profit_factor", "Fator Lucro"),
        ("max_drawdown_pct", "Max DD (%)"),
        ("total_pnl_pct", "PnL Total (%)"),
    ]:
        old = v2[key]
        new = getattr(metrics, key)
        delta = new - old
        print(f"{label:<25} {old:>10.1f} {new:>10.2f} {delta:>+10.2f}")

    # Print last 10 trades
    if trades:
        print()
        print("=" * 60)
        print("ULTIMOS 10 TRADES")
        print("=" * 60)
        for t in trades[-10:]:
            emoji = "✅" if t.pnl_pct > 0 else "❌"
            print(f"  {emoji} {t.type} | entry={t.entry_price:.2f} exit={t.exit_price:.2f} "
                  f"PnL={t.pnl_pct:+.2f}% | {t.exit_reason} | bars={t.bars_held}")

    print()
    print("Backtest concluido!")


if __name__ == "__main__":
    main()
