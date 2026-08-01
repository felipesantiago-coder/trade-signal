# -*- coding: utf-8 -*-
"""validate_15m_backtest.py - Backtest 15m com EMA Cross v8"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest import run_backtest
from strategy_profiles import get_profile

def main():
    # Mostra qual estrategia sera usada
    profile = get_profile("15m")
    print(f"Profile: {profile.name}")
    print(f"SL: {profile.sl_atr_mult}x | TP: {profile.tp_atr_mult}x | R:R {profile.rr_ratio}:1")
    print(f"Max bars: {profile.max_bars_held}")
    print()
    print("Executando backtest 15m 365d com EMA Cross v8...")
    print("(A estrategia EMA Cross sera usada automaticamente pelo router)")
    print()

    metrics, trades = run_backtest(
        symbol="BTC/USDT",
        timeframe="15m",
        days=365,
        regime_switching=False,  # Router decide automaticamente
    )

    print(f"\n{'=' * 60}")
    print(f"  RESULTADO: 15m EMA Cross v8 (365 dias)")
    print(f"{'=' * 60}")
    print(f"  Trades:       {metrics.total_trades}")
    print(f"  Win Rate:     {metrics.win_rate:.1f}%")
    print(f"  Profit Factor: {metrics.profit_factor:.2f}")
    print(f"  PnL Total:    {metrics.total_pnl_pct:+.2f}%")
    print(f"  Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
    print(f"  Sharpe:       {metrics.sharpe_ratio:.2f}")
    print(f"  Buy & Hold:   {metrics.buy_hold_pct:+.2f}%")
    print(f"  Strategy:     {metrics._filter_diag.get('strategy', 'N/A')}")
    print(f"{'=' * 60}")

    if metrics.total_trades > 0:
        beats = metrics.total_pnl_pct > metrics.buy_hold_pct
        print(f"  Supera B&H: {'SIM' if beats else 'NAO'}")


if __name__ == "__main__":
    main()
