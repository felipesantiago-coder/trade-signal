# -*- coding: utf-8 -*-
"""validate_1h_backtest.py - Backtest 1h com CTEV v7.1 Regime-Switching"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest import run_backtest
from strategy_profiles import get_profile

def main():
    profile = get_profile("1h")
    print(f"Profile: {profile.name}")
    print(f"SL: {profile.sl_atr_mult}x | TP: {profile.tp_atr_mult}x | R:R {profile.rr_ratio}:1")
    print(f"Max bars: {profile.max_bars_held}")
    print()
    print("Executando backtest 1h 730d com CTEV v7.1 Regime-Switching...")
    print()

    metrics, trades = run_backtest(
        symbol="BTC/USDT",
        timeframe="1h",
        days=730,
        regime_switching=True,
    )

    print(f"\n{'=' * 60}")
    print(f"  RESULTADO: 1h CTEV v7.1 Regime-Switching (730 dias)")
    print(f"{'=' * 60}")
    print(f"  Trades:       {metrics.total_trades}")
    print(f"  Win Rate:     {metrics.win_rate:.1f}%")
    print(f"  Profit Factor: {metrics.profit_factor:.2f}")
    print(f"  PnL Total:    {metrics.total_pnl_pct:+.2f}%")
    print(f"  Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
    print(f"  Sharpe:       {metrics.sharpe_ratio:.2f}")
    print(f"  Buy & Hold:   {metrics.buy_hold_pct:+.2f}%")
    diag = metrics._filter_diag
    strat = diag.get('strategy', diag.get('regime_switching', 'N/A'))
    print(f"  Strategy:     {strat}")
    print(f"{'=' * 60}")

    if metrics.total_trades > 0:
        beats = metrics.total_pnl_pct > metrics.buy_hold_pct
        print(f"  Supera B&H: {'SIM' if beats else 'NAO'}")


if __name__ == "__main__":
    main()
