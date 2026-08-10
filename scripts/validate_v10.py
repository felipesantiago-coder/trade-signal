"""
validate_v10.py — Backtest CTEV v10.0 em multiplos periodos
Valida: 30d, 90d, 180d, 365d, 730d
"""
import sys, os, logging
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from backtest import run_backtest

PERIODS = [30, 90, 180, 365, 730]

print("=" * 80)
print("CTEV v10.0 — VALIDACAO MULTI-PERIODO")
print("Mudancas: ADX 25 (de 32), RSI 40-68/30-62, EMA prox 1.2%/1.8%,")
print("          cooldown 3 SL consecutivos (24 bars), pos-TP1 SL 1.5x ATR")
print("=" * 80)

results = []
for days in PERIODS:
    print(f"\n{'─' * 60}")
    print(f"Backtest {days} dias...")
    print(f"{'─' * 60}")
    metrics, trades = run_backtest(symbol="BTC/USDT", timeframe="1h", days=days)
    
    wr = metrics.win_rate
    pf = metrics.profit_factor
    pnl = metrics.total_pnl_pct
    bh = metrics.buy_hold_pct
    alpha = pnl - bh
    dd = metrics.max_drawdown_pct
    sharpe = metrics.sharpe_ratio
    n_trades = metrics.total_trades
    
    if pf >= 1.5 and wr >= 40:
        verdict = "BOM"
    elif pf >= 1.0 and pnl > 0:
        verdict = "ACEITAVEL"
    elif pnl > 0:
        verdict = "POSITIVO (marginal)"
    else:
        verdict = "FRACO"
    
    print(f"\n  Trades: {n_trades} | WR: {wr:.1f}% | PF: {pf:.2f} | PnL: {pnl:+.2f}%")
    print(f"  B&H: {bh:+.2f}% | Alpha: {alpha:+.2f}pp | DD: {dd:.2f}% | Sharpe: {sharpe:.2f}")
    print(f"  VEREDICTO: {verdict}")
    
    results.append({
        'days': days, 'trades': n_trades, 'wr': wr, 'pf': pf,
        'pnl': pnl, 'bh': bh, 'alpha': alpha, 'dd': dd, 'sharpe': sharpe,
        'verdict': verdict,
    })

# Summary table
print(f"\n{'=' * 80}")
print("RESUMO COMPARATIVO v10.0")
print(f"{'=' * 80}")
print(f"{'Periodo':>8} {'Trades':>7} {'WR':>6} {'PF':>6} {'PnL':>8} {'Alpha':>8} {'DD':>7} {'Sharpe':>7} {'Veredicto':>20}")
print("-" * 80)
for r in results:
    print(f"{r['days']:>7}d {r['trades']:>7} {r['wr']:>5.1f}% {r['pf']:>6.2f} {r['pnl']:>+7.2f}% {r['alpha']:>+7.2f}pp {r['dd']:>6.2f}% {r['sharpe']:>7.2f} {r['verdict']:>20}")

# Compare v9.0 baseline
print(f"\n{'─' * 80}")
print("COMPARACAO v9.0 -> v10.0")
print(f"{'─' * 80}")
v9_baseline = {
    30:  {'trades': 2, 'wr': 0.0, 'pf': 0.00, 'pnl': -3.06},
    90:  {'trades': 15, 'wr': 33.3, 'pf': 1.00, 'pnl': 0.02},
    180: {'trades': 30, 'wr': 33.3, 'pf': 0.81, 'pnl': -7.98},
    365: {'trades': 67, 'wr': 46.3, 'pf': 1.30, 'pnl': 25.26},
    730: {'trades': 134, 'wr': 44.0, 'pf': 1.15, 'pnl': 25.98},
}
print(f"{'Periodo':>8} | {'v9.0 PnL':>10} | {'v10.0 PnL':>11} | {'Delta':>8} | {'v9.0 PF':>8} | {'v10.0 PF':>9} | {'v9.0 Trades':>12} | {'v10.0 Trades':>13}")
print("-" * 100)
for r in results:
    v9 = v9_baseline.get(r['days'], {})
    d_pnl = r['pnl'] - v9.get('pnl', 0)
    d_pf = r['pf'] - v9.get('pf', 0)
    print(f"{r['days']:>7}d | {v9.get('pnl',0):>+9.2f}% | {r['pnl']:>+10.2f}% | {d_pnl:>+7.2f}pp | {v9.get('pf',0):>8.2f} | {r['pf']:>9.2f} | {v9.get('trades',0):>12} | {r['trades']:>13}")

print(f"\nValidacao concluida!")
