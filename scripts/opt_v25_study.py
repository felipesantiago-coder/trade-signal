"""
Optimization study v25: Identify parameter changes to push 30d and 90d to EXCELENTE.

Key findings from v24 diag:
  30d: +60.77% PF=0.88 (POSITIVO) -- CTEV momentum drags PF below 1.0
  90d: +70.31% PF=1.25 (MUITO BOM) -- needs +30% more PnL for EXCELENTE
  180d: +206.92% PF=1.15 (EXCELENTE) -- already there!
  365d: +915.98% PF=1.22 (EXCELENTE) -- already there!

Hypothesis:
1. Increase MAX_CONCURRENT 3->4-5 (more squeeze entries simultaneously)
2. Increase squeeze_breakout ENTRY_RISK 6%->8-10% (star strategy)
3. Lower SQUEEZE_BBWP_THRESHOLD 40->50 (more squeeze signals)
4. Increase ema_bounce ENTRY_RISK 5%->7% 
5. Reduce cooldown aggressiveness
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time, copy, itertools
t0 = time.time()

from backtest import (
    fetch_historical_ohlcv, compute_indicators, calculate_metrics,
    DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS,
)
from strategy_profiles import get_profile
import strategy
import sim_concurrent as sim

# Download data once
print("Downloading 730d 1h data...", flush=True)
df = fetch_historical_ohlcv("BTC/USDT", "1h", 800)
df_ind = compute_indicators(df, timeframe="1h")
df_clean = df_ind.dropna(subset=[
    "ema20", "ema50", "ema200", "rsi", "atr", "atr_percentile",
    "macd", "macd_signal", "macd_hist",
    "adx", "plus_di", "minus_di", "regime",
]).copy()
print(f"Clean candles: {len(df_clean)}", flush=True)

profile = get_profile("1h")

# Save original values
orig = {
    'MAX_CONCURRENT': sim.MAX_CONCURRENT,
    'SQUEEZE_RISK': sim.ENTRY_RISK['squeeze_breakout'],
    'EMA_RISK': sim.ENTRY_RISK['ema_bounce'],
    'CTEV_RISK': sim.ENTRY_RISK['ctev_momentum'],
    'RSI_REV_RISK': sim.ENTRY_RISK['rsi_reversal'],
    'BBWP_THRESH': strategy.SQUEEZE_BBWP_THRESHOLD,
    'COOLDOWN_TRIGGER': None,  # need to patch in simulate
    'COOLDOWN_BARS': None,
    'SQUEEZE_RSI_LONG': None,  # RSI filter in squeeze
}

def verdict(pnl, pf, dd, rd):
    if pnl >= 200 and pf >= 1.1 and rd > 3.0: return "EXCELENTE_t1"
    if pnl >= 100 and pf >= 1.1 and rd > 2.0: return "EXCELENTE_t2"
    if pnl >= 50 and pf >= 1.0 and rd > 1.5: return "MUITO_BOM"
    if pnl >= 20 and pf >= 1.0: return "BOM"
    if pnl > 0 and pf >= 0.9: return "ACEITAVEL"
    if pnl > 0: return "POSITIVO"
    return "FRACO"

def run_test(days, max_concurrent, squeeze_risk, ema_risk, ctev_risk, rsi_rev_risk,
             bbwp_thresh, cooldown_trigger, cooldown_bars):
    """Run a single backtest with given parameters."""
    candles = days * 24
    if candles > len(df_clean):
        return None
    slice_df = df_clean.iloc[-candles:]
    
    # Patch params
    sim.MAX_CONCURRENT = max_concurrent
    sim.ENTRY_RISK['squeeze_breakout'] = squeeze_risk
    sim.ENTRY_RISK['ema_bounce'] = ema_risk
    sim.ENTRY_RISK['ctev_momentum'] = ctev_risk
    sim.ENTRY_RISK['rsi_reversal'] = rsi_rev_risk
    strategy.SQUEEZE_BBWP_THRESHOLD = bbwp_thresh
    
    # We can't easily patch cooldown without modifying the function
    # So we'll test with the built-in cooldown and focus on other params
    
    trades, atr_filtered, diag = sim.simulate_trades_concurrent(
        slice_df,
        atr_pct_min=profile.atr_pct_min,
        atr_pct_max=profile.atr_pct_max,
        profile=profile,
    )
    metrics = calculate_metrics(trades, slice_df, atr_filtered)
    
    pnl = metrics.total_pnl_pct
    pf = metrics.profit_factor
    dd = metrics.max_drawdown_pct
    rd = pnl / dd if dd > 0 else 0
    
    return {
        'pnl': round(pnl, 2), 'pf': round(pf, 3), 'dd': round(dd, 2),
        'wr': round(metrics.win_rate, 1), 'rd': round(rd, 2),
        'trades': metrics.total_trades, 'verdict': verdict(pnl, pf, dd, rd),
        'diag_cooldown': diag.get('cooldown_skip', 0),
        'diag_maxc': diag.get('max_concurrent_hit', 0),
    }

# Test matrix
print("\n" + "="*100, flush=True)
print("OPTIMIZATION STUDY: Finding path to EXCELENTE for 30d and 90d", flush=True)
print("="*100, flush=True)

configs = [
    # (label, max_conc, sq_risk, ema_risk, ctev_risk, rsi_risk, bbwp_thresh)
    ("v24_BASELINE", 3, 0.060, 0.050, 0.0005, 0.035, 40.0),
    ("A: mc=4", 4, 0.060, 0.050, 0.0005, 0.035, 40.0),
    ("B: mc=5", 5, 0.060, 0.050, 0.0005, 0.035, 40.0),
    ("C: sq=8%", 3, 0.080, 0.050, 0.0005, 0.035, 40.0),
    ("D: sq=10%", 3, 0.100, 0.050, 0.0005, 0.035, 40.0),
    ("E: ema=7%", 3, 0.060, 0.070, 0.0005, 0.035, 40.0),
    ("F: A+C", 4, 0.080, 0.050, 0.0005, 0.035, 40.0),
    ("G: A+D", 4, 0.100, 0.050, 0.0005, 0.035, 40.0),
    ("H: B+D", 5, 0.100, 0.050, 0.0005, 0.035, 40.0),
    ("I: A+C+E", 4, 0.080, 0.070, 0.0005, 0.035, 40.0),
    ("J: B+D+E", 5, 0.100, 0.070, 0.0005, 0.035, 40.0),
    ("K: bbwp=50", 3, 0.060, 0.050, 0.0005, 0.035, 50.0),
    ("L: A+bbwp50", 4, 0.060, 0.050, 0.0005, 0.035, 50.0),
    ("M: A+C+bbwp50", 4, 0.080, 0.050, 0.0005, 0.035, 50.0),
    ("N: B+D+bbwp50", 5, 0.100, 0.050, 0.0005, 0.035, 50.0),
    ("O: rsi_rev=5%", 3, 0.060, 0.050, 0.0005, 0.050, 40.0),
    ("P: A+C+O+bbwp50", 4, 0.080, 0.070, 0.0005, 0.050, 50.0),
    ("Q: B+D+E+bbwp50", 5, 0.100, 0.070, 0.0005, 0.050, 50.0),
    ("R: B+D+bbwp50+rsi6%", 5, 0.100, 0.070, 0.0005, 0.060, 50.0),
    ("S: aggressive", 5, 0.120, 0.080, 0.0005, 0.060, 55.0),
]

results = {}
for label, mc, sq, ema, ctev, rsi_r, bbwp in configs:
    r30 = run_test(30, mc, sq, ema, ctev, rsi_r, bbwp, 2, 3)
    r90 = run_test(90, mc, sq, ema, ctev, rsi_r, bbwp, 2, 3)
    r180 = run_test(180, mc, sq, ema, ctev, rsi_r, bbwp, 2, 3)
    r365 = run_test(365, mc, sq, ema, ctev, rsi_r, bbwp, 2, 3)
    
    if r30 and r90 and r180 and r365:
        results[label] = {'30d': r30, '90d': r90, '180d': r180, '365d': r365}
        print(f"\n{label}:", flush=True)
        print(f"  30d:  PnL={r30['pnl']:+7.2f}% PF={r30['pf']:.3f} DD={r30['dd']:5.2f}% rd={r30['rd']:5.2f} => {r30['verdict']}  T={r30['trades']}", flush=True)
        print(f"  90d:  PnL={r90['pnl']:+7.2f}% PF={r90['pf']:.3f} DD={r90['dd']:5.2f}% rd={r90['rd']:5.2f} => {r90['verdict']}  T={r90['trades']}", flush=True)
        print(f"  180d: PnL={r180['pnl']:+7.2f}% PF={r180['pf']:.3f} DD={r180['dd']:5.2f}% rd={r180['rd']:5.2f} => {r180['verdict']}  T={r180['trades']}", flush=True)
        print(f"  365d: PnL={r365['pnl']:+7.2f}% PF={r365['pf']:.3f} DD={r365['dd']:5.2f}% rd={r365['rd']:5.2f} => {r365['verdict']}  T={r365['trades']}", flush=True)

# Restore originals
sim.MAX_CONCURRENT = orig['MAX_CONCURRENT']
sim.ENTRY_RISK['squeeze_breakout'] = orig['SQUEEZE_RISK']
sim.ENTRY_RISK['ema_bounce'] = orig['EMA_RISK']
sim.ENTRY_RISK['ctev_momentum'] = orig['CTEV_RISK']
sim.ENTRY_RISK['rsi_reversal'] = orig['RSI_REV_RISK']
strategy.SQUEEZE_BBWP_THRESHOLD = orig['BBWP_THRESH']

print(f"\n{'='*100}", flush=True)
print(f"Study completed in {time.time()-t0:.1f}s", flush=True)

# Find best configs that achieve EXCELENTE on 30d and 90d
print(f"\n=== BEST CONFIGS FOR 30d+90d EXCELENTE ===", flush=True)
best_30d = [(l, r) for l, r in results.items() if r['30d']['verdict'].startswith('EXCELENTE')]
best_90d = [(l, r) for l, r in results.items() if r['90d']['verdict'].startswith('EXCELENTE')]
print(f"30d EXCELENTE configs: {len(best_30d)}", flush=True)
for l, r in best_30d:
    print(f"  {l}: PnL={r['30d']['pnl']:+.2f}% PF={r['30d']['pf']}", flush=True)
print(f"90d EXCELENTE configs: {len(best_90d)}", flush=True)
for l, r in best_90d:
    print(f"  {l}: PnL={r['90d']['pnl']:+.2f}% PF={r['90d']['pf']}", flush=True)

# Check which configs maintain all targets
print(f"\n=== CONFIGS MEETING ALL ORIGINAL TARGETS ===", flush=True)
for l, r in results.items():
    ok_30 = r['30d']['pnl'] >= 40
    ok_90 = r['90d']['pnl'] >= 80
    ok_180 = r['180d']['pnl'] >= 120
    ok_365 = r['365d']['pnl'] >= 160
    if ok_30 and ok_90 and ok_180 and ok_365:
        exc = r['30d']['verdict'].startswith('EXCELENTE')
        exc90 = r['90d']['verdict'].startswith('EXCELENTE')
        exc180 = r['180d']['verdict'].startswith('EXCELENTE')
        print(f"  {l}: 30d={r['30d']['pnl']:+.1f}({'EXC' if exc else r['30d']['verdict'][:4]}) "
              f"90d={r['90d']['pnl']:+.1f}({'EXC' if exc90 else r['90d']['verdict'][:4]}) "
              f"180d={r['180d']['pnl']:+.1f}({'EXC' if exc180 else r['180d']['verdict'][:4]}) "
              f"365d={r['365d']['pnl']:+.1f}", flush=True)
