"""
Optimization study v25c: Fine-tune the sweet spot.

Key findings from v25b:
- EMA_OFF is consistently good
- TP=7.0 + SL=1.5 gets 30d to EXCELENTE but drops 180d
- TP=7.0 + SL=1.8 (ADX30+TP7+EMA_OFF) gets 90d+180d+365d EXCELENTE, 30d=MUITO_BOM (PF=1.063)
- Need: SL between 1.5-1.8 that gets 30d PF>=1.1 AND 180d PF>=1.1

Also: ADX_MIN and RSI changes had NO effect because profile overrides them.
The profile is STANDARD from strategy_profiles.py.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import importlib
t0 = time.time()

from backtest import (
    fetch_historical_ohlcv, compute_indicators, calculate_metrics,
    DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS,
)
from strategy_profiles import get_profile
import strategy as strat
import sim_concurrent as sim

# Download data once
print("Downloading data...", flush=True)
df = fetch_historical_ohlcv("BTC/USDT", "1h", 800)
df_ind = compute_indicators(df, timeframe="1h")
df_clean = df_ind.dropna(subset=[
    "ema20", "ema50", "ema200", "rsi", "atr", "atr_percentile",
    "macd", "macd_signal", "macd_hist",
    "adx", "plus_di", "minus_di", "regime",
]).copy()
print(f"Clean candles: {len(df_clean)}", flush=True)

profile = get_profile("1h")

# Save originals
orig_mom_sl = strat.MOMENTUM_SL_ATR_MULT
orig_mom_tp = strat.MOMENTUM_TP_ATR_MULT
orig_sq_risk = sim.ENTRY_RISK['squeeze_breakout']
orig_ema_risk = sim.ENTRY_RISK['ema_bounce']
orig_rsi_rev_risk = sim.ENTRY_RISK['rsi_reversal']

def verdict(pnl, pf, dd, rd):
    if pnl >= 200 and pf >= 1.1 and rd > 3.0: return "EXCELENTE_t1"
    if pnl >= 100 and pf >= 1.1 and rd > 2.0: return "EXCELENTE_t2"
    if pnl >= 50 and pf >= 1.0 and rd > 1.5: return "MUITO_BOM"
    if pnl >= 20 and pf >= 1.0: return "BOM"
    if pnl > 0 and pf >= 0.9: return "ACEITAVEL"
    if pnl > 0: return "POSITIVO"
    return "FRACO"

def run_test(days):
    candles = days * 24
    if candles > len(df_clean): return None
    slice_df = df_clean.iloc[-candles:]
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
    }

def reset_params():
    strat.MOMENTUM_SL_ATR_MULT = orig_mom_sl
    strat.MOMENTUM_TP_ATR_MULT = orig_mom_tp
    sim.ENTRY_RISK['squeeze_breakout'] = orig_sq_risk
    sim.ENTRY_RISK['ema_bounce'] = orig_ema_risk
    sim.ENTRY_RISK['rsi_reversal'] = orig_rsi_rev_risk

# Grid: SL x TP x squeeze_risk x ema_off
# Focus on SL=1.5-1.8, TP=6.0-7.5, sq=8-10%
print(f"\n{'='*100}", flush=True)
print("STUDY C: Fine-tuning SL/TP/sizing for 30d EXCELENTE + 180d EXCELENTE", flush=True)
print(f"{'='*100}", flush=True)

best_configs = []

for sl in [1.5, 1.6, 1.7, 1.8]:
    for tp in [6.0, 6.5, 7.0, 7.5]:
        for sq in [0.08, 0.10]:
            reset_params()
            strat.MOMENTUM_SL_ATR_MULT = sl
            strat.MOMENTUM_TP_ATR_MULT = tp
            sim.ENTRY_RISK['squeeze_breakout'] = sq
            sim.ENTRY_RISK['ema_bounce'] = 0.0  # EMA OFF
            sim.ENTRY_RISK['rsi_reversal'] = 0.05  # original
            
            r30 = run_test(30)
            r90 = run_test(90)
            r180 = run_test(180)
            r365 = run_test(365)
            
            if not all([r30, r90, r180, r365]): continue
            
            # Check if all targets met
            targets_ok = (r30['pnl'] >= 40 and r90['pnl'] >= 80 and 
                         r180['pnl'] >= 120 and r365['pnl'] >= 160)
            
            # Count EXCELENTE
            exc_count = sum(1 for r in [r30, r90, r180, r365] 
                           if r['verdict'].startswith('EXCELENTE'))
            
            label = f"SL={sl} TP={tp} sq={int(sq*100)}%"
            
            if targets_ok and exc_count >= 2:
                best_configs.append({
                    'label': label, 'sl': sl, 'tp': tp, 'sq': sq,
                    '30d': r30, '90d': r90, '180d': r180, '365d': r365,
                    'exc_count': exc_count,
                })
                print(f"{label:25s}: 30d={r30['pnl']:+7.1f}% PF={r30['pf']:.3f}({r30['verdict'][:4]}) "
                      f"90d={r90['pnl']:+7.1f}% PF={r90['pf']:.3f}({r90['verdict'][:4]}) "
                      f"180d={r180['pnl']:+7.1f}% PF={r180['pf']:.3f}({r180['verdict'][:4]}) "
                      f"365d={r365['pnl']:+7.1f}% EXC={exc_count}/4", flush=True)

reset_params()

# Show top configs
print(f"\n{'='*100}", flush=True)
print("TOP CONFIGS (sorted by EXCELENTE count, then 30d PnL):", flush=True)
best_configs.sort(key=lambda x: (-x['exc_count'], -x['30d']['pnl']))
for c in best_configs[:10]:
    r = c
    print(f"  {r['label']:25s}: EXC={r['exc_count']}/4 "
          f"30d={r['30d']['pnl']:+.1f}%(PF={r['30d']['pf']:.3f}) "
          f"90d={r['90d']['pnl']:+.1f}%(PF={r['90d']['pf']:.3f}) "
          f"180d={r['180d']['pnl']:+.1f}%(PF={r['180d']['pf']:.3f}) "
          f"365d={r['365d']['pnl']:+.1f}%", flush=True)

print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
