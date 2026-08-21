"""
Optimization study v25b: Fix 30d PF by reducing losing CTEV momentum trades.

Key insight: 30d has 37 CTEV momentum trades (21.6% WR, -11.20% PnL).
These drag PF from >1.0 to 0.88. 

Approaches:
1. Increase ADX_MIN for CTEV momentum (25->30, 35)
2. Tighten CTEV RSI range 
3. Remove ema_bounce (10% WR in 30d)
4. Increase CTEV momentum SL/TP to improve R:R
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
orig_adx = strat.ADX_MIN
orig_mom_sl = strat.MOMENTUM_SL_ATR_MULT
orig_mom_tp = strat.MOMENTUM_TP_ATR_MULT
orig_sq_risk = sim.ENTRY_RISK['squeeze_breakout']
orig_ema_risk = sim.ENTRY_RISK['ema_bounce']
orig_rsi_long_min = strat.RSI_LONG_MIN
orig_rsi_long_max = strat.RSI_LONG_MAX
orig_mom_rsi_floor = 45.0  # hardcoded in strategy.py line 378
def verdict(pnl, pf, dd, rd):
    if pnl >= 200 and pf >= 1.1 and rd > 3.0: return "EXCELENTE_t1"
    if pnl >= 100 and pf >= 1.1 and rd > 2.0: return "EXCELENTE_t2"
    if pnl >= 50 and pf >= 1.0 and rd > 1.5: return "MUITO_BOM"
    if pnl >= 20 and pf >= 1.0: return "BOM"
    if pnl > 0 and pf >= 0.9: return "ACEITAVEL"
    if pnl > 0: return "POSITIVO"
    return "FRACO"

def run_test(days, label=""):
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
    
    # Breakdown by type
    from collections import defaultdict
    by_type = defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0})
    for t in trades:
        et = t.entry_type
        by_type[et]["count"] += 1
        by_type[et]["pnl"] += t.pnl_pct
        if t.pnl_pct > 0: by_type[et]["wins"] += 1
    
    return {
        'pnl': round(pnl, 2), 'pf': round(pf, 3), 'dd': round(dd, 2),
        'wr': round(metrics.win_rate, 1), 'rd': round(rd, 2),
        'trades': metrics.total_trades, 'verdict': verdict(pnl, pf, dd, rd),
        'by_type': dict(by_type),
        'cooldown': diag.get('cooldown_skip', 0),
        'maxc': diag.get('max_concurrent_hit', 0),
    }

def reset_params():
    strat.ADX_MIN = orig_adx
    strat.MOMENTUM_SL_ATR_MULT = orig_mom_sl
    strat.MOMENTUM_TP_ATR_MULT = orig_mom_tp
    sim.ENTRY_RISK['squeeze_breakout'] = orig_sq_risk
    sim.ENTRY_RISK['ema_bounce'] = orig_ema_risk
    strat.RSI_LONG_MIN = orig_rsi_long_min
    strat.RSI_LONG_MAX = orig_rsi_long_max

# Tests
print(f"\n{'='*100}", flush=True)
print("STUDY B: Fix 30d PF (the EXCELENTE bottleneck)", flush=True)
print(f"{'='*100}", flush=True)

tests = [
    # (label, adx_min, mom_sl, mom_tp, sq_risk, ema_risk, rsi_l_min, rsi_l_max, ema_disable)
    ("BASELINE", 25, 1.8, 5.5, 0.08, 0.05, 44, 66, False),
    # Test ADX increase for CTEV
    ("ADX=30", 30, 1.8, 5.5, 0.08, 0.05, 44, 66, False),
    ("ADX=35", 35, 1.8, 5.5, 0.08, 0.05, 44, 66, False),
    # Test wider TP for CTEV momentum
    ("TP=7.0", 25, 1.8, 7.0, 0.08, 0.05, 44, 66, False),
    ("TP=8.0", 25, 1.8, 8.0, 0.08, 0.05, 44, 66, False),
    # Tighter SL for CTEV momentum
    ("SL=1.5", 25, 1.5, 5.5, 0.08, 0.05, 44, 66, False),
    # Disable ema_bounce
    ("EMA_OFF", 25, 1.8, 5.5, 0.08, 0.0, 44, 66, True),
    # Combined
    ("ADX30+EMA_OFF", 30, 1.8, 5.5, 0.08, 0.0, 44, 66, True),
    ("ADX30+TP7", 30, 1.8, 7.0, 0.08, 0.05, 44, 66, False),
    ("ADX30+TP7+EMA_OFF", 30, 1.8, 7.0, 0.08, 0.0, 44, 66, True),
    # Tighter RSI range for CTEV
    ("RSI48-62", 25, 1.8, 5.5, 0.08, 0.05, 48, 62, False),
    ("RSI50-64", 25, 1.8, 5.5, 0.08, 0.05, 50, 64, False),
    ("RSI48-62+EMA_OFF", 25, 1.8, 5.5, 0.08, 0.0, 48, 62, True),
    # Aggressive combo
    ("BEST_GUESS", 30, 1.5, 7.0, 0.08, 0.0, 48, 62, True),
]

for label, adx, msl, mtp, sq_r, ema_r, rsi_min, rsi_max, ema_off in tests:
    reset_params()
    strat.ADX_MIN = adx
    strat.MOMENTUM_SL_ATR_MULT = msl
    strat.MOMENTUM_TP_ATR_MULT = mtp
    sim.ENTRY_RISK['squeeze_breakout'] = sq_r
    sim.ENTRY_RISK['ema_bounce'] = 0.0 if ema_off else ema_r
    strat.RSI_LONG_MIN = rsi_min
    strat.RSI_LONG_MAX = rsi_max
    
    r30 = run_test(30, label)
    r90 = run_test(90, label)
    r180 = run_test(180, label)
    r365 = run_test(365, label)
    
    if r30 and r90 and r180 and r365:
        print(f"\n{label}:", flush=True)
        print(f"  30d:  PnL={r30['pnl']:+8.2f}% PF={r30['pf']:.3f} DD={r30['dd']:5.2f}% rd={r30['rd']:5.2f} T={r30['trades']:3d} => {r30['verdict']}", flush=True)
        for et, s in sorted(r30['by_type'].items(), key=lambda x: -x[1]['count']):
            wr_t = s['wins']/s['count']*100 if s['count'] > 0 else 0
            print(f"    {et:20s}: {s['count']:3d}T WR={wr_t:5.1f}% PnL={s['pnl']:+8.2f}%", flush=True)
        print(f"  90d:  PnL={r90['pnl']:+8.2f}% PF={r90['pf']:.3f} DD={r90['dd']:5.2f}% rd={r90['rd']:5.2f} T={r90['trades']:3d} => {r90['verdict']}", flush=True)
        print(f"  180d: PnL={r180['pnl']:+8.2f}% PF={r180['pf']:.3f} DD={r180['dd']:5.2f}% rd={r180['rd']:5.2f} T={r180['trades']:3d} => {r180['verdict']}", flush=True)
        print(f"  365d: PnL={r365['pnl']:+8.2f}% PF={r365['pf']:.3f} DD={r365['dd']:5.2f}% rd={r365['rd']:5.2f} T={r365['trades']:3d} => {r365['verdict']}", flush=True)

reset_params()
print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
