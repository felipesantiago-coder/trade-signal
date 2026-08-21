"""
Optimization v25d: Validate the chosen config (SL=1.7 TP=7.5 sq=8% EMA_OFF) on all 5 timeframes including 730d.
Also compare with v24 baseline.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
t0 = time.time()

from backtest import (
    fetch_historical_ohlcv, compute_indicators, calculate_metrics,
)
from strategy_profiles import get_profile
import strategy as strat
import sim_concurrent as sim

print("Downloading 800d data...", flush=True)
df = fetch_historical_ohlcv("BTC/USDT", "1h", 850)
df_ind = compute_indicators(df, timeframe="1h")
df_clean = df_ind.dropna(subset=[
    "ema20", "ema50", "ema200", "rsi", "atr", "atr_percentile",
    "macd", "macd_signal", "macd_hist",
    "adx", "plus_di", "minus_di", "regime",
]).copy()
print(f"Clean candles: {len(df_clean)}", flush=True)

profile = get_profile("1h")

orig_mom_sl = strat.MOMENTUM_SL_ATR_MULT
orig_mom_tp = strat.MOMENTUM_TP_ATR_MULT
orig_sq_risk = sim.ENTRY_RISK['squeeze_breakout']
orig_ema_risk = sim.ENTRY_RISK['ema_bounce']

def verdict(pnl, pf, dd, rd):
    if pnl >= 200 and pf >= 1.1 and rd > 3.0: return "EXCELENTE (tier1)"
    if pnl >= 100 and pf >= 1.1 and rd > 2.0: return "EXCELENTE (tier2)"
    if pnl >= 50 and pf >= 1.0 and rd > 1.5: return "MUITO BOM"
    if pnl >= 20 and pf >= 1.0: return "BOM"
    if pnl > 0 and pf >= 0.9: return "ACEITAVEL"
    if pnl > 0: return "POSITIVO"
    return "FRACO"

def run_test(days, label):
    # Use 800d of data for 730d test
    candles_needed = days * 24
    start_idx = max(0, len(df_clean) - candles_needed)
    slice_df = df_clean.iloc[start_idx:]
    
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
    
    # Breakdown
    from collections import defaultdict
    by_type = defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0})
    for t in trades:
        et = t.entry_type
        by_type[et]["count"] += 1
        by_type[et]["pnl"] += t.pnl_pct
        if t.pnl_pct > 0: by_type[et]["wins"] += 1
    
    v = verdict(pnl, pf, dd, rd)
    exc = "✅" if "EXCELENTE" in v else "❌"
    
    print(f"  {label}: PnL={pnl:+9.2f}% PF={pf:.3f} DD={dd:5.2f}% WR={metrics.win_rate:.1f}% rd={rd:5.2f} => {v} {exc}", flush=True)
    for et, s in sorted(by_type.items(), key=lambda x: -x[1]['count']):
        wr_t = s['wins']/s['count']*100 if s['count'] > 0 else 0
        print(f"    {et:20s}: {s['count']:4d}T WR={wr_t:5.1f}% PnL={s['pnl']:+9.2f}%", flush=True)
    print(f"    Cooldown skips: {diag.get('cooldown_skip', 0)}, Max concurrent hits: {diag.get('max_concurrent_hit', 0)}", flush=True)

# ========== V24 BASELINE ==========
print(f"\n{'='*80}", flush=True)
print("V24 BASELINE (SL=1.8, TP=5.5, sq=6%, EMA=5%):", flush=True)
print(f"{'='*80}", flush=True)
strat.MOMENTUM_SL_ATR_MULT = 1.8
strat.MOMENTUM_TP_ATR_MULT = 5.5
sim.ENTRY_RISK['squeeze_breakout'] = 0.06
sim.ENTRY_RISK['ema_bounce'] = 0.05

for days in [30, 90, 180, 365, 730]:
    run_test(days, f"{days}d")

# ========== V25 CANDIDATE ==========
print(f"\n{'='*80}", flush=True)
print("V25 CANDIDATE (SL=1.7, TP=7.5, sq=8%, EMA=OFF):", flush=True)
print(f"{'='*80}", flush=True)
strat.MOMENTUM_SL_ATR_MULT = 1.7
strat.MOMENTUM_TP_ATR_MULT = 7.5
sim.ENTRY_RISK['squeeze_breakout'] = 0.08
sim.ENTRY_RISK['ema_bounce'] = 0.0

for days in [30, 90, 180, 365, 730]:
    run_test(days, f"{days}d")

# Restore
strat.MOMENTUM_SL_ATR_MULT = orig_mom_sl
strat.MOMENTUM_TP_ATR_MULT = orig_mom_tp
sim.ENTRY_RISK['squeeze_breakout'] = orig_sq_risk
sim.ENTRY_RISK['ema_bounce'] = orig_ema_risk

print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
