"""Diagnostic v24: Run all 5 timeframes from single data download."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
t0 = time.time()

from backtest import (
    fetch_historical_ohlcv, compute_indicators,
    BacktestMetrics, TradeResult, DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS,
    _update_progress,
)
from sim_concurrent import simulate_trades_concurrent
from strategy_profiles import get_profile

# 1. Download data once (730d = max needed)
print("Downloading 730d 1h data...", flush=True)
df = fetch_historical_ohlcv("BTC/USDT", "1h", 730)
print(f"Downloaded {len(df)} candles in {time.time()-t0:.1f}s", flush=True)

# 2. Compute indicators once
df_ind = compute_indicators(df, timeframe="1h")
df_clean = df_ind.dropna(subset=[
    "ema20", "ema50", "ema200", "rsi", "atr", "atr_percentile",
    "macd", "macd_signal", "macd_hist",
    "adx", "plus_di", "minus_di", "regime",
]).copy()
print(f"Clean candles: {len(df_clean)}", flush=True)

profile = get_profile("1h")

def verdict(pnl, pf, dd, rd):
    if pnl >= 200 and pf >= 1.1 and rd > 3.0:
        return "EXCELENTE (tier1)"
    elif pnl >= 100 and pf >= 1.1 and rd > 2.0:
        return "EXCELENTE (tier2)"
    elif pnl >= 50 and pf >= 1.0 and rd > 1.5:
        return "MUITO BOM"
    elif pnl >= 20 and pf >= 1.0:
        return "BOM"
    elif pnl > 0 and pf >= 0.9:
        return "ACEITAVEL"
    elif pnl > 0:
        return "POSITIVO"
    else:
        return "FRACO"

# 3. Run each timeframe
print(f"\n{'='*80}", flush=True)
for days in [30, 90, 180, 365, 730]:
    candles_needed = days * 24
    if candles_needed > len(df_clean):
        print(f"{days}d: SKIP (not enough data)", flush=True)
        continue
    
    slice_df = df_clean.iloc[-candles_needed:]
    
    print(f"\n--- {days}d ({len(slice_df)} candles) ---", flush=True)
    t1 = time.time()
    trades, atr_filtered, diag = simulate_trades_concurrent(
        slice_df,
        atr_pct_min=profile.atr_pct_min,
        atr_pct_max=profile.atr_pct_max,
        profile=profile,
    )
    
    # Calculate metrics manually
    from backtest import calculate_metrics
    metrics = calculate_metrics(trades, slice_df, atr_filtered)
    
    pnl = metrics.total_pnl_pct
    pf = metrics.profit_factor
    dd = metrics.max_drawdown_pct
    wr = metrics.win_rate
    rd = pnl / dd if dd > 0 else 0
    v = verdict(pnl, pf, dd, rd)
    
    # Breakdown by entry_type
    from collections import defaultdict
    by_type = defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0})
    for t in trades:
        et = t.entry_type
        by_type[et]["count"] += 1
        by_type[et]["pnl"] += t.pnl_pct
        if t.pnl_pct > 0:
            by_type[et]["wins"] += 1
    
    elapsed = time.time() - t1
    print(f"  Time: {elapsed:.1f}s", flush=True)
    print(f"  Trades: {metrics.total_trades} (L={metrics.long_trades} S={metrics.short_trades})", flush=True)
    print(f"  W={metrics.wins} L={metrics.losses} WR={wr:.1f}%", flush=True)
    print(f"  PnL={pnl:+.2f}%  PF={pf:.2f}  DD={dd:.2f}%  rd_ratio={rd:.2f}", flush=True)
    print(f"  AvgWin={metrics.avg_win_pct:+.2f}%  AvgLoss={metrics.avg_loss_pct:+.2f}%  R:R={metrics.avg_r_r:.2f}", flush=True)
    print(f"  B&H={metrics.buy_hold_pct:+.2f}%  Alpha={pnl - metrics.buy_hold_pct:+.2f}pp", flush=True)
    print(f"  VERDICT: {v}", flush=True)
    print(f"  By type:", flush=True)
    for et, stats in sorted(by_type.items(), key=lambda x: -x[1]["pnl"]):
        wr_t = stats["wins"]/stats["count"]*100 if stats["count"] > 0 else 0
        print(f"    {et}: {stats['count']}T PnL={stats['pnl']:+.2f}% WR={wr_t:.1f}%", flush=True)
    print(f"  Diag: cooldown_skip={diag.get('cooldown_skip',0)} max_concurrent={diag.get('max_concurrent_hit',0)}", flush=True)

print(f"\n{'='*80}", flush=True)
print(f"Total time: {time.time()-t0:.1f}s", flush=True)
