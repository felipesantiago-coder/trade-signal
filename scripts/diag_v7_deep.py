#!/usr/bin/env python3
"""Diag profundo v7 — entender combinacao fatal RSI+Pullback + EMA proximity fix."""
import sys
sys.path.insert(0, ".")

import ccxt
import pandas as pd
import numpy as np
from indicators import compute_indicators
from strategy import evaluate_long, evaluate_short
from strategy_profiles import PROFILE_STANDARD
import strategy as strat

print("Baixando dados BTC/USDT 1h...")
exchange = ccxt.binance({"enableRateLimit": True})
now = pd.Timestamp.now(tz="UTC")
since = int((now - pd.Timedelta(days=730)).timestamp() * 1000)
all_ohlcv = []
while since < int(now.timestamp() * 1000):
    batch = exchange.fetch_ohlcv("BTC/USDT", "1h", since=since, limit=1000)
    if not batch:
        break
    all_ohlcv.extend(batch)
    since = batch[-1][0] + 1
    if len(batch) < 1000:
        break
df = pd.DataFrame(all_ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
df = df.set_index("ts").drop_duplicates()
print(f"Candles: {len(df)}")
df_ind = compute_indicators(df)

# ── 1. RSI 35-50 em uptrend: o que mais passa? ──
print("\n" + "="*60)
print("RSI 35-50 + trending_up + EMA alignment")
print("="*60)
mask = (
    (df_ind['rsi'] >= 35) & (df_ind['rsi'] <= 50) &
    (df_ind['regime'] == 'trending_up') &
    (df_ind['close'] > df_ind['ema50']) & (df_ind['ema50'] > df_ind['ema200']) &
    (df_ind['ema50_slope'] > -0.3)
)
df_rsi_up = df_ind[mask]
print(f"Total: {len(df_rsi_up)}")
if len(df_rsi_up) > 0:
    print(f"  ADX>=25: {(df_rsi_up['adx']>=25).sum()} | ADX>=30: {(df_rsi_up['adx']>=30).sum()} | ADX>=35: {(df_rsi_up['adx']>=35).sum()}")
    print(f"  ema20_touched: {int(df_rsi_up['ema20_touched'].sum())}")
    print(f"  ema50_touched: {int(df_rsi_up['ema50_touched'].sum())}")
    print(f"  fib_dir==1: {(df_rsi_up['fib_direction']==1).sum()}")
    dist20 = (df_rsi_up['close'] - df_rsi_up['ema20']).abs() / df_rsi_up['ema20'] * 100
    print(f"  Distancia EMA20: mean={dist20.mean():.3f}% median={dist20.median():.3f}%")
    dist50 = (df_rsi_up['close'] - df_rsi_up['ema50']).abs() / df_rsi_up['ema50'] * 100
    print(f"  Distancia EMA50: mean={dist50.mean():.3f}% median={dist50.median():.3f}%")

# ── 2. INVERSA: quando ha pullback, qual RSI? ──
print("\n" + "="*60)
print("Pullback real (ema20_touched=True ou fib_dir==1) + uptrend")
print("="*60)
mask2 = (
    (df_ind['regime'] == 'trending_up') &
    (df_ind['close'] > df_ind['ema50']) & (df_ind['ema50'] > df_ind['ema200']) &
    (df_ind['ema50_slope'] > -0.3) &
    (df_ind['adx'] >= 25) &
    ((df_ind['ema20_touched'] == True) | (df_ind['fib_direction'] == 1))
)
df_pull = df_ind[mask2]
print(f"Total pullbacks: {len(df_pull)}")
if len(df_pull) > 0:
    r = df_pull['rsi']
    print(f"  RSI: mean={r.mean():.1f} median={r.median():.1f} std={r.std():.1f}")
    print(f"  RSI: p5={r.quantile(.05):.1f} p10={r.quantile(.10):.1f} p25={r.quantile(.25):.1f}")
    print(f"  RSI: p75={r.quantile(.75):.1f} p90={r.quantile(.90):.1f}")
    for lo, hi in [(30,40),(35,45),(35,50),(40,50),(40,55),(40,60),(45,55),(45,60),(50,60),(50,65),(55,65),(55,70),(60,70)]:
        n = ((r >= lo) & (r <= hi)).sum()
        if n > 0:
            print(f"    RSI {lo}-{hi}: {n}")

# ── 3. RSI em zona proxima a EMA20 ──
print("\n" + "="*60)
print("RSI quando close esta dentro de X% da EMA20 (uptrend)")
print("="*60)
mask_trend = (
    (df_ind['regime'] == 'trending_up') &
    (df_ind['close'] > df_ind['ema50']) & (df_ind['ema50'] > df_ind['ema200']) &
    (df_ind['ema50_slope'] > -0.3) & (df_ind['adx'] >= 25)
)
df_trend = df_ind[mask_trend]
dist_ema20 = (df_trend['close'] - df_trend['ema20']).abs() / df_trend['ema20'] * 100
for pct in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]:
    near = df_trend[dist_ema20 <= pct]
    if len(near) > 0:
        r = near['rsi']
        print(f"  EMA20 +/-{pct}%: {len(near)} candles | RSI mean={r.mean():.1f} median={r.median():.1f} range=[{r.min():.0f},{r.max():.0f}]")
        for lo, hi in [(35,50),(40,55),(45,60),(50,65)]:
            n = ((r >= lo) & (r <= hi)).sum()
            if n > 0:
                print(f"    RSI {lo}-{hi}: {n} ({100*n/len(near):.1f}%)")

print("\nDone.")
