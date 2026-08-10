#!/usr/bin/env python3
"""Teste rapido v7.1 — verifica se sinais sao gerados."""
import sys
sys.path.insert(0, ".")

import ccxt, pandas as pd
from indicators import compute_indicators
from strategy import evaluate_long, evaluate_short
from strategy_profiles import PROFILE_STANDARD

print("Baixando dados...")
exchange = ccxt.binance({"enableRateLimit": True})
now = pd.Timestamp.now(tz="UTC")
since = int((now - pd.Timedelta(days=730)).timestamp() * 1000)
all_ohlcv = []
while since < int(now.timestamp() * 1000):
    batch = exchange.fetch_ohlcv("BTC/USDT", "1h", since=since, limit=1000)
    if not batch: break
    all_ohlcv.extend(batch)
    since = batch[-1][0] + 1
    if len(batch) < 1000: break
df = pd.DataFrame(all_ohlcv, columns=["ts","open","high","low","close","volume"])
df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
df = df.set_index("ts").drop_duplicates()
print(f"Candles: {len(df)}")

df_ind = compute_indicators(df)

# Count signals
longs = shorts = 0
long_details = []
short_details = []

for idx, row in df_ind.iterrows():
    sig_l = evaluate_long(row, profile=PROFILE_STANDARD)
    if sig_l:
        longs += 1
        long_details.append(f"  {sig_l.timestamp} | RSI={sig_l.rsi:.1f} ADX={sig_l.adx:.1f} pullback={sig_l.pullback_type}")
    sig_s = evaluate_short(row, profile=PROFILE_STANDARD)
    if sig_s:
        shorts += 1
        short_details.append(f"  {sig_s.timestamp} | RSI={sig_s.rsi:.1f} ADX={sig_s.adx:.1f} pullback={sig_s.pullback_type}")

total = longs + shorts
print(f"\nSINAIS v7.1: {total} ({longs}L + {shorts}S)")

if longs > 0:
    print(f"\nPrimeiros 10 LONGs:")
    for d in long_details[:10]:
        print(d)
    if longs > 10:
        print(f"  ... +{longs-10} mais")

if shorts > 0:
    print(f"\nPrimeiros 10 SHORTs:")
    for d in short_details[:10]:
        print(d)
    if shorts > 10:
        print(f"  ... +{shorts-10} mais")

# Pullback type distribution
from collections import Counter
pb_types_l = Counter()
pb_types_s = Counter()
for idx, row in df_ind.iterrows():
    sig_l = evaluate_long(row, profile=PROFILE_STANDARD)
    if sig_l: pb_types_l[sig_l.pullback_type] += 1
    sig_s = evaluate_short(row, profile=PROFILE_STANDARD)
    if sig_s: pb_types_s[sig_s.pullback_type] += 1

print(f"\nPullback types LONG: {dict(pb_types_l)}")
print(f"Pullback types SHORT: {dict(pb_types_s)}")
print("\nOK.")
