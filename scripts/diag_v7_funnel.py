#!/usr/bin/env python3
"""
diag_v7_funnel.py — Diagnostico de funil: qual filtro elimina todos os sinais?
Rodar: python scripts/diag_v7_funnel.py
"""
import sys
sys.path.insert(0, ".")

import ccxt
import pandas as pd
import numpy as np
from indicators import compute_indicators
from strategy import evaluate_long, evaluate_short
from strategy_profiles import PROFILE_STANDARD

# ── 1. Download dados ──
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
print(f"Candles: {len(df)} | {df.index[0]} a {df.index[-1]}")

# ── 2. Calcular indicadores ──
print("Calculando indicadores...")
df_ind = compute_indicators(df)
print(f"Colunas: {list(df_ind.columns)}")
for col in ['obv_trend', 'stoch_rsi_k', 'stoch_rsi_d', 'bbwp', 'macd_hist', 'adx', 'plus_di', 'minus_di', 'ema50_slope', 'ema20_touched', 'ema50_touched', 'fib_direction', 'fib_proximity']:
    has = col in df_ind.columns
    if has:
        nan_pct = 100 * df_ind[col].isna().sum() / len(df_ind)
        print(f"  {col}: EXISTE ({nan_pct:.1f}% NaN)")
    else:
        print(f"  {col}: *** AUSENTE ***")

# ── 3. Funil de filtros LONG ──
print("\n" + "="*60)
print("FUNIL DE FILTROS (LONG)")
print("="*60)

df_clean = df_ind.dropna(subset=['adx', 'rsi', 'ema50', 'ema200', 'atr_percentile', 'close', 'ema50_slope']).copy()
total = len(df_clean)
print(f"Total candles (pos indicadores): {total}")

# Step 1
mask = df_clean['regime'] == 'trending_up'
n = mask.sum()
print(f"\n1. Regime trending_up: {n} ({100*n/total:.1f}%)")
df_s = df_clean[mask].copy()

# Step 2
mask = df_s['adx'] >= 35.0
n = mask.sum()
print(f"2. ADX >= 35: {n} ({100*n/len(df_s):.1f}%)")
df_s = df_s[mask].copy()

# Step 3
mask = (df_s['close'] > df_s['ema50']) & (df_s['ema50'] > df_s['ema200'])
n = mask.sum()
print(f"3. close > EMA50 > EMA200: {n} ({100*n/len(df_s):.1f}%)")
df_s = df_s[mask].copy()

# Step 4
mask = df_s['ema50_slope'] > -0.3
n = mask.sum()
print(f"4. EMA50 slope > -0.3: {n} ({100*n/len(df_s):.1f}%)")
df_s = df_s[mask].copy()

# Step 5
if 'plus_di' in df_s.columns and 'minus_di' in df_s.columns:
    mask = df_s['plus_di'] > df_s['minus_di']
    n = mask.sum()
    print(f"5. +DI > -DI: {n} ({100*n/len(df_s):.1f}%)")
    df_s = df_s[mask].copy()
else:
    print(f"5. +DI > -DI: COLUNAS AUSENTES")

# Step 6
if 'macd_hist' in df_s.columns:
    mask = df_s['macd_hist'] > 0
    n = mask.sum()
    print(f"6. MACD hist > 0: {n} ({100*n/len(df_s):.1f}%)")
    df_s = df_s[mask].copy()
else:
    print(f"6. MACD hist > 0: COLUNA AUSENTE")

# Step 7
if 'obv_trend' in df_s.columns:
    mask = df_s['obv_trend'] >= 1
    n = mask.sum()
    print(f"7. OBV trend >= 1: {n} ({100*n/len(df_s):.1f}%)")
    print(f"   Distribution: {df_s['obv_trend'].value_counts().sort_index().to_dict()}")
    df_s = df_s[mask].copy()
else:
    print(f"7. OBV trend >= 1: *** COLUNA AUSENTE ***")

# Step 8
if 'stoch_rsi_k' in df_s.columns and 'stoch_rsi_d' in df_s.columns:
    mask = df_s['stoch_rsi_k'] > df_s['stoch_rsi_d']
    n = mask.sum()
    print(f"8. Stoch RSI K > D: {n} ({100*n/len(df_s):.1f}%)")
    df_s = df_s[mask].copy()
else:
    print(f"8. Stoch RSI K > D: COLUNAS AUSENTES")

# Step 9 - RSI
mask = (df_s['rsi'] >= 35.0) & (df_s['rsi'] <= 50.0)
n = mask.sum()
print(f"9. RSI 35-50: {n} ({100*n/len(df_s):.1f}%)")
df_s = df_s[mask].copy()

# Step 10 - Pullback
if 'fib_direction' in df_s.columns:
    fib_ok = (df_s['fib_direction'] == 1).sum()
    print(f"10. fib_direction == 1: {fib_ok}")
if 'ema20_touched' in df_s.columns:
    print(f"    ema20_touched=True: {int(df_s['ema20_touched'].sum())}")
if 'ema50_touched' in df_s.columns:
    print(f"    ema50_touched=True: {int(df_s['ema50_touched'].sum())}")

# ── 4. Testar combos reais ──
print("\n" + "="*60)
print("SINAIS REAIS COM DIFERENTES COMBOS")
print("="*60)

import strategy as strat

orig = {
    'macd': strat.MACD_HIST_FILTER,
    'obv': strat.OBV_TREND_FILTER,
    'stoch': strat.STOCH_RSI_FILTER,
    'di': strat.DI_DIRECTION_FILTER,
    'bbwp': strat.BBWP_SQUEEZE_BONUS,
    'adx': strat.ADX_MIN,
    'allow_trans': strat.ALLOW_TRANSITION,
    'rsi_l_min': strat.RSI_LONG_MIN,
    'rsi_l_max': strat.RSI_LONG_MAX,
    'rsi_s_min': strat.RSI_SHORT_MIN,
    'rsi_s_max': strat.RSI_SHORT_MAX,
    'slope': strat.EMA50_SLOPE_MIN,
}

def count_all(df, label):
    longs = shorts = 0
    for idx, row in df.iterrows():
        if evaluate_long(row, profile=PROFILE_STANDARD):
            longs += 1
        if evaluate_short(row, profile=PROFILE_STANDARD):
            shorts += 1
    print(f"  {label}: {longs}L + {shorts}S = {longs+shorts} total")
    return longs + shorts

print("\n[A] v7.0 ATUAL (tudo ON):")
count_all(df_ind, "STANDARD v7.0")

print("\n[B] Desativar filtros novos, manter ADX 35 + no transition:")
strat.MACD_HIST_FILTER = False
strat.OBV_TREND_FILTER = False
strat.STOCH_RSI_FILTER = False
strat.DI_DIRECTION_FILTER = False
strat.BBWP_SQUEEZE_BONUS = False
count_all(df_ind, "Sem filtros novos")

print("\n[C] Como B + ADX 30:")
strat.ADX_MIN = 30.0
count_all(df_ind, "ADX 30")

print("\n[D] Como C + ALLOW_TRANSITION=True:")
strat.ALLOW_TRANSITION = True
count_all(df_ind, "ADX 30 + transition")

print("\n[E] Como D + RSI LONG 30-52:")
strat.RSI_LONG_MIN = 30.0
strat.RSI_LONG_MAX = 52.0
strat.RSI_SHORT_MIN = 48.0
strat.RSI_SHORT_MAX = 73.0
count_all(df_ind, "RSI antigo 30-52/48-73")

print("\n[F] Como E + SLOPE -1.0:")
strat.EMA50_SLOPE_MIN = -1.0
count_all(df_ind, "Slope -1.0")

print("\n[G] Como F + ADX 25:")
strat.ADX_MIN = 25.0
count_all(df_ind, "ADX 25")

# Ativar filtros 1 por vez sobre base F
print("\n[H] Base F + cada filtro individual:")
strat.MACD_HIST_FILTER = True
count_all(df_ind, "+ MACD HIST")
strat.MACD_HIST_FILTER = False

strat.OBV_TREND_FILTER = True
count_all(df_ind, "+ OBV TREND")
strat.OBV_TREND_FILTER = False

strat.STOCH_RSI_FILTER = True
count_all(df_ind, "+ STOCH RSI")
strat.STOCH_RSI_FILTER = False

strat.DI_DIRECTION_FILTER = True
count_all(df_ind, "+ DI DIR")
strat.DI_DIRECTION_FILTER = False

# Restore
for k, v in orig.items():
    setattr(strat, k.upper() if k in ('adx','allow_trans') else k, v)

print("\nDone.")
