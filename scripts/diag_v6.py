import logging, sys, os
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pandas as pd
from backtest import fetch_historical_ohlcv
from indicators import compute_indicators
from strategy import evaluate_long, evaluate_short
from strategy_profiles import get_profile

df = fetch_historical_ohlcv('BTC/USDT', '1h', 730)
df_ind = compute_indicators(df, timeframe='1h')
df_clean = df_ind.dropna(subset=['ema20','ema50','ema200','rsi','atr','atr_percentile','macd','macd_signal','macd_hist','adx','plus_di','minus_di','regime']).copy()

profile = get_profile('1h')
print(f'Candles limpos: {len(df_clean)}')
print(f'Profile: {profile.name}, ADX>={profile.adx_min}, TRANS={profile.allow_transition}')
print(f'Regimes: {dict(df_clean["regime"].value_counts())}')

trending = df_clean[df_clean['regime'].isin(['trending_up','trending_down'])]
print(f'Candles trending: {len(trending)}')

adx_ok = trending[trending['adx'] >= profile.adx_min]
print(f'  Com ADX >= {profile.adx_min}: {len(adx_ok)}')

long_ema = adx_ok[(adx_ok['close'] > adx_ok['ema50']) & (adx_ok['ema50'] > adx_ok['ema200'])]
short_ema = adx_ok[(adx_ok['close'] < adx_ok['ema50']) & (adx_ok['ema50'] < adx_ok['ema200'])]
print(f'  LONG EMA alignment: {len(long_ema)}')
print(f'  SHORT EMA alignment: {len(short_ema)}')

# LONG funnel
if len(long_ema) > 0:
    slope_ok = long_ema[long_ema['ema50_slope'] > profile.ema50_slope_min]
    print(f'    LONG slope > {profile.ema50_slope_min}: {len(slope_ok)}')
    di_ok = slope_ok[slope_ok['plus_di'] > slope_ok['minus_di']]
    print(f'    LONG +DI > -DI: {len(di_ok)}')
    rsi_ok = di_ok[(di_ok['rsi'] >= profile.rsi_long_min) & (di_ok['rsi'] <= profile.rsi_long_max)]
    print(f'    LONG RSI {profile.rsi_long_min}-{profile.rsi_long_max}: {len(rsi_ok)}')
    sigs = sum(1 for _, r in rsi_ok.iterrows() if evaluate_long(r, profile=profile) is not None)
    print(f'    LONG signals (com pullback): {sigs}')

# SHORT funnel
if len(short_ema) > 0:
    slope_ok_s = short_ema[short_ema['ema50_slope'] < -profile.ema50_slope_min]
    print(f'    SHORT slope < {-profile.ema50_slope_min}: {len(slope_ok_s)}')
    di_ok_s = slope_ok_s[slope_ok_s['minus_di'] > slope_ok_s['plus_di']]
    print(f'    SHORT -DI > +DI: {len(di_ok_s)}')
    rsi_ok_s = di_ok_s[(di_ok_s['rsi'] >= profile.rsi_short_min) & (di_ok_s['rsi'] <= profile.rsi_short_max)]
    print(f'    SHORT RSI {profile.rsi_short_min}-{profile.rsi_short_max}: {len(rsi_ok_s)}')
    sigs_s = sum(1 for _, r in rsi_ok_s.iterrows() if evaluate_short(r, profile=profile) is not None)
    print(f'    SHORT signals (com pullback): {sigs_s}')
