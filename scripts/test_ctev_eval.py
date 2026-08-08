import sys, os, traceback as tb
sys.path.insert(0, '/home/z/my-project/trade-signal')

from strategy import evaluate_long, SignalType
from backtest import _apply_costs
print('Import OK')

from strategy_profiles import StrategyProfile
profile = StrategyProfile(
    name='CTEV_GRID', timeframes=('1h',), description='Grid test',
    sl_atr_mult=2.12, tp_atr_mult=8.5,
)
print('Profile OK')

import pandas as pd
df = pd.read_csv('/home/z/my-project/trade-signal/download/btc_1h_cache.csv', index_col=0, parse_dates=True)
print(f'Data OK: {len(df)} rows')

row = df.iloc[500]
try:
    sig = evaluate_long(row, profile=profile)
    print(f'Signal OK: {sig}')
except Exception as e:
    tb.print_exc()
