"""Baixa dados 15m e salva cache para otimizacao."""
import sys, os, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from backtest import fetch_historical_ohlcv
from indicators import compute_indicators

SYMBOL = 'BTC/USDT'; TIMEFRAME = '15m'; DAYS = 365
DATA_CACHE = '/tmp/ctev_15m_data.pkl'

print(f'Baixando {DAYS}d de dados {SYMBOL} {TIMEFRAME}...')
df = fetch_historical_ohlcv(SYMBOL, TIMEFRAME, DAYS)
print(f'{len(df)} candles baixados. Calculando indicadores...')
df = compute_indicators(df, timeframe=TIMEFRAME)
crit = ['ema20','ema50','ema200','rsi','atr','atr_percentile',
        'adx','plus_di','minus_di','regime','bb_lower','bb_upper','bb_width','bb_squeeze_pct']
df = df.dropna(subset=crit).copy()
print(f'{len(df)} candles limpos. Salvando cache...')
with open(DATA_CACHE, 'wb') as f:
    pickle.dump({'df_clean': df}, f)
print(f'Cache salvo em {DATA_CACHE}')
print(f'Periodo: {df.index[0]} a {df.index[-1]}')
print(f'Regimes: {df["regime"].value_counts().to_dict()}')
