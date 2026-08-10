import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from indicators import compute_indicators
from strategy import evaluate_long, evaluate_short, SignalType
from strategy_profiles import StrategyProfile
from backtest import _apply_costs, DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS

FEE, SPREAD, SLIP = DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS

# Load
df = pd.read_csv('/home/z/my-project/trade-signal/download/btc_1h_cache.csv')
df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
df.set_index('datetime', inplace=True)
df_ind = compute_indicators(df, timeframe='1h')
df_ind = df_ind.loc[:, ~df_ind.columns.duplicated()]
need = ['ema20','ema50','ema200','rsi','atr','atr_percentile','macd','macd_signal','macd_hist','adx','plus_di','minus_di','regime']
df_clean = df_ind.dropna(subset=need).copy()
n = len(df_clean)
close_a = df_clean['close'].values.astype(np.float64)
low_a = df_clean['low'].values.astype(np.float64)
high_a = df_clean['high'].values.astype(np.float64)
rsi_a = df_clean['rsi'].values.astype(np.float64)


def simulate_v9(signals, sl_m, tp_m, max_bars=168, 
                partial_pct=0.50, post_tp1_trail=1.5, 
                rsi_exhaustion=True, risk_pct=1.0):
    """
    v9.0 simulation: dynamic trailing after TP1.
    
    After TP1 hit:
      - Close partial_pct at TP
      - Activate trailing stop at post_tp1_trail * ATR from high water mark
      - SL floor = TP - 1.0 * ATR (never worse than this)
    """
    pnls = []
    last_end = -1
    for s in signals:
        si = s['i']
        if si <= last_end: continue
        is_long = s['is_long']; ep = s['ep']; atr = s['atr']
        sl = ep - sl_m * atr if is_long else ep + sl_m * atr
        tp = ep + tp_m * atr if is_long else ep - tp_m * atr
        
        current_sl = sl
        partial_filled = False
        trailing_active = False
        highest_fav = ep
        exit_price = None
        trade_bars = 0
        
        for j in range(si + 1, min(si + max_bars, n)):
            bars = j - si
            fc, fl, fh, fr = close_a[j], low_a[j], high_a[j], rsi_a[j]
            
            # Track high water mark
            if is_long:
                highest_fav = max(highest_fav, fh)
            else:
                highest_fav = min(highest_fav, fl)
            
            # RSI exhaustion
            if rsi_exhaustion and bars >= 24:
                profit = (fc - ep) if is_long else (ep - fc)
                if profit > 0 and ((is_long and fr > 80) or (not is_long and fr < 20)):
                    exit_price = fc; trade_bars = bars; break
            
            # Check SL/TP
            if is_long:
                tp_hit = fh >= tp
                sl_hit = fl <= current_sl
            else:
                tp_hit = fl <= tp
                sl_hit = fh >= current_sl
            
            if sl_hit and not tp_hit:
                exit_price = current_sl; trade_bars = bars; break
            elif tp_hit and not sl_hit:
                if not partial_filled:
                    # First TP: partial close + activate dynamic trailing
                    partial_filled = True
                    trailing_active = True
                    # SL floor at TP - 1.0*ATR
                    if is_long:
                        floor_sl = tp - atr * 1.0
                        new_trail = highest_fav - atr * post_tp1_trail
                        current_sl = max(floor_sl, new_trail, sl)  # ratchet only
                    else:
                        floor_sl = tp + atr * 1.0
                        new_trail = highest_fav + atr * post_tp1_trail
                        current_sl = min(floor_sl, new_trail, sl)
                else:
                    # Second TP hit on remaining position
                    exit_price = tp; trade_bars = bars; break
            elif tp_hit and sl_hit:
                exit_price = current_sl; trade_bars = bars; break
            
            # Dynamic trailing after TP1
            if trailing_active:
                if is_long:
                    new_trail = highest_fav - atr * post_tp1_trail
                    floor_sl = tp - atr * 1.0
                    if new_trail > current_sl:
                        current_sl = max(new_trail, floor_sl)
                else:
                    new_trail = highest_fav + atr * post_tp1_trail
                    floor_sl = tp + atr * 1.0
                    if new_trail < current_sl:
                        current_sl = min(new_trail, floor_sl)
        
        if exit_price is None:
            trade_bars = min(max_bars, n - si - 1)
            exit_price = close_a[si + trade_bars]
        
        last_end = si + trade_bars
        
        # PnL calculation with partial TP
        if partial_filled:
            if is_long:
                partial_pnl = (tp - ep) / ep * 100
            else:
                partial_pnl = (ep - tp) / ep * 100
            _, adj_exit, _ = _apply_costs(ep, exit_price, is_long, FEE, SPREAD, SLIP)
            if is_long:
                remaining_pnl = (adj_exit - ep) / ep * 100
            else:
                remaining_pnl = (ep - adj_exit) / ep * 100
            pnl = partial_pct * partial_pnl + (1 - partial_pct) * remaining_pnl
        else:
            _, adj_exit, _ = _apply_costs(ep, exit_price, is_long, FEE, SPREAD, SLIP)
            if is_long:
                pnl = (adj_exit - ep) / ep * 100
            else:
                pnl = (ep - adj_exit) / ep * 100
        pnls.append(pnl)
    
    if not pnls: return None
    a = np.array(pnls)
    w = a[a>0]; lo = a[a<=0]
    wr = len(w)/len(a)*100
    gp = w.sum() if len(w) > 0 else 0
    gl = abs(lo.sum()) if len(lo) > 0 else 0.001
    pf = gp/gl
    cum = np.cumsum(a)
    dd = abs(min((cum - np.maximum.accumulate(cum)).min(), 0))
    sh = np.mean(a)/max(np.std(a),0.001)*(365**0.5)
    return {'trades':len(a),'wr':round(wr,1),'pf':round(pf,2),'pnl':round(a.sum(),2),'dd':round(dd,2),'sh':round(sh,2),'aw':round(w.mean(),2) if len(w)>0 else 0,'al':round(lo.mean(),2) if len(lo)>0 else 0,'rr':round(abs(w.mean()/lo.mean()),2) if len(lo)>0 and lo.mean()!=0 else 0}


# Generate signals
profile = StrategyProfile(name='P', timeframes=('1h',), description='', adx_min=32.0, allow_transition=False, rsi_long_min=45.0, rsi_long_max=65.0, rsi_short_min=35.0, rsi_short_max=55.0, fib_tolerance_pct=0.025, ema50_slope_min=-0.5, ema20_proximity_pct=0.005, ema50_proximity_pct=0.008, volume_confirm=False, atr_pct_min=0.10, atr_pct_max=0.90, sl_atr_mult=2.0, tp_atr_mult=10.0, max_bars_held=168)

signals = []
i = 0
while i < n:
    row = df_clean.iloc[i]
    regime = str(row['regime'])
    if regime in ('ranging', 'volatile'): i+=1; continue
    atr_pct = float(row['atr_percentile'])
    if atr_pct < 0.10 or atr_pct > 0.90: i+=1; continue
    sig = evaluate_long(row, profile=profile)
    if sig is None: sig = evaluate_short(row, profile=profile)
    if sig is not None:
        signals.append({'i':i,'is_long':sig.type==SignalType.LONG,'ep':sig.entry_price,'atr':sig.atr})
        i += 1
    else: i += 1
print(f'Signals: {len(signals)}')

print(f'{"Config":<40} {"Tr":>4} {"WR":>5} {"PF":>5} {"PnL":>8} {"DD":>6} {"Sh":>5} {"AW":>6} {"AL":>6} {"R:R":>5}')
print('-' * 105)

configs = [
    # (name, sl, tp, partial_pct, post_tp1_trail, rsi_ex)
    ('v8.0 PRODUCTION (fixed SL)', 2.8, 5.5, 0.50, 0.0, True),
    ('v9 trail=1.0, pp=50%', 2.8, 5.5, 0.50, 1.0, True),
    ('v9 trail=1.5, pp=50%', 2.8, 5.5, 0.50, 1.5, True),
    ('v9 trail=2.0, pp=50%', 2.8, 5.5, 0.50, 2.0, True),
    ('v9 trail=2.5, pp=50%', 2.8, 5.5, 0.50, 2.5, True),
    ('v9 trail=1.5, pp=40%', 2.8, 5.5, 0.40, 1.5, True),
    ('v9 trail=2.0, pp=40%', 2.8, 5.5, 0.40, 2.0, True),
    ('v9 trail=1.5, pp=30%', 2.8, 5.5, 0.30, 1.5, True),
    ('v9 trail=2.0, pp=30%', 2.8, 5.5, 0.30, 2.0, True),
    # No RSI exhaustion variants
    ('v9 trail=1.5, pp=50% NOex', 2.8, 5.5, 0.50, 1.5, False),
    ('v9 trail=2.0, pp=50% NOex', 2.8, 5.5, 0.50, 2.0, False),
    ('v9 trail=2.5, pp=50% NOex', 2.8, 5.5, 0.50, 2.5, False),
    # Higher TP with trailing
    ('v9 TP7, trail=1.5, pp=50%', 2.8, 7.0, 0.50, 1.5, True),
    ('v9 TP7, trail=2.0, pp=50%', 2.8, 7.0, 0.50, 2.0, True),
    ('v9 TP8, trail=2.0, pp=50%', 2.8, 8.0, 0.50, 2.0, True),
    ('v9 TP6, trail=1.5, pp=50%', 2.8, 6.0, 0.50, 1.5, True),
    # Different SL
    ('v9 SL3.0, trail=2.0, pp=50%', 3.0, 5.5, 0.50, 2.0, True),
    ('v9 SL2.5, trail=1.5, pp=50%', 2.5, 5.5, 0.50, 1.5, True),
    ('v9 SL3.0, TP6, trail=2.0, pp=50%', 3.0, 6.0, 0.50, 2.0, True),
]

for name, sl, tp, pp, trail, rsi_ex in configs:
    r = simulate_v9(signals, sl, tp, 168, partial_pct=pp, post_tp1_trail=trail, rsi_exhaustion=rsi_ex)
    if r:
        print(f'{name:<40} {r["trades"]:>4} {r["wr"]:>5.1f} {r["pf"]:>5.2f} {r["pnl"]:>+8.2f} {r["dd"]:>6.2f} {r["sh"]:>5.2f} {r["aw"]:>+6.2f} {r["al"]:>+6.2f} {r["rr"]:>5.2f}')
    else:
        print(f'{name:<40} NO TRADES')
