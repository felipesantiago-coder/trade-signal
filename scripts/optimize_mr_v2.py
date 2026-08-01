"""
Optimizacao MR v2: pre-computa sinais, depois testa SL/TP.
Rodagem rapida (< 60s).
"""
import logging, sys, os, json
logging.basicConfig(level=logging.ERROR)

import numpy as np
import pandas as pd

os.environ.setdefault('EXCHANGE_ID', 'bybit')
sys.path.insert(0, '/home/z/my-project/trade-signal')

from indicators import compute_indicators
from regime_engine import classify_regimes_v2
from strategy_regime import evaluate_mean_reversion_long, evaluate_mean_reversion_short
from backtest import (
    TradeResult, _apply_costs, fetch_historical_ohlcv,
    DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS,
)

BASE_SL = 2.12

def precompute_signals(df_ind):
    """Find all RANGING bars that could generate a signal with relaxed params."""
    candidates = []
    for i in range(len(df_ind)):
        row = df_ind.iloc[i]
        rv2 = str(row.get('regime_v2', ''))
        conf = float(row.get('regime_confidence', 0.5))
        if rv2 != 'RANGING' or conf < 0.15:
            continue
        atr_pct = float(row.get('atr_percentile', 0.5))
        if atr_pct < 0.08 or atr_pct > 0.90:
            continue
        close = float(row['close'])
        rsi = float(row['rsi'])
        bb_lower = float(row['bb_lower'])
        bb_upper = float(row['bb_upper'])
        bb_middle = float(row['bb_middle'])
        bb_sq = float(row.get('bb_squeeze_pct', 0.5))
        bb_range = bb_upper - bb_lower
        if bb_range <= 0:
            continue
        bb_pos = (close - bb_lower) / bb_range
        slope = abs(float(row.get('ema50_slope', 0)))
        low = float(row['low'])
        high = float(row['high'])
        atr = float(row['atr'])
        
        # Long candidate: RSI low + near BB lower
        long_rsi_ok = 15 <= rsi <= 50
        long_bb_ok = (low <= bb_lower * 1.02) or (bb_pos < 0.50)
        long_slope_ok = slope < 3.0
        long_bb_sq_ok = 0.08 <= bb_sq <= 0.92
        
        # Short candidate: RSI high + near BB upper  
        short_rsi_ok = 48 <= rsi <= 90
        short_bb_ok = (high >= bb_upper * 0.98) or (bb_pos > 0.50)
        short_slope_ok = slope < 3.0
        short_bb_sq_ok = 0.08 <= bb_sq <= 0.92
        
        if long_rsi_ok and long_bb_ok and long_slope_ok and long_bb_sq_ok:
            candidates.append({'idx': i, 'side': 'LONG', 'rsi': rsi, 'bb_pos': bb_pos, 
                           'bb_lower': bb_lower, 'bb_middle': bb_middle, 'bb_upper': bb_upper,
                           'atr': atr, 'atr_pct': atr_pct, 'close': close,
                           'slope': slope, 'bb_sq': bb_sq})
        if short_rsi_ok and short_bb_ok and short_slope_ok and short_bb_sq_ok:
            candidates.append({'idx': i, 'side': 'SHORT', 'rsi': rsi, 'bb_pos': bb_pos,
                           'bb_lower': bb_lower, 'bb_middle': bb_middle, 'bb_upper': bb_upper,
                           'atr': atr, 'atr_pct': atr_pct, 'close': close,
                           'slope': slope, 'bb_sq': bb_sq})
    return candidates, df_ind


def sim_with_filters(candidates, df_ind, rsi_l_range, rsi_s_range, sl_f, tp_f, max_bars=72):
    """Simulate with specific RSI filters and SL/TP."""
    trades = []
    used_idx = set()
    for c in candidates:
        if c['idx'] in used_idx:
            continue
        # Apply RSI filter
        if c['side'] == 'LONG':
            if not (rsi_l_range[0] <= c['rsi'] <= rsi_l_range[1]):
                continue
        else:
            if not (rsi_s_range[0] <= c['rsi'] <= rsi_s_range[1]):
                continue
        
        close = c['close']
        atr = c['atr']
        is_long = c['side'] == 'LONG'
        
        # Compute SL/TP (same logic as strategy_regime.py)
        sl_mult = BASE_SL * sl_f
        tp_mult = BASE_SL * tp_f
        if is_long:
            sl = min(close - sl_mult * atr, c['bb_lower'] - 0.5 * atr)
            tp_bb = c['bb_middle']
            tp_atr = close + tp_mult * atr
            tp = max(tp_bb, close + 1.0 * atr)
            tp = min(tp, tp_atr)
        else:
            sl = max(close + sl_mult * atr, c['bb_upper'] + 0.5 * atr)
            tp_bb = c['bb_middle']
            tp_atr = close - tp_mult * atr
            tp = min(tp_bb, close - 1.0 * atr)
            tp = max(tp, tp_atr)
        
        if sl <= 0 or (is_long and sl >= close) or (not is_long and sl <= close):
            continue
        
        risk = abs(close - sl)
        reward = abs(tp - close)
        if risk <= 0 or reward / risk < 1.2:
            continue
        
        # Simulate
        i = c['idx']
        n = len(df_ind)
        exit_price, exit_reason, bars = None, None, 0
        for j in range(i + 1, min(i + max_bars, n)):
            f = df_ind.iloc[j]
            bars = j - i
            if is_long:
                if float(f['low']) <= sl:
                    exit_price, exit_reason = sl, 'sl'; break
                if float(f['high']) >= tp:
                    exit_price, exit_reason = tp, 'tp'; break
            else:
                if float(f['high']) >= sl:
                    exit_price, exit_reason = sl, 'sl'; break
                if float(f['low']) <= tp:
                    exit_price, exit_reason = tp, 'tp'; break
        if exit_price is None:
            lj = min(i + max_bars, n) - 1
            exit_price = float(df_ind.iloc[lj]['close'])
            exit_reason, bars = 'timeout', lj - i
        
        _, adj_exit, _ = _apply_costs(close, exit_price, is_long, DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS)
        pnl = (adj_exit - close) / close * 100 if is_long else (close - adj_exit) / close * 100
        
        trades.append(TradeResult(
            entry_ts=df_ind.index[i], exit_ts=df_ind.index[min(i + bars, n - 1)],
            type=c['side'], entry_price=close, exit_price=exit_price,
            stop_loss=sl, take_profit=tp, atr=atr, rsi=c['rsi'],
            pnl_pct=round(pnl, 4), pnl_abs=round(exit_price - close, 2),
            bars_held=bars, exit_reason=exit_reason, atr_percentile=c['atr_pct'],
        ))
        used_idx.add(c['idx'])
    return trades


def score(t):
    if not t: return {'n':0,'wr':0,'pnl':0,'dd':0,'pf':0,'sc':-999}
    w = [x for x in t if x.pnl_pct > 0]
    l = [x for x in t if x.pnl_pct <= 0]
    pnl = sum(x.pnl_pct for x in t)
    wr = len(w)/len(t)*100
    ls = abs(sum(x.pnl_pct for x in l))
    pf = sum(x.pnl_pct for x in w)/ls if ls > 0 else 99
    cum = np.cumsum([x.pnl_pct for x in t])
    dd = abs(min(cum.min(), 0))
    return {'n':len(t),'wr':wr,'pnl':pnl,'dd':dd,'pf':pf,'sc':pnl - 2*dd}


def main():
    print('Fetching 730d 1h...')
    df = fetch_historical_ohlcv('BTC/USDT', '1h', 730)
    df_ind = compute_indicators(df, '1h')
    df_ind = classify_regimes_v2(df_ind, hysteresis_bars=3)
    print(f'{len(df_ind)} bars, RANGING: {(df_ind["regime_v2"]=="RANGING").sum()}')
    
    candidates, df_ind = precompute_signals(df_ind)
    longs_c = [c for c in candidates if c['side'] == 'LONG']
    shorts_c = [c for c in candidates if c['side'] == 'SHORT']
    print(f'Candidates: {len(longs_c)} LONG, {len(shorts_c)} SHORT')
    
    # Grid
    rsi_l_grid = [(20,42),(20,38),(20,35),(25,42),(25,38),(15,45),(20,40),(22,40),(18,42),(20,36),(22,38),(18,40)]
    rsi_s_grid = [(58,80),(55,80),(55,85),(50,80),(58,75),(60,85),(55,75),(50,85),(52,80),(55,82),(52,78),(48,80)]
    sl_grid = [1.5, 2.0, 2.5, 3.0]
    tp_grid = [3.0, 4.0, 5.0, 6.0]
    
    # Phase 1: LONG (fix SHORT at current)
    print('\n=== PHASE 1: LONG ===')
    best_l = {'sc': -999}
    all_l = []
    for rsi in rsi_l_grid:
        for slf in sl_grid:
            for tpf in tp_grid:
                trades = sim_with_filters(candidates, df_ind, rsi, (58,80), slf, tpf)
                longs = [t for t in trades if t.type == 'LONG']
                s = score(longs)
                if s['n'] >= 3:
                    r = {**s, 'rsi': rsi, 'slf': slf, 'tpf': tpf}
                    all_l.append(r)
                    if s['sc'] > best_l['sc']:
                        best_l = r
    all_l.sort(key=lambda x: x['sc'], reverse=True)
    print(f'Top 5 LONG ({len(all_l)} configs):')
    print(f'{"RSI":>10} {"SLf":>4} {"TPf":>4} {"N":>3} {"WR%":>5} {"PnL%":>7} {"DD%":>6} {"PF":>5} {"Sc":>7}')
    for r in all_l[:5]:
        print(f'{str(r["rsi"]):>10} {r["slf"]:4.1f} {r["tpf"]:4.1f} {r["n"]:3d} {r["wr"]:5.1f} {r["pnl"]:+7.2f} {r["dd"]:6.2f} {r["pf"]:5.2f} {r["sc"]:+7.2f}')
    
    # Phase 2: SHORT (fix LONG at best)
    print('\n=== PHASE 2: SHORT ===')
    bl = best_l
    best_s = {'sc': -999}
    all_s = []
    for rsi in rsi_s_grid:
        for slf in sl_grid:
            for tpf in tp_grid:
                trades = sim_with_filters(candidates, df_ind, bl['rsi'], rsi, bl['slf'], bl['tpf'], sl_f_short=slf, tp_f_short=tpf)
                shorts = [t for t in trades if t.type == 'SHORT']
                s = score(shorts)
                if s['n'] >= 2:
                    r = {**s, 'rsi': rsi, 'slf': slf, 'tpf': tpf}
                    all_s.append(r)
                    if s['sc'] > best_s['sc']:
                        best_s = r
    all_s.sort(key=lambda x: x['sc'], reverse=True)
    print(f'Top 5 SHORT ({len(all_s)} configs):')
    print(f'{"RSI":>10} {"SLf":>4} {"TPf":>4} {"N":>3} {"WR%":>5} {"PnL%":>7} {"DD%":>6} {"PF":>5} {"Sc":>7}')
    for r in all_s[:5]:
        print(f'{str(r["rsi"]):>10} {r["slf"]:4.1f} {r["tpf"]:4.1f} {r["n"]:3d} {r["wr"]:5.1f} {r["pnl"]:+7.2f} {r["dd"]:6.2f} {r["pf"]:5.2f} {r["sc"]:+7.2f}')
    
    # Phase 3: Combined
    print('\n=== PHASE 3: COMBINED ===')
    curr = sim_with_filters(candidates, df_ind, (20,42), (58,80), 1.5, 3.0)
    opt = sim_with_filters(candidates, df_ind, bl['rsi'], best_s['rsi'], bl['slf'], bl['tpf'],
                           sl_f_short=best_s['slf'], tp_f_short=best_s['tpf'])
    sc, so = score(curr), score(opt)
    
    print(f'{"":25s} {"CURRENT":>10} {"OPTIMIZED":>10} {"DELTA":>10}')
    print('-'*58)
    for name, cv, ov in [
        ('Total trades', sc['n'], so['n']),
        ('LONG', len([t for t in curr if t.type=='LONG']), len([t for t in opt if t.type=='LONG'])),
        ('SHORT', len([t for t in curr if t.type=='SHORT']), len([t for t in opt if t.type=='SHORT'])),
        ('WR%', sc['wr'], so['wr']),
        ('LONG PnL%', score([t for t in curr if t.type=='LONG'])['pnl'], score([t for t in opt if t.type=='LONG'])['pnl']),
        ('SHORT PnL%', score([t for t in curr if t.type=='SHORT'])['pnl'], score([t for t in opt if t.type=='SHORT'])['pnl']),
        ('Total PnL%', sc['pnl'], so['pnl']),
        ('DD%', sc['dd'], so['dd']),
        ('PF', sc['pf'], so['pf']),
    ]:
        d = ov - cv
        if isinstance(cv, float): print(f'{name:25s} {cv:10.2f} {ov:10.2f} {d:+10.2f}')
        else: print(f'{name:25s} {cv:10d} {ov:10d} {d:+10d}')
    
    # Save results
    result = {
        'best_long': {'rsi': list(bl['rsi']), 'slf': bl['slf'], 'tpf': bl['tpf'],
                     'n': bl['n'], 'wr': round(bl['wr'],1), 'pnl': round(bl['pnl'],2)},
        'best_short': {'rsi': list(best_s['rsi']), 'slf': best_s['slf'], 'tpf': best_s['tpf'],
                      'n': best_s['n'], 'wr': round(best_s['wr'],1), 'pnl': round(best_s['pnl'],2)},
        'combined': {'n': so['n'], 'wr': round(so['wr'],1), 'pnl': round(so['pnl'],2),
                    'dd': round(so['dd'],2), 'pf': round(so['pf'],2)},
        'current': {'n': sc['n'], 'wr': round(sc['wr'],1), 'pnl': round(sc['pnl'],2),
                   'dd': round(sc['dd'],2), 'pf': round(sc['pf'],2)},
    }
    with open('/home/z/my-project/download/mr_optimized.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nSaved to /home/z/my-project/download/mr_optimized.json')


if __name__ == '__main__':
    main()
