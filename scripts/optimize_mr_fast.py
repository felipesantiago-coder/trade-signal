"""
Optimizacao rapida dos parametros MR LONG e SHORT.
Roda em ~60s com grid enxuto focado nos params que mais impactam.
"""
import logging, sys, os
logging.basicConfig(level=logging.ERROR)

import numpy as np
import pandas as pd
from itertools import product

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


def fast_sim_mr(df_ind, params_long, params_short, max_bars=72):
    trades = []
    i = 0
    n = len(df_ind)
    while i < n:
        row = df_ind.iloc[i]
        rv2 = str(row.get('regime_v2', ''))
        conf = float(row.get('regime_confidence', 0.5))
        if rv2 != 'RANGING' or conf < 0.2:
            i += 1
            continue
        atr_pct = float(row.get('atr_percentile', 0.5))
        if atr_pct < 0.10 or atr_pct > 0.85:
            i += 1
            continue
        sig = evaluate_mean_reversion_long(row, params_long)
        if sig is None:
            sig = evaluate_mean_reversion_short(row, params_short)
        if sig is None:
            i += 1
            continue
        entry_price = sig.entry_price
        sl, tp, atr = sig.stop_loss, sig.take_profit, sig.atr
        is_long = sig.type.value == 'LONG'
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
        _, adj_exit, _ = _apply_costs(entry_price, exit_price, is_long, DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS)
        pnl = (adj_exit - entry_price) / entry_price * 100 if is_long else (entry_price - adj_exit) / entry_price * 100
        trades.append(TradeResult(
            entry_ts=row.name, exit_ts=df_ind.iloc[min(i + bars, n - 1)].name,
            type=sig.type.value, entry_price=entry_price, exit_price=exit_price,
            stop_loss=sl, take_profit=tp, atr=atr, rsi=sig.rsi,
            pnl_pct=round(pnl, 4), pnl_abs=round(exit_price - entry_price, 2),
            bars_held=bars, exit_reason=exit_reason, atr_percentile=atr_pct,
        ))
        i += bars + 1
    return trades


def make_params(side, rsi_min, rsi_max, sl_f, tp_f):
    if side == 'L':
        return {'strategy_type':'mr','allow_long':True,'allow_short':False,
                'sl_mult':BASE_SL*sl_f,'tp_mult':BASE_SL*tp_f,
                'rsi_long_range':(rsi_min,rsi_max),'rsi_short_range':(99,99),
                'require_volume':False,'min_confidence':0.2}
    else:
        return {'strategy_type':'mr','allow_long':False,'allow_short':True,
                'sl_mult':BASE_SL*sl_f,'tp_mult':BASE_SL*tp_f,
                'rsi_long_range':(0,0),'rsi_short_range':(rsi_min,rsi_max),
                'require_volume':False,'min_confidence':0.2}


def score_trades(trades):
    if not trades:
        return {'n':0,'wr':0,'pnl':0,'dd':0,'pf':0,'score':-999}
    wins = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct <= 0]
    pnl = sum(t.pnl_pct for t in trades)
    wr = len(wins)/len(trades)*100
    loss_sum = abs(sum(t.pnl_pct for t in losses))
    pf = sum(t.pnl_pct for t in wins)/loss_sum if loss_sum > 0 else 99
    cum = np.cumsum([t.pnl_pct for t in trades])
    dd = abs(min(cum.min(), 0))
    score = pnl - 2*dd
    return {'n':len(trades),'wr':wr,'pnl':pnl,'dd':dd,'pf':pf,'score':score}


def main():
    print('Fetching 730d 1h data...')
    df = fetch_historical_ohlcv('BTC/USDT', '1h', 730)
    df_ind = compute_indicators(df, '1h')
    df_ind = classify_regimes_v2(df_ind, hysteresis_bars=3)
    ranging = (df_ind['regime_v2']=='RANGING').sum()
    print(f'Data: {len(df_ind)} bars, RANGING: {ranging} ({100*ranging/len(df_ind):.1f}%)')

    # ============================================================
    # PHASE 1: LONG optimization (narrow grid)
    # ============================================================
    print('\n' + '='*60)
    print('PHASE 1: MR LONG optimization')
    print('='*60)

    rsi_l_grid = [(20,42),(20,38),(20,35),(25,42),(25,38),(15,45),(20,40),(22,40),(18,42),(20,36)]
    sl_grid = [1.5, 2.0, 2.5, 3.0]
    tp_grid = [3.0, 4.0, 5.0, 6.0]
    fixed_short = make_params('S', 58, 80, 1.5, 3.0)

    best_l = {'score': -999}
    for rsi, slf, tpf in product(rsi_l_grid, sl_grid, tp_grid):
        pl = make_params('L', rsi[0], rsi[1], slf, tpf)
        trades = fast_sim_mr(df_ind, pl, fixed_short)
        longs = [t for t in trades if t.type == 'LONG']
        s = score_trades(longs)
        if s['n'] >= 3 and s['score'] > best_l['score']:
            best_l = {**s, 'rsi': rsi, 'slf': slf, 'tpf': tpf}

    # Top 5 by PnL with >= 3 trades
    all_l = []
    for rsi, slf, tpf in product(rsi_l_grid, sl_grid, tp_grid):
        pl = make_params('L', rsi[0], rsi[1], slf, tpf)
        trades = fast_sim_mr(df_ind, pl, fixed_short)
        longs = [t for t in trades if t.type == 'LONG']
        s = score_trades(longs)
        if s['n'] >= 3:
            all_l.append({**s, 'rsi': rsi, 'slf': slf, 'tpf': tpf})
    all_l.sort(key=lambda x: x['score'], reverse=True)

    print(f'\nTop 5 LONG ({len(all_l)} configs with >=3 trades):')
    print(f'{"RSI":>10} {"SL_f":>5} {"TP_f":>5} {"N":>4} {"WR%":>6} {"PnL%":>7} {"DD%":>6} {"PF":>6} {"Score":>7}')
    print('-'*65)
    for r in all_l[:5]:
        print(f'{str(r["rsi"]):>10} {r["slf"]:5.1f} {r["tpf"]:5.1f} {r["n"]:4d} {r["wr"]:6.1f} {r["pnl"]:+7.2f} {r["dd"]:6.2f} {r["pf"]:6.2f} {r["score"]:+7.2f}')

    # ============================================================
    # PHASE 2: SHORT optimization
    # ============================================================
    print('\n' + '='*60)
    print('PHASE 2: MR SHORT optimization')
    print('='*60)

    rsi_s_grid = [(58,80),(55,80),(55,85),(50,80),(58,75),(60,85),(55,75),(50,85),(52,80),(55,82)]
    sl_grid_s = [1.5, 2.0, 2.5, 3.0]
    tp_grid_s = [3.0, 4.0, 5.0, 6.0]
    best_l_params = make_params('L', best_l.get('rsi',(20,42))[0], best_l.get('rsi',(20,42))[1],
                                  best_l.get('slf',1.5), best_l.get('tpf',3.0))

    best_s = {'score': -999}
    for rsi, slf, tpf in product(rsi_s_grid, sl_grid_s, tp_grid_s):
        ps = make_params('S', rsi[0], rsi[1], slf, tpf)
        trades = fast_sim_mr(df_ind, best_l_params, ps)
        shorts = [t for t in trades if t.type == 'SHORT']
        s = score_trades(shorts)
        if s['n'] >= 2 and s['score'] > best_s['score']:
            best_s = {**s, 'rsi': rsi, 'slf': slf, 'tpf': tpf}

    all_s = []
    for rsi, slf, tpf in product(rsi_s_grid, sl_grid_s, tp_grid_s):
        ps = make_params('S', rsi[0], rsi[1], slf, tpf)
        trades = fast_sim_mr(df_ind, best_l_params, ps)
        shorts = [t for t in trades if t.type == 'SHORT']
        s = score_trades(shorts)
        if s['n'] >= 2:
            all_s.append({**s, 'rsi': rsi, 'slf': slf, 'tpf': tpf})
    all_s.sort(key=lambda x: x['score'], reverse=True)

    print(f'\nTop 5 SHORT ({len(all_s)} configs with >=2 trades):')
    print(f'{"RSI":>10} {"SL_f":>5} {"TP_f":>5} {"N":>4} {"WR%":>6} {"PnL%":>7} {"DD%":>6} {"PF":>6} {"Score":>7}')
    print('-'*65)
    for r in all_s[:5]:
        print(f'{str(r["rsi"]):>10} {r["slf"]:5.1f} {r["tpf"]:5.1f} {r["n"]:4d} {r["wr"]:6.1f} {r["pnl"]:+7.2f} {r["dd"]:6.2f} {r["pf"]:6.2f} {r["score"]:+7.2f}')

    # ============================================================
    # PHASE 3: Combined
    # ============================================================
    print('\n' + '='*60)
    print('PHASE 3: COMBINED VALIDATION')
    print('='*60)

    curr_l = make_params('L', 20, 42, 1.5, 3.0)
    curr_s = make_params('S', 58, 80, 1.5, 3.0)
    trades_c = fast_sim_mr(df_ind, curr_l, curr_s)
    sc = score_trades(trades_c)

    opt_l = make_params('L', best_l['rsi'][0], best_l['rsi'][1], best_l['slf'], best_l['tpf'])
    opt_s = make_params('S', best_s['rsi'][0], best_s['rsi'][1], best_s['slf'], best_s['tpf'])
    trades_o = fast_sim_mr(df_ind, opt_l, opt_s)
    so = score_trades(trades_o)

    longs_c = [t for t in trades_c if t.type=='LONG']
    shorts_c = [t for t in trades_c if t.type=='SHORT']
    longs_o = [t for t in trades_o if t.type=='LONG']
    shorts_o = [t for t in trades_o if t.type=='SHORT']

    print(f'\n{"":30s} {"CURRENT":>10} {"OPTIMIZED":>10} {"DELTA":>10}')
    print('-'*65)
    for name, cv, ov in [
        ('Total trades', sc['n'], so['n']),
        ('LONG trades', len(longs_c), len(longs_o)),
        ('SHORT trades', len(shorts_c), len(shorts_o)),
        ('LONG WR%', score_trades(longs_c)['wr'], score_trades(longs_o)['wr']),
        ('SHORT WR%', score_trades(shorts_c)['wr'], score_trades(shorts_o)['wr']),
        ('Total WR%', sc['wr'], so['wr']),
        ('LONG PnL%', score_trades(longs_c)['pnl'], score_trades(longs_o)['pnl']),
        ('SHORT PnL%', score_trades(shorts_c)['pnl'], score_trades(shorts_o)['pnl']),
        ('Total PnL%', sc['pnl'], so['pnl']),
        ('Max DD%', sc['dd'], so['dd']),
        ('PF', sc['pf'], so['pf']),
    ]:
        d = ov - cv
        if isinstance(cv, float):
            print(f'{name:30s} {cv:10.2f} {ov:10.2f} {d:+10.2f}')
        else:
            print(f'{name:30s} {cv:10d} {ov:10d} {d:+10d}')

    print(f'\nBest LONG:  RSI={best_l["rsi"]}, SL={best_l["slf"]:.1f}x, TP={best_l["tpf"]:.1f}x -> {best_l["n"]} trades, WR={best_l["wr"]:.1f}%, PnL={best_l["pnl"]:+.2f}%')
    print(f'Best SHORT: RSI={best_s["rsi"]}, SL={best_s["slf"]:.1f}x, TP={best_s["tpf"]:.1f}x -> {best_s["n"]} trades, WR={best_s["wr"]:.1f}%, PnL={best_s["pnl"]:+.2f}%')


if __name__ == '__main__':
    main()
