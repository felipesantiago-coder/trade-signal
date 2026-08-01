"""P2 Refinement around the P1 winner: SL=2.5x TP=12x R:R=4.8:1 beats B&H by +6.81pp."""
import sys, os, time, math, json, pickle, logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger('opt')
logger.setLevel(logging.INFO)

import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from indicators import compute_indicators
from backtest import (_apply_costs, DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS)

DATA_CACHE = '/tmp/ctev_1h_study_data.pkl'

def main():
    t0 = time.time()
    print('#' * 140)
    print('#  BTC/USDT 1H P2 REFINEMENT — Around SL=2.5x TP=12x winner')
    print('#' * 140)

    with open(DATA_CACHE, 'rb') as f: df = pickle.load(f)
    df_ind = compute_indicators(df, '1h')
    cc = ['ema20','ema50','ema200','rsi','atr','atr_percentile','macd','macd_signal','macd_hist','adx','plus_di','minus_di','regime']
    dc = df_ind.dropna(subset=cc).copy()
    n = len(dc)
    cl = dc['close'].values.astype(np.float64)
    hi_c = dc['high'].values.astype(np.float64)
    lo_c = dc['low'].values.astype(np.float64)
    e20 = dc['ema20'].values.astype(np.float64)
    e50 = dc['ema50'].values.astype(np.float64)
    e200 = dc['ema200'].values.astype(np.float64)
    rsi_a = dc['rsi'].values.astype(np.float64)
    atr_a = dc['atr'].values.astype(np.float64)
    apct = dc['atr_percentile'].values.astype(np.float64)
    adx_a = dc['adx'].values.astype(np.float64)
    slope_a = dc['ema50_slope'].values.astype(np.float64)
    f382 = dc['fib_0382'].values.astype(np.float64)
    f500 = dc['fib_0500'].values.astype(np.float64)
    f618 = dc['fib_0618'].values.astype(np.float64)
    fdir = dc['fib_direction'].values.astype(np.int32)
    e20t = dc['ema20_touched'].values.astype(bool)
    e50t = dc['ema50_touched'].values.astype(bool)
    e50tu = dc['ema50_touched_up'].values.astype(bool)
    rg = np.array([str(r) for r in dc['regime'].values])
    bh = (cl[-1] - cl[0]) / cl[0] * 100
    print(f'Data: {n:,} candles, B&H={bh:+.2f}%')

    def _check_both(i, _adx, _rl, _rh, _sl2, _sh, _et, _es, _pb, _apl, _aph, _ft, _sl_m, _tp_m):
        _rg = rg[i]; c = cl[i]; e50v = e50[i]; e200v = e200[i]
        if _rg == 'trending_up':
            if adx_a[i] < _adx: return None, True
        elif _rg in ('trending_down', 'transition'): pass
        else: return None, True
        if _et and not (c > e50v > e200v): return None, True
        if _es and slope_a[i] <= -1.0: return None, True
        r = rsi_a[i]
        if _rl <= r <= _rh:
            if apct[i] < _apl or apct[i] > _aph: return None, True
            if _pb:
                ok = False; fd = fdir[i]
                if fd == 1:
                    f38 = f382[i]; f61 = f618[i]
                    if not (np.isnan(f38) or np.isnan(f61)) and f61 <= c <= f38: ok = True
                    if not ok:
                        lo_v = lo_c[i]; tol = c * _ft
                        for fl in (f38, f500[i], f61):
                            if not np.isnan(fl) and fl > 0 and abs(lo_v - fl) <= tol: ok = True; break
                if not ok and e20t[i] and c > e20[i]: ok = True
                if not ok and e50t[i] and c > e50v: ok = True
                if not ok: return None, True
            ep = c; sl = ep - _sl_m * atr_a[i]; tp = ep + _tp_m * atr_a[i]
            if sl > 0: return (ep, sl, tp), True
        if _rg == 'trending_down':
            if adx_a[i] < _adx: return None, False
        if _et and not (c < e50v < e200v): return None, False
        if _es and slope_a[i] >= 1.0: return None, False
        if _sl2 <= r <= _sh:
            if apct[i] < _apl or apct[i] > _aph: return None, False
            if _pb:
                ok = False; fd = fdir[i]
                if fd == -1:
                    f38 = f382[i]; f61 = f618[i]
                    if not (np.isnan(f38) or np.isnan(f61)) and f61 <= c <= f38: ok = True
                    if not ok:
                        hi_v = hi_c[i]; tol = c * _ft
                        for fl in (f38, f500[i], f61):
                            if not np.isnan(fl) and fl > 0 and abs(hi_v - fl) <= tol: ok = True; break
                if not ok and e20t[i] and c < e20[i] and hi_c[i] >= e20[i]: ok = True
                if not ok and e50tu[i] and c < e50v and hi_c[i] >= e50v: ok = True
                if not ok: return None, False
            ep = c; sl = ep + _sl_m * atr_a[i]; tp = ep - _tp_m * atr_a[i]
            return (ep, sl, tp), False
        return None, True

    def sim(combo):
        adx, rl, rh, sl2, sh, et, es, pb, af, sl_m, tp_m, apl, aph, ft = combo
        trades = 0; wins = 0; total_win = 0.0; total_loss = 0.0
        total_pnl = 0.0; equity = [0.0]; longs = 0; shorts = 0
        i = 0
        while i < n:
            sig = None; is_long = True; _rg = rg[i]
            if af and _rg == 'transition':
                sig, is_long = _check_both(i, adx, rl, rh, sl2, sh, et, es, pb, apl, aph, ft, sl_m, tp_m)
            elif _rg in ('trending_up', 'trending_down'):
                sig, is_long = _check_both(i, adx, rl, rh, sl2, sh, et, es, pb, apl, aph, ft, sl_m, tp_m)
            if sig is None: i += 1; continue
            ep, sl, tp = sig; xp = None; bars = 0; mj = min(i + 72, n)
            if is_long:
                for j in range(i + 1, mj):
                    if lo_c[j] <= sl: xp = sl; bars = j - i; break
                    if hi_c[j] >= tp: xp = tp; bars = j - i; break
            else:
                for j in range(i + 1, mj):
                    if hi_c[j] >= sl: xp = sl; bars = j - i; break
                    if lo_c[j] <= tp: xp = tp; bars = j - i; break
            if xp is None: bars = mj - 1 - i; xp = cl[mj - 1]
            _, adj, _ = _apply_costs(ep, xp, is_long, DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS)
            pnl = (adj - ep) / ep * 100 if is_long else (ep - adj) / ep * 100
            trades += 1; total_pnl += pnl
            if pnl > 0: wins += 1; total_win += pnl
            else: total_loss += abs(pnl)
            if is_long: longs += 1
            else: shorts += 1
            equity.append(equity[-1] + pnl)
            i += bars + 1
        if trades < 30: return None
        wr = wins / trades * 100
        pf = total_win / total_loss if total_loss > 0 else 0
        eq = np.array(equity); peak = np.maximum.accumulate(eq)
        dd = float(np.max(peak - eq)) if len(eq) > 1 else 0
        pnls_d = np.diff(equity)
        sharpe = float(np.mean(pnls_d) / np.std(pnls_d) * math.sqrt(365 * 24)) if np.std(pnls_d) > 0 and len(pnls_d) > 1 else 0
        if pf <= 1.0 or wr < 25 or dd > 35: score = 0
        else:
            excess = total_pnl - bh
            bh_f = max(0.01, 1.0 + excess / 10.0) if excess <= 0 else 1.0 + excess / 5.0
            score = (wr / 100) * pf * bh_f * max(0.01, 1.0 - dd / 50) * min(math.log(max(trades, 1)) / math.log(200), 1.5)
        return {'score': round(score, 4), 'combo': combo, 'trades': trades, 'wr': round(wr, 2),
                'pf': round(pf, 4), 'pnl': round(total_pnl, 4), 'dd': round(dd, 4), 'sharpe': round(sharpe, 4),
                'longs': longs, 'shorts': shorts, 'excess': round(total_pnl - bh, 4)}

    # P2 Grid: refine around winner
    combos = []
    # SL: 1.5 to 3.5 in 0.25 steps
    # TP: 6 to 15 in 1.0 steps
    sls = [x * 0.25 for x in range(6, 15)]  # 1.5 to 3.5
    tps = list(range(6, 16))  # 6 to 15
    # ADX: 15, 20, 25, 30, 35
    adxs = [15, 20, 25, 30, 35]
    # RSI Long: vary around 20-55
    rsi_l_ranges = [(15, 50), (20, 50), (20, 55), (20, 60), (25, 50), (25, 55), (25, 60), (30, 55), (30, 60), (15, 55), (18, 52), (22, 58)]
    # RSI Short: vary around 45-80
    rsi_s_ranges = [(40, 75), (45, 75), (45, 80), (50, 75), (50, 80), (55, 80), (40, 80), (45, 70), (40, 85)]
    # Fib tolerance
    fts = [0.015, 0.025, 0.035, 0.050]
    # ATR percentile
    atr_ranges = [(0.05, 0.95), (0.10, 0.90), (0.05, 0.85), (0.15, 0.95)]
    # ET/ES/PB variations
    bools = [(1, 1, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1)]

    for adx in adxs:
        for rl, rh in rsi_l_ranges:
            for sl2, sh in rsi_s_ranges:
                for et, es, pb in bools:
                    for sl_m in sls:
                        for tp_m in tps:
                            if tp_m / sl_m < 1.5: continue  # R:R >= 1.5
                            for ft in fts:
                                for apl, aph in atr_ranges:
                                    combos.append((adx, rl, rh, sl2, sh, et, es, pb, True, sl_m, tp_m, apl, aph, ft))

    print(f'P2 Grid: {len(combos):,} combos')

    # Run
    results = []; best_score = 0; best_str = ''
    t1 = time.time()
    for ci, combo in enumerate(combos):
        r = sim(combo)
        if r is None: continue
        results.append(r)
        if r['score'] > best_score:
            best_score = r['score']
            c = r['combo']
            best_str = f'T={r["trades"]} WR={r["wr"]:.1f}% PF={r["pf"]:.2f} PnL={r["pnl"]:+.2f}% B&H={bh:+.2f}% Exc={r["excess"]:+.2f}% DD={r["dd"]:.2f}% SL={c[9]:.2f}x TP={c[10]:.0f}x ADX={c[0]} RSI_L={c[1]}-{c[2]} RSI_S={c[3]}-{c[4]} ft={c[14]:.3f}'
        if (ci + 1) % 10000 == 0:
            el = time.time() - t1; spd = (ci + 1) / max(el, 0.01)
            eta = (len(combos) - ci - 1) / max(spd, 0.01)
            print(f'  [{ci+1:,}/{len(combos):,}] {spd:.0f}/s ETA={eta:.0f}s BEST: {best_str}')

    print(f'  P2 done in {time.time()-t1:.1f}s')
    results.sort(key=lambda x: x['score'], reverse=True)

    print(f'\n{"=" * 160}')
    print(f'  P2 TOP 30')
    print(f'{"=" * 160}')
    print(f'{"#":>3} {"Score":>7} {"T":>5} {"L/S":>6} {"WR%":>6} {"PF":>6} {"PnL%":>8} {"B&H%":>7} {"Excess":>7} {"DD%":>6} {"Sharpe":>7} {"ADX":>4} {"RSI_L":>8} {"RSI_S":>8} {"ET":>3} {"ES":>3} {"PB":>3} {"SL":>4} {"TP":>5} {"FT%":>4} {"ATR":>10}')
    print(f'{"-" * 160}')
    for i, r in enumerate(results[:30]):
        c = r['combo']; ls = f'{r["longs"]}/{r["shorts"]}'
        print(f'{i+1:>3} {r["score"]:>7.3f} {r["trades"]:>5} {ls:>6} {r["wr"]:>6.1f} {r["pf"]:>6.2f} {r["pnl"]:>8.2f} {bh:>7.2f} {r["excess"]:>+7.2f} {r["dd"]:>6.2f} {r["sharpe"]:>7.2f} {c[0]:>4.0f} {c[1]:>3}-{c[2]:>3} {c[3]:>3}-{c[4]:>3} {"Y" if c[5] else "N":>3} {"Y" if c[6] else "N":>3} {"Y" if c[7] else "N":>3} {c[9]:>4.2f} {c[10]:>5.0f} {c[14]*100:>4.1f} {c[12]:.2f}-{c[13]:.2f}')

    # Also show top by PnL
    by_pnl = sorted(results, key=lambda x: x['pnl'], reverse=True)
    print(f'\n--- TOP 10 BY PNL ---')
    for i, r in enumerate(by_pnl[:10]):
        c = r['combo']
        print(f'  {i+1}. T={r["trades"]} WR={r["wr"]:.1f}% PF={r["pf"]:.2f} PnL={r["pnl"]:+.2f}% B&H={bh:+.2f}% Exc={r["excess"]:+.2f}% DD={r["dd"]:.2f}% SL={c[9]:.2f}x TP={c[10]:.0f}x ADX={c[0]} RSI_L={c[1]}-{c[2]} ft={c[14]:.3f}')

    # Save
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'download')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, '1h_p2_results.json'), 'w') as f:
        json.dump({'buy_hold': round(bh, 4), 'total_combos': len(combos), 'top30': results[:30]}, f, indent=2)
    print(f'\nSaved to {out}/1h_p2_results.json')
    print(f'Total: {time.time()-t0:.1f}s')

if __name__ == '__main__': main()