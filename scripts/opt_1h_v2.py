"""Ultra-fast 1h BTC/USDT optimizer v2 — inline metrics, no calculate_metrics."""
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
COST = DEFAULT_FEE_PCT*2 + DEFAULT_SPREAD_BPS/50 + DEFAULT_SLIPPAGE_BPS/50

def load_data():
    if os.path.exists(DATA_CACHE):
        with open(DATA_CACHE, 'rb') as f: df = pickle.load(f)
    else:
        from backtest import fetch_historical_ohlcv
        df = fetch_historical_ohlcv('BTC/USDT', '1h', 730)
        with open(DATA_CACHE, 'wb') as f: pickle.dump(df, f)
    return df

def main():
    t0 = time.time()
    print('#' * 140)
    print('#  BTC/USDT 1H OPTIMIZER v2 — Inline Metrics — Beat Buy & Hold')
    print('#' * 140)

    df = load_data()
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
    bbl = dc['bb_lower'].values.astype(np.float64)
    bbu = dc['bb_upper'].values.astype(np.float64)
    f382 = dc['fib_0382'].values.astype(np.float64)
    f500 = dc['fib_0500'].values.astype(np.float64)
    f618 = dc['fib_0618'].values.astype(np.float64)
    fdir = dc['fib_direction'].values.astype(np.int32)
    e20t = dc['ema20_touched'].values.astype(bool)
    e50t = dc['ema50_touched'].values.astype(bool)
    e50tu = dc['ema50_touched_up'].values.astype(bool)
    rg = np.array([str(r) for r in dc['regime'].values])
    idx_arr = dc.index
    bh = (cl[-1] - cl[0]) / cl[0] * 100
    print(f'Data: {n:,} candles, B&H={bh:+.2f}%, Cost/trade={COST:.3f}%')

    # === GRID DEFINITION ===
    combos = []
    tcfgs = [
        # (adx, rsi_l_lo, rsi_l_hi, rsi_s_lo, rsi_s_hi, et, es, pb, atr_flag)
        (25, 25, 50, 50, 75, 1, 1, 1, 1),
        (20, 25, 50, 50, 75, 1, 1, 1, 1),
        (30, 25, 50, 50, 75, 1, 1, 1, 1),
        (15, 25, 50, 50, 75, 1, 1, 1, 1),
        (25, 20, 55, 45, 80, 1, 1, 1, 1),
        (20, 20, 55, 45, 80, 1, 1, 1, 1),
        (30, 20, 55, 45, 80, 1, 1, 1, 1),
        (25, 25, 50, 50, 75, 1, 0, 1, 1),
        (20, 20, 55, 45, 80, 1, 0, 1, 1),
        (25, 25, 50, 50, 75, 0, 1, 1, 1),
        (20, 25, 55, 45, 75, 0, 1, 1, 1),
        (25, 30, 50, 50, 70, 1, 1, 0, 1),
        (20, 30, 50, 50, 70, 1, 1, 0, 1),
        (25, 28, 42, 58, 72, 1, 1, 1, 1),
        (20, 28, 42, 58, 72, 1, 1, 1, 1),
        (25, 25, 55, 45, 75, 1, 0, 0, 1),
        (20, 25, 60, 40, 75, 0, 0, 0, 1),
        (0, 30, 65, 35, 70, 0, 0, 0, 1),
        (25, 28, 48, 52, 72, 1, 1, 1, 1),
        (15, 28, 48, 52, 72, 1, 1, 1, 1),
        (0, 25, 60, 40, 70, 0, 0, 0, 1),
        (25, 30, 55, 45, 70, 1, 1, 1, 0),  # NO transition
    ]

    # High R:R SL/TP combos (cost-aware)
    sltps = [
        (0.75, 3.0), (0.75, 4.0), (0.75, 5.0), (0.75, 6.0), (0.75, 8.0),
        (1.0, 3.0), (1.0, 4.0), (1.0, 5.0), (1.0, 6.0), (1.0, 7.0), (1.0, 8.0), (1.0, 10.0),
        (1.25, 4.0), (1.25, 5.0), (1.25, 6.0), (1.25, 7.0), (1.25, 8.0), (1.25, 10.0),
        (1.5, 5.0), (1.5, 6.0), (1.5, 7.0), (1.5, 8.0), (1.5, 10.0),
        (2.0, 6.0), (2.0, 7.0), (2.0, 8.0), (2.0, 10.0),
        (2.5, 8.0), (2.5, 10.0), (2.5, 12.0),
    ]
    mr_sltps = [(0.5, 1.0), (0.5, 1.5), (0.5, 2.0), (0.75, 1.5), (0.75, 2.0), (1.0, 2.0)]
    mrls = [30, 35, 40]; mrss = [60, 65, 70]
    atrs = [(0.05, 0.95), (0.10, 0.90)]

    for tc in tcfgs:
        adx, rl, rh, sl2, sh, et, es, pb, af = tc
        for sl_m, tp_m in sltps:
            for apl, aph in atrs:
                combos.append((adx, rl, rh, sl2, sh, et, es, pb, af, False, sl_m, tp_m, apl, aph, 0.025, 35, 65))
        for sl_m, tp_m in mr_sltps:
            for mrl in mrls:
                for mrs in mrss:
                    combos.append((adx, rl, rh, sl2, sh, et, es, pb, af, True, sl_m, tp_m, 0.05, 0.95, 0.025, mrl, mrs))

    print(f'Grid: {len(combos):,} combos')

    # === HELPER (must be defined BEFORE the loop) ===
    def _check_both(i, _adx, _rl, _rh, _sl2, _sh, _et, _es, _pb, _apl, _aph, _ft, _sl_m, _tp_m):
        _rg = rg[i]; c = cl[i]; e50v = e50[i]; e200v = e200[i]
        # Long
        if _rg == 'trending_up':
            if adx_a[i] < _adx: return None, True
        elif _rg == 'trending_down':
            pass  # check short below
        elif _rg == 'transition':
            pass  # allowed
        else:
            return None, True
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
        # Short
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

    # === RUN GRID ===
    results = []
    best_score = 0.0
    best_str = ''
    t1 = time.time()
    tot_combos = len(combos)

    for ci, combo in enumerate(combos):
        adx, rl, rh, sl2, sh, et, es, pb, af, mr, sl_m, tp_m, apl, aph, ft, mrl, mrs = combo
        trades = 0; wins = 0; losses = 0
        total_pnl = 0.0; total_win = 0.0; total_loss = 0.0
        longs = 0; shorts = 0
        equity = [0.0]  # cumulative PnL

        i = 0
        while i < n:
            sig = None; is_long = True
            _rg = rg[i]

            if mr and _rg == 'ranging':
                c = cl[i]; r = rsi_a[i]
                if r <= mrl and lo_c[i] <= bbl[i]:
                    sig = (c, c - sl_m * atr_a[i], c + tp_m * atr_a[i]); is_long = True
                elif r >= mrs and hi_c[i] >= bbu[i]:
                    sig = (c, c + sl_m * atr_a[i], c - tp_m * atr_a[i]); is_long = False
            elif af and _rg == 'transition':
                sig, is_long = _check_both(i, adx, rl, rh, sl2, sh, et, es, pb, apl, aph, ft, sl_m, tp_m)
            elif _rg in ('trending_up', 'trending_down'):
                sig, is_long = _check_both(i, adx, rl, rh, sl2, sh, et, es, pb, apl, aph, ft, sl_m, tp_m)

            if sig is None:
                i += 1; continue

            ep, sl, tp = sig
            xp = None; bars = 0; mj = min(i + 72, n)
            if is_long:
                for j in range(i + 1, mj):
                    if lo_c[j] <= sl: xp = sl; bars = j - i; break
                    if hi_c[j] >= tp: xp = tp; bars = j - i; break
            else:
                for j in range(i + 1, mj):
                    if hi_c[j] >= sl: xp = sl; bars = j - i; break
                    if lo_c[j] <= tp: xp = tp; bars = j - i; break
            if xp is None:
                bars = mj - 1 - i; xp = cl[mj - 1]

            _, adj, _ = _apply_costs(ep, xp, is_long, DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS)
            pnl = (adj - ep) / ep * 100 if is_long else (ep - adj) / ep * 100
            trades += 1
            total_pnl += pnl
            if pnl > 0: wins += 1; total_win += pnl
            else: losses += 1; total_loss += abs(pnl)
            if is_long: longs += 1
            else: shorts += 1
            equity.append(equity[-1] + pnl)
            i += bars + 1

        if trades < 30:
            results.append((0, combo, trades, 0, 0, 0, 0, 0, 0, 0, longs, shorts))
            continue

        wr = wins / trades * 100
        pf = total_win / total_loss if total_loss > 0 else 0
        # Max DD from equity
        eq = np.array(equity)
        peak = np.maximum.accumulate(eq)
        dd = float(np.max(peak - eq)) if len(eq) > 1 else 0
        # Sharpe (simple)
        pnls_d = np.diff(equity)
        sharpe = float(np.mean(pnls_d) / np.std(pnls_d) * math.sqrt(365 * 24)) if np.std(pnls_d) > 0 and len(pnls_d) > 1 else 0

        # Score
        if pf <= 1.0 or wr < 25 or dd > 35:
            score = 0
        else:
            excess = total_pnl - bh
            bh_f = max(0.01, 1.0 + excess / 10.0) if excess <= 0 else 1.0 + excess / 5.0
            score = (wr / 100) * pf * bh_f * max(0.01, 1.0 - dd / 50) * min(math.log(max(trades, 1)) / math.log(200), 1.5)
            score = round(score, 4)

        results.append((score, combo, trades, wr, pf, total_pnl, bh, dd, sharpe, 0, longs, shorts))

        if score > best_score:
            best_score = score
            excess = total_pnl - bh
            best_str = f'T={trades} WR={wr:.1f}% PF={pf:.2f} PnL={total_pnl:+.2f}% B&H={bh:+.2f}% Exc={excess:+.2f}% DD={dd:.2f}% SL={sl_m}x TP={tp_m}x MR={mr} ADX={adx}'

        if (ci + 1) % 2000 == 0:
            el = time.time() - t1; spd = (ci + 1) / max(el, 0.01)
            eta = (tot_combos - ci - 1) / max(spd, 0.01)
            print(f'  [{ci+1:,}/{tot_combos:,}] {spd:.0f}/s ETA={eta:.0f}s BEST: {best_str}')

    print(f'  Grid done in {time.time()-t1:.1f}s')

    # Sort and display
    results.sort(key=lambda x: x[0], reverse=True)
    print(f'\n{"=" * 160}')
    print(f'  TOP 30 (of {tot_combos:,})')
    print(f'{"=" * 160}')
    print(f'{"#":>3} {"Score":>7} {"T":>5} {"L/S":>6} {"WR%":>6} {"PF":>6} {"PnL%":>8} {"B&H%":>7} {"Excess":>7} {"DD%":>6} {"Sharpe":>7} {"MR":>3} {"ATr":>4} {"ET":>3} {"ES":>3} {"PB":>3} {"ADX":>4} {"RSI_L":>8} {"RSI_S":>8} {"SL":>4} {"TP":>5}')
    print(f'{"-" * 160}')

    for i, (score, combo, trades, wr, pf, pnl, _bh, dd, sharpe, _, longs, shorts) in enumerate(results[:30]):
        adx, rl, rh, sl2, sh, et, es, pb, af, mr, sl_m, tp_m, apl, aph, ft, mrl, mrs = combo
        ls = f'{longs}/{shorts}'
        exc = pnl - bh
        print(f'{i+1:>3} {score:>7.3f} {trades:>5} {ls:>6} {wr:>6.1f} {pf:>6.2f} {pnl:>8.2f} {bh:>7.2f} {exc:>+7.2f} {dd:>6.2f} {sharpe:>7.2f} {"Y" if mr else "N":>3} {"Y" if af else "N":>4} {"Y" if et else "N":>3} {"Y" if es else "N":>3} {"Y" if pb else "N":>3} {adx:>4.0f} {rl:>3}-{rh:>3} {sl2:>3}-{sh:>3} {sl_m:>4.2f} {tp_m:>5.1f}')

    # Show top by PnL
    by_pnl = sorted(results, key=lambda x: x[5], reverse=True)
    print(f'\n--- TOP 10 BY PNL ---')
    for i, (score, combo, trades, wr, pf, pnl, _bh, dd, sharpe, _, longs, shorts) in enumerate(by_pnl[:10]):
        exc = pnl - bh
        print(f'  {i+1}. T={trades} WR={wr:.1f}% PF={pf:.2f} PnL={pnl:+.2f}% B&H={bh:+.2f}% Exc={exc:+.2f}% DD={dd:.2f}%')

    # Save
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'download')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, '1h_v2_results.json'), 'w') as f:
        json.dump({'buy_hold': round(bh, 4), 'total_combos': tot_combos,
                    'top20': [{'score': s, 'combo': list(c), 'trades': t, 'wr': round(w, 2), 'pf': round(p, 4),
                              'pnl': round(pn, 4), 'dd': round(d, 4), 'sharpe': round(sh, 4)}
                             for s, c, t, w, p, pn, bh, d, sh, _, l, s2 in results[:20]]
                    }, f, indent=2, default=str)
    print(f'\nSaved to {out}/1h_v2_results.json')
    print(f'Total: {time.time()-t0:.1f}s')


if __name__ == '__main__': main()
