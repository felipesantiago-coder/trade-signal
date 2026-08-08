r"""refine_v14.py - Refinement grid on v14 best region with sub-period stability validation."""
import sys, os, json, time, logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING)

CACHE = "/home/z/my-project/trade-signal/download/btc_1h_cache.csv"
OUTPUT = "/home/z/my-project/trade-signal/download/refine_v14_results.json"
PHASE1_CACHE = "/home/z/my-project/trade-signal/download/refine_v14_phase1.json"


def load_data():
    df = pd.read_csv(CACHE, index_col=0, parse_dates=True)
    print(f"Cache: {len(df):,} candles", flush=True)
    return df


def run_bt(df, P):
    from strategy_bbwp_squeeze import (
        BBWP_SQUEEZE_PARAMS, reset_cooldown, _check_cooldown,
        _register_signal, _is_squeeze_breakout, _adx_confirms_trend,
        _stoch_rsi_confirms, _volume_confirms, _get_sl_mult,
    )
    from backtest import _apply_costs

    orig = dict(BBWP_SQUEEZE_PARAMS)
    BBWP_SQUEEZE_PARAMS.update(P)
    reset_cooldown()

    results = []
    n = len(df)
    tr_dist = P.get("trailing_atr_mult", 1.5)
    max_b = P.get("max_bars_held", 96)
    use_tr = P.get("use_trailing", True)
    tp1p = P.get("tp1_pct", 0.50)
    pbuf = P.get("post_tp1_sl_buffer", 0.5)
    fee, spr, slp = 0.016, 2.0, 2.0

    i = 0
    while i < n:
        row = df.iloc[i]
        if i < 1:
            i += 1; continue
        cr = ["ema20","ema50","ema200","rsi","atr","atr_percentile",
             "bbwp","stoch_rsi_k","stoch_rsi_d","bb_lower","bb_upper","volume","volume_sma20","adx"]
        if any(pd.isna(row.get(c)) for c in cr):
            i += 1; continue
        ap = float(row.get("atr_percentile", 0.5))
        if ap < 0.10 or ap > 0.90:
            i += 1; continue

        prev = df.iloc[i-1]
        d = None; res = None

        for d_test in ["long", "short"]:
            if _check_cooldown(i, direction=d_test):
                r = _ev(row, prev, d_test, i, df)
                if r:
                    res, d = r, d_test; break

        if res is None:
            i += 1; continue

        sl, tp, atr, _ = res
        entry = float(row["close"])
        isL = (d == "long")
        _register_signal(i, direction=d)
        sl_d = max(abs(entry-sl)/entry*100, 0.05)

        csl = sl; tr_on = False; tp1f = False; tp1p_v = 0.0
        hwm = entry; wtr = False; exit_done = False

        for j in range(i+1, min(i+max_b, n)):
            f = df.iloc[j]
            fc, fl, fh = float(f["close"]), float(f["low"]), float(f["high"])
            bars = j - i
            hwm = max(hwm, fh) if isL else min(hwm, fl)
            sh = (fl <= csl) if isL else (fh >= csl)
            th = (fh >= tp) if isL else (fl <= tp)

            if th and not tp1f:
                tp1f = True; tp1p_v = tp
                if use_tr:
                    tr_on = True
                    b = atr * pbuf
                    csl = (tp - b) if isL else (tp + b)
                else:
                    _, a1, _ = _apply_costs(entry, tp, isL, fee, spr, slp)
                    pn = (a1-entry)/entry*100 if isL else (entry-a1)/entry*100
                    results.append((pn, sl_d)); exit_done = True; break

            if th and sh and tp1f:
                pn = _pp(entry, tp1p_v, csl, isL, fee, spr, slp, tp1p)
                results.append((pn, sl_d)); exit_done = True; wtr = True; break

            if sh and not th:
                if tp1f:
                    pn = _pp(entry, tp1p_v, csl, isL, fee, spr, slp, tp1p)
                else:
                    _, a2, _ = _apply_costs(entry, csl, isL, fee, spr, slp)
                    pn = (a2-entry)/entry*100 if isL else (entry-a2)/entry*100
                results.append((pn, sl_d)); exit_done = True; wtr = tr_on; break

            if tr_on and use_tr:
                td = atr * tr_dist
                if isL:
                    nt = hwm - td
                    if nt > csl: csl = nt
                else:
                    nt = hwm + td
                    if nt < csl: csl = nt

        if not exit_done:
            lj = min(i+max_b, n) - 1
            xp = float(df.iloc[lj]["close"])
            if tp1f:
                pn = _pp(entry, tp1p_v, xp, isL, fee, spr, slp, tp1p)
            else:
                _, a3, _ = _apply_costs(entry, xp, isL, fee, spr, slp)
                pn = (a3-entry)/entry*100 if isL else (entry-a3)/entry*100
            results.append((pn, sl_d))
            wtr = tr_on

        _register_signal(i + (min(i+max_b, n) - 1 - i), was_trailing=wtr, direction=d)
        i = min(i+max_b, n)

    BBWP_SQUEEZE_PARAMS.update(orig)
    return results


def _ev(row, prev, d, idx, df):
    from strategy_bbwp_squeeze import (
        _is_squeeze_breakout, _adx_confirms_trend, _stoch_rsi_confirms,
        _volume_confirms, _get_sl_mult, BBWP_SQUEEZE_PARAMS,
    )
    p = BBWP_SQUEEZE_PARAMS
    if not _is_squeeze_breakout(row, prev, idx=idx, df=df): return None
    if not _adx_confirms_trend(row): return None
    ap = float(row.get("atr_percentile", 0.5))
    if not (p["atr_pct_min"] <= ap <= p["atr_pct_max"]): return None
    if not _volume_confirms(row): return None
    cl = float(row["close"])
    bu = float(row.get("bb_upper", 0))
    bl = float(row.get("bb_lower", 0))
    e50 = float(row.get("ema50", 0))
    e200 = float(row.get("ema200", 0))
    atr = float(row.get("atr", 0))
    if atr <= 0 or cl <= 0 or bu <= 0 or bl <= 0: return None
    sm = _get_sl_mult(row)
    bb = p.get("bb_breakout_buffer", 0.05)
    if d == "long":
        if cl <= bu + bb*(bu-bl): return None
        if not _stoch_rsi_confirms(row, prev, "long"): return None
        if cl <= e50: return None
        if p.get("ema200_filter",True) and e200 > 0 and cl <= e200: return None
        sl = cl - sm*atr
        if sl <= 0: return None
        return (sl, cl + p["tp_atr_mult"]*atr, atr, float(row.get("bbwp",100)))
    else:
        if cl >= bl - bb*(bu-bl): return None
        if not _stoch_rsi_confirms(row, prev, "short"): return None
        if cl >= e50: return None
        if p.get("ema200_filter",True) and e200 > 0 and cl >= e200: return None
        sl = cl + sm*atr
        return (sl, cl - p["tp_atr_mult"]*atr, atr, float(row.get("bbwp",100)))


def _pp(e, t1, ex, isL, f, s, sl, tp1):
    from backtest import _apply_costs
    _, a1, _ = _apply_costs(e, t1, isL, f, s, sl)
    _, a2, _ = _apply_costs(e, ex, isL, f, s, sl)
    if isL: return round(tp1*(a1-e)/e*100 + (1-tp1)*(a2-e)/e*100, 4)
    else: return round(tp1*(e-a1)/e*100 + (1-tp1)*(e-a2)/e*100, 4)


def comp(trades, risk, days=730):
    if not trades: return {"eq":100,"ret":0,"ann":0,"dd":0,"n":0}
    eq, pk, md = 100.0, 100.0, 0.0
    for pn, sd in trades:
        ps = risk / sd
        eq += eq * ps * pn / 100
        eq = max(eq, 0.01)
        if eq > pk: pk = eq
        dd = (pk - eq) / pk * 100
        if dd > md: md = dd
    r = eq - 100
    return {"eq":round(eq,2),"ret":round(r,2),"ann":round(r*365/days,2),"dd":round(md,2),"n":len(trades)}


def met(trades):
    if not trades: return {"n":0,"wr":0,"pf":0}
    pnls = [p for p,_ in trades]
    w = [p for p in pnls if p > 0]
    l = [p for p in pnls if p <= 0]
    return {"n":len(pnls),"wr":round(len(w)/len(pnls)*100,1),"pf":round(sum(w)/max(abs(sum(l)),0.001),2)}


def BP(**kw):
    p = {"bbwp_threshold":15,"squeeze_recent_bars":12,"require_bbwp_expansion":True,
        "volume_mult":0.35,"stoch_rsi_ob":56,"stoch_rsi_os":44,"bb_breakout_buffer":0.05,
        "adx_min":16.0,"sl_atr_mult":2.2,"tp_atr_mult":3.0,"tp1_pct":0.50,
        "use_trailing":True,"trailing_atr_mult":1.5,"post_tp1_sl_buffer":0.5,
        "max_bars_held":96,"cooldown":2,"cooldown_trailing":2,"cooldown_opp_dir":1,
        "use_directional_cooldown":True,"ema200_filter":True,"atr_pct_min":0.10,
        "atr_pct_max":0.90,"sl_atr_mult_high_vol":2.2,"sl_atr_mult_low_vol":2.2,
        "stoch_rsi_cross_enable":True,"stoch_rsi_min_delta":0,"min_bbwp_bars":1,
        "be_trigger_atr_mult":1.0,"use_divergence_exit":False,"divergence_min_bars":3}
    p.update(kw); return p


def main():
    print("="*80, flush=True)
    print("REFINE v14 - Sub-period Stability Validation", flush=True)
    print("="*80, flush=True)

    df = load_data()
    n = len(df)
    print(f"Range: {df.index[0]} to {df.index[-1]}", flush=True)

    tps = [5.0, 5.5, 6.0, 6.5, 7.0]
    trs = [2.0, 2.5, 3.0]
    bufs = [0.1, 0.2, 0.3]
    sls = [2.0, 2.2]
    risks = [2.0, 2.5, 3.0]

    grid = [(tp, tr, buf, sl) for tp in tps for tr in trs for buf in bufs for sl in sls]
    print(f"Grid: {len(grid)} param combos x {len(risks)} risk = {len(grid)*len(risks)}", flush=True)

    t0 = time.time()

    # Phase 1: Full-period backtests
    print("\n--- PHASE 1: Full 730d backtests ---", flush=True)
    full_results = []
    for gi, (tp, tr, buf, sl) in enumerate(grid):
        p = BP(tp_atr_mult=tp, trailing_atr_mult=tr, post_tp1_sl_buffer=buf,
               sl_atr_mult=sl, sl_atr_mult_high_vol=sl, sl_atr_mult_low_vol=sl,
               volume_mult=0.25, cooldown=1, cooldown_trailing=1)
        try:
            trades = run_bt(df, p)
            m = met(trades)
            rr = []
            for risk in risks:
                c = comp(trades, risk)
                sc = c['ann'] / max(c['dd'], 1.0)
                rr.append({'risk': risk, 'ann': c['ann'], 'ret': c['ret'], 'dd': c['dd'],
                           'eq': c['eq'], 'n': m['n'], 'wr': m['wr'], 'pf': m['pf'], 'score': round(sc,2)})
            full_results.append({'tp': tp, 'tr': tr, 'buf': buf, 'sl': sl, 'params': p,
                'trades': trades, 'met': m, 'risk_results': rr})
        except Exception as e:
            print(f"  ERR {tp}_{tr}_{buf}_{sl}: {e}", flush=True)
        if (gi+1) % 15 == 0:
            el = time.time()-t0
            print(f"  [{gi+1}/{len(grid)}] {el:.1f}s", flush=True)

    print(f"Phase 1: {time.time()-t0:.1f}s | {len(full_results)} successful", flush=True)

    # Save phase 1 for resilience
    phase1_save = []
    for r in full_results:
        phase1_save.append({
            'tp': r['tp'], 'tr': r['tr'], 'buf': r['buf'], 'sl': r['sl'],
            'n_trades': r['met']['n'], 'wr': r['met']['wr'], 'pf': r['met']['pf'],
            'risk_results': r['risk_results']
        })
    with open(PHASE1_CACHE, 'w') as f:
        json.dump(phase1_save, f, indent=2)

    # Filter >=50% annual with quality gates
    qualified = []
    for r in full_results:
        m = r['met']
        if m['n'] < 30 or m['wr'] < 35:
            continue
        for rr in r['risk_results']:
            if rr['ann'] >= 50 and rr['dd'] <= 40:
                qualified.append((r, rr))

    print(f"Qualified (>=50% ann, WR>=35, n>=30, DD<=40): {len(qualified)}", flush=True)

    # Phase 2: Sub-period validation
    print("\n--- PHASE 2: Sub-period validation ---", flush=True)
    hpd = 24
    slices = [
        ('365d', n - 365*hpd, n, 365),
        ('180d', n - 180*hpd, n, 180),
        ('90d',  n - 90*hpd,  n, 90),
    ]

    t1 = time.time()
    all_qualified = []
    for qi, (r, rr) in enumerate(qualified):
        tp, tr, buf, sl = r['tp'], r['tr'], r['buf'], r['sl']
        risk = rr['risk']
        label = f'TP{tp}_TR{tr}_BUF{buf}_SL{sl}'
        p = r['params']

        sub_results = {}
        all_positive = True
        for sname, start, end, days in slices:
            if start < 0:
                all_positive = False
                sub_results[sname] = {'ann': None, 'dd': None, 'n': 0, 'wr': 0, 'pf': 0}
                continue
            sub_df = df.iloc[start:end].reset_index(drop=True)
            try:
                sub_trades = run_bt(sub_df, p)
                sub_c = comp(sub_trades, risk, days=days)
                sub_m = met(sub_trades)
                sub_results[sname] = {
                    'ann': sub_c['ann'], 'ret': sub_c['ret'], 'dd': sub_c['dd'],
                    'eq': sub_c['eq'], 'n': sub_m['n'], 'wr': sub_m['wr'], 'pf': sub_m['pf']
                }
                if sub_c['ann'] <= 0:
                    all_positive = False
            except Exception as e:
                all_positive = False
                sub_results[sname] = {'ann': None, 'dd': None, 'n': 0, 'wr': 0, 'pf': 0, 'error': str(e)}

        entry = {
            'label': label, 'tp': tp, 'tr': tr, 'buf': buf, 'sl': sl, 'risk': risk,
            'score': rr['score'], 'stable': all_positive,
            'full': {'ann': rr['ann'], 'ret': rr['ret'], 'dd': rr['dd'], 'eq': rr['eq'],
                     'n': rr['n'], 'wr': rr['wr'], 'pf': rr['pf']},
            'sub_365d': sub_results.get('365d'),
            'sub_180d': sub_results.get('180d'),
            'sub_90d': sub_results.get('90d'),
        }
        all_qualified.append(entry)

        if (qi+1) % 5 == 0:
            el = time.time()-t1
            sc = sum(1 for x in all_qualified if x['stable'])
            print(f"  [{qi+1}/{len(qualified)}] {el:.1f}s stable={sc}", flush=True)

    stable_count = sum(1 for x in all_qualified if x['stable'])
    print(f"Phase 2: {time.time()-t1:.1f}s | stable: {stable_count}/{len(all_qualified)}", flush=True)

    # Sort by score
    all_qualified.sort(key=lambda x: x['score'], reverse=True)

    # Print TOP 5
    print(f"\n{'='*80}", flush=True)
    print(f"TOP 5 by score (annual_return / max_dd)", flush=True)
    print(f"{stable_count} stable of {len(all_qualified)} qualified", flush=True)
    print(f"{'='*80}", flush=True)

    display = [x for x in all_qualified if x['stable']][:5]
    if len(display) < 5:
        display += [x for x in all_qualified if not x['stable']][:5-len(display)]
    if len(display) == 0:
        display = all_qualified[:5]

    for i, r in enumerate(display):
        f = r['full']
        s365 = r.get('sub_365d') or {}
        s180 = r.get('sub_180d') or {}
        s90 = r.get('sub_90d') or {}
        stab = 'STABLE' if r['stable'] else 'UNSTABLE'
        print(f"\n#{i+1} [{r['label']}] risk={r['risk']}% score={r['score']} {stab}", flush=True)
        print(f"  730d: ANN={f['ann']:+.1f}% RET={f['ret']:+.1f}% EQ={f['eq']} DD={f['dd']:.1f}% T={f['n']} WR={f['wr']:.1f}% PF={f['pf']:.2f}", flush=True)
        if s365.get('ann') is not None:
            print(f"  365d: ANN={s365['ann']:+.1f}% DD={s365['dd']:.1f}% T={s365['n']} WR={s365['wr']:.1f}% PF={s365['pf']:.2f}", flush=True)
        if s180.get('ann') is not None:
            print(f"  180d: ANN={s180['ann']:+.1f}% DD={s180['dd']:.1f}% T={s180['n']} WR={s180['wr']:.1f}% PF={s180['pf']:.2f}", flush=True)
        if s90.get('ann') is not None:
            print(f"   90d: ANN={s90['ann']:+.1f}% DD={s90['dd']:.1f}% T={s90['n']} WR={s90['wr']:.1f}% PF={s90['pf']:.2f}", flush=True)
        print(f"  TP={r['tp']}x SL={r['sl']}x TR={r['tr']}x BUF={r['buf']}", flush=True)

    # Save final results
    out = {
        'description': 'Refine v14 — sub-period stability validation',
        'total_param_combos': len(grid),
        'total_tested': len(grid)*len(risks),
        'qualified_50pct': len(qualified),
        'stable_count': stable_count,
        'top5': display,
        'all_qualified': all_qualified,
    }
    with open(OUTPUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUTPUT}", flush=True)
    print(f"Total: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
