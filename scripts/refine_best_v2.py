r"""
refine_best_v2.py — Refinamento fino + validacao por periodo.
Foca na melhor regiao: TP=5-7x, Trail=2-3x, Buf=0.1-0.3, SL=1.8-2.5.
Testa estabilidade em sub-periodos (365d, 180d, 90d).
"""
import sys, os, json, time, logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING)

CACHE = "/home/z/my-project/trade-signal/download/btc_1h_cache.csv"


def load_data():
    df = pd.read_csv(CACHE, index_col=0, parse_dates=True)
    print(f"Cache: {len(df):,} candles")
    return df


def run_bt(df, P, start_idx=0, end_idx=None):
    from strategy_bbwp_squeeze import (
        BBWP_SQUEEZE_PARAMS, reset_cooldown, _check_cooldown,
        _register_signal, _is_squeeze_breakout, _adx_confirms_trend,
        _stoch_rsi_confirms, _volume_confirms, _get_sl_mult,
    )
    from backtest import _apply_costs

    orig = dict(BBWP_SQUEEZE_PARAMS)
    BBWP_SQUEEZE_PARAMS.update(P)
    reset_cooldown()

    if end_idx is None:
        end_idx = len(df)
    df_sub = df.iloc[start_idx:end_idx]
    results = []
    n = len(df_sub)
    tr_dist = P.get("trailing_atr_mult", 1.5)
    max_b = P.get("max_bars_held", 96)
    use_tr = P.get("use_trailing", True)
    tp1p = P.get("tp1_pct", 0.50)
    pbuf = P.get("post_tp1_sl_buffer", 0.5)
    fee, spr, slp = 0.016, 2.0, 2.0

    i = 0
    while i < n:
        row = df_sub.iloc[i]
        if i < 1:
            i += 1; continue
        cr = ["ema20","ema50","ema200","rsi","atr","atr_percentile",
             "bbwp","stoch_rsi_k","stoch_rsi_d","bb_lower","bb_upper","volume","volume_sma20","adx"]
        if any(pd.isna(row.get(c)) for c in cr):
            i += 1; continue
        ap = float(row.get("atr_percentile", 0.5))
        if ap < 0.10 or ap > 0.90:
            i += 1; continue

        prev = df_sub.iloc[i-1]
        d = None; res = None
        for d_test in ["long", "short"]:
            if _check_cooldown(i, direction=d_test):
                r = _ev(row, prev, d_test, i, df_sub)
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
            f = df_sub.iloc[j]
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
            xp = float(df_sub.iloc[lj]["close"])
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
    print("="*70)
    print("REFINAMENTO v14 — Regiao TP=5-7x, Trail=2-3x")
    print("="*70)
    df = load_data()
    n = len(df)
    print(f"Periodo: {df.index[0]} a {df.index[-1]} ({n:,} candles)")

    # Define sub-periods
    periods = {
        "730d": (0, n),
        "365d": (n//2, n),
        "180d": (n - 180*24, n),
        "90d": (n - 90*24, n),
    }

    # Refinement grid around best region
    g = []
    for tp in [5.0, 5.5, 6.0, 6.5, 7.0]:
        for tr in [2.0, 2.5, 3.0]:
            for buf in [0.1, 0.2, 0.3]:
                for sl in [2.0, 2.2]:
                    g.append((f"tp{tp}_tr{tr}_b{buf}_sl{sl}",
                        BP(tp_atr_mult=tp, trailing_atr_mult=tr, post_tp1_sl_buffer=buf,
                           sl_atr_mult=sl, sl_atr_mult_high_vol=sl, sl_atr_mult_low_vol=sl)))

    # Also test with cooldown=1
    for tp in [5.5, 6.0, 6.5]:
        for tr in [2.0, 2.5, 3.0]:
            g.append((f"tp{tp}_tr{tr}_b0.2_sl2.2_cd1",
                BP(tp_atr_mult=tp, trailing_atr_mult=tr, post_tp1_sl_buffer=0.2,
                   cooldown=1, cooldown_trailing=1)))

    print(f"\n{len(g)} combinacoes no refinamento\n")

    # Run grid on full period first
    t0 = time.time()
    results = []

    for idx, (nm, p) in enumerate(g):
        try:
            tp_full = run_bt(df, p)
            m = met(tp_full)
            # Only test configs with reasonable PF on full period
            if m["pf"] < 1.2 or m["n"] < 50:
                continue

            for risk in [2.0, 2.5, 3.0]:
                c_full = comp(tp_full, risk, days=730)

                # Test sub-periods
                sub_results = {}
                all_stable = True
                for pname, (si, ei) in periods.items():
                    if pname == "730d":
                        sub_results[pname] = c_full
                        continue
                    tp_sub = run_bt(df, p, start_idx=si, end_idx=ei)
                    days_sub = (ei - si) / 24  # approximate trading days
                    c_sub = comp(tp_sub, risk, days=max(days_sub, 1))
                    sub_results[pname] = c_sub
                    if c_sub["ann"] < 20:  # must be positive in all sub-periods
                        all_stable = False

                sc = c_full["ann"] / max(c_full["dd"], 1)
                r = {
                    "nm": nm, "p": p, "risk": risk,
                    "m": m, "c": c_full, "sc": round(sc, 1),
                    "sub": sub_results, "stable": all_stable,
                }
                results.append(r)

                if c_full["ann"] >= 50:
                    tag = "STABLE" if all_stable else "unstable"
                    print(f"  [{tag}] {nm} r={risk}% | 730d={c_full['ann']:+.1f}% DD={c_full['dd']:.1f}% | "
                          f"365d={sub_results['365d']['ann']:+.1f}% | "
                          f"180d={sub_results['180d']['ann']:+.1f}% | "
                          f"90d={sub_results['90d']['ann']:+.1f}%")
        except:
            pass

        if (idx+1) % 30 == 0:
            el = time.time()-t0
            print(f"  [{idx+1}/{len(g)}] {el:.0f}s")

    print(f"\nFeito em {time.time()-t0:.1f}s")

    # Sort: prioritize stability, then score
    stable = [r for r in results if r["stable"] and r["c"]["ann"] >= 50]
    stable.sort(key=lambda x: x["sc"], reverse=True)

    all_res = sorted(results, key=lambda x: x["sc"], reverse=True)

    if stable:
        print(f"\n{'='*70}")
        print(f"CONFIGURACOES ESTAVEIS >=50% ANUAL: {len(stable)}")
        print(f"(positivas em todos sub-periodos)")
        print(f"{'='*70}")

        for i, r in enumerate(stable[:15]):
            c, m, p, sub = r["c"], r["m"], r["p"], r["sub"]
            print(f"\n#{i+1} [{r['nm']}] risk={r['risk']}% score={r['sc']}")
            print(f"  730d: ANUAL={c['ann']:+.1f}% DD={c['dd']:.1f}% EQ={c['eq']} | {m['n']}T WR={m['wr']:.1f}% PF={m['pf']:.2f}")
            print(f"  365d: ANUAL={sub['365d']['ann']:+.1f}% DD={sub['365d']['dd']:.1f}% | {sub['365d']['n']}T")
            print(f"  180d: ANUAL={sub['180d']['ann']:+.1f}% DD={sub['180d']['dd']:.1f}% | {sub['180d']['n']}T")
            print(f"   90d: ANUAL={sub['90d']['ann']:+.1f}% DD={sub['90d']['dd']:.1f}% | {sub['90d']['n']}T")
            print(f"  TP={p['tp_atr_mult']}x SL={p['sl_atr_mult']}x TR={p['trailing_atr_mult']}x BUF={p['post_tp1_sl_buffer']}")
            print(f"  CD={p['cooldown']}/CDT={p['cooldown_trailing']}/CDO={p['cooldown_opp_dir']} EMA200={'ON' if p['ema200_filter'] else 'OFF'}")
    else:
        print(f"\nNenhuma config estavel >=50%. Melhores gerais:")
        for i, r in enumerate(all_res[:10]):
            c, m, p, sub = r["c"], r["m"], r["p"], r["sub"]
            print(f"\n#{i+1} [{r['nm']}] risk={r['risk']}% score={r['sc']}")
            print(f"  730d: {c['ann']:+.1f}% DD={c['dd']:.1f}% | {m['n']}T WR={m['wr']:.1f}% PF={m['pf']:.2f}")
            print(f"  365d: {sub['365d']['ann']:+.1f}% | 180d: {sub['180d']['ann']:+.1f}% | 90d: {sub['90d']['ann']:+.1f}%")

    # Save
    pool = stable if stable else all_res
    out = {"refined": len(stable)>0, "total": len(pool), "best": []}
    for r in pool[:5]:
        out["best"].append({
            "name": r["nm"], "risk": r["risk"],
            "annual": r["c"]["ann"], "dd": r["c"]["dd"], "eq": r["c"]["eq"],
            "trades": r["m"]["n"], "wr": r["m"]["wr"], "pf": r["m"]["pf"],
            "sub_365d_ann": r["sub"]["365d"]["ann"],
            "sub_180d_ann": r["sub"]["180d"]["ann"],
            "sub_90d_ann": r["sub"]["90d"]["ann"],
            "stable": r["stable"],
            "params": {k:v for k,v in r["p"].items()
                      if k in ["tp_atr_mult","sl_atr_mult","trailing_atr_mult",
                               "post_tp1_sl_buffer","volume_mult","stoch_rsi_ob",
                               "stoch_rsi_os","adx_min","cooldown","cooldown_trailing",
                               "cooldown_opp_dir","ema200_filter","bbwp_threshold"]},
        })
    with open("/home/z/my-project/trade-signal/download/refine_v14_results.json","w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSalvo: refine_v14_results.json")


if __name__ == "__main__":
    main()
