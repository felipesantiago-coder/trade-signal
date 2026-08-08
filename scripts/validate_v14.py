r"""
validate_v14.py — Validacao por sub-periodos da estrategia v14.
Testa 730d completo + 365d + 180d + 90d.
"""
import sys, os, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging; logging.basicConfig(level=logging.WARNING)

CACHE = "/home/z/my-project/trade-signal/download/btc_1h_cache.csv"


def load_data():
    df = pd.read_csv(CACHE, index_col=0, parse_dates=True)
    print(f"Cache: {len(df):,} candles ({df.index[0].date()} a {df.index[-1].date()})")
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

    if end_idx is None: end_idx = len(df)
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
        if i < 1: i += 1; continue
        cr = ["ema20","ema50","ema200","rsi","atr","atr_percentile",
             "bbwp","stoch_rsi_k","stoch_rsi_d","bb_lower","bb_upper","volume","volume_sma20","adx"]
        if any(pd.isna(row.get(c)) for c in cr): i += 1; continue
        ap = float(row.get("atr_percentile", 0.5))
        if ap < 0.10 or ap > 0.90: i += 1; continue

        prev = df_sub.iloc[i-1]
        d = None; res = None
        for d_test in ["long", "short"]:
            if _check_cooldown(i, direction=d_test):
                r = _ev(row, prev, d_test, i, df_sub)
                if r: res, d = r, d_test; break
        if res is None: i += 1; continue

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
    if not trades: return {"n":0,"wr":0,"pf":0,"pnl":0}
    pnls = [p for p,_ in trades]
    w = [p for p in pnls if p > 0]
    l = [p for p in pnls if p <= 0]
    return {"n":len(pnls),"wr":round(len(w)/len(pnls)*100,1),"pf":round(sum(w)/max(abs(sum(l)),0.001),2),"pnl":round(sum(pnls),2)}


def BP(**kw):
    p = {"bbwp_threshold":15,"squeeze_recent_bars":12,"require_bbwp_expansion":True,
        "volume_mult":0.35,"stoch_rsi_ob":56,"stoch_rsi_os":44,"bb_breakout_buffer":0.05,
        "adx_min":16.0,"sl_atr_mult":2.2,"tp_atr_mult":6.0,"tp1_pct":0.50,
        "use_trailing":True,"trailing_atr_mult":2.5,"post_tp1_sl_buffer":0.2,
        "max_bars_held":96,"cooldown":2,"cooldown_trailing":2,"cooldown_opp_dir":1,
        "use_directional_cooldown":True,"ema200_filter":True,"atr_pct_min":0.10,
        "atr_pct_max":0.90,"sl_atr_mult_high_vol":2.2,"sl_atr_mult_low_vol":2.2,
        "stoch_rsi_cross_enable":True,"stoch_rsi_min_delta":0,"min_bbwp_bars":1,
        "be_trigger_atr_mult":1.0,"use_divergence_exit":False,"divergence_min_bars":3}
    p.update(kw); return p


def main():
    print("="*70)
    print("VALIDACAO v14 — Sub-periodos")
    print("="*70)
    df = load_data()
    n = len(df)

    # v14 params (agora ja sao o default no codigo)
    v14 = BP()
    # Tambem testar variantes proximas
    configs = [
        ("v14_base", v14),
        ("v14_tp5.5_tr2.5_b0.2", BP(tp_atr_mult=5.5, trailing_atr_mult=2.5, post_tp1_sl_buffer=0.2)),
        ("v14_tp6.5_tr2.5_b0.2", BP(tp_atr_mult=6.5, trailing_atr_mult=2.5, post_tp1_sl_buffer=0.2)),
        ("v14_tp6_tr3.0_b0.2", BP(tp_atr_mult=6.0, trailing_atr_mult=3.0, post_tp1_sl_buffer=0.2)),
        ("v14_tp6_tr2.5_b0.3", BP(tp_atr_mult=6.0, trailing_atr_mult=2.5, post_tp1_sl_buffer=0.3)),
        ("v14_tp6_sl2.0", BP(tp_atr_mult=6.0, sl_atr_mult=2.0, sl_atr_mult_high_vol=2.0, sl_atr_mult_low_vol=2.0)),
    ]

    periods = {
        "730d": (0, n),
        "365d": (n//2, n),
        "180d": (n - 180*24, n),
        "90d": (n - 90*24, n),
    }

    print(f"\nTestando {len(configs)} configs x {len(periods)} periodos x 3 risk levels\n")
    t0 = time.time()
    results = []

    for cname, params in configs:
        for pname, (si, ei) in periods.items():
            try:
                tp = run_bt(df, params, start_idx=si, end_idx=ei)
                m = met(tp)
                days = (ei - si) / 24
                for risk in [2.0, 2.5, 3.0]:
                    c = comp(tp, risk, days=days)
                    sc = c["ann"] / max(c["dd"], 1)
                    results.append({
                        "cfg": cname, "period": pname, "risk": risk,
                        "ann": c["ann"], "dd": c["dd"], "eq": c["eq"],
                        "n": m["n"], "wr": m["wr"], "pf": m["pf"],
                        "pnl": m["pnl"], "score": round(sc, 1),
                    })
            except Exception as e:
                print(f"  ERRO {cname}/{pname}: {e}")

        el = time.time() - t0
        print(f"  [{cname}] feito em {el:.0f}s")

    print(f"\nTotal: {time.time()-t0:.1f}s, {len(results)} resultados")

    # Analise: por config, pior periodo
    print(f"\n{'='*70}")
    print("RESUMO POR CONFIG (risk=2.0%)")
    print(f"{'='*70}")

    for cname, _ in configs:
        rows_2 = [r for r in results if r["cfg"] == cname and r["risk"] == 2.0]
        if not rows_2: continue
        anns = {r["period"]: r["ann"] for r in rows_2}
        dds = {r["period"]: r["dd"] for r in rows_2}
        ntrades = {r["period"]: r["n"] for r in rows_2}
        pfs = {r["period"]: r["pf"] for r in rows_2}
        wrs = {r["period"]: r["wr"] for r in rows_2}
        min_ann = min(anns.values())
        all_pos = all(a > 0 for a in anns.values())
        tag = "STABLE" if all_pos else "UNSTABLE"

        print(f"\n{cname} [{tag}] risk=2%:")
        for p in ["730d","365d","180d","90d"]:
            if p in anns:
                print(f"  {p}: ANUAL={anns[p]:+6.1f}% DD={dds[p]:5.1f}% | {ntrades[p]:3d}T WR={wrs[p]:.1f}% PF={pfs[p]:.2f}")
        print(f"  Pior periodo: {min(anns, key=anns.get)} ({min_ann:+.1f}%)")

    # Risk comparison for base v14
    print(f"\n{'='*70}")
    print("v14_base — Comparacao por risco:")
    print(f"{'='*70}")
    for risk in [2.0, 2.5, 3.0]:
        rows = [r for r in results if r["cfg"] == "v14_base" and r["risk"] == risk]
        for r in rows:
            if r["period"] == "730d":
                print(f"  risk={risk}%: ANUAL={r['ann']:+.1f}% DD={r['dd']:.1f}% EQ={r['eq']} PF={r['pf']:.2f}")

    # Save
    out = {"results": results}
    with open("/home/z/my-project/trade-signal/download/validate_v14_results.json","w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSalvo: validate_v14_results.json")


if __name__ == "__main__":
    main()
