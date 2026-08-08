r"""
explore_50pct_v2.py - Busca >=50% retorno anual equilibrado.
Cache de dados + grid otimizado.
"""
import sys, os, json, time, logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING)

CACHE = "/home/z/my-project/trade-signal/download/btc_1h_cache.csv"


def load_data():
    if os.path.exists(CACHE):
        df = pd.read_csv(CACHE, index_col=0, parse_dates=True)
        print(f"Cache: {len(df):,} candles"); return df
    from indicators import compute_indicators
    from backtest import fetch_historical_ohlcv
    print("Baixando dados...")
    t0 = time.time()
    df = fetch_historical_ohlcv("BTC/USDT", "1h", 730)
    print(f"  {len(df):,} candles em {time.time()-t0:.1f}s")
    df = compute_indicators(df, timeframe="1h")
    df = df.dropna(subset=["ema20","ema50","ema200","rsi","atr","atr_percentile",
        "bbwp","stoch_rsi_k","stoch_rsi_d","bb_lower","bb_upper","volume","volume_sma20","adx"]).copy()
    df.to_csv(CACHE)
    print(f"  {len(df):,} limpos, cache salvo")
    return df


def run_bt(df, P):
    """Run BBWP backtest with params dict P. Returns list of (pnl_pct, sl_dist_pct)."""
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
    p_func = BBWP_SQUEEZE_PARAMS.get

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


def grid():
    g = []
    # A: Wider TP + trailing
    for tp in [4.0,5.0,6.0]:
        for tr in [2.5,3.0,4.0]:
            for b in [0.2,0.3,0.5]:
                g.append((f"A_{tp}_{tr}_{b}",BP(tp_atr_mult=tp,trailing_atr_mult=tr,post_tp1_sl_buffer=b)))
    # B: Tighter SL + wider TP
    for sl in [1.5,1.8,2.0]:
        for tp in [4.0,5.0,6.0]:
            g.append((f"B_{sl}_{tp}",BP(sl_atr_mult=sl,sl_atr_mult_high_vol=sl,sl_atr_mult_low_vol=sl,tp_atr_mult=tp,trailing_atr_mult=2.5)))
    # C: Relaxed entries + wider exits
    for v in [0.20,0.25]:
        for ob in [52,54]:
            g.append((f"C_{v}_{ob}",BP(volume_mult=v,stoch_rsi_ob=ob,stoch_rsi_os=100-ob,tp_atr_mult=5.0,trailing_atr_mult=3.0,cooldown=1,cooldown_trailing=1,post_tp1_sl_buffer=0.3)))
    # D: No EMA200
    g.append(("D1",BP(ema200_filter=False,tp_atr_mult=5.0,trailing_atr_mult=3.0,cooldown=1,cooldown_trailing=1,post_tp1_sl_buffer=0.3)))
    # E: Combined F (most promising)
    for tp in [5.0,6.0]:
        for tr in [3.0,4.0]:
            for sl in [1.5,1.8,2.0]:
                for v in [0.20,0.25]:
                    g.append((f"E_{tp}_{tr}_{sl}_{v}",BP(tp_atr_mult=tp,trailing_atr_mult=tr,
                        sl_atr_mult=sl,sl_atr_mult_high_vol=sl,sl_atr_mult_low_vol=sl,
                        volume_mult=v,cooldown=1,cooldown_trailing=1,post_tp1_sl_buffer=0.3)))
    # F: Ultra trailing
    for tr in [4.0,5.0]:
        g.append((f"F_{tr}",BP(trailing_atr_mult=tr,post_tp1_sl_buffer=0.2,volume_mult=0.25,
            cooldown=1,cooldown_trailing=1,tp_atr_mult=4.0)))
    # G: BBWP threshold
    for bwp in [10,12,18,20]:
        g.append((f"G_{bwp}",BP(bbwp_threshold=bwp,tp_atr_mult=5.0,trailing_atr_mult=3.0,
            volume_mult=0.25,cooldown=1,cooldown_trailing=1,post_tp1_sl_buffer=0.3)))
    # H: Everything combined aggressive
    for tp in [6.0,7.0,8.0]:
        for tr in [4.0,5.0]:
            g.append((f"H_{tp}_{tr}",BP(tp_atr_mult=tp,trailing_atr_mult=tr,post_tp1_sl_buffer=0.2,
                volume_mult=0.20,cooldown=1,cooldown_trailing=1,sl_atr_mult=1.5,
                sl_atr_mult_high_vol=1.5,sl_atr_mult_low_vol=1.5,adx_min=12.0,
                stoch_rsi_ob=52,stoch_rsi_os=48,bbwp_threshold=12)))
    return g


def main():
    print("="*70)
    print("EXPLORACAO EQUILIBRADA v2 — Busca >=50% Retorno Anual")
    print("="*70)
    df = load_data()
    g = grid()
    print(f"\n{len(g)} combinacoes\n")
    t0 = time.time()
    qual = []; best = []
    for idx,(nm,p) in enumerate(g):
        try:
            tp = run_bt(df, p)
            m = met(tp)
            for risk in [2.0,3.0,4.0,5.0]:
                c = comp(tp, risk)
                sc = c["ann"] / max(c["dd"], 1)
                e = {"nm":nm,"p":p,"r":risk,"m":m,"c":c,"sc":round(sc,1)}
                if c["ann"]>=50 and m["n"]>=30 and m["wr"]>=35 and c["dd"]<=40:
                    qual.append(e)
                if len(best)<100 or c["ann"]>best[-1]["c"]["ann"]:
                    best.append(e)
                    best.sort(key=lambda x:x["sc"],reverse=True)
                    best=best[:100]
        except: pass
        if (idx+1)%30==0 or idx==0:
            el=time.time()-t0; sp=(idx+1)/max(el,0.01)
            print(f"  [{idx+1}/{len(g)}] {sp:.1f}/s qual={len(qual)}")
    print(f"\nFeito em {time.time()-t0:.1f}s")
    pool = sorted(qual,key=lambda x:x["sc"],reverse=True) if qual else best
    tag = f"{len(qual)} >=50% EQUILIBRADAS" if qual else f"TOP GERAIS (nenhuma >=50% equilibrada)"
    print(f"\n{'='*70}\n{tag}\n{'='*70}")
    for i, r in enumerate(pool[:30]):
        c, m, p = r["c"], r["m"], r["p"]
        print(f"\n#{i+1} [{r['nm']}] risk={r['r']}% score={r['sc']}")
        print(f"  ANUAL={c['ann']:+.1f}% TOTAL={c['ret']:+.1f}% EQ={c['eq']} DD={c['dd']:.1f}%")
        print(f"  Trades={m['n']} WR={m['wr']:.1f}% PF={m['pf']:.2f}")
        print(f"  TP={p['tp_atr_mult']}x SL={p['sl_atr_mult']}x TR={p['trailing_atr_mult']}x BUF={p['post_tp1_sl_buffer']}")
        print(f"  VOL={p['volume_mult']} OB={p['stoch_rsi_ob']} ADX>={p['adx_min']} BBWP<{p['bbwp_threshold']} CD={p['cooldown']} EMA200={'ON' if p['ema200_filter'] else 'OFF'}")
    out={"tested":len(g),"qualified":len(qual),"top":[]}
    for r in pool[:10]:
        out["top"].append({"name":r["nm"],"risk":r["r"],"ann":r["c"]["ann"],"ret":r["c"]["ret"],
            "dd":r["c"]["dd"],"eq":r["c"]["eq"],"trades":r["m"]["n"],
            "wr":r["m"]["wr"],"pf":r["m"]["pf"],
            "params":{k:v for k,v in r["p"].items() if k in [
                "tp_atr_mult","sl_atr_mult","trailing_atr_mult","post_tp1_sl_buffer",
                "volume_mult","stoch_rsi_ob","stoch_rsi_os","adx_min","cooldown",
                "cooldown_trailing","cooldown_opp_dir","ema200_filter","bbwp_threshold"]}})
    with open("/home/z/my-project/trade-signal/download/explore_50pct_v2.json","w") as f:
        json.dump(out,f,indent=2)
    print(f"\nSalvo: explore_50pct_v2.json")

if __name__=="__main__":
    main()
