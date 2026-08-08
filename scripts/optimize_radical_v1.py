import sys, os, json, time, math
import numpy as NP, pandas as PD
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from indicators import compute_indicators

FEE, SPR, SLP = 0.016, 2.0, 2.0

def apply_costs(e, x, lg):
    c = (SPR+SLP)/10000
    if lg: ae=e*(1+c)+e*(1+c)*FEE/100; ax=x*(1-c)-x*(1-c)*FEE/100
    else: ae=e*(1-c)-e*(1-c)*FEE/100; ax=x*(1+c)+x*(1+c)*FEE/100
    return ae, ax

def sim_trade(df, idx, ep, sl, tp, atr, lg, t1p=0.5, trail=2.5, buf=0.2, maxb=96):
    n=len(df); csl=sl; hf=ep; t1f=False; t1px=0.0; be=ta=False
    for j in range(idx+1, min(idx+maxb, n)):
        fc,fl,fh=float(df.iloc[j]['close']),float(df.iloc[j]['low']),float(df.iloc[j]['high'])
        b=j-idx; hf=max(hf,fh) if lg else min(hf,fl)
        slh=(fl<=csl) if lg else (fh>=csl); tph=(fh>=tp) if lg else (fl<=tp)
        if tph and not t1f:
            t1f=True; t1px=tp; be=ta=True
            csl=(tp-buf*atr) if lg else (tp+buf*atr)
        if tph and slh and t1f:
            _,a1=apply_costs(ep,t1px,lg); _,a2=apply_costs(ep,csl,lg)
            if lg: p1=(a1-ep)/ep*100; p2=(a2-ep)/ep*100
            else: p1=(ep-a1)/ep*100; p2=(ep-a2)/ep/100
            return t1p*p1+(1-t1p)*p2,"t1sl",b,be,ta,t1f
        if slh and not tph:
            if t1f:
                _,a1=apply_costs(ep,t1px,lg); _,a2=apply_costs(ep,csl,lg)
                if lg: p1=(a1-ep)/ep*100; p2=(a2-ep)/ep*100
                else: p1=(ep-a1)/ep*100; p2=(ep-a2)/ep*100
                return t1p*p1+(1-t1p)*p2,"tsl",b,be,ta,t1f
            else:
                _,ax=apply_costs(ep,csl,lg); pnl=(ax-ep)/ep*100 if lg else (ep-ax)/ep*100
                return pnl,"sl",b,False,False,False
        if ta:
            td=atr*trail
            if lg:
                nt=hf-td
                if nt>csl: csl=nt
            else:
                nt=hf+td
                if nt<csl: csl=nt
    lj=min(idx+maxb,n)-1; ec=float(df.iloc[lj]['close']); b=lj-idx
    if t1f:
        _,a1=apply_costs(ep,t1px,lg); _,a2=apply_costs(ep,ec,lg)
        if lg: p1=(a1-ep)/ep*100; p2=(a2-ep)/ep*100
        else: p1=(ep-a1)/ep*100; p2=(ep-a2)/ep*100
        return t1p*p1+(1-t1p)*p2,"t1to",b,be,ta,t1f
    else:
        _,ax=apply_costs(ep,ec,lg); pnl=(ax-ep)/ep*100 if lg else (ep-ax)/ep*100
        return pnl,"to",b,False,False,False

def calc_metrics(t):
    if not t: return {"t":0,"wr":0,"pf":0,"pnl":0,"dd":0,"sh":0,"aw":0,"al":0,"ar":0}
    pnls=[x[0] for x in t]; w=[p for p in pnls if p>0]; lo=[p for p in pnls if p<=0]
    tpnl=sum(pnls); wr=len(w)/len(pnls)*100
    gw=sum(w) if w else 0; gl=abs(sum(lo)) if lo else .001; pf=gw/gl
    eq=pk=dd=0
    for p in pnls: eq+=p; pk=max(pk,eq); dd=max(dd,pk-eq)
    sh=NP.mean(pnls)/(NP.std(pnls)+.001)*math.sqrt(365*24/max(NP.mean([x[2] for x in t]),1)) if len(pnls)>1 else 0
    ar=((1+tpnl/100)**(365/730)-1)*100 if tpnl>0 else tpnl
    return {"t":len(pnls),"wr":round(wr,1),"pf":round(pf,2),"pnl":round(tpnl,2),"dd":round(dd,2),"sh":round(sh,2),"aw":round(NP.mean(w),3) if w else 0,"al":round(NP.mean(lo),3) if lo else 0,"ar":round(ar,1)}

def run_backtest(df, sig_fn, p, maxb=96):
    trades=[]; i=0; n=len(df); last_d=""; last_e=-999
    cs=p.get("cd_s",2); co=p.get("cd_o",1)
    while i<n-1:
        row=df.iloc[i]; prev=df.iloc[i-1] if i>0 else row
        cr=["close","atr","rsi","ema20","ema50","ema200","bb_lower","bb_upper","bbwp","stoch_rsi_k","volume","volume_sma20","adx"]
        if any(PD.isna(row.get(c)) for c in cr): i+=1; continue
        sig=sig_fn(row,prev,i,df,p)
        if sig is None: i+=1; continue
        is_long,sl,tp,atr,d=sig
        if last_e>=0:
            bs=i-last_e
            if d==last_d and bs<cs: i+=1; continue
            if d!=last_d and bs<co: i+=1; continue
        ep=float(row["close"])
        r=sim_trade(df,i,ep,sl,tp,atr,is_long,t1p=p.get("t1p",0.5),trail=p.get("tr",2.5),buf=p.get("bf",0.2),maxb=maxb)
        trades.append(r); last_d=d; last_e=i+r[2]; i+=r[2]+1
    return trades

def V(r,k,d=None): return float(r.get(k,d if d is not None else 0))

def chk_sq(df,i,th): 
    if i<1: return False
    w=df["bbwp"].values[max(0,i-12):i+1]; v=w[~NP.isnan(w)]
    return len(v)>=1 and NP.sum(v<th)>=1

def sig_bbwp(row,prev,i,df,p):
    bbwp=V(row,"bbwp"); pb=V(prev,"bbwp")
    if PD.isna(bbwp) or bbwp>=p["bt"]: return None
    if p["exp"] and (PD.isna(pb) or bbwp<=pb): return None
    if not chk_sq(df,i,p["bt"]): return None
    if V(row,"adx")<p["adx"]: return None
    vo=V(row,"volume"); vs=V(row,"volume_sma20")
    if vs>0 and vo<vs*p["vm"]: return None
    cl=V(row,"close"); bu=V(row,"bb_upper"); bl=V(row,"bb_lower")
    bw=bu-bl; e50=V(row,"ema50"); e200=V(row,"ema200")
    atr=V(row,"atr"); sk=V(row,"stoch_rsi_k"); bf=p["bf"]
    if cl>bu+bf*bw and sk>=p["skl"] and cl>e50:
        if p["e2"] and e200>0 and cl<=e200: return None
        return (True,cl-p["sl"]*atr,cl+p["tp"]*atr,atr,"long")
    if cl<bl-bf*bw and sk<=p["sks"] and cl<e50:
        if p["e2"] and e200>0 and cl>=e200: return None
        return (False,cl+p["sl"]*atr,cl-p["tp"]*atr,atr,"short")
    return None

def sig_bb(row,prev,i,df,p):
    cl=V(row,"close"); bu=V(row,"bb_upper"); bl=V(row,"bb_lower")
    bw=bu-bl; e50=V(row,"ema50"); e200=V(row,"ema200")
    atr=V(row,"atr"); adx=V(row,"adx"); rsi=V(row,"rsi")
    sk=V(row,"stoch_rsi_k"); vo=V(row,"volume"); vs=V(row,"volume_sma20")
    if vs>0 and vo<vs*p["vm"]: return None
    bf=p["bf"]
    if cl>bu+bf*bw and cl>e50 and adx>=p["adx"] and rsi>=p["rl"] and sk>=p["skl"]:
        if p["e2"] and e200>0 and cl<=e200: return None
        return (True,cl-p["sl"]*atr,cl+p["tp"]*atr,atr,"long")
    if cl<bl-bf*bw and cl<e50 and adx>=p["adx"] and rsi<=p["rs"] and sk<=p["sks"]:
        if p["e2"] and e200>0 and cl>=e200: return None
        return (False,cl+p["sl"]*atr,cl-p["tp"]*atr,atr,"short")
    return None

def sig_hybrid(row,prev,i,df,p):
    cl=V(row,"close"); e50=V(row,"ema50"); e200=V(row,"ema200")
    atr=V(row,"atr"); adx=V(row,"adx"); mh=V(row,"macd_hist")
    rsi=V(row,"rsi"); sk=V(row,"stoch_rsi_k"); bbwp=V(row,"bbwp")
    bu=V(row,"bb_upper"); bl=V(row,"bb_lower"); bw=bu-bl
    vo=V(row,"volume"); vs=V(row,"volume_sma20")
    sq=not(PD.isna(bbwp) or bbwp>=p["bt"])
    ml=mh>0 and rsi>p["rl"] and sk>p["skl"]
    ms=mh<0 and rsi<p["rs"] and sk<p["sks"]
    if vs>0 and vo<vs*p["vm"]: return None
    if adx<p["adx"]: return None
    bf=p["bf"]
    if cl>bu+bf*bw and cl>e50 and (sq or ml):
        if p["e2"] and e200>0 and cl<=e200: return None
        return (True,cl-p["sl"]*atr,cl+p["tp"]*atr,atr,"long")
    if cl<bl-bf*bw and cl<e50 and (sq or ms):
        if p["e2"] and e200>0 and cl>=e200: return None
        return (False,cl+p["sl"]*atr,cl-p["tp"]*atr,atr,"short")
    return None

def sig_macd(row,prev,i,df,p):
    cl=V(row,"close"); e50=V(row,"ema50"); e200=V(row,"ema200")
    atr=V(row,"atr"); adx=V(row,"adx"); rsi=V(row,"rsi")
    sk=V(row,"stoch_rsi_k"); mh=V(row,"macd_hist"); pmh=V(prev,"macd_hist")
    if pmh<=0 and mh>0 and cl>e50 and adx>=p["adx"] and rsi>p["rl"] and sk>p["skl"]:
        if p["e2"] and e200>0 and cl<=e200: return None
        return (True,cl-p["sl"]*atr,cl+p["tp"]*atr,atr,"long")
    if pmh>=0 and mh<0 and cl<e50 and adx>=p["adx"] and rsi<p["rs"] and sk<p["sks"]:
        if p["e2"] and e200>0 and cl>=e200: return None
        return (False,cl+p["sl"]*atr,cl-p["tp"]*atr,atr,"short")
    return None

def sub_eval(df, sf, p):
    R={}; n=len(df)
    for nm, hrs in [("730d",730*24),("365d",365*24),("180d",180*24),("90d",90*24)]:
        s=max(0,n-hrs); sub=df.iloc[s:n].reset_index(drop=True)
        t=run_backtest(sub,sf,p,maxb=p.get("mb",96)); R[nm]=calc_metrics(t)
    return R

if __name__=="__main__":
    t0=time.time()
    df=PD.read_csv("/home/z/my-project/trade-signal/download/btc_1h_cache.csv")
    df.columns=[c.strip().lower() for c in df.columns]
    for c in ["timestamp","date"]:
        if c in df.columns: df=df.drop(columns=[c])
    print(f"Raw: {len(df)} candles")
    df=compute_indicators(df,"1h")
    df=df.dropna(subset=["ema20","ema50","ema200","rsi","atr","bbwp","stoch_rsi_k","adx","volume","volume_sma20"]).copy()
    print(f"Clean: {len(df)} candles | {df.index[0]} to {df.index[-1]}")
    print(f"Indicators ready in {time.time()-t0:.1f}s")

    grids = {}

    # BBWP Squeeze: 648 combos
    bbwp_base = {"bt":15,"exp":True,"adx":16,"vm":0.35,"sl":2.2,"tp":6.0,"tr":2.5,"bf":0.2,"skl":56,"sks":44,"e2":True,"cd_s":2,"cd_o":1,"t1p":0.50,"mb":96}
    bbwp_grid = []
    for bbt in [10,15,20]:
        for slm in [2.0,2.2,2.5]:
            for tpm in [5.0,6.0,8.0]:
                for trm in [2.0,2.5,3.0]:
                    for bfm in [0.1,0.2]:
                        for adxm in [12,16]:
                            for vmm in [0.25,0.35]:
                                bbwp_grid.append({**bbwp_base,"bt":bbt,"sl":slm,"tp":tpm,"tr":trm,"bf":bfm,"adx":adxm,"vm":vmm})
    grids["bbwp"] = (sig_bbwp, bbwp_grid)

    # BB Breakout: 288 combos
    bb_base = {"adx":20,"vm":0.50,"sl":2.0,"tp":5.0,"tr":2.5,"bf":0.2,"rl":45,"rs":55,"skl":50,"sks":50,"e2":True,"cd_s":2,"cd_o":1,"t1p":0.50,"mb":96}
    bb_grid = []
    for bfm in [0.02,0.03,0.05]:
        for adxm in [15,20]:
            for slm in [2.0,2.5]:
                for tpm in [4.0,5.0,6.0,8.0]:
                    for trm in [2.0,2.5,3.0]:
                        for vmm in [0.35,0.50]:
                            bb_grid.append({**bb_base,"bf":bfm,"adx":adxm,"sl":slm,"tp":tpm,"tr":trm,"vm":vmm})
    grids["bb_break"] = (sig_bb, bb_grid)

    # Hybrid: 432 combos
    hy_base = {"bt":25,"bf":0.03,"adx":15,"vm":0.30,"rl":42,"skl":48,"rs":58,"sks":52,"sl":2.0,"tp":5.0,"tr":2.5,"e2":True,"cd_s":2,"cd_o":1,"t1p":0.50,"mb":96}
    hy_grid = []
    for bbt in [15,20,25,35]:
        for adxm in [12,15,20]:
            for slm in [1.8,2.0,2.5]:
                for tpm in [4.0,5.0,6.0,8.0]:
                    for trm in [2.0,2.5,3.0]:
                        hy_grid.append({**hy_base,"bt":bbt,"adx":adxm,"sl":slm,"tp":tpm,"tr":trm})
    grids["hybrid"] = (sig_hybrid, hy_grid)

    # MACD Momentum: 48 combos
    mo_base = {"adx":20,"sl":2.0,"tp":5.0,"tr":2.5,"bf":0.2,"rl":40,"skl":45,"rs":60,"sks":55,"e2":True,"cd_s":2,"cd_o":1,"t1p":0.50,"mb":96}
    mo_grid = []
    for adxm in [15,20]:
        for slm in [2.0,2.5]:
            for tpm in [4.0,5.0,6.0,8.0]:
                for trm in [2.0,2.5,3.0]:
                    mo_grid.append({**mo_base,"adx":adxm,"sl":slm,"tp":tpm,"tr":trm})
    grids["macd"] = (sig_macd, mo_grid)

    results = []
    total = sum(len(v[1]) for v in grids.values())
    done = 0
    best = {"score": -999}

    for sname, (sf, grid) in grids.items():
        print(f"\n=== {sname}: {len(grid)} combos ===")
        for p in grid:
            done += 1
            if done % 100 == 0:
                el = time.time()-t0
                print(f"  [{done}/{total}] {el:.0f}s elapsed, best_ar={best['score']:.1f}%")
            t = run_backtest(df, sf, p, maxb=p.get("mb",96))
            m = calc_metrics(t)
            score = m["ar"] - m["dd"]*0.5
            if m["t"] < 10: continue
            if m["pf"] < 1.0: continue
            if score > best["score"]:
                best = {"score":score, "strat":sname, "params":{k:v for k,v in p.items()}, "metrics":m}
                print(f"  NEW BEST [{sname}] ar={m['ar']:.1f}% wr={m['wr']:.1f}% pf={m['pf']:.2f} dd={m['dd']:.1f}% t={m['t']}")
                print(f"    sl={p['sl']} tp={p['tp']} tr={p['tr']} bf={p['bf']} adx={p['adx']}")
            results.append({"strat":sname,"params":{k:v for k,v in p.items()},"m":m,"score":round(score,1)})

    results.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n{'='*80}")
    print(f"OPTIMIZATION COMPLETE in {time.time()-t0:.1f}s | {done} combos tested")
    print(f"{'='*80}")
    print(f"\nTOP 20 by score (annual_ret - 0.5*drawdown):")
    for r in results[:20]:
        m=r["m"]; p=r["params"]
        print(f"  {r['strat']:10s} ar={m['ar']:6.1f}% wr={m['wr']:5.1f}% pf={m['pf']:5.2f} dd={m['dd']:5.1f}% t={m['t']:3d} aw={m['aw']:.3f} al={m['al']:.3f} | sl={p.get('sl',0)} tp={p.get('tp',0)} tr={p.get('tr',0)} bf={p.get('bf',0)}")

    print(f"\nSUB-PERIOD VALIDATION (top 5):")
    top5 = results[:5]
    validated = []
    for r in top5:
        sname = r["strat"]
        sf = grids[sname][0]
        p = r["params"]
        sub = sub_eval(df, sf, p)
        r["sub"] = sub
        min_ar = min(s.get("ar",0) for s in sub.values())
        avg_ar = NP.mean([s.get("ar",0) for s in sub.values()])
        r["min_ar"] = round(min_ar, 1)
        r["avg_ar"] = round(avg_ar, 1)
        print(f"\n  {sname}: sl={p.get('sl',0)} tp={p.get('tp',0)} tr={p.get('tr',0)} bf={p.get('bf',0)} adx={p.get('adx',0)}")
        for pn, pm in sub.items():
            print(f"    {pn}: ar={pm['ar']:6.1f}% wr={pm['wr']:5.1f}% pf={pm['pf']:5.2f} dd={pm['dd']:5.1f}% t={pm['t']:3d}")
        print(f"    AVG annual: {r['avg_ar']}% | MIN annual: {r['min_ar']}%")
        validated.append(r)

    validated.sort(key=lambda x: x["min_ar"], reverse=True)
    print(f"\n{'='*80}")
    print(f"BEST BY MIN SUB-PERIOD ANNUAL RETURN:")
    for r in validated[:3]:
        m=r["m"]; p=r["params"]
        print(f"  {r['strat']:10s} avg_ar={r['avg_ar']:.1f}% min_ar={r['min_ar']:.1f}% | wr={m['wr']:.1f}% pf={m['pf']:.2f} dd={m['dd']:.1f}%")
        print(f"    sl={p.get('sl',0)} tp={p.get('tp',0)} tr={p.get('tr',0)} bf={p.get('bf',0)} adx={p.get('adx',0)} vm={p.get('vm',0)} bt={p.get('bt',0)}")

    out = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M UTC"),
        "total_combos": done,
        "best_overall": {"strat":best["strat"],"params":best["params"],"metrics":best["metrics"]},
        "best_by_consistency": [{"strat":r["strat"],"params":r["params"],"metrics":r["m"],"sub":r["sub"],"avg_ar":r["avg_ar"],"min_ar":r["min_ar"]} for r in validated[:5]],
        "top20": [{"strat":r["strat"],"params":r["params"],"metrics":r["m"],"score":r["score"]} for r in results[:20]],
    }
    outf = "/home/z/my-project/trade-signal/download/optimize_radical_v1.json"
    with open(outf,"w") as f: json.dump(out, f, indent=2)
    print(f"\nResults saved to {outf}")
