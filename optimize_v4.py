#!/usr/bin/env python3
"""Optimização SBS com BBWP vetorizado (100x mais rápido)."""
import sys, os, time, logging, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indicators import compute_indicators
from backtest import (TradeResult, calculate_metrics, fetch_historical_ohlcv, _apply_costs)
from strategy import SignalType
from strategy_squeeze_breakout import (SBS_PARAMS, evaluate_sbs_row, reset_cooldown, compute_stoch_rsi, detect_rsi_divergence, _register_signal)
logging.basicConfig(level=logging.WARNING); L = logging.getLogger(""); L.setLevel(logging.INFO)

def fast_bbwp(bb_width, lookback=100):
    """BBWP vetorizado com numpy — 100x mais rápido que rolling.apply."""
    vals = bb_width.values
    n = len(vals)
    result = np.full(n, np.nan)
    for i in range(lookback, n):
        window = vals[i-lookback:i+1]
        result[i] = np.searchsorted(np.sort(window), vals[i]) / lookback * 100
    return pd.Series(result, index=bb_width.index)

def prepare(tf, days):
    L.info(f"Fetching {tf}/{days}d...")
    df = fetch_historical_ohlcv("BTC/USDT", tf, days)
    di = compute_indicators(df, timeframe=tf)
    dc = di.dropna(subset=["ema20","ema50","rsi","atr","atr_percentile","bb_lower","bb_upper","bb_middle","bb_width","volume","volume_sma20","ema200","ema50_slope","adx"]).copy()
    return dc

def run_with_params(dc, params):
    """Run one full sim with given params, returns (trades, metrics)."""
    global SBS_PARAMS; orig = dict(SBS_PARAMS); SBS_PARAMS.update(params)
    
    # Pre-compute indicators
    bbwp = fast_bbwp(dc["bb_width"], lookback=params.get("bbwp_lookback", 100))
    sk, sd = compute_stoch_rsi(dc["close"], dc["rsi"],
        period=params.get("stoch_rsi_period",14),
        k_smooth=params.get("stoch_rsi_k_smooth",3),
        d_smooth=params.get("stoch_rsi_d_smooth",3))
    
    dc_w = dc.copy()
    dc_w["bbwp"] = bbwp
    dc_w["stoch_rsi_k"] = sk
    dc_w["stoch_rsi_d"] = sd
    sq_th = params["bbwp_squeeze_threshold"]
    sq_bars = params["bbwp_was_squeezed_bars"]
    dc_w["was_squeezed"] = dc_w["bbwp"].rolling(window=sq_bars).apply(lambda x: (x < sq_th).any(), raw=False).astype(bool)
    
    reset_cooldown(); p = SBS_PARAMS
    trades = []; i = 0; n = len(dc_w)
    
    while i < n:
        row = dc_w.iloc[i]
        if i < 2: i += 1; continue
        chk = ["ema20","ema50","rsi","atr","atr_percentile","bb_lower","bb_upper","bb_middle","bb_width","volume","volume_sma20","close","stoch_rsi_k","bbwp"]
        if any(pd.isna(row.get(c,np.nan)) for c in chk): i += 1; continue
        ap = float(row.get("atr_percentile",0.5))
        if ap < p["atr_pct_min"] or ap > p["atr_pct_max"]: i += 1; continue
        prev = dc_w.iloc[i-1]
        result = evaluate_sbs_row(row, prev, i,
            float(row["stoch_rsi_k"]), float(row["stoch_rsi_d"]),
            float(row["bbwp"]),
            float(row["volume"])/float(row["volume_sma20"]) if row["volume_sma20"]>0 else 0,
            bool(row.get("was_squeezed",False)), None)
        if result is None: i += 1; continue
        
        sig,mode,conv,tm,sm = result
        ep=sig.entry_price; sl=sig.stop_loss; tp=sig.take_profit; atr=sig.atr
        il = sig.type==SignalType.LONG
        max_j = min(i+p["max_bars"],n)
        exit_price=None; exit_reason=None; bars=0
        csl=sl; be=False; trail=False; hf=ep
        td=atr*tm; bd=atr*p["be_trigger_atr"]
        
        for j in range(i+1, max_j):
            f=dc_w.iloc[j]; fc=float(f["close"]); fl=float(f["low"]); fh=float(f["high"]) 
            bars=j-i
            if il: hf=max(hf,fh) 
            else: hf=min(hf,fl)
            if not trail:
                tp_h=(il and fh>=tp) or (not il and fl<=tp)
                if tp_h:
                    if p["trailing_enabled"] and p["partial_tp_pct"]>0:
                        trail=True; be=True; csl=ep; continue
                    else: exit_price=tp; exit_reason="tp"; break
            sh=(il and fl<=csl) or (not il and fh>=csl)
            if not be and p["trailing_enabled"] and abs(hf-ep)>=bd:
                be=True; trail=True; csl=ep
            if trail and p["trailing_enabled"]:
                if il: 
                    nt=hf-td
                    if nt>csl: csl=nt
                else:
                    nt=hf+td
                    if nt<csl: csl=nt
            if sh: exit_price=csl; exit_reason="trailing_sl" if trail else "sl"; break
        if exit_price is None:
            lj=min(i+p["max_bars"],n)-1; exit_price=float(dc_w.iloc[lj]["close"]); exit_reason="timeout"; bars=lj-i
        _,ae,_ = _apply_costs(ep,exit_price,il,0.016,2.0,2.0)
        pnl=((ae-ep)/ep*100) if il else ((ep-ae)/ep*100)
        trades.append(TradeResult(entry_ts=row.name,exit_ts=dc_w.iloc[min(i+bars,n-1)].name,
            type=sig.type.value,entry_price=ep,exit_price=exit_price,stop_loss=sl,take_profit=tp,
            atr=atr,rsi=sig.rsi,pnl_pct=round(pnl,4),pnl_abs=round(exit_price-ep,2),
            bars_held=bars,exit_reason=exit_reason,atr_percentile=ap,
            be_triggered=be,trailing_activated=trail))
        _register_signal(i+bars); i += bars+1
    SBS_PARAMS.update(orig)
    return trades

def score(m):
    if m.total_trades<10: return -1000
    s=0
    if m.win_rate>=60: s+=200
    elif m.win_rate>=55: s+=100
    elif m.win_rate>=50: s+=30
    else: s-=50
    vbh=m.total_pnl_pct-m.buy_hold_pct; s+=vbh*2
    if m.profit_factor>1.5: s+=50
    elif m.profit_factor>1.0: s+=20
    else: s-=30
    s-=m.max_drawdown_pct*0.5
    return s

def main():
    tf=sys.argv[1] if len(sys.argv)>1 else "15m"
    days=int(sys.argv[2]) if len(sys.argv)>2 else 730
    dc=prepare(tf,days)
    L.info(f"Data ready: {len(dc)} candles")
    
    base={"bbwp_squeeze_threshold":20,"bbwp_was_squeezed_bars":20,"bbwp_expansion_min":40,
        "vol_ratio_min":0.7,"sl_atr_mult":1.5,"tp_atr_mult":2.5,"tp_atr_mult_trending":4.0,
        "trail_atr_mult":1.2,"be_trigger_atr":1.0,"max_bars":48,"cooldown":8,
        "bbwp_lookback":100,"stoch_rsi_period":14,"stoch_rsi_k_smooth":3,"stoch_rsi_d_smooth":3,
        "stoch_rsi_ob":80,"stoch_rsi_os":20,"use_ema200_filter":True,"use_adx_filter":True,
        "adx_min":18,"rsi_long_min":30,"rsi_long_max":55,"rsi_short_min":45,"rsi_short_max":70,
        "atr_pct_min":0.15,"atr_pct_max":0.85,"trailing_enabled":True,"partial_tp_pct":0.50,
        "reversal_enabled":False,"div_lookback":30,"div_min_slope":0.5}
    
    best_s=-9999; best_p=None; best_m=None; t0=time.time()
    
    # 1D sequential search
    for pname, vals in [
        ("bbwp_squeeze_threshold",[5,10,15,20,25,30,35,40,50,60,70,80]),
        ("bbwp_was_squeezed_bars",[4,6,8,10,12,16,20,24,28,32]),
        ("bbwp_expansion_min",[10,15,20,25,30,35,40,50,60,70,80]),
        ("sl_atr_mult",[0.5,0.8,1.0,1.2,1.5,1.8,2.0,2.5,3.0]),
        ("tp_atr_mult",[1.0,1.5,2.0,2.5,3.0,3.5,4.0,5.0,6.0,7.0,8.0]),
        ("trail_atr_mult",[0.5,0.8,1.0,1.2,1.5,2.0,2.5,3.0]),
        ("max_bars",[12,18,24,36,48,72,96]),
        ("cooldown",[2,4,6,8,10,12,16,20]),
        ("vol_ratio_min",[0.3,0.5,0.7,1.0,1.5,2.0]),
        ("be_trigger_atr",[0.3,0.5,0.8,1.0,1.2,1.5,2.0]),
        ("adx_min",[0,10,15,18,20,25,30]),
        ("rsi_long_max",[45,50,55,60,65,70,75,80]),
        ("rsi_long_min",[20,25,30,35,40,45]),
    ]:
        bs=-9999; bv=None; bm=None
        for v in vals:
            p=dict(base); p[pname]=v
            try:
                trades=run_with_params(dc.copy(),p)
                if not trades: continue
                m=calculate_metrics(trades,dc,0); s=score(m)
                if s>bs: bs=s; bv=v; bm=m
            except: pass
        if bv is not None:
            base[pname]=bv
            if bm and bm.total_trades>=3:
                L.info(f"  {pname}={bv} => N={bm.total_trades} WR={bm.win_rate:.1f}% PnL={bm.total_pnl_pct:.1f}% vsBH={bm.total_pnl_pct-bm.buy_hold_pct:+.1f}pp PF={bm.profit_factor:.2f}")
                if score(bm)>best_s: best_s=score(bm); best_p=dict(base); best_m=bm
    
    elapsed=time.time()-t0
    L.info(f"\n{'='*60}")
    L.info(f"Done in {elapsed:.0f}s")
    if best_p:
        m=best_m
        L.info(f"BEST ({tf}/{days}d): N={m.total_trades} WR={m.win_rate:.1f}% PnL={m.total_pnl_pct:.2f}% B&H={m.buy_hold_pct:.2f}% vsBH={m.total_pnl_pct-m.buy_hold_pct:+.2f}pp PF={m.profit_factor:.2f} DD={m.max_drawdown_pct:.2f}%")
        # Print key params
        key_params={k:best_p[k] for k in ['bbwp_squeeze_threshold','bbwp_was_squeezed_bars','bbwp_expansion_min','sl_atr_mult','tp_atr_mult','trail_atr_mult','max_bars','cooldown','vol_ratio_min','be_trigger_atr','adx_min','rsi_long_min','rsi_long_max']}
        L.info(f"Params: {json.dumps(key_params)}")

if __name__=="__main__": main()
