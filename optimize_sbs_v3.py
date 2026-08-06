#!/usr/bin/env python3
"""Otimização SBS v3: dados pré-baixados, grid mínimo e focado."""
import sys, os, time, logging, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indicators import compute_indicators
from backtest import (TradeResult, calculate_metrics, fetch_historical_ohlcv, _apply_costs)
from strategy import SignalType
from strategy_squeeze_breakout import (SBS_PARAMS, evaluate_sbs_row, reset_cooldown, compute_stoch_rsi, compute_bbwp, detect_rsi_divergence, _register_signal)
logging.basicConfig(level=logging.WARNING); logger = logging.getLogger(""); logger.setLevel(logging.INFO)

# Cache data globally
_DATA = {}

def prepare_data(tf, days):
    key = f"{tf}_{days}"
    if key in _DATA: return _DATA[key]
    logger.info(f"Fetching {tf}/{days}d...")
    df = fetch_historical_ohlcv("BTC/USDT", tf, days)
    di = compute_indicators(df, timeframe=tf)
    dc = di.dropna(subset=["ema20","ema50","rsi","atr","atr_percentile","bb_lower","bb_upper","bb_middle","bb_width","volume","volume_sma20"]).copy()
    sk, sd = compute_stoch_rsi(dc["close"], dc["rsi"])
    bbwp = compute_bbwp(dc["bb_width"])
    dc["stoch_rsi_k"] = sk; dc["stoch_rsi_d"] = sd; dc["bbwp"] = bbwp
    _DATA[key] = dc
    return dc

def quick_sim(df_work, params):
    global SBS_PARAMS; orig = dict(SBS_PARAMS); SBS_PARAMS.update(params)
    bbwp = compute_bbwp(df_work["bb_width"], lookback=SBS_PARAMS["bbwp_lookback"])
    df_work["bbwp"] = bbwp
    df_work["was_squeezed"] = bbwp.rolling(window=SBS_PARAMS["bbwp_was_squeezed_bars"]).apply(lambda x: (x < SBS_PARAMS["bbwp_squeeze_threshold"]).any(), raw=False).astype(bool)
    sk, sd = compute_stoch_rsi(df_work["close"], df_work["rsi"], period=SBS_PARAMS["stoch_rsi_period"], k_smooth=SBS_PARAMS["stoch_rsi_k_smooth"], d_smooth=SBS_PARAMS["stoch_rsi_d_smooth"])
    df_work["stoch_rsi_k"] = sk; df_work["stoch_rsi_d"] = sd
    reset_cooldown(); trades = []; i = 0; n = len(df_work); p = SBS_PARAMS
    while i < n:
        row = df_work.iloc[i]
        if i < 2: i += 1; continue
        if any(pd.isna(row.get(c)) for c in ["ema20","ema50","rsi","atr","atr_percentile","bb_lower","bb_upper","bb_middle","bb_width","volume","volume_sma20","close"]) or pd.isna(row.get("stoch_rsi_k",np.nan)) or pd.isna(row.get("bbwp",np.nan)): i += 1; continue
        ap = float(row.get("atr_percentile",0.5))
        if ap < p["atr_pct_min"] or ap > p["atr_pct_max"]: i += 1; continue
        prev = df_work.iloc[i-1]; sk_v = float(row["stoch_rsi_k"]); sd_v = float(row["stoch_rsi_d"])
        bwp = float(row["bbwp"]); vr = float(row["volume"])/float(row["volume_sma20"]) if row["volume_sma20"]>0 else 0
        ws = bool(row.get("was_squeezed",False)); div = detect_rsi_divergence(df_work["close"],df_work["rsi"],i,lookback=p.get("div_lookback",30))
        result = evaluate_sbs_row(row,prev,i,sk_v,sd_v,bwp,vr,ws,div)
        if result is None: i += 1; continue
        sig,mode,conv,tm,sm = result; ep=sig.entry_price; sl=sig.stop_loss; tp=sig.take_profit; atr=sig.atr; il=sig.type==SignalType.LONG
        exp=None; er=None; bars=0; csl=sl; be=False; trail=False; hf=ep; td=atr*tm; bd=atr*p["be_trigger_atr"]
        for j in range(i+1,min(i+p["max_bars"],n)):
            f=df_work.iloc[j]; fc=float(f["close"]); fl=float(f["low"]); fh=float(f["high"]); bars=j-i
            if il: hf=max(hf,fh)
            else: hf=min(hf,fl)
            if not trail:
                tp_h=(il and fh>=tp) or (not il and fl<=tp)
                if tp_h:
                    if p["trailing_enabled"] and p["partial_tp_pct"]>0: trail=True; be=True; csl=ep; continue
                    else: exp=tp; er="tp"; break
            sh=(il and fl<=csl) or (not il and fh>=csl)
            if not be and p["trailing_enabled"] and abs(hf-ep)>=bd: be=True; trail=True; csl=ep
            if trail and p["trailing_enabled"]:
                if il:
                    nt=hf-td
                    if nt>csl: csl=nt
                else:
                    nt=hf+td
                    if nt<csl: csl=nt
            if sh: exp=csl; er="trailing_sl" if trail else "sl"; break
        if exp is None: lj=min(i+p["max_bars"],n)-1; exp=float(df_work.iloc[lj]["close"]); er="timeout"; bars=lj-i
        _,ae,_ = _apply_costs(ep,exp,il,0.016,2.0,2.0)
        pnl=((ae-ep)/ep*100) if il else ((ep-ae)/ep*100)
        trades.append(TradeResult(entry_ts=row.name,exit_ts=df_work.iloc[min(i+bars,n-1)].name,type=sig.type.value,entry_price=ep,exit_price=exp,stop_loss=sl,take_profit=tp,atr=atr,rsi=sig.rsi,pnl_pct=round(pnl,4),pnl_abs=round(exp-ep,2),bars_held=bars,exit_reason=er,atr_percentile=ap,be_triggered=be,trailing_activated=trail))
        _register_signal(i+bars); i += bars+1
    SBS_PARAMS.update(orig); return trades

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
    dc=prepare_data(tf,days)
    logger.info(f"Ready: {len(dc)} candles")
    # Phase 1: Coarse search — 1 param at a time
    base={"bbwp_squeeze_threshold":20,"bbwp_was_squeezed_bars":20,"bbwp_expansion_min":40,"vol_ratio_min":0.7,"sl_atr_mult":1.5,"tp_atr_mult":2.5,"tp_atr_mult_trending":4.0,"trail_atr_mult":1.2,"be_trigger_atr":1.0,"max_bars":48,"cooldown":8}
    best_overall_s=-9999; best_overall_p=None; best_overall_m=None
    for phase,param_name,values in [
        ("sq","bbwp_squeeze_threshold",[5,10,15,20,25,30,35,40,50]),
        ("sb","bbwp_was_squeezed_bars",[6,8,10,12,16,20,24,30]),
        ("sl","sl_atr_mult",[0.8,1.0,1.2,1.5,1.8,2.0,2.5,3.0]),
        ("tp","tp_atr_mult",[1.5,2.0,2.5,3.0,3.5,4.0,5.0,6.0]),
        ("tr","trail_atr_mult",[0.6,0.8,1.0,1.2,1.5,2.0,2.5]),
        ("mb","max_bars",[12,18,24,36,48,72,96]),
        ("cd","cooldown",[2,4,6,8,12,16]),
        ("vr","vol_ratio_min",[0.3,0.5,0.7,1.0,1.3]),
        ("em","bbwp_expansion_min",[20,30,40,50,60]),
    ]:
        best_s=-9999; best_v=None; best_m=None
        for v in values:
            p=dict(base); p[param_name]=v
            try:
                trades=quick_sim(dc.copy(),p)
                if not trades: continue
                m=calculate_metrics(trades,dc,0); s=score(m)
                if s>best_s: best_s=s; best_v=v; best_m=m
            except: pass
        if best_v is not None:
            base[param_name]=best_v
            m=best_m
            logger.info(f"Phase {phase}: {param_name}={best_v} => WR={m.win_rate:.1f}% PnL={m.total_pnl_pct:.1f}% vsBH={m.total_pnl_pct-m.buy_hold_pct:+.1f}pp PF={m.profit_factor:.2f} N={m.total_trades}")
            if score(m)>best_overall_s: best_overall_s=score(m); best_overall_p=dict(base); best_overall_m=m

    # Phase 2: Fine-tune around best
    logger.info(f"\nBest after phase 1: {json.dumps(best_overall_p)}")
    logger.info(f"  WR={best_overall_m.win_rate:.1f}% PnL={best_overall_m.total_pnl_pct:.2f}% vsBH={best_overall_m.total_pnl_pct-best_overall_m.buy_hold_pct:+.2f}pp")

    # Fine-tune: small variations around best
    for param_name, delta_values in [
        ("sl_atr_mult",[-0.3,-0.2,-0.1,0,0.1,0.2,0.3]),
        ("tp_atr_mult",[-0.5,-0.3,-0.1,0,0.1,0.3,0.5]),
        ("trail_atr_mult",[-0.2,-0.1,0,0.1,0.2,0.3]),
    ]:
        best_s=-9999; best_v=None; best_m=None
        base_val=best_overall_p[param_name]
        for dv in delta_values:
            v=round(base_val+dv,2)
            if v<=0: continue
            p=dict(best_overall_p); p[param_name]=v
            try:
                trades=quick_sim(dc.copy(),p)
                if not trades: continue
                m=calculate_metrics(trades,dc,0); s=score(m)
                if s>best_s: best_s=s; best_v=v; best_m=m
            except: pass
        if best_v is not None and best_v!=base_val:
            best_overall_p[param_name]=best_v
            logger.info(f"Fine: {param_name}={best_v} => WR={best_m.win_rate:.1f}% PnL={best_m.total_pnl_pct:.1f}% vsBH={best_m.total_pnl_pct-best_m.buy_hold_pct:+.1f}pp")
            if score(best_m)>best_overall_s: best_overall_s=score(best_m); best_overall_m=best_m

    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL BEST ({tf}/{days}d):")
    logger.info(f"  Params: {json.dumps(best_overall_p)}")
    m=best_overall_m
    logger.info(f"  Trades={m.total_trades} WR={m.win_rate:.1f}% PnL={m.total_pnl_pct:.2f}%")
    logger.info(f"  B&H={m.buy_hold_pct:.2f}% vsBH={m.total_pnl_pct-m.buy_hold_pct:+.2f}pp")
    logger.info(f"  PF={m.profit_factor:.2f} DD={m.max_drawdown_pct:.2f}% Sharpe={m.sharpe_ratio:.2f} R:R={m.avg_r_r:.2f}")

if __name__=="__main__": main()
