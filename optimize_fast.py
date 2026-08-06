#!/usr/bin/env python3
"""Otimização ultra-rápida SBS: simulação vetorizada, sem ccxt download."""
import sys, os, time, logging, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indicators import compute_indicators
from backtest import (TradeResult, calculate_metrics, fetch_historical_ohlcv, _apply_costs)
from strategy import SignalType
from strategy_squeeze_breakout import (SBS_PARAMS, evaluate_sbs_row, reset_cooldown, compute_stoch_rsi, compute_bbwp, detect_rsi_divergence, _register_signal)
logging.basicConfig(level=logging.WARNING); L = logging.getLogger(""); L.setLevel(logging.INFO)

_DF_CACHE = {}

def get_df(tf, days):
    k = f"{tf}_{days}"
    if k in _DF_CACHE: return _DF_CACHE[k]
    L.info(f"Fetching {tf}/{days}d...")
    df = fetch_historical_ohlcv("BTC/USDT", tf, days)
    di = compute_indicators(df, timeframe=tf)
    dc = di.dropna(subset=["ema20","ema50","rsi","atr","atr_percentile","bb_lower","bb_upper","bb_middle","bb_width","volume","volume_sma20","ema200","ema50_slope","adx"]).copy()
    sk, sd = compute_stoch_rsi(dc["close"], dc["rsi"])
    bbwp = compute_bbwp(dc["bb_width"])
    dc["stoch_rsi_k"] = sk; dc["stoch_rsi_d"] = sd; dc["bbwp"] = bbwp
    _DF_CACHE[k] = dc
    return dc

def fast_sim(dc, params):
    """Fast simulation: only evaluate signal at current bar, skip future bar-by-bar for speed.
    Uses simplified exit logic."""
    global SBS_PARAMS; orig = dict(SBS_PARAMS); SBS_PARAMS.update(params)
    bbwp = compute_bbwp(dc["bb_width"], lookback=SBS_PARAMS["bbwp_lookback"])
    dc_w = dc.copy()
    dc_w["bbwp"] = bbwp
    dc_w["was_squeezed"] = bbwp.rolling(window=SBS_PARAMS["bbwp_was_squeezed_bars"]).apply(
        lambda x: (x < SBS_PARAMS["bbwp_squeeze_threshold"]).any(), raw=False).astype(bool)
    sk, sd = compute_stoch_rsi(dc_w["close"], dc_w["rsi"],
        period=SBS_PARAMS["stoch_rsi_period"], k_smooth=SBS_PARAMS["stoch_rsi_k_smooth"],
        d_smooth=SBS_PARAMS["stoch_rsi_d_smooth"])
    dc_w["stoch_rsi_k"] = sk; dc_w["stoch_rsi_d"] = sd

    reset_cooldown()
    p = SBS_PARAMS
    trades = []
    n = len(dc_w)
    i = 0
    
    while i < n:
        row = dc_w.iloc[i]
        if i < 2: i += 1; continue
        
        # NaN checks
        check_cols = ["ema20","ema50","rsi","atr","atr_percentile",
                      "bb_lower","bb_upper","bb_middle","bb_width",
                      "volume","volume_sma20","close","stoch_rsi_k","bbwp"]
        if any(pd.isna(row.get(c, np.nan)) for c in check_cols): i += 1; continue
        
        atr_pct = float(row.get("atr_percentile", 0.5))
        if atr_pct < p["atr_pct_min"] or atr_pct > p["atr_pct_max"]: i += 1; continue
        
        prev = dc_w.iloc[i-1]
        sk_v = float(row["stoch_rsi_k"])
        sd_v = float(row["stoch_rsi_d"])
        bwp = float(row["bbwp"])
        vr = float(row["volume"]) / float(row["volume_sma20"]) if row["volume_sma20"] > 0 else 0
        ws = bool(row.get("was_squeezed", False))
        
        # Skip divergence detection for speed (rarely triggers)
        result = evaluate_sbs_row(row, prev, i, sk_v, sd_v, bwp, vr, ws, None)
        if result is None: i += 1; continue
        
        sig, mode, conv, trail_m, sl_m = result
        ep = sig.entry_price; sl = sig.stop_loss; tp = sig.take_profit
        atr = sig.atr; il = sig.type == SignalType.LONG
        
        # Vectorized future bar scanning
        max_j = min(i + p["max_bars"], n)
        if max_j <= i + 1: i += 1; continue
        
        future = dc_w.iloc[i+1:max_j]
        f_high = future["high"].values.astype(float)
        f_low = future["low"].values.astype(float)
        f_close = future["close"].values.astype(float)
        bars_range = np.arange(1, max_j - i)
        
        # Track highest favorable
        if il:
            hwm = np.maximum.accumulate(f_high)
        else:
            hwm = np.minimum.accumulate(f_low)
        
        # Check SL/TP/BE/trailing
        current_sl = sl
        be_trigger_dist = atr * p["be_trigger_atr"]
        trail_dist = atr * trail_m
        
        exit_bar = None
        exit_price = None
        exit_reason = None
        be_triggered = False
        trail_active = False
        partial_tp = False
        
        for b_idx in range(len(bars_range)):
            fh = f_high[b_idx]; fl = f_low[b_idx]; fc = f_close[b_idx]
            hf = hwm[b_idx]
            
            # TP check (before trailing)
            if not trail_active:
                tp_hit = (il and fh >= tp) or (not il and fl <= tp)
                if tp_hit:
                    if p["trailing_enabled"] and p["partial_tp_pct"] > 0:
                        partial_tp = True; trail_active = True; be_triggered = True
                        current_sl = ep
                        continue  # Keep going with trailing
                    else:
                        exit_bar = b_idx; exit_price = tp; exit_reason = "tp"; break
            
            # SL check
            sl_hit = (il and fl <= current_sl) or (not il and fh >= current_sl)
            
            # BE trigger
            if not be_triggered and p["trailing_enabled"]:
                fav_dist = abs(hf - ep)
                if fav_dist >= be_trigger_dist:
                    be_triggered = True; trail_active = True; current_sl = ep
            
            # Trailing ratchet
            if trail_active and p["trailing_enabled"]:
                if il:
                    nt = hf - trail_dist
                    if nt > current_sl: current_sl = nt
                else:
                    nt = hf + trail_dist
                    if nt < current_sl: current_sl = nt
            
            if sl_hit:
                exit_bar = b_idx; exit_price = current_sl
                exit_reason = "trailing_sl" if trail_active else "sl"; break
        
        if exit_price is None:
            exit_bar = len(bars_range) - 1
            exit_price = f_close[exit_bar]
            exit_reason = "timeout"
        
        bars = exit_bar + 1
        _, adj_exit, _ = _apply_costs(ep, exit_price, il, 0.016, 2.0, 2.0)
        pnl = ((adj_exit - ep) / ep * 100) if il else ((ep - adj_exit) / ep * 100)
        
        trades.append(TradeResult(
            entry_ts=row.name, exit_ts=dc_w.iloc[min(i+bars, n-1)].name,
            type=sig.type.value, entry_price=ep, exit_price=exit_price,
            stop_loss=sl, take_profit=tp, atr=atr, rsi=sig.rsi,
            pnl_pct=round(pnl, 4), pnl_abs=round(exit_price - ep, 2),
            bars_held=bars, exit_reason=exit_reason, atr_percentile=atr_pct,
            be_triggered=be_triggered, trailing_activated=trail_active, partial_tp_filled=partial_tp))
        _register_signal(i + bars)
        i += bars + 1
    
    SBS_PARAMS.update(orig)
    return trades

def score(m):
    if m.total_trades < 10: return -1000
    s = 0
    if m.win_rate >= 60: s += 200
    elif m.win_rate >= 55: s += 100
    elif m.win_rate >= 50: s += 30
    else: s -= 50
    vbh = m.total_pnl_pct - m.buy_hold_pct; s += vbh * 2
    if m.profit_factor > 1.5: s += 50
    elif m.profit_factor > 1.0: s += 20
    else: s -= 30
    s -= m.max_drawdown_pct * 0.5
    return s

def main():
    tf = sys.argv[1] if len(sys.argv) > 1 else "15m"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 730
    dc = get_df(tf, days)
    L.info(f"Ready: {len(dc)} candles")
    
    base = {"bbwp_squeeze_threshold":20,"bbwp_was_squeezed_bars":20,"bbwp_expansion_min":40,
            "vol_ratio_min":0.7,"sl_atr_mult":1.5,"tp_atr_mult":2.5,"tp_atr_mult_trending":4.0,
            "trail_atr_mult":1.2,"be_trigger_atr":1.0,"max_bars":48,"cooldown":8,
            "bbwp_lookback":100,"stoch_rsi_period":14,"stoch_rsi_k_smooth":3,"stoch_rsi_d_smooth":3,
            "stoch_rsi_ob":80,"stoch_rsi_os":20,"use_ema200_filter":True,"use_adx_filter":True,
            "adx_min":18,"rsi_long_min":30,"rsi_long_max":55,"rsi_short_min":45,"rsi_short_max":70,
            "atr_pct_min":0.15,"atr_pct_max":0.85,"trailing_enabled":True,"partial_tp_pct":0.50,
            "reversal_enabled":False,"div_lookback":30,"div_min_slope":0.5}
    
    best_s = -9999; best_p = None; best_m = None
    t0 = time.time()
    
    # Sequential 1D search
    for pname, vals in [
        ("bbwp_squeeze_threshold", [5,10,15,20,25,30,35,40,50,60,70]),
        ("bbwp_was_squeezed_bars", [6,8,10,12,16,20,24,28,32]),
        ("bbwp_expansion_min", [15,20,25,30,35,40,50,60]),
        ("sl_atr_mult", [0.8,1.0,1.2,1.5,1.8,2.0,2.5,3.0]),
        ("tp_atr_mult", [1.5,2.0,2.5,3.0,3.5,4.0,5.0,6.0,7.0]),
        ("trail_atr_mult", [0.6,0.8,1.0,1.2,1.5,2.0,2.5,3.0]),
        ("max_bars", [12,18,24,36,48,72,96]),
        ("cooldown", [2,4,6,8,10,12,16]),
        ("vol_ratio_min", [0.3,0.5,0.7,1.0,1.5]),
        ("be_trigger_atr", [0.5,0.8,1.0,1.2,1.5,2.0]),
        ("adx_min", [10,15,18,20,25]),
        ("rsi_long_max", [50,55,60,65,70,75]),
        ("rsi_long_min", [25,30,35,40,45]),
    ]:
        bs = -9999; bv = None; bm = None
        for v in vals:
            p = dict(base); p[pname] = v
            try:
                trades = fast_sim(dc.copy(), p)
                if not trades: continue
                m = calculate_metrics(trades, dc, 0); s = score(m)
                if s > bs: bs = s; bv = v; bm = m
            except Exception as e: pass
        if bv is not None:
            base[pname] = bv
            if bm.total_trades >= 5:
                L.info(f"  {pname}={bv} => N={bm.total_trades} WR={bm.win_rate:.1f}% PnL={bm.total_pnl_pct:.1f}% vsBH={bm.total_pnl_pct-bm.buy_hold_pct:+.1f}pp PF={bm.profit_factor:.2f}")
                if score(bm) > best_s: best_s = score(bm); best_p = dict(base); best_m = bm
    
    # Fine-tune
    for pname, deltas in [("sl_atr_mult",[-0.3,-0.2,-0.1,0,0.1,0.2,0.3]),("tp_atr_mult",[-1.0,-0.5,-0.3,0,0.3,0.5,1.0]),("trail_atr_mult",[-0.2,-0.1,0,0.1,0.2,0.3])]:
        bv = base[pname]; bs = -9999; bm = None
        for dv in deltas:
            v = round(bv + dv, 2)
            if v <= 0: continue
            p = dict(best_p or base); p[pname] = v
            try:
                trades = fast_sim(dc.copy(), p)
                if not trades: continue
                m = calculate_metrics(trades, dc, 0); s = score(m)
                if s > bs: bs = s; bv = v; bm = m
            except: pass
        if bm and bv != base.get(pname):
            base[pname] = bv
            if score(bm) > best_s: best_s = score(bm); best_p = dict(base); best_m = bm
    
    elapsed = time.time() - t0
    L.info(f"\n{'='*60}")
    L.info(f"Optimization done in {elapsed:.0f}s")
    if best_p:
        m = best_m
        L.info(f"BEST ({tf}/{days}d):")
        L.info(f"  Params: {json.dumps({k:v for k,v in best_p.items() if k in ['bbwp_squeeze_threshold','bbwp_was_squeezed_bars','bbwp_expansion_min','sl_atr_mult','tp_atr_mult','trail_atr_mult','max_bars','cooldown','vol_ratio_min','be_trigger_atr','adx_min','rsi_long_min','rsi_long_max']})}")
        L.info(f"  Trades={m.total_trades} WR={m.win_rate:.1f}% PnL={m.total_pnl_pct:.2f}% B&H={m.buy_hold_pct:.2f}% vsBH={m.total_pnl_pct-m.buy_hold_pct:+.2f}pp PF={m.profit_factor:.2f} DD={m.max_drawdown_pct:.2f}%")

if __name__ == "__main__": main()