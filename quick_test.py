#!/usr/bin/env python3
"""Teste rápido de 15 combinações de parâmetros SBS."""
import sys, os, time, json, logging
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indicators import compute_indicators
from backtest import (TradeResult, calculate_metrics, fetch_historical_ohlcv, _apply_costs)
from strategy import SignalType
from strategy_squeeze_breakout import (SBS_PARAMS, evaluate_sbs_row, reset_cooldown, compute_stoch_rsi, _register_signal)
logging.basicConfig(level=logging.WARNING); L = logging.getLogger(""); L.setLevel(logging.INFO)

def vec_bbwp(bb_width, lookback=100):
    """BBWP com numpy vectorized — muito mais rápido."""
    vals = bb_width.values.astype(float)
    n = len(vals)
    result = np.full(n, 50.0)  # default 50
    for i in range(lookback - 1, n):
        result[i] = np.sum(vals[i-lookback+1:i+1] <= vals[i]) / lookback * 100
    return pd.Series(result, index=bb_width.index)

def sim(dc, params):
    global SBS_PARAMS; orig = dict(SBS_PARAMS); SBS_PARAMS.update(params)
    dc_w = dc.copy()
    bbwp = vec_bbwp(dc_w["bb_width"], lookback=params.get("bbwp_lookback", 100))
    dc_w["bbwp"] = bbwp
    sq_th = params["bbwp_squeeze_threshold"]
    sq_bars = params["bbwp_was_squeezed_bars"]
    dc_w["was_squeezed"] = np.zeros(len(dc_w), dtype=bool)
    bbwp_arr = bbwp.values
    for i in range(sq_bars, len(dc_w)):
        dc_w.iloc[i, dc_w.columns.get_loc("was_squeezed")] = np.any(bbwp_arr[i-sq_bars:i] < sq_th)
    sk, sd = compute_stoch_rsi(dc_w["close"], dc_w["rsi"])
    dc_w["stoch_rsi_k"] = sk; dc_w["stoch_rsi_d"] = sd
    reset_cooldown(); p = SBS_PARAMS; trades = []; i = 0; n = len(dc_w)
    while i < n:
        row = dc_w.iloc[i]
        if i < 2: i += 1; continue
        if any(pd.isna(row.get(c, np.nan)) for c in ["ema20","ema50","rsi","atr","atr_percentile","bb_lower","bb_upper","bb_middle","bb_width","volume","volume_sma20","close","stoch_rsi_k","bbwp"]): i += 1; continue
        ap = float(row.get("atr_percentile", 0.5))
        if ap < p["atr_pct_min"] or ap > p["atr_pct_max"]: i += 1; continue
        prev = dc_w.iloc[i-1]
        result = evaluate_sbs_row(row, prev, i, float(row["stoch_rsi_k"]), float(row["stoch_rsi_d"]), float(row["bbwp"]),
            float(row["volume"])/float(row["volume_sma20"]) if row["volume_sma20"] > 0 else 0,
            bool(row.get("was_squeezed", False)), None)
        if result is None: i += 1; continue
        sig, mode, conv, tm, sm = result
        ep = sig.entry_price; sl = sig.stop_loss; tp = sig.take_profit; atr = sig.atr
        il = sig.type == SignalType.LONG
        max_j = min(i + p["max_bars"], n); exit_price = None; exit_reason = None; bars = 0
        csl = sl; be = False; trail = False; hf = ep; td = atr * tm; bd = atr * p["be_trigger_atr"]
        for j in range(i + 1, max_j):
            f = dc_w.iloc[j]; fh = float(f["high"]); fl = float(f["low"])
            bars = j - i
            if il: hf = max(hf, fh)
            else: hf = min(hf, fl)
            if not trail:
                tp_h = (il and fh >= tp) or (not il and fl <= tp)
                if tp_h:
                    if p["trailing_enabled"] and p["partial_tp_pct"] > 0:
                        trail = True; be = True; csl = ep; continue
                    else: exit_price = tp; exit_reason = "tp"; break
            sh = (il and fl <= csl) or (not il and fh >= csl)
            if not be and p["trailing_enabled"] and abs(hf - ep) >= bd: be = True; trail = True; csl = ep
            if trail and p["trailing_enabled"]:
                if il:
                    nt = hf - td
                    if nt > csl: csl = nt
                else:
                    nt = hf + td
                    if nt < csl: csl = nt
            if sh: exit_price = csl; exit_reason = "trailing_sl" if trail else "sl"; break
        if exit_price is None:
            lj = min(i + p["max_bars"], n) - 1
            exit_price = float(dc_w.iloc[lj]["close"]); exit_reason = "timeout"; bars = lj - i
        _, ae, _ = _apply_costs(ep, exit_price, il, 0.016, 2.0, 2.0)
        pnl = ((ae - ep) / ep * 100) if il else ((ep - ae) / ep * 100)
        trades.append(TradeResult(entry_ts=row.name, exit_ts=dc_w.iloc[min(i + bars, n - 1)].name,
            type=sig.type.value, entry_price=ep, exit_price=exit_price, stop_loss=sl, take_profit=tp,
            atr=atr, rsi=sig.rsi, pnl_pct=round(pnl, 4), pnl_abs=round(exit_price - ep, 2),
            bars_held=bars, exit_reason=exit_reason, atr_percentile=ap,
            be_triggered=be, trailing_activated=trail))
        _register_signal(i + bars); i += bars + 1
    SBS_PARAMS.update(orig); return trades

L.info("Fetching data...")
df = fetch_historical_ohlcv("BTC/USDT", "15m", 730)
di = compute_indicators(df, timeframe="15m")
dc = di.dropna(subset=["ema20","ema50","rsi","atr","atr_percentile","bb_lower","bb_upper","bb_middle","bb_width","volume","volume_sma20","ema200","ema50_slope","adx"]).copy()
L.info(f"Data: {len(dc)} candles")

t0 = time.time()
combos = [
    {"bbwp_squeeze_threshold":10,"bbwp_was_squeezed_bars":8,"sl_atr_mult":1.0,"tp_atr_mult":2.0,"trail_atr_mult":0.8,"max_bars":36,"cooldown":6,"vol_ratio_min":0.5,"be_trigger_atr":0.5},
    {"bbwp_squeeze_threshold":10,"bbwp_was_squeezed_bars":8,"sl_atr_mult":1.0,"tp_atr_mult":3.0,"trail_atr_mult":1.0,"max_bars":48,"cooldown":8,"vol_ratio_min":0.5,"be_trigger_atr":0.8},
    {"bbwp_squeeze_threshold":15,"bbwp_was_squeezed_bars":12,"sl_atr_mult":1.2,"tp_atr_mult":2.5,"trail_atr_mult":1.0,"max_bars":36,"cooldown":6,"vol_ratio_min":0.5,"be_trigger_atr":0.8},
    {"bbwp_squeeze_threshold":20,"bbwp_was_squeezed_bars":16,"sl_atr_mult":1.5,"tp_atr_mult":3.0,"trail_atr_mult":1.2,"max_bars":48,"cooldown":8,"vol_ratio_min":0.7,"be_trigger_atr":1.0},
    {"bbwp_squeeze_threshold":30,"bbwp_was_squeezed_bars":20,"sl_atr_mult":2.0,"tp_atr_mult":4.0,"trail_atr_mult":1.5,"max_bars":72,"cooldown":12,"vol_ratio_min":0.7,"be_trigger_atr":1.2},
    {"bbwp_squeeze_threshold":40,"bbwp_was_squeezed_bars":24,"sl_atr_mult":2.0,"tp_atr_mult":5.0,"trail_atr_mult":2.0,"max_bars":96,"cooldown":16,"vol_ratio_min":0.5,"be_trigger_atr":1.5},
    # Wider TP, tighter SL
    {"bbwp_squeeze_threshold":15,"bbwp_was_squeezed_bars":12,"sl_atr_mult":1.0,"tp_atr_mult":5.0,"trail_atr_mult":2.0,"max_bars":96,"cooldown":8,"vol_ratio_min":0.5,"be_trigger_atr":0.5},
    # Very tight
    {"bbwp_squeeze_threshold":5,"bbwp_was_squeezed_bars":8,"sl_atr_mult":0.8,"tp_atr_mult":1.5,"trail_atr_mult":0.5,"max_bars":24,"cooldown":4,"vol_ratio_min":0.3,"be_trigger_atr":0.3},
    # No trailing
    {"bbwp_squeeze_threshold":20,"bbwp_was_squeezed_bars":16,"sl_atr_mult":1.5,"tp_atr_mult":3.0,"trail_atr_mult":0.0,"max_bars":24,"cooldown":8,"vol_ratio_min":0.7,"be_trigger_atr":1.0,"trailing_enabled":False},
    # No partial TP
    {"bbwp_squeeze_threshold":20,"bbwp_was_squeezed_bars":16,"sl_atr_mult":1.5,"tp_atr_mult":3.0,"trail_atr_mult":1.2,"max_bars":48,"cooldown":8,"vol_ratio_min":0.7,"be_trigger_atr":1.0,"partial_tp_pct":0.0},
    # Disabled EMA200
    {"bbwp_squeeze_threshold":20,"bbwp_was_squeezed_bars":16,"sl_atr_mult":1.5,"tp_atr_mult":3.0,"trail_atr_mult":1.2,"max_bars":48,"cooldown":8,"vol_ratio_min":0.7,"be_trigger_atr":1.0,"use_ema200_filter":False},
    # Disabled ADX
    {"bbwp_squeeze_threshold":20,"bbwp_was_squeezed_bars":16,"sl_atr_mult":1.5,"tp_atr_mult":3.0,"trail_atr_mult":1.2,"max_bars":48,"cooldown":8,"vol_ratio_min":0.7,"be_trigger_atr":1.0,"use_adx_filter":False},
    # Wide RSI
    {"bbwp_squeeze_threshold":20,"bbwp_was_squeezed_bars":16,"sl_atr_mult":1.5,"tp_atr_mult":3.0,"trail_atr_mult":1.2,"max_bars":48,"cooldown":8,"vol_ratio_min":0.7,"be_trigger_atr":1.0,"rsi_long_min":20,"rsi_long_max":70},
    # Very wide expansion
    {"bbwp_squeeze_threshold":20,"bbwp_was_squeezed_bars":16,"sl_atr_mult":1.5,"tp_atr_mult":3.0,"trail_atr_mult":1.2,"max_bars":48,"cooldown":8,"vol_ratio_min":0.7,"be_trigger_atr":1.0,"bbwp_expansion_min":60},
    # Lower expansion threshold
    {"bbwp_squeeze_threshold":20,"bbwp_was_squeezed_bars":16,"sl_atr_mult":1.5,"tp_atr_mult":3.0,"trail_atr_mult":1.2,"max_bars":48,"cooldown":8,"vol_ratio_min":0.7,"be_trigger_atr":1.0,"bbwp_expansion_min":20},
]

best_s = -9999; best_i = -1
for ci, combo in enumerate(combos):
    try:
        trades = sim(dc.copy(), combo)
        if not trades: continue
        m = calculate_metrics(trades, dc, 0)
        vbh = m.total_pnl_pct - m.buy_hold_pct
        s = 0
        if m.win_rate >= 60: s += 200
        elif m.win_rate >= 55: s += 100
        elif m.win_rate >= 50: s += 30
        else: s -= 50
        s += vbh * 2
        if m.profit_factor > 1.5: s += 50
        elif m.profit_factor > 1.0: s += 20
        else: s -= 30
        s -= m.max_drawdown_pct * 0.5
        if s > best_s: best_s = s; best_i = ci
        L.info(f"#{ci} N={m.total_trades} WR={m.win_rate:.1f}% PnL={m.total_pnl_pct:.1f}% B&H={m.buy_hold_pct:.1f}% vsBH={vbh:+.1f}pp PF={m.profit_factor:.2f} DD={m.max_drawdown_pct:.2f}% sc={s:.0f}")
    except Exception as e:
        L.info(f"#{ci} ERROR: {e}")

elapsed = time.time() - t0
L.info(f"\nDone in {elapsed:.0f}s. Best combo: #{best_i}")
if best_i >= 0:
    L.info(f"Best params: {json.dumps(combos[best_i])}")
