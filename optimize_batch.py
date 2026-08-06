#!/usr/bin/env python3
"""Batch optimization: pre-compute signals once, simulate exits per combo."""
import sys, os, time, json, logging
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indicators import compute_indicators
from backtest import (TradeResult, calculate_metrics, fetch_historical_ohlcv, _apply_costs)
from strategy import SignalType
from strategy_squeeze_breakout import (SBS_PARAMS, evaluate_sbs_row, reset_cooldown, compute_stoch_rsi, _register_signal)
logging.basicConfig(level=logging.WARNING)
L = logging.getLogger("")
L.setLevel(logging.INFO)

TF = sys.argv[1] if len(sys.argv) > 1 else "15m"
DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 365

# 1. Fetch & prepare data
L.info(f"Fetching {TF}/{DAYS}d...")
t0 = time.time()
df = fetch_historical_ohlcv("BTC/USDT", TF, DAYS)
di = compute_indicators(df, timeframe=TF)
dc = di.dropna(subset=["ema20","ema50","rsi","atr","atr_percentile","bb_lower","bb_upper","bb_middle","bb_width","volume","volume_sma20","ema200","ema50_slope","adx"]).copy()

# 2. Pre-compute BBWP (vectorized)
bbw = dc["bb_width"].values.astype(float)
n = len(bbw)
bbwp_result = np.full(n, 50.0)
lookback = 100
for i in range(lookback - 1, n):
    bbwp_result[i] = np.sum(bbw[i-lookback+1:i+1] <= bbw[i]) / lookback * 100
dc["bbwp"] = bbwp_result

# 3. Pre-compute Stoch RSI
sk, sd = compute_stoch_rsi(dc["close"], dc["rsi"])
dc["stoch_rsi_k"] = sk
dc["stoch_rsi_d"] = sd

# 4. Pre-compute vol_ratio
dc["vol_ratio"] = dc["volume"] / dc["volume_sma20"].replace(0, np.nan)
dc["vol_ratio"] = dc["vol_ratio"].fillna(0)

L.info(f"Data ready: {n} candles ({time.time()-t0:.1f}s)")

# 5. Pre-compute signals for MANY squeeze threshold/was_squeezed combinations
# Store as list of (bar_index, signal_dict, mode, conviction) for each combo

def precompute_signals(sq_th, sq_bars):
    """Pre-compute all entry signals for given squeeze params."""
    global SBS_PARAMS
    orig_params = dict(SBS_PARAMS)
    SBS_PARAMS["bbwp_squeeze_threshold"] = sq_th
    SBS_PARAMS["bbwp_was_squeezed_bars"] = sq_bars
    
    # Compute was_squeezed
    ws = np.zeros(n, dtype=bool)
    for i in range(sq_bars, n):
        ws[i] = np.any(bbwp_result[i-sq_bars:i] < sq_th)
    
    reset_cooldown()
    signals = []
    p = SBS_PARAMS
    
    for i in range(2, n):
        row = dc.iloc[i]
        prev = dc.iloc[i-1]
        
        if any(pd.isna(row.get(c, np.nan)) for c in ["ema20","ema50","rsi","atr","atr_percentile","bb_lower","bb_upper","bb_middle","bb_width","volume","volume_sma20","close","stoch_rsi_k","bbwp"]):
            continue
        ap = float(row.get("atr_percentile", 0.5))
        if ap < p["atr_pct_min"] or ap > p["atr_pct_max"]:
            continue
        
        result = evaluate_sbs_row(
            row, prev, i,
            float(row["stoch_rsi_k"]), float(row["stoch_rsi_d"]),
            float(row["bbwp"]), float(row["vol_ratio"]),
            bool(ws[i]), None
        )
        if result is None:
            continue
        
        sig, mode, conv, tm, sm = result
        signals.append((
            i,
            sig.entry_price, sig.stop_loss, sig.take_profit, sig.atr, sig.rsi,
            sig.type == SignalType.LONG,  # is_long
            ap  # atr_percentile
        ))
        _register_signal(i)
    
    SBS_PARAMS.update(orig_params)
    return signals


def simulate_exits(signals, sl_mult, tp_mult, trail_mult, max_bars, be_trigger, trailing_on=True, partial_pct=0.5):
    """Simulate exits for pre-computed signals."""
    trades = []
    
    for (entry_bar, ep, orig_sl, orig_tp, atr, rsi, is_long, ap) in signals:
        sl = ep - sl_mult * atr if is_long else ep + sl_mult * atr
        tp = ep + tp_mult * atr if is_long else ep - tp_mult * atr
        
        max_j = min(entry_bar + max_bars, n)
        exit_price = None
        exit_reason = None
        bars = 0
        csl = sl
        be = False
        trail = False
        hf = ep
        td = atr * trail_mult
        bd = atr * be_trigger
        
        for j in range(entry_bar + 1, max_j):
            fh = float(dc.iloc[j]["high"])
            fl = float(dc.iloc[j]["low"])
            bars = j - entry_bar
            
            if is_long:
                hf = max(hf, fh)
            else:
                hf = min(hf, fl)
            
            if not trail:
                tp_h = (is_long and fh >= tp) or (not is_long and fl <= tp)
                if tp_h:
                    if trailing_on and partial_pct > 0:
                        trail = True; be = True; csl = ep; continue
                    else:
                        exit_price = tp; exit_reason = "tp"; break
            
            sh = (is_long and fl <= csl) or (not is_long and fh >= csl)
            
            if not be and trailing_on and abs(hf - ep) >= bd:
                be = True; trail = True; csl = ep
            
            if trail and trailing_on:
                if is_long:
                    nt = hf - td
                    if nt > csl: csl = nt
                else:
                    nt = hf + td
                    if nt < csl: csl = nt
            
            if sh:
                exit_price = csl
                exit_reason = "trailing_sl" if trail else "sl"
                break
        
        if exit_price is None:
            lj = min(entry_bar + max_bars, n) - 1
            exit_price = float(dc.iloc[lj]["close"])
            exit_reason = "timeout"
            bars = lj - entry_bar
        
        _, ae, _ = _apply_costs(ep, exit_price, is_long, 0.016, 2.0, 2.0)
        pnl = ((ae - ep) / ep * 100) if is_long else ((ep - ae) / ep * 100)
        
        trades.append(TradeResult(
            entry_ts=dc.index[entry_bar],
            exit_ts=dc.index[min(entry_bar + bars, n - 1)],
            type="LONG" if is_long else "SHORT",
            entry_price=ep, exit_price=exit_price,
            stop_loss=sl, take_profit=tp, atr=atr, rsi=rsi,
            pnl_pct=round(pnl, 4), pnl_abs=round(exit_price - ep, 2),
            bars_held=bars, exit_reason=exit_reason, atr_percentile=ap,
            be_triggered=be, trailing_activated=trail
        ))
    
    return trades


# 6. Pre-compute signals for a few squeeze combos
L.info("Pre-computing signals...")
precomputed = {}
for sq_th in [10, 15, 20, 25, 30, 40, 50]:
    for sq_bars in [8, 12, 16, 20, 24]:
        key = (sq_th, sq_bars)
        t1 = time.time()
        sigs = precompute_signals(sq_th, sq_bars)
        precomputed[key] = sigs
        L.info(f"  sq={sq_th} sb={sq_bars}: {len(sigs)} signals ({time.time()-t1:.1f}s)")

L.info(f"Pre-computation done: {time.time()-t0:.1f}s")

# 7. Now test exit params (fast — no signal evaluation)
best_s = -9999
best_params = None
best_m = None

exit_combos = [
    {"sl": 0.8, "tp": 1.5, "tr": 0.5, "mb": 24, "be": 0.3, "trail": True, "partial": 0.5},
    {"sl": 1.0, "tp": 2.0, "tr": 0.8, "mb": 36, "be": 0.5, "trail": True, "partial": 0.5},
    {"sl": 1.0, "tp": 2.5, "tr": 1.0, "mb": 36, "be": 0.8, "trail": True, "partial": 0.5},
    {"sl": 1.0, "tp": 3.0, "tr": 1.0, "mb": 48, "be": 0.8, "trail": True, "partial": 0.5},
    {"sl": 1.2, "tp": 2.0, "tr": 0.8, "mb": 36, "be": 0.5, "trail": True, "partial": 0.5},
    {"sl": 1.2, "tp": 2.5, "tr": 1.0, "mb": 48, "be": 0.8, "trail": True, "partial": 0.5},
    {"sl": 1.2, "tp": 3.0, "tr": 1.0, "mb": 48, "be": 1.0, "trail": True, "partial": 0.5},
    {"sl": 1.2, "tp": 3.0, "tr": 1.2, "mb": 48, "be": 1.0, "trail": True, "partial": 0.5},
    {"sl": 1.5, "tp": 2.5, "tr": 1.0, "mb": 48, "be": 0.8, "trail": True, "partial": 0.5},
    {"sl": 1.5, "tp": 3.0, "tr": 1.0, "mb": 48, "be": 1.0, "trail": True, "partial": 0.5},
    {"sl": 1.5, "tp": 3.0, "tr": 1.2, "mb": 48, "be": 1.0, "trail": True, "partial": 0.5},
    {"sl": 1.5, "tp": 4.0, "tr": 1.5, "mb": 72, "be": 1.2, "trail": True, "partial": 0.5},
    {"sl": 1.5, "tp": 5.0, "tr": 2.0, "mb": 96, "be": 1.5, "trail": True, "partial": 0.5},
    {"sl": 2.0, "tp": 3.0, "tr": 1.0, "mb": 48, "be": 1.0, "trail": True, "partial": 0.5},
    {"sl": 2.0, "tp": 4.0, "tr": 1.5, "mb": 72, "be": 1.2, "trail": True, "partial": 0.5},
    {"sl": 2.0, "tp": 5.0, "tr": 2.0, "mb": 96, "be": 1.5, "trail": True, "partial": 0.5},
    {"sl": 2.5, "tp": 4.0, "tr": 1.5, "mb": 72, "be": 1.2, "trail": True, "partial": 0.5},
    {"sl": 2.5, "tp": 5.0, "tr": 2.0, "mb": 96, "be": 1.5, "trail": True, "partial": 0.5},
    # No trailing variants
    {"sl": 1.0, "tp": 2.0, "tr": 0.0, "mb": 24, "be": 99, "trail": False, "partial": 0.0},
    {"sl": 1.2, "tp": 2.5, "tr": 0.0, "mb": 24, "be": 99, "trail": False, "partial": 0.0},
    {"sl": 1.5, "tp": 3.0, "tr": 0.0, "mb": 48, "be": 99, "trail": False, "partial": 0.0},
    {"sl": 1.5, "tp": 4.0, "tr": 0.0, "mb": 48, "be": 99, "trail": False, "partial": 0.0},
    # No partial TP
    {"sl": 1.2, "tp": 3.0, "tr": 1.0, "mb": 48, "be": 1.0, "trail": True, "partial": 0.0},
    {"sl": 1.5, "tp": 3.0, "tr": 1.2, "mb": 48, "be": 1.0, "trail": True, "partial": 0.0},
    # Very wide TP
    {"sl": 1.5, "tp": 6.0, "tr": 2.0, "mb": 96, "be": 1.5, "trail": True, "partial": 0.5},
    {"sl": 2.0, "tp": 7.0, "tr": 2.5, "mb": 96, "be": 2.0, "trail": True, "partial": 0.5},
]

tested = 0
for (sq_th, sq_bars), signals in precomputed.items():
    if not signals:
        continue
    for ec in exit_combos:
        tested += 1
        try:
            trades = simulate_exits(signals, ec["sl"], ec["tp"], ec["tr"], ec["mb"], ec["be"], ec["trail"], ec["partial"])
            if not trades:
                continue
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
            
            if s > best_s:
                best_s = s
                best_params = {"sq_th": sq_th, "sq_bars": sq_bars, **ec}
                best_m = m
                L.info(f"  NEW BEST sc={s:.0f}: sq={sq_th}/{sq_bars} sl={ec['sl']} tp={ec['tp']} tr={ec['tr']} mb={ec['mb']} be={ec['be']} trail={ec['trail']} part={ec['partial']}")
                L.info(f"    N={m.total_trades} WR={m.win_rate:.1f}% PnL={m.total_pnl_pct:.1f}% vsBH={vbh:+.1f}pp PF={m.profit_factor:.2f}")
        except:
            pass

L.info(f"\n{'='*60}")
L.info(f"Tested {tested} combos in {time.time()-t0:.0f}s")
if best_params:
    m = best_m
    L.info(f"BEST: N={m.total_trades} WR={m.win_rate:.1f}% PnL={m.total_pnl_pct:.2f}% B&H={m.buy_hold_pct:.2f}% vsBH={m.total_pnl_pct-m.buy_hold_pct:+.2f}pp PF={m.profit_factor:.2f} DD={m.max_drawdown_pct:.2f}%")
    L.info(f"Params: {json.dumps(best_params)}")
