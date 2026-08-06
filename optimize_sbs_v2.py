#!/usr/bin/env python3
"""Otimizacao focada SBS v3 — grid search reduzido."""
import sys, os, time, logging, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indicators import compute_indicators
from backtest import (
    TradeResult, BacktestMetrics, calculate_metrics,
    fetch_historical_ohlcv, DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS,
    DEFAULT_SLIPPAGE_BPS, _apply_costs,
)
from strategy import SignalType
from strategy_squeeze_breakout import (
    SBS_PARAMS, evaluate_sbs_row, reset_cooldown,
    compute_stoch_rsi, compute_bbwp, detect_rsi_divergence,
    _register_signal,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("sbs_opt")
logger.setLevel(logging.INFO)


def run_sim(df_work, params, fee=0.016, spread=2.0, slip=2.0):
    global SBS_PARAMS
    orig = dict(SBS_PARAMS)
    SBS_PARAMS.update(params)

    bbwp = compute_bbwp(df_work["bb_width"], lookback=SBS_PARAMS["bbwp_lookback"])
    df_work["bbwp"] = bbwp
    df_work["was_squeezed"] = bbwp.rolling(
        window=SBS_PARAMS["bbwp_was_squeezed_bars"]
    ).apply(lambda x: (x < SBS_PARAMS["bbwp_squeeze_threshold"]).any(), raw=False).astype(bool)
    stoch_k, stoch_d = compute_stoch_rsi(df_work["close"], df_work["rsi"],
        period=SBS_PARAMS["stoch_rsi_period"],
        k_smooth=SBS_PARAMS["stoch_rsi_k_smooth"],
        d_smooth=SBS_PARAMS["stoch_rsi_d_smooth"])
    df_work["stoch_rsi_k"] = stoch_k
    df_work["stoch_rsi_d"] = stoch_d

    reset_cooldown()
    trades = []
    i = 0; n = len(df_work); p = SBS_PARAMS

    while i < n:
        row = df_work.iloc[i]
        if i < 2: i += 1; continue
        critical = ["ema20","ema50","rsi","atr","atr_percentile",
                     "bb_lower","bb_upper","bb_middle","bb_width",
                     "volume","volume_sma20","close"]
        if any(pd.isna(row.get(c)) for c in critical) or pd.isna(row.get("stoch_rsi_k",np.nan)) or pd.isna(row.get("bbwp",np.nan)):
            i += 1; continue
        atr_pct = float(row.get("atr_percentile",0.5))
        if atr_pct < p["atr_pct_min"] or atr_pct > p["atr_pct_max"]: i += 1; continue
        prev = df_work.iloc[i-1]
        sk = float(row["stoch_rsi_k"]); sd = float(row["stoch_rsi_d"])
        bwp = float(row["bbwp"])
        vr = float(row["volume"]) / float(row["volume_sma20"]) if row["volume_sma20"] > 0 else 0
        ws = bool(row.get("was_squeezed",False))
        div = detect_rsi_divergence(df_work["close"],df_work["rsi"],i,lookback=p.get("div_lookback",30))
        result = evaluate_sbs_row(row,prev,i,sk,sd,bwp,vr,ws,div)
        if result is None: i += 1; continue
        signal,mode,conviction,trail_mult,sl_mult = result
        ep = signal.entry_price; sl = signal.stop_loss; tp = signal.take_profit
        atr = signal.atr; is_long = signal.type == SignalType.LONG
        exit_price = None; exit_reason = None; bars = 0
        current_sl = sl; be_triggered = False; trailing_activated = False
        highest_fav = ep; trail_dist = atr * trail_mult; be_dist = atr * p["be_trigger_atr"]
        partial = False; was_trail = False
        for j in range(i+1, min(i+p["max_bars"],n)):
            f = df_work.iloc[j]; fc = float(f["close"]); fl = float(f["low"]); fh = float(f["high"])
            bars = j - i
            if is_long: highest_fav = max(highest_fav,fh)
            else: highest_fav = min(highest_fav,fl)
            if not trailing_activated:
                tp_hit = (is_long and fh >= tp) or (not is_long and fl <= tp)
                if tp_hit:
                    if p["trailing_enabled"] and p["partial_tp_pct"] > 0:
                        partial = True; trailing_activated = True; be_triggered = True; current_sl = ep; continue
                    else: exit_price = tp; exit_reason = "tp"; break
            sl_hit = (is_long and fl <= current_sl) or (not is_long and fh >= current_sl)
            if not be_triggered and p["trailing_enabled"]:
                if abs(highest_fav - ep) >= be_dist:
                    be_triggered = True; trailing_activated = True; current_sl = ep
            if trailing_activated and p["trailing_enabled"]:
                if is_long:
                    nt = highest_fav - trail_dist
                    if nt > current_sl: current_sl = nt
                else:
                    nt = highest_fav + trail_dist
                    if nt < current_sl: current_sl = nt
            if sl_hit:
                exit_price = current_sl
                exit_reason = "trailing_sl" if trailing_activated else "sl"
                was_trail = trailing_activated; break
        if exit_price is None:
            lj = min(i+p["max_bars"],n)-1
            exit_price = float(df_work.iloc[lj]["close"]); exit_reason = "timeout"; bars = lj-i
        _,ae,_ = _apply_costs(ep,exit_price,is_long,fee,spread,slip)
        pnl = ((ae-ep)/ep*100) if is_long else ((ep-ae)/ep*100)
        trades.append(TradeResult(
            entry_ts=row.name, exit_ts=df_work.iloc[min(i+bars,n-1)].name,
            type=signal.type.value, entry_price=ep, exit_price=exit_price,
            stop_loss=sl, take_profit=tp, atr=atr, rsi=signal.rsi,
            pnl_pct=round(pnl,4), pnl_abs=round(exit_price-ep,2),
            bars_held=bars, exit_reason=exit_reason, atr_percentile=atr_pct,
            be_triggered=be_triggered, trailing_activated=trailing_activated, partial_tp_filled=partial))
        _register_signal(i+bars,was_trailing=was_trail)
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
    vbh = m.total_pnl_pct - m.buy_hold_pct
    s += vbh * 2
    if m.profit_factor > 1.5: s += 50
    elif m.profit_factor > 1.0: s += 20
    else: s -= 30
    s -= m.max_drawdown_pct * 0.5
    return s


def main():
    tf = sys.argv[1] if len(sys.argv) > 1 else "15m"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 730

    logger.info(f"Data: {tf}/{days}d...")
    df = fetch_historical_ohlcv("BTC/USDT", tf, days)
    df_ind = compute_indicators(df, timeframe=tf)
    df_clean = df_ind.dropna(subset=["ema20","ema50","rsi","atr","atr_percentile","bb_lower","bb_upper","bb_middle","bb_width","volume","volume_sma20"]).copy()
    stoch_k, stoch_d = compute_stoch_rsi(df_clean["close"], df_clean["rsi"])
    bbwp = compute_bbwp(df_clean["bb_width"])
    df_clean["stoch_rsi_k"] = stoch_k; df_clean["stoch_rsi_d"] = stoch_d; df_clean["bbwp"] = bbwp
    logger.info(f"Ready: {len(df_clean)} candles, B&H={((df_clean.iloc[-1]['close']-df_clean.iloc[0]['close'])/df_clean.iloc[0]['close']*100):.1f}%")

    # Focused grid: vary 5 key params
    best_score = -9999; best_p = None; best_m = None; tested = 0; t0 = time.time()

    for sq_th in [10,15,20,25,30,35,40]:
        for sq_bars in [8,12,16,20,24]:
            for sl_m in [1.0,1.2,1.5,1.8,2.0,2.5,3.0]:
                for tp_m in [1.5,2.0,2.5,3.0,4.0,5.0,6.0]:
                    for trail_m in [0.8,1.0,1.2,1.5,2.0]:
                        params = {
                            "bbwp_squeeze_threshold": sq_th,
                            "bbwp_was_squeezed_bars": sq_bars,
                            "sl_atr_mult": sl_m,
                            "tp_atr_mult": tp_m,
                            "trail_atr_mult": trail_m,
                            "max_bars": 48,
                            "cooldown": 8,
                            "bbwp_expansion_min": 40,
                            "vol_ratio_min": 0.7,
                        }
                        tested += 1
                        try:
                            trades = run_sim(df_clean.copy(), params)
                            if not trades: continue
                            m = calculate_metrics(trades, df_clean, 0)
                            s = score(m)
                            if s > best_score:
                                best_score = s; best_p = dict(params); best_m = m
                                if m.win_rate >= 50:
                                    logger.info(f"[{tested}] WR={m.win_rate:.1f}% PnL={m.total_pnl_pct:.1f}% vsBH={m.total_pnl_pct-m.buy_hold_pct:+.1f}pp PF={m.profit_factor:.2f} N={m.total_trades} sc={s:.0f} | sq={sq_th} sb={sq_bars} sl={sl_m} tp={tp_m} tr={trail_m}")
                        except: pass
                        if tested % 2000 == 0:
                            logger.info(f"Progress: {tested} ({time.time()-t0:.0f}s)")

    logger.info(f"\n{'='*60}")
    logger.info(f"Done: {tested} combos in {time.time()-t0:.0f}s")
    if best_p:
        logger.info(f"Best score: {best_score:.1f}")
        logger.info(f"Best params: {json.dumps(best_p)}")
        m = best_m
        logger.info(f"  Trades={m.total_trades} WR={m.win_rate:.1f}% PnL={m.total_pnl_pct:.2f}% B&H={m.buy_hold_pct:.2f}% vsBH={m.total_pnl_pct-m.buy_hold_pct:+.2f}pp PF={m.profit_factor:.2f} DD={m.max_drawdown_pct:.2f}% Sharpe={m.sharpe_ratio:.2f} R:R={m.avg_r_r:.2f}")

if __name__ == "__main__":
    main()
