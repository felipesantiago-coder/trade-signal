#!/usr/bin/env python3
"""
Optimização rápida de parâmetros SBS.
Baixa dados uma vez e testa múltiplos conjuntos de parâmetros.
"""
import sys, os, time, logging, itertools
import numpy as np
import pandas as pd
from typing import List, Tuple

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
    """Run one simulation with given params."""
    global SBS_PARAMS
    # Temporarily update params
    orig = dict(SBS_PARAMS)
    SBS_PARAMS.update(params)

    # Recompute was_squeezed if lookback changed
    bbwp = compute_bbwp(df_work["bb_width"], lookback=SBS_PARAMS["bbwp_lookback"])
    df_work["bbwp"] = bbwp
    df_work["was_squeezed"] = bbwp.rolling(
        window=SBS_PARAMS["bbwp_was_squeezed_bars"]
    ).apply(lambda x: (x < SBS_PARAMS["bbwp_squeeze_threshold"]).any(), raw=False).astype(bool)

    stoch_k, stoch_d = compute_stoch_rsi(
        df_work["close"], df_work["rsi"],
        period=SBS_PARAMS["stoch_rsi_period"],
        k_smooth=SBS_PARAMS["stoch_rsi_k_smooth"],
        d_smooth=SBS_PARAMS["stoch_rsi_d_smooth"],
    )
    df_work["stoch_rsi_k"] = stoch_k
    df_work["stoch_rsi_d"] = stoch_d

    reset_cooldown()
    trades = []
    atr_filtered = 0
    i = 0
    n = len(df_work)
    p = SBS_PARAMS

    while i < n:
        row = df_work.iloc[i]
        if i < 2:
            i += 1; continue

        critical = ["ema20", "ema50", "rsi", "atr", "atr_percentile",
                     "bb_lower", "bb_upper", "bb_middle", "bb_width",
                     "volume", "volume_sma20", "close"]
        if any(pd.isna(row.get(c)) for c in critical) or pd.isna(row.get("stoch_rsi_k", np.nan)) or pd.isna(row.get("bbwp", np.nan)):
            i += 1; continue

        atr_pct = float(row.get("atr_percentile", 0.5))
        if atr_pct < p["atr_pct_min"] or atr_pct > p["atr_pct_max"]:
            atr_filtered += 1; i += 1; continue

        prev = df_work.iloc[i - 1]
        stoch_k_val = float(row["stoch_rsi_k"])
        stoch_d_val = float(row["stoch_rsi_d"])
        bbwp_val = float(row["bbwp"])
        vol = float(row["volume"])
        vol_sma20 = float(row["volume_sma20"])
        vol_ratio = vol / vol_sma20 if vol_sma20 > 0 else 0
        was_squeezed = bool(row.get("was_squeezed", False))
        divergence = detect_rsi_divergence(df_work["close"], df_work["rsi"], i, lookback=p.get("div_lookback", 30))

        result = evaluate_sbs_row(row, prev, i, stoch_k_val, stoch_d_val, bbwp_val, vol_ratio, was_squeezed, divergence)

        if result is None:
            i += 1; continue

        signal, mode, conviction, trail_mult, sl_mult = result
        entry_price = signal.entry_price
        sl = signal.stop_loss
        tp = signal.take_profit
        atr = signal.atr
        is_long = signal.type == SignalType.LONG
        exit_price = None
        exit_reason = None
        bars = 0
        current_sl = sl
        be_triggered = False
        trailing_activated = False
        highest_favorable = entry_price
        trail_distance = atr * trail_mult
        be_trigger_dist = atr * p["be_trigger_atr"]
        partial_tp_hit = False
        was_trailing_exit = False

        for j in range(i + 1, min(i + p["max_bars"], n)):
            future = df_work.iloc[j]
            f_close = float(future["close"])
            f_low = float(future["low"])
            f_high = float(future["high"])
            bars = j - i

            if is_long:
                highest_favorable = max(highest_favorable, f_high)
            else:
                highest_favorable = min(highest_favorable, f_low)

            if not trailing_activated:
                tp_hit = False
                if is_long and f_high >= tp:
                    tp_hit = True
                elif not is_long and f_low <= tp:
                    tp_hit = True
                if tp_hit:
                    if p["trailing_enabled"] and p["partial_tp_pct"] > 0:
                        partial_tp_hit = True
                        trailing_activated = True
                        be_triggered = True
                        current_sl = entry_price
                        continue
                    else:
                        exit_price = tp; exit_reason = "tp"; break

            sl_hit = False
            if is_long:
                if f_low <= current_sl: sl_hit = True
            else:
                if f_high >= current_sl: sl_hit = True

            if not be_triggered and p["trailing_enabled"]:
                fav_dist = abs(highest_favorable - entry_price)
                if fav_dist >= be_trigger_dist:
                    be_triggered = True
                    trailing_activated = True
                    current_sl = entry_price

            if trailing_activated and p["trailing_enabled"]:
                if is_long:
                    new_trail = highest_favorable - trail_distance
                    if new_trail > current_sl: current_sl = new_trail
                else:
                    new_trail = highest_favorable + trail_distance
                    if new_trail < current_sl: current_sl = new_trail

            if sl_hit:
                exit_price = current_sl
                exit_reason = "trailing_sl" if trailing_activated else "sl"
                was_trailing_exit = trailing_activated
                break

        if exit_price is None:
            last_j = min(i + p["max_bars"], n) - 1
            exit_price = float(df_work.iloc[last_j]["close"])
            exit_reason = "timeout"; bars = last_j - i

        _, adj_exit, cost_pct = _apply_costs(entry_price, exit_price, is_long, fee, spread, slip)
        if is_long:
            pnl_pct = (adj_exit - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - adj_exit) / entry_price * 100

        trades.append(TradeResult(
            entry_ts=row.name, exit_ts=df_work.iloc[min(i + bars, n - 1)].name,
            type=signal.type.value, entry_price=entry_price, exit_price=exit_price,
            stop_loss=sl, take_profit=tp, atr=atr, rsi=signal.rsi,
            pnl_pct=round(pnl_pct, 4), pnl_abs=round(exit_price - entry_price, 2),
            bars_held=bars, exit_reason=exit_reason, atr_percentile=atr_pct,
            be_triggered=be_triggered, trailing_activated=trailing_activated,
            partial_tp_filled=partial_tp_hit,
        ))
        _register_signal(i + bars, was_trailing=was_trailing_exit)
        i += bars + 1

    # Restore params
    SBS_PARAMS.update(orig)
    return trades


def score_result(metrics: BacktestMetrics) -> float:
    """Score function: prioritize WR>=60%, then vs B&H, then PF."""
    if metrics.total_trades < 10:
        return -1000
    s = 0
    # Win rate: big bonus for >= 60%
    if metrics.win_rate >= 60:
        s += 200
    elif metrics.win_rate >= 55:
        s += 100
    elif metrics.win_rate >= 50:
        s += 30
    else:
        s -= 50
    # vs B&H
    vs_bh = metrics.total_pnl_pct - metrics.buy_hold_pct
    s += vs_bh * 2
    # Profit factor
    if metrics.profit_factor > 1.5:
        s += 50
    elif metrics.profit_factor > 1.0:
        s += 20
    else:
        s -= 30
    # Max drawdown penalty
    s -= metrics.max_drawdown_pct * 0.5
    return s


def main():
    tf = sys.argv[1] if len(sys.argv) > 1 else "15m"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 730

    logger.info(f"Fetching data: {tf}/{days}d...")
    df = fetch_historical_ohlcv("BTC/USDT", tf, days)
    df_ind = compute_indicators(df, timeframe=tf)
    df_clean = df_ind.dropna(subset=[
        "ema20", "ema50", "rsi", "atr", "atr_percentile",
        "bb_lower", "bb_upper", "bb_middle", "bb_width",
        "volume", "volume_sma20",
    ]).copy()
    logger.info(f"Data ready: {len(df_clean)} candles")

    # Pre-compute stoch RSI and BBWP
    stoch_k, stoch_d = compute_stoch_rsi(df_clean["close"], df_clean["rsi"])
    bbwp = compute_bbwp(df_clean["bb_width"])
    df_clean["stoch_rsi_k"] = stoch_k
    df_clean["stoch_rsi_d"] = stoch_d
    df_clean["bbwp"] = bbwp

    bh_pct = (float(df_clean.iloc[-1]["close"]) - float(df_clean.iloc[0]["close"])) / float(df_clean.iloc[0]["close"]) * 100
    logger.info(f"B&H: {bh_pct:.2f}%")

    # Parameter grid
    param_combos = []
    for sq_th in [10, 15, 20, 25, 30]:
        for sq_bars in [12, 16, 20, 24]:
            for exp_min in [30, 40, 50]:
                for vol_min in [0.5, 0.7, 1.0]:
                    for sl_m in [1.2, 1.5, 1.8, 2.0, 2.5]:
                        for tp_m in [2.0, 2.5, 3.0, 4.0, 5.0]:
                            for trail_m in [1.0, 1.2, 1.5, 2.0]:
                                for max_bars in [24, 36, 48, 72]:
                                    for cooldown in [4, 6, 8, 12]:
                                        param_combos.append({
                                            "bbwp_squeeze_threshold": sq_th,
                                            "bbwp_was_squeezed_bars": sq_bars,
                                            "bbwp_expansion_min": exp_min,
                                            "vol_ratio_min": vol_min,
                                            "sl_atr_mult": sl_m,
                                            "tp_atr_mult": tp_m,
                                            "trail_atr_mult": trail_m,
                                            "max_bars": max_bars,
                                            "cooldown": cooldown,
                                        })

    logger.info(f"Testing {len(param_combos)} parameter combinations...")
    best_score = -9999
    best_params = None
    best_metrics = None
    t0 = time.time()
    tested = 0

    for params in param_combos:
        tested += 1
        if tested % 500 == 0:
            elapsed = time.time() - t0
            logger.info(f"Progress: {tested}/{len(param_combos)} ({elapsed:.0f}s) best_score={best_score:.1f}")

        try:
            trades = run_sim(df_clean.copy(), params)
            if not trades:
                continue
            metrics = calculate_metrics(trades, df_clean, 0)
            s = score_result(metrics)
            if s > best_score:
                best_score = s
                best_params = params
                best_metrics = metrics
                if metrics.win_rate >= 55 and metrics.total_pnl_pct > metrics.buy_hold_pct + 10:
                    logger.info(f"NEW BEST: WR={metrics.win_rate:.1f}% PnL={metrics.total_pnl_pct:.1f}% vsBH={metrics.total_pnl_pct - metrics.buy_hold_pct:+.1f}pp PF={metrics.profit_factor:.2f} score={s:.0f}")
                    for k, v in params.items():
                        logger.info(f"  {k}: {v}")
        except Exception as e:
            pass

    elapsed = time.time() - t0
    logger.info(f"\n{'='*60}")
    logger.info(f"Optimization complete: {tested} combos in {elapsed:.0f}s")
    if best_params:
        logger.info(f"\nBest score: {best_score:.1f}")
        logger.info(f"Best params:")
        for k, v in best_params.items():
            logger.info(f"  {k}: {v}")
        logger.info(f"Best metrics:")
        m = best_metrics
        logger.info(f"  Total Trades: {m.total_trades}")
        logger.info(f"  Win Rate: {m.win_rate:.1f}%")
        logger.info(f"  PnL: {m.total_pnl_pct:.2f}%")
        logger.info(f"  B&H: {m.buy_hold_pct:.2f}%")
        logger.info(f"  vs B&H: {m.total_pnl_pct - m.buy_hold_pct:+.2f} pp")
        logger.info(f"  PF: {m.profit_factor:.2f}")
        logger.info(f"  Max DD: {m.max_drawdown_pct:.2f}%")
        logger.info(f"  Sharpe: {m.sharpe_ratio:.2f}")
        logger.info(f"  Avg R:R: {m.avg_r_r:.2f}")


if __name__ == "__main__":
    main()
