r"""
explore_50pct_balanced.py
------------------------
Exploracao sistematica de variacoes da estrategia BBWP Squeeze
para encontrar configuracao com >=50% retorno anual composto.

Foco: abordagem EQUILIBRADA (DD controlado, WR razoavel, sem alavancagem).

Dimensoes de exploracao:
1. Saidas: TP 3.0-5.0x ATR, Trailing 1.5-3.0x ATR, SL 1.5-2.5x ATR
2. Entradas: relaxar volume, stoch_rsi bounds, ADX, BBWP threshold
3. Cooldown: reduzir para mais trades
4. Position sizing: 1-3% risk/trade com reinvestimento composto
"""
import sys, os, json, time, logging, itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

from indicators import compute_indicators
from backtest import (
    fetch_historical_ohlcv, TradeResult, _apply_costs,
    DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS,
)
from strategy import SignalType


def run_bbwp_backtest(df_clean, params, atr_pct_min=0.10, atr_pct_max=0.90):
    """
    Versao parametrizada do _simulate_bbwp_squeeze.
    Modifica BBWP_SQUEEZE_PARAMS globalmente, roda, restaura.
    """
    from strategy_bbwp_squeeze import (
        BBWP_SQUEEZE_PARAMS, reset_cooldown, _check_cooldown,
        _register_signal, _is_squeeze_breakout, _adx_confirms_trend,
        _stoch_rsi_confirms, _volume_confirms, _get_sl_mult,
    )

    # Save and override params
    orig_params = dict(BBWP_SQUEEZE_PARAMS)
    BBWP_SQUEEZE_PARAMS.update(params)
    reset_cooldown()

    trades = []
    n = len(df_clean)

    _trail_dist = params.get("trailing_atr_mult", 1.5)
    _max_bars = params.get("max_bars_held", 96)
    _use_trailing = params.get("use_trailing", True)
    _tp1_pct = params.get("tp1_pct", 0.50)
    _post_tp1_sl_buf = params.get("post_tp1_sl_buffer", 0.5)
    _fee, _spread, _slip = 0.016, 2.0, 2.0

    i = 0
    while i < n:
        row = df_clean.iloc[i]
        if i < 1:
            i += 1
            continue

        critical = ["ema20","ema50","ema200","rsi","atr","atr_percentile",
                    "bbwp","stoch_rsi_k","stoch_rsi_d","bb_lower","bb_upper",
                    "volume","volume_sma20","adx"]
        if any(pd.isna(row.get(c)) for c in critical):
            i += 1
            continue

        atr_pct = float(row.get("atr_percentile", 0.5))
        if atr_pct < atr_pct_min or atr_pct > atr_pct_max:
            i += 1
            continue

        prev = df_clean.iloc[i - 1]
        direction = None
        result = None

        if _check_cooldown(i, direction="long"):
            r = _eval_dir(row, prev, "long", i, df_clean)
            if r: result, direction = r, "long"

        if result is None and _check_cooldown(i, direction="short"):
            r = _eval_dir(row, prev, "short", i, df_clean)
            if r: result, direction = r, "short"

        if result is None:
            i += 1
            continue

        sl, tp, atr, bbwp = result
        entry_price = float(row["close"])
        is_long = (direction == "long")
        _register_signal(i, direction=direction)

        current_sl = sl
        be_triggered = False
        trailing_activated = False
        highest_favorable = entry_price
        sl_updates = 0
        tp1_filled = False
        tp1_fill_price = 0.0
        was_trailing_exit = False

        for j in range(i + 1, min(i + _max_bars, n)):
            future = df_clean.iloc[j]
            f_close, f_low, f_high = float(future["close"]), float(future["low"]), float(future["high"])
            bars = j - i

            if is_long:
                highest_favorable = max(highest_favorable, f_high)
            else:
                highest_favorable = min(highest_favorable, f_low)

            sl_hit = (f_low <= current_sl) if is_long else (f_high >= current_sl)
            tp_hit = (f_high >= tp) if is_long else (f_low <= tp)

            # TP1 partial fill
            if tp_hit and not tp1_filled:
                tp1_filled = True
                tp1_fill_price = tp
                if _use_trailing:
                    be_triggered = True
                    trailing_activated = True
                    buf = atr * _post_tp1_sl_buf
                    current_sl = (tp - buf) if is_long else (tp + buf)
                if not _use_trailing:
                    break

            # Both TP and SL same bar after TP1
            if tp_hit and sl_hit and tp1_filled:
                pnl = _partial_pnl(entry_price, tp1_fill_price, current_sl, is_long, _fee, _spread, _slip, _tp1_pct)
                trades.append(_make_trade(row, df_clean.iloc[j], entry_price, tp, sl, tp, atr, pnl, bars, "tp1_then_sl",
                                          atr_pct, is_long, True, True, sl_updates))
                was_trailing_exit = True
                break

            # SL hit
            if sl_hit and not tp_hit:
                if tp1_filled:
                    pnl = _partial_pnl(entry_price, tp1_fill_price, current_sl, is_long, _fee, _spread, _slip, _tp1_pct)
                    trades.append(_make_trade(row, df_clean.iloc[j], entry_price, current_sl, sl, tp, atr, pnl, bars, "trailing_sl",
                                              atr_pct, is_long, True, True, sl_updates))
                    was_trailing_exit = True
                else:
                    _, adj_exit, _ = _apply_costs(entry_price, current_sl, is_long, _fee, _spread, _slip)
                    pnl = ((adj_exit - entry_price) / entry_price * 100) if is_long else ((entry_price - adj_exit) / entry_price * 100)
                    trades.append(_make_trade(row, df_clean.iloc[j], entry_price, current_sl, sl, tp, atr, pnl, bars, "sl",
                                              atr_pct, is_long, False, False, 0))
                break

            # Trailing ratchet
            if trailing_activated and _use_trailing:
                td = atr * _trail_dist
                if is_long:
                    new_trail = highest_favorable - td
                    if new_trail > current_sl:
                        current_sl = new_trail
                        sl_updates += 1
                else:
                    new_trail = highest_favorable + td
                    if new_trail < current_sl:
                        current_sl = new_trail
                        sl_updates += 1

        # Timeout
        if not trades or trades[-1].entry_ts != row.name:
            last_j = min(i + _max_bars, n) - 1
            exit_price = float(df_clean.iloc[last_j]["close"])
            bars = last_j - i
            if tp1_filled:
                pnl = _partial_pnl(entry_price, tp1_fill_price, exit_price, is_long, _fee, _spread, _slip, _tp1_pct)
                trades.append(_make_trade(row, df_clean.iloc[last_j], entry_price, exit_price, sl, tp, atr, pnl, bars, "tp1_then_timeout",
                                          atr_pct, is_long, True, True, sl_updates))
                was_trailing_exit = trailing_activated
            else:
                _, adj_exit, _ = _apply_costs(entry_price, exit_price, is_long, _fee, _spread, _slip)
                pnl = ((adj_exit - entry_price) / entry_price * 100) if is_long else ((entry_price - adj_exit) / entry_price * 100)
                trades.append(_make_trade(row, df_clean.iloc[last_j], entry_price, exit_price, sl, tp, atr, pnl, bars, "timeout",
                                          atr_pct, is_long, False, False, 0))

        _register_signal(i + (trades[-1].bars_held if trades else 0),
                         was_trailing=was_trailing_exit, direction=direction)
        i += (trades[-1].bars_held if trades else 1) + 1

    BBWP_SQUEEZE_PARAMS.update(orig_params)
    return trades


def _eval_dir(row, prev_row, direction, idx, df):
    from strategy_bbwp_squeeze import (
        _is_squeeze_breakout, _adx_confirms_trend, _stoch_rsi_confirms,
        _volume_confirms, _get_sl_mult, BBWP_SQUEEZE_PARAMS,
    )
    p = BBWP_SQUEEZE_PARAMS

    if not _is_squeeze_breakout(row, prev_row, idx=idx, df=df): return None
    if not _adx_confirms_trend(row): return None

    atr_pct = float(row.get("atr_percentile", 0.5))
    if not (p["atr_pct_min"] <= atr_pct <= p["atr_pct_max"]): return None
    if not _volume_confirms(row): return None

    close = float(row["close"])
    bb_upper = float(row.get("bb_upper", 0))
    bb_lower = float(row.get("bb_lower", 0))
    ema50 = float(row.get("ema50", 0))
    ema200 = float(row.get("ema200", 0))
    atr = float(row.get("atr", 0))

    if atr <= 0 or close <= 0 or bb_upper <= 0 or bb_lower <= 0: return None

    sl_mult = _get_sl_mult(row)
    bb_buffer = p.get("bb_breakout_buffer", 0.05)

    if direction == "long":
        if close <= bb_upper + bb_buffer * (bb_upper - bb_lower): return None
        if not _stoch_rsi_confirms(row, prev_row, "long"): return None
        if close <= ema50: return None
        if p.get("ema200_filter", True) and ema200 > 0 and close <= ema200: return None
        sl = close - sl_mult * atr
        if sl <= 0: return None
        tp = close + p["tp_atr_mult"] * atr
        return (sl, tp, atr, float(row.get("bbwp", 100)))
    else:
        if close >= bb_lower - bb_buffer * (bb_upper - bb_lower): return None
        if not _stoch_rsi_confirms(row, prev_row, "short"): return None
        if close >= ema50: return None
        if p.get("ema200_filter", True) and ema200 > 0 and close >= ema200: return None
        sl = close + sl_mult * atr
        tp = close - p["tp_atr_mult"] * atr
        return (sl, tp, atr, float(row.get("bbwp", 100)))


def _partial_pnl(entry, tp1_price, exit_price, is_long, fee, spread, slip, tp1_pct):
    _, adj_tp1, _ = _apply_costs(entry, tp1_price, is_long, fee, spread, slip)
    _, adj_exit, _ = _apply_costs(entry, exit_price, is_long, fee, spread, slip)
    if is_long:
        tp1_pnl = (adj_tp1 - entry) / entry * 100
        exit_pnl = (adj_exit - entry) / entry * 100
    else:
        tp1_pnl = (entry - adj_tp1) / entry * 100
        exit_pnl = (entry - adj_exit) / entry * 100
    return round(tp1_pct * tp1_pnl + (1 - tp1_pct) * exit_pnl, 4)


def _make_trade(row_entry, row_exit, entry_price, exit_price, sl, tp, atr, pnl, bars, reason,
                atr_pct, is_long, be, trailing, sl_updates):
    return TradeResult(
        entry_ts=row_entry.name, exit_ts=row_exit.name,
        type=SignalType.LONG.value if is_long else SignalType.SHORT.value,
        entry_price=entry_price, exit_price=exit_price,
        stop_loss=sl, take_profit=tp, atr=atr, rsi=float(row_entry.get("rsi", 0)),
        pnl_pct=round(pnl, 4), pnl_abs=round(exit_price - entry_price, 2),
        bars_held=bars, exit_reason=reason, atr_percentile=atr_pct,
        be_triggered=be, trailing_activated=trailing, partial_tp_filled=("tp1" in reason),
        sl_updates=sl_updates,
    )


def compute_compound_return(trades, risk_pct=2.0, days=730):
    """
    Compound return with fixed-fractional position sizing.
    risk_pct: % of equity risked per trade (e.g., 2.0 = 2%)
    Position size = risk_pct / SL_distance_pct
    """
    if not trades:
        return {"final_equity": 100.0, "total_return_pct": 0, "annual_return_pct": 0,
                "max_dd_pct": 0, "trades": 0, "sharpe": 0}

    equity = 100.0
    peak = equity
    max_dd = 0
    returns_list = []

    for t in trades:
        # Calculate SL distance as % of entry
        if t.stop_loss > 0 and t.entry_price > 0:
            if "long" in t.type:
                sl_dist_pct = max((t.entry_price - t.stop_loss) / t.entry_price * 100, 0.05)
            else:
                sl_dist_pct = max((t.stop_loss - t.entry_price) / t.entry_price * 100, 0.05)
        else:
            sl_dist_pct = 2.0

        # Position size based on risk
        pos_size = risk_pct / sl_dist_pct

        # Equity change
        equity_change = equity * pos_size * (t.pnl_pct / 100.0)
        ret_pct = equity_change / equity * 100 if equity > 0 else 0
        returns_list.append(ret_pct)

        equity += equity_change
        equity = max(equity, 0.01)

        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd

    total_return = (equity - 100.0)
    annual_return = total_return * (365.0 / days)

    # Sharpe (simplified)
    if len(returns_list) > 1:
        mean_r = np.mean(returns_list)
        std_r = np.std(returns_list)
        sharpe = (mean_r / std_r * np.sqrt(365 * 24 / max(np.mean([t.bars_held for t in trades]), 1))) if std_r > 0 else 0
    else:
        sharpe = 0

    return {
        "final_equity": round(equity, 2),
        "total_return_pct": round(total_return, 2),
        "annual_return_pct": round(annual_return, 2),
        "max_dd_pct": round(max_dd, 2),
        "trades": len(trades),
        "sharpe": round(sharpe, 2),
    }


def compute_metrics(trades):
    if not trades:
        return {"total_trades": 0, "win_rate": 0, "profit_factor": 0, "total_pnl": 0, "max_dd": 0}

    wins = [t.pnl_pct for t in trades if t.pnl_pct > 0]
    losses = [t.pnl_pct for t in trades if t.pnl_pct <= 0]
    total_pnl = sum(t.pnl_pct for t in trades)
    wr = len(wins) / len(trades) * 100
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    pf = gross_profit / gross_loss

    peak = max_dd = running = 0
    for t in trades:
        running += t.pnl_pct
        if running > peak: peak = running
        dd = peak - running
        if dd > max_dd: max_dd = dd

    return {
        "total_trades": len(trades), "win_rate": round(wr, 1),
        "profit_factor": round(pf, 2), "total_pnl": round(total_pnl, 2),
        "max_dd": round(max_dd, 2),
    }


def build_param_grid():
    combinations = []
    base = {
        "bbwp_threshold": 15, "squeeze_recent_bars": 12,
        "require_bbwp_expansion": True, "volume_mult": 0.35,
        "stoch_rsi_ob": 56, "stoch_rsi_os": 44,
        "bb_breakout_buffer": 0.05, "adx_min": 16.0,
        "sl_atr_mult": 2.2, "tp_atr_mult": 3.0, "tp1_pct": 0.50,
        "use_trailing": True, "trailing_atr_mult": 1.5,
        "post_tp1_sl_buffer": 0.5, "max_bars_held": 96,
        "cooldown": 2, "cooldown_trailing": 2, "cooldown_opp_dir": 1,
        "use_directional_cooldown": True, "ema200_filter": True,
        "atr_pct_min": 0.10, "atr_pct_max": 0.90,
        "sl_atr_mult_high_vol": 2.2, "sl_atr_mult_low_vol": 2.2,
        "stoch_rsi_cross_enable": True, "stoch_rsi_min_delta": 0,
        "min_bbwp_bars": 1, "be_trigger_atr_mult": 1.0,
        "use_divergence_exit": False, "divergence_min_bars": 3,
    }

    def P(**overrides):
        p = dict(base)
        p.update(overrides)
        return p

    # ---- PHASE 1: Wider TP + trailing ----
    for tp in [3.5, 4.0, 4.5, 5.0]:
        for tr in [2.0, 2.5, 3.0]:
            for buf in [0.3, 0.5, 0.8]:
                combinations.append((f"P1_tp{tp}_tr{tr}_b{buf}",
                    P(tp_atr_mult=tp, trailing_atr_mult=tr, post_tp1_sl_buffer=buf)))

    # ---- PHASE 2: Entry relaxation ----
    for vol in [0.20, 0.25, 0.30]:
        for ob in [52, 54, 56]:
            for adx in [12.0, 14.0, 16.0]:
                combinations.append((f"P2_v{vol}_ob{ob}_adx{adx}",
                    P(volume_mult=vol, stoch_rsi_ob=ob, stoch_rsi_os=100-ob, adx_min=adx)))

    # ---- PHASE 3: SL/TP ratio ----
    for sl in [1.5, 1.8, 2.0, 2.5]:
        for tp in [3.5, 4.0, 5.0]:
            combinations.append((f"P3_sl{sl}_tp{tp}",
                P(sl_atr_mult=sl, sl_atr_mult_high_vol=sl, sl_atr_mult_low_vol=sl, tp_atr_mult=tp)))

    # ---- PHASE 4: Cooldown ----
    for cd in [1, 2]:
        for cdt in [1, 2]:
            combinations.append((f"P4_cd{cd}_cdt{cdt}", P(cooldown=cd, cooldown_trailing=cdt)))

    # ---- PHASE 5: Combined best ----
    for tp in [4.0, 5.0]:
        for sl in [1.8, 2.0]:
            for tr in [2.0, 2.5]:
                for vol in [0.25, 0.30]:
                    for cd in [1, 2]:
                        combinations.append((f"P5_tp{tp}_sl{sl}_tr{tr}_v{vol}_cd{cd}",
                            P(tp_atr_mult=tp, sl_atr_mult=sl, sl_atr_mult_high_vol=sl, sl_atr_mult_low_vol=sl,
                               trailing_atr_mult=tr, volume_mult=vol, cooldown=cd, cooldown_trailing=cd)))

    # ---- PHASE 6: No EMA200 + wider exits ----
    for tp in [4.0, 5.0, 6.0]:
        for tr in [2.0, 3.0]:
            for vol in [0.20, 0.25]:
                combinations.append((f"P6_tp{tp}_tr{tr}_v{vol}_no2k",
                    P(tp_atr_mult=tp, trailing_atr_mult=tr, volume_mult=vol,
                       ema200_filter=False, cooldown=1, cooldown_trailing=1)))

    # ---- PHASE 7: BBWP threshold ----
    for bbwp in [10, 12, 15, 18, 20]:
        for tp in [4.0, 5.0]:
            combinations.append((f"P7_bbwp{bbwp}_tp{tp}",
                P(bbwp_threshold=bbwp, tp_atr_mult=tp, trailing_atr_mult=2.0,
                   volume_mult=0.25, cooldown=1, cooldown_trailing=1)))

    # ---- PHASE 8: Aggressive balanced (wider everything) ----
    for tp in [4.0, 5.0, 6.0]:
        for tr in [2.5, 3.0, 3.5]:
            for sl in [1.5, 1.8, 2.0]:
                combinations.append((f"P8_tp{tp}_tr{tr}_sl{sl}",
                    P(tp_atr_mult=tp, trailing_atr_mult=tr,
                       sl_atr_mult=sl, sl_atr_mult_high_vol=sl, sl_atr_mult_low_vol=sl,
                       volume_mult=0.25, cooldown=1, cooldown_trailing=1,
                       post_tp1_sl_buffer=0.3)))

    # ---- PHASE 9: Ultra-wide trailing (let winners run) ----
    for tp in [3.0, 4.0]:
        for tr in [3.0, 4.0, 5.0]:
            for buf in [0.2, 0.3]:
                combinations.append((f"P9_tp{tp}_tr{tr}_b{buf}",
                    P(tp_atr_mult=tp, trailing_atr_mult=tr, post_tp1_sl_buffer=buf,
                       volume_mult=0.25, cooldown=1, cooldown_trailing=1)))

    # ---- PHASE 10: Everything combined ----
    for tp in [5.0, 6.0]:
        for tr in [3.0, 4.0]:
            for sl in [1.5, 1.8]:
                for vol in [0.20, 0.25]:
                    combinations.append((f"P10_tp{tp}_tr{tr}_sl{sl}_v{vol}",
                        P(tp_atr_mult=tp, trailing_atr_mult=tr,
                           sl_atr_mult=sl, sl_atr_mult_high_vol=sl, sl_atr_mult_low_vol=sl,
                           volume_mult=vol, cooldown=1, cooldown_trailing=1,
                           post_tp1_sl_buffer=0.3, stoch_rsi_ob=52, stoch_rsi_os=48,
                           adx_min=14.0)))

    return combinations


def main():
    print("=" * 70)
    print("EXPLORACAO EQUILIBRADA — Busca >=50% Retorno Anual")
    print("=" * 70)
    print()

    # 1. Fetch data
    print("[1/4] Baixando dados BTC/USDT 1h (730 dias)...")
    t0 = time.time()
    df = fetch_historical_ohlcv("BTC/USDT", "1h", 730)
    print(f"  {len(df):,} candles baixados em {time.time()-t0:.1f}s")

    # 2. Compute indicators
    print("[2/4] Calculando indicadores...")
    t0 = time.time()
    df_ind = compute_indicators(df, timeframe="1h")
    df_clean = df_ind.dropna(subset=[
        "ema20", "ema50", "ema200", "rsi", "atr", "atr_percentile",
        "bbwp", "stoch_rsi_k", "stoch_rsi_d", "bb_lower", "bb_upper",
        "volume", "volume_sma20", "adx",
    ]).copy()
    print(f"  {len(df_clean):,} candles limpos em {time.time()-t0:.1f}s")

    # 3. Build grid
    print("[3/4] Construindo grid de parametros...")
    grid = build_param_grid()
    print(f"  {len(grid)} combinacoes para testar")
    print()

    # 4. Run
    print("[4/4] Executando backtests...")
    print("-" * 70)

    qualified = []
    best_all = []
    t_start = time.time()

    for idx, (name, params) in enumerate(grid):
        try:
            trades = run_bbwp_backtest(df_clean, params)
            m = compute_metrics(trades)

            for risk_pct in [1.0, 2.0, 3.0, 4.0, 5.0]:
                c = compute_compound_return(trades, risk_pct=risk_pct, days=730)

                # Balanced filter
                if (c["annual_return_pct"] >= 50 and
                    m["total_trades"] >= 30 and
                    m["win_rate"] >= 38 and
                    c["max_dd_pct"] <= 35):

                    score = c["annual_return_pct"] / max(c["max_dd_pct"], 1)
                    qualified.append({
                        "name": name, "params": params, "risk_pct": risk_pct,
                        "m": m, "c": c, "score": round(score, 1),
                    })

                # Track best overall
                if len(best_all) < 50 or c["annual_return_pct"] > best_all[-1]["c"]["annual_return_pct"]:
                    score = c["annual_return_pct"] / max(c["max_dd_pct"], 1)
                    best_all.append({
                        "name": name, "params": params, "risk_pct": risk_pct,
                        "m": m, "c": c, "score": round(score, 1),
                    })
                    best_all.sort(key=lambda x: x["score"], reverse=True)
                    best_all = best_all[:50]

        except Exception as e:
            pass

        if (idx + 1) % 100 == 0 or idx == 0:
            elapsed = time.time() - t_start
            speed = (idx + 1) / max(elapsed, 0.01)
            eta = (len(grid) - idx - 1) / max(speed, 0.01)
            print(f"  [{idx+1}/{len(grid)}] {speed:.1f}/s ETA {eta:.0f}s | qualified={len(qualified)}")

    elapsed_total = time.time() - t_start
    print(f"\nConcluido em {elapsed_total:.1f}s")

    # Sort
    qualified.sort(key=lambda x: x["score"], reverse=True)

    if qualified:
        print(f"\n{'='*70}")
        print(f"{len(qualified)} CONFIGURACOES >=50% ANUAL (equilibradas)")
        print(f"{'='*70}")

        for i, r in enumerate(qualified[:30]):
            c, m, p = r["c"], r["m"], r["params"]
            print(f"\n#{i+1} [{r['name']}] risk={r['risk_pct']}% | SCORE={r['score']}")
            print(f"  Anual: {c['annual_return_pct']:+.1f}% | Total: {c['total_return_pct']:+.1f}% | Eq: {c['final_equity']}")
            print(f"  MaxDD: {c['max_dd_pct']:.1f}% | Trades: {m['total_trades']} | WR: {m['win_rate']:.1f}% | PF: {m['profit_factor']:.2f}")
            print(f"  TP={p['tp_atr_mult']}x SL={p['sl_atr_mult']}x Trail={p['trailing_atr_mult']}x Buf={p['post_tp1_sl_buffer']}")
            print(f"  Vol={p['volume_mult']} SRSI=[{p['stoch_rsi_os']},{p['stoch_rsi_ob']}] ADX>={p['adx_min']} BBWP<{p['bbwp_threshold']}")
            print(f"  CD={p['cooldown']}/CDT={p['cooldown_trailing']}/CDO={p['cooldown_opp_dir']} EMA200={'ON' if p['ema200_filter'] else 'OFF'}")

        # Save
        out_data = {"total_tested": len(grid), "qualified": len(qualified), "top": []}
        for r in qualified[:10]:
            out_data["top"].append({
                "name": r["name"], "risk_pct": r["risk_pct"],
                "annual_return_pct": r["c"]["annual_return_pct"],
                "total_return_pct": r["c"]["total_return_pct"],
                "max_dd_pct": r["c"]["max_dd_pct"],
                "final_equity": r["c"]["final_equity"],
                "trades": r["m"]["total_trades"],
                "win_rate": r["m"]["win_rate"],
                "profit_factor": r["m"]["profit_factor"],
                "params": {k: v for k, v in r["params"].items()
                           if k in ["tp_atr_mult","sl_atr_mult","trailing_atr_mult",
                                    "post_tp1_sl_buffer","volume_mult","stoch_rsi_ob",
                                    "stoch_rsi_os","adx_min","cooldown","cooldown_trailing",
                                    "cooldown_opp_dir","ema200_filter","bbwp_threshold",
                                    "bb_breakout_buffer","max_bars_held"]},
            })
    else:
        print(f"\n{'='*70}")
        print("Nenhuma config atingiu >=50% com filtros equilibrados.")
        print("Mostrando melhores resultados gerais:")
        print(f"{'='*70}")

        for i, r in enumerate(best_all[:20]):
            c, m, p = r["c"], r["m"], r["params"]
            print(f"\n#{i+1} [{r['name']}] risk={r['risk_pct']}% | SCORE={r['score']}")
            print(f"  Anual: {c['annual_return_pct']:+.1f}% | Total: {c['total_return_pct']:+.1f}% | Eq: {c['final_equity']}")
            print(f"  MaxDD: {c['max_dd_pct']:.1f}% | Trades: {m['total_trades']} | WR: {m['win_rate']:.1f}% | PF: {m['profit_factor']:.2f}")
            print(f"  TP={p['tp_atr_mult']}x SL={p['sl_atr_mult']}x Trail={p['trailing_atr_mult']}x")
            print(f"  Vol={p['volume_mult']} ADX>={p['adx_min']} BBWP<{p['bbwp_threshold']} CD={p['cooldown']}")

        out_data = {"total_tested": len(grid), "qualified": 0, "best_overall": []}
        for r in best_all[:10]:
            out_data["best_overall"].append({
                "name": r["name"], "risk_pct": r["risk_pct"],
                "annual_return_pct": r["c"]["annual_return_pct"],
                "max_dd_pct": r["c"]["max_dd_pct"],
                "trades": r["m"]["total_trades"],
                "win_rate": r["m"]["win_rate"],
                "profit_factor": r["m"]["profit_factor"],
            })

    out_path = "/home/z/my-project/trade-signal/download/explore_50pct_results.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2, default=str)
    print(f"\nResultados salvos em: {out_path}")

    return qualified if qualified else best_all


if __name__ == "__main__":
    main()
