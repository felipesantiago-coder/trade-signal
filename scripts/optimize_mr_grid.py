"""
optimize_mr_grid.py
-------------------
Grid search otimizacao dos parametros de Mean-Reversion (RANGING regime).

Foco em duas frentes:
  A) LONG: melhorar WR (atualmente 41pct, 7w/10l)
  B) SHORT: aumentar frequencia (atualmente apenas 5 trades)
"""
import logging
import sys
import time
from itertools import product

logging.basicConfig(level=logging.ERROR)

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/z/my-project/trade-signal')

from indicators import compute_indicators
from regime_engine import classify_regimes_v2, get_regime_params
from strategy_regime import (
    evaluate_mean_reversion_long, evaluate_mean_reversion_short,
)
from strategy_profiles import get_profile
from backtest import (
    TradeResult, _apply_costs, fetch_historical_ohlcv,
    DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS,
)


def fast_sim_mr(
    df_ind, params_long, params_short,
    max_bars_held=72,
    fee_pct=DEFAULT_FEE_PCT,
    spread_bps=DEFAULT_SPREAD_BPS,
    slippage_bps=DEFAULT_SLIPPAGE_BPS,
):
    """Simulacao rapida MR com params customizados."""
    trades = []
    i = 0
    n = len(df_ind)

    while i < n:
        row = df_ind.iloc[i]
        regime_v2 = str(row.get('regime_v2', ''))
        confidence = float(row.get('regime_confidence', 0.5))

        if regime_v2 != 'RANGING':
            i += 1
            continue
        if confidence < 0.2:
            i += 1
            continue

        atr_pct = float(row.get('atr_percentile', 0.5))
        if atr_pct < 0.10 or atr_pct > 0.85:
            i += 1
            continue

        signal = None
        signal = evaluate_mean_reversion_long(row, params_long)
        if signal is None:
            signal = evaluate_mean_reversion_short(row, params_short)
        if signal is None:
            i += 1
            continue

        # Simulate trade
        entry_price = signal.entry_price
        sl = signal.stop_loss
        tp = signal.take_profit
        atr = signal.atr
        is_long = signal.type.value == 'LONG'

        exit_price = None
        exit_reason = None
        bars = 0

        for j in range(i + 1, min(i + max_bars_held, n)):
            future = df_ind.iloc[j]
            bars = j - i
            if is_long:
                if float(future['low']) <= sl:
                    exit_price = sl
                    exit_reason = 'sl'
                    break
                if float(future['high']) >= tp:
                    exit_price = tp
                    exit_reason = 'tp'
                    break
            else:
                if float(future['high']) >= sl:
                    exit_price = sl
                    exit_reason = 'sl'
                    break
                if float(future['low']) <= tp:
                    exit_price = tp
                    exit_reason = 'tp'
                    break

        if exit_price is None:
            last_j = min(i + max_bars_held, n) - 1
            exit_price = float(df_ind.iloc[last_j]['close'])
            exit_reason = 'timeout'
            bars = last_j - i

        _, adj_exit, cost_pct = _apply_costs(
            entry_price, exit_price, is_long,
            fee_pct, spread_bps, slippage_bps,
        )
        if is_long:
            pnl_pct = (adj_exit - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - adj_exit) / entry_price * 100

        trades.append(TradeResult(
            entry_ts=row.name,
            exit_ts=df_ind.iloc[min(i + bars, n - 1)].name,
            type=signal.type.value,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=sl,
            take_profit=tp,
            atr=atr,
            rsi=signal.rsi,
            pnl_pct=round(pnl_pct, 4),
            pnl_abs=round(exit_price - entry_price, 2),
            bars_held=bars,
            exit_reason=exit_reason,
            atr_percentile=atr_pct,
        ))

        i += bars + 1

    return trades


def make_long_params(rsi_min, rsi_max, sl_factor, tp_factor):
    base_sl = 2.12
    return {
        'strategy_type': 'mean_reversion',
        'allow_long': True, 'allow_short': False,
        'sl_mult': base_sl * sl_factor,
        'tp_mult': base_sl * tp_factor,
        'rsi_long_range': (rsi_min, rsi_max),
        'rsi_short_range': (99, 99),
        'require_volume': False,
        'min_confidence': 0.2,
    }


def make_short_params(rsi_min, rsi_max, sl_factor, tp_factor):
    base_sl = 2.12
    return {
        'strategy_type': 'mean_reversion',
        'allow_long': False, 'allow_short': True,
        'sl_mult': base_sl * sl_factor,
        'tp_mult': base_sl * tp_factor,
        'rsi_long_range': (0, 0),
        'rsi_short_range': (rsi_min, rsi_max),
        'require_volume': False,
        'min_confidence': 0.2,
    }


def main():
    print('Fetching data...')
    df = fetch_historical_ohlcv('BTC/USDT', '1h', 730)
    df_ind = compute_indicators(df, '1h')
    df_ind = classify_regimes_v2(df_ind, hysteresis_bars=3)
    print(f'Data: {len(df_ind)} bars, RANGING: {(df_ind["regime_v2"]=="RANGING").sum()}')

    # ============================================================
    # PHASE 1: Grid search LONG params
    # ============================================================
    print('\n' + '='*60)
    print('PHASE 1: Optimizing MR LONG')
    print('='*60)

    rsi_long_ranges = [
        (20, 42),   # current
        (20, 38),   # tighter upper
        (20, 35),   # much tighter
        (25, 42),   # tighter lower
        (25, 38),   # tighter both
        (15, 45),   # wider
        (20, 40),   # slight tighten
    ]
    sl_factors = [1.5, 2.0, 2.5]
    tp_factors = [3.0, 4.0, 5.0, 6.0]

    best_long = None
    best_long_score = -999
    results_long = []

    total = len(rsi_long_ranges) * len(sl_factors) * len(tp_factors)
    count = 0

    for rsi_range, sl_f, tp_f in product(rsi_long_ranges, sl_factors, tp_factors):
        count += 1
        params_l = make_long_params(rsi_range[0], rsi_range[1], sl_f, tp_f)
        params_s = make_short_params(58, 80, 1.5, 3.0)  # fixed SHORT

        trades = fast_sim_mr(df_ind, params_l, params_s)
        longs = [t for t in trades if t.type == 'LONG']
        if not longs:
            continue

        wins = [t for t in longs if t.pnl_pct > 0]
        pnl = sum(t.pnl_pct for t in longs)
        wr = len(wins) / len(longs) * 100
        pf = sum(t.pnl_pct for t in wins) / abs(sum(t.pnl_pct for t in longs if t.pnl_pct <= 0)) if any(t.pnl_pct <= 0 for t in longs) else 99
        dd = max(0, -min(np.cumsum([t.pnl_pct for t in longs]))) if longs else 0
        score = pnl - 2 * dd  # reward PnL, penalize DD

        results_long.append({
            'rsi_range': rsi_range, 'sl_f': sl_f, 'tp_f': tp_f,
            'trades': len(longs), 'wr': wr, 'pnl': pnl, 'dd': dd, 'pf': pf, 'score': score
        })

        if len(longs) >= 3 and score > best_long_score:
            best_long_score = score
            best_long = results_long[-1]

    # Sort by score
    results_long.sort(key=lambda x: x['score'], reverse=True)
    print(f'\nTop 10 LONG configurations ({total} tested, {len(results_long)} with trades):')
    print(f'{"RSI":>12} {"SL_f":>5} {"TP_f":>5} {"Trades":>6} {"WR%":>6} {"PnL%":>7} {"DD%":>6} {"PF":>6} {"Score":>7}')
    print('-' * 75)
    for r in results_long[:10]:
        print(f'{str(r["rsi_range"]):>12} {r["sl_f"]:5.1f} {r["tp_f"]:5.1f} {r["trades"]:6d} {r["wr"]:6.1f} {r["pnl"]:+7.2f} {r["dd"]:6.2f} {r["pf"]:6.2f} {r["score"]:+7.2f}')

    # ============================================================
    # PHASE 2: Grid search SHORT params
    # ============================================================
    print('\n' + '='*60)
    print('PHASE 2: Optimizing MR SHORT')
    print('='*60)

    rsi_short_ranges = [
        (58, 80),   # current
        (55, 80),   # wider lower
        (55, 85),   # wider both
        (50, 80),   # much wider lower
        (58, 75),   # tighter upper
        (60, 85),   # shifted up
        (55, 75),   # wider lower, tighter upper
        (50, 85),   # much wider
    ]
    sl_factors_s = [1.5, 2.0, 2.5]
    tp_factors_s = [3.0, 4.0, 5.0, 6.0]

    # Use best LONG params as fixed
    if best_long:
        fixed_long = make_long_params(best_long['rsi_range'][0], best_long['rsi_range'][1], best_long['sl_f'], best_long['tp_f'])
    else:
        fixed_long = make_long_params(20, 42, 1.5, 3.0)

    best_short = None
    best_short_score = -999
    results_short = []

    total = len(rsi_short_ranges) * len(sl_factors_s) * len(tp_factors_s)
    count = 0

    for rsi_range, sl_f, tp_f in product(rsi_short_ranges, sl_factors_s, tp_factors_s):
        count += 1
        params_l = fixed_long
        params_s = make_short_params(rsi_range[0], rsi_range[1], sl_f, tp_f)

        trades = fast_sim_mr(df_ind, params_l, params_s)
        shorts = [t for t in trades if t.type == 'SHORT']
        if not shorts:
            continue

        wins = [t for t in shorts if t.pnl_pct > 0]
        pnl = sum(t.pnl_pct for t in shorts)
        wr = len(wins) / len(shorts) * 100
        pf = sum(t.pnl_pct for t in wins) / abs(sum(t.pnl_pct for t in shorts if t.pnl_pct <= 0)) if any(t.pnl_pct <= 0 for t in shorts) else 99
        dd = max(0, -min(np.cumsum([t.pnl_pct for t in shorts]))) if shorts else 0
        score = pnl - 2 * dd

        results_short.append({
            'rsi_range': rsi_range, 'sl_f': sl_f, 'tp_f': tp_f,
            'trades': len(shorts), 'wr': wr, 'pnl': pnl, 'dd': dd, 'pf': pf, 'score': score
        })

        if len(shorts) >= 2 and score > best_short_score:
            best_short_score = score
            best_short = results_short[-1]

    results_short.sort(key=lambda x: x['score'], reverse=True)
    print(f'\nTop 10 SHORT configurations ({total} tested, {len(results_short)} with trades):')
    print(f'{"RSI":>12} {"SL_f":>5} {"TP_f":>5} {"Trades":>6} {"WR%":>6} {"PnL%":>7} {"DD%":>6} {"PF":>6} {"Score":>7}')
    print('-' * 75)
    for r in results_short[:10]:
        print(f'{str(r["rsi_range"]):>12} {r["sl_f"]:5.1f} {r["tp_f"]:5.1f} {r["trades"]:6d} {r["wr"]:6.1f} {r["pnl"]:+7.2f} {r["dd"]:6.2f} {r["pf"]:6.2f} {r["score"]:+7.2f}')

    # ============================================================
    # PHASE 3: Combined validation
    # ============================================================
    print('\n' + '='*60)
    print('PHASE 3: COMBINED VALIDATION')
    print('='*60)

    # Current system
    params_l_curr = make_long_params(20, 42, 1.5, 3.0)
    params_s_curr = make_short_params(58, 80, 1.5, 3.0)
    trades_curr = fast_sim_mr(df_ind, params_l_curr, params_s_curr)

    # Best combined
    if best_long and best_short:
        params_l_best = make_long_params(best_long['rsi_range'][0], best_long['rsi_range'][1], best_long['sl_f'], best_long['tp_f'])
        params_s_best = make_short_params(best_short['rsi_range'][0], best_short['rsi_range'][1], best_short['sl_f'], best_short['tp_f'])
        trades_best = fast_sim_mr(df_ind, params_l_best, params_s_best)

        print(f'\n{"":30s} {"CURRENT":>10} {"BEST":>10} {"DELTA":>10}')
        print('-' * 65)

        for label, tlist in [('CURRENT', trades_curr), ('BEST', trades_best)]:
            longs_t = [t for t in tlist if t.type == 'LONG']
            shorts_t = [t for t in tlist if t.type == 'SHORT']
            wins = [t for t in tlist if t.pnl_pct > 0]
            pnl = sum(t.pnl_pct for t in tlist)
            dd = max(0, -min(np.cumsum(sorted([t.pnl_pct for t in tlist])))) if tlist else 0
            wr = len(wins)/len(tlist)*100 if tlist else 0
            pf_val = sum(t.pnl_pct for t in wins)/abs(sum(t.pnl_pct for t in tlist if t.pnl_pct <= 0)) if any(t.pnl_pct <= 0 for t in tlist) else 99
            if label == 'CURRENT':
                curr_vals = (len(tlist), len(longs_t), len(shorts_t), wr, pnl, dd, pf_val)
            else:
                best_vals = (len(tlist), len(longs_t), len(shorts_t), wr, pnl, dd, pf_val)

        for i, name in enumerate(['Total trades', 'LONG trades', 'SHORT trades', 'WR%', 'PnL%', 'DD%', 'PF']):
            delta = best_vals[i] - curr_vals[i]
            if isinstance(curr_vals[i], float):
                print(f'{name:30s} {curr_vals[i]:10.2f} {best_vals[i]:10.2f} {delta:+10.2f}')
            else:
                print(f'{name:30s} {curr_vals[i]:10d} {best_vals[i]:10d} {delta:+10d}')

        # Print individual trades
        print(f'\nBest LONG: RSI={best_long["rsi_range"]}, SL={best_long["sl_f"]:.1f}x, TP={best_long["tp_f"]:.1f}x')
        print(f'  {best_long["trades"]} trades, WR={best_long["wr"]:.1f}%, PnL={best_long["pnl"]:+.2f}%')
        print(f'Best SHORT: RSI={best_short["rsi_range"]}, SL={best_short["sl_f"]:.1f}x, TP={best_short["tp_f"]:.1f}x')
        print(f'  {best_short["trades"]} trades, WR={best_short["wr"]:.1f}%, PnL={best_short["pnl"]:+.2f}%')


if __name__ == '__main__':
    main()
