"""
Analise detalhada dos trades MR: padroes win vs loss.
Usa o backtest real para gerar trades com todos os indicadores.
"""
import logging, sys, os, json
logging.basicConfig(level=logging.ERROR)

import numpy as np
import pandas as pd

os.environ.setdefault('EXCHANGE_ID', 'bybit')
sys.path.insert(0, '/home/z/my-project/trade-signal')

from indicators import compute_indicators
from regime_engine import classify_regimes_v2
from strategy_regime import evaluate_mean_reversion_long, evaluate_mean_reversion_short
from regime_engine import get_regime_params
from backtest import (
    TradeResult, _apply_costs, fetch_historical_ohlcv,
    DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS,
)

BASE_SL = 2.12


def sim_mr_with_details(df_ind, params_long, params_short, max_bars=72):
    """Simula MR e retorna trades com indicadores no entry."""
    trades = []
    i = 0
    n = len(df_ind)
    while i < n:
        row = df_ind.iloc[i]
        rv2 = str(row.get('regime_v2', ''))
        conf = float(row.get('regime_confidence', 0.5))
        if rv2 != 'RANGING' or conf < 0.2:
            i += 1; continue
        atr_pct = float(row.get('atr_percentile', 0.5))
        if atr_pct < 0.10 or atr_pct > 0.85:
            i += 1; continue

        sig = evaluate_mean_reversion_long(row, params_long)
        if sig is None:
            sig = evaluate_mean_reversion_short(row, params_short)
        if sig is None:
            i += 1; continue

        # Collect indicators at entry
        entry_info = {
            'rsi': float(row['rsi']),
            'adx': float(row.get('adx', 0)),
            'ema50_slope': float(row.get('ema50_slope', 0)),
            'bb_squeeze_pct': float(row.get('bb_squeeze_pct', 0.5)),
            'atr_percentile': atr_pct,
            'bb_width': float(row.get('bb_width', 0)),
            'bb_position': (float(row['close']) - float(row['bb_lower'])) / max(float(row['bb_upper']) - float(row['bb_lower']), 0.01),
            'close_to_bb_lower_pct': (float(row['close']) - float(row['bb_lower'])) / float(row['close']) * 100,
            'close_to_bb_middle_pct': abs(float(row['close']) - float(row['bb_middle'])) / float(row['close']) * 100,
            'confidence': conf,
            'rsi_delta': float(row.get('rsi_delta', 0)),
            'macd_hist': float(row.get('macd_hist', 0)),
            'volume_ratio': float(row['volume']) / max(float(row.get('volume_sma20', 1)), 1),
        }

        entry_price = sig.entry_price
        sl, tp, atr = sig.stop_loss, sig.take_profit, sig.atr
        is_long = sig.type.value == 'LONG'
        exit_price, exit_reason, bars = None, None, 0
        for j in range(i + 1, min(i + max_bars, n)):
            f = df_ind.iloc[j]
            bars = j - i
            if is_long:
                if float(f['low']) <= sl: exit_price, exit_reason = sl, 'sl'; break
                if float(f['high']) >= tp: exit_price, exit_reason = tp, 'tp'; break
            else:
                if float(f['high']) >= sl: exit_price, exit_reason = sl, 'sl'; break
                if float(f['low']) <= tp: exit_price, exit_reason = tp, 'tp'; break
        if exit_price is None:
            lj = min(i + max_bars, n) - 1
            exit_price = float(df_ind.iloc[lj]['close'])
            exit_reason, bars = 'timeout', lj - i

        _, adj_exit, _ = _apply_costs(entry_price, exit_price, is_long, DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS)
        pnl = (adj_exit - entry_price) / entry_price * 100 if is_long else (entry_price - adj_exit) / entry_price * 100

        trades.append({
            'type': sig.type.value, 'entry_ts': row.name,
            'pnl_pct': pnl, 'exit_reason': exit_reason, 'bars_held': bars,
            'entry_price': entry_price, 'sl': sl, 'tp': tp, 'atr': atr,
            'rr': abs(tp - entry_price) / abs(entry_price - sl) if entry_price != sl else 0,
            **entry_info,
        })
        i += bars + 1
    return trades


def compare_groups(wins, losses, label):
    """Compare indicator distributions between wins and losses."""
    print(f'\n=== {label} ===')
    print(f'  Wins: {len(wins)}, Losses: {len(losses)}')
    indicators = ['rsi', 'adx', 'ema50_slope', 'bb_squeeze_pct', 'atr_percentile',
                   'bb_position', 'close_to_bb_lower_pct', 'close_to_bb_middle_pct',
                   'confidence', 'rsi_delta', 'macd_hist', 'volume_ratio', 'rr']
    print(f'  {"Indicator":>25s} {"Win_mean":>10s} {"Loss_mean":>10s} {"Diff":>8s} {"Win_med":>9s} {"Loss_med":>9s}')
    print('  ' + '-' * 75)
    for ind in indicators:
        wv = [t[ind] for t in wins]
        lv = [t[ind] for t in losses]
        wm, lm = np.mean(wv), np.mean(lv)
        wmd, lmd = np.median(wv), np.median(lv)
        diff = wm - lm
        marker = ' <<<' if abs(diff) > 0.3 * max(abs(wm), abs(lm), 0.01) else ''
        print(f'  {ind:>25s} {wm:10.3f} {lm:10.3f} {diff:+8.3f}{marker}')


def find_filters(wins, losses):
    """Suggest filters that separate wins from losses."""
    print('\n=== FILTER SUGGESTIONS ===')
    all_trades = wins + losses

    # Test BB position thresholds for LONG
    long_wins = [t for t in wins if t['type'] == 'LONG']
    long_losses = [t for t in losses if t['type'] == 'LONG']
    if long_wins and long_losses:
        print('\n  LONG: BB position filter')
        for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
            w_pass = [t for t in long_wins if t['bb_position'] < threshold]
            l_pass = [t for t in long_losses if t['bb_position'] < threshold]
            if w_pass or l_pass:
                wr = len(w_pass) / max(len(w_pass) + len(l_pass), 1) * 100
                print(f'    bb_pos < {threshold:.2f}: {len(w_pass)}W/{len(l_pass)}L (WR={wr:.0f}%)')

        print('  LONG: RSI upper threshold')
        for rsi_max in [30, 33, 35, 38, 40, 42]:
            w_pass = [t for t in long_wins if t['rsi'] <= rsi_max]
            l_pass = [t for t in long_losses if t['rsi'] <= rsi_max]
            if w_pass or l_pass:
                wr = len(w_pass) / max(len(w_pass) + len(l_pass), 1) * 100
                print(f'    RSI <= {rsi_max}: {len(w_pass)}W/{len(l_pass)}L (WR={wr:.0f}%)')

        print('  LONG: EMA50 slope filter')
        for slope_max in [0.3, 0.5, 1.0, 1.5, 2.0]:
            w_pass = [t for t in long_wins if abs(t['ema50_slope']) < slope_max]
            l_pass = [t for t in long_losses if abs(t['ema50_slope']) < slope_max]
            if w_pass or l_pass:
                wr = len(w_pass) / max(len(w_pass) + len(l_pass), 1) * 100
                print(f'    |slope| < {slope_max}: {len(w_pass)}W/{len(l_pass)}L (WR={wr:.0f}%)')

    # For SHORT
    short_wins = [t for t in wins if t['type'] == 'SHORT']
    short_losses = [t for t in losses if t['type'] == 'SHORT']
    if short_wins:
        print('\n  SHORT: RSI thresholds (to increase frequency)')
        for rsi_min in [48, 50, 52, 55, 58]:
            for rsi_max in [75, 78, 80, 85, 90]:
                w_pass = [t for t in short_wins if rsi_min <= t['rsi'] <= rsi_max]
                l_pass = [t for t in short_losses if rsi_min <= t['rsi'] <= rsi_max]
                all_pass = w_pass + l_pass
                if len(all_pass) >= 2:
                    wr = len(w_pass) / len(all_pass) * 100
                    print(f'    RSI {rsi_min}-{rsi_max}: {len(w_pass)}W/{len(l_pass)}L total={len(all_pass)} (WR={wr:.0f}%)')

        print('  SHORT: BB position thresholds')
        for threshold in [0.45, 0.50, 0.55, 0.60, 0.65]:
            w_pass = [t for t in short_wins if t['bb_position'] > threshold]
            l_pass = [t for t in short_losses if t['bb_position'] > threshold]
            if w_pass or l_pass:
                wr = len(w_pass) / max(len(w_pass) + len(l_pass), 1) * 100
                print(f'    bb_pos > {threshold:.2f}: {len(w_pass)}W/{len(l_pass)}L (WR={wr:.0f}%)')


def main():
    print('Fetching 730d 1h...')
    df = fetch_historical_ohlcv('BTC/USDT', '1h', 730)
    df_ind = compute_indicators(df, '1h')
    df_ind = classify_regimes_v2(df_ind, hysteresis_bars=3)
    # Skip warm-up (same as backtest)
    df_ind = df_ind.iloc[300:].copy()
    ranging = (df_ind['regime_v2'] == 'RANGING').sum()
    print(f'{len(df_ind)} bars, RANGING: {ranging}')

    params_l = get_regime_params('RANGING', 0.5)
    params_s = get_regime_params('RANGING', 0.5)

    trades = sim_mr_with_details(df_ind, params_l, params_s)
    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]

    print(f'\nTotal: {len(trades)} trades, {len(wins)} wins, {len(losses)} losses')
    print(f'PnL: {sum(t["pnl_pct"] for t in trades):+.2f}%')

    longs = [t for t in trades if t['type'] == 'LONG']
    shorts = [t for t in trades if t['type'] == 'SHORT']
    print(f'LONG: {len(longs)} ({len([t for t in longs if t["pnl_pct"]>0])}W/{len([t for t in longs if t["pnl_pct"]<=0])}L) PnL={sum(t["pnl_pct"] for t in longs):+.2f}%')
    print(f'SHORT: {len(shorts)} ({len([t for t in shorts if t["pnl_pct"]>0])}W/{len([t for t in shorts if t["pnl_pct"]<=0])}L) PnL={sum(t["pnl_pct"] for t in shorts):+.2f}%')

    # Detailed comparison
    long_wins = [t for t in longs if t['pnl_pct'] > 0]
    long_losses = [t for t in longs if t['pnl_pct'] <= 0]
    short_wins = [t for t in shorts if t['pnl_pct'] > 0]
    short_losses = [t for t in shorts if t['pnl_pct'] <= 0]

    compare_groups(long_wins, long_losses, 'LONG: Win vs Loss')
    compare_groups(short_wins, short_losses, 'SHORT: Win vs Loss')
    compare_groups(wins, losses, 'ALL: Win vs Loss')

    # Filter suggestions
    find_filters(wins, losses)

    # Print each trade detail
    print('\n=== ALL TRADES ===')
    for t in trades:
        w = 'WIN' if t['pnl_pct'] > 0 else 'LOSS'
        print(f'  {str(t["entry_ts"])[:10]} {t["type"]:5s} RSI={t["rsi"]:5.1f} bb_pos={t["bb_position"]:.3f} slope={t["ema50_slope"]:+.3f} conf={t["confidence"]:.2f} RR={t["rr"]:.1f}:1 bars={t["bars_held"]:3d} {t["exit_reason"]:8s} PnL={t["pnl_pct"]:+.2f}% {w}')


if __name__ == '__main__':
    main()
