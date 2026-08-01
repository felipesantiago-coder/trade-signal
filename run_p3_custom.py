#!/usr/bin/env python3
"""Phase 3 custom: test specific SL/TP combos with advanced sim."""
import sys, os, time, pickle, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optimize import Optimizer, GridResult, _save_results
import strategy
from backtest import simulate_trades_advanced, calculate_metrics

log = open('/tmp/p3c_log.txt', 'w')
def p(m): print(m, flush=True); log.write(m+'\n'); log.flush()

t0 = time.time()
opt = Optimizer()
opt.load_data()

# Best filter config from P1: RSI 28-48/55-75, ADX=30, VC=False
# Test multiple SL/TP combos with advanced sim
sl_tp_combos = [
    (1.0, 2.0), (1.0, 2.5), (1.0, 3.0),
    (1.25, 2.5), (1.25, 3.0), (1.25, 3.5), (1.25, 4.0),
    (1.5, 3.0), (1.5, 3.5), (1.5, 4.0), (1.5, 5.0),
    (1.75, 3.5), (1.75, 4.0), (1.75, 4.5), (1.75, 5.0),
    (2.0, 4.0), (2.0, 4.5), (2.0, 5.0), (2.0, 6.0),
    (2.5, 5.0), (2.5, 6.0), (2.5, 7.0),
    (1.5, 2.5),  # baseline from P2
]

p(f'Testing {len(sl_tp_combos)} SL/TP combos with advanced simulation...')

results = []
for sl, tp in sl_tp_combos:
    strategy.ADX_MIN = 30.0
    strategy.RSI_LONG_MIN = 28.0; strategy.RSI_LONG_MAX = 48.0
    strategy.RSI_SHORT_MIN = 55.0; strategy.RSI_SHORT_MAX = 75.0
    strategy.VOLUME_SMA_RATIO = 0.30; strategy.VOLUME_CONFIRM = False
    strategy.FIB_TOLERANCE_PCT = 0.025; strategy.ALLOW_TRANSITION = False
    strategy.SL_ATR_MULT = sl; strategy.TP_ATR_MULT = tp
    strategy.ATR_PCT_MIN = 0.10; strategy.ATR_PCT_MAX = 0.90
    strategy.EMA50_SLOPE_MIN = 0.0; strategy.EMA20_PROXIMITY_PCT = 0.0; strategy.EMA50_PROXIMITY_PCT = 0.0

    trades, _, _ = simulate_trades_advanced(opt.df_clean, 0.10, 0.90)
    m = calculate_metrics(trades, opt.df_clean)

    # Also run basic for comparison
    r_basic = opt.run_combo(0, {
        'rsi_long_min': 28, 'rsi_long_max': 48,
        'rsi_short_min': 55, 'rsi_short_max': 75,
        'adx_min': 30, 'volume_sma_ratio': 0.30,
        'fib_tolerance_pct': 0.025, 'allow_transition': False,
        'sl_atr_mult': sl, 'tp_atr_mult': tp, 'volume_confirm': False,
    }, 'p1')

    rr = tp / sl
    p(f'SL={sl:.2f} TP={tp:.2f} R:R={rr:.1f}:1 | ')
    p(f'  ADV: T={m.total_trades} L/S={m.long_trades}/{m.short_trades} WR={m.win_rate:.1f}% PF={m.profit_factor:.2f} PnL={m.total_pnl_pct:+.2f}% DD={m.max_drawdown_pct:.2f}% Sh={m.sharpe_ratio:.2f} BE={sum(1 for t in trades if t.be_triggered)} Tr={sum(1 for t in trades if t.trailing_activated)}')
    p(f'  BSC: T={r_basic.total_trades} L/S={r_basic.long_trades}/{r_basic.short_trades} WR={r_basic.win_rate:.1f}% PF={r_basic.profit_factor:.2f} PnL={r_basic.total_pnl_pct:+.2f}% DD={r_basic.max_drawdown_pct:.2f}% Sh={r_basic.sharpe_ratio:.2f}')

    r = GridResult(
        combo_id=300000, rsi_long_min=28, rsi_long_max=48,
        rsi_short_min=55, rsi_short_max=75,
        adx_min=30, volume_sma_ratio=0.30,
        fib_tolerance_pct=0.025, allow_transition=False,
        sl_atr_mult=sl, tp_atr_mult=tp, volume_confirm=False,
        total_trades=m.total_trades, long_trades=m.long_trades, short_trades=m.short_trades,
        wins=m.wins, losses=m.losses, win_rate=round(m.win_rate, 2),
        profit_factor=round(m.profit_factor, 4), total_pnl_pct=round(m.total_pnl_pct, 4),
        max_drawdown_pct=round(m.max_drawdown_pct, 4), sharpe_ratio=round(m.sharpe_ratio, 4),
        avg_bars_held=round(m.avg_bars_held, 1), avg_win_pct=round(m.avg_win_pct, 4),
        avg_loss_pct=round(m.avg_loss_pct, 4), best_trade_pct=round(m.best_trade_pct, 4),
        worst_trade_pct=round(m.worst_trade_pct, 4), buy_hold_pct=round(opt.buy_hold, 4), phase='p3',
    )
    r.score = Optimizer._score(r)
    results.append(r)

# Also test with VC=True (Volume Confirm on)
p(f'\nNow testing with VC=True...')
for sl, tp in [(1.5, 3.0), (1.5, 2.5), (1.75, 4.0), (2.0, 5.0)]:
    strategy.ADX_MIN = 30.0
    strategy.RSI_LONG_MIN = 28.0; strategy.RSI_LONG_MAX = 48.0
    strategy.RSI_SHORT_MIN = 55.0; strategy.RSI_SHORT_MAX = 75.0
    strategy.VOLUME_SMA_RATIO = 0.50; strategy.VOLUME_CONFIRM = True
    strategy.FIB_TOLERANCE_PCT = 0.025; strategy.ALLOW_TRANSITION = False
    strategy.SL_ATR_MULT = sl; strategy.TP_ATR_MULT = tp
    strategy.ATR_PCT_MIN = 0.10; strategy.ATR_PCT_MAX = 0.90
    strategy.EMA50_SLOPE_MIN = 0.0; strategy.EMA20_PROXIMITY_PCT = 0.0; strategy.EMA50_PROXIMITY_PCT = 0.0

    trades, _, _ = simulate_trades_advanced(opt.df_clean, 0.10, 0.90)
    m = calculate_metrics(trades, opt.df_clean)
    p(f'SL={sl:.2f} TP={tp:.2f} R:R={tp/sl:.1f}:1 VC=True | ADV: T={m.total_trades} WR={m.win_rate:.1f}% PF={m.profit_factor:.2f} PnL={m.total_pnl_pct:+.2f}% DD={m.max_drawdown_pct:.2f}% Sh={m.sharpe_ratio:.2f}')

    r = GridResult(
        combo_id=400000, rsi_long_min=28, rsi_long_max=48,
        rsi_short_min=55, rsi_short_max=75,
        adx_min=30, volume_sma_ratio=0.50,
        fib_tolerance_pct=0.025, allow_transition=False,
        sl_atr_mult=sl, tp_atr_mult=tp, volume_confirm=True,
        total_trades=m.total_trades, long_trades=m.long_trades, short_trades=m.short_trades,
        wins=m.wins, losses=m.losses, win_rate=round(m.win_rate, 2),
        profit_factor=round(m.profit_factor, 4), total_pnl_pct=round(m.total_pnl_pct, 4),
        max_drawdown_pct=round(m.max_drawdown_pct, 4), sharpe_ratio=round(m.sharpe_ratio, 4),
        avg_bars_held=round(m.avg_bars_held, 1), avg_win_pct=round(m.avg_win_pct, 4),
        avg_loss_pct=round(m.avg_loss_pct, 4), best_trade_pct=round(m.best_trade_pct, 4),
        worst_trade_pct=round(m.worst_trade_pct, 4), buy_hold_pct=round(opt.buy_hold, 4), phase='p3vc',
    )
    r.score = Optimizer._score(r)
    results.append(r)

# Rank all P3 results
p(f'\nALL P3 RESULTS RANKED:')
for i, r in enumerate(sorted(results, key=lambda x: x.score, reverse=True)):
    vc = 'VC=T' if r.volume_confirm else 'VC=F'
    p(f'{i+1:>2} sc={r.score:.3f} T={r.total_trades:>3} WR={r.win_rate:.1f}% PF={r.profit_factor:.2f} PnL={r.total_pnl_pct:>+8.2f}% DD={r.max_drawdown_pct:.2f}% SL={r.sl_atr_mult} TP={r.tp_atr_mult} R:R={r.tp_atr_mult/r.sl_atr_mult:.1f} {vc}')

winner = max(results, key=lambda x: x.score)
tt = time.time()-t0
p(f'\nTotal time: {tt:.1f}s')

# Save winner
wdata = {
    'version': 'v4.4', 'score': winner.score,
    'params': {
        'RSI_LONG_MIN': winner.rsi_long_min, 'RSI_LONG_MAX': winner.rsi_long_max,
        'RSI_SHORT_MIN': winner.rsi_short_min, 'RSI_SHORT_MAX': winner.rsi_short_max,
        'ADX_MIN': winner.adx_min, 'VOLUME_SMA_RATIO': winner.volume_sma_ratio,
        'FIB_TOLERANCE_PCT': winner.fib_tolerance_pct, 'ALLOW_TRANSITION': winner.allow_transition,
        'SL_ATR_MULT': winner.sl_atr_mult, 'TP_ATR_MULT': winner.tp_atr_mult,
        'EMA20_PROXIMITY_PCT': 0.0, 'EMA50_PROXIMITY_PCT': 0.0,
        'VOLUME_CONFIRM': winner.volume_confirm,
    },
    'metrics': {
        'total_trades': winner.total_trades, 'win_rate': winner.win_rate,
        'profit_factor': winner.profit_factor, 'total_pnl_pct': winner.total_pnl_pct,
        'max_drawdown_pct': winner.max_drawdown_pct, 'sharpe_ratio': winner.sharpe_ratio,
        'buy_hold_pct': winner.buy_hold_pct,
    },
    'grid_stats': {'total_tested': 15110 + len(sl_tp_combos) + 4, 'total_time_sec': round(tt, 1)},
}
with open('/tmp/winner_v44.json', 'w') as f:
    json.dump(wdata, f, indent=2)

log.close()
p('Done!')
