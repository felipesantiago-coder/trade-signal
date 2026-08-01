#!/usr/bin/env python3
"""Phase 2 (SL/TP) + Phase 3 (advanced) on top P1 results."""
import sys, os, time, pickle, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optimize import Optimizer, make_p2_grid, _save_results, GridResult
import numpy as np

log = open('/tmp/p2p3_log.txt', 'w')
def p(m): print(m, flush=True); log.write(m+'\n'); log.flush()

t0 = time.time()

# Load P1
with open('/tmp/p1_results.pkl', 'rb') as f:
    p1_res = pickle.load(f)

p1s = sorted(p1_res, key=lambda x: x.score, reverse=True)
viable = [r for r in p1s if r.score > 0]
p(f'P1 loaded: {len(p1_res):,} total, {len(viable)} viable')

# Deduplicate by unique param sets (same metrics = same effective combo)
seen = set()
unique_top = []
for r in viable:
    key = (r.rsi_long_min, r.rsi_long_max, r.rsi_short_min, r.rsi_short_max,
           r.adx_min, r.volume_sma_ratio, r.fib_tolerance_pct, r.allow_transition, r.volume_confirm)
    if key not in seen:
        seen.add(key)
        unique_top.append(r)

p(f'Unique viable param sets: {len(unique_top)}')

# Phase 2
n_top = min(20, len(unique_top))
p1_top = unique_top[:n_top]
p2 = make_p2_grid(p1_top, n_top)
p(f'P2: {len(p2):,} SL/TP combos for top {n_top} filter configs')

opt = Optimizer()
opt.load_data()

p2_res = []
t2 = time.time()
for idx, params in enumerate(p2):
    r = opt.run_combo(100000 + idx, params, 'p2')
    p2_res.append(r)
    if (idx + 1) % 500 == 0:
        el = time.time() - t2
        p(f'  [{idx+1:,}/{len(p2):,}] {el:.1f}s')

p(f'P2 done: {time.time()-t2:.1f}s')

# Combine
all_res = p1_res + p2_res
all_s = sorted(all_res, key=lambda x: x.score, reverse=True)

p(f'\nCOMBINED Top 30:')
for i, r in enumerate(all_s[:30]):
    p(f'{i+1:>2} sc={r.score:.3f} T={r.total_trades:>3} L/S={r.long_trades}/{r.short_trades} WR={r.win_rate:.1f}% PF={r.profit_factor:.2f} PnL={r.total_pnl_pct:>+8.2f}% DD={r.max_drawdown_pct:.2f}% Sh={r.sharpe_ratio:.2f} SL={r.sl_atr_mult} TP={r.tp_atr_mult} RL={r.rsi_long_min:.0f}-{r.rsi_long_max:.0f} RS={r.rsi_short_min:.0f}-{r.rsi_short_max:.0f} ADX={r.adx_min:.0f} V={r.volume_sma_ratio:.2f}')

# Phase 3: Advanced validation on top 10
top10 = all_s[:10]
p(f'\nP3: Advanced validation (trailing/BE/partial)...')
p3_res = []
for i, r in enumerate(top10):
    p3r = opt.run_phase3_advanced(r)
    p3_res.append(p3r)
    p(f'  [{i+1}/10] T={p3r.total_trades} L/S={p3r.long_trades}/{p3r.short_trades} WR={p3r.win_rate:.1f}% PF={p3r.profit_factor:.2f} PnL={p3r.total_pnl_pct:+.2f}% DD={p3r.max_drawdown_pct:.2f}% Sh={p3r.sharpe_ratio:.2f}')

p(f'\nP3 Top 10:')
for i, r in enumerate(sorted(p3_res, key=lambda x: x.score, reverse=True)):
    p(f'{i+1:>2} sc={r.score:.3f} T={r.total_trades:>3} WR={r.win_rate:.1f}% PF={r.profit_factor:.2f} PnL={r.total_pnl_pct:>+8.2f}% DD={r.max_drawdown_pct:.2f}% SL={r.sl_atr_mult} TP={r.tp_atr_mult}')

winner = max(p3_res, key=lambda x: x.score)
tt = time.time() - t0

p(f'\n#{"#"*120}')
p(f'#  VENCEDOR — CTEV v4.4 OPTIMIZED')
p(f'#{"#"*120}')
p(f'#  Score:         {winner.score:.4f}')
p(f'#  Trades:        {winner.total_trades} (L={winner.long_trades} S={winner.short_trades})')
p(f'#  Win Rate:      {winner.win_rate:.1f}%')
p(f'#  Profit Factor: {winner.profit_factor:.2f}')
p(f'#  PnL:           {winner.total_pnl_pct:+.2f}%')
p(f'#  Max DD:        {winner.max_drawdown_pct:.2f}%')
p(f'#  Sharpe:        {winner.sharpe_ratio:.2f}')
p(f'#  Buy & Hold:    {winner.buy_hold_pct:+.2f}%')
p(f'#  Alpha:         {winner.total_pnl_pct - winner.buy_hold_pct:+.2f}pp')
p(f'#')
p(f'#  PARAMETROS:')
p(f'#    RSI LONG:       {winner.rsi_long_min:.0f} - {winner.rsi_long_max:.0f}')
p(f'#    RSI SHORT:      {winner.rsi_short_min:.0f} - {winner.rsi_short_max:.0f}')
p(f'#    ADX_MIN:        {winner.adx_min:.0f}')
p(f'#    VOLUME_RATIO:   {winner.volume_sma_ratio:.2f}')
p(f'#    FIB_TOLERANCE:  {winner.fib_tolerance_pct*100:.1f}%')
p(f'#    ALLOW_TRANS:    {winner.allow_transition}')
p(f'#    SL/TP:          {winner.sl_atr_mult:.2f}x / {winner.tp_atr_mult:.2f}x (R:R {winner.tp_atr_mult/winner.sl_atr_mult:.1f}:1)')
p(f'#    VOLUME_CONFIRM: {winner.volume_confirm}')
p(f'#')
p(f'#  Total: {len(p1_res)+len(p2_res)+len(p3_res):,} combos em {tt:.0f}s ({tt/60:.1f}min)')

# Save
_save_results(p1_res + p2_res + p3_res, opt, t0, len(p1_res), len(p2_res), winner)

# Also save winner as standalone
wdata = {
    'version': 'v4.4', 'score': winner.score,
    'params': {
        'RSI_LONG_MIN': winner.rsi_long_min, 'RSI_LONG_MAX': winner.rsi_long_max,
        'RSI_SHORT_MIN': winner.rsi_short_min, 'RSI_SHORT_MAX': winner.rsi_short_max,
        'ADX_MIN': winner.adx_min, 'VOLUME_SMA_RATIO': winner.volume_sma_ratio,
        'FIB_TOLERANCE_PCT': winner.fib_tolerance_pct, 'ALLOW_TRANSITION': winner.allow_transition,
        'SL_ATR_MULT': winner.sl_atr_mult, 'TP_ATR_MULT': winner.tp_atr_mult,
        'EMA20_PROXIMITY_PCT': winner.ema20_prox_pct, 'EMA50_PROXIMITY_PCT': winner.ema50_prox_pct,
        'VOLUME_CONFIRM': winner.volume_confirm,
    },
    'metrics_p3': {
        'total_trades': winner.total_trades, 'win_rate': winner.win_rate,
        'profit_factor': winner.profit_factor, 'total_pnl_pct': winner.total_pnl_pct,
        'max_drawdown_pct': winner.max_drawdown_pct, 'sharpe_ratio': winner.sharpe_ratio,
    },
}
with open('/tmp/winner_v44.json', 'w') as f:
    json.dump(wdata, f, indent=2)

log.close()
p('Done! Log: /tmp/p2p3_log.txt')
