#!/usr/bin/env python3
"""Run grid search and save results to files."""
import sys, time, pickle, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optimize import Optimizer, make_p1_grid, make_p2_grid, print_top, _save_results

t0 = time.time()
log = open('/tmp/grid_log.txt', 'w')

def p(msg):
    print(msg, flush=True)
    log.write(msg + '\n')
    log.flush()

p(f'Started at {time.strftime("%H:%M:%S")}')

opt = Optimizer()
opt.load_data()

# Phase 1
p1 = make_p1_grid()
p(f'P1: {len(p1):,} combos')

p1_res = []
viable = 0
best_score = 0
best_str = ''
t1 = time.time()

for idx, params in enumerate(p1):
    r = opt.run_combo(idx, params, 'p1')
    p1_res.append(r)
    if r.score > 0:
        viable += 1
    if r.score > best_score:
        best_score = r.score
        best_str = (f'sc={r.score:.3f} WR={r.win_rate:.1f}% PF={r.profit_factor:.2f} '
                    f'PnL={r.total_pnl_pct:+.2f}% DD={r.max_drawdown_pct:.2f}% T={r.total_trades}')
    if (idx + 1) % 3000 == 0:
        el = time.time() - t1
        spd = (idx + 1) / max(el, 0.01)
        eta = (len(p1) - idx - 1) / max(spd, 0.01)
        p(f'  [{idx+1:,}/{len(p1):,}] {spd:.0f}/s viable={viable} ETA={eta:.0f}s | {best_str}')

p(f'P1 done: {time.time()-t1:.1f}s viable={viable}/{len(p1)}')

# Sort and show top 20
p1s = sorted(p1_res, key=lambda x: x.score, reverse=True)
for i, r in enumerate(p1s[:20]):
    p(f'{i+1:>2} sc={r.score:.3f} T={r.total_trades:>3} WR={r.win_rate:.1f}% PF={r.profit_factor:.2f} '
      f'PnL={r.total_pnl_pct:+.2f}% DD={r.max_drawdown_pct:.2f}% '
      f'RL={r.rsi_long_min:.0f}-{r.rsi_long_max:.0f} RS={r.rsi_short_min:.0f}-{r.rsi_short_max:.0f} '
      f'ADX={r.adx_min:.0f} V={r.volume_sma_ratio:.2f} F={r.fib_tolerance_pct:.3f} Tr={r.allow_transition} VC={r.volume_confirm}')

if not p1s or p1s[0].score == 0:
    p('No viable!')
    by_t = sorted(p1_res, key=lambda x: x.total_trades, reverse=True)
    for i, r in enumerate(by_t[:10]):
        p(f'{i+1:>2} T={r.total_trades:>3} WR={r.win_rate:.1f}% PF={r.profit_factor:.2f} PnL={r.total_pnl_pct:+.2f}% DD={r.max_drawdown_pct:.2f}%')
    sys.exit(0)

# Save P1 for safety
with open('/tmp/p1_results.pkl', 'wb') as f:
    pickle.dump(p1_res, f)

# Phase 2
n_top = min(20, len([r for r in p1s if r.score > 0]))
p1_top = [r for r in p1s if r.score > 0][:n_top]
p2 = make_p2_grid(p1_top, n_top)
p(f'P2: {len(p2):,} combos')

p2_res = []
t2 = time.time()
for idx, params in enumerate(p2):
    r = opt.run_combo(100000 + idx, params, 'p2')
    p2_res.append(r)
p(f'P2 done: {time.time()-t2:.1f}s')

all_res = p1_res + p2_res
all_s = sorted(all_res, key=lambda x: x.score, reverse=True)

# Show combined top 20
p('\nCOMBINED Top 20:')
for i, r in enumerate(all_s[:20]):
    p(f'{i+1:>2} sc={r.score:.3f} T={r.total_trades:>3} WR={r.win_rate:.1f}% PF={r.profit_factor:.2f} '
      f'PnL={r.total_pnl_pct:+.2f}% DD={r.max_drawdown_pct:.2f}% SL={r.sl_atr_mult} TP={r.tp_atr_mult}')

# Phase 3
p('P3: Advanced validation...')
top10 = all_s[:10]
p3_res = []
for i, r in enumerate(top10):
    p3r = opt.run_phase3_advanced(r)
    p3_res.append(p3r)
    p(f'  [{i+1}/10] T={p3r.total_trades} WR={p3r.win_rate:.1f}% PF={p3r.profit_factor:.2f} PnL={p3r.total_pnl_pct:+.2f}% DD={p3r.max_drawdown_pct:.2f}%')

winner = max(p3_res, key=lambda x: x.score)
tt = time.time() - t0

p(f'\nVENCEDOR:')
p(f'  Score:     {winner.score:.4f}')
p(f'  Trades:    {winner.total_trades} (L={winner.long_trades} S={winner.short_trades})')
p(f'  WR:        {winner.win_rate:.1f}%')
p(f'  PF:        {winner.profit_factor:.2f}')
p(f'  PnL:       {winner.total_pnl_pct:+.2f}%')
p(f'  DD:        {winner.max_drawdown_pct:.2f}%')
p(f'  Sharpe:    {winner.sharpe_ratio:.2f}')
p(f'  B&H:       {winner.buy_hold_pct:+.2f}%')
p(f'  Alpha:     {winner.total_pnl_pct - winner.buy_hold_pct:+.2f}pp')
p(f'  RSI_L:     {winner.rsi_long_min:.0f}-{winner.rsi_long_max:.0f}')
p(f'  RSI_S:     {winner.rsi_short_min:.0f}-{winner.rsi_short_max:.0f}')
p(f'  ADX:       {winner.adx_min:.0f}')
p(f'  Vol:       {winner.volume_sma_ratio:.2f}')
p(f'  Fib:       {winner.fib_tolerance_pct:.3f}')
p(f'  Trans:     {winner.allow_transition}')
p(f'  VC:        {winner.volume_confirm}')
p(f'  SL/TP:     {winner.sl_atr_mult}/{winner.tp_atr_mult} (R:R {winner.tp_atr_mult/winner.sl_atr_mult:.1f}:1)')
p(f'\nTotal: {len(p1)+len(p2)+len(p3_res):,} combos in {tt:.0f}s ({tt/60:.1f}min)')

_save_results(p1_res + p2_res + p3_res, opt, t0, len(p1), len(p2), winner)

log.close()
p(f'Log: /tmp/grid_log.txt')
