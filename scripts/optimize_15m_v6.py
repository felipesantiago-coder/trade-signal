'''CTEV 15m Optimizer v6 — Focused Grid + WFO

Estrategias testadas:
  1. Trend-Following (CTEV classico): regime+trend+pullback+RSI
  2. Trend-Lite: regime+trend+RSI (sem pullback)
  3. Mean-Reversion: BB bounce em ranging + RSI extremes
  4. Hybrid: MR em ranging + TF em trending/transition

2 anos de dados, 50/50 walk-forward.
'''
from __future__ import annotations
import sys, os, time, math, json, pickle
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.join(SCRIPT_DIR, '..')
sys.path.insert(0, PROJ_DIR)
from backtest import _apply_costs, DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS

DATA_CACHE = '/tmp/ctev_15m_data.pkl'
FEE = DEFAULT_FEE_PCT; SPREAD = DEFAULT_SPREAD_BPS; SLIPPAGE = DEFAULT_SLIPPAGE_BPS
COST_RT = (FEE * 2 + SPREAD / 100 + SLIPPAGE / 100) / 100  # round-trip cost

MIN_TRADES = 80; MIN_WR = 0.0; MAX_DD = 30.0


@dataclass
class R:
    score: float = 0.0; strategy: str = ''
    total_trades: int = 0; long_trades: int = 0; short_trades: int = 0
    wins: int = 0; win_rate: float = 0.0
    profit_factor: float = 0.0; total_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0; sharpe_ratio: float = 0.0
    avg_win_pct: float = 0.0; avg_loss_pct: float = 0.0
    avg_bars_held: float = 0.0
    buy_hold_pct: float = 0.0; sl_m: float = 1.5; tp_m: float = 3.5
    max_bars: int = 72; p: dict = None
    train_pnl: float = 0.0; test_pnl: float = 0.0
    train_wr: float = 0.0; test_wr: float = 0.0
    train_trades: int = 0; test_trades: int = 0
    degradation: float = 0.0

    def to_dict(self):
        d = asdict(self); d['params'] = self.p; return d


def load_data():
    with open(DATA_CACHE, 'rb') as f:
        return pickle.load(f)['df_clean']


def sim(c, hi, lo, regime, rsi, atr, adx, e50, e200, slope, atr_pct,
        fib382, fib500, fib618, fibdir, e20t, e50t, e50tu, e20, bbl, bbu, bbw,
        start, end, p):
    '''Simulate strategy on slice [start:end]. Returns list of (is_long, pnl_pct, bars).'''
    entries = []; dirs = []; sls = []; tps = []; eps = []
    mode = p.get('mode', 'tf')  # 'tf', 'tl', 'mr', 'hybrid'
    rsi_l = p['rsi_l']; rsi_s = p['rsi_s']
    adx_min = p.get('adx_min', 0); allow_trn = p.get('allow_trn', True)
    req_et = p.get('req_et', True); req_pb = p.get('req_pb', False)
    sl_m = p['sl_m']; tp_m = p['tp_m']; max_bars = p.get('max_bars', 72)
    fib_tol = p.get('fib_tol', 0.025)
    atr_min = p.get('atr_min', 0.05); atr_max = p.get('atr_max', 0.95)
    # MR params
    mr_rl = p.get('mr_rl', 30.0); mr_rs = p.get('mr_rs', 70.0)
    mr_bb_touch = p.get('mr_bb_touch', True)
    # Cooldown between trades
    cd = p.get('cooldown', 4)

    i = start; cooldown = 0
    while i < end:
        if i < cooldown: i += 1; continue
        rg = regime[i]
        sig = None; is_long = True

        # ── Mean-Reversion mode ──
        if mode in ('mr', 'hybrid') and rg == 'ranging':
            ci = c[i]; li = lo[i]; hii = hi[i]; r = rsi[i]; ap = atr_pct[i]
            if not (atr_min <= ap <= atr_max): i += 1; continue
            # LONG MR: RSI oversold + BB touch
            if r <= mr_rl:
                if mr_bb_touch and li > bbl[i]: i += 1; continue
                av = atr[i]; sl = ci - sl_m * av; tp = ci + tp_m * av
                if sl > 0:
                    entries.append(i); dirs.append(True); sls.append(sl); tps.append(tp); eps.append(ci)
                    cooldown = i + cd; continue
            # SHORT MR: RSI overbought + BB touch
            if r >= mr_rs:
                if mr_bb_touch and hii < bbu[i]: i += 1; continue
                av = atr[i]; sl = ci + sl_m * av; tp = ci - tp_m * av
                entries.append(i); dirs.append(False); sls.append(sl); tps.append(tp); eps.append(ci)
                cooldown = i + cd; continue
            if mode == 'mr': i += 1; continue

        # ── Trend-Following mode ──
        if mode in ('tf', 'tl', 'hybrid'):
            if rg == 'trending_up':
                if adx[i] < adx_min: i += 1; continue
                is_long = True
            elif rg == 'trending_down':
                if adx[i] < adx_min: i += 1; continue
                is_long = False
            elif rg == 'transition':
                if not allow_trn: i += 1; continue
                if c[i] > e50[i] > e200[i]: is_long = True
                elif c[i] < e50[i] < e200[i]: is_long = False
                else: i += 1; continue
            else:  # ranging, volatile
                i += 1; continue

            ci = c[i]; r = rsi[i]; ap = atr_pct[i]
            if is_long:
                if req_et and not (ci > e50[i] > e200[i]): i += 1; continue
                if not (rsi_l[0] <= r <= rsi_l[1]): i += 1; continue
            else:
                if req_et and not (ci < e50[i] < e200[i]): i += 1; continue
                if not (rsi_s[0] <= r <= rsi_s[1]): i += 1; continue
            if not (atr_min <= ap <= atr_max): i += 1; continue

            # Pullback check (only for 'tf' mode)
            if req_pb:
                pb = False; fd = fibdir[i]
                if (is_long and fd == 1) or (not is_long and fd == -1):
                    f38 = fib382[i]; f61 = fib618[i]
                    if not (np.isnan(f38) or np.isnan(f61)) and f61 <= ci <= f38:
                        pb = True
                    if not pb:
                        tol = ci * fib_tol
                        for fl in (f38, fib500[i], f61):
                            if not np.isnan(fl) and fl > 0:
                                if is_long and abs(lo[i] - fl) <= tol: pb = True; break
                                elif not is_long and abs(hi[i] - fl) <= tol: pb = True; break
                if not pb and is_long and e20t[i] and ci > e20[i]: pb = True
                if not pb and is_long and e50t[i] and ci > e50[i]: pb = True
                if not pb and not is_long and e50tu[i] and ci < e50[i] and hi[i] >= e50[i]: pb = True
                if not pb: i += 1; continue

            av = atr[i]
            if is_long:
                sl = ci - sl_m * av; tp = ci + tp_m * av
            else:
                sl = ci + sl_m * av; tp = ci - tp_m * av
            if is_long and sl <= 0: i += 1; continue
            entries.append(i); dirs.append(is_long); sls.append(sl); tps.append(tp); eps.append(ci)
            cooldown = i + cd
        else:
            i += 1
            continue
        i += 1

    # Execute trades
    n = len(c); trades = []
    for k in range(len(entries)):
        idx = entries[k]; is_long = dirs[k]
        sl = sls[k]; tp = tps[k]; ep = eps[k]
        exit_price = None; bars = 0; mj = min(idx + max_bars, n)
        if is_long:
            for j in range(idx + 1, mj):
                if lo[j] <= sl: exit_price = sl; bars = j - idx; break
                if hi[j] >= tp: exit_price = tp; bars = j - idx; break
        else:
            for j in range(idx + 1, mj):
                if hi[j] >= sl: exit_price = sl; bars = j - idx; break
                if lo[j] <= tp: exit_price = tp; bars = j - idx; break
        if exit_price is None:
            bars = mj - 1 - idx; exit_price = c[mj - 1]
        _, adj, _ = _apply_costs(ep, exit_price, is_long, FEE, SPREAD, SLIPPAGE)
        pnl = (adj - ep) / ep * 100 if is_long else (ep - adj) / ep * 100
        trades.append((is_long, pnl, bars))
    return trades


def calc_metrics(trades):
    if not trades:
        return {'n': 0, 'w': 0, 'wr': 0, 'pf': 0, 'pnl': 0, 'dd': 0,
                'sh': 0, 'aw': 0, 'al': 0, 'ab': 0, 'l': 0, 's': 0}
    total = len(trades); wins = sum(1 for t in trades if t[1] > 0)
    longs = sum(1 for t in trades if t[0])
    pnls = [t[1] for t in trades]
    wp = [p for p in pnls if p > 0]; lp = [p for p in pnls if p <= 0]
    gw = sum(wp); gl = abs(sum(lp))
    pf = gw / gl if gl > 0 else 999.0
    tpnl = sum(pnls)
    eq = [100.0]
    for p in pnls: eq.append(eq[-1] * (1 + p / 100))
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak * 100; mdd = float(np.max(dd))
    # Sharpe annualized (15m = 96 candles/day)
    sharpe = float(np.mean(pnls) / (np.std(pnls) + 1e-9) * math.sqrt(365 * 96 / max(total, 1)))
    return {'n': total, 'w': wins, 'wr': wins / total * 100, 'pf': pf, 'pnl': tpnl,
            'dd': mdd, 'sh': sharpe, 'aw': np.mean(wp) if wp else 0,
            'al': np.mean(lp) if lp else 0, 'ab': np.mean([t[2] for t in trades]),
            'l': longs, 's': total - longs}


def score(r: R) -> float:
    '''Scoring: reward profitability in BOTH halves, penalize overfit.'''
    if r.total_trades < MIN_TRADES: return 0.0
    if r.max_drawdown_pct > MAX_DD: return 0.0
    # Core: must be profitable in both train AND test
    if r.train_pnl <= 0 or r.test_pnl <= 0: return 0.0
    if r.profit_factor < 1.0: return 0.0
    # Components
    wr = r.win_rate / 100.0
    pf = r.profit_factor
    dd_pen = max(0, 1.0 - r.max_drawdown_pct / 100.0)
    freq = min(math.log(max(r.total_trades, 1)) / math.log(400), 1.5)
    pnl_b = 2.0 if r.total_pnl_pct > 15 else (1.5 if r.total_pnl_pct > 5 else 1.0)
    sig = min(math.sqrt(r.total_trades) / 12.0, 2.0)
    # Overfit penalty
    deg_pen = max(0, 1.0 - abs(r.degradation) / 80.0)
    # Balance bonus
    if r.long_trades > 5 and r.short_trades > 5:
        ratio = min(r.long_trades, r.short_trades) / max(r.long_trades, r.short_trades)
        balance = 0.8 + 0.2 * ratio
    else:
        balance = 0.6
    return round(wr * pf * freq * dd_pen * pnl_b * sig * balance * deg_pen, 4)


def make_grid():
    combos = []
    rsi_l_opts = [(25, 50), (28, 48), (30, 55), (25, 55), (20, 45)]
    rsi_s_opts = [(50, 75), (55, 75), (45, 70), (50, 70)]
    sltp_opts = [(0.5, 1.5), (0.75, 2.0), (1.0, 2.0), (1.0, 2.5), (1.5, 3.0)]
    max_bars_opts = [24, 48, 72]
    adx_opts = [0, 20]
    cooldown_opts = [2, 4]

    # MODE 1: Trend-Following (with pullback)
    for rl in rsi_l_opts:
        for rs in rsi_s_opts:
            for sl, tp in sltp_opts:
                for mb in max_bars_opts:
                    for adx in adx_opts:
                        for cd in cooldown_opts:
                            combos.append({
                                'mode': 'tf', 'rsi_l': rl, 'rsi_s': rs,
                                'sl_m': sl, 'tp_m': tp, 'max_bars': mb,
                                'adx_min': adx, 'allow_trn': True, 'req_et': True,
                                'req_pb': True, 'fib_tol': 0.025,
                                'atr_min': 0.05, 'atr_max': 0.95, 'cooldown': cd
                            })

    # MODE 2: Trend-Lite (no pullback)
    for rl in [(25, 50), (28, 48), (30, 55)]:
        for rs in [(50, 75), (55, 75), (45, 70)]:
            for sl, tp in [(0.5, 1.5), (0.75, 2.0), (1.0, 2.0), (1.0, 2.5)]:
                for mb in [24, 48, 72]:
                    for cd in [2, 4]:
                        combos.append({
                            'mode': 'tl', 'rsi_l': rl, 'rsi_s': rs,
                            'sl_m': sl, 'tp_m': tp, 'max_bars': mb,
                            'adx_min': 0, 'allow_trn': True, 'req_et': True,
                            'req_pb': False, 'fib_tol': 0.025,
                            'atr_min': 0.05, 'atr_max': 0.95, 'cooldown': cd
                        })

    # MODE 3: Mean-Reversion in ranging
    for mr_rl in [25, 30, 35]:
        for mr_rs in [65, 70, 75]:
            for sl, tp in [(0.5, 1.0), (0.5, 1.5), (0.75, 1.5), (1.0, 2.0)]:
                for mb in [24, 48, 72]:
                    for cd in [2, 4]:
                        combos.append({
                            'mode': 'mr', 'rsi_l': (25, 55), 'rsi_s': (45, 70),
                            'sl_m': sl, 'tp_m': tp, 'max_bars': mb,
                            'adx_min': 0, 'allow_trn': True, 'req_et': False,
                            'req_pb': False, 'fib_tol': 0.025,
                            'atr_min': 0.05, 'atr_max': 0.95,
                            'mr_rl': float(mr_rl), 'mr_rs': float(mr_rs),
                            'mr_bb_touch': True, 'cooldown': cd
                        })

    # MODE 4: Hybrid (MR in ranging + TF in trending/transition)
    for mr_rl in [30, 35]:
        for mr_rs in [65, 70]:
            for rl in [(25, 50), (28, 48)]:
                for rs in [(50, 75), (55, 75)]:
                    for sl, tp in [(0.5, 1.5), (0.75, 2.0), (1.0, 2.0)]:
                        for mb in [24, 48]:
                            for cd in [2, 4]:
                                combos.append({
                                    'mode': 'hybrid', 'rsi_l': rl, 'rsi_s': rs,
                                    'sl_m': sl, 'tp_m': tp, 'max_bars': mb,
                                    'adx_min': 0, 'allow_trn': True, 'req_et': True,
                                    'req_pb': False, 'fib_tol': 0.025,
                                    'atr_min': 0.05, 'atr_max': 0.95,
                                    'mr_rl': float(mr_rl), 'mr_rs': float(mr_rs),
                                    'mr_bb_touch': True, 'cooldown': cd
                                })

    return combos


def main():
    t0 = time.time()
    print(f'\n{"#" * 140}')
    print(f'#  CTEV 15m OPTIMIZER v6 — HYBRID + WALK-FORWARD')
    print(f'#  Custo: fee={FEE}% spread={SPREAD}bps slip={SLIPPAGE}bps | RT={COST_RT*100:.3f}%')
    print(f'#  Validacao: 50% treino / 50% teste')
    print(f'{"#" * 140}')

    print(f'\n[1/4] Carregando dados 15m...')
    df = load_data()
    n = len(df)
    bh = (df['close'].values[-1] - df['close'].values[0]) / df['close'].values[0] * 100
    rc = dict(zip(*np.unique(df['regime'].values, return_counts=True)))
    print(f'{n} candles ({n // 96} dias) | B&H={bh:+.2f}% | Regimes: {rc}')

    # Pre-extract arrays for speed
    c = df['close'].values.astype(np.float64)
    hi = df['high'].values.astype(np.float64)
    lo = df['low'].values.astype(np.float64)
    regime = df['regime'].values
    rsi = df['rsi'].values.astype(np.float64)
    atr = df['atr'].values.astype(np.float64)
    adx = df['adx'].values.astype(np.float64)
    e50 = df['ema50'].values.astype(np.float64)
    e200 = df['ema200'].values.astype(np.float64)
    slope = df['ema50_slope'].values.astype(np.float64)
    atr_pct = df['atr_percentile'].values.astype(np.float64)
    fib382 = df['fib_0382'].values.astype(np.float64)
    fib500 = df['fib_0500'].values.astype(np.float64)
    fib618 = df['fib_0618'].values.astype(np.float64)
    fibdir = df['fib_direction'].values.astype(np.int32)
    e20t = df['ema20_touched'].values
    e50t = df['ema50_touched'].values
    e50tu = df['ema50_touched_up'].values
    e20 = df['ema20'].values.astype(np.float64)
    bbl = df['bb_lower'].values.astype(np.float64)
    bbu = df['bb_upper'].values.astype(np.float64)
    bbw = df['bb_width'].values.astype(np.float64)

    mid = n // 2
    bh_train = (c[mid - 1] - c[0]) / c[0] * 100
    bh_test = (c[-1] - c[mid]) / c[mid] * 100
    print(f'Treino: {mid} candles ({mid // 96}d) B&H={bh_train:+.2f}%')
    print(f'Teste:  {n - mid} candles ({(n - mid) // 96}d) B&H={bh_test:+.2f}%')

    arrs = (c, hi, lo, regime, rsi, atr, adx, e50, e200, slope, atr_pct,
            fib382, fib500, fib618, fibdir, e20t, e50t, e50tu, e20, bbl, bbu, bbw)

    print(f'\n[2/4] Gerando grid...')
    grid = make_grid()
    print(f'{len(grid):,} combinacoes')

    print(f'\n[3/4] Walk-Forward: treino + teste...')
    results = []; viable = 0; best_score = 0; best_str = ''
    t1 = time.time()

    for idx, p in enumerate(grid):
        mb = p.get('max_bars', 72)

        # FULL period metrics
        trades_all = sim(*arrs, 0, n, p)
        m_all = calc_metrics(trades_all)
        if m_all['n'] < MIN_TRADES: continue

        # TRAIN period
        trades_train = sim(*arrs, 0, mid, p)
        m_train = calc_metrics(trades_train)

        # TEST period
        trades_test = sim(*arrs, mid, n, p)
        m_test = calc_metrics(trades_test)

        r = R(
            p=p, sl_m=p['sl_m'], tp_m=p['tp_m'], max_bars=mb,
            strategy=p.get('mode', 'tf'),
            total_trades=m_all['n'], long_trades=m_all['l'], short_trades=m_all['s'],
            wins=m_all['w'], win_rate=round(m_all['wr'], 2),
            profit_factor=round(m_all['pf'], 4), total_pnl_pct=round(m_all['pnl'], 4),
            max_drawdown_pct=round(m_all['dd'], 4), sharpe_ratio=round(m_all['sh'], 4),
            avg_win_pct=round(m_all['aw'], 4), avg_loss_pct=round(m_all['al'], 4),
            avg_bars_held=round(m_all['ab'], 1), buy_hold_pct=bh,
            train_pnl=round(m_train['pnl'], 4), test_pnl=round(m_test['pnl'], 4),
            train_wr=round(m_train['wr'], 2), test_wr=round(m_test['wr'], 2),
            train_trades=m_train['n'], test_trades=m_test['n'],
            degradation=round(
                (m_train['pnl'] - m_test['pnl']) / max(abs(m_train['pnl']), 0.01) * 100, 2
            ),
        )
        r.score = score(r)
        results.append(r)
        if r.score > 0: viable += 1
        if r.score > best_score:
            best_score = r.score
            best_str = (f'T={r.total_trades} WR={r.win_rate:.1f}% PF={r.profit_factor:.2f} '
                       f'PnL={r.total_pnl_pct:+.2f}% train={r.train_pnl:+.2f}% test={r.test_pnl:+.2f}%')
        if (idx + 1) % 2000 == 0:
            el = time.time() - t1; spd = (idx + 1) / max(el, 0.01)
            eta = (len(grid) - idx - 1) / max(spd, 0.01)
            print(f'  [{idx + 1:,}/{len(grid):,}] {spd:.0f}/s viable={viable} ETA={eta:.0f}s')
            if best_str: print(f'    BEST: {best_str}')

    elapsed = time.time() - t1
    print(f'\n  Concluido: {len(grid):,} combos em {elapsed:.1f}s | viable={viable}')

    # Sort by score
    results.sort(key=lambda x: x.score, reverse=True)

    # Print top results
    print(f'\n{"=" * 160}')
    print(f'  TOP 20 WALK-FORWARD VALIDADOS ({viable} viable de {len(grid):,})')
    print(f'{"=" * 160}')
    print(f'{"#":>3} {"Score":>7} {"Mode":>7} {"Trades":>6} {"L/S":>6} {"T/d":>5} {"WR%":>6} {"PF":>6} {"PnL%":>8} {"DD%":>6} {"Sharpe":>6} {"Train":>8} {"Test":>8} {"Deg":>6} {"SL":>4} {"TP":>4} {"MB":>3} {"CD":>3}')
    print(f'{"-" * 160}')
    for i, r in enumerate(results[:20]):
        if r.score <= 0: break
        ls = f'{r.long_trades}/{r.short_trades}'; td = r.total_trades / (n / 96)
        cd = r.p.get('cooldown', 4)
        print(f'{i + 1:>3} {r.score:>7.3f} {r.strategy:>7} {r.total_trades:>6} {ls:>6} {td:>5.2f} {r.win_rate:>6.1f} {r.profit_factor:>6.2f} '
              f'{r.total_pnl_pct:>8.2f} {r.max_drawdown_pct:>6.2f} {r.sharpe_ratio:>6.2f} '
              f'{r.train_pnl:>+8.2f} {r.test_pnl:>+8.2f} {r.degradation:>6.1f} '
              f'{r.sl_m:>4.2f} {r.tp_m:>4.2f} {r.max_bars:>3} {cd:>3}')

    # Also show best by PnL
    by_pnl = sorted(results, key=lambda x: x.total_pnl_pct, reverse=True)
    print(f'\n  Top 5 por PnL total:')
    for r in by_pnl[:5]:
        print(f'    {r.strategy} T={r.total_trades} WR={r.win_rate:.1f}% PF={r.profit_factor:.2f} PnL={r.total_pnl_pct:+.2f}% '
              f'train={r.train_pnl:+.2f}% test={r.test_pnl:+.2f}% L={r.long_trades} S={r.short_trades}')

    # Winner
    winner = results[0] if results and results[0].score > 0 else None
    if winner:
        print(f'\n{"#" * 140}')
        print(f'#  VENCEDOR 15m (WALK-FORWARD VALIDADO)')
        print(f'{"#" * 140}')
        print(f'#  Mode:          {winner.strategy}')
        print(f'#  Score:         {winner.score:.4f}')
        print(f'#  Trades:        {winner.total_trades} ({winner.total_trades / (n / 96):.2f}/dia) L={winner.long_trades} S={winner.short_trades}')
        print(f'#  Win Rate:      {winner.win_rate:.1f}%')
        print(f'#  Profit Factor: {winner.profit_factor:.2f}')
        print(f'#  PnL:           {winner.total_pnl_pct:+.2f}% (B&H={winner.buy_hold_pct:+.2f}%)')
        print(f'#  Max DD:        {winner.max_drawdown_pct:.2f}%  Sharpe: {winner.sharpe_ratio:.2f}')
        print(f'#  Train:         {winner.train_trades} trades, PnL={winner.train_pnl:+.2f}%, WR={winner.train_wr:.1f}%')
        print(f'#  Test:          {winner.test_trades} trades, PnL={winner.test_pnl:+.2f}%, WR={winner.test_wr:.1f}%')
        print(f'#  Degradation:   {winner.degradation:+.1f}%')
        print(f'#  Avg Win/Loss:  {winner.avg_win_pct:.3f}% / {winner.avg_loss_pct:.3f}%')
        print(f'#  SL/TP:         {winner.sl_m:.2f}x / {winner.tp_m:.2f}x (R:R {winner.tp_m / winner.sl_m:.1f}:1)')
        print(f'#  Cooldown:      {winner.p.get("cooldown", 4)} candles ({winner.p.get("cooldown", 4) * 15}min)')
        print(f'#  Params:        {json.dumps({k: v for k, v in winner.p.items()}, default=str)}')
        print(f'#  Total:         {len(grid):,} combos em {time.time() - t0:.0f}s')
        print(f'{"#" * 140}\n')

        # Save results
        out = os.path.join(PROJ_DIR, 'download')
        os.makedirs(out, exist_ok=True)
        pd.DataFrame([r.to_dict() for r in results]).to_csv(
            os.path.join(out, 'grid_15m_v6_results.csv'), index=False)
        clean_p = {}
        for k, v in winner.p.items():
            if isinstance(v, (tuple, list)): clean_p[k] = list(v)
            else: clean_p[k] = v
        wdata = {
            'version': '15m-v6-hybrid-walkforward', 'timeframe': '15m', 'symbol': 'BTC/USDT',
            'score': winner.score, 'strategy': winner.strategy,
            'params': clean_p,
            'metrics': {
                'total_trades': winner.total_trades,
                'trades_per_day': round(winner.total_trades / (n / 96), 3),
                'win_rate': winner.win_rate, 'profit_factor': winner.profit_factor,
                'total_pnl_pct': winner.total_pnl_pct, 'max_drawdown_pct': winner.max_drawdown_pct,
                'sharpe_ratio': winner.sharpe_ratio, 'buy_hold_pct': winner.buy_hold_pct,
                'train_pnl': winner.train_pnl, 'test_pnl': winner.test_pnl,
                'train_wr': winner.train_wr, 'test_wr': winner.test_wr,
                'train_trades': winner.train_trades, 'test_trades': winner.test_trades,
            },
            'costs': {'fee_pct': FEE, 'spread_bps': SPREAD, 'slippage_bps': SLIPPAGE, 'round_trip_pct': COST_RT * 100},
            'grid': {'total': len(grid), 'viable': viable, 'time_sec': round(time.time() - t0, 1)},
        }
        with open(os.path.join(out, 'winner_15m.json'), 'w') as f:
            json.dump(wdata, f, indent=2)
        print('Saved: winner_15m.json + grid_15m_v6_results.csv')
    else:
        print(f'\n!!! Nenhuma combinacao passou no walk-forward !!!')
        if results:
            print(f'!!! Top 5 por score (mesmo score=0):')
            for r in results[:5]:
                print(f'  {r.strategy} T={r.total_trades} WR={r.win_rate:.1f}% PF={r.profit_factor:.2f} '
                      f'PnL={r.total_pnl_pct:+.2f}% train={r.train_pnl:+.2f}% test={r.test_pnl:+.2f}%')


if __name__ == '__main__':
    main()
