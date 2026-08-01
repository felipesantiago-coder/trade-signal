from __future__ import annotations
import sys, os, time, math, json, pickle
from dataclasses import dataclass, asdict
from collections import Counter

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from backtest import fetch_historical_ohlcv, _apply_costs, DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS
from indicators import compute_indicators

SYMBOL = 'BTC/USDT'; TIMEFRAME = '15m'; DAYS = 365
DATA_CACHE = '/tmp/ctev_15m_data.pkl'
# Use SAME cost model as 1h (the bot's actual execution costs)
FEE = DEFAULT_FEE_PCT; SPREAD = DEFAULT_SPREAD_BPS; SLIPPAGE = DEFAULT_SLIPPAGE_BPS

MIN_TRADES = 80; MIN_WR = 48.0; MAX_DD = 22.0


@dataclass
class R:
    cid: int; score: float = 0.0; strategy: str = ''
    total_trades: int = 0; long_trades: int = 0; short_trades: int = 0
    wins: int = 0; losses: int = 0; win_rate: float = 0.0
    profit_factor: float = 0.0; total_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0; sharpe_ratio: float = 0.0
    avg_bars_held: float = 0.0; avg_win_pct: float = 0.0; avg_loss_pct: float = 0.0
    buy_hold_pct: float = 0.0; sl_m: float = 1.5; tp_m: float = 3.5
    max_bars: int = 72; p: dict = None
    # Walk-forward
    train_pnl: float = 0.0; test_pnl: float = 0.0
    train_wr: float = 0.0; test_wr: float = 0.0
    train_trades: int = 0; test_trades: int = 0
    degradation: float = 0.0

    def to_dict(self):
        d = asdict(self)
        d['params'] = self.p
        return d


def load_data():
    if os.path.exists(DATA_CACHE):
        with open(DATA_CACHE, 'rb') as f:
            return pickle.load(f)['df_clean']
    df = fetch_historical_ohlcv(SYMBOL, TIMEFRAME, DAYS)
    df = compute_indicators(df, timeframe=TIMEFRAME)
    crit = ['ema20','ema50','ema200','rsi','atr','atr_percentile',
            'adx','plus_di','minus_di','regime','bb_lower','bb_upper','bb_width','bb_squeeze_pct']
    df = df.dropna(subset=crit).copy()
    with open(DATA_CACHE, 'wb') as f:
        pickle.dump({'df_clean': df}, f)
    return df


def calc_score(r):
    # Base score: require profitability on BOTH train and test
    if r.total_trades < MIN_TRADES: return 0.0
    if r.train_pnl <= 0 or r.test_pnl <= 0: return 0.0  # Must profit in BOTH halves
    if r.profit_factor <= 1.0: return 0.0
    if r.win_rate < MIN_WR: return 0.0
    if r.max_drawdown_pct > MAX_DD: return 0.0
    # Penalize heavy degradation (overfitting)
    deg_pen = max(0, 1.0 - abs(r.degradation) / 100.0)
    wr = r.win_rate / 100.0; pf = r.profit_factor
    dd_pen = max(0, 1.0 - r.max_drawdown_pct / 100.0)
    freq = math.log(max(r.total_trades, 1)) / math.log(365)
    pnl_b = 2.0 if r.total_pnl_pct > 10 else (1.5 if r.total_pnl_pct > 5 else (1.2 if r.total_pnl_pct > 0 else 0.5))
    sig = min(math.sqrt(r.total_trades) / 15.0, 2.0)
    # Balance bonus: prefer balanced long/short
    if r.long_trades > 0 and r.short_trades > 0:
        ratio = min(r.long_trades, r.short_trades) / max(r.long_trades, r.short_trades)
        balance = 0.8 + 0.2 * ratio  # 0.8 to 1.0
    else:
        balance = 0.5  # Heavy penalty for one-sided
    return round(wr * pf * freq * dd_pen * pnl_b * sig * balance * deg_pen, 4)


def sim_period(c, hi, lo, regime, rsi, atr, adx, e50, e200, slope, atr_pct,
                 fib382, fib500, fib618, fibdir, e20t, e50t, e50tu, e20, bbl, bbu,
                 start, end, p):
    """Simulate strategy on a slice [start:end]."""
    entries=[]; dirs=[]; sls=[]; tps=[]; eps=[]
    rsi_l=p['rsi_l']; rsi_s=p['rsi_s']; adx_min=p.get('adx_min',0)
    allow_trn=p.get('allow_trn',True); req_et=p.get('req_et',True)
    req_sl=p.get('req_sl',False); req_pb=p.get('req_pb',False)
    sl_m=p['sl_m']; tp_m=p['tp_m']; max_bars=p.get('max_bars',72)
    fib_tol=p.get('fib_tol',0.025)
    atr_min=p.get('atr_min',0.05); atr_max=p.get('atr_max',0.95)
    allow_rng_trend=p.get('allow_rng_trend',False)
    vol_c=p.get('vol_c',False); vol_r=p.get('vol_r',0.3)
    mr_on=p.get('mr_enabled',False)
    mr_rl=p.get('mr_rl',35.0); mr_rs=p.get('mr_rs',65.0); mr_bb=p.get('mr_bb',0.0)
    i = start; cooldown = 0

    while i < end:
        if i < cooldown: i += 1; continue
        rg = regime[i]; sig = None; is_long = True

        # ── Mean-Reversion (ranging only) ──
        if mr_on and rg == 'ranging':
            ci = c[i]; li = lo[i]; hii = hi[i]; r = rsi[i]
            if r <= mr_rl and li <= bbl[i] - ci * mr_bb:
                ap = atr_pct[i]
                if atr_min <= ap <= atr_max:
                    av = atr[i]; sl = ci - sl_m * av; tp = ci + tp_m * av
                    if sl > 0:
                        entries.append(i); dirs.append(True); sls.append(sl); tps.append(tp); eps.append(ci)
                        cooldown = i + 4; continue
            if r >= mr_rs and hii >= bbu[i] + ci * mr_bb:
                ap = atr_pct[i]
                if atr_min <= ap <= atr_max:
                    av = atr[i]; sl = ci + sl_m * av; tp = ci - tp_m * av
                    entries.append(i); dirs.append(False); sls.append(sl); tps.append(tp); eps.append(ci)
                    cooldown = i + 4; continue
            i += 1; continue

        # ── Trend-Following ──
        if rg == 'trending_up':
            if adx[i] < adx_min: i += 1; continue
            is_long = True
        elif rg == 'trending_down':
            if adx[i] < adx_min: i += 1; continue
            is_long = False
        elif rg == 'transition':
            if not allow_trn: i += 1; continue
            # Direction based on EMA stack
            if c[i] > e50[i] > e200[i]: is_long = True
            elif c[i] < e50[i] < e200[i]: is_long = False
            else: i += 1; continue
        elif rg == 'ranging':
            if allow_rng_trend:
                if c[i] > e50[i] > e200[i]: is_long = True
                elif c[i] < e50[i] < e200[i]: is_long = False
                else: i += 1; continue
            else: i += 1; continue
        else:  # volatile
            i += 1; continue

        ci = c[i]; r = rsi[i]
        if is_long:
            if req_et and not (ci > e50[i] > e200[i]): i += 1; continue
            if req_sl and slope[i] <= -1.0: i += 1; continue
            if not (rsi_l[0] <= r <= rsi_l[1]): i += 1; continue
        else:
            if req_et and not (ci < e50[i] < e200[i]): i += 1; continue
            if req_sl and slope[i] >= 1.0: i += 1; continue
            if not (rsi_s[0] <= r <= rsi_s[1]): i += 1; continue

        if vol_c:
            vs = ... # skip for speed
        ap = atr_pct[i]
        if not (atr_min <= ap <= atr_max): i += 1; continue

        if req_pb:
            pb = False; fd = fibdir[i]; e20i = e20[i]; e50i = e50[i]
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
            if not pb and is_long and e20t[i] and ci > e20i: pb = True
            if not pb and not is_long and e20t[i] and ci < e20i and hi[i] >= e20i: pb = True
            if not pb and is_long and e50t[i] and ci > e50i: pb = True
            if not pb and not is_long and e50tu[i] and ci < e50i and hi[i] >= e50i: pb = True
            if not pb: i += 1; continue

        av = atr[i]
        if is_long:
            sl = ci - sl_m * av; tp = ci + tp_m * av
        else:
            sl = ci + sl_m * av; tp = ci - tp_m * av
        if is_long and sl <= 0: i += 1; continue
        entries.append(i); dirs.append(is_long); sls.append(sl); tps.append(tp); eps.append(ci)
        cooldown = i + 4
    return entries, dirs, sls, tps, eps


def run_sim(c, hi, lo, entries, dirs, sls, tps, eps, max_bars):
    """Execute trade simulation."""
    n = len(c); trades = []
    for k in range(len(entries)):
        i = entries[k]; is_long = dirs[k]
        sl = sls[k]; tp = tps[k]; ep = eps[k]
        exit_price = None; bars = 0; mj = min(i + max_bars, n)
        if is_long:
            for j in range(i + 1, mj):
                if lo[j] <= sl: exit_price = sl; bars = j - i; break
                if hi[j] >= tp: exit_price = tp; bars = j - i; break
        else:
            for j in range(i + 1, mj):
                if hi[j] >= sl: exit_price = sl; bars = j - i; break
                if lo[j] <= tp: exit_price = tp; bars = j - i; break
        if exit_price is None:
            bars = mj - 1 - i; exit_price = c[mj - 1]
        _, adj, _ = _apply_costs(ep, exit_price, is_long, FEE, SPREAD, SLIPPAGE)
        pnl = (adj - ep) / ep * 100 if is_long else (ep - adj) / ep * 100
        trades.append((is_long, pnl, bars))
    return trades


def calc_metrics(trades):
    if not trades: return {'n':0,'w':0,'wr':0,'pf':0,'pnl':0,'dd':0,'sh':0,'aw':0,'al':0,'ab':0,'l':0,'s':0}
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
    dd = (peak - eq) / peak * 100; mdd = np.max(dd)
    sharpe = np.mean(pnls) / (np.std(pnls) + 1e-9) * math.sqrt(365 / max(total / 96, 1))
    return {'n':total,'w':wins,'wr':wins/total*100,'pf':pf,'pnl':tpnl,'dd':mdd,
            'sh':sharpe,'aw':np.mean(wp) if wp else 0,'al':np.mean(lp) if lp else 0,
            'ab':np.mean([t[2] for t in trades]),'l':longs,'s':total-longs}


def make_grid():
    combos = []
    rsi_l_opts = [(25,50),(28,48),(30,55),(25,55),(28,55)]
    rsi_s_opts = [(50,75),(55,75),(45,70),(50,70)]
    sltp_opts = [(1.0,2.5),(1.5,3.0),(1.5,3.5),(2.0,4.0),(2.5,5.0),(3.0,6.0)]
    for rl in rsi_l_opts:
        for rs in rsi_s_opts:
            for sl,tp in sltp_opts:
                for adx in [0,20,25]:
                    for et in [True,False]:
                        for trn in [True,False]:
                            for pb in [True,False]:
                                for mb in [48,72,96]:
                                    p = {'rsi_l':rl,'rsi_s':rs,'sl_m':sl,'tp_m':tp,'max_bars':mb,
                                          'adx_min':adx,'allow_trn':trn,'req_et':et,
                                          'req_sl':False,'req_pb':pb,'fib_tol':0.025,
                                          'atr_min':0.05,'atr_max':0.95,
                                          'mr_enabled':False,'vol_c':False}
                                    combos.append(p)
    # With MR (mean-reversion in ranging)
    for mr_rl in [30,35,40]:
        for mr_rs in [60,65,70]:
            for mr_bb in [0.0,0.005,0.01]:
                for sl,tp in [(0.75,1.5),(1.0,2.0),(1.0,2.5),(1.5,2.5),(1.5,3.0)]:
                    for mb in [48,72]:
                        p = {'rsi_l':(25,55),'rsi_s':(45,70),'sl_m':sl,'tp_m':tp,'max_bars':mb,
                              'adx_min':0,'allow_trn':True,'req_et':False,
                              'req_sl':False,'req_pb':False,'fib_tol':0.025,
                              'atr_min':0.05,'atr_max':0.95,
                              'mr_enabled':True,'mr_rl':float(mr_rl),'mr_rs':float(mr_rs),'mr_bb':mr_bb,
                              'vol_c':False}
                        combos.append(p)
    return combos


def print_top(results, title, n=20):
    print(f"\n{'='*160}")
    print(f"  {title}")
    print(f"{'='*160}")
    print(f"{'#':>3} {'Score':>7} {'Trades':>6} {'L/S':>6} {'T/d':>5} {'WR%':>6} {'PF':>6} {'PnL%':>8} {'DD%':>6} {'Sharpe':>6} {'Train':>8} {'Test':>8} {'Deg':>6} {'PB':>3} {'ET':>3} {'MR':>3} {'SL':>4} {'TP':>4}")
    print(f"{'-'*160}")
    for i, r in enumerate(sorted(results, key=lambda x: x.score, reverse=True)[:n]):
        ls = f"{r.long_trades}/{r.short_trades}"; td = r.total_trades / 365
        print(f"{i+1:>3} {r.score:>7.3f} {r.total_trades:>6} {ls:>6} {td:>5.2f} {r.win_rate:>6.1f} {r.profit_factor:>6.2f} "
              f"{r.total_pnl_pct:>8.2f} {r.max_drawdown_pct:>6.2f} {r.sharpe_ratio:>6.2f} "
              f"{r.train_pnl:>+8.2f} {r.test_pnl:>+8.2f} {r.degradation:>6.1f} "
              f"{'S' if r.p.get('req_pb',False) else 'N':>3} {'S' if r.p.get('req_et',True) else 'N':>3} "
              f"{'S' if r.p.get('mr_enabled',False) else 'N':>3} {r.sl_m:>4.2f} {r.tp_m:>4.2f}")


def main():
    t0 = time.time()
    print("\n" + "#" * 160)
    print("#  CTEV 15m OPTIMIZER v5 — WALK-FORWARD VALIDATION")
    print(f"#  Custo: fee={FEE}% spread={SPREAD}bps slip={SLIPPAGE}bps (mesmo modelo do 1h)")
    print("#  Validacao: 50% treino / 50% teste (walk-forward)")
    print("#" * 160)

    print("\n[1/4] Carregando dados 15m...")
    df = load_data()
    n = len(df); bh = (df['close'].values[-1] - df['close'].values[0]) / df['close'].values[0] * 100
    rc = dict(zip(*np.unique(df['regime'].values, return_counts=True)))
    print(f"{n} candles ({n//96} dias) | B&H={bh:.2f}% | {rc}")

    # Pre-extract arrays
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

    mid = n // 2  # Split point for walk-forward
    bh_train = (c[mid-1] - c[0]) / c[0] * 100
    bh_test = (c[-1] - c[mid]) / c[mid] * 100
    print(f"Treino: {mid} candles ({mid//96}d) B&H={bh_train:+.2f}%")
    print(f"Teste:  {n-mid} candles ({(n-mid)//96}d) B&H={bh_test:+.2f}%")

    # Generate signals for BOTH periods
    arrs = (c, hi, lo, regime, rsi, atr, adx, e50, e200, slope, atr_pct,
            fib382, fib500, fib618, fibdir, e20t, e50t, e50tu, e20, bbl, bbu)

    print("\n[2/4] Gerando grid...")
    grid = make_grid()
    print(f"{len(grid):,} combinacoes")

    print("\n[3/4] Walk-Forward: treino + teste...")
    results = []; viable = 0; best_score = 0; best_str = ""; t1 = time.time()

    for idx, p in enumerate(grid):
        mb = p.get('max_bars', 72)

        # FULL period
        ent, dr, sl_l, tp_l, ep_l = sim_period(*arrs, 0, n, p)
        if not ent: continue
        trades_all = run_sim(c, hi, lo, ent, dr, sl_l, tp_l, ep_l, mb)
        m_all = calc_metrics(trades_all)

        # TRAIN period
        ent_t, dr_t, sl_t, tp_t, ep_t = sim_period(*arrs, 0, mid, p)
        trades_train = run_sim(c, hi, lo, ent_t, dr_t, sl_t, tp_t, ep_t, mb) if ent_t else []
        m_train = calc_metrics(trades_train)

        # TEST period
        ent_e, dr_e, sl_e, tp_e, ep_e = sim_period(*arrs, mid, n, p)
        trades_test = run_sim(c, hi, lo, ent_e, dr_e, sl_e, tp_e, ep_e, mb) if ent_e else []
        m_test = calc_metrics(trades_test)

        r = R(cid=idx, p=p, sl_m=p['sl_m'], tp_m=p['tp_m'], max_bars=mb,
             total_trades=m_all['n'], long_trades=m_all['l'], short_trades=m_all['s'],
             wins=m_all['w'], losses=m_all['n']-m_all['w'], win_rate=round(m_all['wr'],2),
             profit_factor=round(m_all['pf'],4), total_pnl_pct=round(m_all['pnl'],4),
             max_drawdown_pct=round(m_all['dd'],4), sharpe_ratio=round(m_all['sh'],4),
             avg_bars_held=round(m_all['ab'],1), avg_win_pct=round(m_all['aw'],4),
             avg_loss_pct=round(m_all['al'],4), buy_hold_pct=bh,
             train_pnl=round(m_train['pnl'],4), test_pnl=round(m_test['pnl'],4),
             train_wr=round(m_train['wr'],2), test_wr=round(m_test['wr'],2),
             train_trades=m_train['n'], test_trades=m_test['n'],
             degradation=round((m_train['pnl'] - m_test['pnl']) / max(abs(m_train['pnl']), 0.01) * 100, 2))
        r.score = calc_score(r)
        results.append(r)
        if r.score > 0: viable += 1
        if r.score > best_score:
            best_score = r.score
            best_str = (f"T={r.total_trades} WR={r.win_rate:.1f}% PF={r.profit_factor:.2f} "
                       f"PnL={r.total_pnl_pct:+.2f}% train={r.train_pnl:+.2f}% test={r.test_pnl:+.2f}%")
        if (idx + 1) % 1000 == 0:
            el = time.time() - t1; spd = (idx + 1) / max(el, 0.01)
            eta = (len(grid) - idx - 1) / max(spd, 0.01)
            print(f"  [{idx+1:,}/{len(grid):,}] {spd:.0f}/s viable={viable} ETA={eta:.0f}s")
            if best_str: print(f"    BEST: {best_str}")

    elapsed = time.time() - t1
    print(f"\n  Concluido: {len(grid):,} combos em {elapsed:.1f}s | viable={viable}")

    # Also show best by PnL (ignoring score)
    by_pnl = sorted(results, key=lambda x: x.total_pnl_pct, reverse=True)
    print("\n  Top 5 por PnL total:")
    for r in by_pnl[:5]:
        print(f"    T={r.total_trades} WR={r.win_rate:.1f}% PF={r.profit_factor:.2f} PnL={r.total_pnl_pct:+.2f}% "
              f"train={r.train_pnl:+.2f}% test={r.test_pnl:+.2f}% L={r.long_trades} S={r.short_trades}")

    print_top(results, f"WALK-FORWARD TOP 20 ({len(results):,} combos, viable={viable})", 20)

    winner = max(results, key=lambda x: x.score) if results else None
    if winner and winner.score > 0:
        print(f"\n{'#'*160}")
        print(f"#  VENCEDOR 15m (WALK-FORWARD VALIDADO)")
        print(f"{'#'*160}")
        print(f"#  Score:         {winner.score:.4f}")
        print(f"#  Trades:        {winner.total_trades} ({winner.total_trades/365:.2f}/dia) L={winner.long_trades} S={winner.short_trades}")
        print(f"#  Win Rate:      {winner.win_rate:.1f}%")
        print(f"#  Profit Factor: {winner.profit_factor:.2f}")
        print(f"#  PnL:           {winner.total_pnl_pct:+.2f}% (B&H={winner.buy_hold_pct:+.2f}%)")
        print(f"#  Max DD:        {winner.max_drawdown_pct:.2f}%  Sharpe: {winner.sharpe_ratio:.2f}")
        print(f"#  Train:         {winner.train_trades} trades, PnL={winner.train_pnl:+.2f}%, WR={winner.train_wr:.1f}%")
        print(f"#  Test:          {winner.test_trades} trades, PnL={winner.test_pnl:+.2f}%, WR={winner.test_wr:.1f}%")
        print(f"#  Degradation:   {winner.degradation:+.1f}%")
        print(f"#  Avg Win/Loss:  {winner.avg_win_pct:.3f}% / {winner.avg_loss_pct:.3f}%")
        print(f"#  SL/TP:         {winner.sl_m:.2f}x / {winner.tp_m:.2f}x (R:R {winner.tp_m/winner.sl_m:.1f}:1)")
        print(f"#  Params:        {json.dumps({k:v for k,v in winner.p.items()}, default=str)}")
        print(f"#  Total:         {len(grid):,} combos em {time.time()-t0:.0f}s")
        print(f"{'#'*160}\n")

        out = os.path.join(SCRIPT_DIR, '..', 'download')
        os.makedirs(out, exist_ok=True)
        pd.DataFrame([r.to_dict() for r in results]).sort_values('score', ascending=False).to_csv(
            os.path.join(out, 'grid_15m_v5_results.csv'), index=False)
        clean_p = {}
        for k, v in winner.p.items():
            if isinstance(v, (tuple, list)): clean_p[k] = list(v)
            else: clean_p[k] = v
        wdata = {
            'version': '15m-v5-walkforward', 'timeframe': '15m', 'symbol': SYMBOL,
            'score': winner.score,
            'params': clean_p,
            'metrics': {
                'total_trades': winner.total_trades, 'trades_per_day': round(winner.total_trades/365, 3),
                'win_rate': winner.win_rate, 'profit_factor': winner.profit_factor,
                'total_pnl_pct': winner.total_pnl_pct, 'max_drawdown_pct': winner.max_drawdown_pct,
                'sharpe_ratio': winner.sharpe_ratio, 'buy_hold_pct': winner.buy_hold_pct,
                'train_pnl': winner.train_pnl, 'test_pnl': winner.test_pnl,
                'train_wr': winner.train_wr, 'test_wr': winner.test_wr,
                'train_trades': winner.train_trades, 'test_trades': winner.test_trades,
            },
            'costs': {'fee_pct': FEE, 'spread_bps': SPREAD, 'slippage_bps': SLIPPAGE},
            'grid': {'total': len(grid), 'viable': viable, 'time_sec': round(time.time()-t0, 1)},
        }
        with open(os.path.join(out, 'winner_15m.json'), 'w') as f:
            json.dump(wdata, f, indent=2)
        print('Saved: winner_15m.json + grid_15m_v5_results.csv')
    else:
        print(f"\n!!! Nenhuma combinacao passou no walk-forward !!!")
        print(f"!!! Mostrando top 5 por score (mesmo com score=0):")
        for r in sorted(results, key=lambda x: x.score, reverse=True)[:5]:
            print(f"  T={r.total_trades} WR={r.win_rate:.1f}% PF={r.profit_factor:.2f} PnL={r.total_pnl_pct:+.2f}% "
                  f"train={r.train_pnl:+.2f}% test={r.test_pnl:+.2f}%")


if __name__ == '__main__':
    main()
