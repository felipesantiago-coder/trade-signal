"""
optimize_15m_v2.py
-----------------
Versao otimizada do otimizador 15m. Foco em velocidade:
  - Numpy arrays pre-alocados (sem re-extrair por combo)
  - Grid enxuto: ~800 combos P1 (vs 5180)
  - Fase 2 mais curta
  - Custos realistas para 15m
"""
from __future__ import annotations
import sys, os, time, math, json, logging, pickle
from dataclasses import dataclass, asdict
from typing import List

logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger("ctev.opt15m")
logger.setLevel(logging.INFO)

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from indicators import compute_indicators
from backtest import (fetch_historical_ohlcv, calculate_metrics, TradeResult,
    _apply_costs, DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS)

SYMBOL = "BTC/USDT"; TIMEFRAME = "15m"; DAYS = 365
DATA_CACHE = "/tmp/ctev_15m_data.pkl"
MIN_TRADES_P1 = 60; MIN_TRADES_P2 = 100
MIN_WIN_RATE = 48.0; MAX_DRAWDOWN = 22.0
FEE = 0.025; SPREAD = 10.0; SLIPPAGE = 25.0


@dataclass
class R:
    cid: int; score: float = 0.0; phase: str = "p1"
    total_trades: int = 0; long_trades: int = 0; short_trades: int = 0
    wins: int = 0; losses: int = 0; win_rate: float = 0.0
    profit_factor: float = 0.0; total_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0; sharpe_ratio: float = 0.0
    avg_bars_held: float = 0.0; avg_win_pct: float = 0.0; avg_loss_pct: float = 0.0
    buy_hold_pct: float = 0.0
    # Params
    mr: bool = False; mr_rl: float = 35.0; mr_rs: float = 65.0; mr_bb: float = 0.0
    allow_trn: bool = True; req_pb: bool = True; req_et: bool = True; req_sl: bool = True
    rsi_l: tuple = (28.0, 48.0); rsi_s: tuple = (55.0, 75.0)
    adx: float = 0.0; sl_m: float = 1.5; tp_m: float = 3.5
    vol_c: bool = False; vol_r: float = 0.3
    fib_tol: float = 0.025; e20p: float = 0.0; e50p: float = 0.0
    atr_min: float = 0.05; atr_max: float = 0.95

    def to_dict(self): return asdict(self)


def load_arrays():
    if os.path.exists(DATA_CACHE):
        with open(DATA_CACHE, 'rb') as f:
            df = pickle.load(f)['df_clean']
    else:
        df = fetch_historical_ohlcv(SYMBOL, TIMEFRAME, DAYS)
        df = compute_indicators(df, timeframe=TIMEFRAME)
        crit = ["ema20","ema50","ema200","rsi","atr","atr_percentile",
                "macd","macd_signal","macd_hist","adx","plus_di","minus_di","regime"]
        df = df.dropna(subset=crit).copy()
        with open(DATA_CACHE, 'wb') as f:
            pickle.dump({'df_clean': df}, f)
    return df


def fast_sim(arrs, p, max_bars=72):
    """Ultra-fast simulation using pre-extracted numpy arrays."""
    (n, cl, hi, lo, rg, ap, rsi, atr,
     e20, e50, e200, adx, slp, fib382, fib500, fib618, fibdir,
     vol, vs50, e20t, e50t, e50tu, bbl, bbu) = arrs

    trades = []; i = 0
    mr_on = p['mr']
    allow_trn = p['allow_trn']
    req_pb = p['req_pb']
    req_et = p['req_et']
    req_sl = p['req_sl']
    rsi_lmin, rsi_lmax = p['rsi_l']
    rsi_smin, rsi_smax = p['rsi_s']
    adx_min = p['adx']
    sl_m = p['sl_m']; tp_m = p['tp_m']
    atr_pmin = p['atr_min']; atr_pmax = p['atr_max']
    fib_tol = p['fib_tol']
    vol_c = p['vol_c']; vol_r = p['vol_r']
    mr_rl = p['mr_rl']; mr_rs = p['mr_rs']; mr_bb = p['mr_bb']
    e20p = p['e20p']; e50p = p['e50p']

    while i < n:
        sig = None; is_long = True; cur_rg = rg[i]

        if cur_rg == 'ranging':
            if mr_on:
                c = cl[i]; lo_i = lo[i]; r = rsi[i]
                if r <= mr_rl:
                    bbl_i = bbl[i]
                    if mr_bb > 0:
                        if lo_i <= bbl_i - c * mr_bb:
                            ap_i = ap[i]
                            if atr_pmin <= ap_i <= atr_pmax:
                                av = atr[i]; sl = c - sl_m * av; tp = c + tp_m * av
                                if sl > 0: sig = (c, sl, tp, av, r); is_long = True
                if sig is None:
                    c = cl[i]; hi_i = hi[i]; r = rsi[i]
                    if r >= mr_rs:
                        bbu_i = bbu[i]
                        if mr_bb > 0:
                            if hi_i >= bbu_i + c * mr_bb:
                                ap_i = ap[i]
                                if atr_pmin <= ap_i <= atr_pmax:
                                    av = atr[i]; sl = c + sl_m * av; tp = c - tp_m * av
                                    sig = (c, sl, tp, av, r); is_long = False
                        else:
                            if hi_i >= bbu_i:
                                ap_i = ap[i]
                                if atr_pmin <= ap_i <= atr_pmax:
                                    av = atr[i]; sl = c + sl_m * av; tp = c - tp_m * av
                                    sig = (c, sl, tp, av, r); is_long = False
        elif cur_rg in ('trending_up', 'transition'):
            if cur_rg == 'trending_up' and adx[i] < adx_min:
                i += 1; continue
            if cur_rg == 'transition' and not allow_trn:
                i += 1; continue
            # Try LONG
            c = cl[i]; e50_i = e50[i]; e200_i = e200[i]; r = rsi[i]
            if req_et and not (c > e50_i > e200_i):
                i += 1; continue
            if req_sl and slp[i] <= -1.0:
                i += 1; continue
            if not (rsi_lmin <= r <= rsi_lmax):
                i += 1; continue
            if vol_c and not np.isnan(vs50[i]) and vs50[i] > 0 and vol[i] < vs50[i] * vol_r:
                i += 1; continue
            ap_i = ap[i]
            if not (atr_pmin <= ap_i <= atr_pmax):
                i += 1; continue
            if req_pb:
                pb = False; fd = fibdir[i]; e20_i = e20[i]
                if fd == 1:
                    f38 = fib382[i]; f61 = fib618[i]
                    if not (np.isnan(f38) or np.isnan(f61)) and f61 <= c <= f38:
                        pb = True
                    if not pb:
                        lo_i = lo[i]; tol = c * fib_tol
                        for fl in (f38, fib500[i], f61):
                            if not np.isnan(fl) and fl > 0 and abs(lo_i - fl) <= tol:
                                pb = True; break
                if not pb and e20t[i] and c > e20_i:
                    pb = True
                if not pb and e50t[i] and c > e50_i:
                    pb = True
                if not pb and e20p > 0 and e20_i > 0 and abs(c - e20_i) / e20_i <= e20p:
                    pb = True
                if not pb and e50p > 0 and e50_i > 0 and abs(c - e50_i) / e50_i <= e50p:
                    pb = True
                if not pb:
                    i += 1; continue
            av = atr[i]; sl = c - sl_m * av; tp = c + tp_m * av
            if sl > 0: sig = (c, sl, tp, av, r); is_long = True

        elif cur_rg in ('trending_down', 'transition'):
            if cur_rg == 'trending_down' and adx[i] < adx_min:
                i += 1; continue
            if cur_rg == 'transition' and not allow_trn:
                i += 1; continue
            # Try SHORT
            c = cl[i]; e50_i = e50[i]; e200_i = e200[i]; r = rsi[i]
            if req_et and not (c < e50_i < e200_i):
                i += 1; continue
            if req_sl and slp[i] >= 1.0:
                i += 1; continue
            if not (rsi_smin <= r <= rsi_smax):
                i += 1; continue
            if vol_c and not np.isnan(vs50[i]) and vs50[i] > 0 and vol[i] < vs50[i] * vol_r:
                i += 1; continue
            ap_i = ap[i]
            if not (atr_pmin <= ap_i <= atr_pmax):
                i += 1; continue
            if req_pb:
                pb = False; fd = fibdir[i]; e20_i = e20[i]
                if fd == -1:
                    f38 = fib382[i]; f61 = fib618[i]
                    if not (np.isnan(f38) or np.isnan(f61)) and f61 <= c <= f38:
                        pb = True
                    if not pb:
                        hi_i = hi[i]; tol = c * fib_tol
                        for fl in (f38, fib500[i], f61):
                            if not np.isnan(fl) and fl > 0 and abs(hi_i - fl) <= tol:
                                pb = True; break
                if not pb and e20t[i] and c < e20_i and hi[i] >= e20_i:
                    pb = True
                if not pb and e50tu[i] and c < e50_i and hi[i] >= e50_i:
                    pb = True
                if not pb and e20p > 0 and e20_i > 0 and abs(c - e20_i) / e20_i <= e20p:
                    pb = True
                if not pb and e50p > 0 and e50_i > 0 and abs(c - e50_i) / e50_i <= e50p:
                    pb = True
                if not pb:
                    i += 1; continue
            av = atr[i]; sl = c + sl_m * av; tp = c - tp_m * av
            sig = (c, sl, tp, av, r); is_long = False

        if sig is None:
            i += 1; continue

        ep, sl, tp, atr_v, rsi_v = sig
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
            bars = mj - 1 - i; exit_price = cl[mj - 1]

        _, adj, _ = _apply_costs(ep, exit_price, is_long, FEE, SPREAD, SLIPPAGE)
        pnl = (adj - ep) / ep * 100 if is_long else (ep - adj) / ep * 100
        trades.append((is_long, pnl, bars))
        i += bars + 1

    return trades


def calc_metrics(trades, n_days=365):
    if not trades: return R(0)
    total = len(trades)
    wins = sum(1 for t in trades if t[1] > 0)
    losses = total - wins
    wr = wins / total * 100 if total > 0 else 0
    longs = sum(1 for t in trades if t[0])
    shorts = total - longs
    pnls = [t[1] for t in trades]
    wins_p = [p for p in pnls if p > 0]
    loss_p = [p for p in pnls if p <= 0]
    gross_w = sum(wins_p)
    gross_l = abs(sum(loss_p))
    pf = gross_w / gross_l if gross_l > 0 else 999.0
    total_pnl = sum(pnls)
    avg_w = np.mean(wins_p) if wins_p else 0
    avg_l = np.mean(loss_p) if loss_p else 0
    # Drawdown
    eq = [100.0]
    for p in pnls: eq.append(eq[-1] * (1 + p / 100))
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak * 100
    max_dd = np.max(dd)
    # Sharpe (simplified)
    if len(pnls) > 1:
        sharpe = np.mean(pnls) / (np.std(pnls) + 1e-9) * math.sqrt(365 / max(total / 96, 1))  # 96 candles/day in 15m
    else:
        sharpe = 0
    avg_bars = np.mean([t[2] for t in trades])
    return R(0, total_trades=total, long_trades=longs, short_trades=shorts,
              wins=wins, losses=losses, win_rate=round(wr, 2),
              profit_factor=round(pf, 4), total_pnl_pct=round(total_pnl, 4),
              max_drawdown_pct=round(max_dd, 4), sharpe_ratio=round(sharpe, 4),
              avg_bars_held=round(avg_bars, 1),
              avg_win_pct=round(avg_w, 4), avg_loss_pct=round(avg_l, 4))


def score(r, phase="p1"):
    min_t = MIN_TRADES_P1 if phase == "p1" else MIN_TRADES_P2
    if r.total_trades < min_t or r.profit_factor <= 1.0: return 0.0
    if r.win_rate < MIN_WIN_RATE or r.max_drawdown_pct > MAX_DRAWDOWN: return 0.0
    wr = r.win_rate / 100.0; pf = r.profit_factor
    dd_pen = max(0, 1.0 - r.max_drawdown_pct / 100.0)
    freq = math.log(max(r.total_trades, 1)) / math.log(365)
    pnl_b = 2.0 if r.total_pnl_pct > 10 else (1.5 if r.total_pnl_pct > 5 else (1.2 if r.total_pnl_pct > 0 else 0.5))
    sig = min(math.sqrt(r.total_trades) / 15.0, 2.0)
    return round(wr * pf * freq * dd_pen * pnl_b * sig, 4)


def make_p1():
    combos = []
    # Trend configs: (allow_trn, req_pb, req_et, req_sl, adx, rsi_l, rsi_s)
    tcfgs = [
        (True,  True,  True,  True,  20, (28,48), (55,75)),
        (True,  True,  True,  True,  25, (28,48), (55,75)),
        (True,  True,  True,  True,  15, (28,50), (53,75)),
        (True,  True,  True,  False, 20, (28,50), (50,75)),
        (True,  True,  True,  False, 25, (25,50), (50,75)),
        (True,  False, True,  False, 20, (30,55), (45,70)),
        (True,  False, True,  False, 25, (30,55), (45,70)),
        (True,  False, False, False, 15, (35,60), (40,65)),
        (True,  True,  True,  True,  20, (25,45), (55,75)),
        (True,  True,  True,  False, 20, (25,55), (45,75)),
        (True,  False, True,  False, 20, (25,55), (45,75)),
    ]
    tr_sltp = [(0.75,2.0),(1.0,2.5),(1.0,3.0),(1.25,2.5),(1.25,3.0),(1.5,3.0),(1.5,3.5),(0.75,1.5),(1.0,2.0)]
    mr_rl_v = [30,35,40,45]; mr_rs_v = [55,60,65,70]; mr_bb_v = [0.0,0.005,0.01]
    mr_sltp = [(0.5,1.0),(0.75,1.5),(1.0,1.5),(0.75,2.0),(1.0,2.0),(1.0,2.5)]

    for tc in tcfgs:
        a_trn, r_pb, r_et, r_sl, adx, rsi_l, rsi_s = tc
        for mr in [True, False]:
            if mr:
                for mrl in mr_rl_v:
                    for mrs in mr_rs_v:
                        for mbb in mr_bb_v:
                            for sl,tp in mr_sltp:
                                combos.append({'mr':True,'mr_rl':float(mrl),'mr_rs':float(mrs),'mr_bb':mbb,
                                    'allow_trn':a_trn,'req_pb':r_pb,'req_et':r_et,'req_sl':r_sl,
                                    'rsi_l':rsi_l,'rsi_s':rsi_s,'adx':adx,'fib_tol':0.025,
                                    'e20p':0.0,'e50p':0.0,'vol_c':False,'vol_r':0.3,
                                    'atr_min':0.05,'atr_max':0.95,'sl_m':sl,'tp_m':tp})
            else:
                for sl,tp in tr_sltp:
                    combos.append({'mr':False,'mr_rl':35.0,'mr_rs':65.0,'mr_bb':0.0,
                        'allow_trn':a_trn,'req_pb':r_pb,'req_et':r_et,'req_sl':r_sl,
                        'rsi_l':rsi_l,'rsi_s':rsi_s,'adx':adx,'fib_tol':0.025,
                        'e20p':0.0,'e50p':0.0,'vol_c':False,'vol_r':0.3,
                        'atr_min':0.05,'atr_max':0.95,'sl_m':sl,'tp_m':tp})
    return combos


def make_p2(top, n=10):
    sls=[0.5,0.75,1.0,1.25,1.5,1.75,2.0]
    tps=[1.0,1.25,1.5,2.0,2.5,3.0,3.5,4.0]
    mr_l=[28,32,35,38,42,45]; mr_s=[50,55,58,62,65,68]; mr_b=[0.0,0.003,0.005,0.01,0.015]
    combos=[]
    for r in top[:n]:
        base={'mr':r.mr,'mr_rl':r.mr_rl,'mr_rs':r.mr_rs,'mr_bb':r.mr_bb,
              'allow_trn':r.allow_trn,'req_pb':r.req_pb,'req_et':r.req_et,'req_sl':r.req_sl,
              'rsi_l':r.rsi_l,'rsi_s':r.rsi_s,'adx':r.adx,'fib_tol':r.fib_tol,
              'e20p':r.e20p,'e50p':r.e50p,'vol_c':r.vol_c,'vol_r':r.vol_r,
              'atr_min':r.atr_min,'atr_max':r.atr_max}
        if r.mr:
            for sl in sls:
                for tp in tps:
                    if tp/sl<0.8: continue
                    for mrl in mr_l:
                        for mrs in mr_s:
                            for mbb in mr_b:
                                combos.append({**base,'mr':True,'mr_rl':float(mrl),'mr_rs':float(mrs),'mr_bb':mbb,'sl_m':sl,'tp_m':tp})
        else:
            for sl in sls:
                for tp in tps:
                    if tp/sl<0.8: continue
                    combos.append({**base,'sl_m':sl,'tp_m':tp})
    return combos


def print_top(results, title, n=15):
    print(f"\n{'='*140}")
    print(f"  {title}")
    print(f"{'='*140}")
    print(f"{'#':>3} {'Score':>7} {'Trades':>6} {'L/S':>6} {'T/d':>5} {'WR%':>6} {'PF':>6} {'PnL%':>8} {'DD%':>6} {'Sharpe':>6} "
          f"{'Trn':>3} {'Pb':>3} {'ET':>3} {'MR':>3} {'MR_RL':>5} {'MR_RS':>5} "
          f"{'RSI_L':>8} {'RSI_S':>8} {'ADX':>4} {'SL':>4} {'TP':>4}")
    print(f"{'-'*140}")
    for i,r in enumerate(sorted(results, key=lambda x: x.score, reverse=True)[:n]):
        rl=f"{r.rsi_l[0]:.0f}-{r.rsi_l[1]:.0f}"; rs=f"{r.rsi_s[0]:.0f}-{r.rsi_s[1]:.0f}"
        ls=f"{r.long_trades}/{r.short_trades}"; td=r.total_trades/365
        print(f"{i+1:>3} {r.score:>7.3f} {r.total_trades:>6} {ls:>6} {td:>5.2f} {r.win_rate:>6.1f} {r.profit_factor:>6.2f} "
              f"{r.total_pnl_pct:>8.2f} {r.max_drawdown_pct:>6.2f} {r.sharpe_ratio:>6.2f} "
              f"{'S' if r.allow_trn else 'N':>3} {'S' if r.req_pb else 'N':>3} "
              f"{'S' if r.req_et else 'N':>3} {'S' if r.mr else 'N':>3} "
              f"{r.mr_rl:>5.0f} {r.mr_rs:>5.0f} "
              f"{rl:>8} {rs:>8} {r.adx:>4.0f} {r.sl_m:>4.2f} {r.tp_m:>4.2f}")


def main():
    t0 = time.time()
    print("\n" + "#" * 140)
    print("#  CTEV 15m OPTIMIZER v2 — Grid Search para Timeframe 15 Minutos")
    print(f"#  Custo: fee={FEE}% spread={SPREAD}bps slip={SLIPPAGE}bps")
    print("#" * 140)

    print("\n[1/4] Carregando dados 15m...")
    df = load_arrays()
    n = len(df)
    idx_arr = df.index.values
    buy_hold = (df['close'].values[-1] - df['close'].values[0]) / df['close'].values[0] * 100

    # Pre-extract numpy arrays ONCE
    arrs = (
        n,
        df['close'].values.astype(np.float64),
        df['high'].values.astype(np.float64),
        df['low'].values.astype(np.float64),
        df['regime'].values,
        df['atr_percentile'].values.astype(np.float64),
        df['rsi'].values.astype(np.float64),
        df['atr'].values.astype(np.float64),
        df['ema20'].values.astype(np.float64),
        df['ema50'].values.astype(np.float64),
        df['ema200'].values.astype(np.float64),
        df['adx'].values.astype(np.float64),
        df['ema50_slope'].values.astype(np.float64),
        df['fib_0382'].values.astype(np.float64),
        df['fib_0500'].values.astype(np.float64),
        df['fib_0618'].values.astype(np.float64),
        df['fib_direction'].values.astype(np.int32),
        df['volume'].values.astype(np.float64),
        df['volume_sma50'].values.astype(np.float64),
        df['ema20_touched'].values,
        df['ema50_touched'].values,
        df['ema50_touched_up'].values,
        df['bb_lower'].values.astype(np.float64),
        df['bb_upper'].values.astype(np.float64),
    )

    rc = dict(zip(*np.unique(df['regime'].values, return_counts=True)))
    print(f"{n} candles ({n//96} dias) | B&H={buy_hold:.2f}% | {rc}")

    # Phase 1
    print("\n[2/4] PHASE 1: Grid search...")
    p1 = make_p1()
    print(f"      {len(p1):,} combinacoes")
    p1_res=[]; viable=0; best_score=0; best_str=""; t1=time.time()

    for idx, p in enumerate(p1):
        trades = fast_sim(arrs, p, 72)
        m = calc_metrics(trades)
        r = R(cid=idx, mr=p['mr'], mr_rl=p['mr_rl'], mr_rs=p['mr_rs'], mr_bb=p['mr_bb'],
              allow_trn=p['allow_trn'], req_pb=p['req_pb'], req_et=p['req_et'], req_sl=p['req_sl'],
              rsi_l=p['rsi_l'], rsi_s=p['rsi_s'], adx=p['adx'], sl_m=p['sl_m'], tp_m=p['tp_m'],
              vol_c=p['vol_c'], vol_r=p['vol_r'], fib_tol=p['fib_tol'],
              e20p=p['e20p'], e50p=p['e50p'], atr_min=p['atr_min'], atr_max=p['atr_max'],
              total_trades=m.total_trades, long_trades=m.long_trades, short_trades=m.short_trades,
              wins=m.wins, losses=m.losses, win_rate=m.win_rate, profit_factor=m.profit_factor,
              total_pnl_pct=m.total_pnl_pct, max_drawdown_pct=m.max_drawdown_pct,
              sharpe_ratio=m.sharpe_ratio, avg_bars_held=m.avg_bars_held,
              avg_win_pct=m.avg_win_pct, avg_loss_pct=m.avg_loss_pct, buy_hold_pct=buy_hold,
              phase="p1")
        r.score = score(r, "p1")
        p1_res.append(r)
        if r.score > 0: viable += 1
        if r.score > best_score:
            best_score = r.score
            best_str = f"T={r.total_trades} WR={r.win_rate:.1f}% PF={r.profit_factor:.2f} PnL={r.total_pnl_pct:+.2f}% DD={r.max_drawdown_pct:.2f}% MR={'S' if r.mr else 'N'}"
        if (idx+1) % 1000 == 0:
            el=time.time()-t1; spd=(idx+1)/max(el,0.01); eta=(len(p1)-idx-1)/max(spd,0.01)
            print(f"      [{idx+1:,}/{len(p1):,}] {spd:.0f}/s viable={viable} ETA={eta:.0f}s")
            if best_str: print(f"        BEST: {best_str}")

    p1t = time.time()-t1
    print(f"      Phase 1: {len(p1):,} combos em {p1t:.1f}s | viable={viable}")
    p1s = sorted(p1_res, key=lambda x: x.score, reverse=True)
    print_top(p1s, f"PHASE 1 Top 15 de {len(p1):,}", 15)

    if not p1s or p1s[0].score == 0:
        print("\n!!! Nenhuma viavel no P1 !!!")
        by_pnl = sorted(p1_res, key=lambda x: x.total_pnl_pct, reverse=True)
        for r in by_pnl[:10]:
            print(f"  T={r.total_trades} WR={r.win_rate:.1f}% PF={r.profit_factor:.2f} PnL={r.total_pnl_pct:+.2f}% MR={'S' if r.mr else 'N'}")
        # Save anyway for analysis
        out = os.path.join(SCRIPT_DIR, "..", "download")
        os.makedirs(out, exist_ok=True)
        pd.DataFrame([r.to_dict() for r in p1_res]).sort_values("score", ascending=False).to_csv(
            os.path.join(out, "grid_15m_results.csv"), index=False)
        return

    n_top = min(10, len([r for r in p1s if r.score > 0]))
    p1_top = [r for r in p1s if r.score > 0][:n_top]

    # Phase 2
    print(f"\n[3/4] PHASE 2: Refinamento para top {n_top}...")
    p2 = make_p2(p1_top, n_top)
    print(f"      {len(p2):,} combinacoes")
    p2_res=[]; t2=time.time(); best_p2=""
    for idx, p in enumerate(p2):
        trades = fast_sim(arrs, p, 72)
        m = calc_metrics(trades)
        r = R(cid=200000+idx, mr=p['mr'], mr_rl=p['mr_rl'], mr_rs=p['mr_rs'], mr_bb=p['mr_bb'],
              allow_trn=p['allow_trn'], req_pb=p['req_pb'], req_et=p['req_et'], req_sl=p['req_sl'],
              rsi_l=p['rsi_l'], rsi_s=p['rsi_s'], adx=p['adx'], sl_m=p['sl_m'], tp_m=p['tp_m'],
              vol_c=p['vol_c'], vol_r=p['vol_r'], fib_tol=p['fib_tol'],
              e20p=p['e20p'], e50p=p['e50p'], atr_min=p['atr_min'], atr_max=p['atr_max'],
              total_trades=m.total_trades, long_trades=m.long_trades, short_trades=m.short_trades,
              wins=m.wins, losses=m.losses, win_rate=m.win_rate, profit_factor=m.profit_factor,
              total_pnl_pct=m.total_pnl_pct, max_drawdown_pct=m.max_drawdown_pct,
              sharpe_ratio=m.sharpe_ratio, avg_bars_held=m.avg_bars_held,
              avg_win_pct=m.avg_win_pct, avg_loss_pct=m.avg_loss_pct, buy_hold_pct=buy_hold,
              phase="p2")
        r.score = score(r, "p2")
        p2_res.append(r)
        if r.score > best_score:
            best_score = r.score
            best_p2 = f"T={r.total_trades} WR={r.win_rate:.1f}% PF={r.profit_factor:.2f} PnL={r.total_pnl_pct:+.2f}% SL={r.sl_m:.2f} TP={r.tp_m:.2f}"
        if (idx+1) % 3000 == 0:
            el=time.time()-t2; spd=(idx+1)/max(el,0.01); eta=(len(p2)-idx-1)/max(spd,0.01)
            print(f"      [{idx+1:,}/{len(p2):,}] {spd:.0f}/s ETA={eta:.0f}s")
            if best_p2: print(f"        BEST: {best_p2}")
    p2t = time.time()-t2
    print(f"      Phase 2: {len(p2):,} combos em {p2t:.1f}s")

    all_res = p1_res + p2_res
    all_s = sorted(all_res, key=lambda x: x.score, reverse=True)
    print_top(all_s, f"COMBINED Top 15 ({len(all_res):,} total)", 15)

    winner = all_s[0] if all_s else None
    tt = time.time()-t0
    if winner:
        print(f"\n{'#'*140}")
        print(f"#  VENCEDOR 15m")
        print(f"{'#'*140}")
        print(f"#  Score:         {winner.score:.4f}")
        print(f"#  Trades:        {winner.total_trades} ({winner.total_trades/365:.2f}/dia) L={winner.long_trades} S={winner.short_trades}")
        print(f"#  Win Rate:      {winner.win_rate:.1f}%")
        print(f"#  Profit Factor: {winner.profit_factor:.2f}")
        print(f"#  PnL:           {winner.total_pnl_pct:+.2f}% (B&H={winner.buy_hold_pct:+.2f}%)")
        print(f"#  Max DD:        {winner.max_drawdown_pct:.2f}%  Sharpe: {winner.sharpe_ratio:.2f}")
        print(f"#  Avg Win:       {winner.avg_win_pct:.3f}% | Avg Loss: {winner.avg_loss_pct:.3f}%")
        print(f"#  MR:            {winner.mr} | RSI_L_mr={winner.mr_rl} RSI_S_mr={winner.mr_rs} BB={winner.mr_bb}")
        print(f"#  Trend:         transition={'S' if winner.allow_trn else 'N'} pullback={'S' if winner.req_pb else 'N'} ema_trend={'S' if winner.req_et else 'N'} slope={'S' if winner.req_sl else 'N'}")
        print(f"#  RSI:           L={winner.rsi_l[0]:.0f}-{winner.rsi_l[1]:.0f} S={winner.rsi_s[0]:.0f}-{winner.rsi_s[1]:.0f}")
        print(f"#  SL/TP:         {winner.sl_m:.2f}x / {winner.tp_m:.2f}x (R:R {winner.tp_m/winner.sl_m:.1f}:1)")
        print(f"#  ADX:           {winner.adx:.0f}")
        print(f"#  Total:         {len(p1)+len(p2):,} combos em {tt:.0f}s")
        print(f"{'#'*140}\n")

        # Save results
        out = os.path.join(SCRIPT_DIR, "..", "download")
        os.makedirs(out, exist_ok=True)
        pd.DataFrame([r.to_dict() for r in all_res]).sort_values("score", ascending=False).to_csv(
            os.path.join(out, "grid_15m_results.csv"), index=False)
        wdata = {
            "version": "15m-v1", "timeframe": "15m", "symbol": SYMBOL, "score": winner.score,
            "params": {
                "allow_transition": winner.allow_trn, "require_pullback": winner.req_pb,
                "require_ema_trend": winner.req_et, "require_slope": winner.req_sl,
                "mr_enabled": winner.mr, "mr_rsi_long_max": winner.mr_rl,
                "mr_rsi_short_min": winner.mr_rs, "mr_bb_band_pct": winner.mr_bb,
                "rsi_long_min": winner.rsi_l[0], "rsi_long_max": winner.rsi_l[1],
                "rsi_short_min": winner.rsi_s[0], "rsi_short_max": winner.rsi_s[1],
                "adx_min": winner.adx, "fib_tolerance_pct": winner.fib_tol,
                "sl_atr_mult": winner.sl_m, "tp_atr_mult": winner.tp_m,
                "volume_confirm": winner.vol_c, "volume_sma_ratio": winner.vol_r,
                "atr_pct_min": winner.atr_min, "atr_pct_max": winner.atr_max,
            },
            "metrics": {
                "total_trades": winner.total_trades,
                "trades_per_day": round(winner.total_trades/365, 3),
                "win_rate": winner.win_rate, "profit_factor": winner.profit_factor,
                "total_pnl_pct": winner.total_pnl_pct, "max_drawdown_pct": winner.max_drawdown_pct,
                "sharpe_ratio": winner.sharpe_ratio, "buy_hold_pct": winner.buy_hold_pct,
                "avg_win_pct": winner.avg_win_pct, "avg_loss_pct": winner.avg_loss_pct,
            },
            "costs": {"fee_pct": FEE, "spread_bps": SPREAD, "slippage_bps": SLIPPAGE},
            "grid": {"p1": len(p1), "p2": len(p2), "total": len(p1)+len(p2), "time_sec": round(tt, 1)},
        }
        with open(os.path.join(out, "winner_15m.json"), "w") as f:
            json.dump(wdata, f, indent=2)
        print(f"Saved: winner_15m.json + grid_15m_results.csv")


if __name__ == "__main__":
    main()
