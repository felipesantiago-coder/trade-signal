from __future__ import annotations
import sys, os, time, math, json, logging, pickle
from dataclasses import dataclass, asdict
from typing import List, Optional

logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger("ctev.optv5")
logger.setLevel(logging.INFO)

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from indicators import compute_indicators
from backtest import (fetch_historical_ohlcv, calculate_metrics, TradeResult,
    _apply_costs, DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS)

SYMBOL = "BTC/USDT"; TIMEFRAME = "1h"; DAYS = 730
DATA_CACHE = "/tmp/ctev_v5_data.pkl"
MIN_TRADES_P1 = 50; MIN_TRADES_P2 = 150
MIN_WIN_RATE = 50.0; MAX_DRAWDOWN = 25.0

@dataclass
class GridResult:
    combo_id: int
    allow_ranging: bool = False; allow_volatile: bool = False; allow_transition: bool = False
    require_ema_trend: bool = True; require_slope: bool = True; require_pullback: bool = True
    ema20_prox_pct: float = 0.0; ema50_prox_pct: float = 0.0; fib_tolerance_pct: float = 0.025
    mr_enabled: bool = False; mr_rsi_long_max: float = 35.0
    mr_rsi_short_min: float = 65.0; mr_bb_band_pct: float = 0.0
    rsi_long_min: float = 30.0; rsi_long_max: float = 70.0
    rsi_short_min: float = 30.0; rsi_short_max: float = 70.0
    volume_confirm: bool = False; volume_sma_ratio: float = 0.3
    adx_min: float = 0.0; atr_pct_min: float = 0.05; atr_pct_max: float = 0.95
    sl_atr_mult: float = 1.5; tp_atr_mult: float = 2.5
    total_trades: int = 0; long_trades: int = 0; short_trades: int = 0
    wins: int = 0; losses: int = 0; win_rate: float = 0.0
    profit_factor: float = 0.0; total_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0; sharpe_ratio: float = 0.0
    avg_bars_held: float = 0.0; avg_win_pct: float = 0.0; avg_loss_pct: float = 0.0
    best_trade_pct: float = 0.0; worst_trade_pct: float = 0.0
    buy_hold_pct: float = 0.0; score: float = 0.0; phase: str = "p1"
    def to_dict(self): return asdict(self)

class OptimizerV5:
    def __init__(self):
        self.n = 0
        self.close = self.high = self.low = None
        self.regime = self.atr_pct = self.rsi_a = self.atr_a = None
        self.ema20_a = self.ema50_a = self.ema200_a = self.adx_a = self.eslope = None
        self.fib382 = self.fib500 = self.fib618 = self.fibdir = None
        self.vol_a = self.vs50 = self.e20t = self.e50t = self.e50tu = None
        self.bb_lower = self.bb_upper = self.bb_width = None
        self.df_clean = None; self.buy_hold = 0.0; self.idx_arr = None

    def load_data(self):
        if os.path.exists(DATA_CACHE):
            with open(DATA_CACHE, 'rb') as f: self.df_clean = pickle.load(f)['df_clean']
        else:
            df = fetch_historical_ohlcv(SYMBOL, TIMEFRAME, DAYS)
            df_ind = compute_indicators(df, timeframe=TIMEFRAME)
            crit = ["ema20","ema50","ema200","rsi","atr","atr_percentile",
                    "macd","macd_signal","macd_hist","adx","plus_di","minus_di","regime"]
            self.df_clean = df_ind.dropna(subset=crit).copy()
            with open(DATA_CACHE, 'wb') as f: pickle.dump({'df_clean': self.df_clean}, f)
        df = self.df_clean; self.n = len(df); self.idx_arr = df.index.values
        self.close = df['close'].values.astype(np.float64)
        self.high = df['high'].values.astype(np.float64)
        self.low = df['low'].values.astype(np.float64)
        self.regime = df['regime'].values
        self.atr_pct = df['atr_percentile'].values.astype(np.float64)
        self.rsi_a = df['rsi'].values.astype(np.float64)
        self.atr_a = df['atr'].values.astype(np.float64)
        self.ema20_a = df['ema20'].values.astype(np.float64)
        self.ema50_a = df['ema50'].values.astype(np.float64)
        self.ema200_a = df['ema200'].values.astype(np.float64)
        self.adx_a = df['adx'].values.astype(np.float64)
        self.eslope = df['ema50_slope'].values.astype(np.float64)
        self.fib382 = df['fib_0382'].values.astype(np.float64)
        self.fib500 = df['fib_0500'].values.astype(np.float64)
        self.fib618 = df['fib_0618'].values.astype(np.float64)
        self.fibdir = df['fib_direction'].values.astype(np.int32)
        self.vol_a = df['volume'].values.astype(np.float64)
        self.vs50 = df['volume_sma50'].values.astype(np.float64)
        self.e20t = df['ema20_touched'].values
        self.e50t = df['ema50_touched'].values
        self.e50tu = df['ema50_touched_up'].values
        self.bb_lower = df['bb_lower'].values.astype(np.float64)
        self.bb_upper = df['bb_upper'].values.astype(np.float64)
        self.bb_width = df['bb_width'].values.astype(np.float64)
        self.buy_hold = (self.close[-1] - self.close[0]) / self.close[0] * 100
        rc = dict(zip(*np.unique(self.regime, return_counts=True)))
        logger.info("%d candles | B&H=%.2f%% | %s", self.n, self.buy_hold, rc)

    def _trend_long(self, i, p):
        rg = self.regime[i]
        if rg == 'trending_up':
            if self.adx_a[i] < p['adx_min']: return None
        elif rg == 'transition':
            if not p['allow_transition']: return None
        else: return None
        c = self.close[i]; e50 = self.ema50_a[i]; e200 = self.ema200_a[i]
        if p['require_ema_trend'] and not (c > e50 > e200): return None
        if p['require_slope'] and self.eslope[i] <= 0: return None
        if p['require_pullback']:
            pb = False; fd = self.fibdir[i]; e20 = self.ema20_a[i]
            if fd == 1:
                f38 = self.fib382[i]; f61 = self.fib618[i]
                if not (np.isnan(f38) or np.isnan(f61)) and f61 <= c <= f38: pb = True
                if not pb:
                    lo = self.low[i]; tol = c * p['fib_tolerance_pct']
                    for fl in (f38, self.fib500[i], f61):
                        if not np.isnan(fl) and fl > 0 and abs(lo - fl) <= tol: pb = True; break
            if not pb and self.e20t[i] and c > e20: pb = True
            if not pb and p['ema20_prox_pct'] > 0 and e20 > 0:
                if abs(c - e20) / e20 <= p['ema20_prox_pct']: pb = True
            if not pb and self.e50t[i] and c > e50: pb = True
            if not pb and p['ema50_prox_pct'] > 0 and e50 > 0:
                if abs(c - e50) / e50 <= p['ema50_prox_pct']: pb = True
            if not pb: return None
        r = self.rsi_a[i]
        if not (p['rsi_long_min'] <= r <= p['rsi_long_max']): return None
        if p['volume_confirm']:
            v = self.vs50[i]
            if not np.isnan(v) and v > 0 and self.vol_a[i] < v * p['volume_sma_ratio']: return None
        ap = self.atr_pct[i]
        if ap < p['atr_pct_min'] or ap > p['atr_pct_max']: return None
        av = self.atr_a[i]; ep = c
        sl = ep - p['sl_atr_mult'] * av; tp = ep + p['tp_atr_mult'] * av
        return (ep, sl, tp, av, r) if sl > 0 else None

    def _trend_short(self, i, p):
        rg = self.regime[i]
        if rg == 'trending_down':
            if self.adx_a[i] < p['adx_min']: return None
        elif rg == 'transition':
            if not p['allow_transition']: return None
        else: return None
        c = self.close[i]; e50 = self.ema50_a[i]; e200 = self.ema200_a[i]
        if p['require_ema_trend'] and not (c < e50 < e200): return None
        if p['require_slope'] and self.eslope[i] >= 0: return None
        if p['require_pullback']:
            pb = False; fd = self.fibdir[i]; e20 = self.ema20_a[i]
            if fd == -1:
                f38 = self.fib382[i]; f61 = self.fib618[i]
                if not (np.isnan(f38) or np.isnan(f61)) and f61 <= c <= f38: pb = True
                if not pb:
                    hi = self.high[i]; tol = c * p['fib_tolerance_pct']
                    for fl in (f38, self.fib500[i], f61):
                        if not np.isnan(fl) and fl > 0 and abs(hi - fl) <= tol: pb = True; break
            if not pb and self.e20t[i] and c < e20 and self.high[i] >= e20: pb = True
            if not pb and p['ema20_prox_pct'] > 0 and e20 > 0:
                if abs(c - e20) / e20 <= p['ema20_prox_pct']: pb = True
            if not pb and self.e50tu[i] and c < e50 and self.high[i] >= e50: pb = True
            if not pb and p['ema50_prox_pct'] > 0 and e50 > 0:
                if abs(c - e50) / e50 <= p['ema50_prox_pct']: pb = True
            if not pb: return None
        r = self.rsi_a[i]
        if not (p['rsi_short_min'] <= r <= p['rsi_short_max']): return None
        if p['volume_confirm']:
            v = self.vs50[i]
            if not np.isnan(v) and v > 0 and self.vol_a[i] < v * p['volume_sma_ratio']: return None
        ap = self.atr_pct[i]
        if ap < p['atr_pct_min'] or ap > p['atr_pct_max']: return None
        av = self.atr_a[i]; ep = c
        sl = ep + p['sl_atr_mult'] * av; tp = ep - p['tp_atr_mult'] * av
        return (ep, sl, tp, av, r)

    def _mr_long(self, i, p):
        if self.regime[i] != 'ranging': return None
        c = self.close[i]; lo = self.low[i]; r = self.rsi_a[i]
        if r > p['mr_rsi_long_max']: return None
        bbl = self.bb_lower[i]
        if p['mr_bb_band_pct'] > 0:
            if lo > bbl - c * p['mr_bb_band_pct']: return None
        else:
            if lo > bbl: return None
        ap = self.atr_pct[i]
        if ap < p['atr_pct_min'] or ap > p['atr_pct_max']: return None
        av = self.atr_a[i]; ep = c
        sl = ep - p['sl_atr_mult'] * av; tp = ep + p['tp_atr_mult'] * av
        return (ep, sl, tp, av, r) if sl > 0 else None

    def _mr_short(self, i, p):
        if self.regime[i] != 'ranging': return None
        c = self.close[i]; hi = self.high[i]; r = self.rsi_a[i]
        if r < p['mr_rsi_short_min']: return None
        bbu = self.bb_upper[i]
        if p['mr_bb_band_pct'] > 0:
            if hi < bbu + c * p['mr_bb_band_pct']: return None
        else:
            if hi < bbu: return None
        ap = self.atr_pct[i]
        if ap < p['atr_pct_min'] or ap > p['atr_pct_max']: return None
        av = self.atr_a[i]; ep = c
        sl = ep + p['sl_atr_mult'] * av; tp = ep - p['tp_atr_mult'] * av
        return (ep, sl, tp, av, r)

    def fast_sim(self, p):
        trades = []; i = 0; n = self.n
        cl = self.close; hi = self.high; lo = self.low
        mr_on = p.get('mr_enabled', False)
        allow_rng = p.get('allow_ranging', False)
        allow_vol = p.get('allow_volatile', False)
        while i < n:
            sig = None; is_long = True; rg = self.regime[i]
            if rg == 'ranging':
                if mr_on:
                    sig = self._mr_long(i, p)
                    if sig: is_long = True
                    else: sig = self._mr_short(i, p); is_long = False
                elif allow_rng:
                    sig = self._trend_long(i, p)
                    if sig: is_long = True
                    else: sig = self._trend_short(i, p); is_long = False
            elif rg == 'volatile' and allow_vol:
                sig = self._trend_long(i, p)
                if sig: is_long = True
                else: sig = self._trend_short(i, p); is_long = False
            else:
                sig = self._trend_long(i, p)
                if sig: is_long = True
                else: sig = self._trend_short(i, p); is_long = False
            if sig is None: i += 1; continue
            ep, sl, tp, atr_v, rsi_v = sig
            exit_price = None; bars = 0; max_j = min(i + 72, n)
            if is_long:
                for j in range(i + 1, max_j):
                    if lo[j] <= sl: exit_price = sl; bars = j - i; break
                    if hi[j] >= tp: exit_price = tp; bars = j - i; break
            else:
                for j in range(i + 1, max_j):
                    if hi[j] >= sl: exit_price = sl; bars = j - i; break
                    if lo[j] <= tp: exit_price = tp; bars = j - i; break
            if exit_price is None:
                bars = max_j - 1 - i; exit_price = cl[max_j - 1]
            _, adj, _ = _apply_costs(ep, exit_price, is_long, DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS)
            pnl = (adj - ep) / ep * 100 if is_long else (ep - adj) / ep * 100
            trades.append(TradeResult(
                entry_ts=self.idx_arr[i], exit_ts=self.idx_arr[min(i + bars, n - 1)],
                type='LONG' if is_long else 'SHORT',
                entry_price=ep, exit_price=exit_price, stop_loss=sl, take_profit=tp,
                atr=atr_v, rsi=rsi_v, pnl_pct=round(pnl, 4), pnl_abs=round(exit_price - ep, 2),
                bars_held=bars, exit_reason='x'))
            i += bars + 1
        return trades

    def run_combo(self, cid, p, phase="p1"):
        trades = self.fast_sim(p)
        m = calculate_metrics(trades, self.df_clean)
        r = GridResult(
            combo_id=cid, allow_ranging=p.get('allow_ranging', False),
            allow_volatile=p.get('allow_volatile', False), allow_transition=p.get('allow_transition', False),
            require_ema_trend=p.get('require_ema_trend', True), require_slope=p.get('require_slope', True),
            require_pullback=p.get('require_pullback', True),
            ema20_prox_pct=p.get('ema20_prox_pct', 0.0), ema50_prox_pct=p.get('ema50_prox_pct', 0.0),
            fib_tolerance_pct=p.get('fib_tolerance_pct', 0.025),
            mr_enabled=p.get('mr_enabled', False), mr_rsi_long_max=p.get('mr_rsi_long_max', 35.0),
            mr_rsi_short_min=p.get('mr_rsi_short_min', 65.0), mr_bb_band_pct=p.get('mr_bb_band_pct', 0.0),
            rsi_long_min=p.get('rsi_long_min', 30.0), rsi_long_max=p.get('rsi_long_max', 70.0),
            rsi_short_min=p.get('rsi_short_min', 30.0), rsi_short_max=p.get('rsi_short_max', 70.0),
            volume_confirm=p.get('volume_confirm', False), volume_sma_ratio=p.get('volume_sma_ratio', 0.3),
            adx_min=p.get('adx_min', 0.0), atr_pct_min=p.get('atr_pct_min', 0.05),
            atr_pct_max=p.get('atr_pct_max', 0.95), sl_atr_mult=p.get('sl_atr_mult', 1.5),
            tp_atr_mult=p.get('tp_atr_mult', 2.5),
            total_trades=m.total_trades, long_trades=m.long_trades, short_trades=m.short_trades,
            wins=m.wins, losses=m.losses, win_rate=round(m.win_rate, 2),
            profit_factor=round(m.profit_factor, 4), total_pnl_pct=round(m.total_pnl_pct, 4),
            max_drawdown_pct=round(m.max_drawdown_pct, 4), sharpe_ratio=round(m.sharpe_ratio, 4),
            avg_bars_held=round(m.avg_bars_held, 1), avg_win_pct=round(m.avg_win_pct, 4),
            avg_loss_pct=round(m.avg_loss_pct, 4), best_trade_pct=round(m.best_trade_pct, 4),
            worst_trade_pct=round(m.worst_trade_pct, 4), buy_hold_pct=round(self.buy_hold, 4),
            phase=phase)
        r.score = self._score(r, phase)
        return r

    @staticmethod
    def _score(r, phase="p1"):
        min_t = MIN_TRADES_P1 if phase == "p1" else MIN_TRADES_P2
        if r.total_trades < min_t or r.profit_factor <= 1.0: return 0.0
        if r.win_rate < MIN_WIN_RATE or r.max_drawdown_pct > MAX_DRAWDOWN: return 0.0
        wr = r.win_rate / 100.0; pf = r.profit_factor
        dd_pen = max(0, 1.0 - r.max_drawdown_pct / 100.0)
        freq_bonus = math.log(max(r.total_trades, 1)) / math.log(365)
        pnl_b = 1.5 if r.total_pnl_pct > 5.0 else (1.2 if r.total_pnl_pct > 0 else 0.5)
        sig = min(math.sqrt(r.total_trades) / 10.0, 2.0)
        return round(wr * pf * freq_bonus * dd_pen * pnl_b * sig, 4)

def make_p1_grid():
    combos = []
    trend_cfgs = [
        (True,  True,  True,  False, 20, (25,55), (45,75)),
        (True,  True,  True,  False, 15, (25,50), (50,75)),
        (True,  False, True,  False, 15, (30,60), (40,70)),
        (True,  False, True,  True,  20, (28,52), (48,72)),
        (True,  False, False, False, 15, (30,65), (35,70)),
        (True,  True,  True,  False, 25, (28,48), (52,72)),
        (True,  False, False, False, 0,  (30,70), (30,70)),
        (True,  True,  True,  True,  20, (20,50), (50,80)),
        (True,  False, True,  False, 0,  (35,65), (35,65)),
        (True,  False, False, False, 15, (25,60), (40,75)),
    ]
    mr_rl = [30, 35, 40, 45]; mr_rs = [55, 60, 65, 70]; mr_bb = [0.0, 0.005, 0.01]
    mr_sltp = [(1.0,1.5),(1.0,2.0),(1.5,2.0),(1.5,2.5),(0.75,1.5),(0.75,1.25)]
    tr_sltp = [(1.0,2.5),(1.5,3.0),(1.25,2.5),(1.0,2.0),(1.5,2.5)]
    for tc in trend_cfgs:
        a_tr, r_pb, r_et, r_sl, adx, rsi_l, rsi_s = tc
        for mr_on in [True, False]:
            if mr_on:
                for mrl in mr_rl:
                    for mrs in mr_rs:
                        for mbb in mr_bb:
                            for sl, tp in mr_sltp:
                                combos.append({
                                    'allow_ranging': False, 'allow_volatile': False,
                                    'allow_transition': a_tr, 'require_pullback': r_pb,
                                    'require_ema_trend': r_et, 'require_slope': r_sl,
                                    'rsi_long_min': rsi_l[0], 'rsi_long_max': rsi_l[1],
                                    'rsi_short_min': rsi_s[0], 'rsi_short_max': rsi_s[1],
                                    'adx_min': adx, 'fib_tolerance_pct': 0.025,
                                    'ema20_prox_pct': 0.0, 'ema50_prox_pct': 0.0,
                                    'volume_confirm': False, 'volume_sma_ratio': 0.3,
                                    'atr_pct_min': 0.05, 'atr_pct_max': 0.95,
                                    'mr_enabled': True, 'mr_rsi_long_max': float(mrl),
                                    'mr_rsi_short_min': float(mrs), 'mr_bb_band_pct': mbb,
                                    'sl_atr_mult': sl, 'tp_atr_mult': tp})
            else:
                for sl, tp in tr_sltp:
                    combos.append({
                        'allow_ranging': False, 'allow_volatile': False,
                        'allow_transition': a_tr, 'require_pullback': r_pb,
                        'require_ema_trend': r_et, 'require_slope': r_sl,
                        'rsi_long_min': rsi_l[0], 'rsi_long_max': rsi_l[1],
                        'rsi_short_min': rsi_s[0], 'rsi_short_max': rsi_s[1],
                        'adx_min': adx, 'fib_tolerance_pct': 0.025,
                        'ema20_prox_pct': 0.0, 'ema50_prox_pct': 0.0,
                        'volume_confirm': False, 'volume_sma_ratio': 0.3,
                        'atr_pct_min': 0.05, 'atr_pct_max': 0.95,
                        'mr_enabled': False, 'mr_rsi_long_max': 35.0,
                        'mr_rsi_short_min': 65.0, 'mr_bb_band_pct': 0.0,
                        'sl_atr_mult': sl, 'tp_atr_mult': tp})
    return combos


def make_p2_grid(top, n=15):
    sls = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    tps = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    mr_rsi_l = [28, 32, 35, 38, 42, 45]
    mr_rsi_s = [58, 62, 65, 68, 72]
    mr_bb = [0.0, 0.003, 0.005, 0.008, 0.01, 0.015]
    combos = []
    for r in top[:n]:
        base = {'allow_ranging': r.allow_ranging, 'allow_volatile': r.allow_volatile,
                'allow_transition': r.allow_transition, 'require_pullback': r.require_pullback,
                'require_ema_trend': r.require_ema_trend, 'require_slope': r.require_slope,
                'rsi_long_min': r.rsi_long_min, 'rsi_long_max': r.rsi_long_max,
                'rsi_short_min': r.rsi_short_min, 'rsi_short_max': r.rsi_short_max,
                'adx_min': r.adx_min, 'fib_tolerance_pct': r.fib_tolerance_pct,
                'ema20_prox_pct': r.ema20_prox_pct, 'ema50_prox_pct': r.ema50_prox_pct,
                'volume_confirm': r.volume_confirm, 'volume_sma_ratio': r.volume_sma_ratio,
                'atr_pct_min': r.atr_pct_min, 'atr_pct_max': r.atr_pct_max}
        if r.mr_enabled:
            for sl in sls:
                for tp in tps:
                    if tp / sl < 0.8: continue
                    for mrl in mr_rsi_l:
                        for mrs in mr_rsi_s:
                            for mbb in mr_bb:
                                combos.append({**base, 'mr_enabled': True,
                                    'mr_rsi_long_max': float(mrl), 'mr_rsi_short_min': float(mrs),
                                    'mr_bb_band_pct': mbb, 'sl_atr_mult': sl, 'tp_atr_mult': tp})
        else:
            for sl in sls:
                for tp in tps:
                    if tp / sl < 0.8: continue
                    combos.append({**base, 'mr_enabled': False,
                        'mr_rsi_long_max': 35.0, 'mr_rsi_short_min': 65.0,
                        'mr_bb_band_pct': 0.0, 'sl_atr_mult': sl, 'tp_atr_mult': tp})
    return combos

def print_top(results, title, n=20):
    print(f"\n{'='*160}")
    print(f"  {title}")
    print(f"{'='*160}")
    print(f"{'#':>3} {'Score':>7} {'Trades':>6} {'L/S':>6} {'T/d':>5} {'WR%':>6} {'PF':>6} {'PnL%':>8} {'DD%':>6} {'Sharpe':>7} "
          f"{'Trn':>4} {'Pb':>3} {'ET':>3} {'MR':>3} {'MR_RL':>5} {'MR_RS':>5} "
          f"{'RSI_L':>8} {'RSI_S':>8} {'ADX':>4} {'SL':>4} {'TP':>4}")
    print(f"{'-'*160}")
    for i, r in enumerate(sorted(results, key=lambda x: x.score, reverse=True)[:n]):
        rl = f"{r.rsi_long_min:.0f}-{r.rsi_long_max:.0f}"
        rs = f"{r.rsi_short_min:.0f}-{r.rsi_short_max:.0f}"
        ls = f"{r.long_trades}/{r.short_trades}"
        td = r.total_trades / 730
        print(f"{i+1:>3} {r.score:>7.3f} {r.total_trades:>6} {ls:>6} {td:>5.2f} {r.win_rate:>6.1f} {r.profit_factor:>6.2f} "
              f"{r.total_pnl_pct:>8.2f} {r.max_drawdown_pct:>6.2f} {r.sharpe_ratio:>7.2f} "
              f"{'S' if r.allow_transition else 'N':>4} {'S' if r.require_pullback else 'N':>3} "
              f"{'S' if r.require_ema_trend else 'N':>3} {'S' if r.mr_enabled else 'N':>3} "
              f"{r.mr_rsi_long_max:>5.0f} {r.mr_rsi_short_min:>5.0f} "
              f"{rl:>8} {rs:>8} {r.adx_min:>4.0f} {r.sl_atr_mult:>4.2f} {r.tp_atr_mult:>4.2f}")


def main():
    t0 = time.time()
    print("\n" + "#" * 160)
    print("#  CTEV v5.0 — HIGH-FREQUENCY GRID SEARCH (Trend + Mean-Reversion)")
    print("#  Objetivo: 200+ trades, WR>50%, PF>1.0, DD<25%")
    print("#" * 160)

    opt = OptimizerV5()
    print("\n[1/4] Carregando dados...")
    opt.load_data()

    print("\n[2/4] PHASE 1: Grid massivo...")
    p1 = make_p1_grid()
    print(f"      {len(p1):,} combinacoes")
    p1_res = []; viable = 0; best_score = 0.0; best_str = ""
    t1 = time.time()

    for idx, params in enumerate(p1):
        r = opt.run_combo(idx, params, "p1")
        p1_res.append(r)
        if r.score > 0: viable += 1
        if r.score > best_score:
            best_score = r.score
            best_str = f"T={r.total_trades} WR={r.win_rate:.1f}% PF={r.profit_factor:.2f} PnL={r.total_pnl_pct:+.2f}% DD={r.max_drawdown_pct:.2f}% MR={'S' if r.mr_enabled else 'N'}"
        if (idx + 1) % 2000 == 0:
            el = time.time() - t1; spd = (idx + 1) / max(el, 0.01)
            eta = (len(p1) - idx - 1) / max(spd, 0.01)
            print(f"      [{idx+1:,}/{len(p1):,}] {spd:.0f}/s viable={viable} ETA={eta:.0f}s")
            if best_str: print(f"        BEST: {best_str}")

    p1_time = time.time() - t1
    print(f"      Phase 1: {len(p1):,} combos em {p1_time:.1f}s | viable={viable}")
    p1_sorted = sorted(p1_res, key=lambda x: x.score, reverse=True)
    print_top(p1_sorted, f"PHASE 1 Top 20 de {len(p1):,}", 20)

    if not p1_sorted or p1_sorted[0].score == 0:
        print("\n!!! Nenhuma viavel !!!")
        by_t = sorted(p1_res, key=lambda x: x.total_trades, reverse=True)
        for r in by_t[:10]:
            print(f"  T={r.total_trades} WR={r.win_rate:.1f}% PF={r.profit_factor:.2f} PnL={r.total_pnl_pct:+.2f}% MR={'S' if r.mr_enabled else 'N'}")
        return

    n_top = min(15, len([r for r in p1_sorted if r.score > 0]))
    p1_top = [r for r in p1_sorted if r.score > 0][:n_top]

    print(f"\n[3/4] PHASE 2: Refinamento SL/TP/MR para top {n_top}...")
    p2 = make_p2_grid(p1_top, n_top)
    print(f"      {len(p2):,} combinacoes")
    p2_res = []; t2 = time.time(); best_p2 = ""
    for idx, params in enumerate(p2):
        r = opt.run_combo(200000 + idx, params, "p2")
        p2_res.append(r)
        if r.score > best_score:
            best_score = r.score
            best_p2 = f"T={r.total_trades} WR={r.win_rate:.1f}% PF={r.profit_factor:.2f} PnL={r.total_pnl_pct:+.2f}% SL={r.sl_atr_mult:.2f} TP={r.tp_atr_mult:.2f}"
        if (idx + 1) % 5000 == 0:
            el = time.time() - t2; spd = (idx + 1) / max(el, 0.01)
            eta = (len(p2) - idx - 1) / max(spd, 0.01)
            print(f"      [{idx+1:,}/{len(p2):,}] {spd:.0f}/s ETA={eta:.0f}s")
            if best_p2: print(f"        BEST: {best_p2}")
    p2_time = time.time() - t2
    print(f"      Phase 2: {len(p2):,} combos em {p2_time:.1f}s")

    all_res = p1_res + p2_res
    all_sorted = sorted(all_res, key=lambda x: x.score, reverse=True)
    print_top(all_sorted, f"COMBINED Top 20 ({len(all_res):,} total)", 20)

    winner = all_sorted[0] if all_sorted else None
    tt = time.time() - t0
    if winner:
        print(f"\n{'#'*160}")
        print(f"#  VENCEDOR v5.0")
        print(f"{'#'*160}")
        print(f"#  Score:         {winner.score:.4f}")
        print(f"#  Trades:        {winner.total_trades} ({winner.total_trades/730:.2f}/dia) L={winner.long_trades} S={winner.short_trades}")
        print(f"#  Win Rate:      {winner.win_rate:.1f}%")
        print(f"#  Profit Factor: {winner.profit_factor:.2f}")
        print(f"#  PnL:           {winner.total_pnl_pct:+.2f}% (B&H={winner.buy_hold_pct:+.2f}%)")
        print(f"#  Max DD:        {winner.max_drawdown_pct:.2f}%  Sharpe: {winner.sharpe_ratio:.2f}")
        print(f"#  MR:            {winner.mr_enabled} | RSI_L_mr={winner.mr_rsi_long_max} RSI_S_mr={winner.mr_rsi_short_min} BB={winner.mr_bb_band_pct}")
        print(f"#  Trend:         transition={'S' if winner.allow_transition else 'N'} pullback={'S' if winner.require_pullback else 'N'} ema_trend={'S' if winner.require_ema_trend else 'N'} slope={'S' if winner.require_slope else 'N'}")
        print(f"#  RSI:           L={winner.rsi_long_min:.0f}-{winner.rsi_long_max:.0f} S={winner.rsi_short_min:.0f}-{winner.rsi_short_max:.0f}")
        print(f"#  SL/TP:         {winner.sl_atr_mult:.2f}x / {winner.tp_atr_mult:.2f}x (R:R {winner.tp_atr_mult/winner.sl_atr_mult:.1f}:1)")
        print(f"#  ADX:           {winner.adx_min:.0f} | Total: {len(p1)+len(p2):,} combos em {tt:.0f}s")
        print(f"{'#'*160}\n")
        _save(winner, all_res, opt, t0, len(p1), len(p2))


def _save(winner, all_res, opt, t0, p1c, p2c):
    out = os.path.join(SCRIPT_DIR, "..", "download")
    os.makedirs(out, exist_ok=True)
    if all_res:
        df_r = pd.DataFrame([r.to_dict() for r in all_res]).sort_values("score", ascending=False)
        df_r.to_csv(os.path.join(out, "grid_v5_results.csv"), index=False)
    if winner:
        wdata = {"version": "v5.0", "goal": "high_frequency", "score": winner.score,
            "params": {k: getattr(winner, k) for k in [
                'allow_ranging','allow_volatile','allow_transition','require_ema_trend',
                'require_slope','require_pullback','mr_enabled','mr_rsi_long_max',
                'mr_rsi_short_min','mr_bb_band_pct','rsi_long_min','rsi_long_max',
                'rsi_short_min','rsi_short_max','adx_min','fib_tolerance_pct',
                'ema20_prox_pct','ema50_prox_pct','sl_atr_mult','tp_atr_mult',
                'volume_confirm','volume_sma_ratio','atr_pct_min','atr_pct_max']},
            "metrics": {"total_trades": winner.total_trades,
                "trades_per_day": round(winner.total_trades/730, 3),
                "win_rate": winner.win_rate, "profit_factor": winner.profit_factor,
                "total_pnl_pct": winner.total_pnl_pct, "max_drawdown_pct": winner.max_drawdown_pct,
                "sharpe_ratio": winner.sharpe_ratio, "buy_hold_pct": winner.buy_hold_pct},
            "grid": {"p1": p1c, "p2": p2c, "total": p1c+p2c, "time_sec": round(time.time()-t0, 1)}}
        with open(os.path.join(out, "winner_v5.json"), "w") as f:
            json.dump(wdata, f, indent=2)
        print(f"Saved: winner_v5.json + grid_v5_results.csv")

if __name__ == "__main__":
    main()
