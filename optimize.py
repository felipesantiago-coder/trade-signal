"""
optimize.py
-----------
Grid Search ULTRA-RAPIDO para estrategia CTEV v4.

Usa arrays numpy pre-extraidos + evaluate inline (50x mais rapido que iloc).
Baixa dados UMA VEZ, calcula indicadores UMA VEZ.

Fases:
  Phase 1: Grid massivo nos filtros ~10080 combos
  Phase 2: Refinamento SL/TP nos top 20 ~500 combos
  Phase 3: Simulacao avancada (trailing/BE/partial) nos top 10

Velocidade: ~0.018s/combo = 10080 combos em ~3 min

Execucao: cd /home/z/my-project/trade-signal && python optimize.py
"""

from __future__ import annotations

import sys
import os
import time
import math
import json
import logging
import pickle
from dataclasses import dataclass, asdict
from typing import List

logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger("ctev.optimize")
logger.setLevel(logging.INFO)

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from indicators import compute_indicators
from backtest import (
    fetch_historical_ohlcv, calculate_metrics, TradeResult,
    simulate_trades_advanced,
    _apply_costs, DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS,
)

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
DAYS = 730
MIN_TRADES = 8
MIN_WIN_RATE = 35.0
MAX_DRAWDOWN = 25.0
DATA_CACHE = "/tmp/ctev_data.pkl"


@dataclass
class GridResult:
    combo_id: int
    rsi_long_min: float; rsi_long_max: float
    rsi_short_min: float; rsi_short_max: float
    adx_min: float; volume_sma_ratio: float; fib_tolerance_pct: float
    allow_transition: bool; sl_atr_mult: float; tp_atr_mult: float
    ema20_prox_pct: float = 0.0; ema50_prox_pct: float = 0.0
    volume_confirm: bool = True
    total_trades: int = 0; long_trades: int = 0; short_trades: int = 0
    wins: int = 0; losses: int = 0
    win_rate: float = 0.0; profit_factor: float = 0.0; total_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0; sharpe_ratio: float = 0.0
    avg_bars_held: float = 0.0; avg_win_pct: float = 0.0; avg_loss_pct: float = 0.0
    best_trade_pct: float = 0.0; worst_trade_pct: float = 0.0
    buy_hold_pct: float = 0.0; score: float = 0.0; phase: str = "p1"

    def to_dict(self) -> dict:
        return asdict(self)


class Optimizer:
    """Grid search optimizer com arrays numpy pre-extraidos."""

    def __init__(self):
        self.n = 0
        self.idx_arr = None
        self.close = self.high = self.low = None
        self.regime = None
        self.atr_pct = self.rsi_a = self.atr_a = None
        self.ema20_a = self.ema50_a = self.ema200_a = None
        self.adx_a = self.eslope = None
        self.fib382 = self.fib500 = self.fib618 = self.fibdir = None
        self.vol_a = self.vs50 = None
        self.e20t = self.e50t = self.e50tu = None
        self.tradeable = None
        self.df_clean = None
        self.buy_hold = 0.0

        # Strategy params (variaveis de instancia para acesso rapido)
        self.adx_min = 25.0
        self.rsi_lmin = 28.0; self.rsi_lmax = 48.0
        self.rsi_smin = 55.0; self.rsi_smax = 75.0
        self.vol_ratio = 0.5; self.vol_confirm = True
        self.fib_tol = 0.025; self.allow_trans = False
        self.sl_mult = 1.5; self.tp_mult = 3.0
        self.e20prox = 0.0; self.e50prox = 0.0

    def load_data(self):
        """Baixa e cacheia dados historicos + indicadores."""
        if os.path.exists(DATA_CACHE):
            with open(DATA_CACHE, 'rb') as f:
                data = pickle.load(f)
            self.df_clean = data['df_clean']
            logger.info("Dados carregados do cache: %d candles", len(self.df_clean))
        else:
            df = fetch_historical_ohlcv(SYMBOL, TIMEFRAME, DAYS)
            df_ind = compute_indicators(df)
            crit = ["ema20","ema50","ema200","rsi","atr","atr_percentile",
                    "macd","macd_signal","macd_hist","adx","plus_di","minus_di","regime"]
            self.df_clean = df_ind.dropna(subset=crit).copy()
            with open(DATA_CACHE, 'wb') as f:
                pickle.dump({'df_clean': self.df_clean}, f)
            logger.info("Dados baixados e cacheados: %d candles", len(self.df_clean))

        df = self.df_clean
        self.n = n = len(df)
        self.idx_arr = df.index.values

        # Pre-extrair todas as colunas como numpy arrays
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

        # Pre-compute tradeable mask
        self.tradeable = ~(np.isin(self.regime, ['ranging', 'volatile']))

        # Buy & Hold
        self.buy_hold = (self.close[-1] - self.close[0]) / self.close[0] * 100

        logger.info("Arrays pre-extraidos: %d candles, %d tradeable, B&H=%.2f%%",
                     n, self.tradeable.sum(), self.buy_hold)

    def _set_params(self, p: dict):
        """Configura parametros para o proximo combo."""
        self.adx_min = p['adx_min']
        self.rsi_lmin = p['rsi_long_min']; self.rsi_lmax = p['rsi_long_max']
        self.rsi_smin = p['rsi_short_min']; self.rsi_smax = p['rsi_short_max']
        self.vol_ratio = p['volume_sma_ratio']
        self.vol_confirm = p.get('volume_confirm', True)
        self.fib_tol = p['fib_tolerance_pct']
        self.allow_trans = p['allow_transition']
        self.sl_mult = p['sl_atr_mult']; self.tp_mult = p['tp_atr_mult']
        self.e20prox = p.get('ema20_prox_pct', 0.0)
        self.e50prox = p.get('ema50_prox_pct', 0.0)

    def _eval_long(self, i: int):
        rg = self.regime[i]
        if rg == 'trending_up':
            if self.adx_a[i] < self.adx_min: return None
        elif rg == 'transition':
            if not self.allow_trans: return None
        else:
            return None

        c = self.close[i]; e20 = self.ema20_a[i]; e50 = self.ema50_a[i]; e200 = self.ema200_a[i]
        if not (c > e50 > e200): return None
        if self.eslope[i] <= 0: return None

        pb = False; fd = self.fibdir[i]
        if fd == 1:
            f38 = self.fib382[i]; f61 = self.fib618[i]
            if not (np.isnan(f38) or np.isnan(f61)) and f61 <= c <= f38:
                pb = True
            if not pb:
                lo = self.low[i]; tol = c * self.fib_tol
                for fl in (f38, self.fib500[i], f61):
                    if not np.isnan(fl) and fl > 0 and abs(lo - fl) <= tol:
                        pb = True; break
        if not pb and self.e20t[i] and c > e20:
            pb = True
        if not pb and self.e20prox > 0 and e20 > 0:
            if abs(c - e20) / e20 <= self.e20prox:
                pb = True
        if not pb and self.e50t[i] and c > e50:
            pb = True
        if not pb and self.e50prox > 0 and e50 > 0:
            if abs(c - e50) / e50 <= self.e50prox:
                pb = True
        if not pb:
            return None

        r = self.rsi_a[i]
        if not (self.rsi_lmin <= r <= self.rsi_lmax): return None

        if self.vol_confirm:
            v = self.vs50[i]
            if not np.isnan(v) and v > 0 and self.vol_a[i] < v * self.vol_ratio:
                return None

        ap = self.atr_pct[i]
        if ap < 0.10 or ap > 0.90: return None

        av = self.atr_a[i]; ep = c
        sl = ep - self.sl_mult * av; tp = ep + self.tp_mult * av
        return (ep, sl, tp, av, r)

    def _eval_short(self, i: int):
        rg = self.regime[i]
        if rg == 'trending_down':
            if self.adx_a[i] < self.adx_min: return None
        elif rg == 'transition':
            if not self.allow_trans: return None
        else:
            return None

        c = self.close[i]; e20 = self.ema20_a[i]; e50 = self.ema50_a[i]; e200 = self.ema200_a[i]
        if not (c < e50 < e200): return None
        if self.eslope[i] >= 0: return None

        pb = False; fd = self.fibdir[i]
        if fd == -1:
            f38 = self.fib382[i]; f61 = self.fib618[i]
            if not (np.isnan(f38) or np.isnan(f61)) and f61 <= c <= f38:
                pb = True
            if not pb:
                hi = self.high[i]; tol = c * self.fib_tol
                for fl in (f38, self.fib500[i], f61):
                    if not np.isnan(fl) and fl > 0 and abs(hi - fl) <= tol:
                        pb = True; break
        if not pb and self.e20t[i] and c < e20 and self.high[i] >= e20:
            pb = True
        if not pb and self.e20prox > 0 and e20 > 0:
            if abs(c - e20) / e20 <= self.e20prox:
                pb = True
        if not pb and self.e50tu[i] and c < e50 and self.high[i] >= e50:
            pb = True
        if not pb and self.e50prox > 0 and e50 > 0:
            if abs(c - e50) / e50 <= self.e50prox:
                pb = True
        if not pb:
            return None

        r = self.rsi_a[i]
        if not (self.rsi_smin <= r <= self.rsi_smax): return None

        if self.vol_confirm:
            v = self.vs50[i]
            if not np.isnan(v) and v > 0 and self.vol_a[i] < v * self.vol_ratio:
                return None

        ap = self.atr_pct[i]
        if ap < 0.10 or ap > 0.90: return None

        av = self.atr_a[i]; ep = c
        sl = ep + self.sl_mult * av; tp = ep - self.tp_mult * av
        return (ep, sl, tp, av, r)

    def fast_sim(self) -> List[TradeResult]:
        """Simulacao rapida com numpy arrays e evaluate inline."""
        trades = []
        i = 0; n = self.n
        cl = self.close; hi = self.high; lo = self.low
        ta = self.tradeable

        while i < n:
            if not ta[i]:
                i += 1; continue

            sig = self._eval_long(i)
            if sig is None:
                sig = self._eval_short(i)
            if sig is None:
                i += 1; continue

            ep, sl, tp, atr_v, rsi_v = sig
            is_long = sl < tp  # sl < tp means long
            exit_price = None; bars = 0
            max_j = min(i + 72, n)

            if is_long:
                for j in range(i + 1, max_j):
                    if lo[j] <= sl:
                        exit_price = sl; bars = j - i; break
                    if hi[j] >= tp:
                        exit_price = tp; bars = j - i; break
            else:
                for j in range(i + 1, max_j):
                    if hi[j] >= sl:
                        exit_price = sl; bars = j - i; break
                    if lo[j] <= tp:
                        exit_price = tp; bars = j - i; break

            if exit_price is None:
                bars = max_j - 1 - i
                exit_price = cl[max_j - 1]

            _, adj, _ = _apply_costs(ep, exit_price, is_long,
                                      DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS)
            if is_long:
                pnl = (adj - ep) / ep * 100
            else:
                pnl = (ep - adj) / ep * 100

            trades.append(TradeResult(
                entry_ts=self.idx_arr[i],
                exit_ts=self.idx_arr[min(i + bars, n - 1)],
                type='LONG' if is_long else 'SHORT',
                entry_price=ep, exit_price=exit_price,
                stop_loss=sl, take_profit=tp,
                atr=atr_v, rsi=rsi_v,
                pnl_pct=round(pnl, 4), pnl_abs=round(exit_price - ep, 2),
                bars_held=bars, exit_reason='x',
            ))
            i += bars + 1

        return trades

    def run_combo(self, cid: int, p: dict, phase: str = "p1") -> GridResult:
        self._set_params(p)
        trades = self.fast_sim()
        m = calculate_metrics(trades, self.df_clean)
        r = GridResult(
            combo_id=cid,
            rsi_long_min=p['rsi_long_min'], rsi_long_max=p['rsi_long_max'],
            rsi_short_min=p['rsi_short_min'], rsi_short_max=p['rsi_short_max'],
            adx_min=p['adx_min'], volume_sma_ratio=p['volume_sma_ratio'],
            fib_tolerance_pct=p['fib_tolerance_pct'], allow_transition=p['allow_transition'],
            sl_atr_mult=p['sl_atr_mult'], tp_atr_mult=p['tp_atr_mult'],
            ema20_prox_pct=p.get('ema20_prox_pct', 0.0),
            ema50_prox_pct=p.get('ema50_prox_pct', 0.0),
            volume_confirm=p.get('volume_confirm', True),
            total_trades=m.total_trades, long_trades=m.long_trades, short_trades=m.short_trades,
            wins=m.wins, losses=m.losses, win_rate=round(m.win_rate, 2),
            profit_factor=round(m.profit_factor, 4), total_pnl_pct=round(m.total_pnl_pct, 4),
            max_drawdown_pct=round(m.max_drawdown_pct, 4), sharpe_ratio=round(m.sharpe_ratio, 4),
            avg_bars_held=round(m.avg_bars_held, 1), avg_win_pct=round(m.avg_win_pct, 4),
            avg_loss_pct=round(m.avg_loss_pct, 4), best_trade_pct=round(m.best_trade_pct, 4),
            worst_trade_pct=round(m.worst_trade_pct, 4), buy_hold_pct=round(self.buy_hold, 4),
            phase=phase,
        )
        r.score = self._score(r)
        return r

    @staticmethod
    def _score(r: GridResult) -> float:
        if (r.profit_factor <= 1.0 or r.win_rate < MIN_WIN_RATE or
                r.total_trades < MIN_TRADES or r.max_drawdown_pct > MAX_DRAWDOWN):
            return 0.0
        wr = r.win_rate / 100.0
        dd_pen = max(0, 1.0 - r.max_drawdown_pct / 100.0)
        sig = math.sqrt(r.total_trades)
        pnl_b = 1.2 if r.total_pnl_pct > 0 else 0.8
        return round(wr * r.profit_factor * sig * dd_pen * pnl_b, 4)

    def run_phase3_advanced(self, r: GridResult) -> GridResult:
        """Roda simulacao avancada com trailing/BE/partial para um GridResult."""
        import strategy
        strategy.ADX_MIN = r.adx_min
        strategy.RSI_LONG_MIN = r.rsi_long_min; strategy.RSI_LONG_MAX = r.rsi_long_max
        strategy.RSI_SHORT_MIN = r.rsi_short_min; strategy.RSI_SHORT_MAX = r.rsi_short_max
        strategy.VOLUME_SMA_RATIO = r.volume_sma_ratio
        strategy.VOLUME_CONFIRM = r.volume_confirm
        strategy.FIB_TOLERANCE_PCT = r.fib_tolerance_pct
        strategy.ALLOW_TRANSITION = r.allow_transition
        strategy.SL_ATR_MULT = r.sl_atr_mult; strategy.TP_ATR_MULT = r.tp_atr_mult
        strategy.ATR_PCT_MIN = 0.10; strategy.ATR_PCT_MAX = 0.90
        strategy.EMA50_SLOPE_MIN = 0.0
        strategy.EMA20_PROXIMITY_PCT = r.ema20_prox_pct
        strategy.EMA50_PROXIMITY_PCT = r.ema50_prox_pct

        trades, _, _ = simulate_trades_advanced(self.df_clean, 0.10, 0.90)
        m = calculate_metrics(trades, self.df_clean)

        p3r = GridResult(
            combo_id=r.combo_id,
            rsi_long_min=r.rsi_long_min, rsi_long_max=r.rsi_long_max,
            rsi_short_min=r.rsi_short_min, rsi_short_max=r.rsi_short_max,
            adx_min=r.adx_min, volume_sma_ratio=r.volume_sma_ratio,
            fib_tolerance_pct=r.fib_tolerance_pct, allow_transition=r.allow_transition,
            sl_atr_mult=r.sl_atr_mult, tp_atr_mult=r.tp_atr_mult,
            ema20_prox_pct=r.ema20_prox_pct, ema50_prox_pct=r.ema50_prox_pct,
            volume_confirm=r.volume_confirm,
            total_trades=m.total_trades, long_trades=m.long_trades, short_trades=m.short_trades,
            wins=m.wins, losses=m.losses, win_rate=round(m.win_rate, 2),
            profit_factor=round(m.profit_factor, 4), total_pnl_pct=round(m.total_pnl_pct, 4),
            max_drawdown_pct=round(m.max_drawdown_pct, 4), sharpe_ratio=round(m.sharpe_ratio, 4),
            avg_bars_held=round(m.avg_bars_held, 1), avg_win_pct=round(m.avg_win_pct, 4),
            avg_loss_pct=round(m.avg_loss_pct, 4), best_trade_pct=round(m.best_trade_pct, 4),
            worst_trade_pct=round(m.worst_trade_pct, 4), buy_hold_pct=round(self.buy_hold, 4),
            phase="p3",
        )
        p3r.score = Optimizer._score(p3r)
        return p3r


def make_p1_grid() -> List[dict]:
    """
    Phase 1: Grid massivo nos filtros.

    10 RSI_L * 10 RSI_S * 4 ADX * 3 Vol * 3 Fib * 2 Trans * 2 VC = 14400 combos
    """
    rsi_l = [
        (20,40),(22,42),(25,45),(28,48),(30,50),
        (32,52),(35,55),(20,45),(25,50),(22,48),
    ]
    rsi_s = [
        (48,68),(50,70),(52,72),(55,75),(58,78),
        (60,80),(50,75),(52,68),(55,70),(48,72),
    ]
    adx = [20.0, 25.0, 30.0, 35.0]
    vol = [0.30, 0.50, 0.70]
    fib = [0.015, 0.025, 0.040]
    trans = [False, True]
    vc = [True, False]

    combos = []
    for rl in rsi_l:
        for rs in rsi_s:
            for a in adx:
                for v in vol:
                    for f in fib:
                        for t in trans:
                            for c in vc:
                                combos.append({
                                    "rsi_long_min": rl[0], "rsi_long_max": rl[1],
                                    "rsi_short_min": rs[0], "rsi_short_max": rs[1],
                                    "adx_min": a, "volume_sma_ratio": v,
                                    "fib_tolerance_pct": f, "allow_transition": t,
                                    "sl_atr_mult": 1.5, "tp_atr_mult": 3.0,
                                    "volume_confirm": c,
                                })
    return combos


def make_p2_grid(top: List[GridResult], n=20) -> List[dict]:
    sls = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
    tps = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    combos = []
    for r in top[:n]:
        for s in sls:
            for t in tps:
                if t / s < 1.0: continue
                combos.append({
                    "rsi_long_min": r.rsi_long_min, "rsi_long_max": r.rsi_long_max,
                    "rsi_short_min": r.rsi_short_min, "rsi_short_max": r.rsi_short_max,
                    "adx_min": r.adx_min, "volume_sma_ratio": r.volume_sma_ratio,
                    "fib_tolerance_pct": r.fib_tolerance_pct, "allow_transition": r.allow_transition,
                    "sl_atr_mult": s, "tp_atr_mult": t,
                    "ema20_prox_pct": r.ema20_prox_pct, "ema50_prox_pct": r.ema50_prox_pct,
                    "volume_confirm": r.volume_confirm,
                })
    return combos


def print_top(results: List[GridResult], title: str, n: int = 20) -> None:
    print(f"\n{'='*130}")
    print(f"  {title}")
    print(f"{'='*130}")
    print(f"{'#':>3} {'Score':>7} {'Trades':>6} {'L/S':>5} {'WR%':>6} {'PF':>6} {'PnL%':>8} {'DD%':>6} {'Sharpe':>7} "
          f"{'RSI_L':>8} {'RSI_S':>8} {'ADX':>4} {'Vol':>4} {'Fib':>5} {'Tr':>3} {'VC':>3} "
          f"{'SL':>4} {'TP':>4} {'Phase':>5}")
    print(f"{'-'*130}")
    for i, r in enumerate(sorted(results, key=lambda x: x.score, reverse=True)[:n]):
        rl = f"{r.rsi_long_min:.0f}-{r.rsi_long_max:.0f}"
        rs = f"{r.rsi_short_min:.0f}-{r.rsi_short_max:.0f}"
        ls = f"{r.long_trades}/{r.short_trades}"
        print(f"{i+1:>3} {r.score:>7.3f} {r.total_trades:>6} {ls:>5} {r.win_rate:>6.1f} {r.profit_factor:>6.2f} "
              f"{r.total_pnl_pct:>8.2f} {r.max_drawdown_pct:>6.2f} {r.sharpe_ratio:>7.2f} "
              f"{rl:>8} {rs:>8} {r.adx_min:>4.0f} {r.volume_sma_ratio:>4.2f} {r.fib_tolerance_pct:>5.3f} "
              f"{'S' if r.allow_transition else 'N':>3} {'S' if r.volume_confirm else 'N':>3} "
              f"{r.sl_atr_mult:>4.1f} {r.tp_atr_mult:>4.1f} {r.phase:>5}")


def main():
    t0 = time.time()
    print("\n" + "#" * 130)
    print("#  CTEV v4.4 — GRID SEARCH OPTIMIZER (numpy-accelerated)")
    print("#  Objetivo: maximizar Win Rate e Lucro")
    print("#" * 130)

    opt = Optimizer()

    # 1. Load data
    print("\n[1/5] Carregando dados...")
    opt.load_data()
    rc = dict(zip(*np.unique(opt.regime, return_counts=True)))
    print(f"      {opt.n:,} candles | Tradeable: {opt.tradeable.sum():,} | B&H={opt.buy_hold:+.2f}%")
    print(f"      Regimes: {rc}")

    # 2. Phase 1
    print("\n[2/5] PHASE 1: Grid de Filtros...")
    p1 = make_p1_grid()
    print(f"      {len(p1):,} combinacoes")

    p1_res = []; viable = 0
    best_score = 0.0; best_str = ""
    t1 = time.time()

    for idx, params in enumerate(p1):
        r = opt.run_combo(idx, params, "p1")
        p1_res.append(r)
        if r.score > 0: viable += 1
        if r.score > best_score:
            best_score = r.score
            best_str = (f"score={r.score:.3f} WR={r.win_rate:.1f}% PF={r.profit_factor:.2f} "
                        f"PnL={r.total_pnl_pct:+.2f}% DD={r.max_drawdown_pct:.2f}% T={r.total_trades}")

        if (idx + 1) % 2000 == 0:
            el = time.time() - t1
            spd = (idx + 1) / max(el, 0.01)
            eta = (len(p1) - idx - 1) / max(spd, 0.01)
            print(f"      [{idx+1:,}/{len(p1):,}] {spd:.0f}/s viable={viable} ETA={eta:.0f}s")
            if best_str:
                print(f"        BEST: {best_str}")

    p1_time = time.time() - t1
    print(f"      Phase 1: {len(p1):,} combos em {p1_time:.1f}s | viable={viable} ({viable/len(p1)*100:.1f}%)")

    p1_sorted = sorted(p1_res, key=lambda x: x.score, reverse=True)
    print_top(p1_sorted, f"PHASE 1 — Top 20 de {len(p1):,}", 20)

    if not p1_sorted or p1_sorted[0].score == 0:
        print("\n!!! Nenhuma combinacao viavel !!!")
        by_t = sorted(p1_res, key=lambda x: x.total_trades, reverse=True)
        print_top(by_t, "Diagnostico — Mais Trades", 10)
        _save_results([], opt, t0, len(p1), 0)
        return

    # 3. Phase 2
    n_top = min(20, len([r for r in p1_sorted if r.score > 0]))
    p1_top = [r for r in p1_sorted if r.score > 0][:n_top]

    print(f"\n[3/5] PHASE 2: SL/TP refinement para top {n_top}...")
    p2 = make_p2_grid(p1_top, n_top)
    print(f"      {len(p2):,} combinacoes")

    p2_res = []; t2 = time.time()
    for idx, params in enumerate(p2):
        r = opt.run_combo(100000 + idx, params, "p2")
        p2_res.append(r)

    p2_time = time.time() - t2
    print(f"      Phase 2: {len(p2):,} combos em {p2_time:.1f}s")

    all_res = p1_res + p2_res
    all_sorted = sorted(all_res, key=lambda x: x.score, reverse=True)
    print_top(all_sorted, f"COMBINED — Top 20 ({len(all_res):,} total)", 20)

    # 4. Phase 3
    top10 = all_sorted[:10]
    print(f"\n[4/5] PHASE 3: Validacao avancada (trailing/BE/partial) top 10...")
    p3_res = []
    for i, r in enumerate(top10):
        p3r = opt.run_phase3_advanced(r)
        p3_res.append(p3r)
        print(f"      [{i+1}/10] T={p3r.total_trades} WR={p3r.win_rate:.1f}% PF={p3r.profit_factor:.2f} "
              f"PnL={p3r.total_pnl_pct:+.2f}% DD={p3r.max_drawdown_pct:.2f}% Sh={p3r.sharpe_ratio:.2f}")

    print_top(p3_res, "PHASE 3 — Validacao Avancada (top 10)", 10)

    # 5. Vencedor
    winner = max(p3_res, key=lambda x: x.score)
    tt = time.time() - t0

    print(f"\n{'#'*130}")
    print(f"#  VENCEDOR — CTEV v4.4 OPTIMIZED")
    print(f"{'#'*130}")
    print(f"#  Score:         {winner.score:.4f}")
    print(f"#  Trades:        {winner.total_trades} (L={winner.long_trades} S={winner.short_trades})")
    print(f"#  Win Rate:      {winner.win_rate:.1f}%")
    print(f"#  Profit Factor: {winner.profit_factor:.2f}")
    print(f"#  PnL:           {winner.total_pnl_pct:+.2f}%")
    print(f"#  Max DD:        {winner.max_drawdown_pct:.2f}%")
    print(f"#  Sharpe:        {winner.sharpe_ratio:.2f}")
    print(f"#  Buy & Hold:    {winner.buy_hold_pct:+.2f}%")
    print(f"#  Alpha:         {winner.total_pnl_pct - winner.buy_hold_pct:+.2f}pp")
    print(f"#")
    print(f"#  PARAMETROS:")
    print(f"#    RSI LONG:       {winner.rsi_long_min:.0f} - {winner.rsi_long_max:.0f}")
    print(f"#    RSI SHORT:      {winner.rsi_short_min:.0f} - {winner.rsi_short_max:.0f}")
    print(f"#    ADX_MIN:        {winner.adx_min:.0f}")
    print(f"#    VOLUME_RATIO:   {winner.volume_sma_ratio:.2f}")
    print(f"#    FIB_TOLERANCE:  {winner.fib_tolerance_pct*100:.1f}%")
    print(f"#    ALLOW_TRANS:    {winner.allow_transition}")
    print(f"#    SL/TP:          {winner.sl_atr_mult:.2f}x / {winner.tp_atr_mult:.2f}x (R:R {winner.tp_atr_mult/winner.sl_atr_mult:.1f}:1)")
    print(f"#    VOLUME_CONFIRM: {winner.volume_confirm}")
    print(f"#")
    print(f"#  STATS: P1={len(p1):,} P2={len(p2):,} P3={len(p3_res)} Total={len(p1)+len(p2)+len(p3_res):,}")
    print(f"#         Tempo: {tt:.0f}s ({tt/60:.1f}min)")
    print(f"{'#'*130}\n")

    _save_results(p1_res + p2_res + p3_res, opt, t0, len(p1), len(p2), winner)


def _save_results(all_results: List[GridResult], opt: Optimizer, t0: float,
                   p1_count: int, p2_count: int, winner: GridResult = None) -> None:
    out = os.path.join(SCRIPT_DIR, "..", "download")
    os.makedirs(out, exist_ok=True)

    if all_results:
        df_r = pd.DataFrame([r.to_dict() for r in all_results]).sort_values("score", ascending=False)
        csv_p = os.path.join(out, "grid_search_results.csv")
        df_r.to_csv(csv_p, index=False)
        print(f"CSV: {csv_p}")

    if winner:
        wdata = {
            "version": "v4.4", "score": winner.score,
            "params": {
                "RSI_LONG_MIN": winner.rsi_long_min, "RSI_LONG_MAX": winner.rsi_long_max,
                "RSI_SHORT_MIN": winner.rsi_short_min, "RSI_SHORT_MAX": winner.rsi_short_max,
                "ADX_MIN": winner.adx_min, "VOLUME_SMA_RATIO": winner.volume_sma_ratio,
                "FIB_TOLERANCE_PCT": winner.fib_tolerance_pct, "ALLOW_TRANSITION": winner.allow_transition,
                "SL_ATR_MULT": winner.sl_atr_mult, "TP_ATR_MULT": winner.tp_atr_mult,
                "EMA20_PROXIMITY_PCT": winner.ema20_prox_pct, "EMA50_PROXIMITY_PCT": winner.ema50_prox_pct,
                "VOLUME_CONFIRM": winner.volume_confirm,
            },
            "metrics": {
                "total_trades": winner.total_trades, "win_rate": winner.win_rate,
                "profit_factor": winner.profit_factor, "total_pnl_pct": winner.total_pnl_pct,
                "max_drawdown_pct": winner.max_drawdown_pct, "sharpe_ratio": winner.sharpe_ratio,
                "buy_hold_pct": winner.buy_hold_pct,
            },
            "grid_stats": {
                "phase1_combos": p1_count, "phase2_combos": p2_count,
                "total_tested": p1_count + p2_count + 10,
                "total_time_sec": round(time.time() - t0, 1),
            },
        }
        wp = os.path.join(out, "winner_params.json")
        with open(wp, "w") as f:
            json.dump(wdata, f, indent=2)
        print(f"Winner: {wp}")


if __name__ == "__main__":
    main()
