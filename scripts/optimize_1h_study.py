"""
optimize_1h_study.py
-------------------
Estudo abrangente de otimizacao para BTC/USDT 1H.

Objetivo: Encontrar estrategia que SUPERE o Buy & Hold em 2 anos.

Abordagens testadas:
  Phase 0: Analise exploratoria dos dados (regimes, volatilidade, caracteristicas)
  Phase 1: Grid massivo trend-following + mean-reversion (~20k combos)
  Phase 2: Refinamento SL/TP/MR para top 20 (~8k combos)
  Phase 3: Validacao walk-forward (6 janelas)
  Phase 4: Estudo de custos e sensibilidade

Scoring: Penaliza estrategias abaixo do Buy & Hold.
"""
from __future__ import annotations
import sys, os, time, math, json, logging, pickle
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger("ctev.1hstudy")
logger.setLevel(logging.INFO)

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)
from indicators import compute_indicators
from backtest import (
    fetch_historical_ohlcv, calculate_metrics, TradeResult,
    _apply_costs, DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS
)

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
DAYS = 730
DATA_CACHE = "/tmp/ctev_1h_study_data.pkl"


# ============================================================
# Data Structures
# ============================================================
@dataclass
class GridResult:
    combo_id: int
    phase: str = "p1"
    # Strategy config
    allow_ranging: bool = False
    allow_volatile: bool = False
    allow_transition: bool = False
    require_ema_trend: bool = True
    require_slope: bool = True
    require_pullback: bool = True
    ema20_prox_pct: float = 0.0
    ema50_prox_pct: float = 0.0
    fib_tolerance_pct: float = 0.025
    mr_enabled: bool = False
    mr_rsi_long_max: float = 35.0
    mr_rsi_short_min: float = 65.0
    mr_bb_band_pct: float = 0.0
    rsi_long_min: float = 30.0
    rsi_long_max: float = 70.0
    rsi_short_min: float = 30.0
    rsi_short_max: float = 70.0
    volume_confirm: bool = False
    volume_sma_ratio: float = 0.3
    adx_min: float = 0.0
    atr_pct_min: float = 0.05
    atr_pct_max: float = 0.95
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 2.5
    # Results
    total_trades: int = 0
    long_trades: int = 0
    short_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    avg_bars_held: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    buy_hold_pct: float = 0.0
    # Computed
    score: float = 0.0
    beats_bh: bool = False
    excess_return: float = 0.0  # pnl - buy_hold

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ============================================================
# Fast Backtester (numpy arrays for speed)
# ============================================================
class FastBacktester:
    """Backtester otimizado com arrays numpy para velocidade."""

    def __init__(self, df: pd.DataFrame, timeframe: str = "1h"):
        self.df = df
        self.timeframe = timeframe

        # Pre-compute indicators
        t0 = time.time()
        df_ind = compute_indicators(df, timeframe=timeframe)
        self.df_ind = df_ind

        # Drop NaN rows for clean arrays
        clean_cols = [
            "ema20", "ema50", "ema200", "rsi", "atr", "atr_percentile",
            "macd", "macd_signal", "macd_hist", "adx", "plus_di", "minus_di",
            "regime", "ema50_slope", "bb_lower", "bb_upper", "bb_middle",
            "bb_width", "bb_squeeze_pct", "volume", "volume_sma20", "volume_sma50",
            "fib_0382", "fib_0500", "fib_0618", "fib_direction", "fib_proximity",
            "ema20_touched", "ema50_touched", "ema50_touched_up",
        ]
        self.df_clean = df_ind.dropna(subset=clean_cols[:14]).copy()
        self.n = len(self.df_clean)  # USE CLEAN LENGTH for array indexing
        n_clean = self.n
        logger.info("Indicators computed in %.1fs. Clean rows: %d/%d", time.time() - t0, n_clean, len(df))

        # Extract numpy arrays for speed
        self.close = self.df_clean["close"].values.astype(np.float64)
        self.high = self.df_clean["high"].values.astype(np.float64)
        self.low = self.df_clean["low"].values.astype(np.float64)
        self.ema20_a = self.df_clean["ema20"].values.astype(np.float64)
        self.ema50_a = self.df_clean["ema50"].values.astype(np.float64)
        self.ema200_a = self.df_clean["ema200"].values.astype(np.float64)
        self.rsi_a = self.df_clean["rsi"].values.astype(np.float64)
        self.atr_a = self.df_clean["atr"].values.astype(np.float64)
        self.atr_pct = self.df_clean["atr_percentile"].values.astype(np.float64)
        self.adx_a = self.df_clean["adx"].values.astype(np.float64)
        self.plus_di = self.df_clean["plus_di"].values.astype(np.float64)
        self.minus_di = self.df_clean["minus_di"].values.astype(np.float64)
        self.eslope = self.df_clean["ema50_slope"].values.astype(np.float64)
        self.bb_lower = self.df_clean["bb_lower"].values.astype(np.float64)
        self.bb_upper = self.df_clean["bb_upper"].values.astype(np.float64)
        self.bb_middle = self.df_clean["bb_middle"].values.astype(np.float64)
        self.vol_a = self.df_clean["volume"].values.astype(np.float64)
        self.vs20 = self.df_clean["volume_sma20"].values.astype(np.float64)
        self.vs50 = self.df_clean["volume_sma50"].values.astype(np.float64)
        self.fib382 = self.df_clean["fib_0382"].values.astype(np.float64)
        self.fib500 = self.df_clean["fib_0500"].values.astype(np.float64)
        self.fib618 = self.df_clean["fib_0618"].values.astype(np.float64)
        self.fibdir = self.df_clean["fib_direction"].values.astype(np.int32)
        self.e20t = self.df_clean["ema20_touched"].values.astype(bool)
        self.e50t = self.df_clean["ema50_touched"].values.astype(bool)
        self.e50tu = self.df_clean["ema50_touched_up"].values.astype(bool)

        # Regime as string array
        self.regime = np.array([str(r) for r in self.df_clean["regime"].values])
        self.idx_arr = self.df_clean.index

        # Buy & Hold (from clean data to match backtest.py)
        first_c = float(self.df_clean.iloc[0]["close"])
        last_c = float(self.df_clean.iloc[-1]["close"])
        self.buy_hold = (last_c - first_c) / first_c * 100
        logger.info("Buy & Hold: %.2f%% (%s -> %s)", self.buy_hold,
                     self.df_clean.index[0], self.df_clean.index[-1])

    def _trend_long(self, i, p):
        rg = self.regime[i]
        if rg == 'trending_up':
            if self.adx_a[i] < p['adx_min']: return None
        elif rg == 'transition':
            if not p['allow_transition']: return None
        elif rg == 'ranging':
            if not p.get('allow_ranging', False): return None
        elif rg == 'volatile':
            if not p.get('allow_volatile', False): return None
        else:
            return None
        c = self.close[i]; e50 = self.ema50_a[i]; e200 = self.ema200_a[i]
        if p['require_ema_trend'] and not (c > e50 > e200): return None
        if p['require_slope'] and self.eslope[i] <= p.get('slope_min_val', -1.0): return None
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
        elif rg == 'ranging':
            if not p.get('allow_ranging', False): return None
        elif rg == 'volatile':
            if not p.get('allow_volatile', False): return None
        else:
            return None
        c = self.close[i]; e50 = self.ema50_a[i]; e200 = self.ema200_a[i]
        if p['require_ema_trend'] and not (c < e50 < e200): return None
        if p['require_slope'] and self.eslope[i] >= -p.get('slope_min_val', -1.0): return None
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
            if not pb and self.e50tu[i] and c < e50 and self.high[i] >= e50: pb = True
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
        if self.regime[i] != 'ranging' and not p.get('allow_ranging', False): return None
        if self.regime[i] not in ('ranging', 'transition'): return None
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
        if self.regime[i] != 'ranging' and not p.get('allow_ranging', False): return None
        if self.regime[i] not in ('ranging', 'transition'): return None
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

    def fast_sim(self, p, max_bars=72):
        trades = []; i = 0; n = self.n
        cl = self.close; hi = self.high; lo = self.low
        mr_on = p.get('mr_enabled', False)
        while i < n:
            sig = None; is_long = True; rg = self.regime[i]
            if rg == 'ranging':
                if mr_on:
                    sig = self._mr_long(i, p)
                    if sig: is_long = True
                    else: sig = self._mr_short(i, p); is_long = False
                elif p.get('allow_ranging', False):
                    sig = self._trend_long(i, p)
                    if sig: is_long = True
                    else: sig = self._trend_short(i, p); is_long = False
            elif rg == 'volatile':
                if p.get('allow_volatile', False):
                    sig = self._trend_long(i, p)
                    if sig: is_long = True
                    else: sig = self._trend_short(i, p); is_long = False
            elif rg == 'transition':
                if p.get('allow_transition', False) or mr_on:
                    if mr_on:
                        sig = self._mr_long(i, p)
                        if sig: is_long = True
                        else: sig = self._mr_short(i, p); is_long = False
                    else:
                        sig = self._trend_long(i, p)
                        if sig: is_long = True
                        else: sig = self._trend_short(i, p); is_long = False
            else:  # trending_up / trending_down
                sig = self._trend_long(i, p)
                if sig: is_long = True
                else: sig = self._trend_short(i, p); is_long = False

            if sig is None: i += 1; continue
            ep, sl, tp, atr_v, rsi_v = sig
            exit_price = None; bars = 0; max_j = min(i + max_bars, n)
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

    def run_combo(self, cid, p, phase="p1", max_bars=72):
        trades = self.fast_sim(p, max_bars=max_bars)
        m = calculate_metrics(trades, self.df_clean)
        r = GridResult(
            combo_id=cid, phase=phase,
            allow_ranging=p.get('allow_ranging', False),
            allow_volatile=p.get('allow_volatile', False),
            allow_transition=p.get('allow_transition', False),
            require_ema_trend=p.get('require_ema_trend', True),
            require_slope=p.get('require_slope', True),
            require_pullback=p.get('require_pullback', True),
            ema20_prox_pct=p.get('ema20_prox_pct', 0.0),
            ema50_prox_pct=p.get('ema50_prox_pct', 0.0),
            fib_tolerance_pct=p.get('fib_tolerance_pct', 0.025),
            mr_enabled=p.get('mr_enabled', False),
            mr_rsi_long_max=p.get('mr_rsi_long_max', 35.0),
            mr_rsi_short_min=p.get('mr_rsi_short_min', 65.0),
            mr_bb_band_pct=p.get('mr_bb_band_pct', 0.0),
            rsi_long_min=p.get('rsi_long_min', 30.0),
            rsi_long_max=p.get('rsi_long_max', 70.0),
            rsi_short_min=p.get('rsi_short_min', 30.0),
            rsi_short_max=p.get('rsi_short_max', 70.0),
            volume_confirm=p.get('volume_confirm', False),
            volume_sma_ratio=p.get('volume_sma_ratio', 0.3),
            adx_min=p.get('adx_min', 0.0),
            atr_pct_min=p.get('atr_pct_min', 0.05),
            atr_pct_max=p.get('atr_pct_max', 0.95),
            sl_atr_mult=p.get('sl_atr_mult', 1.5),
            tp_atr_mult=p.get('tp_atr_mult', 2.5),
            total_trades=m.total_trades, long_trades=m.long_trades,
            short_trades=m.short_trades, wins=m.wins, losses=m.losses,
            win_rate=round(m.win_rate, 2), profit_factor=round(m.profit_factor, 4),
            total_pnl_pct=round(m.total_pnl_pct, 4),
            max_drawdown_pct=round(m.max_drawdown_pct, 4),
            sharpe_ratio=round(m.sharpe_ratio, 4),
            avg_bars_held=round(m.avg_bars_held, 1),
            avg_win_pct=round(m.avg_win_pct, 4),
            avg_loss_pct=round(m.avg_loss_pct, 4),
            best_trade_pct=round(m.best_trade_pct, 4),
            worst_trade_pct=round(m.worst_trade_pct, 4),
            buy_hold_pct=round(self.buy_hold, 4),
        )
        r.excess_return = round(r.total_pnl_pct - r.buy_hold_pct, 4)
        r.beats_bh = r.total_pnl_pct > r.buy_hold_pct
        r.score = self._score(r)
        return r

    @staticmethod
    def _score(r):
        """
        Scoring que PENALIZA estrategias abaixo do Buy & Hold.
        Prioridade: PnL > B&H > WR > PF > DD > Frequencia.
        """
        # Hard filters
        if r.total_trades < 30: return 0.0
        if r.profit_factor <= 1.0: return 0.0
        if r.win_rate < 30.0: return 0.0
        if r.max_drawdown_pct > 35.0: return 0.0

        # B&H bonus/penalty (CRITICAL)
        excess = r.total_pnl_pct - r.buy_hold_pct
        if excess <= 0:
            # Penaliza proporcionalmente ao gap
            bh_factor = max(0.01, 1.0 + excess / 10.0)  # excess=-10 -> 0.0
        else:
            bh_factor = 1.0 + (excess / 5.0)  # excess=+10 -> 3.0

        # Core factors
        wr = r.win_rate / 100.0
        pf = r.profit_factor
        dd_pen = max(0.01, 1.0 - r.max_drawdown_pct / 50.0)
        freq_bonus = min(math.log(max(r.total_trades, 1)) / math.log(200), 1.5)
        sharpe_bonus = 1.0 + max(0, r.sharpe_ratio) * 0.1

        score = wr * pf * bh_factor * dd_pen * freq_bonus * sharpe_bonus
        return round(score, 4)


# ============================================================
# Grid Generators
# ============================================================
def make_p1_grid():
    """Phase 1: Grid massivo explorando multiplas abordagens."""
    combos = []

    # Trend-following configs: (allow_tr, req_pb, req_et, req_sl, adx, rsi_l, rsi_s)
    trend_cfgs = [
        # Config 1: v5.0 base + variants
        (True,  True,  True,  True,  25, (25, 50), (50, 75)),
        (True,  True,  True,  True,  20, (25, 50), (50, 75)),
        (True,  True,  True,  True,  30, (25, 50), (50, 75)),
        (True,  True,  True,  True,  15, (25, 50), (50, 75)),
        # Config 2: RSI alargado
        (True,  True,  True,  True,  25, (20, 55), (45, 80)),
        (True,  True,  True,  True,  20, (20, 55), (45, 80)),
        (True,  True,  True,  True,  30, (20, 55), (45, 80)),
        # Config 3: Sem slope (mais sinais)
        (True,  True,  True,  False, 25, (25, 50), (50, 75)),
        (True,  True,  True,  False, 20, (20, 55), (45, 80)),
        (True,  True,  True,  False, 15, (25, 55), (45, 75)),
        # Config 4: Sem pullback (apenas regime+RSI)
        (True,  False, True,  True,  25, (30, 50), (50, 70)),
        (True,  False, True,  False, 25, (30, 50), (50, 70)),
        (True,  False, True,  True,  20, (25, 55), (45, 75)),
        (True,  False, True,  False, 20, (25, 55), (45, 75)),
        # Config 5: Sem EMA trend (mais flexivel)
        (True,  True,  False,  False, 25, (25, 55), (45, 75)),
        (True,  False, False, False, 20, (25, 60), (40, 75)),
        # Config 6: RSI justo (pullback forte)
        (True,  True,  True,  True,  25, (28, 42), (58, 72)),
        (True,  True,  True,  True,  20, (28, 42), (58, 72)),
        (True,  True,  True,  False, 25, (28, 45), (55, 72)),
        # Config 7: RSI ultra-largo (muitos sinais)
        (True,  False, False, False, 0,  (30, 65), (35, 70)),
        (True,  False, False, False, 15, (25, 60), (40, 75)),
        # Config 8: Com ranging + volatile
        (True,  False, True,  False, 0,  (30, 60), (40, 70)),
        # Config 9: Apenas transicao (regime fraco)
        (True,  True,  True,  True,  20, (28, 48), (52, 72)),
        (True,  True,  True,  True,  15, (28, 48), (52, 72)),
        ]

    # MR configs for mean-reversion
    mr_rl = [30, 35, 40, 45]
    mr_rs = [55, 60, 65, 70]
    mr_bb = [0.0, 0.005, 0.01]

    # SL/TP configs — FOCO EM ALTO R:R (custos = 100% ATR!)
    # Com custo = 0.65% e ATR medio = 0.65%, precisamos R:R >= 3:1
    # Ideal: 1.0x SL / 5.0x+ TP para breakeven com WR ~33%
    tr_sltp = [
        (0.75, 3.0), (0.75, 4.0), (0.75, 5.0), (0.75, 6.0),
        (1.0, 3.0), (1.0, 3.5), (1.0, 4.0), (1.0, 5.0), (1.0, 6.0), (1.0, 7.0),
        (1.25, 4.0), (1.25, 5.0), (1.25, 6.0), (1.25, 7.0),
        (1.5, 4.0), (1.5, 5.0), (1.5, 6.0), (1.5, 7.0),
        (2.0, 5.0), (2.0, 6.0), (2.0, 7.0), (2.0, 8.0),
    ]
    mr_sltp = [
        (0.5, 1.0), (0.5, 1.5), (0.5, 2.0), (0.5, 2.5),
        (0.75, 1.25), (0.75, 1.5), (0.75, 2.0), (0.75, 2.5),
        (1.0, 1.5), (1.0, 2.0), (1.0, 2.5),
        (1.25, 2.0), (1.25, 2.5), (1.25, 3.0),
    ]

    # Fib tolerances (reduced for speed)
    fib_tols = [0.025]
    # EMA proximity (disabled for speed)
    ema20_proxs = [0.0]
    ema50_proxs = [0.0]
    # Slope min values
    slope_vals = [-1.0]
    # ATR pct ranges (reduced)
    atr_ranges = [(0.05, 0.95), (0.10, 0.90)]

    for tc in trend_cfgs:
        a_tr, r_pb, r_et, r_sl, adx, rsi_l, rsi_s = tc

        for mr_on in [False, True]:
            if mr_on:
                for mrl in mr_rl:
                    for mrs in mr_rs:
                        for mbb in mr_bb:
                            for sl, tp in mr_sltp:
                                for fib_tol in [0.025]:  # keep fixed in P1 for MR
                                    combos.append({
                                        'allow_ranging': True, 'allow_volatile': False,
                                        'allow_transition': a_tr, 'require_pullback': r_pb,
                                        'require_ema_trend': r_et, 'require_slope': r_sl,
                                        'rsi_long_min': rsi_l[0], 'rsi_long_max': rsi_l[1],
                                        'rsi_short_min': rsi_s[0], 'rsi_short_max': rsi_s[1],
                                        'adx_min': adx, 'fib_tolerance_pct': fib_tol,
                                        'ema20_prox_pct': 0.0, 'ema50_prox_pct': 0.0,
                                        'slope_min_val': -1.0,
                                        'volume_confirm': False, 'volume_sma_ratio': 0.3,
                                        'atr_pct_min': 0.05, 'atr_pct_max': 0.95,
                                        'mr_enabled': True, 'mr_rsi_long_max': float(mrl),
                                        'mr_rsi_short_min': float(mrs), 'mr_bb_band_pct': mbb,
                                        'sl_atr_mult': sl, 'tp_atr_mult': tp})
            else:
                for sl, tp in tr_sltp:
                    for fib_tol in fib_tols:
                        for e20p in ema20_proxs:
                            for e50p in ema50_proxs:
                                for slope_v in (slope_vals if r_sl else [-1.0]):
                                    for atr_min, atr_max in atr_ranges:
                                        combos.append({
                                            'allow_ranging': False, 'allow_volatile': False,
                                            'allow_transition': a_tr, 'require_pullback': r_pb,
                                            'require_ema_trend': r_et, 'require_slope': r_sl,
                                            'rsi_long_min': rsi_l[0], 'rsi_long_max': rsi_l[1],
                                            'rsi_short_min': rsi_s[0], 'rsi_short_max': rsi_s[1],
                                            'adx_min': adx, 'fib_tolerance_pct': fib_tol,
                                            'ema20_prox_pct': e20p, 'ema50_prox_pct': e50p,
                                            'slope_min_val': slope_v,
                                            'volume_confirm': False, 'volume_sma_ratio': 0.3,
                                            'atr_pct_min': atr_min, 'atr_pct_max': atr_max,
                                            'mr_enabled': False, 'mr_rsi_long_max': 35.0,
                                            'mr_rsi_short_min': 65.0, 'mr_bb_band_pct': 0.0,
                                            'sl_atr_mult': sl, 'tp_atr_mult': tp})
    return combos


def make_p2_grid(top_results, n=20):
    """Phase 2: Refinamento fino de SL/TP/MR em volta dos melhores."""
    sls = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
    tps = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    mr_rsi_l = [28, 30, 32, 35, 38, 40, 42, 45]
    mr_rsi_s = [55, 58, 60, 62, 65, 68, 70, 72]
    mr_bb = [0.0, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02]
    rsi_l_offsets = [-5, -3, -1, 0, 1, 3, 5]
    rsi_s_offsets = [-5, -3, -1, 0, 1, 3, 5]

    combos = []
    for r in top_results[:n]:
        base = {
            'allow_ranging': r.allow_ranging, 'allow_volatile': r.allow_volatile,
            'allow_transition': r.allow_transition, 'require_pullback': r.require_pullback,
            'require_ema_trend': r.require_ema_trend, 'require_slope': r.require_slope,
            'fib_tolerance_pct': r.fib_tolerance_pct,
            'ema20_prox_pct': r.ema20_prox_pct, 'ema50_prox_pct': r.ema50_prox_pct,
            'volume_confirm': r.volume_confirm, 'volume_sma_ratio': r.volume_sma_ratio,
            'atr_pct_min': r.atr_pct_min, 'atr_pct_max': r.atr_pct_max,
            'adx_min': r.adx_min, 'slope_min_val': -1.0,
        }
        if r.mr_enabled:
            for sl in sls:
                for tp in tps:
                    if tp / sl < 0.8: continue
                    for mrl in mr_rsi_l:
                        for mrs in mr_rsi_s:
                            for mbb in mr_bb:
                                combos.append({**base, 'mr_enabled': True,
                                    'rsi_long_min': r.rsi_long_min, 'rsi_long_max': r.rsi_long_max,
                                    'rsi_short_min': r.rsi_short_min, 'rsi_short_max': r.rsi_short_max,
                                    'mr_rsi_long_max': float(mrl), 'mr_rsi_short_min': float(mrs),
                                    'mr_bb_band_pct': mbb, 'sl_atr_mult': sl, 'tp_atr_mult': tp})
        else:
            for sl in sls:
                for tp in tps:
                    if tp / sl < 0.8: continue
                    for rsi_l_off in rsi_l_offsets:
                        for rsi_s_off in rsi_s_offsets:
                            combos.append({**base, 'mr_enabled': False,
                                'rsi_long_min': max(10, r.rsi_long_min + rsi_l_off),
                                'rsi_long_max': min(85, r.rsi_long_max + rsi_l_off),
                                'rsi_short_min': max(15, r.rsi_short_min + rsi_s_off),
                                'rsi_short_max': min(90, r.rsi_short_max + rsi_s_off),
                                'mr_rsi_long_max': 35.0, 'mr_rsi_short_min': 65.0,
                                'mr_bb_band_pct': 0.0, 'sl_atr_mult': sl, 'tp_atr_mult': tp})
    return combos


# ============================================================
# Phase 0: Exploratory Data Analysis
# ============================================================
def phase0_analysis(bt: FastBacktester):
    """Analise exploratoria dos dados."""
    print("\n" + "=" * 80)
    print("  PHASE 0: EXPLORATORY DATA ANALYSIS")
    print("=" * 80)

    df = bt.df_clean
    n = len(df)
    print(f"  Candles: {n:,} | Periodo: {df.index[0]} a {df.index[-1]}")
    print(f"  Buy & Hold: {bt.buy_hold:+.2f}%")
    print(f"  Preco: {float(df.iloc[0]['close']):,.2f} -> {float(df.iloc[-1]['close']):,.2f}")

    # Regime distribution
    regimes = bt.regime
    regime_counts = {r: int(np.sum(regimes == r)) for r in np.unique(regimes)}
    print(f"\n  Regime Distribution:")
    for rg, cnt in sorted(regime_counts.items(), key=lambda x: -x[1]):
        print(f"    {rg:>15}: {cnt:>6} ({cnt/n*100:>5.1f}%)")

    # Volatility analysis
    atr = bt.atr_a
    close = bt.close
    print(f"\n  Volatility (ATR):")
    print(f"    ATR medio:  {np.mean(atr):.2f} ({np.mean(atr)/np.mean(close)*100:.3f}%)")
    print(f"    ATR mediano: {np.median(atr):.2f}")
    print(f"    ATR p10:     {np.percentile(atr, 10):.2f}")
    print(f"    ATR p90:     {np.percentile(atr, 90):.2f}")

    # RSI distribution
    rsi = bt.rsi_a
    print(f"\n  RSI Distribution:")
    print(f"    RSI medio:  {np.mean(rsi):.1f}")
    print(f"    RSI < 30:   {np.sum(rsi < 30):,} ({np.sum(rsi < 30)/n*100:.1f}%)")
    print(f"    RSI 30-50:  {np.sum((rsi >= 30) & (rsi <= 50)):,} ({np.sum((rsi >= 30) & (rsi <= 50))/n*100:.1f}%)")
    print(f"    RSI 50-70:  {np.sum((rsi > 50) & (rsi <= 70)):,} ({np.sum((rsi > 50) & (rsi <= 70))/n*100:.1f}%)")
    print(f"    RSI > 70:   {np.sum(rsi > 70):,} ({np.sum(rsi > 70)/n*100:.1f}%)")

    # EMA alignment
    e50 = bt.ema50_a; e200 = bt.ema200_a; c = bt.close
    uptrend = np.sum((c > e50) & (e50 > e200))
    downtrend = np.sum((c < e50) & (e50 < e200))
    mixed = n - uptrend - downtrend
    print(f"\n  EMA Alignment (close vs EMA50 vs EMA200):")
    print(f"    Uptrend (c>e50>e200):   {uptrend:>6} ({uptrend/n*100:>5.1f}%)")
    print(f"    Downtrend (c<e50<e200): {downtrend:>6} ({downtrend/n*100:>5.1f}%)")
    print(f"    Mixed:                  {mixed:>6} ({mixed/n*100:>5.1f}%)")

    # ADX distribution
    adx = bt.adx_a
    print(f"\n  ADX Distribution:")
    print(f"    ADX medio:  {np.mean(adx):.1f}")
    print(f"    ADX > 20:   {np.sum(adx > 20):,} ({np.sum(adx > 20)/n*100:.1f}%)")
    print(f"    ADX > 25:   {np.sum(adx > 25):,} ({np.sum(adx > 25)/n*100:.1f}%)")
    print(f"    ADX > 30:   {np.sum(adx > 30):,} ({np.sum(adx > 30)/n*100:.1f}%)")

    # Cost impact analysis
    print(f"\n  Cost Impact (per trade):")
    print(f"    Fee:       {DEFAULT_FEE_PCT}% x 2 = {DEFAULT_FEE_PCT*2:.3f}%")
    print(f"    Spread:    {DEFAULT_SPREAD_BPS/100:.3f}% x 2 = {DEFAULT_SPREAD_BPS/50:.3f}%")
    print(f"    Slippage:  {DEFAULT_SLIPPAGE_BPS/100:.3f}% x 2 = {DEFAULT_SLIPPAGE_BPS/50:.3f}%")
    total_cost = DEFAULT_FEE_PCT*2 + DEFAULT_SPREAD_BPS/50 + DEFAULT_SLIPPAGE_BPS/50
    print(f"    TOTAL:     {total_cost:.3f}% por trade")
    avg_atr_pct = np.mean(atr) / np.mean(close) * 100
    print(f"    ATR medio: {avg_atr_pct:.3f}% do preco")
    print(f"    Custo/ATR: {total_cost/avg_atr_pct*100:.1f}%")


# ============================================================
# Phase 3: Walk-Forward Validation
# ============================================================
def phase3_walkforward(bt: FastBacktester, best_params: dict, n_windows=6):
    """Validacao walk-forward em janelas separadas."""
    print("\n" + "=" * 80)
    print("  PHASE 3: WALK-FORWARD VALIDATION")
    print("=" * 80)

    df = bt.df_clean
    n = len(df)
    window_size = n // (n_windows + 1)  # leave room for train
    train_size = window_size  # same size for simplicity

    results = []
    for w in range(n_windows):
        test_start = w * window_size
        test_end = min((w + 1) * window_size, n)

        # Create sub-backtester for this window
        sub_df = df.iloc[test_start:test_end].copy()
        if len(sub_df) < 200:
            continue

        # Manual simulation on subset
        trades = []
        i = 0
        sub_close = sub_df["close"].values.astype(np.float64)
        sub_high = sub_df["high"].values.astype(np.float64)
        sub_low = sub_df["low"].values.astype(np.float64)
        sub_n = len(sub_df)

        # We need the full bt arrays but restricted to window
        offset = test_start

        while i < sub_n:
            gi = offset + i  # global index
            if gi >= bt.n: break

            sig = None; is_long = True; rg = bt.regime[gi]
            mr_on = best_params.get('mr_enabled', False)

            if rg == 'ranging':
                if mr_on:
                    sig = bt._mr_long(gi, best_params)
                    if sig: is_long = True
                    else: sig = bt._mr_short(gi, best_params); is_long = False
            elif rg == 'volatile':
                if best_params.get('allow_volatile', False):
                    sig = bt._trend_long(gi, best_params)
                    if sig: is_long = True
                    else: sig = bt._trend_short(gi, best_params); is_long = False
            elif rg == 'transition':
                if best_params.get('allow_transition', False) or mr_on:
                    if mr_on:
                        sig = bt._mr_long(gi, best_params)
                        if sig: is_long = True
                        else: sig = bt._mr_short(gi, best_params); is_long = False
                    else:
                        sig = bt._trend_long(gi, best_params)
                        if sig: is_long = True
                        else: sig = bt._trend_short(gi, best_params); is_long = False
            else:
                sig = bt._trend_long(gi, best_params)
                if sig: is_long = True
                else: sig = bt._trend_short(gi, best_params); is_long = False

            if sig is None: i += 1; continue
            ep, sl, tp, atr_v, rsi_v = sig
            exit_price = None; bars = 0
            max_j = min(i + 72, sub_n)
            if is_long:
                for j in range(i + 1, max_j):
                    if sub_low[j] <= sl: exit_price = sl; bars = j - i; break
                    if sub_high[j] >= tp: exit_price = tp; bars = j - i; break
            else:
                for j in range(i + 1, max_j):
                    if sub_high[j] >= sl: exit_price = sl; bars = j - i; break
                    if sub_low[j] <= tp: exit_price = tp; bars = j - i; break
            if exit_price is None:
                bars = max_j - 1 - i; exit_price = sub_close[max_j - 1]
            _, adj, _ = _apply_costs(ep, exit_price, is_long, DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS)
            pnl = (adj - ep) / ep * 100 if is_long else (ep - adj) / ep * 100
            trades.append(TradeResult(
                entry_ts=sub_df.index[i], exit_ts=sub_df.index[min(i + bars, sub_n - 1)],
                type='LONG' if is_long else 'SHORT',
                entry_price=ep, exit_price=exit_price, stop_loss=sl, take_profit=tp,
                atr=atr_v, rsi=rsi_v, pnl_pct=round(pnl, 4), pnl_abs=round(exit_price - ep, 2),
                bars_held=bars, exit_reason='x'))
            i += bars + 1

        m = calculate_metrics(trades, sub_df)
        bh = (float(sub_df.iloc[-1]['close']) - float(sub_df.iloc[0]['close'])) / float(sub_df.iloc[0]['close']) * 100
        beats = m.total_pnl_pct > bh

        results.append({
            'window': w + 1,
            'start': str(sub_df.index[0])[:10],
            'end': str(sub_df.index[-1])[:10],
            'trades': m.total_trades,
            'wr': round(m.win_rate, 1),
            'pf': round(m.profit_factor, 2),
            'pnl': round(m.total_pnl_pct, 2),
            'bh': round(bh, 2),
            'dd': round(m.max_drawdown_pct, 2),
            'beats_bh': beats,
        })
        print(f"    W{w+1}: {sub_df.index[0].strftime('%Y-%m-%d')} a {sub_df.index[-1].strftime('%Y-%m-%d')} | "
              f"T={m.total_trades:>3} WR={m.win_rate:>5.1f}% PF={m.profit_factor:>5.2f} "
              f"PnL={m.total_pnl_pct:>+7.2f}% B&H={bh:>+7.2f}% DD={m.max_drawdown_pct:>5.2f}% "
              f"{'BEATS' if beats else 'LOSES'}")

    if results:
        avg_pnl = np.mean([r['pnl'] for r in results])
        avg_bh = np.mean([r['bh'] for r in results])
        avg_wr = np.mean([r['wr'] for r in results])
        avg_pf = np.mean([r['pf'] for r in results])
        windows_beating = sum(1 for r in results if r['beats_bh'])
        total_trades = sum(r['trades'] for r in results)

        print(f"\n    RESUMO WALK-FORWARD ({len(results)} janelas):")
        print(f"    Avg PnL:  {avg_pnl:>+7.2f}% | Avg B&H: {avg_bh:>+7.2f}%")
        print(f"    Avg WR:   {avg_wr:>5.1f}% | Avg PF:  {avg_pf:>5.2f}")
        print(f"    Total Trades: {total_trades}")
        print(f"    Janelas batendo B&H: {windows_beating}/{len(results)}")
        return {'avg_pnl': avg_pnl, 'avg_bh': avg_bh, 'windows_beating': windows_beating,
                'total_windows': len(results), 'total_trades': total_trades}
    return None


# ============================================================
# Phase 4: Cost Sensitivity
# ============================================================
def phase4_sensitivity(bt: FastBacktester, best_params: dict):
    """Testa sensibilidade a custos."""
    print("\n" + "=" * 80)
    print("  PHASE 4: COST SENSITIVITY ANALYSIS")
    print("=" * 80)

    cost_scenarios = [
        ("Zero costs", 0.0, 0.0, 0.0),
        ("Low (Binance VIP)", 0.02, 5.0, 10.0),
        ("Current", DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS),
        ("High (small pair)", 0.05, 15.0, 30.0),
        ("Very High", 0.10, 20.0, 40.0),
    ]

    for name, fee, spread, slip in cost_scenarios:
        trades = bt.fast_sim(best_params)
        # Re-apply costs with different values
        total_pnl = 0.0
        wins = 0; losses = 0; total_win = 0.0; total_loss = 0.0
        for t in trades:
            _, adj, _ = _apply_costs(t.entry_price, t.exit_price, t.type == 'LONG', fee, spread, slip)
            pnl = (adj - t.entry_price) / t.entry_price * 100 if t.type == 'LONG' else (t.entry_price - adj) / t.entry_price * 100
            total_pnl += pnl
            if pnl > 0: wins += 1; total_win += pnl
            else: losses += 1; total_loss += abs(pnl)

        pf = total_win / total_loss if total_loss > 0 else 0
        wr = wins / len(trades) * 100 if trades else 0
        beats = total_pnl > bt.buy_hold
        total_cost = fee * 2 + spread / 50 + slip / 50
        print(f"    {name:>20}: PnL={total_pnl:>+8.2f}% B&H={bt.buy_hold:>+8.2f}% "
              f"WR={wr:>5.1f}% PF={pf:>5.2f} Cost={total_cost:.3f}% "
              f"{'BEATS' if beats else 'LOSES'}")


# ============================================================
# Output
# ============================================================
def print_top(results, title, n=25):
    print(f"\n{'='*180}")
    print(f"  {title}")
    print(f"{'='*180}")
    print(f"{'#':>3} {'Score':>7} {'Trades':>6} {'L/S':>6} {'T/d':>5} {'WR%':>6} {'PF':>6} {'PnL%':>8} {'B&H%':>7} {'Excess':>7} {'DD%':>6} {'Sharpe':>7} "
          f"{'Trn':>4} {'Rng':>4} {'Vol':>4} {'Pb':>3} {'ET':>3} {'Sl':>3} {'MR':>3} "
          f"{'RSI_L':>8} {'RSI_S':>8} {'ADX':>4} {'SL':>4} {'TP':>4}")
    print(f"{'-'*180}")
    for i, r in enumerate(sorted(results, key=lambda x: x.score, reverse=True)[:n]):
        rl = f"{r.rsi_long_min:.0f}-{r.rsi_long_max:.0f}"
        rs = f"{r.rsi_short_min:.0f}-{r.rsi_short_max:.0f}"
        ls = f"{r.long_trades}/{r.short_trades}"
        td = r.total_trades / DAYS
        excess = r.total_pnl_pct - r.buy_hold_pct
        print(f"{i+1:>3} {r.score:>7.3f} {r.total_trades:>6} {ls:>6} {td:>5.2f} {r.win_rate:>6.1f} {r.profit_factor:>6.2f} "
              f"{r.total_pnl_pct:>8.2f} {r.buy_hold_pct:>7.2f} {excess:>+7.2f} {r.max_drawdown_pct:>6.2f} {r.sharpe_ratio:>7.2f} "
              f"{'S' if r.allow_transition else 'N':>4} {'S' if r.allow_ranging else 'N':>4} "
              f"{'S' if r.allow_volatile else 'N':>4} "
              f"{'S' if r.require_pullback else 'N':>3} {'S' if r.require_ema_trend else 'N':>3} "
              f"{'S' if r.require_slope else 'N':>3} {'S' if r.mr_enabled else 'N':>3} "
              f"{rl:>8} {rs:>8} {r.adx_min:>4.0f} {r.sl_atr_mult:>4.2f} {r.tp_atr_mult:>4.2f}")


def save_results(winner, all_results, bt, t0, p1c, p2c, wf_result):
    out = os.path.join(SCRIPT_DIR, "download")
    os.makedirs(out, exist_ok=True)

    # Also save EDA report
    eda_path = os.path.join(out, "1h_study_eda.txt")
    with open(eda_path, "w") as f:
        f.write(f"BTC/USDT 1H Study - Data Summary\n")
        f.write(f"Candles: {len(bt.df_clean):,}\n")
        f.write(f"Period: {bt.df_clean.index[0]} to {bt.df_clean.index[-1]}\n")
        f.write(f"Buy & Hold: {bt.buy_hold:+.2f}%\n")
    logger.info("EDA saved to %s", eda_path)

    if all_results:
        df_r = pd.DataFrame([r.to_dict() for r in all_results]).sort_values("score", ascending=False)
        df_r.to_csv(os.path.join(out, "grid_1h_study_results.csv"), index=False)

    if winner:
        params = {k: getattr(winner, k) for k in [
            'allow_ranging', 'allow_volatile', 'allow_transition', 'require_ema_trend',
            'require_slope', 'require_pullback', 'mr_enabled', 'mr_rsi_long_max',
            'mr_rsi_short_min', 'mr_bb_band_pct', 'rsi_long_min', 'rsi_long_max',
            'rsi_short_min', 'rsi_short_max', 'adx_min', 'fib_tolerance_pct',
            'ema20_prox_pct', 'ema50_prox_pct', 'sl_atr_mult', 'tp_atr_mult',
            'volume_confirm', 'volume_sma_ratio', 'atr_pct_min', 'atr_pct_max']}
        wdata = {
            "version": "1h-study-v1",
            "goal": "beat_buy_and_hold",
            "score": winner.score,
            "params": params,
            "metrics": {
                "total_trades": winner.total_trades,
                "trades_per_day": round(winner.total_trades / DAYS, 3),
                "win_rate": winner.win_rate,
                "profit_factor": winner.profit_factor,
                "total_pnl_pct": winner.total_pnl_pct,
                "max_drawdown_pct": winner.max_drawdown_pct,
                "sharpe_ratio": winner.sharpe_ratio,
                "buy_hold_pct": winner.buy_hold_pct,
                "excess_return": winner.excess_return,
                "beats_bh": winner.beats_bh,
            },
            "walk_forward": wf_result,
            "grid": {"p1": p1c, "p2": p2c, "total": p1c + p2c, "time_sec": round(time.time() - t0, 1)}
        }
        with open(os.path.join(out, "winner_1h_study.json"), "w") as f:
            json.dump(wdata, f, indent=2)
        print(f"\n  Saved: winner_1h_study.json + grid_1h_study_results.csv")


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    print("\n" + "#" * 180)
    print("#  BTC/USDT 1H — ESTUDO ABRANGENTE: Encontrar estrategia que supera Buy & Hold")
    print("#  Meta: PnL > B&H ({DAYS} dias) com WR>35%, PF>1.0, DD<30%")
    print("#" * 180)

    # Load data
    print("\n[1/5] Carregando dados...")
    if os.path.exists(DATA_CACHE):
        logger.info("Loading cached data from %s", DATA_CACHE)
        with open(DATA_CACHE, "rb") as f:
            df = pickle.load(f)
        logger.info("Cached data: %d candles", len(df))
    else:
        df = fetch_historical_ohlcv(SYMBOL, TIMEFRAME, DAYS)
        try:
            with open(DATA_CACHE, "wb") as f:
                pickle.dump(df, f)
            logger.info("Data cached to %s", DATA_CACHE)
        except Exception as e:
            logger.warning("Could not cache data: %s", e)

    bt = FastBacktester(df, TIMEFRAME)

    # Phase 0: EDA
    phase0_analysis(bt)

    # Phase 1: Massive grid
    print(f"\n[2/5] PHASE 1: Grid massivo...")
    p1 = make_p1_grid()
    print(f"      {len(p1):,} combinacoes")
    p1_res = []; viable = 0; best_score = 0.0; best_str = ""
    t1 = time.time()

    for idx, params in enumerate(p1):
        r = bt.run_combo(idx, params, "p1")
        p1_res.append(r)
        if r.score > 0: viable += 1
        if r.score > best_score:
            best_score = r.score
            best_str = (f"T={r.total_trades} WR={r.win_rate:.1f}% PF={r.profit_factor:.2f} "
                       f"PnL={r.total_pnl_pct:+.2f}% B&H={r.buy_hold_pct:+.2f}% "
                       f"Excess={r.excess_return:+.2f}% MR={'S' if r.mr_enabled else 'N'}")
        if (idx + 1) % 5000 == 0:
            el = time.time() - t1; spd = (idx + 1) / max(el, 0.01)
            eta = (len(p1) - idx - 1) / max(spd, 0.01)
            print(f"      [{idx+1:,}/{len(p1):,}] {spd:.0f}/s viable={viable} ETA={eta:.0f}s")
            if best_str: print(f"        BEST: {best_str}")

    p1_time = time.time() - t1
    print(f"      Phase 1: {len(p1):,} combos em {p1_time:.1f}s | viable={viable}")
    p1_sorted = sorted(p1_res, key=lambda x: x.score, reverse=True)
    print_top(p1_sorted, f"PHASE 1 Top 25 de {len(p1):,}", 25)

    if not p1_sorted or p1_sorted[0].score == 0:
        print("\n!!! Nenhuma combinacao viavel no P1 !!!")
        by_pnl = sorted(p1_res, key=lambda x: x.total_pnl_pct, reverse=True)
        print("\n  Top 10 por PnL absoluto:")
        for r in by_pnl[:10]:
            print(f"    T={r.total_trades} WR={r.win_rate:.1f}% PF={r.profit_factor:.2f} "
                  f"PnL={r.total_pnl_pct:+.2f}% B&H={r.buy_hold_pct:+.2f}% MR={'S' if r.mr_enabled else 'N'}")
        # Save whatever we have
        save_results(by_pnl[0] if by_pnl else None, p1_res, bt, t0, len(p1), 0, None)
        return

    # Phase 2: Refinement
    n_top = min(20, len([r for r in p1_sorted if r.score > 0]))
    p1_top = [r for r in p1_sorted if r.score > 0][:n_top]

    print(f"\n[3/5] PHASE 2: Refinamento para top {n_top}...")
    p2 = make_p2_grid(p1_top, n_top)
    print(f"      {len(p2):,} combinacoes")
    p2_res = []; t2 = time.time(); best_p2 = ""
    for idx, params in enumerate(p2):
        r = bt.run_combo(500000 + idx, params, "p2")
        p2_res.append(r)
        if r.score > best_score:
            best_score = r.score
            best_p2 = (f"T={r.total_trades} WR={r.win_rate:.1f}% PF={r.profit_factor:.2f} "
                       f"PnL={r.total_pnl_pct:+.2f}% SL={r.sl_atr_mult:.2f} TP={r.tp_atr_mult:.2f}")
        if (idx + 1) % 10000 == 0:
            el = time.time() - t2; spd = (idx + 1) / max(el, 0.01)
            eta = (len(p2) - idx - 1) / max(spd, 0.01)
            print(f"      [{idx+1:,}/{len(p2):,}] {spd:.0f}/s ETA={eta:.0f}s")
            if best_p2: print(f"        BEST: {best_p2}")

    p2_time = time.time() - t2
    print(f"      Phase 2: {len(p2):,} combos em {p2_time:.1f}s")

    all_res = p1_res + p2_res
    all_sorted = sorted(all_res, key=lambda x: x.score, reverse=True)
    print_top(all_sorted, f"COMBINED Top 25 ({len(all_res):,} total)", 25)

    winner = all_sorted[0] if all_sorted else None

    if winner:
        print(f"\n{'#'*180}")
        print(f"#  VENCEDOR DO ESTUDO 1H")
        print(f"{'#'*180}")
        print(f"#  Score:         {winner.score:.4f}")
        print(f"#  Trades:        {winner.total_trades} ({winner.total_trades/DAYS:.2f}/dia) L={winner.long_trades} S={winner.short_trades}")
        print(f"#  Win Rate:      {winner.win_rate:.1f}%")
        print(f"#  Profit Factor: {winner.profit_factor:.2f}")
        print(f"#  PnL:           {winner.total_pnl_pct:+.2f}% (B&H={winner.buy_hold_pct:+.2f}%)")
        print(f"#  Excess Return: {winner.excess_return:+.2f}% ({'BEATS B&H' if winner.beats_bh else 'LOSES TO B&H'})")
        print(f"#  Max DD:        {winner.max_drawdown_pct:.2f}%  Sharpe: {winner.sharpe_ratio:.2f}")
        print(f"#  MR:            {winner.mr_enabled} | RSI_L_mr={winner.mr_rsi_long_max} RSI_S_mr={winner.mr_rsi_short_min} BB={winner.mr_bb_band_pct}")
        print(f"#  Trend:         transition={'S' if winner.allow_transition else 'N'} pullback={'S' if winner.require_pullback else 'N'} ema_trend={'S' if winner.require_ema_trend else 'N'} slope={'S' if winner.require_slope else 'N'}")
        print(f"#  RSI:           L={winner.rsi_long_min:.0f}-{winner.rsi_long_max:.0f} S={winner.rsi_short_min:.0f}-{winner.rsi_short_max:.0f}")
        print(f"#  SL/TP:         {winner.sl_atr_mult:.2f}x / {winner.tp_atr_mult:.2f}x (R:R {winner.tp_atr_mult/winner.sl_atr_mult:.1f}:1)")
        print(f"#  ADX:           {winner.adx_min:.0f} | ATR_pct: [{winner.atr_pct_min:.2f}, {winner.atr_pct_max:.2f}]")
        print(f"#  FIB_tol:       {winner.fib_tolerance_pct*100:.1f}% | EMA20_prox={winner.ema20_prox_pct*100:.1f}% | EMA50_prox={winner.ema50_prox_pct*100:.1f}%")
        print(f"#  Total combos:  {len(p1)+len(p2):,} em {time.time()-t0:.0f}s")
        print(f"{'#'*180}")

        # Extract best params for WF and sensitivity
        best_params = {
            'allow_ranging': winner.allow_ranging, 'allow_volatile': winner.allow_volatile,
            'allow_transition': winner.allow_transition, 'require_pullback': winner.require_pullback,
            'require_ema_trend': winner.require_ema_trend, 'require_slope': winner.require_slope,
            'rsi_long_min': winner.rsi_long_min, 'rsi_long_max': winner.rsi_long_max,
            'rsi_short_min': winner.rsi_short_min, 'rsi_short_max': winner.rsi_short_max,
            'adx_min': winner.adx_min, 'fib_tolerance_pct': winner.fib_tolerance_pct,
            'ema20_prox_pct': winner.ema20_prox_pct, 'ema50_prox_pct': winner.ema50_prox_pct,
            'slope_min_val': -1.0,
            'volume_confirm': winner.volume_confirm, 'volume_sma_ratio': winner.volume_sma_ratio,
            'atr_pct_min': winner.atr_pct_min, 'atr_pct_max': winner.atr_pct_max,
            'mr_enabled': winner.mr_enabled, 'mr_rsi_long_max': winner.mr_rsi_long_max,
            'mr_rsi_short_min': winner.mr_rsi_short_min, 'mr_bb_band_pct': winner.mr_bb_band_pct,
            'sl_atr_mult': winner.sl_atr_mult, 'tp_atr_mult': winner.tp_atr_mult,
        }

        # Phase 3: Walk-Forward
        print(f"\n[4/5] PHASE 3: Walk-Forward...")
        wf_result = phase3_walkforward(bt, best_params)

        # Phase 4: Cost sensitivity
        print(f"\n[5/5] PHASE 4: Cost sensitivity...")
        phase4_sensitivity(bt, best_params)

        # Save
        save_results(winner, all_res, bt, t0, len(p1), len(p2), wf_result)
    else:
        save_results(None, all_res, bt, t0, len(p1), len(p2), None)

    tt = time.time() - t0
    print(f"\n  Estudo completo em {tt:.0f}s ({tt/60:.1f}min)")


if __name__ == "__main__":
    main()
