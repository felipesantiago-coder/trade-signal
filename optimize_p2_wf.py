"""
optimize_p2_wf.py
------------------
P2: Refinamento SL/TP ao redor de 2.5x/12.0x para timeframe 1h
Walk-Forward: 6 janelas (7 splits) com validacao out-of-sample

Estrategia:
  1. Carregar dados 730d BTC/USDT 1h
  2. P2: Grid fino de SL [1.5-3.5] x TP [8.0-16.0] com params STANDARD
  3. Walk-Forward: 6 passos, treinar em janela i, testar em janela i+1
  4. Output: melhor combinacao OOS + metricas agregadas
"""
from __future__ import annotations
import sys, os, time, math, json, logging, pickle
collections_abc = __import__('collections.abc')
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict

logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger("ctev.p2wf")
logger.setLevel(logging.INFO)

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from indicators import compute_indicators
from backtest import (fetch_historical_ohlcv, calculate_metrics, TradeResult,
    _apply_costs, DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS)

SYMBOL = "BTC/USDT"; TIMEFRAME = "1h"; DAYS = 730
DATA_CACHE = "/tmp/ctev_p2wf_data.pkl"

# Base STANDARD entry params (fixed from P1)
BASE_PARAMS = {
    'allow_ranging': False, 'allow_volatile': False,
    'allow_transition': True, 'require_pullback': True,
    'require_ema_trend': True, 'require_slope': True,
    'rsi_long_min': 28.0, 'rsi_long_max': 48.0,
    'rsi_short_min': 55.0, 'rsi_short_max': 75.0,
    'adx_min': 30.0, 'fib_tolerance_pct': 0.025,
    'ema20_prox_pct': 0.0, 'ema50_prox_pct': 0.0,
    'volume_confirm': False, 'volume_sma_ratio': 0.3,
    'atr_pct_min': 0.10, 'atr_pct_max': 0.90,
    'mr_enabled': False, 'mr_rsi_long_max': 35.0,
    'mr_rsi_short_min': 65.0, 'mr_bb_band_pct': 0.0,
}

# P2 Grid: SL around 2.5x, TP around 12.0x
SL_VALUES = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5]
TP_VALUES = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]

# Walk-Forward: 7 splits -> 6 WF steps
N_SPLITS = 7


@dataclass
class SimResult:
    combo_id: int
    sl_atr_mult: float
    tp_atr_mult: float
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    buy_hold_pct: float = 0.0
    score: float = 0.0
    window: str = "full"
    phase: str = "p2"


class P2Optimizer:
    """Otimizador P2 com suporte a sub-arrays para walk-forward."""

    def __init__(self):
        self.df_clean = None
        self.n = 0
        # Full arrays
        self.close = self.high = self.low = None
        self.regime = self.atr_pct = self.rsi_a = self.atr_a = None
        self.ema20_a = self.ema50_a = self.ema200_a = None
        self.adx_a = self.eslope = None
        self.fib382 = self.fib500 = self.fib618 = self.fibdir = None
        self.vol_a = self.vs50 = None
        self.e20t = self.e50t = self.e50tu = None
        self.bb_lower = self.bb_upper = None
        self.idx_arr = None
        self.buy_hold = 0.0

    def load_data(self):
        if os.path.exists(DATA_CACHE):
            with open(DATA_CACHE, 'rb') as f:
                self.df_clean = pickle.load(f)['df_clean']
            logger.info("Dados carregados do cache %s", DATA_CACHE)
        else:
            df = fetch_historical_ohlcv(SYMBOL, TIMEFRAME, DAYS)
            df_ind = compute_indicators(df, timeframe=TIMEFRAME)
            crit = ["ema20", "ema50", "ema200", "rsi", "atr", "atr_percentile",
                    "macd", "macd_signal", "macd_hist", "adx", "plus_di", "minus_di", "regime"]
            self.df_clean = df_ind.dropna(subset=crit).copy()
            with open(DATA_CACHE, 'wb') as f:
                pickle.dump({'df_clean': self.df_clean}, f)
            logger.info("Dados baixados e cacheados")
        self._load_arrays(self.df_clean)
        self.buy_hold = (self.close[-1] - self.close[0]) / self.close[0] * 100
        logger.info("%d candles | B&H=%.2f%%", self.n, self.buy_hold)

    def _load_arrays(self, df: pd.DataFrame):
        """Carrega arrays numpy de um DataFrame."""
        self.n = len(df)
        self.idx_arr = df.index.values
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

    def _trend_long(self, i, p):
        rg = self.regime[i]
        if rg == 'trending_up':
            if self.adx_a[i] < p['adx_min']: return None
        elif rg == 'transition':
            if not p['allow_transition']: return None
        else:
            return None
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
        else:
            return None
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

    def fast_sim(self, p, start=0, end=None, max_bars=72):
        """Simula trades em sub-range [start, end)."""
        if end is None:
            end = self.n
        trades = []; i = start; n = end
        cl = self.close; hi = self.high; lo = self.low
        while i < n:
            sig = None; is_long = True; rg = self.regime[i]
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
                entry_ts=self.idx_arr[i], exit_ts=self.idx_arr[min(i + bars, self.n - 1)],
                type='LONG' if is_long else 'SHORT',
                entry_price=ep, exit_price=exit_price, stop_loss=sl, take_profit=tp,
                atr=atr_v, rsi=rsi_v, pnl_pct=round(pnl, 4), pnl_abs=round(exit_price - ep, 2),
                bars_held=bars, exit_reason='x'))
            i += max(bars + 1, 1)
        return trades

    def run_sl_tp(self, sl, tp, start=0, end=None, max_bars=72):
        """Roda sim com SL/TP especificos, retorna metricas."""
        p = {**BASE_PARAMS, 'sl_atr_mult': sl, 'tp_atr_mult': tp}
        trades = self.fast_sim(p, start, end, max_bars)
        if not trades:
            return SimResult(combo_id=0, sl_atr_mult=sl, tp_atr_mult=tp)
        m = calculate_metrics(trades, self.df_clean.iloc[start:end])
        bh = (self.close[min(end, self.n) - 1] - self.close[start]) / self.close[start] * 100 if end and end > start else self.buy_hold
        r = SimResult(
            combo_id=0, sl_atr_mult=sl, tp_atr_mult=tp,
            total_trades=m.total_trades, wins=m.wins, losses=m.losses,
            win_rate=round(m.win_rate, 2), profit_factor=round(m.profit_factor, 4),
            total_pnl_pct=round(m.total_pnl_pct, 4), max_drawdown_pct=round(m.max_drawdown_pct, 4),
            sharpe_ratio=round(m.sharpe_ratio, 4), buy_hold_pct=round(bh, 4))
        r.score = self._score(r)
        return r

    @staticmethod
    def _score(r):
        if r.total_trades < 20 or r.profit_factor <= 1.0: return 0.0
        if r.max_drawdown_pct > 30.0: return 0.0
        wr = r.win_rate / 100.0; pf = r.profit_factor
        dd_pen = max(0, 1.0 - r.max_drawdown_pct / 100.0)
        freq_bonus = math.log(max(r.total_trades, 1)) / math.log(200)
        pnl_b = 1.5 if r.total_pnl_pct > 5.0 else (1.2 if r.total_pnl_pct > 0 else 0.5)
        # Bonus: supera Buy & Hold
        bh_beat = 1.3 if r.total_pnl_pct > r.buy_hold_pct else 0.8
        return round(wr * pf * freq_bonus * dd_pen * pnl_b * bh_beat, 4)

    def get_window_indices(self):
        """Retorna lista de (start, end) para cada split."""
        window_size = self.n // N_SPLITS
        splits = []
        for i in range(N_SPLITS):
            s = i * window_size
            e = (i + 1) * window_size if i < N_SPLITS - 1 else self.n
            splits.append((s, e))
        return splits


def run_p2_full(opt: P2Optimizer):
    """P2: Grid completo SL x TP nos 730 dias."""
    print(f"\n{'='*120}")
    print(f"  P2: Refinamento SL/TP — {len(SL_VALUES)} SL x {len(TP_VALUES)} TP = {len(SL_VALUES)*len(TP_VALUES)} combos")
    print(f"  SL range: [{min(SL_VALUES)}, {max(SL_VALUES)}] | TP range: [{min(TP_VALUES)}, {max(TP_VALUES)}]")
    print(f"{'='*120}")

    results = []; cid = 0
    t0 = time.time()
    for sl in SL_VALUES:
        for tp in TP_VALUES:
            r = opt.run_sl_tp(sl, tp)
            r.combo_id = cid; r.phase = "p2_full"; r.window = "full"
            results.append(r); cid += 1

    elapsed = time.time() - t0
    print(f"  P2 completo em {elapsed:.1f}s ({len(results)} combos, {elapsed/max(len(results),1)*1000:.1f}ms/combo)")

    # Top results
    viable = [r for r in results if r.score > 0]
    sorted_r = sorted(viable, key=lambda x: x.score, reverse=True)
    print(f"\n  Viaveis: {len(viable)}/{len(results)}")

    print(f"\n  {'#':>3} {'SL':>5} {'TP':>5} {'R:R':>5} {'Trades':>7} {'WR%':>6} {'PF':>6} {'PnL%':>8} {'DD%':>6} {'Sharpe':>7} {'B&H%':>7} {'Score':>7}")
    print(f"  {'-'*90}")
    for i, r in enumerate(sorted_r[:30]):
        rr = r.tp_atr_mult / r.sl_atr_mult if r.sl_atr_mult > 0 else 0
        bh_flag = "*" if r.total_pnl_pct > r.buy_hold_pct else " "
        print(f"  {i+1:>3} {r.sl_atr_mult:>5.2f} {r.tp_atr_mult:>5.2f} {rr:>5.1f} {r.total_trades:>7} {r.win_rate:>6.1f} {r.profit_factor:>6.2f} "
              f"{r.total_pnl_pct:>8.2f} {r.max_drawdown_pct:>6.2f} {r.sharpe_ratio:>7.2f} {r.buy_hold_pct:>7.2f} {r.score:>7.3f}{bh_flag}")

    return sorted_r


def run_walk_forward(opt: P2Optimizer, top_n=10):
    """Walk-Forward: 6 passos com top_n melhores SL/TP do P2."""
    # Primeiro, encontrar os top_n SL/TP do P2 full
    p2_results = []
    for sl in SL_VALUES:
        for tp in TP_VALUES:
            r = opt.run_sl_tp(sl, tp)
            r.combo_id = 0; r.phase = "p2_full"; r.window = "full"
            p2_results.append(r)

    p2_sorted = sorted([r for r in p2_results if r.score > 0], key=lambda x: x.score, reverse=True)
    top_combos = [(r.sl_atr_mult, r.tp_atr_mult) for r in p2_sorted[:top_n]]

    if not top_combos:
        print("\n!!! Nenhuma combinacao viavel para Walk-Forward !!!")
        return None, None

    print(f"\n{'='*120}")
    print(f"  WALK-FORWARD: {len(top_combos)} combos x 6 janelas")
    print(f"  Combos testados: {[(f'{s:.2f}x/{t:.1f}x') for s,t in top_combos]}")
    print(f"{'='*120}")

    splits = opt.get_window_indices()
    print(f"\n  Janelas ({N_SPLITS} splits):")
    for i, (s, e) in enumerate(splits):
        ts_start = opt.idx_arr[s]; ts_end = opt.idx_arr[min(e, opt.n) - 1]
        days = (e - s) / 24  # candles to days for 1h
        print(f"    W{i+1}: candles [{s:>5}, {e:>5}) | {ts_start} ~ {ts_end} | ~{days:.0f}d")

    # Walk-Forward: treinar na janela i, testar na janela i+1
    # Para cada WF step, encontrar melhor SL/TP no treino, aplicar no teste
    wf_results = []  # lista de dicts com train/test results

    print(f"\n  {'Step':>4} {'Train W':>8} {'Test W':>7} {'Best SL':>8} {'Best TP':>8} {'T_Trades':>9} {'T_WR%':>6} {'T_PF':>6} {'T_PnL%':>8} "
          f"{'OOS_Trades':>10} {'OOS_WR%':>8} {'OOS_PF':>7} {'OOS_PnL%':>9} {'OOS_BH%':>8}")
    print(f"  {'-'*130}")

    all_oos_trades = []

    for step in range(N_SPLITS - 1):
        train_start, train_end = splits[step]
        test_start, test_end = splits[step + 1]

        # Treinar: encontrar melhor SL/TP na janela de treino
        best_train = None; best_score = -1
        for sl, tp in top_combos:
            r = opt.run_sl_tp(sl, tp, train_start, train_end)
            if r.score > best_score:
                best_score = r.score
                best_train = r

        if best_train is None or best_train.score == 0:
            # Nenhum viavel no treino — pegar o que tem mais trades
            for sl, tp in top_combos:
                r = opt.run_sl_tp(sl, tp, train_start, train_end)
                if best_train is None or r.total_trades > best_train.total_trades:
                    best_train = r

        # Testar OOS com o melhor SL/TP do treino
        best_sl = best_train.sl_atr_mult
        best_tp = best_train.tp_atr_mult
        oos_result = opt.run_sl_tp(best_sl, best_tp, test_start, test_end)
        oos_result.window = f"oos_w{step+2}"
        oos_result.phase = "wf_oos"

        # Coletar trades OOS para metricas agregadas
        p = {**BASE_PARAMS, 'sl_atr_mult': best_sl, 'tp_atr_mult': best_tp}
        oos_trades = opt.fast_sim(p, test_start, test_end)
        all_oos_trades.extend(oos_trades)

        bh_test = oos_result.buy_hold_pct
        beat_bh = "*" if oos_result.total_pnl_pct > bh_test else " "

        print(f"  {step+1:>4} W{step+1:>7} W{step+2:>6} {best_sl:>8.2f} {best_tp:>8.1f} "
              f"{best_train.total_trades:>9} {best_train.win_rate:>6.1f} {best_train.profit_factor:>6.2f} {best_train.total_pnl_pct:>8.2f} "
              f"{oos_result.total_trades:>10} {oos_result.win_rate:>8.1f} {oos_result.profit_factor:>7.2f} {oos_result.total_pnl_pct:>9.2f} {bh_test:>8.2f}{beat_bh}")

        wf_results.append({
            'step': step + 1, 'train_window': step + 1, 'test_window': step + 2,
            'sl': best_sl, 'tp': best_tp,
            'train_trades': best_train.total_trades, 'train_wr': best_train.win_rate,
            'train_pf': best_train.profit_factor, 'train_pnl': best_train.total_pnl_pct,
            'oos_trades': oos_result.total_trades, 'oos_wr': oos_result.win_rate,
            'oos_pf': oos_result.profit_factor, 'oos_pnl': oos_result.total_pnl_pct,
            'oos_dd': oos_result.max_drawdown_pct, 'oos_bh': oos_result.buy_hold_pct,
            'oos_beat_bh': bool(oos_result.total_pnl_pct > oos_result.buy_hold_pct)
        })

    # Metricas agregadas OOS
    print(f"\n{'='*120}")
    print(f"  WALK-FORWARD: Metricas Agregadas OOS")
    print(f"{'='*120}")

    if all_oos_trades:
        oos_metrics = calculate_metrics(all_oos_trades, opt.df_clean)
        oos_bh = (opt.close[-1] - opt.close[splits[1][0]]) / opt.close[splits[1][0]] * 100
        print(f"  Total OOS Trades:  {oos_metrics.total_trades}")
        print(f"  Win Rate:          {oos_metrics.win_rate:.1f}%")
        print(f"  Profit Factor:     {oos_metrics.profit_factor:.2f}")
        print(f"  PnL OOS:           {oos_metrics.total_pnl_pct:+.2f}%")
        print(f"  B&H OOS:           {oos_bh:+.2f}%")
        print(f"  Max DD:            {oos_metrics.max_drawdown_pct:.2f}%")
        print(f"  Sharpe:            {oos_metrics.sharpe_ratio:.2f}")
        print(f"  Avg Win:           {oos_metrics.avg_win_pct:+.2f}%")
        print(f"  Avg Loss:          {oos_metrics.avg_loss_pct:+.2f}%")
        print(f"  Supera B&H:        {'SIM' if oos_metrics.total_pnl_pct > oos_bh else 'NAO'}")
    else:
        oos_metrics = None
        oos_bh = 0

    # Estatisticas por step
    steps_beat = sum(1 for w in wf_results if w['oos_beat_bh'])
    avg_oos_pnl = np.mean([w['oos_pnl'] for w in wf_results]) if wf_results else 0
    avg_oos_wr = np.mean([w['oos_wr'] for w in wf_results]) if wf_results else 0
    avg_oos_pf = np.mean([w['oos_pf'] for w in wf_results]) if wf_results else 0
    print(f"\n  Steps que superam B&H: {steps_beat}/{len(wf_results)}")
    print(f"  Avg OOS PnL/step:      {avg_oos_pnl:+.2f}%")
    print(f"  Avg OOS WR:           {avg_oos_wr:.1f}%")
    print(f"  Avg OOS PF:           {avg_oos_pf:.2f}")

    # Melhor SL/TP consistente (mais frequente nos WF steps)
    sl_tp_counts = {}
    for w in wf_results:
        key = (w['sl'], w['tp'])
        sl_tp_counts[key] = sl_tp_counts.get(key, 0) + 1
    most_common = max(sl_tp_counts.items(), key=lambda x: x[1])
    print(f"\n  SL/TP mais frequente nos WF steps: {most_common[0][0]:.2f}x / {most_common[0][1]:.1f}x ({most_common[1]}/{len(wf_results)} steps)")

    return wf_results, oos_metrics


def find_best_full_oos(opt: P2Optimizer, wf_results):
    """Encontra o melhor SL/TP considerando P2 full + WF OOS."""
    if not wf_results:
        return None

    # Para cada SL/TP unico que apareceu nos WF steps, rodar no full period
    unique_sl_tp = list(set((w['sl'], w['tp']) for w in wf_results))
    print(f"\n{'='*120}")
    print(f"  VALIDACAO FINAL: Top SL/TP dos WF steps no periodo completo")
    print(f"{'='*120}")

    best_overall = None; best_score = -1
    for sl, tp in unique_sl_tp:
        r = opt.run_sl_tp(sl, tp)
        r.window = "full_validation"; r.phase = "final"
        bh_flag = "*" if r.total_pnl_pct > r.buy_hold_pct else " "
        rr = r.tp_atr_mult / r.sl_atr_mult
        print(f"  SL={sl:>5.2f} TP={tp:>5.1f} R:R={rr:>5.1f}:1 | T={r.total_trades:>4} WR={r.win_rate:>5.1f}% PF={r.profit_factor:>5.2f} "
              f"PnL={r.total_pnl_pct:>+8.2f}% DD={r.max_drawdown_pct:>5.2f}% Sharpe={r.sharpe_ratio:>6.2f} B&H={r.buy_hold_pct:>+7.2f}%{bh_flag} Score={r.score:.3f}")
        if r.score > best_score:
            best_score = r.score
            best_overall = r

    # Tambem testar o SL/TP mais frequente com variacoes finas
    if best_overall:
        sl_center = best_overall.sl_atr_mult
        tp_center = best_overall.tp_atr_mult
        fine_sl = [sl_center - 0.25, sl_center - 0.125, sl_center, sl_center + 0.125, sl_center + 0.25]
        fine_tp = [tp_center - 1.0, tp_center - 0.5, tp_center, tp_center + 0.5, tp_center + 1.0]
        fine_sl = [s for s in fine_sl if s >= 0.5]
        fine_tp = [t for t in fine_tp if t >= 1.0]

        print(f"\n  Refinamento fino ao redor de SL={sl_center:.2f}x / TP={tp_center:.1f}x:")
        for sl in fine_sl:
            for tp in fine_tp:
                r = opt.run_sl_tp(sl, tp)
                r.window = "full_fine"; r.phase = "final_fine"
                bh_flag = "*" if r.total_pnl_pct > r.buy_hold_pct else " "
                rr = r.tp_atr_mult / r.sl_atr_mult
                if r.total_trades > 0 and r.score > 0:
                    print(f"    SL={sl:>5.2f} TP={tp:>5.1f} R:R={rr:>5.1f}:1 | T={r.total_trades:>4} WR={r.win_rate:>5.1f}% PF={r.profit_factor:>5.2f} "
                          f"PnL={r.total_pnl_pct:>+8.2f}% DD={r.max_drawdown_pct:>5.2f}%{bh_flag} Score={r.score:.3f}")
                if r.score > best_score:
                    best_score = r.score
                    best_overall = r

    return best_overall


def main():
    t0 = time.time()
    print("\n" + "#" * 120)
    print("#  CTEV P2 + Walk-Forward — Refinamento SL/TP ao redor de 2.5x/12.0x")
    print("#  Timeframe: 1h | Symbol: BTC/USDT | Period: 730 days")
    print("#" * 120)

    opt = P2Optimizer()
    print("\n[1/4] Carregando dados...")
    opt.load_data()

    print("\n[2/4] P2: Grid SL/TP...")
    p2_top = run_p2_full(opt)

    print("\n[3/4] Walk-Forward 6 janelas...")
    wf_results, oos_metrics = run_walk_forward(opt, top_n=15)

    print("\n[4/4] Validacao final...")
    winner = find_best_full_oos(opt, wf_results)

    tt = time.time() - t0
    print(f"\n{'#'*120}")
    if winner:
        rr = winner.tp_atr_mult / winner.sl_atr_mult
        print(f"#  VENCEDOR P2+WF")
        print(f"{'#'*120}")
        print(f"#  SL/TP:         {winner.sl_atr_mult:.2f}x / {winner.tp_atr_mult:.1f}x (R:R {rr:.1f}:1)")
        print(f"#  Trades:        {winner.total_trades} ({winner.total_trades/730:.2f}/dia)")
        print(f"#  Win Rate:      {winner.win_rate:.1f}%")
        print(f"#  Profit Factor: {winner.profit_factor:.2f}")
        print(f"#  PnL:           {winner.total_pnl_pct:+.2f}% (B&H={winner.buy_hold_pct:+.2f}%)")
        print(f"#  Max DD:        {winner.max_drawdown_pct:.2f}%  Sharpe: {winner.sharpe_ratio:.2f}")
        print(f"#  Score:         {winner.score:.4f}")
        print(f"#  Supera B&H:    {'SIM' if winner.total_pnl_pct > winner.buy_hold_pct else 'NAO'}")
        print(f"#  Tempo total:   {tt:.0f}s")
        print(f"{'#'*120}")

        # Salvar resultados
        out_dir = os.path.join(SCRIPT_DIR, "..", "download")
        os.makedirs(out_dir, exist_ok=True)

        save_data = {
            'version': 'p2_wf',
            'winner': {
                'sl_atr_mult': winner.sl_atr_mult,
                'tp_atr_mult': winner.tp_atr_mult,
                'rr_ratio': round(winner.tp_atr_mult / winner.sl_atr_mult, 2),
                'total_trades': winner.total_trades,
                'win_rate': winner.win_rate,
                'profit_factor': winner.profit_factor,
                'total_pnl_pct': winner.total_pnl_pct,
                'max_drawdown_pct': winner.max_drawdown_pct,
                'sharpe_ratio': winner.sharpe_ratio,
                'buy_hold_pct': winner.buy_hold_pct,
                'score': winner.score,
            },
            'wf_steps': wf_results,
            'oos_aggregate': {
                'total_trades': oos_metrics.total_trades if oos_metrics else 0,
                'win_rate': round(oos_metrics.win_rate, 2) if oos_metrics else 0,
                'profit_factor': round(oos_metrics.profit_factor, 4) if oos_metrics else 0,
                'total_pnl_pct': round(oos_metrics.total_pnl_pct, 4) if oos_metrics else 0,
                'max_drawdown_pct': round(oos_metrics.max_drawdown_pct, 4) if oos_metrics else 0,
                'sharpe_ratio': round(oos_metrics.sharpe_ratio, 4) if oos_metrics else 0,
            } if oos_metrics else None,
            'time_sec': round(tt, 1)
        }
        with open(os.path.join(out_dir, 'winner_p2_wf.json'), 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"\n  Salvo: winner_p2_wf.json")
    else:
        print("#  NENHUM VENCEDOR ENCONTRADO")
        print(f"{'#'*120}")

    return winner


if __name__ == "__main__":
    main()
