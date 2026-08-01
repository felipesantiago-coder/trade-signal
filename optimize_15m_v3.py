"""
optimize_15m_v3.py
-----------------
Abordagem FUNDAMENTALMENTE diferente para 15m.

O trend-following com pullback (CTEV 1h) NAO funciona em 15m porque:
  1. ATR em 15m e ~0.15-0.30% → custos ~0.35% >> ATR
  2. Ruido destrói o sinal de pullback
  3. Tendencias em 15m duram poucos candles

Abordagem alternativa para 15m:
  A) BB Bounce (Mean-Reversion): Entrar no toco da BB e capturar
     retorno a media (SMA20). SL justo, TP na SMA20.
  B) RSI Extreme + BB: RSI < 25 + preco na BB lower → LONG.
     RSI > 75 + preco na BB upper → SHORT.
  C) EMA Cross Momentum: EMA8 > EMA21 + RSI > 50 → LONG.
  D) Volatility Breakout: BB squeeze → expansao → entrada na direcao.
  E) Trend-Lite: Simplificacao do CTEV sem pullback, sem fib,
     apenas regime + RSI + EMA trend.
"""
from __future__ import annotations
import sys, os, time, math, json, pickle
from dataclasses import dataclass, asdict
from typing import List

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from backtest import _apply_costs, fetch_historical_ohlcv, calculate_metrics, TradeResult
from indicators import compute_indicators

SYMBOL = "BTC/USDT"; TIMEFRAME = "15m"; DAYS = 365
DATA_CACHE = "/tmp/ctev_15m_data.pkl"
FEE = 0.025; SPREAD = 10.0; SLIPPAGE = 25.0
MIN_TRADES = 80; MIN_WR = 48.0; MAX_DD = 20.0


def load_data():
    if os.path.exists(DATA_CACHE):
        with open(DATA_CACHE, 'rb') as f:
            return pickle.load(f)['df_clean']
    df = fetch_historical_ohlcv(SYMBOL, TIMEFRAME, DAYS)
    df = compute_indicators(df, timeframe=TIMEFRAME)
    crit = ["ema20","ema50","ema200","rsi","atr","atr_percentile",
            "adx","plus_di","minus_di","regime","bb_lower","bb_upper","bb_width","bb_squeeze_pct"]
    df = df.dropna(subset=crit).copy()
    with open(DATA_CACHE, 'wb') as f:
        pickle.dump({'df_clean': df}, f)
    return df


@dataclass
class R:
    cid: int; score: float = 0.0; strategy: str = ""
    total_trades: int = 0; long_trades: int = 0; short_trades: int = 0
    wins: int = 0; losses: int = 0; win_rate: float = 0.0
    profit_factor: float = 0.0; total_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0; sharpe_ratio: float = 0.0
    avg_bars_held: float = 0.0; avg_win_pct: float = 0.0; avg_loss_pct: float = 0.0
    buy_hold_pct: float = 0.0
    sl_m: float = 1.5; tp_m: float = 3.5; max_bars: int = 72
    # Strategy-specific params
    p: dict = None

    def to_dict(self):
        d = asdict(self)
        d['params'] = self.p
        return d


def calc_score(r):
    if r.total_trades < MIN_TRADES or r.profit_factor <= 1.0: return 0.0
    if r.win_rate < MIN_WR or r.max_drawdown_pct > MAX_DD: return 0.0
    wr = r.win_rate / 100.0
    pf = r.profit_factor
    dd_pen = max(0, 1.0 - r.max_drawdown_pct / 100.0)
    freq = math.log(max(r.total_trades, 1)) / math.log(365)
    pnl_b = 2.0 if r.total_pnl_pct > 10 else (1.5 if r.total_pnl_pct > 5 else (1.2 if r.total_pnl_pct > 0 else 0.5))
    sig = min(math.sqrt(r.total_trades) / 15.0, 2.0)
    return round(wr * pf * freq * dd_pen * pnl_b * sig, 4)


def sim_trades(entry_indices, directions, sl_prices, tp_prices, entries,
               close, high, low, max_bars=72):
    """Fast trade simulation given pre-computed signals."""
    trades = []
    n = len(close)
    for k in range(len(entry_indices)):
        i = entry_indices[k]
        is_long = directions[k]
        sl = sl_prices[k]; tp = tp_prices[k]; ep = entries[k]
        exit_price = None; bars = 0
        mj = min(i + max_bars, n)
        if is_long:
            for j in range(i + 1, mj):
                if low[j] <= sl: exit_price = sl; bars = j - i; break
                if high[j] >= tp: exit_price = tp; bars = j - i; break
        else:
            for j in range(i + 1, mj):
                if high[j] >= sl: exit_price = sl; bars = j - i; break
                if low[j] <= tp: exit_price = tp; bars = j - i; break
        if exit_price is None:
            bars = mj - 1 - i; exit_price = close[mj - 1]
        _, adj, _ = _apply_costs(ep, exit_price, is_long, FEE, SPREAD, SLIPPAGE)
        pnl = (adj - ep) / ep * 100 if is_long else (ep - adj) / ep * 100
        trades.append((is_long, pnl, bars))
    return trades


def metrics_from_trades(trades, buy_hold):
    if not trades: return R(0, buy_hold_pct=buy_hold)
    total = len(trades)
    wins = sum(1 for t in trades if t[1] > 0)
    losses = total - wins
    wr = wins / total * 100
    longs = sum(1 for t in trades if t[0])
    pnls = [t[1] for t in trades]
    wins_p = [p for p in pnls if p > 0]
    loss_p = [p for p in pnls if p <= 0]
    gw = sum(wins_p); gl = abs(sum(loss_p))
    pf = gw / gl if gl > 0 else 999.0
    tpnl = sum(pnls)
    aw = np.mean(wins_p) if wins_p else 0
    al = np.mean(loss_p) if loss_p else 0
    eq = [100.0]
    for p in pnls: eq.append(eq[-1] * (1 + p / 100))
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak * 100
    mdd = np.max(dd)
    sharpe = np.mean(pnls) / (np.std(pnls) + 1e-9) * math.sqrt(365 / max(total / 96, 1))
    ab = np.mean([t[2] for t in trades])
    return R(0, total_trades=total, long_trades=longs, short_trades=total-longs,
              wins=wins, losses=losses, win_rate=round(wr,2),
              profit_factor=round(pf,4), total_pnl_pct=round(tpnl,4),
              max_drawdown_pct=round(mdd,4), sharpe_ratio=round(sharpe,4),
              avg_bars_held=round(ab,1), avg_win_pct=round(aw,4), avg_loss_pct=round(al,4),
              buy_hold_pct=buy_hold)


# ======================================================================
# STRATEGY A: BB Bounce (Mean-Reversion)
# ======================================================================
def strat_a_bb_bounce(df, params):
    """BB Bounce: buy at lower band, sell at upper band."""
    rsi_min = params['rsi_min']  # e.g., 20, 25, 30
    rsi_max_long = params['rsi_max_long']  # e.g., 40, 45
    rsi_max_short = params['rsi_min_short']  # e.g., 60, 65
    rsi_max_short_exit = params['rsi_max_short_exit']  # e.g., 75, 80
    sl_m = params['sl_m']; tp_m = params['tp_m']; max_bars = params['max_bars']
    require_vol = params.get('require_vol', False)
    vol_ratio = params.get('vol_ratio', 0.5)

    c = df['close'].values; lo = df['low'].values; hi = df['high'].values
    bbl = df['bb_lower'].values; bbu = df['bb_upper'].values; bbm = df['bb_middle'].values
    rsi = df['rsi'].values; atr = df['atr'].values; vol = df['volume'].values
    vs = df['volume_sma50'].values; ap = df['atr_percentile'].values
    n = len(c)

    entries = []; dirs = []; sls = []; tps = []; eps = []
    i = 0
    while i < n:
        # LONG: RSI oversold + price at/below BB lower
        if rsi[i] <= rsi_max_long and lo[i] <= bbl[i]:
            if require_vol and not np.isnan(vs[i]) and vs[i] > 0:
                if vol[i] < vs[i] * vol_ratio:
                    i += 1; continue
            ep = c[i]; sl = ep - sl_m * atr[i]; tp = ep + tp_m * atr[i]
            if sl > 0:
                entries.append(i); dirs.append(True); sls.append(sl); tps.append(tp); eps.append(ep)
                i += max(4, int(sl_m * 2))  # skip ahead
                continue
        # SHORT: RSI overbought + price at/above BB upper
        elif rsi[i] >= rsi_max_short and hi[i] >= bbu[i]:
            if require_vol and not np.isnan(vs[i]) and vs[i] > 0:
                if vol[i] < vs[i] * vol_ratio:
                    i += 1; continue
            ep = c[i]; sl = ep + sl_m * atr[i]; tp = ep - tp_m * atr[i]
            entries.append(i); dirs.append(False); sls.append(sl); tps.append(tp); eps.append(ep)
            i += max(4, int(sl_m * 2))
            continue
        i += 1

    if not entries: return R(0, strategy='A-BB-Bounce', sl_m=sl_m, tp_m=tp_m, max_bars=max_bars, p=params)
    trades = sim_trades(entries, dirs, sls, tps, eps, c, hi, lo, max_bars)
    r = metrics_from_trades(trades, 0)
    r.strategy = 'A-BB-Bounce'; r.sl_m = sl_m; r.tp_m = tp_m; r.max_bars = max_bars; r.p = params
    return r


# ======================================================================
# STRATEGY B: RSI Extreme + BB + Regime
# ======================================================================
def strat_b_rsi_extreme(df, params):
    """RSI extreme (oversold/overbought) + BB confirmation + regime filter."""
    rsi_long_max = params['rsi_long_max']  # e.g., 25, 30, 35
    rsi_short_min = params['rsi_short_min']  # e.g., 70, 75, 80
    use_bb = params.get('use_bb', True)
    allow_ranging = params.get('allow_ranging', True)
    sl_m = params['sl_m']; tp_m = params['tp_m']; max_bars = params['max_bars']
    adx_min = params.get('adx_min', 0)
    vol_c = params.get('vol_c', False)

    c = df['close'].values; lo = df['low'].values; hi = df['high'].values
    bbl = df['bb_lower'].values; bbu = df['bb_upper'].values
    rsi = df['rsi'].values; atr = df['atr'].values
    regime = df['regime'].values; adx = df['adx'].values
    vol = df['volume'].values; vs = df['volume_sma50'].values
    e50 = df['ema50'].values; e200 = df['ema200'].values
    n = len(c)

    entries = []; dirs = []; sls = []; tps = []; eps = []
    i = 0
    while i < n:
        rg = regime[i]
        # Skip volatile
        if rg == 'volatile': i += 1; continue
        # For LONG: trending_up or transition (or ranging if allowed)
        if rsi[i] <= rsi_long_max:
            ok_regime = rg in ('trending_up', 'transition') or (allow_ranging and rg == 'ranging')
            if ok_regime:
                if rg == 'trending_up' and adx[i] < adx_min: i += 1; continue
                if use_bb and lo[i] > bbl[i]: i += 1; continue  # price not at BB
                if vol_c and not np.isnan(vs[i]) and vs[i] > 0 and vol[i] < vs[i] * 0.3: i += 1; continue
                # Optional: require price above EMA50 (trend filter)
                if params.get('req_ema_trend', False) and not (c[i] > e50[i]): i += 1; continue
                ep = c[i]; sl = ep - sl_m * atr[i]; tp = ep + tp_m * atr[i]
                if sl > 0:
                    entries.append(i); dirs.append(True); sls.append(sl); tps.append(tp); eps.append(ep)
                    i += max(4, int(sl_m * 2)); continue
        # For SHORT
        elif rsi[i] >= rsi_short_min:
            ok_regime = rg in ('trending_down', 'transition') or (allow_ranging and rg == 'ranging')
            if ok_regime:
                if rg == 'trending_down' and adx[i] < adx_min: i += 1; continue
                if use_bb and hi[i] < bbu[i]: i += 1; continue
                if vol_c and not np.isnan(vs[i]) and vs[i] > 0 and vol[i] < vs[i] * 0.3: i += 1; continue
                if params.get('req_ema_trend', False) and not (c[i] < e50[i]): i += 1; continue
                ep = c[i]; sl = ep + sl_m * atr[i]; tp = ep - tp_m * atr[i]
                entries.append(i); dirs.append(False); sls.append(sl); tps.append(tp); eps.append(ep)
                i += max(4, int(sl_m * 2)); continue
        i += 1

    if not entries: return R(0, strategy='B-RSI-Extreme', sl_m=sl_m, tp_m=tp_m, max_bars=max_bars, p=params)
    trades = sim_trades(entries, dirs, sls, tps, eps, c, hi, lo, max_bars)
    r = metrics_from_trades(trades, 0)
    r.strategy = 'B-RSI-Extreme'; r.sl_m = sl_m; r.tp_m = tp_m; r.max_bars = max_bars; r.p = params
    return r


# ======================================================================
# STRATEGY C: EMA Cross Momentum
# ======================================================================
def strat_c_ema_cross(df, params):
    """Fast EMA cross with momentum confirmation."""
    # Compute fast EMAs
    fast_span = params.get('fast_ema', 8)
    slow_span = params.get('slow_ema', 21)
    rsi_min = params.get('rsi_min', 45)
    rsi_max_long = params.get('rsi_max_long', 70)
    rsi_min_short = params.get('rsi_min_short', 30)
    rsi_max_short = params.get('rsi_max_short', 55)
    sl_m = params['sl_m']; tp_m = params['tp_m']; max_bars = params['max_bars']
    vol_c = params.get('vol_c', False)

    c = df['close'].values; lo = df['low'].values; hi = df['high'].values
    rsi = df['rsi'].values; atr = df['atr'].values
    vol = df['volume'].values; vs = df['volume_sma50'].values
    regime = df['regime'].values
    n = len(c)

    ema_fast = pd.Series(c).ewm(span=fast_span, adjust=False).mean().values
    ema_slow = pd.Series(c).ewm(span=slow_span, adjust=False).mean().values

    entries = []; dirs = []; sls = []; tps = []; eps = []
    i = 1
    while i < n:
        rg = regime[i]
        if rg == 'volatile': i += 1; continue

        # Cross detection
        bull_cross = ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]
        bear_cross = ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]

        if bull_cross and rsi_min <= rsi[i] <= rsi_max_long:
            if vol_c and not np.isnan(vs[i]) and vs[i] > 0 and vol[i] < vs[i] * 0.3: i += 1; continue
            ep = c[i]; sl = ep - sl_m * atr[i]; tp = ep + tp_m * atr[i]
            if sl > 0:
                entries.append(i); dirs.append(True); sls.append(sl); tps.append(tp); eps.append(ep)
                i += max(4, int(sl_m * 2)); continue
        elif bear_cross and rsi_min_short <= rsi[i] <= rsi_max_short:
            if vol_c and not np.isnan(vs[i]) and vs[i] > 0 and vol[i] < vs[i] * 0.3: i += 1; continue
            ep = c[i]; sl = ep + sl_m * atr[i]; tp = ep - tp_m * atr[i]
            entries.append(i); dirs.append(False); sls.append(sl); tps.append(tp); eps.append(ep)
            i += max(4, int(sl_m * 2)); continue
        i += 1

    if not entries: return R(0, strategy='C-EMA-Cross', sl_m=sl_m, tp_m=tp_m, max_bars=max_bars, p=params)
    trades = sim_trades(entries, dirs, sls, tps, eps, c, hi, lo, max_bars)
    r = metrics_from_trades(trades, 0)
    r.strategy = 'C-EMA-Cross'; r.sl_m = sl_m; r.tp_m = tp_m; r.max_bars = max_bars; r.p = params
    return r


# ======================================================================
# STRATEGY D: BB Squeeze Breakout
# ======================================================================
def strat_d_squeeze_breakout(df, params):
    """Trade breakouts after BB squeeze (low bb_squeeze_pct)."""
    sq_max = params.get('squeeze_max', 0.20)  # squeeze = low percentile of BB width
    sl_m = params['sl_m']; tp_m = params['tp_m']; max_bars = params['max_bars']
    vol_c = params.get('vol_c', False)
    rsi_long_max = params.get('rsi_long_max', 65)
    rsi_short_min = params.get('rsi_short_min', 35)

    c = df['close'].values; lo = df['low'].values; hi = df['high'].values
    bbl = df['bb_lower'].values; bbu = df['bb_upper'].values
    rsi = df['rsi'].values; atr = df['atr'].values
    bb_sq = df['bb_squeeze_pct'].values; bb_w = df['bb_width'].values
    regime = df['regime'].values
    vol = df['volume'].values; vs = df['volume_sma50'].values
    n = len(c)

    entries = []; dirs = []; sls = []; tps = []; eps = []
    i = 1
    while i < n:
        rg = regime[i]
        if rg == 'volatile': i += 1; continue

        # Detect squeeze breakout: squeeze was low, now expanding
        if bb_sq[i-1] < sq_max and bb_w[i] > bb_w[i-1] * 1.1:
            # Direction based on price vs BB middle
            bbm = (bbl[i] + bbu[i]) / 2
            if c[i] > bbm and rsi[i] <= rsi_long_max:  # Bullish breakout
                if vol_c and not np.isnan(vs[i]) and vs[i] > 0 and vol[i] < vs[i] * 0.3: i += 1; continue
                ep = c[i]; sl = ep - sl_m * atr[i]; tp = ep + tp_m * atr[i]
                if sl > 0:
                    entries.append(i); dirs.append(True); sls.append(sl); tps.append(tp); eps.append(ep)
                    i += max(4, int(sl_m * 2)); continue
            elif c[i] < bbm and rsi[i] >= rsi_short_min:  # Bearish breakout
                if vol_c and not np.isnan(vs[i]) and vs[i] > 0 and vol[i] < vs[i] * 0.3: i += 1; continue
                ep = c[i]; sl = ep + sl_m * atr[i]; tp = ep - tp_m * atr[i]
                entries.append(i); dirs.append(False); sls.append(sl); tps.append(tp); eps.append(ep)
                i += max(4, int(sl_m * 2)); continue
        i += 1

    if not entries: return R(0, strategy='D-Squeeze-Breakout', sl_m=sl_m, tp_m=tp_m, max_bars=max_bars, p=params)
    trades = sim_trades(entries, dirs, sls, tps, eps, c, hi, lo, max_bars)
    r = metrics_from_trades(trades, 0)
    r.strategy = 'D-Squeeze-Breakout'; r.sl_m = sl_m; r.tp_m = tp_m; r.max_bars = max_bars; r.p = params
    return r


# ======================================================================
# STRATEGY E: Trend-Lite (simplified CTEV)
# ======================================================================
def strat_e_trend_lite(df, params):
    """Simplified trend: regime + RSI + EMA trend only (no pullback, no fib)."""
    rsi_l = params['rsi_l']; rsi_s = params['rsi_s']
    adx_min = params.get('adx_min', 0)
    allow_trn = params.get('allow_trn', True)
    req_et = params.get('req_et', True)
    sl_m = params['sl_m']; tp_m = params['tp_m']; max_bars = params['max_bars']
    vol_c = params.get('vol_c', False)
    atr_min = params.get('atr_min', 0.05); atr_max = params.get('atr_max', 0.95)

    c = df['close'].values; lo = df['low'].values; hi = df['high'].values
    e50 = df['ema50'].values; e200 = df['ema200'].values
    rsi = df['rsi'].values; atr = df['atr'].values; adx = df['adx'].values
    regime = df['regime'].values; ap = df['atr_percentile'].values
    vol = df['volume'].values; vs = df['volume_sma50'].values
    n = len(c)

    entries = []; dirs = []; sls = []; tps = []; eps = []
    i = 0
    while i < n:
        rg = regime[i]
        # LONG
        if rg in ('trending_up', 'transition'):
            if rg == 'trending_up' and adx[i] < adx_min: i += 1; continue
            if rg == 'transition' and not allow_trn: i += 1; continue
            if req_et and not (c[i] > e50[i] > e200[i]): i += 1; continue
            if not (rsi_l[0] <= rsi[i] <= rsi_l[1]): i += 1; continue
            if vol_c and not np.isnan(vs[i]) and vs[i] > 0 and vol[i] < vs[i] * 0.3: i += 1; continue
            if not (atr_min <= ap[i] <= atr_max): i += 1; continue
            ep = c[i]; sl = ep - sl_m * atr[i]; tp = ep + tp_m * atr[i]
            if sl > 0:
                entries.append(i); dirs.append(True); sls.append(sl); tps.append(tp); eps.append(ep)
                i += max(4, int(sl_m * 2)); continue
        # SHORT
        elif rg in ('trending_down', 'transition'):
            if rg == 'trending_down' and adx[i] < adx_min: i += 1; continue
            if rg == 'transition' and not allow_trn: i += 1; continue
            if req_et and not (c[i] < e50[i] < e200[i]): i += 1; continue
            if not (rsi_s[0] <= rsi[i] <= rsi_s[1]): i += 1; continue
            if vol_c and not np.isnan(vs[i]) and vs[i] > 0 and vol[i] < vs[i] * 0.3: i += 1; continue
            if not (atr_min <= ap[i] <= atr_max): i += 1; continue
            ep = c[i]; sl = ep + sl_m * atr[i]; tp = ep - tp_m * atr[i]
            entries.append(i); dirs.append(False); sls.append(sl); tps.append(tp); eps.append(ep)
            i += max(4, int(sl_m * 2)); continue
        i += 1

    if not entries: return R(0, strategy='E-Trend-Lite', sl_m=sl_m, tp_m=tp_m, max_bars=max_bars, p=params)
    trades = sim_trades(entries, dirs, sls, tps, eps, c, hi, lo, max_bars)
    r = metrics_from_trades(trades, 0)
    r.strategy = 'E-Trend-Lite'; r.sl_m = sl_m; r.tp_m = tp_m; r.max_bars = max_bars; r.p = params
    return r


def make_grids():
    all_combos = []  # (strategy_func, params_dict, label)

    # ── STRATEGY A: BB Bounce ──
    for rsi_min_l in [20, 25, 30, 35]:
        for rsi_max_l in [40, 45, 50]:
            for rsi_min_s in [55, 60, 65]:
                for rsi_max_s in [75, 80]:
                    for sl, tp in [(0.75,1.5),(1.0,2.0),(1.0,2.5),(1.5,2.5),(1.5,3.0),(2.0,3.0),(2.0,4.0)]:
                        for vol in [False, True]:
                            p = {'rsi_min': rsi_min_l, 'rsi_max_long': rsi_max_l,
                                 'rsi_min_short': rsi_min_s, 'rsi_max_short_exit': rsi_max_s,
                                 'sl_m': sl, 'tp_m': tp, 'max_bars': 48,
                                 'require_vol': vol, 'vol_ratio': 0.5}
                            all_combos.append((strat_a_bb_bounce, p, f'A-BB-Bounce rsiL<={rsi_max_l} rsiS>={rsi_min_s} vol={vol}'))

    # ── STRATEGY B: RSI Extreme ──
    for rsi_l in [25, 30, 35, 40]:
        for rsi_s in [60, 65, 70, 75]:
            for sl, tp in [(0.75,1.5),(1.0,2.0),(1.5,2.5),(1.5,3.0),(2.0,3.0),(2.0,4.0)]:
                for use_bb in [True, False]:
                    for allow_rng in [True, False]:
                        for adx in [0, 15, 20]:
                            p = {'rsi_long_max': rsi_l, 'rsi_short_min': rsi_s,
                                 'sl_m': sl, 'tp_m': tp, 'max_bars': 48,
                                 'use_bb': use_bb, 'allow_ranging': allow_rng,
                                 'adx_min': adx, 'vol_c': False, 'req_ema_trend': False}
                            all_combos.append((strat_b_rsi_extreme, p, f'B-RSI-Extreme rsiL<={rsi_l} rsiS>={rsi_s}'))

    # ── STRATEGY C: EMA Cross ──
    for fast in [5, 8, 10, 13]:
        for slow in [13, 21, 34]:
            if fast >= slow: continue
            for sl, tp in [(0.75,1.5),(1.0,2.0),(1.5,2.5),(1.5,3.0),(2.0,3.0)]:
                for vol in [False, True]:
                    p = {'fast_ema': fast, 'slow_ema': slow,
                         'rsi_min': 40, 'rsi_max_long': 70,
                         'rsi_min_short': 30, 'rsi_max_short': 60,
                         'sl_m': sl, 'tp_m': tp, 'max_bars': 48, 'vol_c': vol}
                    all_combos.append((strat_c_ema_cross, p, f'C-EMA-Cross EMA{fast}/{slow}'))

    # ── STRATEGY D: Squeeze Breakout ──
    for sq in [0.10, 0.15, 0.20, 0.30]:
        for sl, tp in [(1.0,2.0),(1.5,2.5),(1.5,3.0),(2.0,3.0),(2.0,4.0),(2.5,4.0)]:
            for rsi_l in [55, 60, 65, 70]:
                for rsi_s in [30, 35, 40, 45]:
                    p = {'squeeze_max': sq, 'sl_m': sl, 'tp_m': tp, 'max_bars': 48,
                         'vol_c': False, 'rsi_long_max': rsi_l, 'rsi_short_min': rsi_s}
                    all_combos.append((strat_d_squeeze_breakout, p, f'D-Squeeze sq<{sq}'))

    # ── STRATEGY E: Trend-Lite ──
    for rsi_l in [(25,50),(28,48),(30,55),(25,55),(20,50),(30,50)]:
        for rsi_s in [(50,75),(55,75),(45,70),(50,70)]:
            for sl, tp in [(0.75,1.5),(1.0,2.0),(1.0,2.5),(1.25,2.5),(1.5,2.5),(1.5,3.0),(2.0,3.0),(2.0,4.0)]:
                for adx in [0, 15, 20, 25]:
                    for et in [True, False]:
                        for trn in [True, False]:
                            p = {'rsi_l': rsi_l, 'rsi_s': rsi_s, 'sl_m': sl, 'tp_m': tp,
                                 'max_bars': 72, 'adx_min': adx, 'allow_trn': trn,
                                 'req_et': et, 'vol_c': False, 'atr_min': 0.05, 'atr_max': 0.95}
                            all_combos.append((strat_e_trend_lite, p, f'E-Trend-Lite adx={adx}'))

    return all_combos


def print_top(results, title, n=15):
    print(f"\n{'='*130}")
    print(f"  {title}")
    print(f"{'='*130}")
    print(f"{'#':>3} {'Score':>7} {'Trades':>6} {'L/S':>6} {'T/d':>5} {'WR%':>6} {'PF':>6} {'PnL%':>8} {'DD%':>6} {'Sharpe':>6} {'Strategy':>20} {'SL':>4} {'TP':>4}")
    print(f"{'-'*130}")
    for i, r in enumerate(sorted(results, key=lambda x: x.score, reverse=True)[:n]):
        ls = f"{r.long_trades}/{r.short_trades}"
        td = r.total_trades / 365
        print(f"{i+1:>3} {r.score:>7.3f} {r.total_trades:>6} {ls:>6} {td:>5.2f} {r.win_rate:>6.1f} {r.profit_factor:>6.2f} "
              f"{r.total_pnl_pct:>8.2f} {r.max_drawdown_pct:>6.2f} {r.sharpe_ratio:>6.2f} {r.strategy:>20} {r.sl_m:>4.2f} {r.tp_m:>4.2f}")


def main():
    t0 = time.time()
    print("\n" + "#" * 130)
    print("#  CTEV 15m OPTIMIZER v3 — MULTI-STRATEGY SEARCH")
    print(f"#  Custo: fee={FEE}% spread={SPREAD}bps slip={SLIPPAGE}bps")
    print("#" * 130)

    print("\n[1/3] Carregando dados 15m...")
    df = load_data()
    n = len(df)
    buy_hold = (df['close'].values[-1] - df['close'].values[0]) / df['close'].values[0] * 100
    rc = dict(zip(*np.unique(df['regime'].values, return_counts=True)))
    print(f"{n} candles ({n//96} dias) | B&H={buy_hold:.2f}% | {rc}")

    print("\n[2/3] Gerando grid multi-estrategia...")
    grid = make_grids()
    print(f"{len(grid):,} combinacoes total")

    # Count per strategy
    from collections import Counter
    strat_counts = Counter()
    for _, _, label in grid:
        strat_counts[label.split()[0]] += 1
    for s, c in strat_counts.most_common():
        print(f"  {s}: {c:,} combos")

    print("\n[3/3] Executando simulacoes...")
    results = []
    viable = 0
    best_score = 0; best_str = ""
    t1 = time.time()

    for idx, (func, params, label) in enumerate(grid):
        r = func(df, params)
        r.cid = idx
        r.score = calc_score(r)
        results.append(r)
        if r.score > 0: viable += 1
        if r.score > best_score:
            best_score = r.score
            best_str = (f"{r.strategy} T={r.total_trades} WR={r.win_rate:.1f}% "
                       f"PF={r.profit_factor:.2f} PnL={r.total_pnl_pct:+.2f}% DD={r.max_drawdown_pct:.2f}%")
        if (idx + 1) % 1000 == 0:
            el = time.time() - t1; spd = (idx + 1) / max(el, 0.01)
            eta = (len(grid) - idx - 1) / max(spd, 0.01)
            print(f"  [{idx+1:,}/{len(grid):,}] {spd:.0f}/s viable={viable} ETA={eta:.0f}s")
            if best_str: print(f"    BEST: {best_str}")

    elapsed = time.time() - t1
    print(f"\n  Concluido: {len(grid):,} combos em {elapsed:.1f}s | viable={viable}")

    # Per-strategy best
    print("\n" + "=" * 130)
    print("  MELHOR POR ESTRATEGIA")
    print("=" * 130)
    strat_groups = {}
    for r in results:
        s = r.strategy
        if s not in strat_groups or r.score > strat_groups[s].score:
            strat_groups[s] = r
    for s, r in sorted(strat_groups.items(), key=lambda x: x[1].score, reverse=True):
        print(f"  {s:>20}: Score={r.score:.3f} T={r.total_trades} WR={r.win_rate:.1f}% PF={r.profit_factor:.2f} PnL={r.total_pnl_pct:+.2f}% DD={r.max_drawdown_pct:.2f}% SL={r.sl_m:.2f} TP={r.tp_m:.2f}")

    print_top(results, f"TOP 20 GERAL ({len(results):,} combos)", 20)

    # Save
    winner = max(results, key=lambda x: x.score) if results else None
    if winner:
        print(f"\n{'#'*130}")
        print(f"#  VENCEDOR 15m: {winner.strategy}")
        print(f"{'#'*130}")
        print(f"#  Score:         {winner.score:.4f}")
        print(f"#  Trades:        {winner.total_trades} ({winner.total_trades/365:.2f}/dia) L={winner.long_trades} S={winner.short_trades}")
        print(f"#  Win Rate:      {winner.win_rate:.1f}%")
        print(f"#  Profit Factor: {winner.profit_factor:.2f}")
        print(f"#  PnL:           {winner.total_pnl_pct:+.2f}% (B&H={winner.buy_hold_pct:+.2f}%)")
        print(f"#  Max DD:        {winner.max_drawdown_pct:.2f}%  Sharpe: {winner.sharpe_ratio:.2f}")
        print(f"#  SL/TP:         {winner.sl_m:.2f}x / {winner.tp_m:.2f}x (R:R {winner.tp_m/winner.sl_m:.1f}:1)")
        print(f"#  Params:        {json.dumps(winner.p, indent=2, default=str)}")
        print(f"#  Total:         {len(grid):,} combos em {time.time()-t0:.0f}s")
        print(f"{'#'*130}\n")

        out = os.path.join(SCRIPT_DIR, "..", "download")
        os.makedirs(out, exist_ok=True)
        pd.DataFrame([r.to_dict() for r in results]).sort_values("score", ascending=False).to_csv(
            os.path.join(out, "grid_15m_v3_results.csv"), index=False)
        wdata = {
            "version": "15m-v3-multi-strategy", "timeframe": "15m", "symbol": SYMBOL,
            "score": winner.score, "strategy": winner.strategy,
            "params": {k: v for k, v in winner.p.items()},
            "metrics": {
                "total_trades": winner.total_trades,
                "trades_per_day": round(winner.total_trades/365, 3),
                "win_rate": winner.win_rate, "profit_factor": winner.profit_factor,
                "total_pnl_pct": winner.total_pnl_pct, "max_drawdown_pct": winner.max_drawdown_pct,
                "sharpe_ratio": winner.sharpe_ratio, "buy_hold_pct": winner.buy_hold_pct,
            },
            "costs": {"fee_pct": FEE, "spread_bps": SPREAD, "slippage_bps": SLIPPAGE},
            "grid": {"total": len(grid), "time_sec": round(time.time()-t0, 1)},
        }
        with open(os.path.join(out, "winner_15m.json"), "w") as f:
            json.dump(wdata, f, indent=2)
        print("Saved: winner_15m.json + grid_15m_v3_results.csv")


if __name__ == "__main__":
    main()
