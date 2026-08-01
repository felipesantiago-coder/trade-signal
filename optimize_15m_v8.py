r"""
optimize_15m_v8.py
------------------
15m BTC/USDT — Mean-Reversion + Momentum com custos realistas

Problema do CTEV trend-following em 15m:
  - Filtros muito restritivos (EMA200, Fibonacci, ADX regime)
  - Movimentos pequenos vs custos (~0.325%/lado)
  - Ruido destrói sinais de tendencia

Abordagem v8:
  1. BB Bounce (mean-reversion): entra no toque da BB inferior/superior
  2. RSI Extreme: entra em RSI extremo com confirmacao BB
  3. Squeeze Breakout: entra na expansao apos squeeze
  4. EMA Cross: cruze EMA20/50 com filtro de momentum
  5. Custos realistas (limit orders): 0.016% fee + 2bps spread + 2bps slip
  6. Walk-Forward 4 janelas (5 splits)
"""
from __future__ import annotations
import sys, os, time, math, json, pickle
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from backtest import fetch_historical_ohlcv, _apply_costs
from indicators import compute_indicators

SYMBOL = 'BTC/USDT'; TIMEFRAME = '15m'; DAYS = 180
DATA_CACHE = '/tmp/ctev_15m_v8_data.pkl'

# Custos realistas para limit orders em 15m (Binance maker)
FEE = 0.016; SPREAD = 2.0; SLIPPAGE = 2.0  # ~0.05% RT


@dataclass
class R:
    score: float = 0.0; strategy: str = ''
    total_trades: int = 0; long_trades: int = 0; short_trades: int = 0
    wins: int = 0; win_rate: float = 0.0
    profit_factor: float = 0.0; total_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0; sharpe_ratio: float = 0.0
    avg_win_pct: float = 0.0; avg_loss_pct: float = 0.0
    avg_bars_held: float = 0.0
    buy_hold_pct: float = 0.0
    train_pnl: float = 0.0; test_pnl: float = 0.0
    train_wr: float = 0.0; test_wr: float = 0.0
    train_trades: int = 0; test_trades: int = 0
    p: dict = None

    def to_dict(self):
        d = asdict(self)
        d['params'] = self.p
        return d


def calc_m(trades):
    if not trades:
        return {'n': 0, 'w': 0, 'wr': 0, 'pf': 0, 'pnl': 0, 'dd': 0, 'sh': 0,
                'aw': 0, 'al': 0, 'ab': 0, 'l': 0, 's': 0}
    total = len(trades)
    wins = sum(1 for t in trades if t[1] > 0)
    longs = sum(1 for t in trades if t[0])
    pnls = [t[1] for t in trades]
    wp = [p for p in pnls if p > 0]
    lp = [p for p in pnls if p <= 0]
    gw = sum(wp); gl = abs(sum(lp))
    pf = gw / gl if gl > 0 else 999.0
    tpnl = sum(pnls)
    eq = [100.0]
    for p in pnls:
        eq.append(eq[-1] * (1 + p / 100))
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak * 100
    mdd = float(np.max(dd))
    # Sharpe anualizado (15min = 96 candles/dia)
    sharpe = float(np.mean(pnls) / (np.std(pnls) + 1e-9) * math.sqrt(365 * 96 / max(total, 1)))
    return {'n': total, 'w': wins, 'wr': wins / total * 100, 'pf': pf, 'pnl': tpnl,
            'dd': mdd, 'sh': sharpe, 'aw': np.mean(wp) if wp else 0,
            'al': np.mean(lp) if lp else 0,
            'ab': np.mean([t[2] for t in trades]), 'l': longs, 's': total - longs}


def calc_score(r):
    # Requisitos minimos
    if r.total_trades < 80 or r.max_drawdown_pct > 25:
        return 0.0
    if r.total_pnl_pct <= 0 or r.profit_factor < 1.0:
        return 0.0
    # Penalizar se NAO supera B&H
    bh_beat = 1.3 if r.total_pnl_pct > r.buy_hold_pct else 0.7
    # Penalizar se treino ou teste negativo
    if r.train_pnl <= 0 and r.test_pnl <= 0:
        return 0.0
    wf_b = 1.2 if (r.train_pnl > 0 and r.test_pnl > 0) else 0.6
    wr = r.win_rate / 100.0
    pf = r.profit_factor
    dd_pen = max(0, 1.0 - r.max_drawdown_pct / 100.0)
    freq = min(math.log(max(r.total_trades, 1)) / math.log(400), 1.5)
    pnl_b = 2.0 if r.total_pnl_pct > 15 else (1.5 if r.total_pnl_pct > 5 else 1.0)
    sig = min(math.sqrt(r.total_trades) / 10.0, 2.0)
    return round(wr * pf * freq * dd_pen * pnl_b * sig * wf_b * bh_beat, 4)


def sim_trades(c, hi, lo, atr, entries, dirs, sls, tps, eps, max_bars=48):
    """Simula trades com custos realistas."""
    n = len(c); trades = []
    for k in range(len(entries)):
        idx = entries[k]; is_l = dirs[k]; sl = sls[k]; tp = tps[k]; ep = eps[k]
        xp = None; bars = 0; mj = min(idx + max_bars, n)
        if is_l:
            for j in range(idx + 1, mj):
                if lo[j] <= sl:
                    xp = sl; bars = j - idx; break
                if hi[j] >= tp:
                    xp = tp; bars = j - idx; break
        else:
            for j in range(idx + 1, mj):
                if hi[j] >= sl:
                    xp = sl; bars = j - idx; break
                if lo[j] <= tp:
                    xp = tp; bars = j - idx; break
        if xp is None:
            bars = mj - 1 - idx; xp = c[mj - 1]
        _, adj, _ = _apply_costs(ep, xp, is_l, FEE, SPREAD, SLIPPAGE)
        pnl = (adj - ep) / ep * 100 if is_l else (ep - adj) / ep * 100
        trades.append((is_l, pnl, bars))
    return trades


def gen_entries_bb_bounce(c, hi, lo, rsi, atr, bbl, bbu, bbm, bbw, bbsq,
                          regime, vol, vs20, start, end, p):
    """BB Bounce: mean-reversion nas bandas de Bollinger."""
    E = []; D = []; S = []; T = []; EP = []
    rsi_lo = p['rsi_lo']; rsi_hi = p['rsi_hi']
    sl_m = p['sl_m']; tp_m = p['tp_m']
    bb_touch = p.get('bb_touch', 0.0)  # tolerancia para toque
    vol_min = p.get('vol_min', 0.0)  # min volume ratio
    mb = p.get('max_bars', 48); cd = p.get('cooldown', 4)
    i = start; cd_end = 0
    while i < end:
        if i < cd_end: i += 1; continue
        ci = c[i]; li = lo[i]; hii = hi[i]; r = rsi[i]; av = atr[i]
        rg = regime[i]

        # LONG: price toca/passa BB inferior, RSI baixo, NAO trending_down
        if rg != 'trending_down':
            if li <= bbl[i] * (1 + bb_touch) and r <= rsi_lo:
                if vol_min > 0 and vs20[i] > 0 and vol[i] < vs20[i] * vol_min:
                    i += 1; continue
                sl = li - sl_m * av
                tp = ci + tp_m * av
                if sl > 0:
                    E.append(i); D.append(True); S.append(sl); T.append(tp); EP.append(ci)
                    cd_end = i + cd; i += cd; continue

        # SHORT: price toca/passa BB superior, RSI alto, NAO trending_up
        if rg != 'trending_up':
            if hii >= bbu[i] * (1 - bb_touch) and r >= rsi_hi:
                if vol_min > 0 and vs20[i] > 0 and vol[i] < vs20[i] * vol_min:
                    i += 1; continue
                sl = hii + sl_m * av
                tp = ci - tp_m * av
                E.append(i); D.append(False); S.append(sl); T.append(tp); EP.append(ci)
                cd_end = i + cd; i += cd; continue

        i += 1
    return E, D, S, T, EP


def gen_entries_rsi_extreme(c, hi, lo, rsi, atr, bbl, bbu, bbm,
                            regime, vol, vs20, start, end, p):
    """RSI Extreme: entrada em RSI extremo com BB confirmacao."""
    E = []; D = []; S = []; T = []; EP = []
    rsi_l = p['rsi_l']; rsi_s = p['rsi_s']
    sl_m = p['sl_m']; tp_m = p['tp_m']
    mb = p.get('max_bars', 48); cd = p.get('cooldown', 4)
    bb_conf = p.get('bb_confirm', True)  # exige preco proximo a BB
    i = start; cd_end = 0
    while i < end:
        if i < cd_end: i += 1; continue
        ci = c[i]; li = lo[i]; hii = hi[i]; r = rsi[i]; av = atr[i]
        rg = regime[i]

        # LONG: RSI muito baixo, preco na parte baixa do BB
        if r <= rsi_l and rg != 'trending_down':
            if bb_conf and ci > bbm[i]:
                i += 1; continue
            sl = li - sl_m * av
            tp = ci + tp_m * av
            if sl > 0:
                E.append(i); D.append(True); S.append(sl); T.append(tp); EP.append(ci)
                cd_end = i + cd; i += cd; continue

        # SHORT: RSI muito alto, preco na parte alta do BB
        if r >= rsi_s and rg != 'trending_up':
            if bb_conf and ci < bbm[i]:
                i += 1; continue
            sl = hii + sl_m * av
            tp = ci - tp_m * av
            E.append(i); D.append(False); S.append(sl); T.append(tp); EP.append(ci)
            cd_end = i + cd; i += cd; continue

        i += 1
    return E, D, S, T, EP


def gen_entries_squeeze(c, hi, lo, rsi, rsi_delta, atr, bbl, bbu, bbm,
                      bbw, bbsq, regime, vol, vs20, adx, start, end, p):
    """Squeeze Breakout: entra na expansao de vol apos BB squeeze."""
    E = []; D = []; S = []; T = []; EP = []
    sq_thresh = p.get('sq_thresh', 0.20)
    sl_m = p['sl_m']; tp_m = p['tp_m']
    mb = p.get('max_bars', 48); cd = p.get('cooldown', 8)
    rsi_l = p.get('rsi_lo', 40); rsi_s = p.get('rsi_hi', 60)
    i = start; cd_end = 0
    while i < end:
        if i < cd_end: i += 1; continue
        if i < 2: i += 1; continue
        ci = c[i]; li = lo[i]; hii = hi[i]; r = rsi[i]; av = atr[i]
        rg = regime[i]

        # Squeeze: BB width no percentil baixo
        if bbsq[i] <= sq_thresh:
            i += 1; continue

        # LONG: squeeze rompe para cima (price acima BB upper)
        if hii >= bbu[i] and ci > bbm[i] and r >= rsi_l and rg != 'trending_down':
            sl = ci - sl_m * av
            tp = ci + tp_m * av
            if sl > 0:
                E.append(i); D.append(True); S.append(sl); T.append(tp); EP.append(ci)
                cd_end = i + cd; i += cd; continue

        # SHORT: squeeze rompe para baixo
        if li <= bbl[i] and ci < bbm[i] and r <= rsi_s and rg != 'trending_up':
            sl = ci + sl_m * av
            tp = ci - tp_m * av
            E.append(i); D.append(False); S.append(sl); T.append(tp); EP.append(ci)
            cd_end = i + cd; i += cd; continue

        i += 1
    return E, D, S, T, EP


def gen_entries_ema_cross(c, hi, lo, rsi, atr, e20, e50, e200,
                         rsi_delta, regime, adx, start, end, p):
    """EMA Cross: cruze EMA20/50 com filtro de momentum."""
    E = []; D = []; S = []; T = []; EP = []
    sl_m = p['sl_m']; tp_m = p['tp_m']
    mb = p.get('max_bars', 48); cd = p.get('cooldown', 8)
    adx_min = p.get('adx_min', 0)
    rsi_l = p.get('rsi_lo', 30); rsi_s = p.get('rsi_hi', 70)
    use_200 = p.get('use_200', False)
    i = start; cd_end = 0
    while i < end:
        if i < 1: i += 1; continue
        if i < cd_end: i += 1; continue
        ci = c[i]; li = lo[i]; hii = hi[i]; r = rsi[i]; av = atr[i]
        rg = regime[i]

        # LONG: EMA20 cruza acima da EMA50 + momentum
        if e20[i] > e50[i] and e20[i-1] <= e50[i-1]:
            if rg == 'trending_down': i += 1; continue
            if adx_min > 0 and adx[i] < adx_min: i += 1; continue
            if use_200 and ci < e200[i]: i += 1; continue
            if rsi_delta[i] <= 0: i += 1; continue
            if not (rsi_l <= r <= 80): i += 1; continue
            sl = ci - sl_m * av
            tp = ci + tp_m * av
            if sl > 0:
                E.append(i); D.append(True); S.append(sl); T.append(tp); EP.append(ci)
                cd_end = i + cd; i += cd; continue

        # SHORT: EMA20 cruza abaixo da EMA50
        if e20[i] < e50[i] and e20[i-1] >= e50[i-1]:
            if rg == 'trending_up': i += 1; continue
            if adx_min > 0 and adx[i] < adx_min: i += 1; continue
            if use_200 and ci > e200[i]: i += 1; continue
            if rsi_delta[i] >= 0: i += 1; continue
            if not (20 <= r <= rsi_s): i += 1; continue
            sl = ci + sl_m * av
            tp = ci - tp_m * av
            E.append(i); D.append(False); S.append(sl); T.append(tp); EP.append(ci)
            cd_end = i + cd; i += cd; continue

        i += 1
    return E, D, S, T, EP


STRATEGY_FUNCS = {
    'bb_bounce': (gen_entries_bb_bounce,
                  ['c', 'hi', 'lo', 'rsi', 'atr', 'bbl', 'bbu', 'bbm', 'bbw', 'bbsq',
                   'regime', 'vol', 'vs20']),
    'rsi_extreme': (gen_entries_rsi_extreme,
                    ['c', 'hi', 'lo', 'rsi', 'atr', 'bbl', 'bbu', 'bbm',
                     'regime', 'vol', 'vs20']),
    'squeeze': (gen_entries_squeeze,
                ['c', 'hi', 'lo', 'rsi', 'rsi_delta', 'atr', 'bbl', 'bbu', 'bbm',
                 'bbw', 'bbsq', 'regime', 'vol', 'vs20', 'adx']),
    'ema_cross': (gen_entries_ema_cross,
                  ['c', 'hi', 'lo', 'rsi', 'atr', 'e20', 'e50', 'e200',
                   'rsi_delta', 'regime', 'adx']),
}


def make_grid():
    C = []
    # === BB BOUNCE (mean-reversion) ===
    # Foco: RSI baixo no toque da BB, saida rapida no meio/superior
    for rsi_lo in [25, 30, 35]:
        for rsi_hi in [65, 70, 75]:
            for sl, tp in [(0.5, 1.0), (0.75, 1.5), (1.0, 2.0), (1.0, 2.5), (1.0, 3.0)]:
                for cd in [2, 4]:
                    for mb in [24, 48]:
                        C.append({'strat': 'bb_bounce', 'rsi_lo': rsi_lo, 'rsi_hi': rsi_hi,
                                  'sl_m': sl, 'tp_m': tp, 'cooldown': cd,
                                  'max_bars': mb, 'bb_touch': 0.0, 'vol_min': 0.0})

    # === RSI EXTREME ===
    for rsi_l in [20, 25, 30]:
        for rsi_s in [70, 75, 80]:
            for sl, tp in [(0.5, 1.0), (0.5, 1.5), (0.75, 1.5), (0.75, 2.0),
                           (1.0, 2.0), (1.0, 2.5)]:
                for cd in [2, 4, 8]:
                    for bb_conf in [True, False]:
                        C.append({'strat': 'rsi_extreme', 'rsi_l': rsi_l, 'rsi_s': rsi_s,
                                  'sl_m': sl, 'tp_m': tp, 'cooldown': cd,
                                  'max_bars': 48, 'bb_confirm': bb_conf})

    # === SQUEEZE BREAKOUT ===
    for sq in [0.10, 0.20, 0.30]:
        for sl, tp in [(0.75, 2.0), (1.0, 2.5), (1.0, 3.0), (1.5, 3.0), (1.5, 4.0)]:
            for cd in [4, 8, 12]:
                for rsi_lo in [35, 40]:
                    C.append({'strat': 'squeeze', 'sq_thresh': sq, 'sl_m': sl, 'tp_m': tp,
                              'cooldown': cd, 'max_bars': 48, 'rsi_lo': rsi_lo, 'rsi_hi': 65})

    # === EMA CROSS ===
    for sl, tp in [(0.75, 2.0), (1.0, 2.5), (1.0, 3.0), (1.5, 3.0), (1.5, 4.0)]:
        for cd in [8, 12]:
            for adx in [0, 15, 25]:
                for use_200 in [False, True]:
                    C.append({'strat': 'ema_cross', 'sl_m': sl, 'tp_m': tp, 'cooldown': cd,
                              'max_bars': 48, 'adx_min': adx, 'use_200': use_200,
                              'rsi_lo': 30, 'rsi_hi': 70})

    return C


def run_one(strat_name, gen_func, arr_names, arrs, start, end, p):
    """Executa uma estrategia em um sub-range."""
    kwargs = {arr_names[k]: arrs[k] for k in range(len(arr_names))}
    kwargs['start'] = start; kwargs['end'] = end; kwargs['p'] = p
    ea, da, sa, ta, epa = gen_func(**kwargs)
    if not ea:
        return []
    c_arr = arrs[0]; hi_arr = arrs[1]; lo_arr = arrs[2]; atr_arr = arrs[3]
    return sim_trades(c_arr, hi_arr, lo_arr, atr_arr, ea, da, sa, ta, epa,
                     p.get('max_bars', 48))


def main():
    t0 = time.time()
    print(f'\n{"#" * 160}')
    print(f'#  CTEV 15m v8 — MEAN-REVERSION + MULTI-STRATEGY + WALK-FORWARD')
    print(f'#  Custos: fee={FEE}% spread={SPREAD}bps slip={SLIPPAGE}bps (~0.05% RT)')
    print(f'#  4 estrategias: BB Bounce, RSI Extreme, Squeeze Breakout, EMA Cross')
    print(f'#{"#" * 160}')

    # Load data
    print(f'\n[1/4] Carregando dados 15m ({DAYS}d)...')
    if os.path.exists(DATA_CACHE):
        with open(DATA_CACHE, 'rb') as f:
            df = pickle.load(f)['df_clean']
        print(f'  Cache carregado')
    else:
        df = fetch_historical_ohlcv(SYMBOL, TIMEFRAME, DAYS)
        df = compute_indicators(df, timeframe=TIMEFRAME)
        crit = ['ema20', 'ema50', 'ema200', 'rsi', 'rsi_delta', 'atr', 'atr_percentile',
                'adx', 'plus_di', 'minus_di', 'regime', 'bb_lower', 'bb_upper', 'bb_middle',
                'bb_width', 'bb_squeeze_pct', 'macd', 'macd_signal', 'macd_hist',
                'volume_sma20', 'volume_sma50', 'ema50_slope']
        df = df.dropna(subset=crit).copy()
        with open(DATA_CACHE, 'wb') as f:
            pickle.dump({'df_clean': df}, f)
        print(f'  Dados baixados e cacheados')

    n = len(df)
    bh = (df['close'].values[-1] - df['close'].values[0]) / df['close'].values[0] * 100
    rc = dict(zip(*np.unique(df['regime'].values, return_counts=True)))
    print(f'  {n} candles ({n // 96}d) | B&H={bh:+.2f}% | {rc}')

    # Arrays
    c = df['close'].values.astype(np.float64)
    hi = df['high'].values.astype(np.float64)
    lo = df['low'].values.astype(np.float64)
    rsi = df['rsi'].values.astype(np.float64)
    rd = df['rsi_delta'].values.astype(np.float64)
    atr = df['atr'].values.astype(np.float64)
    adx = df['adx'].values.astype(np.float64)
    e20 = df['ema20'].values.astype(np.float64)
    e50 = df['ema50'].values.astype(np.float64)
    e200 = df['ema200'].values.astype(np.float64)
    bbl = df['bb_lower'].values.astype(np.float64)
    bbu = df['bb_upper'].values.astype(np.float64)
    bbm = df['bb_middle'].values.astype(np.float64)
    bbw = df['bb_width'].values.astype(np.float64)
    bbsq = df['bb_squeeze_pct'].values.astype(np.float64)
    regime = df['regime'].values
    vol = df['volume'].values.astype(np.float64)
    vs20 = df['volume_sma20'].values.astype(np.float64)

    # Walk-Forward: 5 splits -> 4 WF steps
    N_SPLITS = 5
    split_size = n // N_SPLITS
    splits = []
    for i in range(N_SPLITS):
        s = i * split_size
        e = (i + 1) * split_size if i < N_SPLITS - 1 else n
        splits.append((s, e))

    print(f'\n  Walk-Forward: {N_SPLITS} splits')
    for i, (s, e) in enumerate(splits):
        ts = df.index[s]; te = df.index[min(e, n) - 1]
        print(f'    W{i + 1}: [{s:>6}, {e:>6}) {ts} ~ {te} ({(e - s) // 96}d)')

    # Grid
    print(f'\n[2/4] Gerando grid...')
    grid = make_grid()
    print(f'  {len(grid):,} combos')

    # Contagem por estrategia
    from collections import Counter
    sc = Counter(p['strat'] for p in grid)
    for k, v in sc.items():
        print(f'    {k}: {v:,} combos')

    # Simular
    print(f'\n[3/4] Simulando (grid + walk-forward)...')
    results = []; viable = 0; best_score = 0; best_str = ''
    t1 = time.time()

    for idx, p in enumerate(grid):
        st = p['strat']
        if st not in STRATEGY_FUNCS:
            continue
        gen_func, arr_names = STRATEGY_FUNCS[st]

        # Mapear arrays pelo nome
        arr_map = {
            'c': c, 'hi': hi, 'lo': lo, 'rsi': rsi, 'rsi_delta': rd,
            'atr': atr, 'adx': adx, 'e20': e20, 'e50': e50, 'e200': e200,
            'bbl': bbl, 'bbu': bbu, 'bbm': bbm, 'bbw': bbw, 'bbsq': bbsq,
            'regime': regime, 'vol': vol, 'vs20': vs20
        }
        arrs = [arr_map[a] for a in arr_names]

        # Full period
        try:
            trades_all = run_one(st, gen_func, arr_names, arrs, 0, n, p)
        except Exception:
            continue
        if len(trades_all) < 80:
            continue

        # Walk-Forward: treinar em cada janela, testar na proxima
        wf_pnls = []
        for step in range(N_SPLITS - 1):
            tr_s, tr_e = splits[step]
            te_s, te_e = splits[step + 1]
            try:
                tr_trades = run_one(st, gen_func, arr_names, arrs, tr_s, tr_e, p)
                te_trades = run_one(st, gen_func, arr_names, arrs, te_s, te_e, p)
            except Exception:
                continue
            tr_m = calc_m(tr_trades)
            te_m = calc_m(te_trades)
            wf_pnls.append({'tr_pnl': tr_m['pnl'], 'te_pnl': te_m['pnl'],
                           'tr_n': tr_m['n'], 'te_n': te_m['n']})

        ma = calc_m(trades_all)
        # Agregar WF
        avg_tr_pnl = np.mean([w['tr_pnl'] for w in wf_pnls]) if wf_pnls else 0
        avg_te_pnl = np.mean([w['te_pnl'] for w in wf_pnls]) if wf_pnls else 0
        avg_tr_n = int(np.mean([w['tr_n'] for w in wf_pnls])) if wf_pnls else 0
        avg_te_n = int(np.mean([w['te_n'] for w in wf_pnls])) if wf_pnls else 0

        r = R(p=p, strategy=st,
              total_trades=ma['n'], long_trades=ma['l'], short_trades=ma['s'],
              wins=ma['w'], win_rate=round(ma['wr'], 2),
              profit_factor=round(ma['pf'], 4), total_pnl_pct=round(ma['pnl'], 4),
              max_drawdown_pct=round(ma['dd'], 4), sharpe_ratio=round(ma['sh'], 4),
              avg_win_pct=round(ma['aw'], 4), avg_loss_pct=round(ma['al'], 4),
              avg_bars_held=round(ma['ab'], 1), buy_hold_pct=bh,
              train_pnl=round(avg_tr_pnl, 4), test_pnl=round(avg_te_pnl, 4),
              train_wr=0, test_wr=0, train_trades=avg_tr_n, test_trades=avg_te_n)
        r.score = calc_score(r)
        results.append(r)
        if r.score > 0:
            viable += 1
        if r.score > best_score:
            best_score = r.score
            best_str = (f'{r.strategy} T={r.total_trades} WR={r.win_rate:.1f}% '
                       f'PF={r.profit_factor:.2f} PnL={r.total_pnl_pct:+.2f}% '
                       f'tr={r.train_pnl:+.2f}% te={r.test_pnl:+.2f}%')
        if (idx + 1) % 2000 == 0:
            el = time.time() - t1
            spd = (idx + 1) / max(el, 0.01)
            eta = (len(grid) - idx - 1) / max(spd, 0.01)
            print(f'  [{idx + 1:,}/{len(grid):,}] {spd:.0f}/s viable={viable} ETA={eta:.0f}s')
            if best_str:
                print(f'    BEST: {best_str}')

    elapsed = time.time() - t1
    print(f'\n  Grid: {len(grid):,} combos em {elapsed:.1f}s | viable={viable}')

    results.sort(key=lambda x: x.score, reverse=True)

    # Results table
    hdr = (f'{"#":>3} {"Score":>7} {"Strat":>12} {"Trades":>6} {"L/S":>6} '
           f'{"T/d":>5} {"WR%":>6} {"PF":>6} {"PnL%":>8} {"DD%":>6} {"Sharpe":>6} '
           f'{"AvgTR":>7} {"AvgTE":>7} {"SL":>4} {"TP":>4}')
    sep = '-' * 140
    print(f'\n{sep}\n  TOP 30 WALK-FORWARD ({viable} viable)\n{sep}')
    print(hdr)
    for i, r in enumerate(results[:30]):
        if r.score <= 0:
            break
        ls = f'{r.long_trades}/{r.short_trades}'
        td = r.total_trades / (n / 96)
        print(f'{i + 1:>3} {r.score:>7.3f} {r.strategy:>12} {r.total_trades:>6} {ls:>6} '
              f'{td:>5.2f} {r.win_rate:>6.1f} {r.profit_factor:>6.2f} '
              f'{r.total_pnl_pct:>8.2f} {r.max_drawdown_pct:>6.2f} {r.sharpe_ratio:>6.2f} '
              f'{r.train_pnl:>+7.2f} {r.test_pnl:>+7.2f} '
              f'{r.p["sl_m"]:>4.2f} {r.p["tp_m"]:>4.2f}')

    # Winner
    winner = results[0] if results and results[0].score > 0 else None
    if winner:
        rr = winner.p['tp_m'] / winner.p['sl_m']
        print(f'\n{"#" * 160}')
        print(f'#  VENCEDOR 15m v8 (WALK-FORWARD)')
        print(f'{"#" * 160}')
        print(f'#  Estrategia:    {winner.strategy}')
        print(f'#  Score:         {winner.score:.4f}')
        print(f'#  Trades:        {winner.total_trades} ({winner.total_trades / (n / 96):.2f}/dia)')
        print(f'#  Win Rate:      {winner.win_rate:.1f}%')
        print(f'#  Profit Factor: {winner.profit_factor:.2f}')
        print(f'#  PnL:           {winner.total_pnl_pct:+.2f}% (B&H={winner.buy_hold_pct:+.2f}%)')
        print(f'#  Max DD:        {winner.max_drawdown_pct:.2f}%')
        print(f'#  Sharpe:        {winner.sharpe_ratio:.2f}')
        print(f'#  SL/TP:         {winner.p["sl_m"]}x / {winner.p["tp_m"]}x (R:R {rr:.1f}:1)')
        print(f'#  Avg WF Train:  {winner.train_pnl:+.2f}%')
        print(f'#  Avg WF Test:   {winner.test_pnl:+.2f}%')
        print(f'#  Supera B&H:    {"SIM" if winner.total_pnl_pct > winner.buy_hold_pct else "NAO"}')
        print(f'#  Params:        {json.dumps({k: v for k, v in winner.p.items() if k != "strat"}, default=str)}')
        print(f'#  Grid:          {len(grid):,} combos em {time.time() - t0:.0f}s')
        print(f'{"#" * 160}\n')

        # Salvar
        out = os.path.join(SCRIPT_DIR, '..', 'download')
        os.makedirs(out, exist_ok=True)
        pd.DataFrame([r.to_dict() for r in results]).to_csv(
            os.path.join(out, 'grid_15m_v8_results.csv'), index=False)
        cp = {}
        for k, v in winner.p.items():
            cp[k] = list(v) if isinstance(v, (tuple, list)) else v
        with open(os.path.join(out, 'winner_15m_v8.json'), 'w') as f:
            json.dump({
                'version': '15m-v8-wf', 'timeframe': '15m', 'symbol': SYMBOL,
                'score': winner.score, 'strategy': winner.strategy, 'params': cp,
                'metrics': {
                    'total_trades': winner.total_trades,
                    'trades_per_day': round(winner.total_trades / (n / 96), 3),
                    'win_rate': winner.win_rate, 'profit_factor': winner.profit_factor,
                    'total_pnl_pct': winner.total_pnl_pct,
                    'max_drawdown_pct': winner.max_drawdown_pct,
                    'sharpe_ratio': winner.sharpe_ratio,
                    'buy_hold_pct': winner.buy_hold_pct,
                    'avg_wf_train_pnl': winner.train_pnl,
                    'avg_wf_test_pnl': winner.test_pnl,
                },
                'costs': {'fee_pct': FEE, 'spread_bps': SPREAD, 'slippage_bps': SLIPPAGE},
                'grid': {'total': len(grid), 'viable': viable,
                        'time_sec': round(time.time() - t0, 1)}
            }, f, indent=2)
        print(f'  Salvo: winner_15m_v8.json + grid_15m_v8_results.csv')
    else:
        print(f'\n!!! Nenhuma combinacao viavel encontrada !!!')
        # Mostrar os melhores por PnL mesmo sem score
        bp = sorted(results, key=lambda x: x.total_pnl_pct, reverse=True)
        print(f'  Top 5 por PnL (sem filtro):')
        for r in bp[:5]:
            print(f'    {r.strategy} T={r.total_trades} WR={r.win_rate:.1f}% '
                  f'PF={r.profit_factor:.2f} PnL={r.total_pnl_pct:+.2f}% '
                  f'DD={r.max_drawdown_pct:.2f}%')


if __name__ == '__main__':
    main()
