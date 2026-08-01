from __future__ import annotations
import sys, os, time, math, json, pickle
from dataclasses import dataclass, asdict
from typing import List

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from backtest import fetch_historical_ohlcv, _apply_costs
from indicators import compute_indicators

SYMBOL = "BTC/USDT"; TIMEFRAME = "15m"; DAYS = 365
DATA_CACHE = "/tmp/ctev_15m_data.pkl"

# Custos realistas para 15m com ordens LIMIT:
# - Spread BTC/USDT Binance: ~1-2 bps (mercado muito liquido)
# - Slippage em ordens limit: ~2-5 bps (quase zero)
# - Fee maker Binance: 0.01% (com BNB) ou 0.02% (sem BNB)
# Total round-trip: ~0.10-0.15%
FEE = 0.02; SPREAD = 3.0; SLIPPAGE = 5.0  # Muito mais realistas para 15m limit orders
MIN_TRADES = 80; MIN_WR = 48.0; MAX_DD = 20.0


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
    p: dict = None

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
    crit = ["ema20","ema50","ema200","rsi","atr","atr_percentile",
            "adx","plus_di","minus_di","regime","bb_lower","bb_upper","bb_width","bb_squeeze_pct"]
    df = df.dropna(subset=crit).copy()
    with open(DATA_CACHE, 'wb') as f:
        pickle.dump({'df_clean': df}, f)
    return df


def calc_score(r):
    if r.total_trades < MIN_TRADES or r.profit_factor <= 1.0: return 0.0
    if r.win_rate < MIN_WR or r.max_drawdown_pct > MAX_DD: return 0.0
    wr = r.win_rate / 100.0; pf = r.profit_factor
    dd_pen = max(0, 1.0 - r.max_drawdown_pct / 100.0)
    freq = math.log(max(r.total_trades, 1)) / math.log(365)
    pnl_b = 2.0 if r.total_pnl_pct > 10 else (1.5 if r.total_pnl_pct > 5 else (1.2 if r.total_pnl_pct > 0 else 0.5))
    sig = min(math.sqrt(r.total_trades) / 15.0, 2.0)
    return round(wr * pf * freq * dd_pen * pnl_b * sig, 4)


def sim(c, hi, lo, entries, dirs, sls, tps, eps, max_bars):
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


def mkt(trades, buy_hold):
    if not trades: return R(0, buy_hold_pct=buy_hold)
    total = len(trades)
    wins = sum(1 for t in trades if t[1] > 0)
    losses = total - wins
    wr = wins / total * 100
    longs = sum(1 for t in trades if t[0])
    pnls = [t[1] for t in trades]
    wp = [p for p in pnls if p > 0]; lp = [p for p in pnls if p <= 0]
    gw = sum(wp); gl = abs(sum(lp))
    pf = gw / gl if gl > 0 else 999.0
    tpnl = sum(pnls)
    aw = np.mean(wp) if wp else 0; al = np.mean(lp) if lp else 0
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


def run_strat(df, strategy_fn, params, label):
    entries, dirs, sls, tps, eps = strategy_fn(df, params)
    if not entries:
        r = R(0, strategy=label, p=params)
        return r
    c = df['close'].values; hi = df['high'].values; lo = df['low'].values
    bh = (c[-1] - c[0]) / c[0] * 100
    trades = sim(c, hi, lo, entries, dirs, sls, tps, eps, params.get('max_bars', 72))
    r = mkt(trades, bh)
    r.strategy = label; r.sl_m = params.get('sl_m', 1.5); r.tp_m = params.get('tp_m', 3.5)
    r.max_bars = params.get('max_bars', 72); r.p = params
    return r


# ======================================================================
# STRATEGIES
# ======================================================================

def s_trend_lite(df, p):
    """Regime + RSI + EMA trend only (no pullback, no fib)."""
    c=df['close'].values; lo=df['low'].values; hi=df['high'].values
    e50=df['ema50'].values; e200=df['ema200'].values; e20=df['ema20'].values
    rsi=df['rsi'].values; atr=df['atr'].values; adx=df['adx'].values
    regime=df['regime'].values; ap=df['atr_percentile'].values
    rg_d=df['regime'].values; n=len(c)
    entries=[]; dirs=[]; sls=[]; tps=[]; eps=[]
    rsi_l=p['rsi_l']; rsi_s=p['rsi_s']; adx_min=p.get('adx_min',0)
    allow_trn=p.get('allow_trn',True); req_et=p.get('req_et',True)
    sl_m=p['sl_m']; tp_m=p['tp_m']; max_bars=p.get('max_bars',72)
    atr_min=p.get('atr_min',0.05); atr_max=p.get('atr_max',0.95)
    cooldown=0
    for i in range(n):
        if i < cooldown: continue
        rg=regime[i]
        if rg=='volatile': continue
        # LONG
        if rg in ('trending_up','transition'):
            if rg=='trending_up' and adx[i]<adx_min: continue
            if rg=='transition' and not allow_trn: continue
            if req_et and not (c[i]>e50[i]>e200[i]): continue
            if not (rsi_l[0]<=rsi[i]<=rsi_l[1]): continue
            if not (atr_min<=ap[i]<=atr_max): continue
            ep=c[i]; sl=ep-sl_m*atr[i]; tp=ep+tp_m*atr[i]
            if sl>0:
                entries.append(i); dirs.append(True); sls.append(sl); tps.append(tp); eps.append(ep)
                cooldown=i+4; continue
        # SHORT
        if rg in ('trending_down','transition'):
            if rg=='trending_down' and adx[i]<adx_min: continue
            if rg=='transition' and not allow_trn: continue
            if req_et and not (c[i]<e50[i]<e200[i]): continue
            if not (rsi_s[0]<=rsi[i]<=rsi_s[1]): continue
            if not (atr_min<=ap[i]<=atr_max): continue
            ep=c[i]; sl=ep+sl_m*atr[i]; tp=ep-tp_m*atr[i]
            entries.append(i); dirs.append(False); sls.append(sl); tps.append(tp); eps.append(ep)
            cooldown=i+4
    return entries, dirs, sls, tps, eps


def s_ema_cross(df, p):
    """Fast EMA cross with momentum."""
    c=df['close'].values; lo=df['low'].values; hi=df['high'].values
    rsi=df['rsi'].values; atr=df['atr'].values; regime=df['regime'].values
    n=len(c)
    fs=p.get('fast_ema',8); ss=p.get('slow_ema',21)
    ef=pd.Series(c).ewm(span=fs,adjust=False).mean().values
    es=pd.Series(c).ewm(span=ss,adjust=False).mean().values
    entries=[]; dirs=[]; sls=[]; tps=[]; eps=[]
    sl_m=p['sl_m']; tp_m=p['tp_m']; max_bars=p.get('max_bars',48)
    rsi_l_range=p.get('rsi_l_range',(40,70)); rsi_s_range=p.get('rsi_s_range',(30,60))
    cooldown=0
    for i in range(1, n):
        if i < cooldown: continue
        if regime[i]=='volatile': continue
        bull=ef[i]>es[i] and ef[i-1]<=es[i-1]
        bear=ef[i]<es[i] and ef[i-1]>=es[i-1]
        if bull and rsi_l_range[0]<=rsi[i]<=rsi_l_range[1]:
            ep=c[i]; sl=ep-sl_m*atr[i]; tp=ep+tp_m*atr[i]
            if sl>0:
                entries.append(i); dirs.append(True); sls.append(sl); tps.append(tp); eps.append(ep)
                cooldown=i+4; continue
        if bear and rsi_s_range[0]<=rsi[i]<=rsi_s_range[1]:
            ep=c[i]; sl=ep+sl_m*atr[i]; tp=ep-tp_m*atr[i]
            entries.append(i); dirs.append(False); sls.append(sl); tps.append(tp); eps.append(ep)
            cooldown=i+4
    return entries, dirs, sls, tps, eps


def s_bb_bounce(df, p):
    """BB Bounce: buy at lower band, sell at upper band, with cooldown."""
    c=df['close'].values; lo=df['low'].values; hi=df['high'].values
    bbl=df['bb_lower'].values; bbu=df['bb_upper'].values
    rsi=df['rsi'].values; atr=df['atr'].values
    entries=[]; dirs=[]; sls=[]; tps=[]; eps=[]
    rsi_l_max=p.get('rsi_l_max',40); rsi_s_min=p.get('rsi_s_min',60)
    sl_m=p['sl_m']; tp_m=p['tp_m']; max_bars=p.get('max_bars',48)
    cooldown=0; n=len(c)
    for i in range(n):
        if i < cooldown: continue
        # LONG: RSI oversold + price at/below BB lower
        if rsi[i]<=rsi_l_max and lo[i]<=bbl[i]:
            ep=c[i]; sl=ep-sl_m*atr[i]; tp=ep+tp_m*atr[i]
            if sl>0:
                entries.append(i); dirs.append(True); sls.append(sl); tps.append(tp); eps.append(ep)
                cooldown=i+12; continue
        # SHORT: RSI overbought + price at/above BB upper
        if rsi[i]>=rsi_s_min and hi[i]>=bbu[i]:
            ep=c[i]; sl=ep+sl_m*atr[i]; tp=ep-tp_m*atr[i]
            entries.append(i); dirs.append(False); sls.append(sl); tps.append(tp); eps.append(ep)
            cooldown=i+12
    return entries, dirs, sls, tps, eps


def s_rsi_extreme(df, p):
    """RSI extreme + BB + optional regime filter."""
    c=df['close'].values; lo=df['low'].values; hi=df['high'].values
    bbl=df['bb_lower'].values; bbu=df['bb_upper'].values
    rsi=df['rsi'].values; atr=df['atr'].values; regime=df['regime'].values
    entries=[]; dirs=[]; sls=[]; tps=[]; eps=[]
    rsi_l_max=p.get('rsi_l_max',35); rsi_s_min=p.get('rsi_s_min',65)
    use_bb=p.get('use_bb',True); sl_m=p['sl_m']; tp_m=p['tp_m']
    max_bars=p.get('max_bars',48); cooldown=0; n=len(c)
    for i in range(n):
        if i < cooldown: continue
        if regime[i]=='volatile': continue
        # LONG
        if rsi[i]<=rsi_l_max:
            if use_bb and lo[i]>bbl[i]: continue
            ep=c[i]; sl=ep-sl_m*atr[i]; tp=ep+tp_m*atr[i]
            if sl>0:
                entries.append(i); dirs.append(True); sls.append(sl); tps.append(tp); eps.append(ep)
                cooldown=i+8; continue
        # SHORT
        if rsi[i]>=rsi_s_min:
            if use_bb and hi[i]<bbu[i]: continue
            ep=c[i]; sl=ep+sl_m*atr[i]; tp=ep-tp_m*atr[i]
            entries.append(i); dirs.append(False); sls.append(sl); tps.append(tp); eps.append(ep)
            cooldown=i+8
    return entries, dirs, sls, tps, eps


def make_grid():
    combos = []
    # ── TREND-LITE ──
    rsi_l_opts = [(25,50),(28,48),(30,55),(25,55)]
    rsi_s_opts = [(50,75),(55,75),(45,70),(50,70)]
    sltp_opts = [(1.0,2.5),(1.25,3.0),(1.5,3.0),(1.5,3.5),(2.0,4.0),(2.5,5.0),(3.0,5.0),(3.0,6.0)]
    for rl in rsi_l_opts:
        for rs in rsi_s_opts:
            for sl,tp in sltp_opts:
                for adx in [0,20]:
                    for et in [True,False]:
                        for trn in [True,False]:
                            for mb in [48,96]:
                                p = {'rsi_l':rl,'rsi_s':rs,'sl_m':sl,'tp_m':tp,'max_bars':mb,
                                      'adx_min':adx,'allow_trn':trn,'req_et':et,
                                      'atr_min':0.05,'atr_max':0.95}
                                combos.append((s_trend_lite, p, 'E-Trend-Lite'))

    # ── EMA CROSS ──
    for fs,ss in [(5,13),(5,21),(8,21),(8,34)]:
        for sl,tp in sltp_opts:
            for mb in [48,96]:
                for rl in [(40,70),(35,65)]:
                    for rs in [(30,60),(25,55)]:
                        p = {'fast_ema':fs,'slow_ema':ss,'sl_m':sl,'tp_m':tp,'max_bars':mb,
                              'rsi_l_range':rl,'rsi_s_range':rs}
                        combos.append((s_ema_cross, p, 'C-EMA-Cross'))

    # ── BB BOUNCE ──
    for rsi_l in [30,40]:
        for rsi_s in [60,70]:
            for sl,tp in sltp_opts:
                for mb in [48,96]:
                    p = {'rsi_l_max':rsi_l,'rsi_s_min':rsi_s,'sl_m':sl,'tp_m':tp,'max_bars':mb}
                    combos.append((s_bb_bounce, p, 'A-BB-Bounce'))

    # ── RSI EXTREME ──
    for rsi_l in [30,40]:
        for rsi_s in [60,70]:
            for sl,tp in sltp_opts:
                for use_bb in [True,False]:
                    for mb in [48,96]:
                        p = {'rsi_l_max':rsi_l,'rsi_s_min':rsi_s,'sl_m':sl,'tp_m':tp,
                              'use_bb':use_bb,'max_bars':mb}
                        combos.append((s_rsi_extreme, p, 'B-RSI-Extreme'))

    return combos


def print_top(results, title, n=20):
    print(f"\n{'='*130}")
    print(f"  {title}")
    print(f"{'='*130}")
    print(f"{'#':>3} {'Score':>7} {'Trades':>6} {'L/S':>6} {'T/d':>5} {'WR%':>6} {'PF':>6} {'PnL%':>8} {'DD%':>6} {'Sharpe':>6} {'Strategy':>15} {'SL':>4} {'TP':>4}")
    print(f"{'-'*130}")
    for i, r in enumerate(sorted(results, key=lambda x: x.score, reverse=True)[:n]):
        ls = f"{r.long_trades}/{r.short_trades}"; td = r.total_trades / 365
        print(f"{i+1:>3} {r.score:>7.3f} {r.total_trades:>6} {ls:>6} {td:>5.2f} {r.win_rate:>6.1f} {r.profit_factor:>6.2f} "
              f"{r.total_pnl_pct:>8.2f} {r.max_drawdown_pct:>6.2f} {r.sharpe_ratio:>6.2f} {r.strategy:>15} {r.sl_m:>4.2f} {r.tp_m:>4.2f}")


def main():
    t0 = time.time()
    print("\n" + "#" * 130)
    print("#  CTEV 15m OPTIMIZER v4 — REALISTIC COSTS + COOLDOWN")
    print(f"#  Custo: fee={FEE}% spread={SPREAD}bps slip={SLIPPAGE}bps (~0.15% round-trip)")
    print("#" * 130)

    print("\n[1/3] Carregando dados 15m...")
    df = load_data()
    n = len(df); bh = (df['close'].values[-1] - df['close'].values[0]) / df['close'].values[0] * 100
    rc = dict(zip(*np.unique(df['regime'].values, return_counts=True)))
    print(f"{n} candles ({n//96} dias) | B&H={bh:.2f}% | {rc}")
    print(f"  ATR medio: {df['atr'].values.mean():.2f} ({df['atr'].values.mean()/df['close'].values.mean()*100:.3f}% do preco)")

    print("\n[2/3] Gerando grid...")
    grid = make_grid()
    print(f"{len(grid):,} combinacoes")
    from collections import Counter
    sc = Counter(l for _,_,l in grid)
    for s,c in sc.most_common(): print(f"  {s}: {c:,}")

    print("\n[3/3] Executando simulacoes...")
    results = []; viable = 0; best_score = 0; best_str = ""; t1 = time.time()

    for idx, (func, params, label) in enumerate(grid):
        r = run_strat(df, func, params, label)
        r.cid = idx; r.score = calc_score(r)
        results.append(r)
        if r.score > 0: viable += 1
        if r.score > best_score:
            best_score = r.score
            best_str = (f"{r.strategy} T={r.total_trades} WR={r.win_rate:.1f}% "
                       f"PF={r.profit_factor:.2f} PnL={r.total_pnl_pct:+.2f}% DD={r.max_drawdown_pct:.2f}%")
        if (idx + 1) % 2000 == 0:
            el = time.time() - t1; spd = (idx + 1) / max(el, 0.01)
            eta = (len(grid) - idx - 1) / max(spd, 0.01)
            print(f"  [{idx+1:,}/{len(grid):,}] {spd:.0f}/s viable={viable} ETA={eta:.0f}s")
            if best_str: print(f"    BEST: {best_str}")

    elapsed = time.time() - t1
    print(f"\n  Concluido: {len(grid):,} combos em {elapsed:.1f}s | viable={viable}")

    print("\n" + "=" * 130)
    print("  MELHOR POR ESTRATEGIA")
    print("=" * 130)
    sg = {}
    for r in results:
        s = r.strategy
        if s not in sg or r.score > sg[s].score: sg[s] = r
    for s, r in sorted(sg.items(), key=lambda x: x[1].score, reverse=True):
        print(f"  {s:>15}: Score={r.score:.3f} T={r.total_trades} WR={r.win_rate:.1f}% PF={r.profit_factor:.2f} PnL={r.total_pnl_pct:+.2f}% DD={r.max_drawdown_pct:.2f}%")

    print_top(results, f"TOP 25 ({len(results):,} combos)", 25)

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
        print(f"#  Avg Win/Loss:  {winner.avg_win_pct:.3f}% / {winner.avg_loss_pct:.3f}%")
        print(f"#  SL/TP:         {winner.sl_m:.2f}x / {winner.tp_m:.2f}x (R:R {winner.tp_m/winner.sl_m:.1f}:1)")
        print(f"#  Max Bars:      {winner.max_bars}")
        print(f"#  Params:        {json.dumps({k:v for k,v in winner.p.items() if k not in ('rsi_l','rsi_s')}, default=str)}")
        print(f"#  Total:         {len(grid):,} combos em {time.time()-t0:.0f}s")
        print(f"{'#'*130}\n")

        out = os.path.join(SCRIPT_DIR, "..", "download")
        os.makedirs(out, exist_ok=True)
        pd.DataFrame([r.to_dict() for r in results]).sort_values("score", ascending=False).to_csv(
            os.path.join(out, "grid_15m_v4_results.csv"), index=False)
        # Clean params for JSON serialization
        clean_p = {}
        for k, v in winner.p.items():
            if isinstance(v, (tuple, list)):
                clean_p[k] = list(v)
            else:
                clean_p[k] = v
        wdata = {
            "version": "15m-v4", "timeframe": "15m", "symbol": SYMBOL,
            "score": winner.score, "strategy": winner.strategy,
            "params": clean_p,
            "metrics": {
                "total_trades": winner.total_trades,
                "trades_per_day": round(winner.total_trades/365, 3),
                "win_rate": winner.win_rate, "profit_factor": winner.profit_factor,
                "total_pnl_pct": winner.total_pnl_pct, "max_drawdown_pct": winner.max_drawdown_pct,
                "sharpe_ratio": winner.sharpe_ratio, "buy_hold_pct": winner.buy_hold_pct,
                "avg_win_pct": winner.avg_win_pct, "avg_loss_pct": winner.avg_loss_pct,
            },
            "costs": {"fee_pct": FEE, "spread_bps": SPREAD, "slippage_bps": SLIPPAGE,
                       "est_round_trip_pct": round(FEE*2 + SPREAD*2/10000 + SLIPPAGE*2/10000, 4)},
            "grid": {"total": len(grid), "time_sec": round(time.time()-t0, 1)},
        }
        with open(os.path.join(out, "winner_15m.json"), "w") as f:
            json.dump(wdata, f, indent=2)
        print("Saved: winner_15m.json + grid_15m_v4_results.csv")


if __name__ == "__main__":
    main()
