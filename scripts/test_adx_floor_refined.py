"""Testa ADX floors refinados para WEAK_UPTREND LONG.
Mostra quais trades sao eliminados em cada nivel e se algum era winner."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, logging, ccxt, numpy as np, pandas as pd
from datetime import datetime, timezone
from indicators import compute_indicators
from regime_engine import classify_regimes_v2, get_regime_params
from strategy import evaluate_long, evaluate_short, SignalType
from strategy_regime import evaluate_mean_reversion_long, evaluate_mean_reversion_short
from strategy_profiles import get_profile
from backtest import DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS, _apply_costs

logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
logger = logging.getLogger("adx_refined")
logger.setLevel(logging.INFO)


def download_btc():
    exchange = ccxt.binance({"enableRateLimit": True})
    since_ms = int((datetime.now(timezone.utc).timestamp() - 730 * 86400) * 1000)
    all_ohlcv, last_ts, it = [], 0, 0
    while it < 730*24+100:
        it += 1
        batch = exchange.fetch_ohlcv("BTC/USDT", "1h", since=since_ms, limit=1000)
        if not batch or batch[-1][0] <= last_ts: break
        last_ts = batch[-1][0]
        all_ohlcv.extend(batch)
        since_ms = last_ts + 3600000
    df = pd.DataFrame(all_ohlcv, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("ts").astype(float).drop_duplicates(keep="first")


def simulate(df_ind, profile, adx_floor=0):
    trades = []
    i, n = 0, len(df_ind)
    mb = profile.max_bars_held if profile else 72
    while i < n:
        row = df_ind.iloc[i]
        crit = ["ema20","ema50","ema200","rsi","atr","atr_percentile","adx","plus_di","minus_di","regime","regime_v2","regime_confidence"]
        if any(pd.isna(row.get(c)) for c in crit): i += 1; continue
        rv2 = str(row.get("regime_v2", ""))
        conf = float(row.get("regime_confidence", 0.5))
        params = get_regime_params(rv2, conf, base_profile=profile)
        st = params["strategy_type"]
        if st == "neutral" or conf < params["min_confidence"]: i += 1; continue
        signal = None
        ap = float(row.get("atr_percentile", 0.5))
        if st == "trend_follow":
            if adx_floor > 0 and rv2 == "WEAK_UPTREND" and float(row.get("adx",0)) < adx_floor:
                i += 1; continue
            if ap < 0.10 or ap > 0.90: i += 1; continue
            signal = evaluate_long(row, profile=profile)
            if signal is None: signal = evaluate_short(row, profile=profile)
        elif st == "mean_reversion":
            if ap < 0.10 or ap > 0.85: i += 1; continue
            if params["allow_long"]: signal = evaluate_mean_reversion_long(row, params, base_profile=profile)
            if signal is None and params["allow_short"]: signal = evaluate_mean_reversion_short(row, params, base_profile=profile)
        if signal is None: i += 1; continue
        ep, sl, tp, atr = signal.entry_price, signal.stop_loss, signal.take_profit, signal.atr
        is_l = signal.type == SignalType.LONG
        xp, xr, bars = None, None, 0
        for j in range(i+1, min(i+mb, n)):
            f = df_ind.iloc[j]; bars = j - i
            if is_l:
                if float(f["low"]) <= sl: xp, xr = sl, "sl"; break
                if float(f["high"]) >= tp: xp, xr = tp, "tp"; break
            else:
                if float(f["high"]) >= sl: xp, xr = sl, "sl"; break
                if float(f["low"]) <= tp: xp, xr = tp, "tp"; break
        if xp is None:
            lj = min(i+mb, n) - 1; xp = float(df_ind.iloc[lj]["close"]); xr = "timeout"; bars = lj - i
        _, ax, _ = _apply_costs(ep, xp, is_l, DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS)
        pnl = ((ax - ep) / ep * 100) if is_l else ((ep - ax) / ep * 100)
        trades.append({"type": "LONG" if is_l else "SHORT", "pnl": round(pnl,4), "exit": xr, "regime": rv2, "adx": round(float(row.get("adx",0)),1), "ts": str(row.name)[:16]})
        i += bars + 1
    return trades


def metrics(trades):
    if not trades: return {"t":0,"l":0,"s":0,"wr":0,"pnl":0,"dd":0}
    longs = [t for t in trades if t["type"]=="LONG"]
    shorts = [t for t in trades if t["type"]=="SHORT"]
    w = sum(1 for t in trades if t["pnl"]>0)
    lw = sum(1 for t in longs if t["pnl"]>0)
    sw = sum(1 for t in shorts if t["pnl"]>0)
    pnl = sum(t["pnl"] for t in trades)
    r, p, dd = 0, 0, 0
    for t in trades:
        r += t["pnl"]; p = max(p, r); dd = max(dd, p - r)
    return {"t": len(trades), "l": len(longs), "s": len(shorts),
            "wr": round(100*w/len(trades),1), "lwr": round(100*lw/max(len(longs),1),1),
            "swr": round(100*sw/max(len(shorts),1),1), "pnl": round(pnl,2), "dd": round(dd,2),
            "lpnl": round(sum(t["pnl"] for t in longs),2), "spnl": round(sum(t["pnl"] for t in shorts),2)}


logger.info("Baixando dados...")
df = download_btc()
logger.info(f"Dados: {len(df)} candles")
df_ind = compute_indicators(df, "1h")
df_ind = classify_regimes_v2(df_ind, hysteresis_bars=3)
profile = get_profile("STANDARD")

# Run all variants
floors = [0, 22, 24, 25, 26, 27, 28, 30]
print(f"\n{'Floor':>6s} {'Trades':>7s} {'L':>3s} {'S':>3s} {'WR%':>6s} {'PnL%':>7s} {'DD%':>6s} {'L_WR%':>6s} {'S_WR%':>6s} {'L_PnL':>7s} {'S_PnL':>7s}")
print("-"*82)

all_results = {}
for fl in floors:
    trades = simulate(df_ind, profile, adx_floor=fl)
    m = metrics(trades)
    all_results[fl] = (trades, m)
    label = "Baseline" if fl == 0 else f"ADX>{fl}"
    print(f"{label:>6s} {m['t']:7d} {m['l']:3d} {m['s']:3d} {m['wr']:6.1f} {m['pnl']:7.2f} {m['dd']:6.2f} {m['lwr']:6.1f} {m['swr']:6.1f} {m['lpnl']:7.2f} {m['spnl']:7.2f}")

# Detail: which trades are removed at ADX 25?
base_trades = all_results[0][0]
fl25_trades = all_results[25][0]
base_ts = {(t["ts"], t["type"]) for t in base_trades}
fl25_ts = {(t["ts"], t["type"]) for t in fl25_trades}
removed = [t for t in base_trades if (t["ts"], t["type"]) not in fl25_ts]

print(f"\n{'='*80}")
print(f"Trades REMOVIDOS pelo ADX floor 25 ({len(removed)} trades):")
for t in removed:
    w = "WIN " if t["pnl"] > 0 else "LOSS"
    print(f"  {t['ts']} {t['type']:5s} ADX={t['adx']:5.1f} PnL={t['pnl']:+6.2f}% {w} {t['regime']}")

removed_wins = [t for t in removed if t["pnl"] > 0]
removed_losses = [t for t in removed if t["pnl"] <= 0]
print(f"\n  Resumo: {len(removed_wins)} wins removidos (+{sum(t['pnl'] for t in removed_wins):.2f}%), {len(removed_losses)} losses removidos ({sum(t['pnl'] for t in removed_losses):.2f}%)")
print(f"  Liquido: {sum(t['pnl'] for t in removed):+.2f}%")

# Best variant
best_fl = max(floors, key=lambda f: all_results[f][1]["pnl"])
bm = all_results[best_fl][1]
print(f"\n{'='*80}")
print(f"MELHOR ADX floor: {best_fl if best_fl > 0 else '0 (baseline)'}")
print(f"  Trades: {bm['t']}, WR: {bm['wr']}%, PnL: {bm['pnl']}%, DD: {bm['dd']}%")
