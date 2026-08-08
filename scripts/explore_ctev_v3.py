# explore_ctev_v3.py - CTEV saidas otimizadas (pre-compute entries)
# Etapa 1: Roda CTEV uma vez para encontrar todos os pontos de entrada.
# Etapa 2: Testa diferentes saidas (trailing, buffer) sobre os mesmos entries.
import sys, os, json, time, logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING)

CACHE = "/home/z/my-project/trade-signal/download/btc_1h_cache.csv"
OUTPUT = "/home/z/my-project/trade-signal/download/explore_ctev_v3.json"


def precompute_entries(df):
    from strategy import evaluate_long, evaluate_short, SignalType
    from strategy_profiles import StrategyProfile

    profile = StrategyProfile(
        name="CTEV_BASE", timeframes=("1h",), description="Pre-compute",
        sl_atr_mult=2.12, tp_atr_mult=8.50,
    )

    entries = []
    n = len(df)
    for i in range(1, n):
        row = df.iloc[i]
        critical = ["ema20", "ema50", "ema200", "rsi", "atr", "atr_percentile",
                   "adx", "plus_di", "minus_di", "regime"]
        if any(pd.isna(row.get(c)) for c in critical):
            continue
        ap = float(row.get("atr_percentile", 0.5))
        if ap < 0.20 or ap > 0.80:
            continue
        regime = str(row.get("regime", ""))
        if regime in ("ranging", "volatile"):
            continue

        signal = evaluate_long(row, profile=profile)
        if signal is None:
            signal = evaluate_short(row, profile=profile)
        if signal is None:
            continue

        entries.append({
            "idx": i,
            "entry": float(signal.entry_price),
            "atr": float(signal.atr),
            "is_long": signal.type == SignalType.LONG,
            "sl_dist_pct": max(abs(signal.entry_price - signal.stop_loss) / signal.entry_price * 100, 0.05),
        })
    print(f"  Pre-computed {len(entries)} CTEV entries", flush=True)
    return entries


def simulate_exits(df, entries, P):
    from backtest import _apply_costs

    tr_mult = P["trailing_atr_mult"]
    buf = P["post_tp1_sl_buffer"]
    max_b = P["max_bars"]
    tp_mult = P["tp_atr_mult"]
    sl_mult = P["sl_atr_mult"]
    risk = P["risk"]
    fee, spr, slp = 0.016, 2.0, 2.0
    n = len(df)
    results = []
    eq, pk, md = 100.0, 100.0, 0.0
    last_exit = -999

    for e in entries:
        idx = e["idx"]
        if idx - last_exit < 2:
            continue

        entry = e["entry"]
        atr = e["atr"]
        is_long = e["is_long"]
        sl = entry - sl_mult * atr if is_long else entry + sl_mult * atr
        tp = entry + tp_mult * atr if is_long else entry - tp_mult * atr
        sl_d = sl_mult * atr / entry * 100
        if sl_d < 0.05:
            sl_d = 0.05

        csl = sl
        tr_on = False
        tp1f = False
        tp1p_v = 0.0
        hwm = entry
        exit_done = False
        pn = 0.0

        for j in range(idx + 1, min(idx + max_b, n)):
            f = df.iloc[j]
            fl, fh = float(f["low"]), float(f["high"])

            if is_long:
                hwm = max(hwm, fh)
                sh = fl <= csl
                th = fh >= tp
            else:
                hwm = min(hwm, fl)
                sh = fh >= csl
                th = fl <= tp

            if th and not tp1f:
                tp1f = True
                tp1p_v = tp
                tr_on = True
                buffer = atr * buf
                if is_long:
                    csl = tp - buffer
                else:
                    csl = tp + buffer

            if th and sh and tp1f:
                pn = _pp(entry, tp1p_v, csl, is_long, fee, spr, slp, 0.50)
                exit_done = True
                break

            if sh and not th:
                if tp1f:
                    pn = _pp(entry, tp1p_v, csl, is_long, fee, spr, slp, 0.50)
                else:
                    _, a2, _ = _apply_costs(entry, csl, is_long, fee, spr, slp)
                    pn = (a2 - entry) / entry * 100 if is_long else (entry - a2) / entry * 100
                exit_done = True
                break

            if th and tp1f:
                pn = _pp(entry, tp1p_v, tp, is_long, fee, spr, slp, 0.50)
                exit_done = True
                break

            if tr_on:
                td = atr * tr_mult
                if is_long:
                    nt = hwm - td
                    if nt > csl:
                        csl = nt
                else:
                    nt = hwm + td
                    if nt < csl:
                        csl = nt

        if not exit_done:
            lj = min(idx + max_b, n) - 1
            xp = float(df.iloc[lj]["close"])
            if tp1f:
                pn = _pp(entry, tp1p_v, xp, is_long, fee, spr, slp, 0.50)
            else:
                _, a3, _ = _apply_costs(entry, xp, is_long, fee, spr, slp)
                pn = (a3 - entry) / entry * 100 if is_long else (entry - a3) / entry * 100

        results.append((pn, sl_d))
        ps = risk / sl_d
        eq += eq * ps * pn / 100
        eq = max(eq, 0.01)
        if eq > pk:
            pk = eq
        dd = (pk - eq) / pk * 100
        if dd > md:
            md = dd
        last_exit = idx + max_b

    return results, eq, md


def _pp(e, t1, ex, isL, f, s, sl, tp1):
    from backtest import _apply_costs
    _, a1, _ = _apply_costs(e, t1, isL, f, s, sl)
    _, a2, _ = _apply_costs(e, ex, isL, f, s, sl)
    if isL:
        return round(tp1 * (a1 - e) / e * 100 + (1 - tp1) * (a2 - e) / e * 100, 4)
    else:
        return round(tp1 * (e - a1) / e * 100 + (1 - tp1) * (e - a2) / e * 100, 4)


def comp(trades, risk, days=730):
    if not trades:
        return {"eq": 100, "ret": 0, "ann": 0, "dd": 0, "n": 0}
    eq, pk, md = 100.0, 100.0, 0.0
    for pn, sd in trades:
        ps = risk / sd
        eq += eq * ps * pn / 100
        eq = max(eq, 0.01)
        if eq > pk:
            pk = eq
        dd = (pk - eq) / pk * 100
        if dd > md:
            md = dd
    r = eq - 100
    return {"eq": round(eq, 2), "ret": round(r, 2), "ann": round(r * 365 / days, 2), "dd": round(md, 2), "n": len(trades)}


def met(trades):
    if not trades:
        return {"n": 0, "wr": 0, "pf": 0}
    pnls = [p for p, _ in trades]
    w = [p for p in pnls if p > 0]
    l = [p for p in pnls if p <= 0]
    return {"n": len(pnls), "wr": round(len(w) / len(pnls) * 100, 1),
            "pf": round(sum(w) / max(abs(sum(l)), 0.001), 2)}


def main():
    print("=" * 80, flush=True)
    print("EXPLORE CTEV v3 — Pre-computed entries + exit grid", flush=True)
    print("=" * 80, flush=True)

    df = pd.read_csv(CACHE, index_col=0, parse_dates=True)
    print(f"Cache: {len(df):,} candles | {df.index[0]} to {df.index[-1]}", flush=True)

    print("\n--- Phase 1: Pre-computing entries ---", flush=True)
    t0 = time.time()
    entries = precompute_entries(df)
    print(f"  {time.time() - t0:.1f}s", flush=True)

    if len(entries) < 10:
        print("  FEW ENTRIES - aborting")
        return

    print(f"\n--- Phase 2: Exit grid ({len(entries)} entries) ---", flush=True)
    trailings = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    buffers = [0.0, 0.1, 0.2, 0.3, 0.5]
    tps = [5.0, 8.5, 10.0, 12.0]
    sls = [2.0, 2.5, 3.0]
    risks = [1.0, 2.0, 2.5, 3.0]

    grid = [(tr, buf, tp, sl) for tr in trailings for buf in buffers
            for tp in tps for sl in sls]
    print(f"  Grid: {len(grid)} combos", flush=True)

    t1 = time.time()
    qualified = []

    for gi, (tr, buf, tp, sl) in enumerate(grid):
        P = {
            "sl_atr_mult": sl, "tp_atr_mult": tp,
            "trailing_atr_mult": tr, "post_tp1_sl_buffer": buf,
            "max_bars": 72, "risk": 2.0,
        }
        try:
            trades, eq, dd = simulate_exits(df, entries, P)
            m = met(trades)
            c = comp(trades, 2.0)
            ann, dd_val = c["ann"], c["dd"]
            score = ann / max(dd_val, 1.0)

            if ann >= 50 and dd_val <= 40 and m["n"] >= 10:
                qualified.append({
                    "trail": tr, "buf": buf, "tp": tp, "sl": sl,
                    "n": m["n"], "wr": m["wr"], "pf": m["pf"],
                    "ann": ann, "ret": c["ret"], "dd": dd_val,
                    "eq": c["eq"], "score": round(score, 2),
                })
        except Exception:
            pass

        if (gi + 1) % 50 == 0:
            el = time.time() - t1
            print(f"  [{gi + 1}/{len(grid)}] {el:.1f}s | qualified={len(qualified)}", flush=True)

    print(f"\n  Done: {time.time() - t1:.1f}s | {len(qualified)} qualified", flush=True)

    qualified.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n{'=' * 80}")
    print(f"TOP 10 configs (>=50% ann, DD<=40%)")
    print(f"{'=' * 80}")
    for i, r in enumerate(qualified[:10]):
        bl = f"BE" if r["buf"] == 0 else f"{r['buf']}"
        print(f"#{i + 1} TR={r['trail']}x BUF={bl} TP={r['tp']}x SL={r['sl']}x | "
              f"ANN={r['ann']:+.1f}% DD={r['dd']:.1f}% T={r['n']} WR={r['wr']:.1f}% PF={r['pf']:.2f} sc={r['score']}", flush=True)

    print(f"\n--- Risk variation for top 5 ---", flush=True)
    risk_results = []
    for r in qualified[:5]:
        for risk in risks:
            P2 = dict(r)
            P2["risk"] = risk
            try:
                trades, eq, dd = simulate_exits(df, entries, P2)
                c = comp(trades, risk)
                m = met(trades)
                sc = c["ann"] / max(c["dd"], 1.0)
                risk_results.append({**r, "risk": risk, "ann": c["ann"], "ret": c["ret"],
                                    "dd": c["dd"], "eq": c["eq"], "wr": m["wr"], "pf": m["pf"], "score": round(sc, 2)})
            except Exception:
                pass

    risk_results.sort(key=lambda x: x["score"], reverse=True)
    print(f"\nTOP 5 with risk variation:")
    for i, r in enumerate(risk_results[:5]):
        bl = f"BE" if r["buf"] == 0 else f"{r['buf']}"
        print(f"#{i + 1} TR={r['trail']}x BUF={bl} TP={r['tp']}x SL={r['sl']}x risk={r['risk']}% | "
              f"ANN={r['ann']:+.1f}% DD={r['dd']:.1f}% EQ={r['eq']} WR={r['wr']:.1f}% PF={r['pf']:.2f}", flush=True)

    # Baseline: original CTEV
    print(f"\n--- Baseline (original CTEV: BE + tr=1.5) ---", flush=True)
    for risk in [1, 2, 3]:
        P2 = {"sl_atr_mult": 2.12, "tp_atr_mult": 8.5, "trailing_atr_mult": 1.5,
               "post_tp1_sl_buffer": 0.0, "max_bars": 72, "risk": float(risk)}
        trades, eq, dd = simulate_exits(df, entries, P2)
        c = comp(trades, float(risk))
        m = met(trades)
        print(f"  risk={risk}%: ANN={c['ann']:+.1f}% DD={c['dd']:.1f}% T={m['n']} WR={m['wr']:.1f}% PF={m['pf']:.2f}", flush=True)

    out = {
        "description": "CTEV v3 exit exploration (pre-computed entries)",
        "entries_found": len(entries),
        "total_combos": len(grid),
        "qualified": len(qualified),
        "top10": qualified[:10],
        "risk_top5": risk_results[:5],
    }
    with open(OUTPUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
