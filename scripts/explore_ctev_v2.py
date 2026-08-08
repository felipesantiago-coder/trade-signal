r"""explore_ctev_v2.py - Grid search de saidas alargadas para CTEV v4 strategy.

Encontra configs que garantam >=50% de retorno anual composto.
Reusa o padrao v14: trailing largo + post-TP1 floor (nao breakeven).
"""
import sys, os, json, time, logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING)

CACHE = "/home/z/my-project/trade-signal/download/btc_1h_cache.csv"
OUTPUT = "/home/z/my-project/trade-signal/download/explore_ctev_v2.json"


def load_data():
    df = pd.read_csv(CACHE, index_col=0, parse_dates=True)
    print(f"Cache: {len(df):,} candles", flush=True)
    return df


def run_bt(df, P):
    """Run CTEV backtest with custom exit parameters."""
    from strategy import evaluate_long, evaluate_short, SignalType
    from backtest import _apply_costs
    from strategy_profiles import StrategyProfile

    # Build profile with custom TP/SL
    profile = StrategyProfile(
        name="CTEV_GRID", timeframes=("1h",), description="Grid test",
        sl_atr_mult=P["sl_atr_mult"],
        tp_atr_mult=P["tp_atr_mult"],
    )

    results = []
    n = len(df)
    tr_mult = P["trailing_atr_mult"]
    buf = P["post_tp1_sl_buffer"]
    max_b = P.get("max_bars", 72)
    tp1p = P.get("tp1_pct", 0.50)
    risk = P["risk"]
    fee, spr, slp = 0.016, 2.0, 2.0

    eq = 100.0
    pk = 100.0
    md = 0.0
    i = 0
    last_exit_bar = -999

    while i < n:
        row = df.iloc[i]
        if i < 1:
            i += 1; continue

        critical = ["ema20", "ema50", "ema200", "rsi", "atr", "atr_percentile",
                   "adx", "plus_di", "minus_di", "regime"]
        if any(pd.isna(row.get(c)) for c in critical):
            i += 1; continue

        # ATR filter
        ap = float(row.get("atr_percentile", 0.5))
        if ap < 0.20 or ap > 0.80:
            i += 1; continue

        # Regime filter: only trending and transition
        regime = str(row.get("regime", ""))
        if regime in ("ranging", "volatile"):
            i += 1; continue

        # Cooldown: skip if last exit was recent
        if i - last_exit_bar < 2:
            i += 1; continue

        # Evaluate signals
        signal = evaluate_long(row, profile=profile)
        if signal is None:
            signal = evaluate_short(row, profile=profile)

        if signal is None:
            i += 1; continue

        entry = signal.entry_price
        sl = signal.stop_loss
        tp = signal.take_profit
        atr = signal.atr
        is_long = signal.type == SignalType.LONG
        sl_d = max(abs(entry - sl) / entry * 100, 0.05)

        # Simulate trade with custom exit logic
        csl = sl
        tr_on = False
        tp1f = False
        tp1p_v = 0.0
        hwm = entry
        exit_done = False

        for j in range(i + 1, min(i + max_b, n)):
            f = df.iloc[j]
            fc, fl, fh = float(f["close"]), float(f["low"]), float(f["high"])

            if is_long:
                hwm = max(hwm, fh)
                sh = fl <= csl
                th = fh >= tp
            else:
                hwm = min(hwm, fl)
                sh = fh >= csl
                th = fl <= tp

            # TP1 hit (first time)
            if th and not tp1f:
                tp1f = True
                tp1p_v = tp
                tr_on = True
                # v14 logic: SL -> TP - buffer*ATR (floor, NOT breakeven)
                buffer = atr * buf
                if is_long:
                    csl = tp - buffer
                else:
                    csl = tp + buffer

            # Both TP and SL hit in same bar
            if th and sh and tp1f:
                pn = _pp(entry, tp1p_v, csl, is_long, fee, spr, slp, tp1p)
                results.append((pn, sl_d))
                exit_done = True
                break

            # SL hit
            if sh and not th:
                if tp1f:
                    pn = _pp(entry, tp1p_v, csl, is_long, fee, spr, slp, tp1p)
                else:
                    _, a2, _ = _apply_costs(entry, csl, is_long, fee, spr, slp)
                    pn = (a2 - entry) / entry * 100 if is_long else (entry - a2) / entry * 100
                results.append((pn, sl_d))
                exit_done = True
                break

            # Second TP hit (close remaining)
            if th and tp1f:
                pn = _pp(entry, tp1p_v, tp, is_long, fee, spr, slp, tp1p)
                results.append((pn, sl_d))
                exit_done = True
                break

            # Trailing update
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

        # Timeout
        if not exit_done:
            lj = min(i + max_b, n) - 1
            xp = float(df.iloc[lj]["close"])
            if tp1f:
                pn = _pp(entry, tp1p_v, xp, is_long, fee, spr, slp, tp1p)
            else:
                _, a3, _ = _apply_costs(entry, xp, is_long, fee, spr, slp)
                pn = (a3 - entry) / entry * 100 if is_long else (entry - a3) / entry * 100
            results.append((pn, sl_d))

        # Compound equity
        pn = results[-1][0]
        ps = risk / sl_d
        eq += eq * ps * pn / 100
        eq = max(eq, 0.01)
        if eq > pk:
            pk = eq
        dd = (pk - eq) / pk * 100
        if dd > md:
            md = dd

        last_exit_bar = i + max_b  # skip ahead
        i += max_b  # advance past the trade

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
    if not trades: return {"eq": 100, "ret": 0, "ann": 0, "dd": 0, "n": 0}
    eq, pk, md = 100.0, 100.0, 0.0
    for pn, sd in trades:
        ps = risk / sd
        eq += eq * ps * pn / 100
        eq = max(eq, 0.01)
        if eq > pk: pk = eq
        dd = (pk - eq) / pk * 100
        if dd > md: md = dd
    r = eq - 100
    return {"eq": round(eq, 2), "ret": round(r, 2), "ann": round(r * 365 / days, 2), "dd": round(md, 2), "n": len(trades)}


def met(trades):
    if not trades: return {"n": 0, "wr": 0, "pf": 0}
    pnls = [p for p, _ in trades]
    w = [p for p in pnls if p > 0]
    l = [p for p in pnls if p <= 0]
    return {"n": len(pnls), "wr": round(len(w) / len(pnls) * 100, 1),
            "pf": round(sum(w) / max(abs(sum(l)), 0.001), 2)}


def main():
    print("=" * 80, flush=True)
    print("EXPLORE CTEV v2 — Grid search saidas alargadas", flush=True)
    print("=" * 80, flush=True)

    df = load_data()
    n = len(df)
    print(f"Range: {df.index[0]} to {df.index[-1]}", flush=True)

    # Grid: trailing x buffer x TP x SL x risk
    trailings = [1.5, 2.0, 2.5, 3.0, 3.5]
    buffers = [0.0, 0.1, 0.2, 0.3, 0.5]  # 0.0 = breakeven (original)
    tps = [5.0, 6.0, 8.5, 10.0, 12.0]
    sls = [2.0, 2.12, 2.5, 3.0]
    risks = [1.0, 2.0, 2.5, 3.0]

    grid = [(tr, buf, tp, sl) for tr in trailings for buf in buffers
            for tp in tps for sl in sls]
    print(f"Grid: {len(grid)} param combos x {len(risks)} risk = {len(grid) * len(risks)}", flush=True)

    t0 = time.time()
    qualified = []
    best_score = 0

    for gi, (tr, buf, tp, sl) in enumerate(grid):
        P = {
            "sl_atr_mult": sl,
            "tp_atr_mult": tp,
            "trailing_atr_mult": tr,
            "post_tp1_sl_buffer": buf,
            "max_bars": 72,
            "risk": 2.0,  # use fixed risk for grid, test multiple later
        }
        try:
            trades, eq, dd = run_bt(df, P)
            m = met(trades)
            c = comp(trades, 2.0)
            ann = c["ann"]
            dd_val = c["dd"]
            score = ann / max(dd_val, 1.0)

            if ann >= 50 and dd_val <= 40 and m["n"] >= 20:
                qualified.append({
                    "trail": tr, "buf": buf, "tp": tp, "sl": sl,
                    "n": m["n"], "wr": m["wr"], "pf": m["pf"],
                    "ann": ann, "ret": c["ret"], "dd": dd_val,
                    "eq": c["eq"], "score": round(score, 2),
                })
                if score > best_score:
                    best_score = score

        except Exception as e:
            print(f"  ERR tr={tr} buf={buf} tp={tp} sl={sl}: {e}", flush=True)

        if (gi + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  [{gi + 1}/{len(grid)}] {el:.1f}s | qualified={len(qualified)} best_score={best_score:.1f}", flush=True)

    print(f"\nDone: {time.time() - t0:.1f}s | {len(qualified)} qualified configs", flush=True)

    # Sort by score
    qualified.sort(key=lambda x: x["score"], reverse=True)

    # Print top 10
    print(f"\n{'=' * 80}")
    print(f"TOP 10 configs (>=50% ann, DD<=40%, n>=20)")
    print(f"{'=' * 80}")
    for i, r in enumerate(qualified[:10]):
        buf_label = f"BE" if r["buf"] == 0 else f"{r['buf']}"
        print(f"#{i + 1} TR={r['trail']}x BUF={buf_label} TP={r['tp']}x SL={r['sl']}x | "
              f"ANN={r['ann']:+.1f}% DD={r['dd']:.1f}% T={r['n']} WR={r['wr']:.1f}% PF={r['pf']:.2f} score={r['score']}", flush=True)

    # Test top 5 with different risk levels
    print(f"\n--- Risk variation for top 5 ---", flush=True)
    risk_results = []
    for r in qualified[:5]:
        for risk in risks:
            P = {
                "sl_atr_mult": r["sl"],
                "tp_atr_mult": r["tp"],
                "trailing_atr_mult": r["trail"],
                "post_tp1_sl_buffer": r["buf"],
                "max_bars": 72,
                "risk": risk,
            }
            try:
                trades, eq, dd = run_bt(df, P)
                c = comp(trades, risk)
                m = met(trades)
                sc = c["ann"] / max(c["dd"], 1.0)
                risk_results.append({
                    **r, "risk": risk, "ann": c["ann"], "ret": c["ret"],
                    "dd": c["dd"], "eq": c["eq"], "wr": m["wr"], "pf": m["pf"],
                    "score": round(sc, 2),
                })
            except:
                pass

    risk_results.sort(key=lambda x: x["score"], reverse=True)
    print(f"\nTOP 5 with risk variation:")
    for i, r in enumerate(risk_results[:5]):
        buf_label = f"BE" if r["buf"] == 0 else f"{r['buf']}"
        print(f"#{i + 1} TR={r['trail']}x BUF={buf_label} TP={r['tp']}x SL={r['sl']}x risk={r['risk']}% | "
              f"ANN={r['ann']:+.1f}% DD={r['dd']:.1f}% EQ={r['eq']} WR={r['wr']:.1f}% PF={r['pf']:.2f}", flush=True)

    # Save
    out = {
        "description": "CTEV v2 exit exploration — trailing + post-TP1 floor",
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
