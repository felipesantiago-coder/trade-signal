r"""
optimize_v15_fast.py — Otimizacao RAPIDA da Confluence v15.

Abordagem: usar gen_confluence existente + grid focado nas saidas.
Foco: testar variações de saida que melhorem o 90d sem quebrar o resto.
"""
import sys, os, json, time
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.explore_new_strategies import (
    load_data, run_trades, gen_confluence, compute_metrics
)

CACHE = "/home/z/my-project/trade-signal/download/btc_1h_cache.csv"
OUTPATH = "/home/z/my-project/trade-signal/download/optimize_v15_robust_results.json"


def main():
    print("=" * 70)
    print("OTIMIZACAO RAPIDA CONFLUENCE v15")
    print("=" * 70)

    df = load_data()
    total_days = (df.index[-1] - df.index[0]).days

    # Sub-periods
    sub_periods = []
    for label, offset in [("730d", 0), ("365d", total_days - 365),
                            ("180d", total_days - 180), ("90d", total_days - 90)]:
        if offset < 0: continue
        start_ts = df.index[0] + pd.Timedelta(days=offset)
        mask = df.index >= start_ts
        sub_df = df[mask]
        start_idx = df.index.get_loc(sub_df.index[0])
        days = (sub_df.index[-1] - sub_df.index[0]).days
        sub_periods.append((label, start_idx, len(sub_df), days))

    print(f"Sub-periodos: {[(s[0], s[3]) for s in sub_periods]}")

    # =================================================================
    # PHASE 1: Generate signal sets (use existing fast function)
    # =================================================================
    print("\n--- Gerando sinais ---")
    t0 = time.time()

    sig_configs = [
        ("adx15_rsi50_vol0.3", 15, 50, 0.3),
        ("adx15_rsi55_vol0.3", 15, 55, 0.3),
        ("adx15_rsi60_vol0.3", 15, 60, 0.3),
        ("adx18_rsi50_vol0.3", 18, 50, 0.3),
        ("adx18_rsi55_vol0.3", 18, 55, 0.3),
        ("adx18_rsi60_vol0.3", 18, 60, 0.3),
        ("adx20_rsi50_vol0.3", 20, 50, 0.3),
        ("adx20_rsi55_vol0.3", 20, 55, 0.3),
        ("adx20_rsi60_vol0.3", 20, 60, 0.3),
        ("adx25_rsi50_vol0.3", 25, 50, 0.3),
        ("adx25_rsi55_vol0.3", 25, 55, 0.3),
        ("adx15_rsi50_vol0.35", 15, 50, 0.35),
        ("adx15_rsi55_vol0.35", 15, 55, 0.35),
        ("adx18_rsi50_vol0.35", 18, 50, 0.35),
        ("adx18_rsi55_vol0.35", 18, 55, 0.35),
        ("adx20_rsi50_vol0.35", 20, 50, 0.35),
        ("adx20_rsi55_vol0.35", 20, 55, 0.35),
        ("adx25_rsi50_vol0.35", 25, 50, 0.35),
        ("adx25_rsi55_vol0.35", 25, 55, 0.35),
        ("adx15_rsi50_vol0.4", 15, 50, 0.4),
        ("adx15_rsi55_vol0.4", 15, 55, 0.4),
        ("adx18_rsi55_vol0.4", 18, 55, 0.4),
        ("adx20_rsi55_vol0.4", 20, 55, 0.4),
        ("adx20_rsi50_vol0.4", 20, 50, 0.4),
        ("adx25_rsi55_vol0.4", 25, 55, 0.4),
        ("adx15_rsi50_vol0.5", 15, 50, 0.5),
        ("adx18_rsi55_vol0.5", 18, 55, 0.5),
        ("adx20_rsi55_vol0.5", 20, 55, 0.5),
    ]

    sig_sets = {}
    for sname, adx, rsi_l, vol in sig_configs:
        sig_sets[sname] = gen_confluence(df, adx_min=adx, rsi_long_max=rsi_l,
                                         rsi_short_min=100-rsi_l, vol_mult=vol)

    print(f"  {len(sig_sets)} signal sets em {time.time()-t0:.1f}s")
    for sname, sigs in sorted(sig_sets.items()):
        if len(sigs) >= 15:
            print(f"    {sname}: {len(sigs)}")

    # =================================================================
    # PHASE 2: Pre-filter on 730d (quick)
    # =================================================================
    print("\n--- Pre-filtro 730d ---")
    t0 = time.time()

    baseline = [(8.0, 2.5, 3.0, 0.2, 120), (6.0, 2.0, 3.0, 0.2, 120)]
    promising = {}

    for sname, signals in sig_sets.items():
        if len(signals) < 20: continue
        best_ann = -999
        for tp, sl, tr, buf, mb in baseline:
            for risk in [2.0, 3.0]:
                trades = run_trades(df, signals, tp_mult=tp, sl_mult=sl,
                                    trailing_mult=tr, tp1_pct=0.50,
                                    post_tp1_buf=buf, max_bars=mb)
                m = compute_metrics(trades, risk, sub_periods[0][3])
                if m["n"] >= 20 and m["wr"] >= 30 and m["ann"] > best_ann:
                    best_ann = m["ann"]
        if best_ann >= 50:
            promising[sname] = best_ann

    print(f"  {len(promising)} signal sets promissores em {time.time()-t0:.1f}s")
    for s, a in sorted(promising.items(), key=lambda x: -x[1]):
        print(f"    {s}: {a:+.1f}%")

    # =================================================================
    # PHASE 3: Full grid on promising signals
    # =================================================================
    exit_combos = [
        # (name, tp, sl, tr, buf, mb)
        ("TP6_SL2_TR2.5_b0.1_m96", 6, 2.0, 2.5, 0.1, 96),
        ("TP6_SL2_TR3_b0.1_m96", 6, 2.0, 3.0, 0.1, 96),
        ("TP6_SL2_TR3_b0.2_m96", 6, 2.0, 3.0, 0.2, 96),
        ("TP6_SL2_TR3_b0.2_m120", 6, 2.0, 3.0, 0.2, 120),
        ("TP6_SL2.5_TR3_b0.1_m96", 6, 2.5, 3.0, 0.1, 96),
        ("TP6_SL2.5_TR3_b0.2_m96", 6, 2.5, 3.0, 0.2, 96),
        ("TP6_SL2.5_TR3_b0.2_m120", 6, 2.5, 3.0, 0.2, 120),
        ("TP8_SL2_TR2.5_b0.1_m96", 8, 2.0, 2.5, 0.1, 96),
        ("TP8_SL2_TR3_b0.1_m96", 8, 2.0, 3.0, 0.1, 96),
        ("TP8_SL2_TR3_b0.2_m96", 8, 2.0, 3.0, 0.2, 96),
        ("TP8_SL2_TR3_b0.2_m120", 8, 2.0, 3.0, 0.2, 120),
        ("TP8_SL2_TR4_b0.2_m120", 8, 2.0, 4.0, 0.2, 120),
        ("TP8_SL2.5_TR2.5_b0.1_m96", 8, 2.5, 2.5, 0.1, 96),
        ("TP8_SL2.5_TR3_b0.1_m96", 8, 2.5, 3.0, 0.1, 96),
        ("TP8_SL2.5_TR3_b0.2_m96", 8, 2.5, 3.0, 0.2, 96),
        ("TP8_SL2.5_TR3_b0.2_m120", 8, 2.5, 3.0, 0.2, 120),
        ("TP8_SL2.5_TR3.5_b0.2_m120", 8, 2.5, 3.5, 0.2, 120),
        ("TP8_SL2.5_TR4_b0.2_m120", 8, 2.5, 4.0, 0.2, 120),
        ("TP10_SL2_TR3_b0.2_m120", 10, 2.0, 3.0, 0.2, 120),
        ("TP10_SL2.5_TR3_b0.2_m120", 10, 2.5, 3.0, 0.2, 120),
        ("TP10_SL2.5_TR4_b0.2_m120", 10, 2.5, 4.0, 0.2, 120),
        ("TP10_SL3_TR3_b0.2_m120", 10, 3.0, 3.0, 0.2, 120),
        ("TP10_SL3_TR4_b0.2_m120", 10, 3.0, 4.0, 0.2, 120),
    ]

    risks = [2.0, 3.0]
    total = len(promising) * len(exit_combos) * len(risks)
    print(f"\n--- Grid search ({total} combinacoes) ---")
    t0 = time.time()

    candidates = []
    tested = 0

    for sname in promising:
        signals = sig_sets[sname]
        for ename, tp, sl, tr, buf, mb in exit_combos:
            for risk in risks:
                tested += 1

                # Quick 730d
                trades_730 = run_trades(df, signals, tp_mult=tp, sl_mult=sl,
                                        trailing_mult=tr, tp1_pct=0.50,
                                        post_tp1_buf=buf, max_bars=mb)
                m730 = compute_metrics(trades_730, risk, sub_periods[0][3])
                if m730["n"] < 25 or m730["wr"] < 30 or m730["dd"] > 40:
                    continue

                # All sub-periods
                sub_anns = []
                sub_dds = []
                sub_ns = []
                ok = True
                for pname, sidx, slen, days in sub_periods:
                    ssigs = [(i, d, a, p) for i, d, a, p in signals if sidx <= i < sidx + slen]
                    trs = run_trades(df, ssigs, tp_mult=tp, sl_mult=sl,
                                     trailing_mult=tr, tp1_pct=0.50,
                                     post_tp1_buf=buf, max_bars=mb)
                    m = compute_metrics(trs, risk, days)
                    sub_anns.append(m["ann"])
                    sub_dds.append(m["dd"])
                    sub_ns.append(m["n"])
                    if m["ann"] < 0: ok = False

                min_ann = min(sub_anns)
                avg_wr = np.mean([compute_metrics(
                    run_trades(df, [(i,d,a,p) for i,d,a,p in signals if sidx <= i < sidx+slen],
                                tp_mult=tp, sl_mult=sl, trailing_mult=tr, tp1_pct=0.50,
                                post_tp1_buf=buf, max_bars=mb), risk, days)["wr"]
                    for pname, sidx, slen, days in sub_periods])
                max_dd = max(sub_dds)
                min_n = min(sub_ns)

                if min_ann >= 50 and max_dd <= 40 and min_n >= 10:
                    candidates.append({
                        "sig": sname, "exit": ename,
                        "tp": tp, "sl": sl, "tr": tr,
                        "buf": buf, "mb": mb, "risk": risk,
                        "min_ann": round(min_ann, 1),
                        "avg_ann": round(np.mean(sub_anns), 1),
                        "max_dd": round(max_dd, 1),
                        "min_n": min_n,
                        "sub_anns": [round(a, 1) for a in sub_anns],
                        "sub_dds": [round(d, 1) for d in sub_dds],
                        "sub_ns": sub_ns,
                        "score": round(min_ann - max_dd * 0.5, 1),
                    })

    print(f"  {tested:,} testados em {time.time()-t0:.1f}s")
    print(f"  Candidatos (min>=50): {len(candidates)}")

    # If no strict candidates, try relaxed
    if not candidates:
        print("\n  Busca relaxada (min>=0, todos positivos)...")
        for sname in promising:
            signals = sig_sets[sname]
            for ename, tp, sl, tr, buf, mb in exit_combos:
                for risk in risks:
                    sub_anns = []
                    sub_dds = []
                    sub_ns = []
                    for pname, sidx, slen, days in sub_periods:
                        ssigs = [(i, d, a, p) for i, d, a, p in signals if sidx <= i < sidx + slen]
                        trs = run_trades(df, ssigs, tp_mult=tp, sl_mult=sl,
                                         trailing_mult=tr, tp1_pct=0.50,
                                         post_tp1_buf=buf, max_bars=mb)
                        m = compute_metrics(trs, risk, days)
                        sub_anns.append(m["ann"])
                        sub_dds.append(m["dd"])
                        sub_ns.append(m["n"])

                    min_ann = min(sub_anns)
                    max_dd = max(sub_dds)
                    min_n = min(sub_ns)

                    if (min_ann >= 0 and max_dd <= 45 and min_n >= 8
                            and all(a > 0 for a in sub_anns)):
                        candidates.append({
                            "sig": sname, "exit": ename,
                            "tp": tp, "sl": sl, "tr": tr,
                            "buf": buf, "mb": mb, "risk": risk,
                            "min_ann": round(min_ann, 1),
                            "avg_ann": round(np.mean(sub_anns), 1),
                            "max_dd": round(max_dd, 1),
                            "min_n": min_n,
                            "sub_anns": [round(a, 1) for a in sub_anns],
                            "sub_dds": [round(d, 1) for d in sub_dds],
                            "sub_ns": sub_ns,
                            "score": round(min_ann - max_dd * 0.5, 1),
                        })
        print(f"  Total relaxado: {len(candidates)}")

    # Sort
    candidates.sort(key=lambda x: (x["min_ann"], -x["max_dd"], x["avg_ann"]),
                    reverse=True)

    # Print top 20
    print(f"\nTop 20:")
    for i, c in enumerate(candidates[:20]):
        print(f"  #{i+1} {c['sig']} | {c['exit']} r{c['risk']} | "
              f"min={c['min_ann']:+.1f} avg={c['avg_ann']:+.1f} "
              f"dd={c['max_dd']:.1f} n_min={c['min_n']}")
        print(f"       anns={[f'{a:+.1f}' for a in c['sub_anns']]} ns={c['sub_ns']}")

    # Detailed validation top 5
    print(f"\n{'='*70}")
    print("VALIDACAO DETALHADA TOP 5")
    print(f"{'='*70}")

    for i, c in enumerate(candidates[:5]):
        sig = sig_sets[c["sig"]]
        label = f"{c['sig']} | {c['exit']} | r{c['risk']}%"
        print(f"\n--- #{i+1}: {label} ---")

        for pname, sidx, slen, days in sub_periods:
            ssigs = [(j, d, a, p) for j, d, a, p in sig if sidx <= j < sidx + slen]
            trs = run_trades(df, ssigs, tp_mult=c["tp"], sl_mult=c["sl"],
                             trailing_mult=c["tr"], tp1_pct=0.50,
                             post_tp1_buf=c["buf"], max_bars=c["mb"])
            m = compute_metrics(trs, c["risk"], days)
            print(f"  {pname}: ann={m['ann']:+7.1f}% dd={m['dd']:5.1f}% "
                  f"wr={m['wr']:5.1f}% n={m['n']:3d} pf={m['pf']:.2f} eq={m['eq']:.0f}")

    # Save
    if candidates:
        best = candidates[0]
        parts = best["sig"].split("_")
        adx_min = int(parts[0].replace("adx", ""))
        rsi_l = int(parts[1].replace("rsi", ""))
        vol = float(parts[2].replace("vol", ""))

        rec = {
            "strategy": "confluence_v15_robust",
            "signal_params": {
                "adx_min": adx_min, "rsi_long_max": rsi_l,
                "rsi_short_min": 100 - rsi_l, "vol_mult": vol,
                "confluence_score_min": 5,
            },
            "exit_params": {
                "tp_atr_mult": best["tp"], "sl_atr_mult": best["sl"],
                "trailing_atr_mult": best["tr"],
                "post_tp1_sl_buffer": best["buf"],
                "tp1_pct": 0.50, "max_bars_held": best["mb"],
                "use_trailing": True,
            },
            "risk_per_trade": best["risk"],
            "validation": {
                "min_annual": best["min_ann"],
                "avg_annual": best["avg_ann"],
                "max_drawdown": best["max_dd"],
                "sub_anns": best["sub_anns"],
                "sub_ns": best["sub_ns"],
            },
            "total_candidates": len(candidates),
        }

        with open(OUTPATH, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"\nSalvo: {OUTPATH}")

        # Print recommendation
        print(f"\n{'='*70}")
        print("RECOMENDACAO")
        print(f"{'='*70}")
        print(f"  Sinal: adx_min={adx_min} rsi_long_max={rsi_l} vol_mult={vol}")
        print(f"  Saida: TP={best['tp']}x SL={best['sl']}x TR={best['tr']}x BUF={best['buf']} MB={best['mb']}")
        print(f"  Risk: {best['risk']}%")
        print(f"  Sub-anuais: {best['sub_anns']}")
        print(f"  Min: {best['min_ann']:+.1f}% | DD: {best['max_dd']:.1f}%")

    # Save all candidates
    with open(OUTPATH.replace(".json", "_all.json"), "w") as f:
        json.dump(candidates[:200], f, indent=2)


if __name__ == "__main__":
    main()
