r"""
validate_confluence.py — Validacao rapida da Confluencia em sub-periodos.
"""
import json, time, sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.explore_new_strategies import (
    run_trades, gen_confluence, compute_metrics
)

CACHE = "/home/z/my-project/trade-signal/download/btc_1h_cache.csv"


def main():
    print("=" * 70)
    print("VALIDACAO CONFLUENCIA — Sub-periodos")
    print("=" * 70)

    df = pd.read_csv(CACHE, index_col=0, parse_dates=True)
    total_days = (df.index[-1] - df.index[0]).days
    print(f"Dados: {len(df):,} candles, {total_days} dias")

    # Generate base signals with the top configs
    signal_configs = [
        ("adx20_rsi55_vol0.4", 20, 55, 0.4),
        ("adx20_rsi60_vol0.4", 20, 60, 0.4),
        ("adx15_rsi55_vol0.4", 15, 55, 0.4),
        ("adx15_rsi60_vol0.4", 15, 60, 0.4),
        ("adx18_rsi55_vol0.4", 18, 55, 0.4),
        ("adx20_rsi55_vol0.35", 20, 55, 0.35),
        ("adx20_rsi55_vol0.5", 20, 55, 0.5),
        ("adx20_rsi50_vol0.4", 20, 50, 0.4),
    ]

    # Exit params to test
    exit_configs = [
        ("TP8_SL2_TR3_buf0.2_mb120", 8.0, 2.0, 3.0, 0.2, 120),
        ("TP8_SL2.5_TR3_buf0.2_mb120", 8.0, 2.5, 3.0, 0.2, 120),
        ("TP6_SL2_TR3_buf0.2_mb120", 6.0, 2.0, 3.0, 0.2, 120),
        ("TP8_SL2_TR4_buf0.2_mb120", 8.0, 2.0, 4.0, 0.2, 120),
        ("TP10_SL2_TR3_buf0.2_mb120", 10.0, 2.0, 3.0, 0.2, 120),
        ("TP8_SL1.5_TR3_buf0.2_mb120", 8.0, 1.5, 3.0, 0.2, 120),
        ("TP8_SL2_TR3.5_buf0.2_mb120", 8.0, 2.0, 3.5, 0.2, 120),
        ("TP8_SL2_TR3_buf0.1_mb120", 8.0, 2.0, 3.0, 0.1, 120),
    ]

    risk_levels = [2.0, 3.0]

    # Sub-period definitions (start_offset_days, label)
    sub_periods = []
    for label, offset in [("730d", 0), ("365d", total_days-365), ("180d", total_days-180), ("90d", total_days-90)]:
        if offset < 0:
            continue
        start_ts = df.index[0] + pd.Timedelta(days=offset)
        mask = df.index >= start_ts
        sub_df = df[mask]
        start_idx = df.index.get_loc(sub_df.index[0])
        days = (sub_df.index[-1] - sub_df.index[0]).days
        sub_periods.append((label, start_idx, len(sub_df), days))

    print(f"\nSub-periodos: {[(s[0], s[3]) for s in sub_periods]}")

    # Generate all signal sets
    print("\nGerando sinais...")
    t0 = time.time()
    sig_sets = {}
    for sname, adx, rsi_l, vol in signal_configs:
        sig_sets[sname] = gen_confluence(df, adx_min=adx, rsi_long_max=rsi_l,
                                         rsi_short_min=100-rsi_l, vol_mult=vol)
        print(f"  {sname}: {len(sig_sets[sname])} sinais")
    print(f"  {time.time()-t0:.1f}s")

    # Quick scan: find best (sig_config, exit_config, risk) on full period
    print(f"\n--- Scan 730d (full) ---")
    full_results = []
    count = 0
    for sname, signals in sig_sets.items():
        if len(signals) < 20:
            continue
        for ename, tp, sl, tr, buf, mb in exit_configs:
            try:
                trades = run_trades(df, signals, tp_mult=tp, sl_mult=sl,
                                    trailing_mult=tr, tp1_pct=0.50,
                                    post_tp1_buf=buf, max_bars=mb)
            except:
                continue
            count += 1
            for risk in risk_levels:
                m = compute_metrics(trades, risk, total_days)
                if m["n"] >= 30 and m["wr"] >= 35 and m["dd"] <= 40 and m["ann"] >= 50:
                    sc = m["ann"] / max(m["dd"], 1)
                    full_results.append({
                        "sig": sname, "exit": ename, "risk": risk,
                        "tp": tp, "sl": sl, "tr": tr, "buf": buf, "mb": mb,
                        "ann": m["ann"], "dd": m["dd"], "wr": m["wr"],
                        "pf": m["pf"], "n": m["n"], "score": round(sc, 1)
                    })
    
    print(f"  {count} testes, {len(full_results)} com >=50% ann")
    full_results.sort(key=lambda x: x["score"], reverse=True)

    print(f"\nTop 15 (730d):")
    for i, r in enumerate(full_results[:15]):
        print(f"  #{i+1} [{r['sig']}] {r['exit']} r={r['risk']}% "
              f"ann={r['ann']:+.1f}% dd={r['dd']:.1f}% wr={r['wr']:.1f}% pf={r['pf']:.2f} n={r['n']} sc={r['score']}")

    # Validate top 10 on sub-periods
    print(f"\n{'='*70}")
    print("VALIDACAO SUB-PERIODOS (top 10)")
    print(f"{'='*70}")

    best_robust = None
    best_robust_score = -999

    for r in full_results[:10]:
        signals = sig_sets[r["sig"]]
        label = f"{r['sig']} | {r['exit']} | r{r['risk']}"
        print(f"\n--- {label} ---")

        sub_anns = []
        for pname, start_idx, sub_len, days in sub_periods:
            sub_sigs = [(i, d, a, p) for i, d, a, p in signals
                        if start_idx <= i < start_idx + sub_len]
            trades = run_trades(df, sub_sigs, tp_mult=r["tp"], sl_mult=r["sl"],
                                trailing_mult=r["tr"], tp1_pct=0.50,
                                post_tp1_buf=r["buf"], max_bars=r["mb"])
            m = compute_metrics(trades, r["risk"], days)
            sub_anns.append(m["ann"])
            print(f"  {pname}: ann={m['ann']:+7.1f}% dd={m['dd']:5.1f}% wr={m['wr']:5.1f}% n={m['n']:3d} pf={m['pf']:.2f}")

        all_pos = all(a > 0 for a in sub_anns)
        min_ann = min(sub_anns)
        avg_ann = np.mean(sub_anns)
        robust_sc = avg_ann / max(avg_ann * 0.1 + 1, 1)  # normalized
        if all_pos:
            robust_sc = avg_ann  # simple: higher avg = better, as long as all positive
        else:
            robust_sc = avg_ann * 0.3  # penalty

        print(f"  >> min={min_ann:+.1f}% avg={avg_ann:+.1f}% all_pos={all_pos} robust_sc={robust_sc:.1f}")

        if robust_sc > best_robust_score:
            best_robust_score = robust_sc
            best_robust = {"result": r, "sub_anns": sub_anns, "all_pos": all_pos,
                          "label": label, "robust_sc": robust_sc}

    # Final recommendation
    print(f"\n{'='*70}")
    print("RECOMENDACAO FINAL")
    print(f"{'='*70}")
    if best_robust:
        r = best_robust["result"]
        print(f"  Config: {best_robust['label']}")
        print(f"  TP={r['tp']}x SL={r['sl']}x TR={r['tr']}x BUF={r['buf']} MB={r['mb']}")
        print(f"  Risk={r['risk']}% | 730d: ann={r['ann']:+.1f}% dd={r['dd']:.1f}% wr={r['wr']:.1f}% pf={r['pf']:.2f}")
        print(f"  Robustez: min_ann={min(best_robust['sub_anns']):+.1f}% all_positive={best_robust['all_pos']}")

        # Parse signal params from sig name
        sig_name = r["sig"]
        parts = sig_name.split("_")
        adx_min = int(parts[0].replace("adx", ""))
        rsi_l = int(parts[1].replace("rsi", ""))
        vol = float(parts[2].replace("vol", ""))

        out = {
            "strategy": "confluence_v15",
            "signal_params": {
                "adx_min": adx_min,
                "rsi_long_max": rsi_l,
                "rsi_short_min": 100 - rsi_l,
                "vol_mult": vol,
                "confluence_score_min": 5,
            },
            "exit_params": {
                "tp_atr_mult": r["tp"],
                "sl_atr_mult": r["sl"],
                "trailing_atr_mult": r["tr"],
                "post_tp1_sl_buffer": r["buf"],
                "tp1_pct": 0.50,
                "max_bars_held": r["mb"],
                "use_trailing": True,
            },
            "risk_per_trade": r["risk"],
            "full_period_730d": {"ann": r["ann"], "dd": r["dd"], "wr": r["wr"], "pf": r["pf"], "n": r["n"]},
            "sub_period_anns": best_robust["sub_anns"],
            "all_subperiods_positive": best_robust["all_pos"],
        }
        outpath = "/home/z/my-project/trade-signal/download/confluence_v15_params.json"
        with open(outpath, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Salvo: {outpath}")
    else:
        print("  Nenhum config robusto encontrado.")


if __name__ == "__main__":
    main()
