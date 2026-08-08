r"""
optimize_v15_quick.py — Teste direcionado de variantes do v15.

Em vez de grid search, testa variantes especificas que podem
melhorar o 90d sem quebrar o resto.
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


def test_config(df, sub_periods, signals, label, tp, sl, tr, buf, mb, risk):
    """Testa config e retorna detalhamento por sub-periodo."""
    results = {}
    all_anns = []
    for pname, sidx, slen, days in sub_periods:
        ssigs = [(i, d, a, p) for i, d, a, p in signals if sidx <= i < sidx + slen]
        trs = run_trades(df, ssigs, tp_mult=tp, sl_mult=sl,
                         trailing_mult=tr, tp1_pct=0.50,
                         post_tp1_buf=buf, max_bars=mb)
        m = compute_metrics(trs, risk, days)
        results[pname] = m
        all_anns.append(m["ann"])
    results["min_ann"] = min(all_anns)
    results["avg_ann"] = np.mean(all_anns)
    results["max_dd"] = max(results[p]["dd"] for p, _, _, _ in sub_periods)
    results["label"] = label
    results["sub_anns"] = all_anns
    results["all_pos"] = all(a > 0 for a in all_anns)
    return results


def main():
    print("=" * 70)
    print("TESTE DIRECIONADO CONFLUENCE v15")
    print("=" * 70)

    df = load_data()
    total_days = (df.index[-1] - df.index[0]).days

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

    sp_labels = [s[0] for s in sub_periods]
    print(f"Sub-periodos: {sp_labels}")

    # =================================================================
    # Generate signals (only the most promising configs)
    # =================================================================
    print("\nGerando sinais...")
    t0 = time.time()

    # Current v15 params + variants
    sig_configs = [
        # Current v15 (baseline)
        ("v15_baseline", 20, 55, 0.35),
        # Lower ADX (more signals)
        ("v15_adx15", 15, 55, 0.35),
        ("v15_adx18", 18, 55, 0.35),
        # Wider RSI
        ("v15_rsi60", 20, 60, 0.35),
        ("v15_adx15_rsi60", 15, 60, 0.35),
        # Lower volume (more signals)
        ("v15_vol0.3", 20, 55, 0.3),
        ("v15_adx15_vol0.3", 15, 55, 0.3),
        ("v15_adx18_vol0.3", 18, 55, 0.3),
        ("v15_adx15_rsi60_vol0.3", 15, 60, 0.3),
        # Even more aggressive
        ("v15_adx15_rsi65", 15, 65, 0.3),
        ("v15_adx15_rsi65_vol0.25", 15, 65, 0.25),
    ]

    sig_sets = {}
    for sname, adx, rsi_l, vol in sig_configs:
        sig_sets[sname] = gen_confluence(df, adx_min=adx, rsi_long_max=rsi_l,
                                         rsi_short_min=100-rsi_l, vol_mult=vol)
        print(f"  {sname}: {len(sig_sets[sname])} sinais")

    print(f"Sinais gerados em {time.time()-t0:.1f}s")

    # =================================================================
    # Test exit variants on all signal sets
    # =================================================================
    print("\n--- Testando combinacoes ---")
    t0 = time.time()

    exit_configs = [
        # Current v15 exits
        ("cur", 8.0, 2.5, 3.0, 0.2, 120),
        # Tighter SL
        ("sl2", 8.0, 2.0, 3.0, 0.2, 120),
        ("sl2_b0.1", 8.0, 2.0, 3.0, 0.1, 120),
        ("sl2_tr4", 8.0, 2.0, 4.0, 0.2, 120),
        # Tighter TP (faster exits)
        ("tp6", 6.0, 2.0, 3.0, 0.2, 120),
        ("tp6_sl2.5", 6.0, 2.5, 3.0, 0.2, 120),
        # Wider TP
        ("tp10", 10.0, 2.5, 3.0, 0.2, 120),
        ("tp10_tr4", 10.0, 2.5, 4.0, 0.2, 120),
        # Shorter max bars
        ("mb96", 8.0, 2.5, 3.0, 0.2, 96),
        ("sl2_mb96", 8.0, 2.0, 3.0, 0.2, 96),
    ]

    risks = [2.0, 3.0]
    all_results = []
    tested = 0

    for sname, signals in sig_sets.items():
        if len(signals) < 15:
            continue
        for ename, tp, sl, tr, buf, mb in exit_configs:
            for risk in risks:
                tested += 1
                label = f"{sname} | {ename} | r{risk}%"
                r = test_config(df, sub_periods, signals, label,
                               tp, sl, tr, buf, mb, risk)
                all_results.append(r)

                if tested % 20 == 0:
                    print(f"  {tested} testados... {time.time()-t0:.0f}s")

    print(f"\n{tested} combinacoes testadas em {time.time()-t0:.1f}s")

    # =================================================================
    # Filter and rank
    # =================================================================
    # First: all sub-periods positive
    positive = [r for r in all_results if r["all_pos"] and r["max_dd"] <= 45]
    print(f"\nTodos sub-periodos positivos: {len(positive)}")

    # Sort by min_ann
    positive.sort(key=lambda x: x["min_ann"], reverse=True)

    # Print top 30
    print(f"\n{'='*70}")
    print(f"TOP 30 (todos positivos)")
    print(f"{'='*70}")
    print(f"{'#':>3} {'Config':<55} {'min':>6} {'avg':>7} {'dd':>5}")
    print("-" * 85)
    for i, r in enumerate(positive[:30]):
        anns_str = " ".join(f"{sp_labels[j]}={r['sub_anns'][j]:+.0f}" for j in range(len(sp_labels)))
        print(f"{i+1:>3} {r['label']:<55} {r['min_ann']:>+6.1f} {r['avg_ann']:>+7.1f} {r['max_dd']:>5.1f}")
        print(f"    {anns_str}")

    # Those with min_ann >= 50
    robust = [r for r in positive if r["min_ann"] >= 50 and r["max_dd"] <= 40]
    print(f"\n*** Robustos (min>=50%, dd<=40%): {len(robust)} ***")
    for i, r in enumerate(robust[:10]):
        anns_str = " ".join(f"{sp_labels[j]}={r['sub_anns'][j]:+.0f}" for j in range(len(sp_labels)))
        print(f"  #{i+1} {r['label']} | min={r['min_ann']:+.1f} dd={r['max_dd']:.1f}")
        print(f"       {anns_str}")

    # =================================================================
    # Detailed validation of best
    # =================================================================
    if robust:
        best = robust[0]
        print(f"\n{'='*70}")
        print(f"DETALHAMENTO DO MELHOR: {best['label']}")
        print(f"{'='*70}")
        for pname in sp_labels:
            m = best[pname]
            print(f"  {pname}: ann={m['ann']:+7.1f}% dd={m['dd']:5.1f}% "
                  f"wr={m['wr']:5.1f}% n={m['n']:3d} pf={m['pf']:.2f} eq={m['eq']:.0f}")

        # Save
        rec = {"best_label": best["label"], "min_ann": best["min_ann"],
               "sub_anns": best["sub_anns"], "all_pos": best["all_pos"]}
        with open("/home/z/my-project/trade-signal/download/optimize_v15_quick.json", "w") as f:
            json.dump(rec, f, indent=2)
    elif positive:
        best = positive[0]
        print(f"\nMelhor (todos positivos, mas min<50): {best['label']}")
        print(f"  Sub-anuais: {[f'{a:+.1f}' for a in best['sub_anns']]}")

        # Try 3% risk if best was 2%
        # Or try other tricks

    # Save all results
    with open("/home/z/my-project/trade-signal/download/optimize_v15_quick_all.json", "w") as f:
        json.dump([{k: v for k, v in r.items() if k != 'label' and not isinstance(v, dict)}
                   for r in positive[:100]], f, indent=2)
    print(f"\nResultados salvos.")


if __name__ == "__main__":
    main()
