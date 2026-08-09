r"""
optimize_v15_robust.py — Otimizacao robusta da Confluence v15.

Objetivo: Encontrar parametros que garantam >= 50% anual no MINIMO
de todos os sub-periodos (90d, 180d, 365d, 730d).

Abordagem eficiente:
1. Computar raw scores para TODOS os candles uma vez
2. Filtrar por score_min/adx/rsi/vol sem re-escanear
3. 2-fase: (a) encontrar melhores sinais no 730d, (b) refinar saidas nos sub-periodos
"""
import sys, os, json, time
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.explore_new_strategies import (
    load_data, run_trades, compute_metrics
)

CACHE = "/home/z/my-project/trade-signal/download/btc_1h_cache.csv"
OUTPATH = "/home/z/my-project/trade-signal/download/optimize_v15_robust_results.json"


def get_subperiods(df):
    """Retorna lista de (label, start_idx, sub_len, days)."""
    total_days = (df.index[-1] - df.index[0]).days
    sub_periods = []
    for label, offset in [("730d", 0), ("365d", total_days - 365),
                            ("180d", total_days - 180), ("90d", total_days - 90)]:
        if offset < 0:
            continue
        start_ts = df.index[0] + pd.Timedelta(days=offset)
        mask = df.index >= start_ts
        sub_df = df[mask]
        start_idx = df.index.get_loc(sub_df.index[0])
        days = (sub_df.index[-1] - sub_df.index[0]).days
        sub_periods.append((label, start_idx, len(sub_df), days))
    return sub_periods


def precompute_all_scores(df):
    """
    Computa raw scores para LONG e SHORT em todos os candles.
    Retorna DataFrame com colunas: long_score, short_score, + todas as
    variaveis de filtro (adx, rsi, vol_ratio, srk, srd, obv_trend, etc.)
    """
    n = len(df)
    results = []

    for i in range(200, n - 1):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        cr = ["ema20", "ema50", "ema200", "rsi", "rsi_delta", "atr",
             "macd", "macd_signal", "macd_hist", "volume", "volume_sma20",
             "bb_lower", "bb_upper", "bb_middle", "adx", "plus_di", "minus_di",
             "stoch_rsi_k", "stoch_rsi_d", "obv", "obv_sma20", "obv_trend",
             "atr_percentile"]
        if any(pd.isna(row.get(c)) for c in cr):
            continue

        cl = float(row["close"])
        e20 = float(row["ema20"])
        e50 = float(row["ema50"])
        e200 = float(row["ema200"])
        rsi = float(row["rsi"])
        atr = float(row["atr"])
        macd = float(row["macd"])
        macd_s = float(row["macd_signal"])
        p_macd = float(prev["macd"])
        p_macd_s = float(prev["macd_signal"])
        macd_h = float(row["macd_hist"])
        vol = float(row["volume"])
        vsma = float(row["volume_sma20"])
        adx = float(row["adx"])
        srk = float(row["stoch_rsi_k"])
        srd = float(row["stoch_rsi_d"])
        obv = float(row["obv"])
        obv_sma = float(row["obv_sma20"])
        obv_t = int(row["obv_trend"])
        ap = float(row.get("atr_percentile", 0.5))

        if ap < 0.10 or ap > 0.90 or atr <= 0 or cl <= 0:
            continue

        vol_ratio = vol / vsma if vsma > 0 else 0

        # LONG score
        ls = 0
        if cl > e50 and cl > e200: ls += 1
        if adx > 0: ls += 1  # will filter later
        if rsi > 40 and rsi < 100: ls += 1  # will filter later
        if srk < 100 and srk > srd: ls += 1  # will filter later
        if macd > macd_s and p_macd <= p_macd_s: ls += 2
        elif macd > macd_s and macd_h > 0: ls += 1
        if obv > obv_sma and obv_t == 1: ls += 1
        if vol_ratio >= 0: ls += 1  # will filter later
        near_ema = cl <= e20 * 1.005 or float(prev["low"]) <= e20
        if near_ema: ls += 1

        # SHORT score
        ss = 0
        if cl < e50 and cl < e200: ss += 1
        if adx > 0: ss += 1
        if rsi < 60 and rsi > 0: ss += 1
        if srk > 0 and srk < srd: ss += 1
        if macd < macd_s and p_macd >= p_macd_s: ss += 2
        elif macd < macd_s and macd_h < 0: ss += 1
        if obv < obv_sma and obv_t == -1: ss += 1
        if vol_ratio >= 0: ss += 1
        near_ema_s = cl >= e20 * 0.995 or float(prev["high"]) >= e20
        if near_ema_s: ss += 1

        results.append({
            "idx": i, "close": cl, "atr": atr,
            "adx": adx, "rsi": rsi, "srk": srk, "srd": srd,
            "vol_ratio": vol_ratio,
            "obv_trend": obv_t, "obv": obv, "obv_sma": obv_sma,
            "long_score": ls, "short_score": ss,
            "trend_long": 1 if (cl > e50 and cl > e200) else 0,
            "trend_short": 1 if (cl < e50 and cl < e200) else 0,
        })

    return pd.DataFrame(results)


def filter_signals(score_df, adx_min, rsi_long_max, vol_mult, score_min):
    """Filtra sinais a partir do DataFrame pre-computado."""
    rsi_short_min = 100 - rsi_long_max

    # LONG
    long_mask = (
        (score_df["long_score"] >= score_min) &
        (score_df["adx"] > adx_min) &
        (score_df["rsi"] > 40) &
        (score_df["rsi"] < rsi_long_max) &
        (score_df["srk"] < 65) &
        (score_df["srk"] > score_df["srd"]) &
        (score_df["vol_ratio"] >= vol_mult)
    )
    long_sigs = score_df.loc[long_mask, ["idx", "close", "atr"]].values.tolist()
    long_sigs = [(int(r[0]), "long", float(r[2]), float(r[1])) for r in long_sigs]

    # SHORT
    short_mask = (
        (score_df["short_score"] >= score_min) &
        (score_df["adx"] > adx_min) &
        (score_df["rsi"] < 60) &
        (score_df["rsi"] > rsi_short_min) &
        (score_df["srk"] > 35) &
        (score_df["srk"] < score_df["srd"]) &
        (score_df["vol_ratio"] >= vol_mult)
    )
    short_sigs = score_df.loc[short_mask, ["idx", "close", "atr"]].values.tolist()
    short_sigs = [(int(r[0]), "short", float(r[2]), float(r[1])) for r in short_sigs]

    return long_sigs + short_sigs


def eval_config_fast(signals, df, sub_periods, tp, sl, tr, buf, mb, risk):
    """Avalia config em todos sub-periodos. Early exit se algum negativo."""
    sub_anns = []
    sub_dds = []
    sub_wrs = []
    sub_ns = []
    all_positive = True
    total_trades = 0

    for pname, start_idx, sub_len, days in sub_periods:
        sub_sigs = [(i, d, a, p) for i, d, a, p in signals
                    if start_idx <= i < start_idx + sub_len]
        trades = run_trades(df, sub_sigs, tp_mult=tp, sl_mult=sl,
                            trailing_mult=tr, tp1_pct=0.50,
                            post_tp1_buf=buf, max_bars=mb)
        m = compute_metrics(trades, risk, days)
        sub_anns.append(m["ann"])
        sub_dds.append(m["dd"])
        sub_wrs.append(m["wr"])
        sub_ns.append(m["n"])
        total_trades += m["n"]
        if m["ann"] < 0:
            all_positive = False

    min_ann = min(sub_anns)
    avg_ann = np.mean(sub_anns)
    max_dd = max(sub_dds)
    avg_wr = np.mean(sub_wrs)
    min_n = min(sub_ns)

    return {
        "min_ann": min_ann, "avg_ann": avg_ann, "max_dd": max_dd,
        "avg_wr": avg_wr, "min_n": min_n, "total_trades": total_trades,
        "all_positive": all_positive,
        "sub_anns": sub_anns, "sub_dds": sub_dds,
        "sub_wrs": sub_wrs, "sub_ns": sub_ns,
    }


def main():
    print("=" * 70)
    print("OTIMIZACAO ROBUSTA CONFLUENCE v15")
    print("Objetivo: min(sub_period_annual) >= 50%")
    print("=" * 70)

    df = load_data()
    sub_periods = get_subperiods(df)
    print(f"Sub-periodos: {[(s[0], s[3]) for s in sub_periods]}")

    # =================================================================
    # PHASE 1: Pre-compute all raw scores (ONE PASS)
    # =================================================================
    print("\n--- FASE 1: Pre-computando scores (1 pass) ---")
    t0 = time.time()
    score_df = precompute_all_scores(df)
    print(f"  {len(score_df):,} candles com indicadores validos em {time.time()-t0:.1f}s")

    # =================================================================
    # PHASE 2: Generate signal sets via filtering (fast)
    # =================================================================
    print("\n--- FASE 2: Gerando sinais via filtro ---")
    t0 = time.time()

    signal_params = [
        # (adx_min, rsi_long_max, vol_mult, score_min)
        (15, 50, 0.30, 4), (15, 50, 0.30, 5), (15, 50, 0.30, 6),
        (15, 55, 0.30, 4), (15, 55, 0.30, 5), (15, 55, 0.30, 6),
        (15, 60, 0.30, 4), (15, 60, 0.30, 5), (15, 60, 0.30, 6),
        (15, 50, 0.35, 4), (15, 50, 0.35, 5), (15, 50, 0.35, 6),
        (15, 55, 0.35, 4), (15, 55, 0.35, 5), (15, 55, 0.35, 6),
        (15, 60, 0.35, 4), (15, 60, 0.35, 5), (15, 60, 0.35, 6),
        (15, 50, 0.40, 4), (15, 50, 0.40, 5), (15, 50, 0.40, 6),
        (15, 55, 0.40, 4), (15, 55, 0.40, 5), (15, 55, 0.40, 6),
        (15, 60, 0.40, 4), (15, 60, 0.40, 5), (15, 60, 0.40, 6),
        (18, 50, 0.30, 4), (18, 50, 0.30, 5), (18, 50, 0.30, 6),
        (18, 55, 0.30, 4), (18, 55, 0.30, 5), (18, 55, 0.30, 6),
        (18, 60, 0.30, 4), (18, 60, 0.30, 5), (18, 60, 0.30, 6),
        (18, 50, 0.35, 4), (18, 50, 0.35, 5), (18, 50, 0.35, 6),
        (18, 55, 0.35, 4), (18, 55, 0.35, 5), (18, 55, 0.35, 6),
        (18, 60, 0.35, 4), (18, 60, 0.35, 5), (18, 60, 0.35, 6),
        (18, 50, 0.40, 4), (18, 50, 0.40, 5), (18, 50, 0.40, 6),
        (18, 55, 0.40, 4), (18, 55, 0.40, 5), (18, 55, 0.40, 6),
        (18, 60, 0.40, 4), (18, 60, 0.40, 5), (18, 60, 0.40, 6),
        (20, 50, 0.30, 4), (20, 50, 0.30, 5), (20, 50, 0.30, 6),
        (20, 55, 0.30, 4), (20, 55, 0.30, 5), (20, 55, 0.30, 6),
        (20, 60, 0.30, 4), (20, 60, 0.30, 5), (20, 60, 0.30, 6),
        (20, 50, 0.35, 4), (20, 50, 0.35, 5), (20, 50, 0.35, 6),
        (20, 55, 0.35, 4), (20, 55, 0.35, 5), (20, 55, 0.35, 6),
        (20, 60, 0.35, 4), (20, 60, 0.35, 5), (20, 60, 0.35, 6),
        (20, 50, 0.40, 4), (20, 50, 0.40, 5), (20, 50, 0.40, 6),
        (20, 55, 0.40, 4), (20, 55, 0.40, 5), (20, 55, 0.40, 6),
        (20, 60, 0.40, 4), (20, 60, 0.40, 5), (20, 60, 0.40, 6),
        (25, 50, 0.30, 4), (25, 50, 0.30, 5), (25, 50, 0.30, 6),
        (25, 55, 0.30, 4), (25, 55, 0.30, 5), (25, 55, 0.30, 6),
        (25, 60, 0.30, 4), (25, 60, 0.30, 5), (25, 60, 0.30, 6),
        (25, 50, 0.35, 4), (25, 50, 0.35, 5), (25, 50, 0.35, 6),
        (25, 55, 0.35, 4), (25, 55, 0.35, 5), (25, 55, 0.35, 6),
        (25, 60, 0.35, 4), (25, 60, 0.35, 5), (25, 60, 0.35, 6),
        (25, 50, 0.40, 4), (25, 50, 0.40, 5), (25, 50, 0.40, 6),
        (25, 55, 0.40, 4), (25, 55, 0.40, 5), (25, 55, 0.40, 6),
        (25, 60, 0.40, 4), (25, 60, 0.40, 5), (25, 60, 0.40, 6),
        # Extras: vol 0.50
        (15, 55, 0.50, 5), (18, 55, 0.50, 5), (20, 55, 0.50, 5),
        (25, 55, 0.50, 5),
        (15, 55, 0.50, 4), (18, 55, 0.50, 4), (20, 55, 0.50, 4),
        (25, 55, 0.50, 4),
    ]

    sig_sets = {}
    for adx, rsi_l, vol, sm in signal_params:
        sname = f"adx{adx}_rsi{rsi_l}_vol{vol}_s{sm}"
        sig_sets[sname] = filter_signals(score_df, adx, rsi_l, vol, sm)

    print(f"  {len(sig_sets)} signal sets gerados em {time.time()-t0:.1f}s")
    for sname, sigs in sorted(sig_sets.items()):
        if len(sigs) >= 15:
            print(f"    {sname}: {len(sigs)} sinais")

    # =================================================================
    # PHASE 2b: Pre-filter signal sets on 730d with baseline exits
    # =================================================================
    print("\n--- FASE 2b: Pre-filtro de sinais (730d, TP8 SL2.5 TR3) ---")
    t0 = time.time()

    baseline_exits = [
        ("TP8_SL2.5_TR3", 8.0, 2.5, 3.0, 0.2, 120),
        ("TP6_SL2_TR3", 6.0, 2.0, 3.0, 0.2, 120),
        ("TP10_SL3_TR4", 10.0, 3.0, 4.0, 0.2, 120),
    ]

    promising_sigs = {}  # sname -> best_730_ann
    for sname, signals in sig_sets.items():
        if len(signals) < 20:
            continue
        best_ann = -999
        for ename, tp, sl, tr, buf, mb in baseline_exits:
            for risk in [2.0, 3.0]:
                trades = run_trades(df, signals, tp_mult=tp, sl_mult=sl,
                                    trailing_mult=tr, tp1_pct=0.50,
                                    post_tp1_buf=buf, max_bars=mb)
                m = compute_metrics(trades, risk, sub_periods[0][3])
                if m["n"] >= 20 and m["wr"] >= 30 and m["ann"] > best_ann:
                    best_ann = m["ann"]
        if best_ann >= 100:  # Promising on 730d
            promising_sigs[sname] = best_ann

    print(f"  {len(promising_sigs)}/{len(sig_sets)} signal sets promissores (730d >= 100%)")
    for sname, ann in sorted(promising_sigs.items(), key=lambda x: -x[1]):
        print(f"    {sname}: {ann:+.1f}%")

    # =================================================================
    # PHASE 3: Full grid search on promising signals
    # =================================================================
    exit_grid = [
        ("TP6_SL2_TR2.5", 6.0, 2.0, 2.5, 0.1, 96),
        ("TP6_SL2_TR3", 6.0, 2.0, 3.0, 0.1, 96),
        ("TP6_SL2_TR3", 6.0, 2.0, 3.0, 0.2, 96),
        ("TP6_SL2.5_TR3", 6.0, 2.5, 3.0, 0.1, 96),
        ("TP6_SL2.5_TR3", 6.0, 2.5, 3.0, 0.2, 96),
        ("TP6_SL2.5_TR3", 6.0, 2.5, 3.0, 0.2, 120),
        ("TP8_SL2_TR2.5", 8.0, 2.0, 2.5, 0.1, 96),
        ("TP8_SL2_TR3", 8.0, 2.0, 3.0, 0.1, 96),
        ("TP8_SL2_TR3", 8.0, 2.0, 3.0, 0.2, 96),
        ("TP8_SL2_TR3", 8.0, 2.0, 3.0, 0.2, 120),
        ("TP8_SL2_TR4", 8.0, 2.0, 4.0, 0.2, 120),
        ("TP8_SL2.5_TR2.5", 8.0, 2.5, 2.5, 0.1, 96),
        ("TP8_SL2.5_TR3", 8.0, 2.5, 3.0, 0.1, 96),
        ("TP8_SL2.5_TR3", 8.0, 2.5, 3.0, 0.2, 96),
        ("TP8_SL2.5_TR3", 8.0, 2.5, 3.0, 0.2, 120),
        ("TP8_SL2.5_TR3.5", 8.0, 2.5, 3.5, 0.2, 120),
        ("TP8_SL2.5_TR4", 8.0, 2.5, 4.0, 0.2, 120),
        ("TP8_SL3_TR3", 8.0, 3.0, 3.0, 0.2, 120),
        ("TP8_SL3_TR4", 8.0, 3.0, 4.0, 0.2, 120),
        ("TP10_SL2_TR3", 10.0, 2.0, 3.0, 0.2, 120),
        ("TP10_SL2.5_TR3", 10.0, 2.5, 3.0, 0.2, 120),
        ("TP10_SL2.5_TR4", 10.0, 2.5, 4.0, 0.2, 120),
        ("TP10_SL3_TR4", 10.0, 3.0, 4.0, 0.2, 120),
    ]

    risk_levels = [2.0, 3.0]
    total_combos = len(promising_sigs) * len(exit_grid) * len(risk_levels)
    print(f"\n--- FASE 3: Grid search ({total_combos:,} combinacoes) ---")
    t0 = time.time()

    candidates = []
    tested = 0
    passed = 0

    for sname in promising_sigs:
        signals = sig_sets[sname]
        for ename, tp, sl, tr, buf, mb in exit_grid:
            for risk in risk_levels:
                tested += 1

                # Quick 730d pre-filter
                trades_730 = run_trades(df, signals, tp_mult=tp, sl_mult=sl,
                                        trailing_mult=tr, tp1_pct=0.50,
                                        post_tp1_buf=buf, max_bars=mb)
                m730 = compute_metrics(trades_730, risk, sub_periods[0][3])

                if m730["n"] < 25 or m730["wr"] < 30 or m730["dd"] > 40:
                    continue

                # Full evaluation
                ev = eval_config_fast(signals, df, sub_periods,
                                       tp, sl, tr, buf, mb, risk)

                if (ev["min_ann"] >= 50 and ev["max_dd"] <= 40
                        and ev["avg_wr"] >= 35 and ev["min_n"] >= 10):
                    passed += 1
                    candidates.append({
                        "sig": sname, "exit": ename,
                        "tp": tp, "sl": sl, "tr": tr,
                        "buf": buf, "mb": mb, "risk": risk,
                        "min_ann": ev["min_ann"],
                        "avg_ann": round(ev["avg_ann"], 1),
                        "max_dd": round(ev["max_dd"], 1),
                        "avg_wr": round(ev["avg_wr"], 1),
                        "min_n": ev["min_n"],
                        "total_trades": ev["total_trades"],
                        "sub_anns": [round(a, 1) for a in ev["sub_anns"]],
                        "sub_dds": [round(d, 1) for d in ev["sub_dds"]],
                        "sub_ns": ev["sub_ns"],
                        "score": round(ev["min_ann"] - ev["max_dd"] * 0.5, 1),
                    })

                if tested % 500 == 0:
                    elapsed = time.time() - t0
                    rate = tested / max(elapsed, 0.01)
                    print(f"  {tested:,}/{total_combos:,} | {rate:.0f}/s | candidatos: {passed}")

    elapsed = time.time() - t0
    print(f"\n  Scan: {tested:,} em {elapsed:.1f}s")
    print(f"  Candidatos (min_ann>=50): {passed}")

    # =================================================================
    # PHASE 4: Ranking
    # =================================================================
    print(f"\n--- FASE 4: Ranking ---")

    if not candidates:
        print("  Nenhum config com min_ann>=50. Salvando tudo para analise...")
        # Try to find best available anyway
        # Run a broader search without the min_ann constraint
        print("  Executando busca relaxada (min_ann >= 0, todos positivos)...")
        for sname in promising_sigs:
            signals = sig_sets[sname]
            for ename, tp, sl, tr, buf, mb in exit_grid:
                for risk in risk_levels:
                    ev = eval_config_fast(signals, df, sub_periods,
                                           tp, sl, tr, buf, mb, risk)
                    if (ev["all_positive"] and ev["min_n"] >= 8
                            and ev["max_dd"] <= 45 and ev["avg_wr"] >= 33):
                        candidates.append({
                            "sig": sname, "exit": ename,
                            "tp": tp, "sl": sl, "tr": tr,
                            "buf": buf, "mb": mb, "risk": risk,
                            "min_ann": ev["min_ann"],
                            "avg_ann": round(ev["avg_ann"], 1),
                            "max_dd": round(ev["max_dd"], 1),
                            "avg_wr": round(ev["avg_wr"], 1),
                            "min_n": ev["min_n"],
                            "sub_anns": [round(a, 1) for a in ev["sub_anns"]],
                            "sub_dds": [round(d, 1) for d in ev["sub_dds"]],
                            "sub_ns": ev["sub_ns"],
                            "score": round(ev["min_ann"] - ev["max_dd"] * 0.5, 1),
                        })
        print(f"  Total com busca relaxada: {len(candidates)}")

    candidates.sort(key=lambda x: (x["min_ann"], -x["max_dd"], x["avg_ann"]),
                    reverse=True)

    print(f"\nTop 20:")
    print(f"{'#':>3} {'Config':<50} {'min':>6} {'avg':>7} {'dd':>5} {'wr':>5} {'n_min':>5} {'sc':>6}")
    print("-" * 100)
    for i, c in enumerate(candidates[:20]):
        cfg = f"{c['sig']} | {c['exit']} r{c['risk']}"
        print(f"{i+1:>3} {cfg:<50} {c['min_ann']:>+6.1f} {c['avg_ann']:>+7.1f} "
              f"{c['max_dd']:>5.1f} {c['avg_wr']:>5.1f} {c['min_n']:>5} {c['score']:>+6.1f}")
        print(f"    anns={[f'{a:+.1f}' for a in c['sub_anns']]}  ns={c['sub_ns']}")

    # =================================================================
    # PHASE 5: Detailed validation + save
    # =================================================================
    print(f"\n{'='*70}")
    print("VALIDACAO DETALHADA TOP 5")
    print(f"{'='*70}")

    top5_validated = []
    for i, c in enumerate(candidates[:5]):
        sig = sig_sets[c["sig"]]
        label = f"{c['sig']} | {c['exit']} | r{c['risk']}"
        print(f"\n--- #{i+1}: {label} ---")

        for pname, start_idx, sub_len, days in sub_periods:
            sub_sigs = [(j, d, a, p) for j, d, a, p in sig
                        if start_idx <= j < start_idx + sub_len]
            trades = run_trades(df, sub_sigs, tp_mult=c["tp"], sl_mult=c["sl"],
                                trailing_mult=c["tr"], tp1_pct=0.50,
                                post_tp1_buf=c["buf"], max_bars=c["mb"])
            m = compute_metrics(trades, c["risk"], days)
            print(f"  {pname}: ann={m['ann']:+7.1f}% dd={m['dd']:5.1f}% "
                  f"wr={m['wr']:5.1f}% n={m['n']:3d} pf={m['pf']:.2f} eq={m['eq']:.0f}")

        top5_validated.append({"config": c, "label": label})

    # Save recommendation
    print(f"\n{'='*70}")
    print("RECOMENDACAO FINAL")
    print(f"{'='*70}")

    if candidates:
        best = candidates[0]
        parts = best["sig"].split("_")
        adx_min = int(parts[0].replace("adx", ""))
        rsi_l = int(parts[1].replace("rsi", ""))
        vol = float(parts[2].replace("vol", ""))
        score_min = int(parts[3].replace("s", ""))

        recommendation = {
            "strategy": "confluence_v15_robust",
            "signal_params": {
                "adx_min": adx_min,
                "rsi_long_max": rsi_l,
                "rsi_short_min": 100 - rsi_l,
                "vol_mult": vol,
                "confluence_score_min": score_min,
            },
            "exit_params": {
                "tp_atr_mult": best["tp"],
                "sl_atr_mult": best["sl"],
                "trailing_atr_mult": best["tr"],
                "post_tp1_sl_buffer": best["buf"],
                "tp1_pct": 0.50,
                "max_bars_held": best["mb"],
                "use_trailing": True,
            },
            "risk_per_trade": best["risk"],
            "full_validation": {
                "min_annual": best["min_ann"],
                "avg_annual": best["avg_ann"],
                "max_drawdown": best["max_dd"],
                "avg_winrate": best["avg_wr"],
                "min_trades_subperiod": best["min_n"],
            },
            "sub_period_details": {
                "labels": [s[0] for s in sub_periods],
                "annual_returns": best["sub_anns"],
                "drawdowns": best["sub_dds"],
                "trade_counts": best["sub_ns"],
            },
            "total_candidates": len(candidates),
            "top5": [{
                "label": v["label"],
                "min_ann": v["config"]["min_ann"],
                "avg_ann": v["config"]["avg_ann"],
                "max_dd": v["config"]["max_dd"],
                "sub_anns": v["config"]["sub_anns"],
            } for v in top5_validated],
        }

        print(f"  Sinal: {best['sig']}")
        print(f"  Saida: {best['exit']}")
        print(f"  Risk: {best['risk']}%")
        print(f"  Sub-anuais: {best['sub_anns']}")
        print(f"  Min anual: {best['min_ann']:+.1f}% | Max DD: {best['max_dd']:.1f}%")

        with open(OUTPATH, "w") as f:
            json.dump(recommendation, f, indent=2)
        print(f"\n  Salvo: {OUTPATH}")
    else:
        print("  Nenhum config encontrado.")

    # Save all candidates
    all_out = OUTPATH.replace(".json", "_all.json")
    with open(all_out, "w") as f:
        json.dump(candidates[:200], f, indent=2)
    print(f"  Todos candidatos: {all_out}")


if __name__ == "__main__":
    main()
