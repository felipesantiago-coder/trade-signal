import time, sys, json, numpy as np, pandas as pd
sys.path.insert(0, '.')
from scripts.explore_new_strategies import load_data, gen_confluence, run_trades, compute_metrics

df = load_data()
total_days = (df.index[-1] - df.index[0]).days

sub_periods = []
for label, offset in [('730d',0),('365d',total_days-365),('180d',total_days-180),('90d',total_days-90)]:
    if offset < 0: continue
    start_ts = df.index[0] + pd.Timedelta(days=offset)
    mask = df.index >= start_ts
    sub_df = df[mask]
    start_idx = df.index.get_loc(sub_df.index[0])
    days = (sub_df.index[-1] - sub_df.index[0]).days
    sub_periods.append((label, start_idx, len(sub_df), days))

sig = gen_confluence(df, adx_min=20, rsi_long_max=55, rsi_short_min=45, vol_mult=0.35)
print(f'Sinais: {len(sig)}')

top_configs = [
    ('SL2_BUF0.1_TR3 (NEW BEST)', 8.0, 2.0, 3.0, 0.1, 120, 3.0),
    ('SL2_BUF0.2_TR3', 8.0, 2.0, 3.0, 0.2, 120, 3.0),
    ('SL2_BUF0.2_TR4', 8.0, 2.0, 4.0, 0.2, 120, 3.0),
    ('SL2.5_BUF0.2_TR3 (current v15)', 8.0, 2.5, 3.0, 0.2, 120, 3.0),
]

for name, tp, sl, tr, buf, mb, risk in top_configs:
    print(f'\n{"="*70}')
    print(f'CONFIG: {name}')
    print(f'  TP={tp}x SL={sl}x TR={tr}x BUF={buf} MB={mb} RISK={risk}%')
    print(f'{"="*70}')
    all_anns = []
    all_dds = []
    for pn, sidx, slen, days in sub_periods:
        ss = [(i,d,a,p) for i,d,a,p in sig if sidx <= i < sidx+slen]
        trs = run_trades(df, ss, tp_mult=tp, sl_mult=sl, trailing_mult=tr,
                         tp1_pct=0.50, post_tp1_buf=buf, max_bars=mb)
        m = compute_metrics(trs, risk, days)
        all_anns.append(m['ann'])
        all_dds.append(m['dd'])
        print(f'  {pn}: ann={m["ann"]:+7.1f}% dd={m["dd"]:5.1f}% '
              f'wr={m["wr"]:5.1f}% n={m["n"]:3d} pf={m["pf"]:.2f} eq={m["eq"]:.0f}')
    min_a = min(all_anns)
    max_d = max(all_dds)
    ap = all(a > 0 for a in all_anns)
    print(f'  >> MIN_ANNUAL={min_a:+.1f}% MAX_DD={max_d:.1f}% ALL_POS={ap}')

# Save recommendation
rec = {
    "strategy": "confluence_v15_optimized",
    "signal_params": {"adx_min": 20, "rsi_long_max": 55, "rsi_short_min": 45, "vol_mult": 0.35, "confluence_score_min": 5},
    "exit_params": {"tp_atr_mult": 8.0, "sl_atr_mult": 2.0, "trailing_atr_mult": 3.0, "post_tp1_sl_buffer": 0.1, "tp1_pct": 0.50, "max_bars_held": 120, "use_trailing": True},
    "risk_per_trade": 3.0,
    "validation": {"730d_ann": all_anns[0], "365d_ann": all_anns[1], "180d_ann": all_anns[2], "90d_ann": all_anns[3]},
    "min_annual": min_a, "all_subperiods_positive": ap
}
with open('download/confluence_v15_optimized.json', 'w') as f:
    json.dump(rec, f, indent=2)
print(f'\nSalvo: download/confluence_v15_optimized.json')
