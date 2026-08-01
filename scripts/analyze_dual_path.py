'''
analyze_dual_path.py
------------------
Analise detalhada dos 22 trades do regime-switching v7 para entender:

PATH 1: Por que LONG WR e 41% (7/17)? Quais regimes/condicoes causam perdas?
PATH 2: Por que apenas 5 SHORT trades em 730 dias? Onde estao os gargalos?

Para cada trade, recupera o regime_v2 no momento da entrada,
e analisa RSI, ATR percentile, ADX, EMA slope, volume.
'''
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import ccxt
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from indicators import compute_indicators
from regime_engine import classify_regimes_v2, get_regime_params, REGIME_STRATEGY
from strategy import evaluate_long, evaluate_short, SignalType
from strategy_regime import (
    evaluate_mean_reversion_long, evaluate_mean_reversion_short,
)
from strategy_profiles import get_profile
from backtest import (
    DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS,
    _apply_costs, TradeResult,
)

logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
logger = logging.getLogger("dual_path")
logger.setLevel(logging.INFO)


def _simulate_with_metadata(
    df_ind: pd.DataFrame,
    profile=None,
    fee_pct=DEFAULT_FEE_PCT,
    spread_bps=DEFAULT_SPREAD_BPS,
    slippage_bps=DEFAULT_SLIPPAGE_BPS,
):
    """
    Simula trades com regime-switching e coleta metadados completos
    de cada trade (regime, RSI, ATR, ADX, slope, volume, etc.)
    """
    trades = []
    i = 0
    n = len(df_ind)
    _max_bars = profile.max_bars_held if profile else 72

    while i < n:
        row = df_ind.iloc[i]

        # Check NaN
        critical = [
            "ema20", "ema50", "ema200", "rsi", "atr", "atr_percentile",
            "adx", "plus_di", "minus_di", "regime",
            "regime_v2", "regime_confidence",
        ]
        if any(pd.isna(row.get(c)) for c in critical):
            i += 1
            continue

        regime_v2 = str(row.get("regime_v2", ""))
        confidence = float(row.get("regime_confidence", 0.5))
        params = get_regime_params(regime_v2, confidence, base_profile=profile)
        st = params["strategy_type"]

        if st == "neutral":
            i += 1
            continue

        if confidence < params["min_confidence"]:
            i += 1
            continue

        signal = None
        atr_pct = float(row.get("atr_percentile", 0.5))

        if st == "trend_follow":
            if atr_pct < 0.10 or atr_pct > 0.90:
                i += 1
                continue
            signal = evaluate_long(row, profile=profile)
            if signal is None:
                signal = evaluate_short(row, profile=profile)

        elif st == "mean_reversion":
            if atr_pct < 0.10 or atr_pct > 0.85:
                i += 1
                continue
            if params["allow_long"]:
                signal = evaluate_mean_reversion_long(row, params, base_profile=profile)
            if signal is None and params["allow_short"]:
                signal = evaluate_mean_reversion_short(row, params, base_profile=profile)

        if signal is None:
            i += 1
            continue

        # Simulate trade
        entry_price = signal.entry_price
        sl = signal.stop_loss
        tp = signal.take_profit
        atr = signal.atr
        is_long = signal.type == SignalType.LONG
        exit_price = None
        exit_reason = None
        bars = 0

        for j in range(i + 1, min(i + _max_bars, n)):
            future = df_ind.iloc[j]
            bars = j - i
            if is_long:
                if float(future["low"]) <= sl:
                    exit_price = sl
                    exit_reason = "sl"
                    break
                if float(future["high"]) >= tp:
                    exit_price = tp
                    exit_reason = "tp"
                    break
            else:
                if float(future["high"]) >= sl:
                    exit_price = sl
                    exit_reason = "sl"
                    break
                if float(future["low"]) <= tp:
                    exit_price = tp
                    exit_reason = "tp"
                    break

        if exit_price is None:
            last_j = min(i + _max_bars, n) - 1
            exit_price = float(df_ind.iloc[last_j]["close"])
            exit_reason = "timeout"
            bars = last_j - i

        # Apply costs
        _, adj_exit, cost_pct = _apply_costs(
            entry_price, exit_price, is_long,
            fee_pct, spread_bps, slippage_bps,
        )
        if is_long:
            pnl_pct = (adj_exit - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - adj_exit) / entry_price * 100

        # Collect metadata
        trade_meta = {
            "entry_ts": str(row.name),
            "type": "LONG" if is_long else "SHORT",
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "pnl_pct": round(pnl_pct, 4),
            "exit_reason": exit_reason,
            "bars_held": bars,
            # Regime metadata
            "regime_v2": regime_v2,
            "regime_strategy": st,
            "regime_confidence": round(confidence, 3),
            # Entry conditions
            "rsi": round(float(row.get("rsi", 0)), 1),
            "atr_pct": round(atr_pct, 3),
            "adx": round(float(row.get("adx", 0)), 1),
            "plus_di": round(float(row.get("plus_di", 0)), 1),
            "minus_di": round(float(row.get("minus_di", 0)), 1),
            "di_spread": round(float(row.get("plus_di", 0)) - float(row.get("minus_di", 0)), 1),
            "ema50_slope": round(float(row.get("ema50_slope", 0)), 3),
            "volume_ratio": round(float(row.get("volume", 0)) / max(float(row.get("volume_sma20", 1)), 1), 2),
            "bb_width": round(float(row.get("bb_width", 0)), 4),
            "bb_squeeze_pct": round(float(row.get("bb_squeeze_pct", 0)), 3),
            # Price context
            "close_vs_ema50_pct": round((float(row["close"]) - float(row["ema50"])) / float(row["close"]) * 100, 2),
            "close_vs_ema200_pct": round((float(row["close"]) - float(row["ema200"])) / float(row["close"]) * 100, 2),
            "pullback_type": getattr(signal, "pullback_type", ""),
            # R:R
            "risk": round(abs(entry_price - sl), 2),
            "reward": round(abs(tp - entry_price), 2),
            "rr_ratio": round(abs(tp - entry_price) / max(abs(entry_price - sl), 0.01), 2),
        }
        trades.append(trade_meta)

        i += bars + 1

    return trades


def analyze_long_wr(trades):
    """
    PATH 1: Analise detalhada dos LONG trades para entender porque WR = 41%
    """
    longs = [t for t in trades if t["type"] == "LONG"]
    print("\n" + "="*80)
    print("PATH 1: ANALISE LONG TRADES (WR otimizacao)")
    print("="*80)

    # 1. Por regime
    print("\n--- 1.1 LONG Trades por Regime ---")
    regime_stats = {}
    for t in longs:
        r = t["regime_v2"]
        if r not in regime_stats:
            regime_stats[r] = {"total": 0, "wins": 0, "pnl": 0, "pnl_wins": 0, "pnl_losses": 0}
        regime_stats[r]["total"] += 1
        regime_stats[r]["pnl"] += t["pnl_pct"]
        if t["pnl_pct"] > 0:
            regime_stats[r]["wins"] += 1
            regime_stats[r]["pnl_wins"] += t["pnl_pct"]
        else:
            regime_stats[r]["pnl_losses"] += t["pnl_pct"]

    for r, s in sorted(regime_stats.items(), key=lambda x: -x[1]["total"]):
        wr = 100 * s["wins"] / max(s["total"], 1)
        avg = s["pnl"] / max(s["total"], 1)
        print(f"  {r:20s}: {s['total']:2d} trades, WR={wr:5.1f}%, PnL={s['pnl']:+7.2f}%, avg={avg:+.2f}%")
        if s["pnl_wins"]:
            print(f"    wins PnL: {s['pnl_wins']:+.2f}% avg={s['pnl_wins']/s['wins']:+.2f}%")
        if s["total"] - s["wins"]:
            losses = s["total"] - s["wins"]
            print(f"    losses PnL: {s['pnl_losses']:+.2f}% avg={s['pnl_losses']/losses:+.2f}%")

    # 2. Por exit reason
    print("\n--- 1.2 LONG Trades por Exit Reason ---")
    reason_stats = {}
    for t in longs:
        r = t["exit_reason"]
        if r not in reason_stats:
            reason_stats[r] = {"total": 0, "wins": 0, "pnl": 0}
        reason_stats[r]["total"] += 1
        reason_stats[r]["pnl"] += t["pnl_pct"]
        if t["pnl_pct"] > 0:
            reason_stats[r]["wins"] += 1
    for r, s in sorted(reason_stats.items()):
        wr = 100 * s["wins"] / max(s["total"], 1)
        print(f"  {r:10s}: {s['total']:2d} trades, WR={wr:5.1f}%, PnL={s['pnl']:+7.2f}%")

    # 3. Condições dos losers vs winners
    print("\n--- 1.3 LONG Winners vs Losers: Condicoes de Entrada ---")
    winners = [t for t in longs if t["pnl_pct"] > 0]
    losers = [t for t in longs if t["pnl_pct"] <= 0]
    print(f"  Winners: {len(winners)}, Losers: {len(losers)}")

    metrics = ["rsi", "atr_pct", "adx", "di_spread", "ema50_slope",
               "volume_ratio", "bb_squeeze_pct", "close_vs_ema50_pct", "rr_ratio"]
    print(f"  {'Metric':25s} {'Winners_avg':>12s} {'Losers_avg':>12s} {'Diff':>8s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*8}")
    for m in metrics:
        w_avg = np.mean([t[m] for t in winners]) if winners else 0
        l_avg = np.mean([t[m] for t in losers]) if losers else 0
        diff = w_avg - l_avg
        marker = " ***" if abs(diff) > 0.5 * max(abs(w_avg), abs(l_avg), 0.01) else ""
        print(f"  {m:25s} {w_avg:12.3f} {l_avg:12.3f} {diff:+8.3f}{marker}")

    # 4. RSI distribution of losers
    print("\n--- 1.4 LONG Losers: RSI Distribution ---")
    for t in losers:
        print(f"  {t['entry_ts'][:16]} RSI={t['rsi']:5.1f} ADX={t['adx']:5.1f} ATR%={t['atr_pct']:.3f} "
              f"regime={t['regime_v2']:20s} slope={t['ema50_slope']:.3f} "
              f"PnL={t['pnl_pct']:+.2f}% exit={t['exit_reason']} pullback={t['pullback_type']}")

    # 5. Perda por regime - quais regimes estao destruindo valor?
    print("\n--- 1.5 LONG: Regimes com Maior Destruição de Valor ---")
    for t in sorted(losers, key=lambda x: x["pnl_pct"]):
        print(f"  PnL={t['pnl_pct']:+6.2f}% | regime={t['regime_v2']:20s} | RSI={t['rsi']:5.1f} | "
              f"ADX={t['adx']:5.1f} | slope={t['ema50_slope']:+.3f} | entry={t['entry_ts'][:16]}")

    return regime_stats


def analyze_short_frequency(df_ind, trades):
    """
    PATH 2: Analise porque apenas 5 SHORT trades em 730 dias.
    """
    shorts = [t for t in trades if t["type"] == "SHORT"]
    print("\n" + "="*80)
    print("PATH 2: ANALISE SHORT TRADES (frequencia)")
    print("="*80)

    # 1. Regime distribution
    print("\n--- 2.1 Regime Distribution no Dataset (730d 1h) ---")
    total = len(df_ind)
    dist = df_ind["regime_v2"].value_counts()
    for regime, count in dist.items():
        pct = 100 * count / total
        strat = REGIME_STRATEGY.get(regime, "?")
        allows_short = "YES" if regime in {"STRONG_DOWNTREND", "WEAK_DOWNTREND", "RANGING"} else "no"
        if allows_short == "YES":
            if regime in ("STRONG_DOWNTREND", "WEAK_DOWNTREND"):
                min_conf = 0.4 if "WEAK" in regime else 0.3
            else:  # RANGING
                min_conf = 0.2
            mask_conf = (df_ind["regime_v2"] == regime) & (df_ind["regime_confidence"] >= min_conf)
            conf_count = mask_conf.sum()
            print(f"  {regime:20s}: {count:6,} ({pct:5.1f}%) -> {strat:20s} SHORT={allows_short} (conf>={min_conf}: {conf_count:,})")

    # 2. Downtrend regime analysis
    print("\n--- 2.2 DOWNTREND Regimes: Quao frequentes e quao fortes? ---")
    for regime in ["STRONG_DOWNTREND", "WEAK_DOWNTREND"]:
        mask = df_ind["regime_v2"] == regime
        count = mask.sum()
        if count == 0:
            print(f"  {regime}: NUNCA detectado!")
            continue

        subset = df_ind[mask]
        avg_adx = subset["adx"].mean()
        avg_di_spread = (subset["plus_di"] - subset["minus_di"]).abs().mean()
        avg_slope = subset["ema50_slope"].mean()
        avg_conf = subset["regime_confidence"].mean()

        # Check how often EMA alignment (close < ema50 < ema200) is satisfied
        ema_align = ((subset["close"] < subset["ema50"]) & (subset["ema50"] < subset["ema200"])).sum()
        # Check ADX >= 30
        adx_ok = (subset["adx"] >= 30).sum()

        print(f"  {regime}: {count} bars ({100*count/total:.1f}%)")
        print(f"    avg ADX={avg_adx:.1f}, avg |DI spread|={avg_di_spread:.1f}, avg slope={avg_slope:.3f}")
        print(f"    avg confidence={avg_conf:.3f}")
        print(f"    EMA alignment (c<50<200): {ema_align}/{count} ({100*ema_align/count:.1f}%)")
        print(f"    ADX >= 30: {adx_ok}/{count} ({100*adx_ok/count:.1f}%)")

        # Check RSI range for shorts in these regimes
        valid_rsi = (subset["rsi"] >= 55) & (subset["rsi"] <= 75)
        print(f"    RSI in [55,75]: {valid_rsi.sum()}/{count} ({100*valid_rsi.sum()/count:.1f}%)")

        # Check pullback conditions
        # For shorts: ema20_touched_up or ema50_touched_up
        ema20_touch_up = subset.get("ema20_touched_up", pd.Series(False, index=subset.index))
        if "ema20_touched_up" not in subset.columns:
            ema20_touch_up = pd.Series(False, index=subset.index)
        ema50_touch_up = subset.get("ema50_touched_up", pd.Series(False, index=subset.index))
        if "ema50_touched_up" not in subset.columns:
            ema50_touch_up = pd.Series(False, index=subset.index)
        any_pullback = ema20_touch_up | ema50_touch_up
        print(f"    Pullback touch (EMA20/50 from below): {any_pullback.sum()}/{count} ({100*any_pullback.sum()/count:.1f}%)")

    # 3. RANGING regime shorts
    print("\n--- 2.3 RANGING Regime: Potencial para Shorts ---")
    mask_r = df_ind["regime_v2"] == "RANGING"
    ranging = df_ind[mask_r]
    print(f"  Total RANGING bars: {len(ranging)} ({100*len(ranging)/total:.1f}%)")

    # In RANGING, shorts need: RSI [58, 80], close > bb_middle, close >= bb_upper
    # (from evaluate_mean_reversion_short)
    if len(ranging) > 0:
        rsi_short_ok = (ranging["rsi"] >= 58) & (ranging["rsi"] <= 80)
        close_upper = ranging["close"] >= ranging["bb_upper"]
        macd_ok = (ranging["macd_hist"] > 0) | (ranging["macd"] <= ranging["macd_signal"])

        both = rsi_short_ok & close_upper
        all_three = both & macd_ok

        print(f"  RSI in [58,80]: {rsi_short_ok.sum()}/{len(ranging)} ({100*rsi_short_ok.sum()/len(ranging):.1f}%)")
        print(f"  Close >= BB upper: {close_upper.sum()}/{len(ranging)} ({100*close_upper.sum()/len(ranging):.1f}%)")
        print(f"  RSI + BB upper: {both.sum()}/{len(ranging)} ({100*both.sum()/len(ranging):.1f}%)")
        print(f"  RSI + BB + MACD: {all_three.sum()}/{len(ranging)} ({100*all_three.sum()/len(ranging):.1f}%)")

        # Volume filter
        vol_ok = ranging["volume"] >= ranging["volume_sma20"] * 1.0
        print(f"  + Volume >= SMA20: {(both & vol_ok).sum()}/{len(ranging)} ({100*(both & vol_ok).sum()/len(ranging):.1f}%)")

    # 4. Short trade details
    print(f"\n--- 2.4 Todos os SHORT Trades ({len(shorts)}) ---")
    for t in shorts:
        print(f"  {t['entry_ts'][:16]} regime={t['regime_v2']:20s} RSI={t['rsi']:5.1f} ADX={t['adx']:5.1f} "
              f"PnL={t['pnl_pct']:+.2f}% exit={t['exit_reason']} RR={t['rr_ratio']:.1f}:1")

    # 5. Funnel analysis: where are shorts being filtered out?
    print("\n--- 2.5 Funnel Analysis: Por que tao poucos SHORTS? ---")
    print("  Filtros sequenciais para SHORT:")
    print("  1. Regime deve ser DOWNTREND ou RANGING")
    down_mask = df_ind["regime_v2"].isin({"STRONG_DOWNTREND", "WEAK_DOWNTREND"})
    print(f"     DOWNTREND bars: {down_mask.sum():,}/{total:,} ({100*down_mask.sum()/total:.1f}%)")
    print(f"     RANGING bars:   {mask_r.sum():,}/{total:,} ({100*mask_r.sum()/total:.1f}%)")

    # In downtrend: EMA alignment
    if down_mask.sum() > 0:
        dt = df_ind[down_mask]
        ema_ok = (dt["close"] < dt["ema50"]) & (dt["ema50"] < dt["ema200"])
        print(f"  2. EMA alignment (c<50<200): {ema_ok.sum():,}/{down_mask.sum():,}")

        # Slope
        slope_ok = dt["ema50_slope"] < 0
        print(f"  3. EMA50 slope < 0: {slope_ok.sum():,}/{down_mask.sum():,}")

        # ADX
        adx_ok = dt["adx"] >= 30
        print(f"  4. ADX >= 30: {adx_ok.sum():,}/{down_mask.sum():,}")

        # RSI
        rsi_ok = (dt["rsi"] >= 55) & (dt["rsi"] <= 75)
        print(f"  5. RSI in [55,75]: {rsi_ok.sum():,}/{down_mask.sum():,}")

        # All combined
        all_ok = ema_ok & slope_ok & adx_ok & rsi_ok
        print(f"  6. ALL filters: {all_ok.sum():,}/{down_mask.sum():,}")

        # Pullback
        if "ema20_touched" in dt.columns:
            pb = dt["ema20_touched"] | dt.get("ema50_touched_up", pd.Series(False, index=dt.index))
            print(f"  7. + Pullback: {(all_ok & pb).sum():,}/{down_mask.sum():,}")

        # ATR percentile
        atr_ok = (dt["atr_percentile"] >= 0.10) & (dt["atr_percentile"] <= 0.90)
        print(f"  8. + ATR [0.1, 0.9]: {(all_ok & atr_ok).sum():,}/{down_mask.sum():,}")

    # Summary
    print("\n--- 2.6 Diagnostico Principal ---")
    if down_mask.sum() < 500:
        print(f"  >> GARGALO PRINCIPAL: DOWNTREND regimes cobrem apenas {100*down_mask.sum()/total:.1f}% do tempo")
        print(f"     BTC em 730d esteve majoritariamente em uptrend/ranging.")
        print(f"     Solucao: Ativar BREAKOUT_BEAR para capturar reversoes abruptas,")
        print(f"     ou relaxar condicoes de WEAK_DOWNTREND (adx_min reduzido).")
    elif shorts and all(t["regime_v2"] == "RANGING" for t in shorts):
        print(f"  >> Todos os SHORTs vieram de RANGING (mean-reversion).")
        print(f"     DOWNTREND nao gera trades — EMA alignment muito raro ou pullback raro.")
    else:
        print(f"  >> Analisar filtros individualmente acima.")


def main():
    logger.info("Baixando dados BTC/USDT 1h 730d...")
    exchange = ccxt.binance({"enableRateLimit": True})
    symbol = "BTC/USDT"
    timeframe = "1h"
    since_ms = int((datetime.now(timezone.utc).timestamp() - 730 * 86400) * 1000)

    # Paginate download (1000 per batch)
    all_ohlcv = []
    last_ts = 0
    max_iter = 730 * 24 + 100
    iteration = 0
    tf_ms = 3600 * 1000  # 1h in ms

    while iteration < max_iter:
        iteration += 1
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=1000)
        if not batch:
            break
        batch_ts = batch[-1][0]
        if batch_ts <= last_ts:
            break
        last_ts = batch_ts
        all_ohlcv.extend(batch)
        since_ms = batch_ts + tf_ms
        if len(all_ohlcv) % 5000 < 1000:
            logger.info(f"  Downloaded {len(all_ohlcv):,} candles...")

    df = pd.DataFrame(all_ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts")
    df = df.astype(float)
    df = df[~df.index.duplicated(keep="first")]

    logger.info(f"Dados: {len(df)} candles")

    # Compute indicators
    logger.info("Computando indicadores...")
    df_ind = compute_indicators(df, timeframe)

    # Classify regimes
    logger.info("Classificando regimes...")
    df_ind = classify_regimes_v2(df_ind, hysteresis_bars=3)

    # Get profile
    profile = get_profile("STANDARD")

    # Simulate with metadata
    logger.info("Simulando trades com metadados...")
    trades = _simulate_with_metadata(df_ind, profile=profile)

    print(f"\nTotal trades: {len(trades)}")
    longs = [t for t in trades if t["type"] == "LONG"]
    shorts = [t for t in trades if t["type"] == "SHORT"]
    print(f"  LONG: {len(longs)} (WR={100*sum(1 for t in longs if t['pnl_pct']>0)/max(len(longs),1):.1f}%)")
    print(f"  SHORT: {len(shorts)} (WR={100*sum(1 for t in shorts if t['pnl_pct']>0)/max(len(shorts),1):.1f}%)")

    # Path 1: Long WR analysis
    long_regime_stats = analyze_long_wr(trades)

    # Path 2: Short frequency analysis
    analyze_short_frequency(df_ind, trades)

    # Save results to JSON
    results = {
        "total_trades": len(trades),
        "longs": len(longs),
        "shorts": len(shorts),
        "long_wr": round(100 * sum(1 for t in longs if t["pnl_pct"] > 0) / max(len(longs), 1), 1),
        "short_wr": round(100 * sum(1 for t in shorts if t["pnl_pct"] > 0) / max(len(shorts), 1), 1),
        "trades": trades,
    }
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "download", "dual_path_analysis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResultados salvos em: {out_path}")


if __name__ == "__main__":
    main()
