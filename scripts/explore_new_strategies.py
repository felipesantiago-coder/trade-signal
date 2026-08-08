r"""
explore_new_strategies.py
------------------------
Exploracao de estrategias FUNDAMENTALMENTE DIFERENTES do BBWP Squeeze.
Testa abordagens: Trend Pullback, EMA Cross, BB Mean Reversion,
RSI Extreme Reversal, MACD Cross, OBV Breakout, e Confluencia Multi-sinal.

Busca: >=50% retorno anual com DD <= 40%, WR >= 35%, >= 30 trades.
"""
import sys, os, json, time, logging, itertools
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)

CACHE = "/home/z/my-project/trade-signal/download/btc_1h_cache.csv"
FEE, SPR, SLP = 0.016, 2.0, 2.0  # maker fee 0.016% + spread 2bps + slip 2bps


def load_data():
    if os.path.exists(CACHE):
        df = pd.read_csv(CACHE, index_col=0, parse_dates=True)
        print(f"Cache: {len(df):,} candles  [{df.index[0]} a {df.index[-1]}]")
        return df
    print("ERRO: Cache nao encontrado. Rode explore_50pct_v2.py primeiro.")
    sys.exit(1)


def _apply_costs_local(entry_price, exit_price, is_long):
    """Custos: fee 0.016% + spread 2bps + slippage 2bps por lado."""
    spread_pct = SPR / 10000.0
    slippage_pct = SLP / 10000.0
    entry_cost = spread_pct + slippage_pct
    if is_long:
        adj_e = entry_price * (1 + entry_cost)
    else:
        adj_e = entry_price * (1 - entry_cost)
    exit_cost = spread_pct + slippage_pct
    if is_long:
        adj_x = exit_price * (1 - exit_cost)
    else:
        adj_x = exit_price * (1 + exit_cost)
    fee_e = adj_e * (FEE / 100.0)
    fee_x = adj_x * (FEE / 100.0)
    if is_long:
        adj_e += fee_e; adj_x -= fee_x
    else:
        adj_e -= fee_e; adj_x += fee_x
    return adj_e, adj_x


def apply_costs(entry, exit_price, is_long):
    """Retorna preco ajustado de saida."""
    _, adj_exit = _apply_costs_local(entry, exit_price, is_long)
    return adj_exit


def run_trades(df, signals, tp_mult=3.0, sl_mult=2.0, trailing_mult=2.0,
               use_trailing=True, tp1_pct=0.50, post_tp1_buf=0.3, max_bars=96):
    """
    Roda backtest a partir de lista de sinais [(idx, direction, atr, price)] and returns
    list of (pnl_pct, sl_dist_pct).
    """
    results = []
    n = len(df)
    last_exit_idx = -999

    for sig_idx, sig_dir, sig_atr, entry_price in signals:
        if sig_idx <= last_exit_idx:
            continue

        isL = (sig_dir == "long")
        atr = sig_atr
        entry = entry_price

        sl = entry - sl_mult * atr if isL else entry + sl_mult * atr
        tp = entry + tp_mult * atr if isL else entry - tp_mult * atr
        sl_d = max(abs(entry - sl) / entry * 100, 0.05)

        csl = sl; tr_on = False; tp1f = False; tp1_price = 0.0
        hwm = entry; exit_done = False; exit_pnl = 0.0

        for j in range(sig_idx + 1, min(sig_idx + max_bars, n)):
            row = df.iloc[j]
            fc, fl, fh = float(row["close"]), float(row["low"]), float(row["high"])
            hwm = max(hwm, fh) if isL else min(hwm, fl)

            hit_sl = (fl <= csl) if isL else (fh >= csl)
            hit_tp = (fh >= tp) if isL else (fl <= tp)

            # TP1 hit
            if hit_tp and not tp1f:
                tp1f = True
                tp1_price = tp
                if use_trailing:
                    tr_on = True
                    buf = atr * post_tp1_buf
                    csl = (tp - buf) if isL else (tp + buf)
                else:
                    adj_tp = apply_costs(entry, tp, isL)
                    pnl = (adj_tp - entry) / entry * 100 if isL else (entry - adj_tp) / entry * 100
                    results.append((pnl, sl_d)); exit_done = True; break

            # Both TP and SL hit same bar after TP1
            if hit_tp and hit_sl and tp1f:
                exit_pnl = calc_partial_pnl(entry, tp1_price, csl, isL, tp1_pct)
                results.append((exit_pnl, sl_d)); exit_done = True; break

            # SL hit
            if hit_sl and not hit_tp:
                if tp1f:
                    exit_pnl = calc_partial_pnl(entry, tp1_price, csl, isL, tp1_pct)
                else:
                    adj_sl = apply_costs(entry, csl, isL)
                    pnl = (adj_sl - entry) / entry * 100 if isL else (entry - adj_sl) / entry * 100
                    exit_pnl = pnl
                results.append((exit_pnl, sl_d)); exit_done = True; break

            # Trailing stop update
            if tr_on and use_trailing:
                td = atr * trailing_mult
                if isL:
                    new_sl = hwm - td
                    if new_sl > csl:
                        csl = new_sl
                else:
                    new_sl = hwm + td
                    if new_sl < csl:
                        csl = new_sl

        # Max bars exit
        if not exit_done:
            last_j = min(sig_idx + max_bars, n) - 1
            xp = float(df.iloc[last_j]["close"])
            if tp1f:
                exit_pnl = calc_partial_pnl(entry, tp1_price, xp, isL, tp1_pct)
            else:
                adj_xp = apply_costs(entry, xp, isL)
                pnl = (adj_xp - entry) / entry * 100 if isL else (entry - adj_xp) / entry * 100
                exit_pnl = pnl
            results.append((exit_pnl, sl_d))

        last_exit_idx = min(sig_idx + max_bars, n) - 1

    return results


def calc_partial_pnl(entry, tp1, exit_p, isL, tp1_pct):
    _, a1 = _apply_costs_local(entry, tp1, isL)
    _, a2 = _apply_costs_local(entry, exit_p, isL)
    if isL:
        return round(tp1_pct * (a1 - entry) / entry * 100 + (1 - tp1_pct) * (a2 - entry) / entry * 100, 4)
    else:
        return round(tp1_pct * (entry - a1) / entry * 100 + (1 - tp1_pct) * (entry - a2) / entry * 100, 4)


def compute_metrics(trades, risk=2.0, days=730):
    if not trades:
        return {"eq": 100, "ret": 0, "ann": 0, "dd": 0, "n": 0, "wr": 0, "pf": 0}
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
    pnls = [p for p, _ in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    r = eq - 100
    return {
        "eq": round(eq, 2), "ret": round(r, 2),
        "ann": round(r * 365 / days, 2), "dd": round(md, 2),
        "n": len(trades),
        "wr": round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
        "pf": round(sum(wins) / max(abs(sum(losses)), 0.001), 2) if pnls else 0,
    }


# =====================================================================
# STRATEGY 1: TREND PULLBACK
# Compra pullbacks em tendencias fortes (ADX>25, DI alinhado)
# Entrada: pullback para EMA20 ou BB middle em tendencia forte
# =====================================================================
def gen_trend_pullback(df):
    signals = []
    n = len(df)
    for i in range(200, n - 1):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        cr = ["ema20", "ema50", "ema200", "rsi", "atr", "adx", "plus_di", "minus_di",
             "bb_lower", "bb_middle", "bb_upper", "volume", "volume_sma20", "stoch_rsi_k"]
        if any(pd.isna(row.get(c)) for c in cr):
            continue

        cl = float(row["close"])
        e20 = float(row["ema20"])
        e50 = float(row["ema50"])
        e200 = float(row["ema200"])
        adx = float(row["adx"])
        pdi = float(row["plus_di"])
        mdi = float(row["minus_di"])
        atr = float(row["atr"])
        rsi = float(row["rsi"])
        srk = float(row["stoch_rsi_k"])
        vol = float(row["volume"])
        vsma = float(row["volume_sma20"])
        bb_mid = float(row["bb_middle"])
        ap = float(row.get("atr_percentile", 0.5))

        if ap < 0.10 or ap > 0.90:
            continue
        if atr <= 0 or cl <= 0:
            continue

        # LONG: Uptrend (ADX strong, +DI > -DI, close > EMA50)
        # Pullback: close <= EMA20 or close near BB middle (within 0.5*ATR)
        # RSI: 30-55 (not overbought, showing dip)
        # Volume: >= 0.5 * SMA20
        if adx > 25 and pdi > mdi and cl > e50 and cl > e200:
            near_ema20 = cl <= e20 * 1.005  # within 0.5% above EMA20
            near_bb_mid = abs(cl - bb_mid) < 0.5 * atr
            touched_ema20 = float(prev["low"]) <= e20 or float(row["low"]) <= e20
            if (near_ema20 or near_bb_mid or touched_ema20) and 25 <= rsi <= 55:
                if vol >= vsma * 0.4:
                    if srk < 60:  # Stoch RSI not overbought
                        signals.append((i, "long", atr, cl))

        # SHORT: Downtrend (ADX strong, -DI > +DI, close < EMA50)
        if adx > 25 and mdi > pdi and cl < e50 and cl < e200:
            near_ema20_s = cl >= e20 * 0.995
            near_bb_mid_s = abs(cl - bb_mid) < 0.5 * atr
            touched_ema20_s = float(prev["high"]) >= e20 or float(row["high"]) >= e20
            if (near_ema20_s or near_bb_mid_s or touched_ema20_s) and 45 <= rsi <= 75:
                if vol >= vsma * 0.4:
                    if srk > 40:  # Stoch RSI not oversold
                        signals.append((i, "short", atr, cl))

    return signals


# =====================================================================
# STRATEGY 2: EMA CROSSOVER MOMENTUM
# EMA20 cruza EMA50 + MACD confirma + Volume confirma
# =====================================================================
def gen_ema_cross(df):
    signals = []
    n = len(df)
    for i in range(200, n - 1):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        cr = ["ema20", "ema50", "ema200", "rsi", "atr", "macd", "macd_signal",
             "macd_hist", "volume", "volume_sma20", "adx", "plus_di", "minus_di"]
        if any(pd.isna(row.get(c)) for c in cr):
            continue

        cl = float(row["close"])
        e20 = float(row["ema20"])
        e50 = float(row["ema50"])
        e200 = float(row["ema200"])
        p_e20 = float(prev["ema20"])
        p_e50 = float(prev["ema50"])
        macd_h = float(row["macd_hist"])
        p_macd_h = float(prev["macd_hist"])
        macd_s = float(row["macd_signal"])
        macd_v = float(row["macd"])
        atr = float(row["atr"])
        vol = float(row["volume"])
        vsma = float(row["volume_sma20"])
        rsi = float(row["rsi"])
        adx = float(row["adx"])
        ap = float(row.get("atr_percentile", 0.5))

        if ap < 0.10 or ap > 0.90:
            continue
        if atr <= 0 or cl <= 0:
            continue

        # LONG: EMA20 crosses above EMA50
        if p_e20 <= p_e50 and e20 > e50:
            if macd_h > 0 and p_macd_h <= 0:  # MACD hist crosses positive
                if vol >= vsma * 0.5:
                    if rsi > 45 and rsi < 75:
                        if cl > e200:  # macro uptrend
                            signals.append((i, "long", atr, cl))

        # SHORT: EMA20 crosses below EMA50
        if p_e20 >= p_e50 and e20 < e50:
            if macd_h < 0 and p_macd_h >= 0:  # MACD hist crosses negative
                if vol >= vsma * 0.5:
                    if rsi < 55 and rsi > 25:
                        if cl < e200:
                            signals.append((i, "short", atr, cl))

    return signals


# =====================================================================
# STRATEGY 3: BB MEAN REVERSION IN TREND
# Preco toca banda BB inferior em uptrend (ou superior em downtrend)
# RSI extremo + EMA filter
# =====================================================================
def gen_bb_mean_reversion(df):
    signals = []
    n = len(df)
    for i in range(200, n - 1):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        cr = ["ema20", "ema50", "ema200", "rsi", "atr", "bb_lower", "bb_upper",
             "bb_middle", "volume", "volume_sma20", "stoch_rsi_k", "adx"]
        if any(pd.isna(row.get(c)) for c in cr):
            continue

        cl = float(row["close"])
        lo = float(row["low"])
        hi = float(row["high"])
        p_lo = float(prev["low"])
        p_hi = float(prev["high"])
        e50 = float(row["ema50"])
        e200 = float(row["ema200"])
        bbl = float(row["bb_lower"])
        bbu = float(row["bb_upper"])
        bbm = float(row["bb_middle"])
        atr = float(row["atr"])
        rsi = float(row["rsi"])
        srk = float(row["stoch_rsi_k"])
        vol = float(row["volume"])
        vsma = float(row["volume_sma20"])
        ap = float(row.get("atr_percentile", 0.5))

        if ap < 0.10 or ap > 0.90:
            continue
        if atr <= 0 or cl <= 0:
            continue

        bb_width = (bbu - bbl) / bbm * 100

        # LONG: Low touches/crosses BB lower in uptrend
        if cl > e50 and cl > e200:
            touched_bbl = lo <= bbl or p_lo <= bbl
            near_bbl = lo <= bbl * 1.005  # within 0.5% of BB lower
            if touched_bbl or near_bbl:
                if rsi < 40:  # oversold-ish
                    if srk < 25:  # Stoch RSI oversold
                        if vol >= vsma * 0.3:
                            signals.append((i, "long", atr, cl))

        # SHORT: High touches/crosses BB upper in downtrend
        if cl < e50 and cl < e200:
            touched_bbu = hi >= bbu or p_hi >= bbu
            near_bbu = hi >= bbu * 0.995
            if touched_bbu or near_bbu:
                if rsi > 60:
                    if srk > 75:
                        if vol >= vsma * 0.3:
                            signals.append((i, "short", atr, cl))

    return signals


# =====================================================================
# STRATEGY 4: RSI EXTREME REVERSAL IN TREND
# RSI atinge extremo (oversold em uptrend / overbought em downtrend)
# Reversal candle confirmation
# =====================================================================
def gen_rsi_extreme(df):
    signals = []
    n = len(df)
    for i in range(200, n - 1):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        cr = ["ema20", "ema50", "ema200", "rsi", "rsi_delta", "atr",
             "volume", "volume_sma20", "stoch_rsi_k", "stoch_rsi_d", "adx"]
        if any(pd.isna(row.get(c)) for c in cr):
            continue

        cl = float(row["close"])
        p_cl = float(prev["close"])
        e50 = float(row["ema50"])
        e200 = float(row["ema200"])
        rsi = float(row["rsi"])
        p_rsi = float(prev["rsi"])
        rsi_d = float(row["rsi_delta"])
        atr = float(row["atr"])
        srk = float(row["stoch_rsi_k"])
        srd = float(row["stoch_rsi_d"])
        vol = float(row["volume"])
        vsma = float(row["volume_sma20"])
        ap = float(row.get("atr_percentile", 0.5))
        adx = float(row["adx"])

        if ap < 0.10 or ap > 0.90:
            continue
        if atr <= 0 or cl <= 0:
            continue

        # LONG: RSI was oversold, now recovering in uptrend
        if cl > e50 and cl > e200 and adx > 15:
            if p_rsi < 30 and rsi >= 30:  # RSI crossed above 30
                if cl > p_cl:  # bullish candle
                    if srk > srd and srk < 50:  # Stoch RSI turning up, not overbought
                        signals.append((i, "long", atr, cl))

        # SHORT: RSI was overbought, now falling in downtrend
        if cl < e50 and cl < e200 and adx > 15:
            if p_rsi > 70 and rsi <= 70:  # RSI crossed below 70
                if cl < p_cl:  # bearish candle
                    if srk < srd and srk > 50:
                        signals.append((i, "short", atr, cl))

    return signals


# =====================================================================
# STRATEGY 5: MACD CROSS + TREND + BB POSITION
# MACD line cruza signal line + tendencia alinhada + preco nao longe das BB
# =====================================================================
def gen_macd_cross(df):
    signals = []
    n = len(df)
    for i in range(200, n - 1):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        cr = ["ema20", "ema50", "ema200", "rsi", "atr", "macd", "macd_signal",
             "macd_hist", "volume", "volume_sma20", "bb_lower", "bb_upper",
             "bb_middle", "adx", "plus_di", "minus_di"]
        if any(pd.isna(row.get(c)) for c in cr):
            continue

        cl = float(row["close"])
        e50 = float(row["ema50"])
        e200 = float(row["ema200"])
        macd = float(row["macd"])
        macd_s = float(row["macd_signal"])
        p_macd = float(prev["macd"])
        p_macd_s = float(prev["macd_signal"])
        atr = float(row["atr"])
        vol = float(row["volume"])
        vsma = float(row["volume_sma20"])
        rsi = float(row["rsi"])
        bbl = float(row["bb_lower"])
        bbu = float(row["bb_upper"])
        bbm = float(row["bb_middle"])
        adx = float(row["adx"])
        pdi = float(row["plus_di"])
        mdi = float(row["minus_di"])
        ap = float(row.get("atr_percentile", 0.5))

        if ap < 0.10 or ap > 0.90:
            continue
        if atr <= 0 or cl <= 0:
            continue

        # LONG: MACD crosses above signal + uptrend
        if p_macd <= p_macd_s and macd > macd_s:
            if cl > e50 and cl > e200:
                if adx > 18:
                    if rsi > 40 and rsi < 70:
                        if vol >= vsma * 0.4:
                            # Not too far from BB middle (within 1 BB width)
                            bb_dist = (cl - bbm) / (bbu - bbl) if (bbu - bbl) > 0 else 0.5
                            if bb_dist < 0.7:  # not already extended
                                signals.append((i, "long", atr, cl))

        # SHORT: MACD crosses below signal + downtrend
        if p_macd >= p_macd_s and macd < macd_s:
            if cl < e50 and cl < e200:
                if adx > 18:
                    if rsi < 60 and rsi > 30:
                        if vol >= vsma * 0.4:
                            bb_dist = (cl - bbm) / (bbu - bbl) if (bbu - bbl) > 0 else 0.5
                            if bb_dist > -0.7:
                                signals.append((i, "short", atr, cl))

    return signals


# =====================================================================
# STRATEGY 6: OBV BREAKOUT + MOMENTUM
# OBV rompe acima da SMA20 + preco confirma com EMA alinhada
# =====================================================================
def gen_obv_breakout(df):
    signals = []
    n = len(df)
    for i in range(200, n - 1):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        cr = ["ema20", "ema50", "ema200", "rsi", "atr", "volume", "volume_sma20",
             "obv", "obv_sma20", "obv_trend", "adx", "stoch_rsi_k"]
        if any(pd.isna(row.get(c)) for c in cr):
            continue

        cl = float(row["close"])
        e50 = float(row["ema50"])
        e200 = float(row["ema200"])
        atr = float(row["atr"])
        rsi = float(row["rsi"])
        srk = float(row["stoch_rsi_k"])
        vol = float(row["volume"])
        vsma = float(row["volume_sma20"])
        obv = float(row["obv"])
        obv_sma = float(row["obv_sma20"])
        obv_t = int(row["obv_trend"])
        p_obv = float(prev["obv"])
        p_obv_sma = float(prev["obv_sma20"])
        adx = float(row["adx"])
        ap = float(row.get("atr_percentile", 0.5))

        if ap < 0.10 or ap > 0.90:
            continue
        if atr <= 0 or cl <= 0:
            continue

        # LONG: OBV crosses above SMA20 + uptrend
        if p_obv <= p_obv_sma and obv > obv_sma:
            if cl > e50 and cl > e200 and adx > 18:
                if rsi > 40 and rsi < 70 and srk < 70:
                    if vol >= vsma * 0.4:
                        signals.append((i, "long", atr, cl))

        # SHORT: OBV crosses below SMA20 + downtrend
        if p_obv >= p_obv_sma and obv < obv_sma:
            if cl < e50 and cl < e200 and adx > 18:
                if rsi < 60 and rsi > 30 and srk > 30:
                    if vol >= vsma * 0.4:
                        signals.append((i, "short", atr, cl))

    return signals


# =====================================================================
# STRATEGY 7: CONFLUENCE - Combina 2+ sinais diferentes
# Requer pelo menos 2 condicoes de entrada simultaneas
# =====================================================================
def gen_confluence(df, adx_min=20, rsi_long_max=55, rsi_short_min=45,
                    vol_mult=0.4, stoch_long_max=65, stoch_short_min=35):
    signals = []
    n = len(df)
    for i in range(200, n - 1):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        cr = ["ema20", "ema50", "ema200", "rsi", "rsi_delta", "atr",
             "macd", "macd_signal", "macd_hist", "volume", "volume_sma20",
             "bb_lower", "bb_upper", "bb_middle", "adx", "plus_di", "minus_di",
             "stoch_rsi_k", "stoch_rsi_d", "obv", "obv_sma20", "obv_trend"]
        if any(pd.isna(row.get(c)) for c in cr):
            continue

        cl = float(row["close"])
        p_cl = float(prev["close"])
        e20 = float(row["ema20"])
        e50 = float(row["ema50"])
        e200 = float(row["ema200"])
        p_e20 = float(prev["ema20"])
        p_e50 = float(prev["ema50"])
        rsi = float(row["rsi"])
        rsi_d = float(row["rsi_delta"])
        atr = float(row["atr"])
        macd = float(row["macd"])
        macd_s = float(row["macd_signal"])
        p_macd = float(prev["macd"])
        p_macd_s = float(prev["macd_signal"])
        macd_h = float(row["macd_hist"])
        vol = float(row["volume"])
        vsma = float(row["volume_sma20"])
        bbl = float(row["bb_lower"])
        bbu = float(row["bb_upper"])
        bbm = float(row["bb_middle"])
        adx = float(row["adx"])
        pdi = float(row["plus_di"])
        mdi = float(row["minus_di"])
        srk = float(row["stoch_rsi_k"])
        srd = float(row["stoch_rsi_d"])
        p_srk = float(prev["stoch_rsi_k"])
        obv = float(row["obv"])
        obv_sma = float(row["obv_sma20"])
        obv_t = int(row["obv_trend"])
        ap = float(row.get("atr_percentile", 0.5))

        if ap < 0.10 or ap > 0.90:
            continue
        if atr <= 0 or cl <= 0:
            continue

        # Count confluence signals for LONG
        long_score = 0
        if cl > e50 and cl > e200:
            long_score += 1  # trend alignment
        if adx > adx_min:
            long_score += 1  # strong trend
        if rsi > 40 and rsi < rsi_long_max:
            long_score += 1  # RSI room to run
        if srk < stoch_long_max and srk > srd:
            long_score += 1  # Stoch RSI momentum
        if macd > macd_s and p_macd <= p_macd_s:
            long_score += 2  # MACD cross (stronger signal)
        elif macd > macd_s and macd_h > 0:
            long_score += 1  # MACD bullish
        if obv > obv_sma and obv_t == 1:
            long_score += 1  # OBV confirms
        if vol >= vsma * vol_mult:
            long_score += 1  # Volume ok
        # Pullback proximity
        near_ema = cl <= e20 * 1.005 or float(prev["low"]) <= e20
        if near_ema:
            long_score += 1  # Pullback entry

        if long_score >= 5 and rsi < rsi_long_max and srk < stoch_long_max:
            signals.append((i, "long", atr, cl))

        # Count confluence signals for SHORT
        short_score = 0
        if cl < e50 and cl < e200:
            short_score += 1
        if adx > adx_min:
            short_score += 1
        if rsi < 60 and rsi > rsi_short_min:
            short_score += 1
        if srk > stoch_short_min and srk < srd:
            short_score += 1
        if macd < macd_s and p_macd >= p_macd_s:
            short_score += 2
        elif macd < macd_s and macd_h < 0:
            short_score += 1
        if obv < obv_sma and obv_t == -1:
            short_score += 1
        if vol >= vsma * vol_mult:
            short_score += 1
        near_ema_s = cl >= e20 * 0.995 or float(prev["high"]) >= e20
        if near_ema_s:
            short_score += 1

        if short_score >= 5 and rsi > rsi_short_min and srk > stoch_short_min:
            signals.append((i, "short", atr, cl))

    return signals


# =====================================================================
# STRATEGY 8: MOMENTUM BREAKOUT (sem squeeze - qualquer BB breakout com momentum)
# =====================================================================
def gen_momentum_breakout(df):
    signals = []
    n = len(df)
    for i in range(200, n - 1):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        cr = ["ema20", "ema50", "ema200", "rsi", "atr", "bb_lower", "bb_upper",
             "volume", "volume_sma20", "adx", "plus_di", "minus_di",
             "stoch_rsi_k", "stoch_rsi_d", "macd_hist"]
        if any(pd.isna(row.get(c)) for c in cr):
            continue

        cl = float(row["close"])
        p_cl = float(prev["close"])
        e50 = float(row["ema50"])
        e200 = float(row["ema200"])
        bbl = float(row["bb_lower"])
        bbu = float(row["bb_upper"])
        atr = float(row["atr"])
        rsi = float(row["rsi"])
        srk = float(row["stoch_rsi_k"])
        srd = float(row["stoch_rsi_d"])
        p_srk = float(prev["stoch_rsi_k"])
        vol = float(row["volume"])
        vsma = float(row["volume_sma20"])
        adx = float(row["adx"])
        pdi = float(row["plus_di"])
        mdi = float(row["minus_di"])
        macd_h = float(row["macd_hist"])
        ap = float(row.get("atr_percentile", 0.5))
        bbwp = float(row.get("bbwp", 100))

        if ap < 0.10 or ap > 0.90:
            continue
        if atr <= 0 or cl <= 0:
            continue

        bb_w = bbu - bbl

        # LONG: Close above BB upper + momentum
        if cl > bbu and p_cl <= bbu:  # Fresh breakout
            if cl > e50:
                if adx > 18:
                    if srk > 50 and srk > srd:  # Stoch RSI momentum up
                        if vol >= vsma * 0.5:
                            if macd_h > 0:
                                signals.append((i, "long", atr, cl))

        # SHORT: Close below BB lower + momentum
        if cl < bbl and p_cl >= bbl:
            if cl < e50:
                if adx > 18:
                    if srk < 50 and srk < srd:
                        if vol >= vsma * 0.5:
                            if macd_h < 0:
                                signals.append((i, "short", atr, cl))

    return signals


# =====================================================================
# MAIN EXPLORATION
# =====================================================================
def main():
    print("=" * 70)
    print("EXPLORACAO NOVAS ESTRATEGIAS — Busca >=50% Retorno Anual")
    print("Abordagens: Trend Pullback, EMA Cross, BB MeanReversion,")
    print("RSI Extreme, MACD Cross, OBV, Confluencia, Momentum Breakout")
    print("=" * 70)

    df = load_data()
    days = (df.index[-1] - df.index[0]).days
    print(f"Periodo: {days} dias")

    # Generate signals for each strategy type
    print("\nGerando sinais...")
    t0 = time.time()

    all_strategies = {}

    # --- Generate base signals ---
    print("  [1/8] Trend Pullback...")
    all_strategies["trend_pullback"] = gen_trend_pullback(df)
    print(f"       {len(all_strategies['trend_pullback'])} sinais")

    print("  [2/8] EMA Crossover...")
    all_strategies["ema_cross"] = gen_ema_cross(df)
    print(f"       {len(all_strategies['ema_cross'])} sinais")

    print("  [3/8] BB Mean Reversion...")
    all_strategies["bb_meanrev"] = gen_bb_mean_reversion(df)
    print(f"       {len(all_strategies['bb_meanrev'])} sinais")

    print("  [4/8] RSI Extreme Reversal...")
    all_strategies["rsi_extreme"] = gen_rsi_extreme(df)
    print(f"       {len(all_strategies['rsi_extreme'])} sinais")

    print("  [5/8] MACD Cross...")
    all_strategies["macd_cross"] = gen_macd_cross(df)
    print(f"       {len(all_strategies['macd_cross'])} sinais")

    print("  [6/8] OBV Breakout...")
    all_strategies["obv_breakout"] = gen_obv_breakout(df)
    print(f"       {len(all_strategies['obv_breakout'])} sinais")

    print("  [7/8] Momentum Breakout...")
    all_strategies["momentum_bo"] = gen_momentum_breakout(df)
    print(f"       {len(all_strategies['momentum_bo'])} sinais")

    # Confluence with different params
    print("  [8/8] Confluencia (multi-param)...")
    confluence_variants = {}
    for adx_m in [15, 20, 25]:
        for rsi_l in [50, 55, 60]:
            key = f"conf_adx{adx_m}_rsi{rsi_l}"
            confluence_variants[key] = gen_confluence(df, adx_min=adx_m, rsi_long_max=rsi_l,
                                                      rsi_short_min=100 - rsi_l)
    print(f"       {len(confluence_variants)} variantes")

    print(f"\nSinais gerados em {time.time() - t0:.1f}s")

    # --- Parameter grids for SL/TP/Trailing ---
    param_grid = [
        # (tp_mult, sl_mult, trailing_mult, use_trailing, tp1_pct, post_tp1_buf, max_bars)
        (3.0, 2.0, 1.5, True, 0.50, 0.3, 72),
        (3.0, 2.0, 2.0, True, 0.50, 0.3, 72),
        (4.0, 2.0, 2.0, True, 0.50, 0.3, 96),
        (4.0, 2.0, 2.5, True, 0.50, 0.3, 96),
        (5.0, 2.0, 2.5, True, 0.50, 0.2, 96),
        (5.0, 2.0, 3.0, True, 0.50, 0.2, 96),
        (6.0, 2.0, 2.5, True, 0.50, 0.2, 96),
        (6.0, 2.0, 3.0, True, 0.50, 0.2, 96),
        (6.0, 2.0, 4.0, True, 0.50, 0.2, 96),
        (4.0, 1.5, 2.0, True, 0.50, 0.3, 96),
        (4.0, 1.5, 2.5, True, 0.50, 0.2, 96),
        (5.0, 1.5, 2.5, True, 0.50, 0.2, 96),
        (5.0, 1.5, 3.0, True, 0.50, 0.2, 96),
        (6.0, 1.5, 2.5, True, 0.50, 0.2, 96),
        (6.0, 1.5, 3.0, True, 0.50, 0.2, 96),
        (8.0, 2.0, 3.0, True, 0.50, 0.2, 120),
        (8.0, 2.0, 4.0, True, 0.50, 0.2, 120),
        (3.0, 2.0, 0, False, 0.50, 0.3, 48),  # No trailing
        (4.0, 2.0, 0, False, 0.50, 0.3, 48),
        (5.0, 2.5, 2.5, True, 0.50, 0.3, 96),
        (5.0, 2.5, 3.0, True, 0.50, 0.2, 96),
        (6.0, 2.5, 3.0, True, 0.50, 0.2, 96),
        (6.0, 2.5, 4.0, True, 0.50, 0.2, 96),
        (8.0, 2.5, 3.0, True, 0.50, 0.2, 120),
        (8.0, 2.5, 4.0, True, 0.50, 0.2, 120),
    ]

    risk_levels = [2.0, 3.0, 4.0, 5.0]

    # --- Run exploration ---
    print(f"\nTestando {len(param_grid)} conjuntos de params x {len(risk_levels)} risks...")
    t0 = time.time()

    qualified = []
    all_results = []
    total_tests = 0

    # Combine all signal generators
    all_signal_sets = []
    for name, sigs in all_strategies.items():
        all_signal_sets.append((name, sigs))
    for name, sigs in confluence_variants.items():
        all_signal_sets.append((name, sigs))

    for strat_name, signals in all_signal_sets:
        if len(signals) < 10:
            continue  # Skip strategies with too few signals

        for pg in param_grid:
            tp_m, sl_m, tr_m, use_tr, tp1, buf, mb = pg
            try:
                trades = run_trades(df, signals, tp_mult=tp_m, sl_mult=sl_m,
                                    trailing_mult=tr_m, use_trailing=use_tr,
                                    tp1_pct=tp1, post_tp1_buf=buf, max_bars=mb)
            except Exception as e:
                continue

            total_tests += 1

            for risk in risk_levels:
                m = compute_metrics(trades, risk, days)

                if m["n"] >= 30 and m["wr"] >= 35 and m["dd"] <= 40:
                    score = m["ann"] / max(m["dd"], 1)
                    entry = {
                        "strat": strat_name,
                        "risk": risk,
                        "tp": tp_m, "sl": sl_m, "tr": tr_m,
                        "trailing": use_tr, "tp1": tp1, "buf": buf, "mb": mb,
                        "ann": m["ann"], "ret": m["ret"], "eq": m["eq"],
                        "dd": m["dd"], "n": m["n"], "wr": m["wr"], "pf": m["pf"],
                        "score": round(score, 1)
                    }
                    all_results.append(entry)
                    if m["ann"] >= 50:
                        qualified.append(entry)

        if total_tests % 500 == 0:
            el = time.time() - t0
            print(f"  {total_tests} tests | {el:.1f}s | {len(qualified)} qualificados")

    print(f"\n{total_tests} tests em {time.time() - t0:.1f}s")
    print(f"Qualificados (>=50% ann, DD<=40, WR>=35, N>=30): {len(qualified)}")

    # Sort by score (ann/dd ratio)
    qualified.sort(key=lambda x: x["score"], reverse=True)
    all_results.sort(key=lambda x: x["score"], reverse=True)

    pool = qualified if qualified else all_results[:50]
    tag = f"{len(qualified)} QUALIFICADOS (>=50% ann)" if qualified else "TOP GERAIS (nenhum >=50%)"

    print(f"\n{'=' * 70}")
    print(f"{tag}")
    print(f"{'=' * 70}")

    for i, r in enumerate(pool[:30]):
        print(f"\n#{i + 1} [{r['strat']}] risk={r['risk']}% score={r['score']}")
        print(f"  ANUAL={r['ann']:+.1f}% TOTAL={r['ret']:+.1f}% EQ={r['eq']} DD={r['dd']:.1f}%")
        print(f"  Trades={r['n']} WR={r['wr']:.1f}% PF={r['pf']:.2f}")
        print(f"  TP={r['tp']}x SL={r['sl']}x TR={r['tr']}x BUF={r['buf']} MB={r['mb']} trail={'ON' if r['trailing'] else 'OFF'}")

    # Save results
    out = {
        "total_tests": total_tests,
        "qualified_count": len(qualified),
        "period_days": days,
        "top": []
    }
    for r in pool[:20]:
        out["top"].append({
            "strat": r["strat"], "risk": r["risk"],
            "ann": r["ann"], "ret": r["ret"], "dd": r["dd"],
            "eq": r["eq"], "n": r["n"], "wr": r["wr"], "pf": r["pf"],
            "score": r["score"],
            "params": {"tp": r["tp"], "sl": r["sl"], "tr": r["tr"],
                       "trailing": r["trailing"], "tp1": r["tp1"],
                       "buf": r["buf"], "mb": r["mb"]}
        })

    outpath = "/home/z/my-project/trade-signal/download/explore_new_strategies.json"
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSalvo: {outpath}")

    # Print strategy summary
    print(f"\n{'=' * 70}")
    print("RESUMO POR ESTRATEGIA (melhor resultado de cada)")
    print(f"{'=' * 70}")
    strat_best = {}
    for r in all_results:
        s = r["strat"].split("_")[0] if "_" in r["strat"] else r["strat"]
        # Group by base strategy name
        if r["strat"].startswith("conf_"):
            base = "confluence"
        else:
            base = r["strat"]
        if base not in strat_best or r["score"] > strat_best[base]["score"]:
            strat_best[base] = r

    for name, r in sorted(strat_best.items(), key=lambda x: x[1]["score"], reverse=True):
        print(f"  {name:20s} | ann={r['ann']:+6.1f}% dd={r['dd']:5.1f}% wr={r['wr']:5.1f}% n={r['n']:3d} score={r['score']:5.1f}")


if __name__ == "__main__":
    main()
