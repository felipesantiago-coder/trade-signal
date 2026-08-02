# -*- coding: utf-8 -*-
"""analyze_weaknesses.py - Diagnostico detalhado dos pontos fracos de ambas estrategias"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from backtest import (
    run_backtest, fetch_historical_ohlcv,
    _apply_costs, TradeResult, SignalType,
)
from indicators import compute_indicators
from strategy_profiles import get_profile
from strategy_ema_cross import evaluate_ema_cross_row, reset_cooldown, EMA_CROSS_PARAMS
from regime_engine import classify_regimes_v2, get_regime_params
from strategy_regime import evaluate_mean_reversion_long, evaluate_mean_reversion_short
from strategy import evaluate_long, evaluate_short


def analyze_15m_trades():
    """Analise detalhada dos trades 15m para encontrar padroes de perda."""
    print("\n" + "#" * 80)
    print("#  ANALISE DETALHADA: 15m EMA Cross v8")
    print("#" * 80)

    profile = get_profile("15m")
    df = fetch_historical_ohlcv("BTC/USDT", "15m", 365)
    df = compute_indicators(df, timeframe="15m")
    crit = ['ema20', 'ema50', 'ema200', 'rsi', 'rsi_delta', 'atr', 'atr_percentile',
            'adx', 'plus_di', 'minus_di', 'regime',
            'bb_lower', 'bb_upper', 'bb_middle', 'bb_width', 'bb_squeeze_pct',
            'macd', 'macd_signal', 'macd_hist', 'volume_sma20', 'volume_sma50', 'ema50_slope']
    df = df.dropna(subset=crit).copy()

    reset_cooldown()
    fee, spread, slip = 0.016, 2.0, 2.0

    c = df['close'].values
    hi = df['high'].values
    lo = df['low'].values
    atr = df['atr'].values
    adx = df['adx'].values
    rsi = df['rsi'].values
    rsi_delta = df['rsi_delta'].values
    bb_width = df['bb_width'].values
    bb_squeeze = df['bb_squeeze_pct'].values
    vol = df['volume'].values
    vol_sma20 = df['volume_sma20'].values
    atr_pct = df['atr_percentile'].values
    regime = df['regime'].values
    ema50_slope = df['ema50_slope'].values
    macd_hist = df['macd_hist'].values

    # Simular todos os trades com detalhes
    trades_detail = []
    n = len(df)
    i = 0
    last_signal_bar = -999
    cooldown = EMA_CROSS_PARAMS['cooldown']

    while i < n:
        if i < 1 or (i - last_signal_bar) < cooldown:
            i += 1; continue

        row = df.iloc[i]
        prev = df.iloc[i-1]
        signal = evaluate_ema_cross_row(row, prev, i, profile=profile)
        if signal is None:
            i += 1; continue

        last_signal_bar = i
        entry = signal.entry_price
        sl = signal.stop_loss
        tp = signal.take_profit
        is_long = signal.type == SignalType.LONG

        # Simular
        exit_p = None; reason = None; bars = 0
        for j in range(i+1, min(i+48, n)):
            if is_long:
                if lo[j] <= sl: exit_p = sl; reason = "sl"; bars = j-i; break
                if hi[j] >= tp: exit_p = tp; reason = "tp"; bars = j-i; break
            else:
                if hi[j] >= sl: exit_p = sl; reason = "sl"; bars = j-i; break
                if lo[j] <= tp: exit_p = tp; reason = "tp"; bars = j-i; break
        if exit_p is None:
            exit_p = c[min(i+48, n)-1]; reason = "timeout"; bars = min(i+48, n)-1 - i

        _, adj_exit, _ = _apply_costs(entry, exit_p, is_long, fee, spread, slip)
        pnl = ((adj_exit - entry) / entry * 100) if is_long else ((entry - adj_exit) / entry * 100)

        trades_detail.append({
            'idx': i, 'type': 'LONG' if is_long else 'SHORT',
            'entry': entry, 'exit': adj_exit, 'pnl': pnl,
            'reason': reason, 'bars': bars,
            'adx': adx[i], 'rsi': rsi[i], 'rsi_delta': rsi_delta[i],
            'atr_pct': atr_pct[i], 'bb_width': bb_width[i],
            'bb_squeeze': bb_squeeze[i],
            'vol_ratio': vol[i] / vol_sma20[i] if vol_sma20[i] > 0 else 1.0,
            'regime': str(regime[i]),
            'ema50_slope': ema50_slope[i],
            'macd_hist': macd_hist[i],
        })
        i += bars + 1

    td = pd.DataFrame(trades_detail)
    if td.empty:
        print("  Nenhum trade encontrado!")
        return

    print(f"\n  Total de trades: {len(td)}")
    print(f"  Longs: {len(td[td.type=='LONG'])} | Shorts: {len(td[td.type=='SHORT'])}")

    # 1. Analise por resultado
    wins = td[td.pnl > 0]
    losses = td[td.pnl <= 0]
    print(f"\n  WINS ({len(wins)}):  avg={wins['pnl'].mean():+.3f}%  median={wins['pnl'].median():+.3f}%")
    print(f"  LOSSES ({len(losses)}): avg={losses['pnl'].mean():+.3f}%  median={losses['pnl'].median():+.3f}%")
    print(f"  Avg win / Avg loss ratio: {abs(wins['pnl'].mean() / losses['pnl'].mean()):.2f}")

    # 2. Analise por razao de saida
    print(f"\n  --- SAIDA POR RAZAO ---")
    for reason in ['tp', 'sl', 'timeout']:
        sub = td[td.reason == reason]
        if len(sub) > 0:
            wr = len(sub[sub.pnl > 0]) / len(sub) * 100
            print(f"  {reason:>8}: {len(sub):>3} trades ({wr:.0f}% win) avg_pnl={sub['pnl'].mean():+.3f}%")

    # 3. Analise por direcao
    print(f"\n  --- POR DIRECAO ---")
    for direction in ['LONG', 'SHORT']:
        sub = td[td.type == direction]
        if len(sub) > 0:
            wr = len(sub[sub.pnl > 0]) / len(sub) * 100
            pf_w = sub[sub.pnl > 0]['pnl'].sum()
            pf_l = abs(sub[sub.pnl <= 0]['pnl'].sum())
            pf = pf_w / pf_l if pf_l > 0 else 999
            print(f"  {direction:>5}: {len(sub):>3} trades WR={wr:.0f}% PF={pf:.2f} PnL={sub['pnl'].sum():+.2f}%")

    # 4. Analise por ADX no momento da entrada
    print(f"\n  --- FILTRO ADX (momento da entrada) ---")
    for adx_min in [0, 10, 15, 20, 25, 30]:
        if adx_min == 0:
            sub = td
        else:
            sub = td[td.adx >= adx_min]
        if len(sub) >= 5:
            wr = len(sub[sub.pnl > 0]) / len(sub) * 100
            pf_w = sub[sub.pnl > 0]['pnl'].sum()
            pf_l = abs(sub[sub.pnl <= 0]['pnl'].sum())
            pf = pf_w / pf_l if pf_l > 0 else 999
            print(f"  ADX>={adx_min:>2}: {len(sub):>3} trades WR={wr:.0f}% PF={pf:.2f} PnL={sub['pnl'].sum():+.2f}%")

    # 5. Analise por Volume ratio
    print(f"\n  --- FILTRO VOLUME RATIO ---")
    for vol_min in [0.0, 0.5, 0.8, 1.0, 1.2, 1.5]:
        if vol_min == 0.0:
            sub = td
        else:
            sub = td[td.vol_ratio >= vol_min]
        if len(sub) >= 5:
            wr = len(sub[sub.pnl > 0]) / len(sub) * 100
            pf_w = sub[sub.pnl > 0]['pnl'].sum()
            pf_l = abs(sub[sub.pnl <= 0]['pnl'].sum())
            pf = pf_w / pf_l if pf_l > 0 else 999
            print(f"  Vol>={vol_min:.1f}x: {len(sub):>3} trades WR={wr:.0f}% PF={pf:.2f} PnL={sub['pnl'].sum():+.2f}%")

    # 6. Analise por BB width (volatilidade)
    print(f"\n  --- FILTRO BB WIDTH (percentil) ---")
    for bb_min in [0.0, 0.2, 0.3, 0.4, 0.5]:
        if bb_min == 0.0:
            sub = td
        else:
            sub = td[td.bb_squeeze >= bb_min]
        if len(sub) >= 5:
            wr = len(sub[sub.pnl > 0]) / len(sub) * 100
            pf_w = sub[sub.pnl > 0]['pnl'].sum()
            pf_l = abs(sub[sub.pnl <= 0]['pnl'].sum())
            pf = pf_w / pf_l if pf_l > 0 else 999
            print(f"  BB_sq>={bb_min:.1f}: {len(sub):>3} trades WR={wr:.0f}% PF={pf:.2f} PnL={sub['pnl'].sum():+.2f}%")

    # 7. Analise por ATR percentile
    print(f"\n  --- FILTRO ATR PERCENTILE ---")
    for atr_lo, atr_hi in [(0, 1), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.8)]:
        sub = td[(td.atr_pct >= atr_lo) & (td.atr_pct <= atr_hi)]
        if len(sub) >= 5:
            wr = len(sub[sub.pnl > 0]) / len(sub) * 100
            pf_w = sub[sub.pnl > 0]['pnl'].sum()
            pf_l = abs(sub[sub.pnl <= 0]['pnl'].sum())
            pf = pf_w / pf_l if pf_l > 0 else 999
            print(f"  ATR [{atr_lo:.1f}-{atr_hi:.1f}]: {len(sub):>3} trades WR={wr:.0f}% PF={pf:.2f} PnL={sub['pnl'].sum():+.2f}%")

    # 8. Analise por RSI delta (momentum strength)
    print(f"\n  --- FILTRO RSI DELTA (forca do momentum) ---")
    for rd_min in [0, 0.5, 1.0, 2.0]:
        if rd_min == 0:
            sub = td
        else:
            sub = td[td.rsi_delta.abs() >= rd_min]
        if len(sub) >= 5:
            wr = len(sub[sub.pnl > 0]) / len(sub) * 100
            pf_w = sub[sub.pnl > 0]['pnl'].sum()
            pf_l = abs(sub[sub.pnl <= 0]['pnl'].sum())
            pf = pf_w / pf_l if pf_l > 0 else 999
            print(f"  |RSI_delta|>={rd_min:.1f}: {len(sub):>3} trades WR={wr:.0f}% PF={pf:.2f} PnL={sub['pnl'].sum():+.2f}%")

    # 9. Analise por regime
    print(f"\n  --- POR REGIME ---")
    for reg in td['regime'].unique():
        sub = td[td.regime == reg]
        if len(sub) >= 3:
            wr = len(sub[sub.pnl > 0]) / len(sub) * 100
            pf_w = sub[sub.pnl > 0]['pnl'].sum()
            pf_l = abs(sub[sub.pnl <= 0]['pnl'].sum())
            pf = pf_w / pf_l if pf_l > 0 else 999
            print(f"  {reg:>20}: {len(sub):>3} trades WR={wr:.0f}% PF={pf:.2f} PnL={sub['pnl'].sum():+.2f}%")

    # 10. Combinacao dos melhores filtros
    print(f"\n  --- MELHORES COMBINACOES ---")
    best_pnl = -999
    best_combo = ""
    for adx_m in [0, 15, 20]:
        for vol_m in [0.0, 0.8, 1.0]:
            for rd_m in [0, 0.5, 1.0]:
                for bb_m in [0.0, 0.2]:
                    sub = td.copy()
                    if adx_m > 0: sub = sub[sub.adx >= adx_m]
                    if vol_m > 0: sub = sub[sub.vol_ratio >= vol_m]
                    if rd_m > 0: sub = sub[sub.rsi_delta.abs() >= rd_m]
                    if bb_m > 0: sub = sub[sub.bb_squeeze >= bb_m]
                    if len(sub) >= 15:
                        wr = len(sub[sub.pnl > 0]) / len(sub) * 100
                        pf_w = sub[sub.pnl > 0]['pnl'].sum()
                        pf_l = abs(sub[sub.pnl <= 0]['pnl'].sum())
                        pf = pf_w / pf_l if pf_l > 0 else 999
                        total_pnl = sub['pnl'].sum()
                        if total_pnl > best_pnl and pf > 1.0:
                            best_pnl = total_pnl
                            best_combo = f"ADX>={adx_m} Vol>={vol_m} |RSI_d|>={rd_m} BBsq>={bb_m}"
                            print(f"  ADX>={adx_m} Vol>={vol_m} |RSId|>={rd_m} BBsq>={bb_m}: "
                                  f"{len(sub):>3}T WR={wr:.0f}% PF={pf:.2f} PnL={total_pnl:+.2f}%")

    print(f"\n  MELHOR COMBINACAO: {best_combo} -> PnL={best_pnl:+.2f}%")

    # 11. Timeout trades analysis
    timeouts = td[td.reason == 'timeout']
    if len(timeouts) > 0:
        print(f"\n  --- TIMEOUT TRADES ({len(timeouts)}) ---")
        print(f"  WR timeouts: {len(timeouts[timeouts.pnl > 0])/len(timeouts)*100:.0f}%")
        print(f"  Avg PnL timeout: {timeouts['pnl'].mean():+.3f}%")
        print(f"  Se eliminados: PnL seria {td[td.reason != 'timeout']['pnl'].sum():+.2f}%")

    return td


def analyze_1h_trades():
    """Analise detalhada dos trades 1h."""
    print("\n" + "#" * 80)
    print("#  ANALISE DETALHADA: 1h CTEV v7.1 Regime-Switching")
    print("#" * 80)

    profile = get_profile("1h")
    metrics, trades = run_backtest(
        symbol="BTC/USDT", timeframe="1h", days=730, regime_switching=True,
    )

    if not trades:
        print("  Nenhum trade!")
        return

    td = pd.DataFrame([{
        'type': t.type, 'entry': t.entry_price, 'exit': t.exit_price,
        'pnl': t.pnl_pct, 'reason': t.exit_reason, 'bars': t.bars_held,
        'atr_pct': t.atr_percentile,
    } for t in trades])

    wins = td[td.pnl > 0]
    losses = td[td.pnl <= 0]

    print(f"\n  Total: {len(td)} trades")
    print(f"  WINS:  avg={wins['pnl'].mean():+.2f}%")
    print(f"  LOSSES: avg={losses['pnl'].mean():+.2f}%")
    print(f"  Avg win/loss: {abs(wins['pnl'].mean() / losses['pnl'].mean()):.2f}")

    # Por direcao
    print(f"\n  --- POR DIRECAO ---")
    for d in ['LONG', 'SHORT']:
        sub = td[td.type == d]
        if len(sub) > 0:
            wr = len(sub[sub.pnl > 0]) / len(sub) * 100
            print(f"  {d}: {len(sub)} trades WR={wr:.0f}% PnL={sub['pnl'].sum():+.2f}%")

    # Por razao
    print(f"\n  --- POR RAZAO ---")
    for r in td['reason'].unique():
        sub = td[td.reason == r]
        wr = len(sub[sub.pnl > 0]) / len(sub) * 100 if len(sub) > 0 else 0
        print(f"  {r}: {len(sub)} trades WR={wr:.0f}% avg_pnl={sub['pnl'].mean():+.2f}%")

    # Por ATR percentile
    print(f"\n  --- POR ATR PERCENTILE ---")
    for lo, hi in [(0, 0.3), (0.3, 0.7), (0.7, 1.0)]:
        sub = td[(td.atr_pct >= lo) & (td.atr_pct <= hi)]
        if len(sub) > 0:
            wr = len(sub[sub.pnl > 0]) / len(sub) * 100
            print(f"  ATR [{lo:.1f}-{hi:.1f}]: {len(sub)} trades WR={wr:.0f}% PnL={sub['pnl'].sum():+.2f}%")

    print(f"\n  PnL Total: {td['pnl'].sum():+.2f}%")
    return td


if __name__ == "__main__":
    analyze_15m_trades()
    analyze_1h_trades()
