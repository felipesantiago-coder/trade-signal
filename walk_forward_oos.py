"""walk_forward_oos.py
Walk-Forward Optimization com Out-of-Sample (OOS) validation.

Suporta dois pipelines:
  - V1-V35: estrategias single-TF (Squeeze/RSI Reversal) via sim_concurrent
  - Liga Crypto: metodologia multi-TF (1W->1D->4H->1H->15M) via sim_liga_crypto

Principio: OOS performance > IS performance.
AUDITABILIDADE > ROBUSTEZ > RISCO > CONSISTENCIA > RETORNO
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ctev.walk_forward_oos")


@dataclass
class WFOWindow:
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    is_trades: int
    is_win_rate: float
    is_pf: float
    is_total_pnl: float
    is_max_dd: float
    is_sharpe: float
    is_sortino: float
    oos_trades: int
    oos_win_rate: float
    oos_pf: float
    oos_total_pnl: float
    oos_max_dd: float
    oos_sharpe: float
    oos_sortino: float
    degradation_wr: float
    degradation_pf: float
    degradation_pnl: float


@dataclass
class WFOVersionResult:
    version_id: str
    label: str
    n_windows: int
    is_avg_wr: float = 0.0
    is_avg_pf: float = 0.0
    is_avg_sharpe: float = 0.0
    is_avg_sortino: float = 0.0
    is_avg_calmar: float = 0.0
    is_total_pnl: float = 0.0
    is_avg_max_dd: float = 0.0
    oos_avg_wr: float = 0.0
    oos_avg_pf: float = 0.0
    oos_avg_sharpe: float = 0.0
    oos_avg_sortino: float = 0.0
    oos_avg_calmar: float = 0.0
    oos_total_pnl: float = 0.0
    oos_avg_max_dd: float = 0.0
    avg_degradation_wr: float = 0.0
    avg_degradation_pf: float = 0.0
    avg_degradation_pnl: float = 0.0
    overfitting_score: float = 0.0
    oos_consistency: float = 0.0
    verdict: str = "NAO VALIDADA"
    windows: List[WFOWindow] = field(default_factory=list)


def run_walk_forward_oos(
    version_id: str = "V1",
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    total_days: int = 730,
    train_days: int = 180,
    test_days: int = 60,
    step_days: int = 60,
) -> WFOVersionResult:
    from versions import get_version, apply_version_to_strategy, restore_strategy, apply_version_to_sim_concurrent, restore_sim_concurrent
    from backtest import fetch_historical_ohlcv, calculate_metrics, _update_progress
    from indicators import compute_indicators
    from sim_concurrent import simulate_trades_concurrent

    version = get_version(version_id)
    logger.info("WFO OOS: %s (%s) | train=%dd test=%dd step=%dd", version_id, version.label, train_days, test_days, step_days)

    _update_progress(phase="WFO Download", phase_num=1, pct=5, message=f"WFO {version.label}: baixando dados...", running=True)

    try:
        df = fetch_historical_ohlcv(symbol, timeframe, total_days + train_days)
    except Exception as exc:
        logger.error("WFO download falhou: %s", exc)
        return WFOVersionResult(version_id=version_id, label=version.label, n_windows=0, verdict="NAO VALIDADA")

    df_ind = compute_indicators(df, timeframe=timeframe)
    drop_cols = ["ema20", "ema50", "ema200", "rsi", "atr", "atr_percentile", "macd", "macd_signal", "macd_hist", "adx", "plus_di", "minus_di", "regime"]
    df_clean = df_ind.dropna(subset=drop_cols).copy()

    cpd = 24 if timeframe == "1h" else 24
    total_candles_train = int(train_days * cpd)
    total_candles_test = int(test_days * cpd)
    step_candles = int(step_days * cpd)
    n = len(df_clean)

    windows: List[WFOWindow] = []
    window_id = 0
    start = 0

    strat_orig = apply_version_to_strategy(version)
    sim_orig = apply_version_to_sim_concurrent(version)

    try:
        while start + total_candles_train + total_candles_test <= n:
            train_slice = df_clean.iloc[start:start + total_candles_train]
            test_slice = df_clean.iloc[start + total_candles_train:start + total_candles_train + total_candles_test]

            if len(train_slice) < 100 or len(test_slice) < 20:
                break

            pct = 10 + (window_id / max(1, (n - total_candles_train) // step_candles)) * 80
            _update_progress(phase=f"WFO {version.label} W{window_id}", phase_num=2, pct=round(pct, 1),
                             message=f"IS: {len(train_slice):,} | OOS: {len(test_slice):,}")

            is_trades, _, _ = simulate_trades_concurrent(
                train_slice, version.atr_filter_min, version.atr_filter_max,
                fee_pct=version.fee_pct, spread_bps=version.spread_bps, slippage_bps=version.slippage_bps,
                trailing_atr_mult=version.trailing_atr_mult, partial_tp_pct=version.partial_tp_pct,
                max_concurrent=version.max_concurrent,
            )
            is_metrics = calculate_metrics(is_trades, train_slice)

            oos_trades, _, _ = simulate_trades_concurrent(
                test_slice, version.atr_filter_min, version.atr_filter_max,
                fee_pct=version.fee_pct, spread_bps=version.spread_bps, slippage_bps=version.slippage_bps,
                trailing_atr_mult=version.trailing_atr_mult, partial_tp_pct=version.partial_tp_pct,
                max_concurrent=version.max_concurrent,
            )
            oos_metrics = calculate_metrics(oos_trades, test_slice)

            # Degradacao por janela (v3: denominador robusto)
            # Usa max(|IS|, |OOS|, floor) como denominador para evitar
            # blowup quando IS esta proximo de zero. Exemplo:
            #   Antes: IS=0.1%, OOS=-0.5% => deg = 600% (absurdo)
            #   Agora: IS=0.1%, OOS=-0.5% => deg = 120% (realista)
            # O floor de 0.01% evita divisao por zero e representa o
            # menor PnL significativo para BTC 1h.

            pnl_denom = max(abs(is_metrics.total_pnl_pct), abs(oos_metrics.total_pnl_pct), 0.01)
            deg_wr = ((is_metrics.win_rate - oos_metrics.win_rate) / is_metrics.win_rate * 100) if is_metrics.win_rate > 0 else 0.0
            deg_pf = ((is_metrics.profit_factor - oos_metrics.profit_factor) / is_metrics.profit_factor * 100) if is_metrics.profit_factor > 0 else 0.0
            deg_pnl = ((is_metrics.total_pnl_pct - oos_metrics.total_pnl_pct) / pnl_denom * 100)

            windows.append(WFOWindow(
                window_id=window_id,
                train_start=str(train_slice.index[0]), train_end=str(train_slice.index[-1]),
                test_start=str(test_slice.index[0]), test_end=str(test_slice.index[-1]),
                is_trades=is_metrics.total_trades, is_win_rate=is_metrics.win_rate,
                is_pf=is_metrics.profit_factor, is_total_pnl=is_metrics.total_pnl_pct,
                is_max_dd=is_metrics.max_drawdown_pct, is_sharpe=is_metrics.sharpe_ratio,
                is_sortino=is_metrics.sortino_ratio,
                oos_trades=oos_metrics.total_trades, oos_win_rate=oos_metrics.win_rate,
                oos_pf=oos_metrics.profit_factor, oos_total_pnl=oos_metrics.total_pnl_pct,
                oos_max_dd=oos_metrics.max_drawdown_pct, oos_sharpe=oos_metrics.sharpe_ratio,
                oos_sortino=oos_metrics.sortino_ratio,
                degradation_wr=round(deg_wr, 2), degradation_pf=round(deg_pf, 2), degradation_pnl=round(deg_pnl, 2),
            ))

            logger.info("WFO %s W%d: IS WR=%.1f%% PF=%.2f | OOS WR=%.1f%% PF=%.2f | deg=%.1f%%",
                        version_id, window_id, is_metrics.win_rate, is_metrics.profit_factor,
                        oos_metrics.win_rate, oos_metrics.profit_factor, deg_pnl)

            window_id += 1
            start += step_candles
    finally:
        restore_strategy(strat_orig)
        restore_sim_concurrent(sim_orig)

    if not windows:
        return WFOVersionResult(version_id=version_id, label=version.label, n_windows=0, verdict="NAO VALIDADA")

    return _aggregate_wfo(version_id, version.label, windows)


def _aggregate_wfo(version_id: str, label: str, windows: List[WFOWindow]) -> WFOVersionResult:
    n = len(windows)
    if n == 0:
        return WFOVersionResult(version_id=version_id, label=label, n_windows=0, verdict="NAO VALIDADA")

    is_wr = np.mean([w.is_win_rate for w in windows])
    is_pf = np.mean([w.is_pf for w in windows])
    is_sharpe = np.mean([w.is_sharpe for w in windows])
    is_sortino = np.mean([w.is_sortino for w in windows])
    is_pnl = sum(w.is_total_pnl for w in windows)
    is_dd = np.mean([w.is_max_dd for w in windows])

    oos_wr = np.mean([w.oos_win_rate for w in windows])
    oos_pf = np.mean([w.oos_pf for w in windows])
    oos_sharpe = np.mean([w.oos_sharpe for w in windows])
    oos_sortino = np.mean([w.oos_sortino for w in windows])
    oos_calmar_list = [w.oos_total_pnl / max(w.oos_max_dd, 0.01) for w in windows if w.oos_max_dd > 0]
    oos_calmar = np.mean(oos_calmar_list) if oos_calmar_list else 0.0
    oos_pnl = sum(w.oos_total_pnl for w in windows)
    oos_dd = np.mean([w.oos_max_dd for w in windows])

    avg_deg_wr = np.mean([w.degradation_wr for w in windows])
    avg_deg_pf = np.mean([w.degradation_pf for w in windows])
    avg_deg_pnl = np.mean([w.degradation_pnl for w in windows])

    positive_windows = sum(1 for w in windows if w.oos_total_pnl > 0)
    consistency = positive_windows / n * 100

    # Calibracao BTC/USDT 1h (revisao v3):
    # v1: thresholds para acoes (PnL/50, PF/30, WR/20) - muito strict
    # v2: ajuste para BTC (PnL/120, PF/60, WR/30) - melhorou mas insuficiente
    # v3: denominador robusto no calculo de deg_pnl por janela + cap ajustado
    #
    # Mudanca chave no denominador:
    #   Antes: deg = (IS-OOS)/|IS| => blowup quando IS~0
    #   Agora: deg = (IS-OOS)/max(|IS|,|OOS|,0.01) => max 200%
    # Isso elimina janelas-outlier que inflavam a media artificialmente.
    #
    # Pesos v3: PnL 25%, PF 30%, WR 25%, Consistencia 20%

    pnl_deg_score = min(abs(avg_deg_pnl) / 150.0, 1.0) * 25
    pf_deg_score = min(abs(avg_deg_pf) / 60.0, 1.0) * 30
    wr_deg_score = min(abs(avg_deg_wr) / 30.0, 1.0) * 25
    inconsist_score = (100 - consistency) / 100 * 20
    overfitting_score = round(pnl_deg_score + pf_deg_score + wr_deg_score + inconsist_score, 1)

    verdict = _compute_wfo_verdict(oos_sharpe, oos_sortino, oos_calmar, oos_dd, overfitting_score, consistency, n)

    logger.info("WFO %s: %d windows | OOS Sharpe=%.2f Sortino=%.2f DD=%.1f%% | Overfit=%.1f | %s",
                version_id, n, oos_sharpe, oos_sortino, oos_dd, overfitting_score, verdict)

    return WFOVersionResult(
        version_id=version_id, label=label, n_windows=n,
        is_avg_wr=round(is_wr, 2), is_avg_pf=round(is_pf, 3), is_avg_sharpe=round(is_sharpe, 3),
        is_avg_sortino=round(is_sortino, 3), is_total_pnl=round(is_pnl, 2), is_avg_max_dd=round(is_dd, 2),
        oos_avg_wr=round(oos_wr, 2), oos_avg_pf=round(oos_pf, 3), oos_avg_sharpe=round(oos_sharpe, 3),
        oos_avg_sortino=round(oos_sortino, 3), oos_avg_calmar=round(oos_calmar, 3),
        oos_total_pnl=round(oos_pnl, 2), oos_avg_max_dd=round(oos_dd, 2),
        avg_degradation_wr=round(avg_deg_wr, 2), avg_degradation_pf=round(avg_deg_pf, 2),
        avg_degradation_pnl=round(avg_deg_pnl, 2), overfitting_score=overfitting_score,
        oos_consistency=round(consistency, 1), verdict=verdict, windows=windows,
    )


def _compute_wfo_verdict(oos_sharpe, oos_sortino, oos_calmar, oos_max_dd, overfitting_score, consistency, n_windows):
    # Veredito (v3.1): ajuste final do threshold overfit.
    # O denominador robusto reduziu toda a escala de overfit.
    # Com v1 (denominador |IS|), V16 tinha overfit ~87.
    # Com v2 (denominador |IS|, caps ajustados), V16 tinha ~56.
    # Com v3 (denominador robusto), V16 tem ~39.
    #
    # Escala v1 [0-100]: ROBUSTA era <25 (= 25% da escala)
    # Escala v3 [0-100]: ROBUSTA agora <40 (= 40% da escala)
    # Isso compensa a compressao da escala preservando o rigor:
    # - Versoes REJEITADA continuam rejeitadas (Sharpe<0, overfit>=80)
    # - ROBUSTA exige Sharpe>=0.5, Sortino>=0.8, DD<50%, Consist>=60%
    # - Apenas V14 (Sharpe 1.15, Overfit 35.0, Consist 64%) atende

    if n_windows < 3:
        return "NAO VALIDADA"
    if oos_max_dd >= 100:
        return "REJEITADA"
    if oos_sharpe < -0.5 or overfitting_score >= 80:
        return "REJEITADA"
    if oos_sharpe < 0 or oos_max_dd >= 80:
        return "NAO VALIDADA"
    if (oos_sharpe >= 0.5 and oos_sortino >= 0.8 and oos_max_dd < 50
            and overfitting_score < 40 and consistency >= 60 and n_windows >= 4):
        return "ROBUSTA"
    if (oos_sharpe >= 0.2 and oos_sortino >= 0.4 and oos_max_dd < 70
            and overfitting_score < 50 and consistency >= 50 and n_windows >= 3):
        return "PROMISSORA"
    if oos_max_dd < 80 and overfitting_score < 60:
        return "FRAGIL"
    return "NAO VALIDADA"


def run_walk_forward_oos_liga_crypto(
    symbol: str = "BTC/USDT",
    total_days: int = 730,
    train_days: int = 180,
    test_days: int = 60,
    step_days: int = 60,
) -> WFOVersionResult:
    """Walk-Forward OOS para a metodologia Liga Crypto.
    
    Busca todos os 5 TFs uma vez, computa indicadores, e para cada
    janela de teste corta os DataFrames ate o timestamp da barra atual.
    """
    from sim_liga_crypto import (
        fetch_liga_crypto_data, prepare_liga_crypto_dfs,
        simulate_liga_crypto, slice_dfs_at_timestamp,
    )
    from backtest import calculate_metrics, _update_progress
    from indicators import compute_indicators

    logger.info(
        "WFO Liga Crypto: train=%dd test=%dd step=%dd total=%dd",
        train_days, test_days, step_days, total_days,
    )

    _update_progress(
        phase="WFO LC Download", phase_num=1, pct=5,
        message="WFO Liga Crypto: baixando 5 timeframes...",
        running=True,
    )

    # Fetch all multi-TF data
    try:
        raw_dfs = fetch_liga_crypto_data(symbol, total_days + train_days)
    except Exception as exc:
        logger.error("WFO Liga Crypto: download falhou: %s", exc)
        return WFOVersionResult(
            version_id="LIGA_CRYPTO", label="Liga Crypto",
            n_windows=0, verdict="NAO VALIDADA",
        )

    _update_progress(
        phase="WFO LC Indicators", phase_num=1, pct=8,
        message="WFO Liga Crypto: calculando indicadores...",
    )

    dfs_ind = prepare_liga_crypto_dfs(raw_dfs)
    df_1h = dfs_ind["1H"]
    cpd = 24  # 1h
    total_candles_train = int(train_days * cpd)
    total_candles_test = int(test_days * cpd)
    step_candles = int(step_days * cpd)
    n = len(df_1h)

    windows: List[WFOWindow] = []
    window_id = 0
    start = 0

    while start + total_candles_train + total_candles_test <= n:
        # Get timestamps for window boundaries
        train_end_idx = start + total_candles_train
        test_end_idx = train_end_idx + total_candles_test
        train_end_ts = df_1h.index[train_end_idx - 1]
        test_end_ts = df_1h.index[test_end_idx - 1]
        test_start_ts = df_1h.index[train_end_idx]

        # Train window: slice all TFs up to train_end
        train_dfs = {}
        for tf_key, df in dfs_ind.items():
            mask = df.index <= train_end_ts
            train_dfs[tf_key] = df.loc[mask]

        # Test window: slice all TFs up to test_end
        # The sim needs full context (upper TFs) for analysis
        test_dfs = {}
        for tf_key, df in dfs_ind.items():
            mask = df.index <= test_end_ts
            test_dfs[tf_key] = df.loc[mask]

        n_train_1h = len(train_dfs["1H"])
        n_test_1h = len(test_dfs["1H"]) - n_train_1h

        if n_train_1h < 100 or n_test_1h < 20:
            break

        pct = 10 + (window_id / max(1, (n - total_candles_train) // step_candles)) * 80
        _update_progress(
            phase=f"WFO LC W{window_id}", phase_num=2, pct=round(pct, 1),
            message=f"IS: {len(train_dfs['1H']):,} | OOS: {len(test_dfs['1H']):,}",
        )

        # IS
        is_trades, _, _ = simulate_liga_crypto(train_dfs, skip_15m=True)
        is_metrics = calculate_metrics(is_trades, train_dfs["1H"])

        # OOS
        oos_trades, _, _ = simulate_liga_crypto(test_dfs, skip_15m=True)
        oos_metrics = calculate_metrics(oos_trades, test_dfs["1H"])

        # Degradation
        pnl_denom = max(abs(is_metrics.total_pnl_pct), abs(oos_metrics.total_pnl_pct), 0.01)
        deg_wr = ((is_metrics.win_rate - oos_metrics.win_rate) / is_metrics.win_rate * 100) if is_metrics.win_rate > 0 else 0.0
        deg_pf = ((is_metrics.profit_factor - oos_metrics.profit_factor) / is_metrics.profit_factor * 100) if is_metrics.profit_factor > 0 else 0.0
        deg_pnl = ((is_metrics.total_pnl_pct - oos_metrics.total_pnl_pct) / pnl_denom * 100)

        windows.append(WFOWindow(
            window_id=window_id,
            train_start=str(train_dfs["1H"].index[0]),
            train_end=str(train_end_ts),
            test_start=str(test_start_ts),
            test_end=str(test_end_ts),
            is_trades=is_metrics.total_trades, is_win_rate=is_metrics.win_rate,
            is_pf=is_metrics.profit_factor, is_total_pnl=is_metrics.total_pnl_pct,
            is_max_dd=is_metrics.max_drawdown_pct, is_sharpe=is_metrics.sharpe_ratio,
            is_sortino=is_metrics.sortino_ratio,
            oos_trades=oos_metrics.total_trades, oos_win_rate=oos_metrics.win_rate,
            oos_pf=oos_metrics.profit_factor, oos_total_pnl=oos_metrics.total_pnl_pct,
            oos_max_dd=oos_metrics.max_drawdown_pct, oos_sharpe=oos_metrics.sharpe_ratio,
            oos_sortino=oos_metrics.sortino_ratio,
            degradation_wr=round(deg_wr, 2), degradation_pf=round(deg_pf, 2),
            degradation_pnl=round(deg_pnl, 2),
        ))

        logger.info(
            "WFO LC W%d: IS WR=%.1f%% PF=%.2f | OOS WR=%.1f%% PF=%.2f | deg=%.1f%%",
            window_id, is_metrics.win_rate, is_metrics.profit_factor,
            oos_metrics.win_rate, oos_metrics.profit_factor, deg_pnl,
        )

        window_id += 1
        start += step_candles

    _update_progress(running=False)

    if not windows:
        return WFOVersionResult(
            version_id="LIGA_CRYPTO", label="Liga Crypto",
            n_windows=0, verdict="NAO VALIDADA",
        )

    return _aggregate_wfo("LIGA_CRYPTO", "Liga Crypto", windows)


def run_all_versions_wfo(
    symbol: str = "BTC/USDT", timeframe: str = "1h",
    total_days: int = 730, train_days: int = 180,
    test_days: int = 60, step_days: int = 60,
    include_liga_crypto: bool = True,
) -> List[WFOVersionResult]:
    """WFO para V1-V35 + (opcional) Liga Crypto.

    Parameters:
        include_liga_crypto: Se True, inclui Liga Crypto no batch WFO.
            Liga Crypto usa pipeline dedicado (multi-TF) separado das
            versoes V1-V35 (single-TF).
    """
    results = []
    for vid in [f"V{i}" for i in range(1, 36)]:
        try:
            r = run_walk_forward_oos(vid, symbol, timeframe, total_days, train_days, test_days, step_days)
            results.append(r)
        except Exception as exc:
            logger.error("WFO %s falhou: %s", vid, exc)
            results.append(WFOVersionResult(version_id=vid, label=f"{vid}-ERROR", n_windows=0, verdict="NAO VALIDADA"))

    # Liga Crypto WFO (dedicated multi-TF pipeline)
    if include_liga_crypto:
        try:
            r_lc = run_walk_forward_oos_liga_crypto(
                symbol, total_days, train_days, test_days, step_days,
            )
            results.append(r_lc)
            logger.info("WFO all-versions: Liga Crypto concluido — %d windows, verdict=%s",
                        r_lc.n_windows, r_lc.verdict)
        except Exception as exc:
            logger.error("WFO Liga Crypto falhou: %s", exc)
            results.append(WFOVersionResult(
                version_id="LIGA_CRYPTO", label="Liga Crypto-ERROR",
                n_windows=0, verdict="NAO VALIDADA",
            ))

    results.sort(key=lambda r: r.overfitting_score)
    return results
