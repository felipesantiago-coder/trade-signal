"""sim_liga_crypto.py
------------------
Liga Crypto Multi-Timeframe Backtest Simulator.

Simula a metodologia Liga Crypto (1W->1D->4H->1H->15M) em backtest,
buscando dados de 5 timeframes e avaliando a hierarquia a cada barra.

Performance optimization:
  - Fetches all 5 TFs ONCE at start
  - Computes indicators ONCE per TF
  - For each 1H bar, slices the pre-computed DataFrames up to that timestamp
  - Calls analyze_liga_crypto() which uses .iloc[-1] internally
  - Re-analyzes every 4 bars (4h) to reduce computation

Trade simulation:
  - Single position at a time (Liga Crypto = sinalizacao seletiva)
  - SL/TP based on 4H ATR (1.8x SL, Fib projections for TP)
  - Partial TP: 50% at TP1, trailing stop after TP1
  - Cost modeling: fees + spread + slippage (realistic for BTC/USDT)

Gestao de risco:
  - R:R minimo 2:1
  - Cooldown: 2 SL -> 6 bars
  - Max bars: 168 (7 dias)
  - Risk: 2% por trade (Liga Crypto = maior confianca)

Filtros quantitativos:
  - Weekend (Saturday/Sunday UTC): bloqueia entrada
  - Sazonalidade: score <= -2 bloqueia (ciclo BTC + mes + dow)
  - Macro eventos: FOMC/CPI/NFP (janelas de 1h) bloqueiam entrada
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("ctev.sim_liga_crypto")

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest import (
    TradeResult, BacktestMetrics, _apply_costs, _compute_audit_fields,
    _update_progress, DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS,
    fetch_historical_ohlcv,
)
from indicators import compute_indicators


# ── Liga Crypto Backtest Parameters ──
MAX_BARS_HELD = 168         # 7 dias (1h bars)
RISK_PER_TRADE_PCT = 0.02  # 2% por trade (confianca da metodologia)
COOLDOWN_SL_TRIGGER = 2    # 2 SLs consecutivos
COOLDOWN_BARS = 6           # ~6 horas de pausa
TRAILING_ATR_MULT = 1.0    # 1.0x ATR trailing apos TP1
PARTIAL_TP_PCT = 0.50       # 50% em TP1

# Timeframe mapping: Liga Crypto key -> ccxt timeframe
LIGA_TF_MAP = {
    "1W": "1W",
    "1D": "1d",
    "4H": "4h",
    "1H": "1h",
    "15M": "15m",
}


# ═══════════════════════════════════════════════════════════════════
# DATA FETCHING — Multi-Timeframe
# ═══════════════════════════════════════════════════════════════════

def _resample_1d_to_1w(df: pd.DataFrame) -> pd.DataFrame:
    """Resample DataFrame 1D para 1W (semanal, domingo->sabado).

    OHLCV padrao: open=primeira abertura, high=maximo, low=minimo,
    close=ultimo fechamento, volume=soma dos volumes.
    """
    if df.empty:
        return df
    rule = "W-FRI"  # semana termina na sexta (padrao financeiro)
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    # Filtra apenas colunas OHLCV (pode haver colunas de indicadores)
    ohlcv_cols = [c for c in agg if c in df.columns]
    agg_filtered = {c: agg[c] for c in ohlcv_cols}
    df_w = df[ohlcv_cols].resample(rule, label="left", closed="left").agg(agg_filtered)
    df_w.dropna(subset=["close"], inplace=True)
    return df_w


def fetch_liga_crypto_data(
    symbol: str = "BTC/USDT",
    days: int = 730,
) -> Dict[str, pd.DataFrame]:
    """
    Busca dados OHLCV para os 5 timeframes da Liga Crypto.

    Para 1H, busca o periodo completo. Para os demais TFs,
    busca com margem para cobrir o periodo do backtest.

    Nota: 1W nao e suportado por nenhuma exchange (Coinbase, Binance, Bybit),
    entao os dados 1W sao gerados por resample dos dados 1D.

    Returns:
        Dict["1W"|"1D"|"4H"|"1H"|"15M", DataFrame] com OHLCV.
    """
    dfs = {}

    # 1W: resample de 1D (exchange nao suporta granularidade semanal)
    limit_days_1d = 5 * 365  # 5 anos para ter weeklys suficientes
    try:
        _update_progress(
            phase="Baixando dados MTF", phase_num=1, pct=5,
            message="Baixando 1D (para 1W + 1D)...",
        )
        df_1d = fetch_historical_ohlcv(symbol, "1d", limit_days_1d)
        df_1w = _resample_1d_to_1w(df_1d)
        dfs["1W"] = df_1w
        dfs["1D"] = df_1d
        logger.info(
            "Liga Crypto backtest: 1W = %d candles (%s a %s) [resample de 1D]",
            len(df_1w), df_1w.index[0], df_1w.index[-1],
        )
        logger.info(
            "Liga Crypto backtest: 1D = %d candles (%s a %s)",
            len(df_1d), df_1d.index[0], df_1d.index[-1],
        )
    except Exception as exc:
        logger.error("Liga Crypto backtest: falha ao buscar 1D: %s", exc)
        raise RuntimeError(f"Falha ao buscar dados 1D: {exc}") from exc

    # Timeframes restantes: 4H, 1H, 15M
    remaining_tfs = {
        "4H": ("4h", days + 60),
        "1H": ("1h", days),
        "15M": ("15m", 90),
    }

    for tf_key, (ccxt_tf, tf_days) in remaining_tfs.items():
        try:
            df = fetch_historical_ohlcv(symbol, ccxt_tf, tf_days)
            dfs[tf_key] = df
            logger.info(
                "Liga Crypto backtest: %s = %d candles (%s a %s)",
                tf_key, len(df), df.index[0], df.index[-1],
            )
        except Exception as exc:
            logger.error("Liga Crypto backtest: falha ao buscar %s: %s", tf_key, exc)
            raise RuntimeError(f"Falha ao buscar dados {tf_key}: {exc}") from exc

    return dfs


def prepare_liga_crypto_dfs(
    raw_dfs: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    Computa indicadores para todos os TFs de uma vez.

    Returns:
        Dict com DataFrames processados (com indicadores) por TF.
    """
    tf_name_map = {"1W": "1W", "1D": "1d", "4H": "4h", "1H": "1h", "15M": "15m"}
    dfs_ind = {}
    for tf_key, df in raw_dfs.items():
        indicator_tf = tf_name_map.get(tf_key, tf_key.lower())
        dfs_ind[tf_key] = compute_indicators(df, timeframe=indicator_tf)
        logger.debug("Liga Crypto: indicadores calculados para %s (%d candles)", tf_key, len(df))
    return dfs_ind


def slice_dfs_at_timestamp(
    dfs_ind: Dict[str, pd.DataFrame],
    ts: pd.Timestamp,
) -> Dict[str, pd.DataFrame]:
    """
    Corta todos os DataFrames para incluir apenas barras ate `ts`.

    Para cada TF, pega todas as barras com indice <= ts.
    Isso permite que analyze_liga_crypto() use .iloc[-1] para a barra "atual".
    """
    sliced = {}
    for tf_key, df in dfs_ind.items():
        mask = df.index <= ts
        sliced_df = df.loc[mask]
        if len(sliced_df) > 0:
            sliced[tf_key] = sliced_df
    return sliced


# ═══════════════════════════════════════════════════════════════════
# MAIN SIMULATION
# ═══════════════════════════════════════════════════════════════════

def simulate_liga_crypto(
    dfs_ind: Dict[str, pd.DataFrame],
    fee_pct: float = DEFAULT_FEE_PCT,
    spread_bps: float = DEFAULT_SPREAD_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    initial_balance: float = 10000.0,
    skip_15m: bool = True,
) -> Tuple[List[TradeResult], int, dict]:
    """
    Simulacao da metodologia Liga Crypto em dados historicos.

    Avanca 1 bar (1H) por iteracao. A cada barra, corta os DFs dos
    outros TFs ate o timestamp atual e chama analyze_liga_crypto().

    Parameters:
        dfs_ind: Dict com DataFrames processados (indicadores calculados).
        fee_pct/spread_bps/slippage_bps: costos de transacao.
        initial_balance: Balance inicial.
        skip_15m: Se True, nao inclui 15M nos dados (mais rapido, menos preciso).

    Returns:
        (List[TradeResult], atr_filtered_count, diagnostics_dict)
    """
    from strategy_liga_crypto import (
        analyze_liga_crypto, liga_crypto_to_signal,
        _get_seasonal_context_quant, _is_macro_event_window, Decision,
    )

    trades: List[TradeResult] = []
    atr_filtered = 0
    balance = initial_balance
    n = len(dfs_ind["1H"])
    _scan_t0 = time.time()
    _last_progress_time = 0.0
    _eq_points: list = []
    _running_pnl = 0.0

    # Diagnostics
    _diag_no_signal = 0
    _diag_blocked = 0
    _diag_rr_rejected = 0
    _diag_weekend_filtered = 0
    _diag_seasonal_filtered = 0
    _diag_macro_filtered = 0
    _diag_cooldown_skip = 0

    # Position state
    _in_position = False
    _pos = None  # dict with all position fields

    # Cooldown
    _consecutive_sl = 0
    _cooldown_until_bar = 0

    df_1h = dfs_ind["1H"]

    # Re-analyze frequency: every 4 bars (4h) to reduce computation
    _REANALYZE_INTERVAL = 4
    _last_analysis_bar = -999
    _cached_signal = None
    _cached_lc_result = None

    def _close_position(i: int, exit_price: float, exit_reason: str):
        """Close the current position and record the trade."""
        nonlocal _in_position, _pos, balance, _running_pnl, _consecutive_sl

        is_long = _pos["is_long"]
        entry_price = _pos["entry_price"]

        if fee_pct > 0:
            _, adj_exit, cost_pct = _apply_costs(
                entry_price, exit_price, is_long,
                fee_pct, spread_bps, slippage_bps,
            )
        else:
            adj_exit = exit_price
            cost_pct = 0.0

        if is_long:
            pnl_pct = (adj_exit - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - adj_exit) / entry_price * 100

        # Partial TP PnL adjustment
        if _pos["tp1_filled"] and exit_reason not in ("tp", "sl"):
            if is_long:
                partial_pnl = (_pos["tp1"] - entry_price) / entry_price * 100
            else:
                partial_pnl = (entry_price - _pos["tp1"]) / entry_price * 100
            remaining_pnl = pnl_pct
            pnl_pct = PARTIAL_TP_PCT * partial_pnl + (1 - PARTIAL_TP_PCT) * remaining_pnl

        pnl_usd = _pos["position_usd"] * (pnl_pct / 100) if _pos["position_usd"] > 0 else 0
        balance += pnl_usd

        signal = _pos["signal"]
        _sl_risk_pct = abs((entry_price - signal.stop_loss) / entry_price * 100) if signal.stop_loss != 0 and entry_price > 0 else 0.0
        _r_mult = round(abs(pnl_pct) / max(_sl_risk_pct, 0.01), 2) if _sl_risk_pct > 0 else 0.0

        trades.append(TradeResult(
            entry_ts=df_1h.index[_pos["entry_idx"]],
            exit_ts=df_1h.index[min(i, n - 1)],
            type="LONG" if is_long else "SHORT",
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=signal.stop_loss,
            take_profit=_pos["tp1"],
            atr=_pos["atr"],
            rsi=float(df_1h.iloc[i].get("rsi", 50.0)),
            pnl_pct=round(pnl_pct, 4),
            pnl_abs=round(exit_price - entry_price, 2),
            bars_held=i - _pos["entry_idx"],
            exit_reason=exit_reason,
            atr_percentile=float(df_1h.iloc[i].get("atr_percentile", 0.5)),
            position_size=round(_pos["position_size"], 8),
            position_usd=round(_pos["position_usd"], 2),
            risk_usd=round(_pos["risk_usd"], 2),
            trailing_activated=_pos["tp1_filled"],
            partial_tp_filled=_pos["tp1_filled"],
            sl_updates=_pos["sl_updates"],
            entry_type="liga_crypto",
            regime_at_entry="",
            concurrent_count=0,
            equity_before=0.0,
            equity_after=0.0,
            capital_allocated=round(_pos["position_usd"], 2),
            quantity=round(_pos["position_size"], 8),
            r_multiple=_r_mult,
            **_compute_audit_fields(
                entry_price=entry_price, exit_price=exit_price,
                is_long=is_long, position_size=_pos["position_size"],
                position_usd=_pos["position_usd"], bars_held=i - _pos["entry_idx"],
                entry_ts=df_1h.index[_pos["entry_idx"]],
                exit_ts=df_1h.index[min(i, n - 1)],
                fee_pct=fee_pct, spread_bps=spread_bps, slippage_bps=slippage_bps,
            ),
        ))

        _running_pnl += pnl_pct
        _eq_points.append(round(_running_pnl, 2))

        # Cooldown update
        if exit_reason == "sl" and pnl_pct < 0:
            _consecutive_sl += 1
            if _consecutive_sl >= COOLDOWN_SL_TRIGGER:
                _cooldown_until_bar = i + 1 + COOLDOWN_BARS
                _consecutive_sl = 0
        else:
            _consecutive_sl = 0

        _in_position = False
        _pos = None

    for i in range(n):
        row = df_1h.iloc[i]
        ts = df_1h.index[i]

        # Progress (throttled)
        _now = time.time()
        if _now - _last_progress_time > 0.5:
            _last_progress_time = _now
            _elapsed = max(_now - _scan_t0, 0.01)
            _speed = i / _elapsed if _elapsed > 0.01 else 0
            _pct = 20 + (i / max(n, 1)) * 60
            _update_progress(
                phase="Escaneando candles (Liga Crypto MTF)",
                phase_num=4, pct=round(_pct, 1),
                message=f"Bar {i:,}/{n:,} | Trades: {len(trades)} | In pos: {_in_position}",
                candles_total=n, candles_scanned=i, signals_found=len(trades),
                scan_speed=round(_speed),
            )

        # ── MANAGE OPEN POSITION ──
        if _in_position:
            f_close = float(row["close"])
            f_low = float(row["low"])
            f_high = float(row["high"])
            bars = i - _pos["entry_idx"]
            is_long = _pos["is_long"]

            # Update highest favorable excursion
            if is_long:
                _pos["highest_fav"] = max(_pos["highest_fav"], f_high)
            else:
                _pos["highest_fav"] = min(_pos["highest_fav"], f_low)

            # Trailing after TP1
            if _pos["tp1_filled"]:
                trail_dist = _pos["atr"] * TRAILING_ATR_MULT
                if is_long:
                    new_sl = _pos["highest_fav"] - trail_dist
                    if new_sl > _pos["current_sl"]:
                        _pos["current_sl"] = new_sl
                        _pos["sl_updates"] += 1
                else:
                    new_sl = _pos["highest_fav"] + trail_dist
                    if new_sl < _pos["current_sl"]:
                        _pos["current_sl"] = new_sl
                        _pos["sl_updates"] += 1

            # SL/TP check
            current_sl = _pos["current_sl"]
            tp1 = _pos["tp1"]

            if is_long:
                tp_hit = f_high >= tp1
                sl_hit = f_low <= current_sl
            else:
                tp_hit = f_low <= tp1
                sl_hit = f_high >= current_sl

            if sl_hit and not tp_hit:
                _close_position(i, current_sl, "sl")
                _cached_signal = None
            elif tp_hit and not sl_hit:
                if not _pos["tp1_filled"]:
                    # First TP hit: partial close, move SL to breakeven+buffer
                    _pos["tp1_filled"] = True
                    if is_long:
                        _pos["current_sl"] = tp1 - _pos["atr"] * 1.5
                    else:
                        _pos["current_sl"] = tp1 + _pos["atr"] * 1.5
                    _pos["sl_updates"] += 1
                else:
                    _close_position(i, tp1, "tp")
                    _cached_signal = None
            elif tp_hit and sl_hit:
                _close_position(i, current_sl, "sl")
                _cached_signal = None
            elif bars >= MAX_BARS_HELD:
                _close_position(i, f_close, "timeout")
                _cached_signal = None

            continue  # Skip signal generation while in position

        # ── NOT IN POSITION: CHECK FOR SIGNAL ──
        # Cooldown check
        if i < _cooldown_until_bar:
            _diag_cooldown_skip += 1
            continue

        # Weekend filter (UTC)
        if hasattr(ts, 'weekday') and ts.weekday() >= 5:
            _diag_weekend_filtered += 1
            continue

        # Seasonal filter (quantitative)
        seasonal_score = _get_seasonal_context_quant(ts)
        if seasonal_score <= -2:
            _diag_seasonal_filtered += 1
            continue

        # Macro event filter (FOMC, CPI, NFP windows)
        if _is_macro_event_window(ts):
            _diag_macro_filtered += 1
            continue

        # Re-analyze only every REANALYZE_INTERVAL bars
        if i - _last_analysis_bar >= _REANALYZE_INTERVAL or _cached_signal is None:
            sliced = slice_dfs_at_timestamp(dfs_ind, ts)
            if skip_15m and "15M" in sliced:
                del sliced["15M"]

            if len(sliced) < 4:
                continue

            try:
                lc_result = analyze_liga_crypto(sliced)
                signal = liga_crypto_to_signal(lc_result)
                _cached_signal = signal
                _cached_lc_result = lc_result
                _last_analysis_bar = i

                # Track diagnostics
                if lc_result.decision == Decision.AGUARDAR:
                    _diag_no_signal += 1
                    _cached_signal = None
                elif 'R:R' in (lc_result.justification or ''):
                    _diag_rr_rejected += 1
                    _cached_signal = None
                elif any('bloquead' in r.lower() for r in (lc_result.invalidation_reasons or [])):
                    _diag_blocked += 1
                    _cached_signal = None
            except Exception as exc:
                logger.debug("Liga Crypto analysis error at bar %d: %s", i, exc)
                _cached_signal = None
                continue
        else:
            signal = _cached_signal
            lc_result = _cached_lc_result

        if signal is None:
            continue

        # ── OPEN POSITION ──
        entry_price = signal.entry_price
        sl = signal.stop_loss
        atr = signal.atr
        is_long = signal.type.value == "LONG"

        if fee_pct > 0:
            adj_entry, _, _ = _apply_costs(
                entry_price, entry_price, is_long,
                fee_pct, spread_bps, slippage_bps,
            )
        else:
            adj_entry = entry_price

        sl_distance_pct = abs(adj_entry - sl) / adj_entry if adj_entry > 0 else 0
        risk_usd = balance * RISK_PER_TRADE_PCT
        if sl_distance_pct > 0:
            position_size = risk_usd / (sl_distance_pct * adj_entry)
        else:
            position_size = 0.0
        position_usd = position_size * adj_entry

        if position_usd < 1.0 or sl_distance_pct <= 0:
            continue

        # Get TPs from Liga Crypto result
        tp1 = lc_result.tp1 if lc_result and lc_result.tp1 > 0 else signal.take_profit
        tp2 = lc_result.tp2 if lc_result and lc_result.tp2 > 0 else signal.take_profit
        tp3 = lc_result.tp3 if lc_result and lc_result.tp3 > 0 else signal.take_profit

        _in_position = True
        _pos = {
            "entry_idx": i,
            "signal": signal,
            "entry_price": entry_price,
            "current_sl": sl,
            "tp1": tp1, "tp2": tp2, "tp3": tp3,
            "is_long": is_long,
            "atr": atr,
            "tp1_filled": False,
            "highest_fav": entry_price,
            "sl_updates": 0,
            "position_size": position_size,
            "position_usd": position_usd,
            "risk_usd": risk_usd,
        }
        _cached_signal = None
        _cached_lc_result = None

    # Close any remaining position
    if _in_position:
        last_close = float(df_1h.iloc[n - 1]["close"])
        _close_position(n - 1, last_close, "timeout_eod")

    logger.info(
        "Liga Crypto simulacao completa: %d trades | balance=$%.2f | no_signal=%d blocked=%d rr_rej=%d weekend=%d seasonal=%d macro=%d cooldown=%d",
        len(trades), balance, _diag_no_signal, _diag_blocked, _diag_rr_rejected,
        _diag_weekend_filtered, _diag_seasonal_filtered, _diag_macro_filtered, _diag_cooldown_skip,
    )

    _diag = {
        "no_signal": _diag_no_signal,
        "blocked": _diag_blocked,
        "rr_rejected": _diag_rr_rejected,
        "weekend_filtered": _diag_weekend_filtered,
        "seasonal_filtered": _diag_seasonal_filtered,
        "macro_filtered": _diag_macro_filtered,
        "cooldown_skip": _diag_cooldown_skip,
        "strategy": "liga_crypto",
    }

    return trades, atr_filtered, _diag
