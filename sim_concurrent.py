"""
sim_concurrent.py
----------------
v17.0 Concurrent Position Simulator para CTEV Multi-Strategy.

Diferenca vs simulate_trades_advanced:
  - Suporta ate N posicoes abertas simultaneamente
  - Avanca 1 bar por iteracao (nao pula bars durante trades)
  - Cada posicao tem SL/TP/trailing/max_bars independente
  - Compartilha cooldown e anti-martingale entre todas posicoes

Isso permite atingir 1+ trade/dia mesmo com sinais de 0.30-0.40/dia.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

from strategy import (
    Signal, SignalType, evaluate_row_signals,
)

logger = logging.getLogger("ctev.sim_concurrent")

# Import from backtest module
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest import (
    TradeResult, BacktestMetrics, _apply_costs,
    _update_progress, DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS,
)


MAX_CONCURRENT = 4       # v18.2: 4 para atingir 1+/dia
RISK_PER_TRADE = 0.01   # 1% do balance por trade


@dataclass
class _OpenPosition:
    """Posicao aberta com estado de gestao."""
    entry_idx: int           # Bar index de entrada
    signal: Signal           # Sinal original
    entry_price: float
    current_sl: float
    take_profit: float
    atr: float
    is_long: bool
    position_size: float
    position_usd: float
    risk_usd: float
    be_triggered: bool = False
    trailing_activated: bool = False
    partial_tp_filled: bool = False
    highest_favorable: float = 0.0
    sl_updates: int = 0
    bars: int = 0
    early_scalp_filled: bool = False  # v18.2: disabled


def simulate_trades_concurrent(
    df_ind: pd.DataFrame,
    atr_pct_min: float = 0.10,
    atr_pct_max: float = 0.90,
    initial_balance: float = 10000.0,
    risk_per_trade_pct: float = 0.01,
    fee_pct: float = DEFAULT_FEE_PCT,
    spread_bps: float = DEFAULT_SPREAD_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    trailing_atr_mult: float = 1.0,
    partial_tp_pct: float = 0.50,
    profile=None,
    max_concurrent: int = MAX_CONCURRENT,
) -> Tuple[List[TradeResult], int, dict]:
    """
    Simulacao v17.0 com posicoes concorrentes.

    Avanca 1 bar por iteracao, permitindo ate max_concurrent posicoes
    abertas simultaneamente. Cada posicao tem sua propria gestao SL/TP/trailing.
    """
    trades: List[TradeResult] = []
    atr_filtered = 0
    balance = initial_balance
    n = len(df_ind)
    _scan_t0 = time.time()
    _last_progress_time = 0.0
    _eq_points: list = []
    _running_pnl = 0.0

    # Diagnostics
    _diag_nan = 0
    _diag_regime_ranging = 0
    _diag_regime_volatile = 0
    _diag_regime_transition = 0
    _diag_trending_up = 0
    _diag_trending_down = 0
    _diag_atr_filtered = 0
    _diag_no_signal = 0
    _diag_cooldown_skip = 0
    _diag_max_concurrent_hit = 0

    # v18.1: Cooldown 8 bars (de 16) para mais rapidez em periodos curtos
    _consecutive_sl_long = 0
    _consecutive_sl_short = 0
    _cooldown_direction = None
    _cooldown_until_bar = 0
    _COOLDOWN_TRIGGER = 2
    _COOLDOWN_BARS = 8  # v18.1: 8 bars (~1/3 dia, de 16)

    # Anti-martingale DESATIVADO (v18.1: revert — nao ajuda com alta frequencia)
    _base_risk = risk_per_trade_pct
    _current_risk = _base_risk
    _consecutive_losses = 0
    _RISK_REDUCTION = 0.00
    _MIN_RISK_FRACTION = 1.00

    # Open positions
    open_positions: List[_OpenPosition] = []

    for i in range(n):
        row = df_ind.iloc[i]

        # Progress (throttled)
        _now = time.time()
        if _now - _last_progress_time > 0.5:
            _last_progress_time = _now
            _elapsed = max(_now - _scan_t0, 0.01)
            _speed = (i / _elapsed if _elapsed > 0.01 else 0)
            _pct = 20 + (i / max(n, 1)) * 60
            _update_progress(
                phase="Escaneando candles (concurrent)",
                phase_num=4, pct=round(_pct, 1),
                message=f"Bar {i:,}/{n:,} | Pos abertas: {len(open_positions)} | Trades: {len(trades)}",
                candles_total=n, candles_scanned=i, signals_found=len(trades),
                scan_speed=round(_speed),
            )

        # Check NaN
        critical = [
            "ema20", "ema50", "ema200", "rsi", "atr", "atr_percentile",
            "macd", "macd_signal", "macd_hist",
            "adx", "plus_di", "minus_di", "regime",
        ]
        if any(pd.isna(row.get(c)) for c in critical):
            _diag_nan += 1
            continue

        regime = str(row.get("regime", ""))
        if regime == "volatile":
                _diag_regime_volatile += 1
                atr_filtered += 1
                continue
        if regime == "ranging":
                _diag_regime_ranging += 1
        if regime == "trending_up":
                _diag_trending_up += 1
        if regime == "trending_down":
                _diag_trending_down += 1

        atr_pct = float(row.get("atr_percentile", 0.5))
        if atr_pct < atr_pct_min or atr_pct > atr_pct_max:
                _diag_atr_filtered += 1
                atr_filtered += 1
                continue

        # ── Check ALL open positions for SL/TP ──
        closed_positions = []
        for pos in open_positions:
            f_close = float(row["close"])
            f_low = float(row["low"])
            f_high = float(row["high"])
            f_rsi = float(row.get("rsi", 50.0))
            pos.bars = i - pos.entry_idx

            # Update highest favorable
            if pos.is_long:
                pos.highest_favorable = max(pos.highest_favorable, f_high)
            else:
                pos.highest_favorable = min(pos.highest_favorable, f_low)

            fav_dist = abs(pos.highest_favorable - pos.entry_price)

            # v18.2: Early Scalp DESATIVADO — custos de trading tornam BE negativo
            # if not pos.early_scalp_filled and fav_dist >= pos.atr * 1.5:
            #     pos.early_scalp_filled = True
            #     pos.be_triggered = True
            #     pos.current_sl = pos.entry_price
            #     pos.sl_updates += 1

            # Trailing (pos-TP1 only, same as advanced)
            if pos.partial_tp_filled:
                trail_dist = pos.atr * trailing_atr_mult
                if pos.is_long:
                    new_trail = pos.highest_favorable - trail_dist
                    if new_trail > pos.current_sl:
                        pos.current_sl = new_trail
                        pos.sl_updates += 1
                else:
                    new_trail = pos.highest_favorable + trail_dist
                    if new_trail < pos.current_sl:
                        pos.current_sl = new_trail
                        pos.sl_updates += 1

            # RSI exhaustion (after 24+ bars with profit)
            if pos.bars >= 24:
                if pos.is_long and f_rsi > 80.0:
                    _cp = f_close - pos.entry_price
                    if _cp > 0:
                        closed_positions.append((pos, f_close, "rsi_exhaustion"))
                        continue
                elif not pos.is_long and f_rsi < 20.0:
                    _cp = pos.entry_price - f_close
                    if _cp > 0:
                        closed_positions.append((pos, f_close, "rsi_exhaustion"))
                        continue

            # SL/TP check
            max_bars = getattr(pos.signal, "max_bars", 96)
            exit_price = None
            exit_reason = None

            if pos.is_long:
                tp_hit = f_high >= pos.take_profit
                sl_hit = f_low <= pos.current_sl
            else:
                tp_hit = f_low <= pos.take_profit
                sl_hit = f_high >= pos.current_sl

            if sl_hit and not tp_hit:
                exit_price = pos.current_sl
                exit_reason = "sl"
            elif tp_hit and not sl_hit:
                if not pos.partial_tp_filled:
                    pos.partial_tp_filled = True
                    # Pos-TP1 SL buffer
                    if pos.is_long:
                        pos.current_sl = pos.take_profit - pos.atr * 1.5
                    else:
                        pos.current_sl = pos.take_profit + pos.atr * 1.5
                    pos.sl_updates += 1
                else:
                    exit_price = pos.take_profit
                    exit_reason = "tp"
            elif tp_hit and sl_hit:
                exit_price = pos.current_sl
                exit_reason = "sl"
            elif pos.bars >= max_bars:
                exit_price = f_close
                exit_reason = "timeout"

            if exit_price is not None:
                closed_positions.append((pos, exit_price, exit_reason))

        # ── Process closed positions ──
        for pos, exit_price, exit_reason in closed_positions:
            open_positions.remove(pos)

            # Cost modeling
            entry_price = pos.entry_price
            is_long = pos.is_long
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

            # v18.2: Early Scalp DESATIVADO

            # Partial TP handling
            if pos.partial_tp_filled and exit_reason == "tp":
                pass  # Full TP — pnl_pct is correct
            elif pos.partial_tp_filled and exit_reason != "tp":
                tp = pos.take_profit
                if is_long:
                    partial_pnl = (tp - entry_price) / entry_price * 100
                else:
                    partial_pnl = (entry_price - tp) / entry_price * 100
                remaining_pnl = pnl_pct
                pnl_pct = partial_tp_pct * partial_pnl + (1 - partial_tp_pct) * remaining_pnl

            if pos.position_usd > 0:
                pnl_usd = pos.position_usd * (pnl_pct / 100)
            else:
                pnl_usd = 0

            balance += pnl_usd

            # Record trade
            trades.append(TradeResult(
                entry_ts=df_ind.index[pos.entry_idx],
                exit_ts=df_ind.index[min(i, n - 1)],
                type=pos.signal.type.value,
                entry_price=entry_price,
                exit_price=exit_price,
                stop_loss=pos.signal.stop_loss,
                take_profit=pos.take_profit,
                atr=pos.atr,
                rsi=pos.signal.rsi,
                pnl_pct=round(pnl_pct, 4),
                pnl_abs=round(exit_price - entry_price, 2),
                bars_held=pos.bars,
                exit_reason=exit_reason,
                atr_percentile=float(row.get("atr_percentile", 0.5)),
                position_size=round(pos.position_size, 8),
                position_usd=round(pos.position_usd, 2),
                risk_usd=round(pos.risk_usd, 2),
                be_triggered=pos.be_triggered,
                trailing_activated=pos.trailing_activated,
                partial_tp_filled=pos.partial_tp_filled,
                sl_updates=pos.sl_updates,
                entry_type=getattr(pos.signal, 'entry_type', 'unknown'),
            ))

            # Equity tracking
            _running_pnl += pnl_pct
            _eq_points.append(round(_running_pnl, 2))

            # Cooldown update
            if exit_reason == "sl" and pnl_pct < 0:
                if is_long:
                    _consecutive_sl_long += 1
                    _consecutive_sl_short = 0
                    if _consecutive_sl_long >= _COOLDOWN_TRIGGER:
                        _cooldown_direction = "LONG"
                        _cooldown_until_bar = i + 1 + _COOLDOWN_BARS
                        _consecutive_sl_long = 0
                else:
                    _consecutive_sl_short += 1
                    _consecutive_sl_long = 0
                    if _consecutive_sl_short >= _COOLDOWN_TRIGGER:
                        _cooldown_direction = "SHORT"
                        _cooldown_until_bar = i + 1 + _COOLDOWN_BARS
                        _consecutive_sl_short = 0
            else:
                if is_long:
                    _consecutive_sl_long = 0
                else:
                    _consecutive_sl_short = 0
                if _cooldown_direction == pos.signal.type.value:
                    _cooldown_direction = None

            # Anti-martingale
            if pnl_pct < 0:
                _consecutive_losses += 1
                _current_risk = max(
                    _base_risk * _MIN_RISK_FRACTION,
                    _current_risk * (1 - _RISK_REDUCTION),
                )
            else:
                _consecutive_losses = 0
                _current_risk = _base_risk

        # ── Check for new signal ──
        if len(open_positions) < max_concurrent:
            signal = evaluate_row_signals(row, profile=profile)
            if signal is not None:
                # Cooldown check
                _sig_dir = signal.type.value
                if (_cooldown_direction is not None
                        and _sig_dir == _cooldown_direction
                        and i < _cooldown_until_bar):
                    _diag_cooldown_skip += 1
                    continue

                # v18.0: Allow same-direction entries (removed directional loss guard)
                # O cooldown e anti-martingale ja protegem contra perdas cascata.
                # O loss guard era excessivamente restritivo em periodos curtos.

                entry_price = signal.entry_price
                sl = signal.stop_loss
                tp = signal.take_profit
                atr = signal.atr
                is_long = signal.type == SignalType.LONG

                # Cost-adjusted entry
                if fee_pct > 0:
                    adj_entry, _, _ = _apply_costs(
                        entry_price, entry_price, is_long,
                        fee_pct, spread_bps, slippage_bps,
                    )
                else:
                    adj_entry = entry_price

                sl_distance_pct = abs(adj_entry - sl) / adj_entry if adj_entry > 0 else 0
                risk_usd = balance * _current_risk
                if sl_distance_pct > 0:
                    position_size = risk_usd / (sl_distance_pct * adj_entry)
                else:
                    position_size = 0.0
                position_usd = position_size * adj_entry

                if position_usd < 1.0:
                    continue

                open_positions.append(_OpenPosition(
                    entry_idx=i,
                    signal=signal,
                    entry_price=entry_price,
                    current_sl=sl,
                    take_profit=tp,
                    atr=atr,
                    is_long=is_long,
                    position_size=position_size,
                    position_usd=position_usd,
                    risk_usd=risk_usd,
                    highest_favorable=entry_price,
                ))
        elif len(open_positions) >= max_concurrent:
            _diag_max_concurrent_hit += 1

    # Close any remaining open positions at last bar
    for pos in open_positions:
        last_close = float(df_ind.iloc[n - 1]["close"])
        if pos.position_usd > 0:
            if pos.is_long:
                pnl_pct = (last_close - pos.entry_price) / pos.entry_price * 100
            else:
                pnl_pct = (pos.entry_price - last_close) / pos.entry_price * 100
            pnl_usd = pos.position_usd * (pnl_pct / 100)
        else:
            pnl_pct = 0.0
            pnl_usd = 0.0
        balance += pnl_usd
        trades.append(TradeResult(
            entry_ts=df_ind.index[pos.entry_idx],
            exit_ts=df_ind.index[n - 1],
            type=pos.signal.type.value,
            entry_price=pos.entry_price,
            exit_price=last_close,
            stop_loss=pos.signal.stop_loss,
            take_profit=pos.take_profit,
            atr=pos.atr,
            rsi=pos.signal.rsi,
            pnl_pct=round(pnl_pct, 4),
            pnl_abs=round(last_close - pos.entry_price, 2),
            bars_held=n - 1 - pos.entry_idx,
            exit_reason="timeout_eod",
            atr_percentile=0.5,
            position_size=round(pos.position_size, 8),
            position_usd=round(pos.position_usd, 2),
            risk_usd=round(pos.risk_usd, 2),
            entry_type=getattr(pos.signal, 'entry_type', 'unknown'),
        ))

    logger.info(
        "Simulacao concurrent v17 completa: %d trades, %d filtrados vol, %d regime, balance=$%.2f",
        len(trades), atr_filtered, 0, balance,
    )

    _diag = {
        "nan_filtered": _diag_nan,
        "regime_ranging": _diag_regime_ranging,
        "regime_volatile": _diag_regime_volatile,
        "regime_transition": _diag_regime_transition,
        "trending_up": _diag_trending_up,
        "trending_down": _diag_trending_down,
        "atr_filtered_diag": _diag_atr_filtered,
        "no_signal": _diag_no_signal,
        "cooldown_skip": _diag_cooldown_skip,
        "max_concurrent_hit": _diag_max_concurrent_hit,
    }

    return trades, atr_filtered, _diag
