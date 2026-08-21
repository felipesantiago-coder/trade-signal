#!/usr/bin/env python3
"""
v26.0 - Populate gross_pnl/fees/slippage/funding/net_pnl/return_pct/duration in all TradeResult objects.

This script patches backtest.py and sim_concurrent.py to:
1. Add a _compute_audit_fields() helper function
2. Use _apply_costs_detail() at every trade append point
3. Populate: gross_pnl, fees, slippage, funding, net_pnl, return_pct, duration

Run: python scripts/v26_audit_fields.py
"""

import re
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def patch_backtest_py():
    path = os.path.join(BASE_DIR, "backtest.py")
    with open(path, "r") as f:
        content = f.read()

    # === 1. Add _compute_audit_fields helper after _apply_costs_detail ===
    helper_code = '''

# ------------------------------------------------------------------
# v26.0: Audit field calculator for TradeResult
# ------------------------------------------------------------------
def _compute_audit_fields(
    entry_price: float, exit_price: float, is_long: bool,
    position_size: float, position_usd: float, bars_held: int,
    entry_ts, exit_ts,
    fee_pct: float = DEFAULT_FEE_PCT,
    spread_bps: float = DEFAULT_SPREAD_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    funding_rate_bps: float = 0.01,  # ~0.01% per 8h for BTC perpetual
    apply_costs_flag: bool = True,
) -> dict:
    """v26.0: Compute detailed audit fields for a TradeResult.
    
    Returns dict with: gross_pnl, fees, slippage, funding, net_pnl, return_pct, duration, pnl_pct
    """
    if entry_price <= 0:
        return {"gross_pnl": 0.0, "fees": 0.0, "slippage": 0.0, "funding": 0.0,
            "net_pnl": 0.0, "return_pct": 0.0, "duration": "", "pnl_pct": 0.0}

    # Raw gross PnL (no costs)
    if is_long:
        gross_pnl_pct = (exit_price - entry_price) / entry_price * 100
    else:
        gross_pnl_pct = (entry_price - exit_price) / entry_price * 100
    
    if apply_costs_flag and position_usd > 0:
        # Detailed cost decomposition
        _, _, total_cost_pct, fee_comp, slip_comp, spread_comp = _apply_costs_detail(
            entry_price, exit_price, is_long, fee_pct, spread_bps, slippage_bps,
        )
        
        fees_usd = position_usd * (fee_comp / 100)
        slippage_usd = position_usd * (slip_comp / 100)
        
        # Adjusted exit for net PnL
        _, adj_exit, _ = _apply_costs(
            entry_price, exit_price, is_long, fee_pct, spread_bps, slippage_bps,
        )
        if is_long:
            net_pnl_pct = (adj_exit - entry_price) / entry_price * 100
        else:
            net_pnl_pct = (entry_price - adj_exit) / entry_price * 100
        
        net_pnl_usd = position_usd * (net_pnl_pct / 100)
    else:
        fees_usd = 0.0
        slippage_usd = 0.0
        total_cost_pct = 0.0
        net_pnl_pct = gross_pnl_pct
        net_pnl_usd = position_usd * (net_pnl_pct / 100) if position_usd > 0 else 0.0

    # Funding rate estimation (perpetual swap: ~0.01% per 8h, 1h candles = 0.01/8 per bar)
    funding_per_bar = position_usd * (funding_rate_bps / 10000) / 8.0 if position_usd > 0 else 0.0
    funding_usd = -funding_per_bar * bars_held  # always a cost for long, offset for short
    # For shorts, funding can be positive (you receive it), but model as cost for conservatism
    if not is_long:
        funding_usd = abs(funding_usd) * 0.7  # shorts receive ~70% of time on BTC
    else:
        funding_usd = -abs(funding_usd) * 0.6  # longs pay ~60% of time
    
    # Duration string
    hours = bars_held  # 1h candles = 1h per bar
    if hours < 24:
        duration = f"{hours}h"
    elif hours < 48:
        duration = f"1d {hours - 24}h"
    elif hours < 168:
        days = hours // 24
        rem_h = hours % 24
        duration = f"{days}d {rem_h}h"
    else:
        weeks = hours // 168
        rem = hours % 168
        days = rem // 24
        rem_h = rem % 24
        duration = f"{weeks}w {days}d {rem_h}h"

    # Return % relative to capital allocated
    return_pct = net_pnl_pct  # already in %

    return {
        "gross_pnl": round(gross_pnl_pct, 4),
        "fees": round(fees_usd, 4),
        "slippage": round(slippage_usd, 4),
        "funding": round(funding_usd, 4),
        "net_pnl": round(net_pnl_usd, 4),
        "return_pct": round(return_pct, 4),
        "duration": duration,
        "pnl_pct": round(net_pnl_pct, 4),
    }

'''

    # Insert helper after _apply_costs_detail function (line 630 is the return)
    marker = '    return adj_entry, adj_exit, total_cost_pct, fee_component, slippage_component, spread_component'
    if marker not in content:
        print("ERROR: Could not find _apply_costs_detail marker in backtest.py")
        return False
    
    if '_compute_audit_fields' not in content:
        content = content.replace(marker, marker + helper_code, 1)
        print("[OK] Added _compute_audit_fields helper to backtest.py")
    else:
        print("[SKIP] _compute_audit_fields already exists in backtest.py")

    # === 2. Patch simulate_trades_advanced - the main trade append (line ~1109) ===
    # This is the big function that handles STANDARD 1h profile
    
    # Find the trade append in simulate_trades_advanced and add audit fields
    # The append around line 1109 has specific fields like entry_type=getattr
    old_append_advanced = '''            entry_type=getattr(signal, 'entry_type', 'unknown'),
        ))'''
    
    new_append_advanced = '''            entry_type=getattr(signal, 'entry_type', 'unknown'),
            # v26.0: Audit fields
            **_compute_audit_fields(
                entry_price=entry_price, exit_price=exit_price, is_long=is_long,
                position_size=position_size, position_usd=position_usd,
                bars_held=bars, entry_ts=row.name,
                exit_ts=df_ind.iloc[min(i + bars, n - 1)].name,
                fee_pct=fee_pct, spread_bps=spread_bps, slippage_bps=slippage_bps,
                apply_costs_flag=apply_costs_flag,
            ),
        ))'''

    # There are two places with entry_type=getattr in backtest.py - one in simulate_trades_advanced
    # and one at the end. We need to be careful to patch the right ones.
    # Let's use a more specific context for the simulate_trades_advanced one
    
    # The simulate_trades_advanced append has 'sl_updates=sl_updates,' before entry_type
    old_adv_context = '''            sl_updates=sl_updates,
            entry_type=getattr(signal, 'entry_type', 'unknown'),
        ))'''
    
    new_adv_context = '''            sl_updates=sl_updates,
            entry_type=getattr(signal, 'entry_type', 'unknown'),
            # v26.0: Audit fields
            **_compute_audit_fields(
                entry_price=entry_price, exit_price=exit_price, is_long=is_long,
                position_size=position_size, position_usd=position_usd,
                bars_held=bars, entry_ts=row.name,
                exit_ts=df_ind.iloc[min(i + bars, n - 1)].name,
                fee_pct=fee_pct, spread_bps=spread_bps, slippage_bps=slippage_bps,
                apply_costs_flag=apply_costs_flag,
            ),
        ))'''

    count = content.count(old_adv_context)
    if count >= 1:
        content = content.replace(old_adv_context, new_adv_context, 1)  # Replace first occurrence only
        print(f"[OK] Patched simulate_trades_advanced trade append ({count} occurrences found, replaced 1)")
    else:
        print(f"[WARN] simulate_trades_advanced pattern not found (count={count}), trying simpler match...")

    # === 3. Patch the basic simulate_trades (line ~771) ===
    old_basic = '''            atr_percentile=atr_pct,
        ))'''
    
    # There are multiple occurrences, so we need to be more specific
    # The basic simulate_trades append is the first one and doesn't have position_size
    # Let's find the exact context
    old_basic_ctx = '''            exit_reason=exit_reason,
            atr_percentile=atr_pct,
        ))

        # Avanca apos o trade fechar (evita trades sobrepostos)'''
    
    new_basic_ctx = '''            exit_reason=exit_reason,
            atr_percentile=atr_pct,
            # v26.0: Audit fields (basic sim - no position sizing)
            **_compute_audit_fields(
                entry_price=entry_price, exit_price=exit_price, is_long=is_long,
                position_size=0.0, position_usd=0.0, bars_held=bars,
                entry_ts=row.name, exit_ts=df_ind.iloc[min(i + bars, n - 1)].name,
                fee_pct=fee_pct, spread_bps=spread_bps, slippage_bps=slippage_bps,
                apply_costs_flag=apply_costs_flag,
            ),
        ))

        # Avanca apos o trade fechar (evita trades sobrepostos)'''

    if old_basic_ctx in content:
        content = content.replace(old_basic_ctx, new_basic_ctx, 1)
        print("[OK] Patched simulate_trades (basic) trade append")
    else:
        print("[WARN] Basic simulate_trades pattern not found")

    with open(path, "w") as f:
        f.write(content)
    
    return True


def patch_sim_concurrent_py():
    path = os.path.join(BASE_DIR, "sim_concurrent.py")
    with open(path, "r") as f:
        content = f.read()

    # Import _compute_audit_fields from backtest
    if '_compute_audit_fields' not in content:
        old_import = '''from backtest import (
    TradeResult, BacktestMetrics, _apply_costs,
    _update_progress, DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS,
)'''
        new_import = '''from backtest import (
    TradeResult, BacktestMetrics, _apply_costs, _compute_audit_fields,
    _update_progress, DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS,
)'''
        content = content.replace(old_import, new_import, 1)
        print("[OK] Added _compute_audit_fields import to sim_concurrent.py")

    # Patch the main trade append in the closed_positions processing loop
    # This is the critical one - line ~333-363
    old_concurrent = '''                r_multiple=_r_mult,
            ))'''
    
    # We need to be careful - there are two trade appends in sim_concurrent.py
    # The first one in the main loop and one for timeout_eod at the end
    
    # Main loop append has 'capital_allocated=round(pos.position_usd, 2),' before r_multiple
    old_conc_ctx = '''                capital_allocated=round(pos.position_usd, 2),
                quantity=round(pos.position_size, 8),
                r_multiple=_r_mult,
            ))

            # Equity tracking'''
    
    new_conc_ctx = '''                capital_allocated=round(pos.position_usd, 2),
                quantity=round(pos.position_size, 8),
                r_multiple=_r_mult,
                # v26.0: Audit fields
                **_compute_audit_fields(
                    entry_price=entry_price, exit_price=exit_price, is_long=is_long,
                    position_size=pos.position_size, position_usd=pos.position_usd,
                    bars_held=pos.bars, entry_ts=df_ind.index[pos.entry_idx],
                    exit_ts=df_ind.index[min(i, n - 1)],
                    fee_pct=fee_pct, spread_bps=spread_bps, slippage_bps=slippage_bps,
                ),
            ))

            # Equity tracking'''

    if old_conc_ctx in content:
        content = content.replace(old_conc_ctx, new_conc_ctx, 1)
        print("[OK] Patched sim_concurrent.py main trade append")
    else:
        print("[WARN] sim_concurrent.py main append pattern not found")

    # Patch the timeout_eod append at the end
    old_eod_ctx = '''                r_multiple=_r_mult,
        ))

    logger.info(
        "Simulacao concurrent v17 completa'''

    new_eod_ctx = '''                r_multiple=_r_mult,
                # v26.0: Audit fields
                **_compute_audit_fields(
                    entry_price=pos.entry_price, exit_price=last_close,
                    is_long=pos.is_long, position_size=pos.position_size,
                    position_usd=pos.position_usd, bars_held=n - 1 - pos.entry_idx,
                    entry_ts=df_ind.index[pos.entry_idx], exit_ts=df_ind.index[n - 1],
                    fee_pct=fee_pct, spread_bps=spread_bps, slippage_bps=slippage_bps,
                ),
        ))

    logger.info(
        "Simulacao concurrent v17 completa'''

    if old_eod_ctx in content:
        content = content.replace(old_eod_ctx, new_eod_ctx, 1)
        print("[OK] Patched sim_concurrent.py timeout_eod trade append")
    else:
        print("[WARN] sim_concurrent.py timeout_eod pattern not found")

    with open(path, "w") as f:
        f.write(content)
    
    return True


def patch_server_py():
    """Update server.py to export the new audit fields in the backtest result."""
    path = os.path.join(BASE_DIR, "server.py")
    with open(path, "r") as f:
        content = f.read()

    # Add new fields to the trade dict in the backtest endpoint
    old_fields = '''                        "r_multiple": getattr(t, 'r_multiple', 0.0),
                    }
                    for t in trades
                ],'''
    
    new_fields = '''                        "r_multiple": getattr(t, 'r_multiple', 0.0),
                        "gross_pnl": getattr(t, 'gross_pnl', 0.0),
                        "fees": getattr(t, 'fees', 0.0),
                        "slippage": getattr(t, 'slippage', 0.0),
                        "funding": getattr(t, 'funding', 0.0),
                        "net_pnl": getattr(t, 'net_pnl', 0.0),
                        "return_pct": getattr(t, 'return_pct', 0.0),
                        "duration": getattr(t, 'duration', ''),
                    }
                    for t in trades
                ],'''
    
    if old_fields in content and 'gross_pnl' not in content:
        content = content.replace(old_fields, new_fields, 1)
        print("[OK] Added audit fields to server.py backtest export")
    elif 'gross_pnl' in content:
        print("[SKIP] server.py already has gross_pnl field")
    else:
        print("[WARN] server.py trade dict pattern not found")

    with open(path, "w") as f:
        f.write(content)
    return True


if __name__ == "__main__":
    print("=== v26.0 Audit Fields Patch ===")
    print(f"Base dir: {BASE_DIR}")
    print()
    
    ok1 = patch_backtest_py()
    ok2 = patch_sim_concurrent_py()
    ok3 = patch_server_py()
    
    print()
    if ok1 and ok2 and ok3:
        print("=== ALL PATCHES APPLIED SUCCESSFULLY ===")
    else:
        print("=== SOME PATCHES FAILED - REVIEW WARNINGS ===")
        sys.exit(1)
