"""
apply_v16.py — Apply v16.0 multi-strategy changes.
"""
import re


def apply_strategy():
    path = "/home/z/my-project/trade-signal/strategy.py"
    with open(path) as f:
        c = f.read()

    # 1. Signal dataclass: add entry_type and max_bars after pullback_type
    c = c.replace(
        'pullback_type: str  # "fibonacci", "ema20_touch", "ema50_touch", "fib_ema_combo"\n    ema50_slope: float',
        'pullback_type: str  # "fibonacci", "ema20_touch", "ema50_touch", "none"\n    entry_type: str = "ctev_pullback"  # "ctev_pullback", "ctev_momentum", "momentum", "ranging_mr"\n    max_bars: int = 168  # max bars to hold — signal-specific\n    ema50_slope: float'
    )

    # 2. to_dict: add entry_type and max_bars
    c = c.replace(
        '"pullback_type": self.pullback_type,\n            "ema50_slope": round(self.ema50_slope, 6),',
        '"pullback_type": self.pullback_type,\n            "entry_type": self.entry_type,\n            "max_bars": self.max_bars,\n            "ema50_slope": round(self.ema50_slope, 6),'
    )

    # 3. Constants: ADX_MIN 32->25, ALLOW_TRANSITION False->True
    c = c.replace(
        'ADX_MIN = 32.0                # v14.3 FINAL: 32 (de 36)',
        'ADX_MIN = 25.0                # v16.0: 25 (de 32)'
    )
    c = c.replace(
        'ALLOW_TRANSITION = False      # v14.3: OFF',
        'ALLOW_TRANSITION = True       # v16.0: ON'
    )

    # 4. evaluate_long: make pullback optional
    c = c.replace(
        '    # v14.1: Pullback OBRIGATORIO — sem pullback = sem entrada\n    if pullback_type is None:\n        return None\n\n    # 5. RSI: Zona de pullback (adaptada ao profile)\n    if not (_rsi_l_min <= rsi <= _rsi_l_max):\n        return None\n\n    # 6. VOLUME: Soft confirmation (adaptada ao profile)\n    if _vol_confirm and (not pd.isna(volume_sma50) and volume_sma50 > 0):\n        if volume < volume_sma50 * _vol_ratio:\n            return None\n\n    # 7. ATR: Volatilidade na faixa normal (adaptada ao profile)\n    if not (_atr_pct_min <= atr_pct <= _atr_pct_max):\n        return None\n\n    # NOTA v4.2: Filtros MACD, RSI Delta e BB Width REMOVIDOS\n    # (eram redundantes/restritivos demais — ver docstring do modulo)\n    # ── Gestao de risco LONG — SL/TP adaptados ao profile ──\n    entry = close\n    stop_loss = entry - (_sl_mult * atr)\n    take_profit = entry + (_tp_mult * atr)',
        '    # v16.0: Pullback OPCIONAL — sem pullback = momentum (SL/TP mais justos)\n    if pullback_type is not None:\n        _entry_type = "ctev_pullback"\n        _sl_use = _sl_mult       # profile SL (2.8x default)\n        _tp_use = _tp_mult       # profile TP (5.5x default)\n        _max_bars_use = 168\n    else:\n        pullback_type = "none"\n        _entry_type = "ctev_momentum"\n        _sl_use = 2.0             # Momentum: SL mais justo\n        _tp_use = 3.5             # Momentum: TP mais justo\n        _max_bars_use = 72\n\n    # 5. RSI: Zona de pullback (adaptada ao profile)\n    if not (_rsi_l_min <= rsi <= _rsi_l_max):\n        return None\n\n    # 6. VOLUME: Soft confirmation (adaptada ao profile)\n    if _vol_confirm and (not pd.isna(volume_sma50) and volume_sma50 > 0):\n        if volume < volume_sma50 * _vol_ratio:\n            return None\n\n    # 7. ATR: Volatilidade na faixa normal (adaptada ao profile)\n    if not (_atr_pct_min <= atr_pct <= _atr_pct_max):\n        return None\n\n    # NOTA v4.2: Filtros MACD, RSI Delta e BB Width REMOVIDOS\n    # ── Gestao de risco LONG — SL/TP por entry type ──\n    entry = close\n    stop_loss = entry - (_sl_use * atr)\n    take_profit = entry + (_tp_use * atr)'
    )

    # 5. evaluate_long Signal: add entry_type and max_bars
    c = c.replace(
        'pullback_type=pullback_type,\n        ema50_slope=ema50_slope,\n        timestamp=ts,\n    )\n\n\ndef evaluate_short(',
        'pullback_type=pullback_type,\n        entry_type=_entry_type,\n        max_bars=_max_bars_use,\n        ema50_slope=ema50_slope,\n        timestamp=ts,\n    )\n\n\ndef evaluate_short('
    )

    # 6. evaluate_short: make pullback optional
    c = c.replace(
        '    # v14.1: Pullback OBRIGATORIO — sem pullback = sem entrada\n    if pullback_type is None:\n        return None\n\n    # 5. RSI: Zona de rally (adaptada ao profile)',
        '    # v16.0: Pullback OPCIONAL — sem pullback = momentum\n    if pullback_type is not None:\n        _entry_type_s = "ctev_pullback"\n        _sl_use_s = _sl_mult\n        _tp_use_s = _tp_mult\n        _max_bars_use_s = 168\n    else:\n        pullback_type = "none"\n        _entry_type_s = "ctev_momentum"\n        _sl_use_s = 2.0\n        _tp_use_s = 3.5\n        _max_bars_use_s = 72\n\n    # 5. RSI: Zona de rally (adaptada ao profile)'
    )

    # 7. evaluate_short SL/TP
    c = c.replace(
        '    # ── Gestao de risco SHORT — SL/TP adaptados ao profile ──\n    entry = close\n    stop_loss = entry + (_sl_mult * atr)\n    take_profit = entry - (_tp_mult * atr)',
        '    # ── Gestao de risco SHORT — SL/TP por entry type ──\n    entry = close\n    stop_loss = entry + (_sl_use_s * atr)\n    take_profit = entry - (_tp_use_s * atr)'
    )

    # 8. evaluate_short Signal: add entry_type and max_bars
    c = c.replace(
        'pullback_type=pullback_type,\n        ema50_slope=ema50_slope,\n        timestamp=ts,\n    )\n\n\ndef evaluate_signal(',
        'pullback_type=pullback_type,\n        entry_type=_entry_type_s,\n        max_bars=_max_bars_use_s,\n        ema50_slope=ema50_slope,\n        timestamp=ts,\n    )\n\n\ndef evaluate_signal('
    )

    # 9-12. Momentum/MR: add entry_type and max_bars to Signal
    # Each momentum/MR Signal ends with: fib_proximity=0.0, pullback_type="...", ema50_slope=ema50_slope,\n    #                           timestamp=ts,\n    #                       )\n    # We need to match each one uniquely by what follows

    # momentum long (followed by def evaluate_momentum_short)
    c = c.replace(
        'pullback_type="momentum", ema50_slope=ema50_slope,\n        timestamp=ts,\n    )\n\n\ndef evaluate_momentum_short(',
        'pullback_type="momentum", ema50_slope=ema50_slope,\n        entry_type="momentum", max_bars=72,\n        timestamp=ts,\n    )\n\n\ndef evaluate_momentum_short('
    )

    # momentum short (followed by def evaluate_mean_reversion_long)
    c = c.replace(
        'pullback_type="momentum", ema50_slope=ema50_slope,\n        timestamp=ts,\n    )\n\n\ndef evaluate_mean_reversion_long(',
        'pullback_type="momentum", ema50_slope=ema50_slope,\n        entry_type="momentum", max_bars=72,\n        timestamp=ts,\n    )\n\n\ndef evaluate_mean_reversion_long('
    )

    # MR long (followed by def evaluate_mean_reversion_short)
    c = c.replace(
        'pullback_type="mean_reversion", ema50_slope=ema50_slope,\n        timestamp=ts,\n    )\n\n\ndef evaluate_mean_reversion_short(',
        'pullback_type="mean_reversion", ema50_slope=ema50_slope,\n        entry_type="ranging_mr", max_bars=48,\n        timestamp=ts,\n    )\n\n\ndef evaluate_mean_reversion_short('
    )

    # MR short (last Signal in file — followed by end of file or nothing)
    c = c.replace(
        'pullback_type="mean_reversion", ema50_slope=ema50_slope,\n        timestamp=ts,\n    )',
        'pullback_type="mean_reversion", ema50_slope=ema50_slope,\n        entry_type="ranging_mr", max_bars=48,\n        timestamp=ts,\n    )'
    )

    with open(path, 'w') as f:
        f.write(c)
    print(f"  strategy.py: applied")


def apply_backtest():
    path = "/home/z/my-project/trade-signal/backtest.py"
    with open(path) as f:
        c = f.read()

    # 1. Import momentum and MR functions
    c = c.replace(
        'from strategy import (\n    SL_ATR_MULT,\n    TP_ATR_MULT,\n    Signal,\n    SignalType,\n    evaluate_long,\n    evaluate_short,\n    ATR_PCT_MIN as _ATR_PCT_MIN_STRATEGY,\n    ATR_PCT_MAX as _ATR_PCT_MAX_STRATEGY,\n    ADX_MIN as _ADX_MIN_STRATEGY,\n)',
        'from strategy import (\n    SL_ATR_MULT,\n    TP_ATR_MULT,\n    Signal,\n    SignalType,\n    evaluate_long,\n    evaluate_short,\n    evaluate_momentum_long,\n    evaluate_momentum_short,\n    evaluate_mean_reversion_long,\n    evaluate_mean_reversion_short,\n    ATR_PCT_MIN as _ATR_PCT_MIN_STRATEGY,\n    ATR_PCT_MAX as _ATR_PCT_MAX_STRATEGY,\n    ADX_MIN as _ADX_MIN_STRATEGY,\n)'
    )

    # 2. Don't skip ranging regime — only skip volatile
    c = c.replace(
        '        if regime in ("ranging", "volatile"):\n            if regime == "ranging":\n                _diag_regime_ranging += 1\n            else:\n                _diag_regime_volatile += 1\n            # v13.2: CTEV-only — skip ranging/volatile (CTEV only trades trending)\n            regime_filtered += 1\n            i += 1\n            continue',
        '        if regime == "volatile":\n            _diag_regime_volatile += 1\n            regime_filtered += 1\n            i += 1\n            continue\n        if regime == "ranging":\n            _diag_regime_ranging += 1\n            # v16.0: ranging NOT skipped — MR strategy handles it'
    )

    # 3. Multi-strategy entry chain
    c = c.replace(
        '        # v13.2: CTEV-only entry\n        signal = evaluate_long(row, profile=profile)\n        if signal is None:\n            signal = evaluate_short(row, profile=profile)\n\n        # v13.2: CTEV-only — sem fallback (momentum tinha WR 20%, sem edge)\n        if signal is None:\n            _diag_no_signal += 1\n            i += 1\n            continue',
        '        # v16.0: Multi-strategy entry (CTEV > Momentum > MR)\n        signal = evaluate_long(row, profile=profile)\n        if signal is None:\n            signal = evaluate_short(row, profile=profile)\n        if signal is None:\n            signal = evaluate_momentum_long(row)\n        if signal is None:\n            signal = evaluate_momentum_short(row)\n        if signal is None:\n            signal = evaluate_mean_reversion_long(row)\n        if signal is None:\n            signal = evaluate_mean_reversion_short(row)\n\n        if signal is None:\n            _diag_no_signal += 1\n            i += 1\n            continue'
    )

    # 4. Use signal.max_bars instead of profile max_bars
    c = c.replace(
        '        # v15.0: max_bars from profile (168 bars = 7 dias, grid-otimizado)\n        _max_bars = max_bars if profile is None else profile.max_bars_held',
        '        # v16.0: max_bars from signal (entry-type specific)\n        _max_bars = getattr(signal, "max_bars", 72)'
    )

    # 5. Lower $10 minimum to $1
    c = c.replace('if position_usd < 10.0:', 'if position_usd < 1.0:')

    # 6. Tone down anti-martingale
    c = c.replace('_RISK_REDUCTION = 0.25   # reduz 25% apos cada loss', '_RISK_REDUCTION = 0.15   # v16.0: 15% (era 25%)')
    c = c.replace('_MIN_RISK_FRACTION = 0.50  # minimo 50% do base risk', '_MIN_RISK_FRACTION = 0.65  # v16.0: 65% (era 50%)')

    with open(path, 'w') as f:
        f.write(c)
    print(f"  backtest.py: applied")


def apply_profiles():
    path = "/home/z/my-project/trade-signal/strategy_profiles.py"
    with open(path) as f:
        c = f.read()

    c = c.replace(
        'adx_min=32.0,\n    allow_transition=False,',
        'adx_min=25.0,\n    allow_transition=True,'
    )

    with open(path, 'w') as f:
        f.write(c)
    print(f"  strategy_profiles.py: applied")


if __name__ == "__main__":
    print("Applying v16.0 multi-strategy changes...")
    apply_strategy()
    apply_backtest()
    apply_profiles()
    print("Done.")
