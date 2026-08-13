"""versions.py
Registro de versoes do sistema BTC/USDT 1h Multi-Strategy.

V1 = BASELINE (v25.0 current)
V2 = CONSERVATIVE (tighter SL, wider R:R, lower leverage)
V3 = MODERATE (slightly wider ATR filter, stricter ADX)
V4 = AGGRESSIVE_RR (wider TP for trend strategies, reduced position sizes)
V5 = LOW_FREQ_QUALITY (higher ADX, BBWP squeeze focus only)

Walk-Forward OOS validation per version.
Principio: AUDITABILIDADE > ROBUSTEZ > RISCO > CONSISTENCIA > RETORNO
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import logging

logger = logging.getLogger("ctev.versions")


@dataclass(frozen=True)
class StrategyVersion:
    version_id: str
    label: str
    description: str
    adx_min: float = 25.0
    allow_transition: bool = True
    rsi_long_min: float = 44.0
    rsi_long_max: float = 66.0
    rsi_short_min: float = 34.0
    rsi_short_max: float = 56.0
    atr_pct_min: float = 0.08
    atr_pct_max: float = 0.95
    strategies: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    fee_pct: float = 0.016
    spread_bps: float = 2.0
    slippage_bps: float = 5.0
    max_concurrent: int = 3
    risk_per_trade: float = 0.01
    cooldown_trigger: int = 2
    cooldown_bars: int = 3
    trailing_atr_mult: float = 0.7
    partial_tp_pct: float = 0.50
    atr_filter_min: float = 0.08
    atr_filter_max: float = 0.95


VERSIONS = {
    "V1": StrategyVersion(
        version_id="V1", label="V1-BASELINE",
        description=("V1 Baseline (v25.0). 4 estrategias. 3 concurrent. ADX>25. "
                      "Momentum SL=1.7x/TP=7.5x. Squeeze risk=8%. Fee=0.016%+2bps+5bps."),
        adx_min=25.0, allow_transition=True,
        rsi_long_min=44.0, rsi_long_max=66.0, rsi_short_min=34.0, rsi_short_max=56.0,
        atr_pct_min=0.08, atr_pct_max=0.95,
        strategies={
            "ctev_pullback": {"sl_mult": 1.80, "tp_mult": 5.50, "max_bars": 168, "risk_pct": 0.0005, "enabled": True},
            "ctev_momentum": {"sl_mult": 1.70, "tp_mult": 7.50, "max_bars": 120, "risk_pct": 0.0005, "enabled": True},
            "squeeze_breakout": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 120, "risk_pct": 0.080, "enabled": True},
            "rsi_reversal": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 120, "risk_pct": 0.035, "enabled": True},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=5.0,
        max_concurrent=3, risk_per_trade=0.01, cooldown_trigger=2, cooldown_bars=3,
        trailing_atr_mult=0.7, partial_tp_pct=0.50,
        atr_filter_min=0.08, atr_filter_max=0.95,
    ),
    "V2": StrategyVersion(
        version_id="V2", label="V2-CONSERVATIVE",
        description=("V2 Conservative. Tighter SL (1.5x pullback, 1.4x momentum), wider TP. "
                      "ADX>28. Squeeze risk=5%. Cooldown 2SL/6bars. Slip=3bps."),
        adx_min=28.0, allow_transition=True,
        rsi_long_min=42.0, rsi_long_max=64.0, rsi_short_min=36.0, rsi_short_max=58.0,
        atr_pct_min=0.10, atr_pct_max=0.90,
        strategies={
            "ctev_pullback": {"sl_mult": 1.50, "tp_mult": 6.00, "max_bars": 168, "risk_pct": 0.0005, "enabled": True},
            "ctev_momentum": {"sl_mult": 1.40, "tp_mult": 8.50, "max_bars": 120, "risk_pct": 0.0005, "enabled": True},
            "squeeze_breakout": {"sl_mult": 1.80, "tp_mult": 5.50, "max_bars": 120, "risk_pct": 0.050, "enabled": True},
            "rsi_reversal": {"sl_mult": 1.80, "tp_mult": 5.50, "max_bars": 120, "risk_pct": 0.025, "enabled": True},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=3.0,
        max_concurrent=3, risk_per_trade=0.008, cooldown_trigger=2, cooldown_bars=6,
        trailing_atr_mult=0.8, partial_tp_pct=0.50,
        atr_filter_min=0.10, atr_filter_max=0.90,
    ),
    "V3": StrategyVersion(
        version_id="V3", label="V3-MODERATE",
        description=("V3 Moderate. ADX>27, ATR 0.12-0.88. Squeeze TP=6.0x. "
                      "RSI Rev risk=2.5%. Cooldown 2SL/4bars."),
        adx_min=27.0, allow_transition=True,
        rsi_long_min=43.0, rsi_long_max=65.0, rsi_short_min=35.0, rsi_short_max=57.0,
        atr_pct_min=0.12, atr_pct_max=0.88,
        strategies={
            "ctev_pullback": {"sl_mult": 1.70, "tp_mult": 5.50, "max_bars": 168, "risk_pct": 0.0005, "enabled": True},
            "ctev_momentum": {"sl_mult": 1.60, "tp_mult": 7.50, "max_bars": 120, "risk_pct": 0.0005, "enabled": True},
            "squeeze_breakout": {"sl_mult": 2.00, "tp_mult": 6.00, "max_bars": 120, "risk_pct": 0.060, "enabled": True},
            "rsi_reversal": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 120, "risk_pct": 0.025, "enabled": True},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=5.0,
        max_concurrent=3, risk_per_trade=0.01, cooldown_trigger=2, cooldown_bars=4,
        trailing_atr_mult=0.7, partial_tp_pct=0.50,
        atr_filter_min=0.12, atr_filter_max=0.88,
    ),
    "V4": StrategyVersion(
        version_id="V4", label="V4-AGGRESSIVE_RR",
        description=("V4 Aggressive R:R. Momentum TP=9.0x (R:R=5.29). ADX>24. "
                      "Max 4 concurrent. Reduced position sizes."),
        adx_min=24.0, allow_transition=True,
        rsi_long_min=44.0, rsi_long_max=68.0, rsi_short_min=32.0, rsi_short_max=56.0,
        atr_pct_min=0.08, atr_pct_max=0.92,
        strategies={
            "ctev_pullback": {"sl_mult": 1.80, "tp_mult": 6.50, "max_bars": 168, "risk_pct": 0.0005, "enabled": True},
            "ctev_momentum": {"sl_mult": 1.50, "tp_mult": 9.00, "max_bars": 168, "risk_pct": 0.0004, "enabled": True},
            "squeeze_breakout": {"sl_mult": 1.80, "tp_mult": 6.50, "max_bars": 144, "risk_pct": 0.060, "enabled": True},
            "rsi_reversal": {"sl_mult": 1.80, "tp_mult": 5.50, "max_bars": 120, "risk_pct": 0.030, "enabled": True},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=5.0,
        max_concurrent=4, risk_per_trade=0.008, cooldown_trigger=2, cooldown_bars=3,
        trailing_atr_mult=0.8, partial_tp_pct=0.50,
        atr_filter_min=0.08, atr_filter_max=0.92,
    ),
    "V5": StrategyVersion(
        version_id="V5", label="V5-LOW_FREQ_QUALITY",
        description=("V5 Low-Frequency Quality. ADX>30, ATR 0.15-0.85. "
                      "Only Squeeze+Momentum active. Max 2 concurrent."),
        adx_min=30.0, allow_transition=False,
        rsi_long_min=42.0, rsi_long_max=62.0, rsi_short_min=38.0, rsi_short_max=58.0,
        atr_pct_min=0.15, atr_pct_max=0.85,
        strategies={
            "ctev_pullback": {"sl_mult": 1.80, "tp_mult": 5.50, "max_bars": 168, "risk_pct": 0.000, "enabled": False},
            "ctev_momentum": {"sl_mult": 2.20, "tp_mult": 8.00, "max_bars": 144, "risk_pct": 0.040, "enabled": True},
            "squeeze_breakout": {"sl_mult": 2.50, "tp_mult": 6.00, "max_bars": 144, "risk_pct": 0.100, "enabled": True},
            "rsi_reversal": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=5.0,
        max_concurrent=2, risk_per_trade=0.008, cooldown_trigger=2, cooldown_bars=6,
        trailing_atr_mult=0.8, partial_tp_pct=0.50,
        atr_filter_min=0.15, atr_filter_max=0.85,
    ),
}


def get_version(version_id: str) -> StrategyVersion:
    if version_id in VERSIONS:
        return VERSIONS[version_id]
    raise ValueError(f"Versao desconhecida: {version_id}. Disponiveis: {list(VERSIONS.keys())}")


def list_versions() -> Dict[str, StrategyVersion]:
    return dict(VERSIONS)


def apply_version_to_strategy(version: StrategyVersion) -> Dict[str, Any]:
    import strategy as strat
    orig = {}
    for k, v in [
        ('ADX_MIN', 'adx_min'), ('ALLOW_TRANSITION', 'allow_transition'),
        ('RSI_LONG_MIN', 'rsi_long_min'), ('RSI_LONG_MAX', 'rsi_long_max'),
        ('RSI_SHORT_MIN', 'rsi_short_min'), ('RSI_SHORT_MAX', 'rsi_short_max'),
        ('ATR_PCT_MIN', 'atr_pct_min'), ('ATR_PCT_MAX', 'atr_pct_max'),
    ]:
        orig[k] = getattr(strat, k, None)
        setattr(strat, k, getattr(version, v))

    strats_map = {
        "ctev_pullback": [('SL_ATR_MULT', 'sl_mult'), ('TP_ATR_MULT', 'tp_mult')],
        "ctev_momentum": [('MOMENTUM_SL_ATR_MULT', 'sl_mult'), ('MOMENTUM_TP_ATR_MULT', 'tp_mult'), ('MOMENTUM_MAX_BARS', 'max_bars')],
        "squeeze_breakout": [('SQUEEZE_SL_ATR_MULT', 'sl_mult'), ('SQUEEZE_TP_ATR_MULT', 'tp_mult'), ('SQUEEZE_MAX_BARS', 'max_bars')],
        "rsi_reversal": [('RSI_REV_SL_ATR_MULT', 'sl_mult'), ('RSI_REV_TP_ATR_MULT', 'tp_mult'), ('RSI_REV_MAX_BARS', 'max_bars')],
    }
    for strat_name, param_map in strats_map.items():
        if strat_name in version.strategies:
            sv = version.strategies[strat_name]
            for strat_const, sv_key in param_map:
                if sv_key in sv:
                    orig[strat_const] = getattr(strat, strat_const, None)
                    setattr(strat, strat_const, sv[sv_key])

    logger.info("Version %s applied: ADX>=%.0f, ATR=[%.2f,%.2f]", version.version_id, version.adx_min, version.atr_pct_min, version.atr_pct_max)
    return orig


def restore_strategy(orig: Dict[str, Any]) -> None:
    import strategy as strat
    for k, v in orig.items():
        if v is not None:
            setattr(strat, k, v)


def apply_version_to_sim_concurrent(version: StrategyVersion) -> Dict[str, Any]:
    import sim_concurrent as sim
    orig = {}
    for k, v in [('MAX_CONCURRENT', 'max_concurrent'), ('RISK_PER_TRADE', 'risk_per_trade')]:
        orig[k] = getattr(sim, k)
        setattr(sim, k, getattr(version, v))
    orig["ENTRY_RISK"] = dict(sim.ENTRY_RISK)
    for strat_name, params in version.strategies.items():
        if strat_name in sim.ENTRY_RISK:
            sim.ENTRY_RISK[strat_name] = params.get("risk_pct", 0.0)
    sim._COOLDOWN_TRIGGER = version.cooldown_trigger
    logger.info("Sim concurrent: max_conc=%d, risk=%.3f", version.max_concurrent, version.risk_per_trade)
    return orig


def restore_sim_concurrent(orig: Dict[str, Any]) -> None:
    import sim_concurrent as sim
    for k, v in orig.items():
        if v is not None:
            setattr(sim, k, v)
