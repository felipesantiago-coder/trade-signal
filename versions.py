"""versions.py
Registro de versoes do sistema BTC/USDT 1h Multi-Strategy.

V1-V35 = Historico de versoes Multi-Strategy (Squeeze + RSI Reversal).
Estrategia ativa: LIGA_CRYPTO (analise hierarquica multi-timeframe).

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
    # ═══════════════════════════════════════════════════════════════════
    # V6-V10: SQUEEZE-CENTRIC — simplicidade = robustez
    # Hipotese: remover CTEV (WR~22%, ruído) melhora Sharpe e consistência
    # ═══════════════════════════════════════════════════════════════════
    "V6": StrategyVersion(
        version_id="V6", label="V6-SQUEEZE_PURE",
        description=("V6 Squeeze Pure. Apenas Squeeze (WR~50%). CTEV desativado. "
                      "ADX>26, ATR 0.08-0.92. Trailing 0.6. Max 3 conc."),
        adx_min=26.0, allow_transition=True,
        rsi_long_min=44.0, rsi_long_max=66.0, rsi_short_min=34.0, rsi_short_max=56.0,
        atr_pct_min=0.08, atr_pct_max=0.92,
        strategies={
            "ctev_pullback": {"sl_mult": 1.80, "tp_mult": 5.50, "max_bars": 168, "risk_pct": 0.000, "enabled": False},
            "ctev_momentum": {"sl_mult": 1.70, "tp_mult": 7.50, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
            "squeeze_breakout": {"sl_mult": 1.80, "tp_mult": 5.00, "max_bars": 144, "risk_pct": 0.080, "enabled": True},
            "rsi_reversal": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=5.0,
        max_concurrent=3, risk_per_trade=0.008, cooldown_trigger=2, cooldown_bars=3,
        trailing_atr_mult=0.6, partial_tp_pct=0.50,
        atr_filter_min=0.08, atr_filter_max=0.92,
    ),
    "V7": StrategyVersion(
        version_id="V7", label="V7-SQUEEZE_ADX28",
        description=("V7 Squeeze ADX28. Apenas Squeeze. ADX>28 (trends mais fortes). "
                      "Trailing 0.5. ATR 0.10-0.90. Max 2 conc."),
        adx_min=28.0, allow_transition=True,
        rsi_long_min=44.0, rsi_long_max=66.0, rsi_short_min=34.0, rsi_short_max=56.0,
        atr_pct_min=0.10, atr_pct_max=0.90,
        strategies={
            "ctev_pullback": {"sl_mult": 1.80, "tp_mult": 5.50, "max_bars": 168, "risk_pct": 0.000, "enabled": False},
            "ctev_momentum": {"sl_mult": 1.70, "tp_mult": 7.50, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
            "squeeze_breakout": {"sl_mult": 1.80, "tp_mult": 5.00, "max_bars": 144, "risk_pct": 0.070, "enabled": True},
            "rsi_reversal": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=5.0,
        max_concurrent=2, risk_per_trade=0.008, cooldown_trigger=2, cooldown_bars=4,
        trailing_atr_mult=0.5, partial_tp_pct=0.50,
        atr_filter_min=0.10, atr_filter_max=0.90,
    ),
    "V8": StrategyVersion(
        version_id="V8", label="V8-SQUEEZE_ADX30",
        description=("V8 Squeeze ADX30. Apenas Squeeze. ADX>30 (trends fortes). "
                      "Trailing 0.5. ATR 0.12-0.88. Max 2 conc. Slip 3bps."),
        adx_min=30.0, allow_transition=False,
        rsi_long_min=44.0, rsi_long_max=66.0, rsi_short_min=34.0, rsi_short_max=56.0,
        atr_pct_min=0.12, atr_pct_max=0.88,
        strategies={
            "ctev_pullback": {"sl_mult": 1.80, "tp_mult": 5.50, "max_bars": 168, "risk_pct": 0.000, "enabled": False},
            "ctev_momentum": {"sl_mult": 1.70, "tp_mult": 7.50, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
            "squeeze_breakout": {"sl_mult": 2.00, "tp_mult": 4.50, "max_bars": 120, "risk_pct": 0.070, "enabled": True},
            "rsi_reversal": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=3.0,
        max_concurrent=2, risk_per_trade=0.008, cooldown_trigger=2, cooldown_bars=4,
        trailing_atr_mult=0.5, partial_tp_pct=0.50,
        atr_filter_min=0.12, atr_filter_max=0.88,
    ),
    "V9": StrategyVersion(
        version_id="V9", label="V9-SQUEEZE_RSIR_ADX26",
        description=("V9 Squeeze+RSIRev ADX26. Duas estrategias. Trailing 0.6. "
                      "ATR 0.10-0.90. Max 3 conc. RSI Rev risk 2.5%."),
        adx_min=26.0, allow_transition=True,
        rsi_long_min=44.0, rsi_long_max=66.0, rsi_short_min=34.0, rsi_short_max=56.0,
        atr_pct_min=0.10, atr_pct_max=0.90,
        strategies={
            "ctev_pullback": {"sl_mult": 1.80, "tp_mult": 5.50, "max_bars": 168, "risk_pct": 0.000, "enabled": False},
            "ctev_momentum": {"sl_mult": 1.70, "tp_mult": 7.50, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
            "squeeze_breakout": {"sl_mult": 1.80, "tp_mult": 5.00, "max_bars": 144, "risk_pct": 0.070, "enabled": True},
            "rsi_reversal": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 120, "risk_pct": 0.025, "enabled": True},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=5.0,
        max_concurrent=3, risk_per_trade=0.008, cooldown_trigger=2, cooldown_bars=4,
        trailing_atr_mult=0.6, partial_tp_pct=0.50,
        atr_filter_min=0.10, atr_filter_max=0.90,
    ),
    "V10": StrategyVersion(
        version_id="V10", label="V10-SQUEEZE_RSIR_ADX28",
        description=("V10 Squeeze+RSIRev ADX28. Duas estrategias. Trailing 0.5. "
                       "ATR 0.10-0.90. Max 2 conc. Slip 3bps."),
        adx_min=28.0, allow_transition=True,
        rsi_long_min=44.0, rsi_long_max=66.0, rsi_short_min=34.0, rsi_short_max=56.0,
        atr_pct_min=0.10, atr_pct_max=0.90,
        strategies={
            "ctev_pullback": {"sl_mult": 1.80, "tp_mult": 5.50, "max_bars": 168, "risk_pct": 0.000, "enabled": False},
            "ctev_momentum": {"sl_mult": 1.70, "tp_mult": 7.50, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
            "squeeze_breakout": {"sl_mult": 1.80, "tp_mult": 5.00, "max_bars": 144, "risk_pct": 0.060, "enabled": True},
            "rsi_reversal": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 120, "risk_pct": 0.025, "enabled": True},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=3.0,
        max_concurrent=2, risk_per_trade=0.008, cooldown_trigger=2, cooldown_bars=4,
        trailing_atr_mult=0.5, partial_tp_pct=0.50,
        atr_filter_min=0.10, atr_filter_max=0.90,
    ),
    # ═══════════════════════════════════════════════════════════════════
    # V11-V14: DERIVADAS DAS MELHORES (V2/V4) com ajustes finos
    # V2 = menor MaxDD (35%), V4 = menor overfitting (52.8)
    # ═══════════════════════════════════════════════════════════════════
    "V11": StrategyVersion(
        version_id="V11", label="V11-V4_SQUEEZE_BOOST",
        description=("V11 V4-derivative. CTEV desativado. Squeeze 10%. RSI Rev 2.5%. "
                       "Trailing 0.6. Partial TP 55%."),
        adx_min=24.0, allow_transition=True,
        rsi_long_min=44.0, rsi_long_max=68.0, rsi_short_min=32.0, rsi_short_max=56.0,
        atr_pct_min=0.08, atr_pct_max=0.92,
        strategies={
            "ctev_pullback": {"sl_mult": 1.80, "tp_mult": 6.50, "max_bars": 168, "risk_pct": 0.000, "enabled": False},
            "ctev_momentum": {"sl_mult": 1.50, "tp_mult": 9.00, "max_bars": 168, "risk_pct": 0.000, "enabled": False},
            "squeeze_breakout": {"sl_mult": 1.80, "tp_mult": 6.50, "max_bars": 144, "risk_pct": 0.100, "enabled": True},
            "rsi_reversal": {"sl_mult": 1.80, "tp_mult": 5.50, "max_bars": 120, "risk_pct": 0.025, "enabled": True},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=5.0,
        max_concurrent=3, risk_per_trade=0.008, cooldown_trigger=2, cooldown_bars=3,
        trailing_atr_mult=0.6, partial_tp_pct=0.55,
        atr_filter_min=0.08, atr_filter_max=0.92,
    ),
    "V12": StrategyVersion(
        version_id="V12", label="V12-V2_TIGHT_TRAIL",
        description=("V12 V2-derivative. Trailing 0.5, partial TP 60%. "
                       "Maior protecao de lucro. ADX>28. Slip 3bps."),
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
        trailing_atr_mult=0.5, partial_tp_pct=0.60,
        atr_filter_min=0.10, atr_filter_max=0.90,
    ),
    "V13": StrategyVersion(
        version_id="V13", label="V13-V4_HALF_RISK",
        description=("V13 V4-derivative. Todas posicoes com 50% do risco. "
                       "Menor MaxDD. Trailing 0.6. ADX>24."),
        adx_min=24.0, allow_transition=True,
        rsi_long_min=44.0, rsi_long_max=68.0, rsi_short_min=32.0, rsi_short_max=56.0,
        atr_pct_min=0.08, atr_pct_max=0.92,
        strategies={
            "ctev_pullback": {"sl_mult": 1.80, "tp_mult": 6.50, "max_bars": 168, "risk_pct": 0.000, "enabled": False},
            "ctev_momentum": {"sl_mult": 1.50, "tp_mult": 9.00, "max_bars": 168, "risk_pct": 0.000, "enabled": False},
            "squeeze_breakout": {"sl_mult": 1.80, "tp_mult": 6.50, "max_bars": 144, "risk_pct": 0.030, "enabled": True},
            "rsi_reversal": {"sl_mult": 1.80, "tp_mult": 5.50, "max_bars": 120, "risk_pct": 0.015, "enabled": True},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=5.0,
        max_concurrent=3, risk_per_trade=0.005, cooldown_trigger=2, cooldown_bars=3,
        trailing_atr_mult=0.6, partial_tp_pct=0.50,
        atr_filter_min=0.08, atr_filter_max=0.92,
    ),
    "V14": StrategyVersion(
        version_id="V14", label="V14-V2_NO_CTEV",
        description=("V14 V2-derivative. CTEV desativado. Squeeze+RSI Rev. "
                       "Trailing 0.5. ADX>28. Slip 3bps. Max 2 conc."),
        adx_min=28.0, allow_transition=True,
        rsi_long_min=42.0, rsi_long_max=64.0, rsi_short_min=36.0, rsi_short_max=58.0,
        atr_pct_min=0.10, atr_pct_max=0.90,
        strategies={
            "ctev_pullback": {"sl_mult": 1.50, "tp_mult": 6.00, "max_bars": 168, "risk_pct": 0.000, "enabled": False},
            "ctev_momentum": {"sl_mult": 1.40, "tp_mult": 8.50, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
            "squeeze_breakout": {"sl_mult": 1.80, "tp_mult": 5.50, "max_bars": 120, "risk_pct": 0.050, "enabled": True},
            "rsi_reversal": {"sl_mult": 1.80, "tp_mult": 5.50, "max_bars": 120, "risk_pct": 0.025, "enabled": True},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=3.0,
        max_concurrent=2, risk_per_trade=0.008, cooldown_trigger=2, cooldown_bars=6,
        trailing_atr_mult=0.5, partial_tp_pct=0.50,
        atr_filter_min=0.10, atr_filter_max=0.90,
    ),
    # ═══════════════════════════════════════════════════════════════════
    # V15-V17: FILTRO ATR MAIS SELETIVO
    # Hipotese: ATR mais estreito = menos sinais em volatilidade extrema = mais consistente
    # ═══════════════════════════════════════════════════════════════════
    "V15": StrategyVersion(
        version_id="V15", label="V15-WIDE_ATR_SQUEEZE",
        description=("V15 Wide ATR Squeeze. ATR 0.15-0.85. Squeeze only. "
                       "ADX>28. Trailing 0.5. Max 2 conc."),
        adx_min=28.0, allow_transition=True,
        rsi_long_min=44.0, rsi_long_max=66.0, rsi_short_min=34.0, rsi_short_max=56.0,
        atr_pct_min=0.15, atr_pct_max=0.85,
        strategies={
            "ctev_pullback": {"sl_mult": 1.80, "tp_mult": 5.50, "max_bars": 168, "risk_pct": 0.000, "enabled": False},
            "ctev_momentum": {"sl_mult": 1.70, "tp_mult": 7.50, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
            "squeeze_breakout": {"sl_mult": 1.80, "tp_mult": 5.00, "max_bars": 144, "risk_pct": 0.070, "enabled": True},
            "rsi_reversal": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=5.0,
        max_concurrent=2, risk_per_trade=0.008, cooldown_trigger=2, cooldown_bars=4,
        trailing_atr_mult=0.5, partial_tp_pct=0.50,
        atr_filter_min=0.15, atr_filter_max=0.85,
    ),
    "V16": StrategyVersion(
        version_id="V16", label="V16-TIGHT_ATR_ADX30",
        description=("V16 Tight ATR ADX30. ATR 0.15-0.85. Squeeze+RSI Rev. "
                       "ADX>30. No transition. Trailing 0.5."),
        adx_min=30.0, allow_transition=False,
        rsi_long_min=44.0, rsi_long_max=66.0, rsi_short_min=34.0, rsi_short_max=56.0,
        atr_pct_min=0.15, atr_pct_max=0.85,
        strategies={
            "ctev_pullback": {"sl_mult": 1.80, "tp_mult": 5.50, "max_bars": 168, "risk_pct": 0.000, "enabled": False},
            "ctev_momentum": {"sl_mult": 1.70, "tp_mult": 7.50, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
            "squeeze_breakout": {"sl_mult": 2.00, "tp_mult": 4.50, "max_bars": 120, "risk_pct": 0.060, "enabled": True},
            "rsi_reversal": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 120, "risk_pct": 0.025, "enabled": True},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=3.0,
        max_concurrent=2, risk_per_trade=0.008, cooldown_trigger=2, cooldown_bars=4,
        trailing_atr_mult=0.5, partial_tp_pct=0.50,
        atr_filter_min=0.15, atr_filter_max=0.85,
    ),
    "V17": StrategyVersion(
        version_id="V17", label="V17-VERY_TIGHT_ATR",
        description=("V17 Very Tight ATR. ATR 0.20-0.80. Squeeze only. "
                       "ADX>28. Ultra-seletivo. Trailing 0.4."),
        adx_min=28.0, allow_transition=True,
        rsi_long_min=44.0, rsi_long_max=66.0, rsi_short_min=34.0, rsi_short_max=56.0,
        atr_pct_min=0.20, atr_pct_max=0.80,
        strategies={
            "ctev_pullback": {"sl_mult": 1.80, "tp_mult": 5.50, "max_bars": 168, "risk_pct": 0.000, "enabled": False},
            "ctev_momentum": {"sl_mult": 1.70, "tp_mult": 7.50, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
            "squeeze_breakout": {"sl_mult": 1.80, "tp_mult": 5.00, "max_bars": 144, "risk_pct": 0.070, "enabled": True},
            "rsi_reversal": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=5.0,
        max_concurrent=2, risk_per_trade=0.008, cooldown_trigger=2, cooldown_bars=4,
        trailing_atr_mult=0.4, partial_tp_pct=0.50,
        atr_filter_min=0.20, atr_filter_max=0.80,
    ),
    # ═══════════════════════════════════════════════════════════════════
    # V18-V20: ULTRA-CONSERVADORAS
    # Hipotese: trades raros mas de altissima qualidade = maxima consistencia
    # ═══════════════════════════════════════════════════════════════════
    "V18": StrategyVersion(
        version_id="V18", label="V18-ULTRA_CONSERV",
        description=("V18 Ultra Conservative. ADX>32. Squeeze+RSI Rev. "
                       "Max 2 conc. Trailing 0.4. Partial TP 60%. ATR 0.12-0.88."),
        adx_min=32.0, allow_transition=False,
        rsi_long_min=44.0, rsi_long_max=66.0, rsi_short_min=34.0, rsi_short_max=56.0,
        atr_pct_min=0.12, atr_pct_max=0.88,
        strategies={
            "ctev_pullback": {"sl_mult": 1.80, "tp_mult": 5.50, "max_bars": 168, "risk_pct": 0.000, "enabled": False},
            "ctev_momentum": {"sl_mult": 1.70, "tp_mult": 7.50, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
            "squeeze_breakout": {"sl_mult": 2.00, "tp_mult": 4.50, "max_bars": 120, "risk_pct": 0.050, "enabled": True},
            "rsi_reversal": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 120, "risk_pct": 0.020, "enabled": True},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=3.0,
        max_concurrent=2, risk_per_trade=0.006, cooldown_trigger=2, cooldown_bars=6,
        trailing_atr_mult=0.4, partial_tp_pct=0.60,
        atr_filter_min=0.12, atr_filter_max=0.88,
    ),
    "V19": StrategyVersion(
        version_id="V19", label="V19-TREND_ONLY",
        description=("V19 Trend Only. ADX>30. No transition. Squeeze+Momentum (3%). "
                       "Trailing 0.5. Max 2 conc. ATR 0.12-0.88."),
        adx_min=30.0, allow_transition=False,
        rsi_long_min=44.0, rsi_long_max=66.0, rsi_short_min=34.0, rsi_short_max=56.0,
        atr_pct_min=0.12, atr_pct_max=0.88,
        strategies={
            "ctev_pullback": {"sl_mult": 1.80, "tp_mult": 5.50, "max_bars": 168, "risk_pct": 0.000, "enabled": False},
            "ctev_momentum": {"sl_mult": 2.20, "tp_mult": 8.00, "max_bars": 144, "risk_pct": 0.030, "enabled": True},
            "squeeze_breakout": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 144, "risk_pct": 0.060, "enabled": True},
            "rsi_reversal": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=5.0,
        max_concurrent=2, risk_per_trade=0.008, cooldown_trigger=2, cooldown_bars=6,
        trailing_atr_mult=0.5, partial_tp_pct=0.50,
        atr_filter_min=0.12, atr_filter_max=0.88,
    ),
    "V20": StrategyVersion(
        version_id="V20", label="V20-SQUEEZE_WIDE_TP",
        description=("V20 Squeeze Wide TP. Apenas Squeeze. SL 1.5x / TP 7.0x (R:R=4.67). "
                       "ADX>26. Trailing 0.6. Max 3 conc. Higher risk per squeeze."),
        adx_min=26.0, allow_transition=True,
        rsi_long_min=44.0, rsi_long_max=66.0, rsi_short_min=34.0, rsi_short_max=56.0,
        atr_pct_min=0.08, atr_pct_max=0.92,
        strategies={
            "ctev_pullback": {"sl_mult": 1.80, "tp_mult": 5.50, "max_bars": 168, "risk_pct": 0.000, "enabled": False},
            "ctev_momentum": {"sl_mult": 1.70, "tp_mult": 7.50, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
            "squeeze_breakout": {"sl_mult": 1.50, "tp_mult": 7.00, "max_bars": 168, "risk_pct": 0.100, "enabled": True},
            "rsi_reversal": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=5.0,
        max_concurrent=3, risk_per_trade=0.008, cooldown_trigger=2, cooldown_bars=3,
        trailing_atr_mult=0.6, partial_tp_pct=0.50,
        atr_filter_min=0.08, atr_filter_max=0.92,
    ),
    # ═══════════════════════════════════════════════════════════════════
    # V21-V30: MINIMALISTAS
    # Hipotese: maxima simplicidade = minima diferenca IS/OOS = overfit baixo
    # - 1 estrategia habilitada, parametros padrao, sem filtros seletivos extras
    # - R:R moderado (3:1) que funciona em todos os regimes
    # - Sem trailing/partial TP que adicionam complexidade e overfit
    # ═══════════════════════════════════════════════════════════════════
    "V21": StrategyVersion(
        version_id="V21", label="V21-MINIMAL_SQUEEZE",
        description=("V21 Minimal Squeeze. Apenas Squeeze. SL=2.0x / TP=6.0x (R:R=3:1). "
                       "Padroes todos. Sem trailing. Sem partial TP. Max 2 conc."),
        adx_min=25.0, allow_transition=True,
        rsi_long_min=44.0, rsi_long_max=66.0, rsi_short_min=34.0, rsi_short_max=56.0,
        atr_pct_min=0.08, atr_pct_max=0.95,
        strategies={
            "ctev_pullback": {"sl_mult": 2.00, "tp_mult": 6.00, "max_bars": 168, "risk_pct": 0.000, "enabled": False},
            "ctev_momentum": {"sl_mult": 2.00, "tp_mult": 6.00, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
            "squeeze_breakout": {"sl_mult": 2.00, "tp_mult": 6.00, "max_bars": 168, "risk_pct": 0.050, "enabled": True},
            "rsi_reversal": {"sl_mult": 2.00, "tp_mult": 6.00, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=5.0,
        max_concurrent=2, risk_per_trade=0.010, cooldown_trigger=2, cooldown_bars=3,
        trailing_atr_mult=1.0, partial_tp_pct=1.0,
        atr_filter_min=0.08, atr_filter_max=0.95,
    ),
    "V22": StrategyVersion(
        version_id="V22", label="V22-MINIMAL_SQUEEZE_NT",
        description=("V22 Minimal Squeeze No Transition. Sem allow_transition. "
                       "SL=2.0x / TP=6.0x. Padroes todos. Sem trailing. Max 2 conc."),
        adx_min=25.0, allow_transition=False,
        rsi_long_min=44.0, rsi_long_max=66.0, rsi_short_min=34.0, rsi_short_max=56.0,
        atr_pct_min=0.08, atr_pct_max=0.95,
        strategies={
            "ctev_pullback": {"sl_mult": 2.00, "tp_mult": 6.00, "max_bars": 168, "risk_pct": 0.000, "enabled": False},
            "ctev_momentum": {"sl_mult": 2.00, "tp_mult": 6.00, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
            "squeeze_breakout": {"sl_mult": 2.00, "tp_mult": 6.00, "max_bars": 168, "risk_pct": 0.050, "enabled": True},
            "rsi_reversal": {"sl_mult": 2.00, "tp_mult": 6.00, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=5.0,
        max_concurrent=2, risk_per_trade=0.010, cooldown_trigger=2, cooldown_bars=3,
        trailing_atr_mult=1.0, partial_tp_pct=1.0,
        atr_filter_min=0.08, atr_filter_max=0.95,
    ),
    "V23": StrategyVersion(
        version_id="V23", label="V23-MINIMAL_SQUEEZE_ADX22",
        description=("V23 Squeeze ADX>22. Limiar mais baixo = mais sinais = mais estatistica. "
                       "SL=2.0x / TP=6.0x. Padroes todos. Sem trailing."),
        adx_min=22.0, allow_transition=True,
        rsi_long_min=44.0, rsi_long_max=66.0, rsi_short_min=34.0, rsi_short_max=56.0,
        atr_pct_min=0.08, atr_pct_max=0.95,
        strategies={
            "ctev_pullback": {"sl_mult": 2.00, "tp_mult": 6.00, "max_bars": 168, "risk_pct": 0.000, "enabled": False},
            "ctev_momentum": {"sl_mult": 2.00, "tp_mult": 6.00, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
            "squeeze_breakout": {"sl_mult": 2.00, "tp_mult": 6.00, "max_bars": 168, "risk_pct": 0.050, "enabled": True},
            "rsi_reversal": {"sl_mult": 2.00, "tp_mult": 6.00, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=5.0,
        max_concurrent=2, risk_per_trade=0.010, cooldown_trigger=2, cooldown_bars=3,
        trailing_atr_mult=1.0, partial_tp_pct=1.0,
        atr_filter_min=0.08, atr_filter_max=0.95,
    ),
    "V24": StrategyVersion(
        version_id="V24", label="V24-MINIMAL_SQUEEZE_RR25",
        description=("V24 Squeeze R:R 2.5:1 (SL=2.0x / TP=5.0x). TP mais curto = mais wins = menos variabilidade. "
                       "Padroes todos. Sem trailing."),
        adx_min=25.0, allow_transition=True,
        rsi_long_min=44.0, rsi_long_max=66.0, rsi_short_min=34.0, rsi_short_max=56.0,
        atr_pct_min=0.08, atr_pct_max=0.95,
        strategies={
            "ctev_pullback": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 168, "risk_pct": 0.000, "enabled": False},
            "ctev_momentum": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
            "squeeze_breakout": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 168, "risk_pct": 0.050, "enabled": True},
            "rsi_reversal": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=5.0,
        max_concurrent=2, risk_per_trade=0.010, cooldown_trigger=2, cooldown_bars=3,
        trailing_atr_mult=1.0, partial_tp_pct=1.0,
        atr_filter_min=0.08, atr_filter_max=0.95,
    ),
    "V25": StrategyVersion(
        version_id="V25", label="V25-MINIMAL_SQUEEZE_RR20",
        description=("V25 Squeeze R:R 2:1 (SL=2.0x / TP=4.0x). TP ainda mais curto = WR mais alto. "
                       "Padroes todos. Sem trailing. Menos overfit por WR mais estavel."),
        adx_min=25.0, allow_transition=True,
        rsi_long_min=44.0, rsi_long_max=66.0, rsi_short_min=34.0, rsi_short_max=56.0,
        atr_pct_min=0.08, atr_pct_max=0.95,
        strategies={
            "ctev_pullback": {"sl_mult": 2.00, "tp_mult": 4.00, "max_bars": 168, "risk_pct": 0.000, "enabled": False},
            "ctev_momentum": {"sl_mult": 2.00, "tp_mult": 4.00, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
            "squeeze_breakout": {"sl_mult": 2.00, "tp_mult": 4.00, "max_bars": 168, "risk_pct": 0.050, "enabled": True},
            "rsi_reversal": {"sl_mult": 2.00, "tp_mult": 4.00, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=5.0,
        max_concurrent=2, risk_per_trade=0.010, cooldown_trigger=2, cooldown_bars=3,
        trailing_atr_mult=1.0, partial_tp_pct=1.0,
        atr_filter_min=0.08, atr_filter_max=0.95,
    ),
    "V26": StrategyVersion(
        version_id="V26", label="V26-MINIMAL_DUAL_SQ_RSI",
        description=("V26 Squeeze + RSI Reversal. Duas estrategias simples com mesmos parametros. "
                       "SL=2.0x / TP=5.0x. Sem trailing. Padroes todos."),
        adx_min=25.0, allow_transition=True,
        rsi_long_min=44.0, rsi_long_max=66.0, rsi_short_min=34.0, rsi_short_max=56.0,
        atr_pct_min=0.08, atr_pct_max=0.95,
        strategies={
            "ctev_pullback": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 168, "risk_pct": 0.000, "enabled": False},
            "ctev_momentum": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
            "squeeze_breakout": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 168, "risk_pct": 0.050, "enabled": True},
            "rsi_reversal": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 168, "risk_pct": 0.030, "enabled": True},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=5.0,
        max_concurrent=2, risk_per_trade=0.010, cooldown_trigger=2, cooldown_bars=3,
        trailing_atr_mult=1.0, partial_tp_pct=1.0,
        atr_filter_min=0.08, atr_filter_max=0.95,
    ),
    "V27": StrategyVersion(
        version_id="V27", label="V27-MINIMAL_SQUEEZE_1CONC",
        description=("V27 Squeeze 1 concurrent. Max 1 trade por vez. "
                       "SL=2.0x / TP=6.0x. Sem trailing. Menos sobreposicao = mais robustez."),
        adx_min=25.0, allow_transition=True,
        rsi_long_min=44.0, rsi_long_max=66.0, rsi_short_min=34.0, rsi_short_max=56.0,
        atr_pct_min=0.08, atr_pct_max=0.95,
        strategies={
            "ctev_pullback": {"sl_mult": 2.00, "tp_mult": 6.00, "max_bars": 168, "risk_pct": 0.000, "enabled": False},
            "ctev_momentum": {"sl_mult": 2.00, "tp_mult": 6.00, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
            "squeeze_breakout": {"sl_mult": 2.00, "tp_mult": 6.00, "max_bars": 168, "risk_pct": 0.050, "enabled": True},
            "rsi_reversal": {"sl_mult": 2.00, "tp_mult": 6.00, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=5.0,
        max_concurrent=1, risk_per_trade=0.010, cooldown_trigger=2, cooldown_bars=3,
        trailing_atr_mult=1.0, partial_tp_pct=1.0,
        atr_filter_min=0.08, atr_filter_max=0.95,
    ),
    "V28": StrategyVersion(
        version_id="V28", label="V28-MINIMAL_SQ_ADX22_NT",
        description=("V28 Squeeze ADX>22 No Transition. Combina V23+V22. "
                       "SL=2.0x / TP=5.0x. Padroes todos. Sem trailing."),
        adx_min=22.0, allow_transition=False,
        rsi_long_min=44.0, rsi_long_max=66.0, rsi_short_min=34.0, rsi_short_max=56.0,
        atr_pct_min=0.08, atr_pct_max=0.95,
        strategies={
            "ctev_pullback": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 168, "risk_pct": 0.000, "enabled": False},
            "ctev_momentum": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
            "squeeze_breakout": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 168, "risk_pct": 0.050, "enabled": True},
            "rsi_reversal": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=5.0,
        max_concurrent=2, risk_per_trade=0.010, cooldown_trigger=2, cooldown_bars=3,
        trailing_atr_mult=1.0, partial_tp_pct=1.0,
        atr_filter_min=0.08, atr_filter_max=0.95,
    ),
    "V29": StrategyVersion(
        version_id="V29", label="V29-MINIMAL_SQ_1CONC_ADX22",
        description=("V29 Squeeze 1 conc ADX>22. Combina V23+V27. "
                       "SL=2.0x / TP=6.0x. Max 1 trade. Sem trailing. Maxima simplicidade."),
        adx_min=22.0, allow_transition=True,
        rsi_long_min=44.0, rsi_long_max=66.0, rsi_short_min=34.0, rsi_short_max=56.0,
        atr_pct_min=0.08, atr_pct_max=0.95,
        strategies={
            "ctev_pullback": {"sl_mult": 2.00, "tp_mult": 6.00, "max_bars": 168, "risk_pct": 0.000, "enabled": False},
            "ctev_momentum": {"sl_mult": 2.00, "tp_mult": 6.00, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
            "squeeze_breakout": {"sl_mult": 2.00, "tp_mult": 6.00, "max_bars": 168, "risk_pct": 0.050, "enabled": True},
            "rsi_reversal": {"sl_mult": 2.00, "tp_mult": 6.00, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=5.0,
        max_concurrent=1, risk_per_trade=0.010, cooldown_trigger=2, cooldown_bars=3,
        trailing_atr_mult=1.0, partial_tp_pct=1.0,
        atr_filter_min=0.08, atr_filter_max=0.95,
    ),
    "V30": StrategyVersion(
        version_id="V30", label="V30-MINIMAL_SQ_RR25_1CONC",
        description=("V30 Squeeze R:R 2.5:1 + 1 conc. Combina V24+V27. "
                       "SL=2.0x / TP=5.0x. Max 1 trade. Sem trailing. Equilibrio WR/R:R."),
        adx_min=25.0, allow_transition=True,
        rsi_long_min=44.0, rsi_long_max=66.0, rsi_short_min=34.0, rsi_short_max=56.0,
        atr_pct_min=0.08, atr_pct_max=0.95,
        strategies={
            "ctev_pullback": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 168, "risk_pct": 0.000, "enabled": False},
            "ctev_momentum": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
            "squeeze_breakout": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 168, "risk_pct": 0.050, "enabled": True},
            "rsi_reversal": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=5.0,
        max_concurrent=1, risk_per_trade=0.010, cooldown_trigger=2, cooldown_bars=3,
        trailing_atr_mult=1.0, partial_tp_pct=1.0,
        atr_filter_min=0.08, atr_filter_max=0.95,
    ),
    # ═══════════════════════════════════════════════════════════════════
    # V31-V35: OTIMIZACOES FINAIS DE V16
    # V16 e a melhor candidata (Sharpe 1.78, DD 39.7%, Consistency 67%)
    # Hipotese: ajustes minimos em trailing/partial TP/risco podem reduzir
    # overfit sem comprometer Sharpe e Consistency
    # ═══════════════════════════════════════════════════════════════════
    "V31": StrategyVersion(
        version_id="V31", label="V31-V16_TRAIL65",
        description=("V31 = V16 + trailing 0.65 (era 0.5). Trailing mais largo"
                       "captura mais das tendencias fortes sem sacrificar amortecimento."),
        adx_min=30.0, allow_transition=False,
        rsi_long_min=44.0, rsi_long_max=66.0, rsi_short_min=34.0, rsi_short_max=56.0,
        atr_pct_min=0.15, atr_pct_max=0.85,
        strategies={
            "ctev_pullback": {"sl_mult": 2.00, "tp_mult": 5.50, "max_bars": 168, "risk_pct": 0.000, "enabled": False},
            "ctev_momentum": {"sl_mult": 2.00, "tp_mult": 7.50, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
            "squeeze_breakout": {"sl_mult": 2.00, "tp_mult": 4.50, "max_bars": 120, "risk_pct": 0.060, "enabled": True},
            "rsi_reversal": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 120, "risk_pct": 0.025, "enabled": True},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=3.0,
        max_concurrent=2, risk_per_trade=0.008, cooldown_trigger=2, cooldown_bars=4,
        trailing_atr_mult=0.65, partial_tp_pct=0.50,
        atr_filter_min=0.15, atr_filter_max=0.85,
    ),
    "V32": StrategyVersion(
        version_id="V32", label="V32-V16_PARTIAL55",
        description=("V32 = V16 + partial TP 55% (era 50%). Fecha mais cedo em"
                       "victorias parciais, reduzindo dependencia de TP exato."),
        adx_min=30.0, allow_transition=False,
        rsi_long_min=44.0, rsi_long_max=66.0, rsi_short_min=34.0, rsi_short_max=56.0,
        atr_pct_min=0.15, atr_pct_max=0.85,
        strategies={
            "ctev_pullback": {"sl_mult": 2.00, "tp_mult": 5.50, "max_bars": 168, "risk_pct": 0.000, "enabled": False},
            "ctev_momentum": {"sl_mult": 2.00, "tp_mult": 7.50, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
            "squeeze_breakout": {"sl_mult": 2.00, "tp_mult": 4.50, "max_bars": 120, "risk_pct": 0.060, "enabled": True},
            "rsi_reversal": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 120, "risk_pct": 0.025, "enabled": True},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=3.0,
        max_concurrent=2, risk_per_trade=0.008, cooldown_trigger=2, cooldown_bars=4,
        trailing_atr_mult=0.5, partial_tp_pct=0.55,
        atr_filter_min=0.15, atr_filter_max=0.85,
    ),
    "V33": StrategyVersion(
        version_id="V33", label="V33-V16_TRAIL65_PART55",
        description=("V33 = V16 + trailing 0.65 + partial TP 55%. Combina V31+V32."
                       "Maior amortecimento de PnL em ambas direcoes."),
        adx_min=30.0, allow_transition=False,
        rsi_long_min=44.0, rsi_long_max=66.0, rsi_short_min=34.0, rsi_short_max=56.0,
        atr_pct_min=0.15, atr_pct_max=0.85,
        strategies={
            "ctev_pullback": {"sl_mult": 2.00, "tp_mult": 5.50, "max_bars": 168, "risk_pct": 0.000, "enabled": False},
            "ctev_momentum": {"sl_mult": 2.00, "tp_mult": 7.50, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
            "squeeze_breakout": {"sl_mult": 2.00, "tp_mult": 4.50, "max_bars": 120, "risk_pct": 0.060, "enabled": True},
            "rsi_reversal": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 120, "risk_pct": 0.025, "enabled": True},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=3.0,
        max_concurrent=2, risk_per_trade=0.008, cooldown_trigger=2, cooldown_bars=4,
        trailing_atr_mult=0.65, partial_tp_pct=0.55,
        atr_filter_min=0.15, atr_filter_max=0.85,
    ),
    "V34": StrategyVersion(
        version_id="V34", label="V34-V16_RISK6_SQ",
        description=("V34 = V16 + squeeze risk 5% (era 6%). Menos exposicao"
                       "por trade squeeze = PnL mais estavel entre janelas."),
        adx_min=30.0, allow_transition=False,
        rsi_long_min=44.0, rsi_long_max=66.0, rsi_short_min=34.0, rsi_short_max=56.0,
        atr_pct_min=0.15, atr_pct_max=0.85,
        strategies={
            "ctev_pullback": {"sl_mult": 2.00, "tp_mult": 5.50, "max_bars": 168, "risk_pct": 0.000, "enabled": False},
            "ctev_momentum": {"sl_mult": 2.00, "tp_mult": 7.50, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
            "squeeze_breakout": {"sl_mult": 2.00, "tp_mult": 4.50, "max_bars": 120, "risk_pct": 0.050, "enabled": True},
            "rsi_reversal": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 120, "risk_pct": 0.025, "enabled": True},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=3.0,
        max_concurrent=2, risk_per_trade=0.008, cooldown_trigger=2, cooldown_bars=4,
        trailing_atr_mult=0.5, partial_tp_pct=0.50,
        atr_filter_min=0.15, atr_filter_max=0.85,
    ),
    "V35": StrategyVersion(
        version_id="V35", label="V35-V16_ADX28_TRAIL55",
        description=("V35 = V16 + ADX>28 (era 30) + trailing 0.55. Limiar ADX"
                       "mais baixo = mais sinais = mais estatistica = menos variancia."),
        adx_min=28.0, allow_transition=False,
        rsi_long_min=44.0, rsi_long_max=66.0, rsi_short_min=34.0, rsi_short_max=56.0,
        atr_pct_min=0.15, atr_pct_max=0.85,
        strategies={
            "ctev_pullback": {"sl_mult": 2.00, "tp_mult": 5.50, "max_bars": 168, "risk_pct": 0.000, "enabled": False},
            "ctev_momentum": {"sl_mult": 2.00, "tp_mult": 7.50, "max_bars": 120, "risk_pct": 0.000, "enabled": False},
            "squeeze_breakout": {"sl_mult": 2.00, "tp_mult": 4.50, "max_bars": 120, "risk_pct": 0.060, "enabled": True},
            "rsi_reversal": {"sl_mult": 2.00, "tp_mult": 5.00, "max_bars": 120, "risk_pct": 0.025, "enabled": True},
        },
        fee_pct=0.016, spread_bps=2.0, slippage_bps=3.0,
        max_concurrent=2, risk_per_trade=0.008, cooldown_trigger=2, cooldown_bars=4,
        trailing_atr_mult=0.55, partial_tp_pct=0.50,
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
