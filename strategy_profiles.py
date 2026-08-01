"""
strategy_profiles.py
-------------------
Perfis de estrategia CTEV otimizados por grupo de timeframe.

Cada timeframe tem caracteristicas unicas de ruido, duracao de tendencia
e volatilidade relativa. Os parametros do 1h (v5.0) NAO sao otimos para
outros timeframes — usar os mesmos thresholds gera sinais de baixa qualidade.

Principio de design:
  - SCALP (1m/3m/5m): Muito ruido → indicadores mais rapidos, SL justo,
    RSI alargado, volume obrigatorio para filtrar falsos sinais.
  - INTRADAY (15m/30m): Balance entre ruido e sinal → similar ao 1h
    com ajustes finos de sensibilidade.
  - STANDARD (1h): Estrategia original CTEV v5.0 otimizada via grid search.
  - SWING (2h/4h): Tendencias mais limpas e longas → filtros mais
    restritivos (qualidade > quantidade), SL/TP mais largos.
  - POSITION (1d): Tendencias duram dias/semanas → SL largo, TP agressivo,
    sem transition (apenas sinais de alta qualidade).

Cada perfil ajusta:
  1. Filtros de entrada: ADX minimo, RSI zones, EMA slope, fib tolerance
  2. Gestao de risco: SL/TP ATR multipliers, max bars per trade
  3. Volume: confirmacao obrigatoria ou nao
  4. Volatilidade: faixa de ATR percentile aceitavel

Uso:
    from strategy_profiles import get_profile
    profile = get_profile("15m")  # retorna StrategyProfile para 15m
    signal = evaluate_long(row, profile=profile)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class StrategyProfile:
    """
    Perfil completo de parametros da estrategia CTEV para um timeframe.

    Atributos:
        name: Nome descritivo do grupo
        timeframes: Lista de timeframes que usam este perfil
        description: Rationale do perfil

        # Filtros de entrada
        adx_min: ADX minimo para regime trending
        allow_transition: Se True, aceita regime 'transition'
        rsi_long_min/max: Zona RSI para entrada LONG
        rsi_short_min/max: Zona RSI para entrada SHORT
        fib_tolerance_pct: Tolerancia % para proximidade Fibonacci
        ema50_slope_min: Slope minimo da EMA50 (relaxado < 0)
        volume_confirm: Se True, exige volume > SMA * ratio
        volume_sma_ratio: Ratio minimo volume/SMA quando confirm=True

        # Filtros de volatilidade
        atr_pct_min/max: Faixa aceitavel de ATR percentile

        # Gestao de risco
        sl_atr_mult: Stop-loss = entry - mult * ATR
        tp_atr_mult: Take-profit = entry + mult * ATR
        max_bars_held: Maximo de candles para segurar posicao
    """
    name: str
    timeframes: tuple
    description: str

    # Filtros de entrada
    adx_min: float = 30.0
    allow_transition: bool = True
    rsi_long_min: float = 28.0
    rsi_long_max: float = 48.0
    rsi_short_min: float = 55.0
    rsi_short_max: float = 75.0
    fib_tolerance_pct: float = 0.025
    ema50_slope_min: float = -1.0
    volume_confirm: bool = False
    volume_sma_ratio: float = 0.30

    # Filtros de volatilidade
    atr_pct_min: float = 0.10
    atr_pct_max: float = 0.90

    # Gestao de risco
    sl_atr_mult: float = 1.50
    tp_atr_mult: float = 3.50
    max_bars_held: int = 72

    @property
    def rr_ratio(self) -> float:
        """Risk:Reward ratio implicito no perfil."""
        return round(self.tp_atr_mult / self.sl_atr_mult, 2)

    def summary(self) -> str:
        return (
            f"{self.name} | ADX>={self.adx_min:.0f} "
            f"RSI_L[{self.rsi_long_min:.0f}-{self.rsi_long_max:.0f}] "
            f"RSI_S[{self.rsi_short_min:.0f}-{self.rsi_short_max:.0f}] "
            f"SL={self.sl_atr_mult:.1f}x TP={self.tp_atr_mult:.1f}x "
            f"R:R={self.rr_ratio}:1 "
            f"FIB_tol={self.fib_tolerance_pct*100:.1f}% "
            f"Vol={'ON' if self.volume_confirm else 'OFF'}"
        )


# ==================================================================
# PERFIS POR GRUPO DE TIMEFRAME
# ==================================================================

PROFILE_SCALP = StrategyProfile(
    name="SCALP",
    timeframes=("1m", "3m", "5m"),
    description=(
        "[NAO VALIDADO] Timeframes sub-15min. Backtest 90d: 48 trades, "
        "WR 22.9%, PF 0.06. Custos (~0.24% por trade) consomem toda "
        "a edge porque os movimentos em ATR sao muito pequenos. "
        "Recomendacao: usar estrategia de mean-reversion (BB bounce) "
        "ao inves de trend-following para estes timeframes."
    ),
    # Filtros de entrada
    adx_min=25.0,          # Mais restritivo para filtrar ruido
    allow_transition=False,  # Sem transition — apenas tendencias reais
    rsi_long_min=30.0,     # Mais justo — evitar entrada no ruido
    rsi_long_max=50.0,
    rsi_short_min=50.0,
    rsi_short_max=70.0,
    fib_tolerance_pct=0.035,  # 3.5%
    ema50_slope_min=-0.5,   # Requer slope real
    volume_confirm=True,    # Obrigatorio no ruido
    volume_sma_ratio=0.60,  # 60% da SMA — filtro strict

    # Volatilidade
    atr_pct_min=0.30,       # Apenas volatilidade significativa
    atr_pct_max=0.70,       # Evita extremos

    # Gestao de risco — SL mais largo para sobreviver ruido
    sl_atr_mult=1.5,        # SL mais largo — 1.5x ATR
    tp_atr_mult=3.5,        # TP 3.5x para compensar custos
    max_bars_held=48,
)

PROFILE_INTRADAY = StrategyProfile(
    name="INTRADAY",
    timeframes=("15m", "30m"),
    description=(
        "[VALIDADO v8] EMA Cross strategy para 15m. Diferente do CTEV trend-following, "
        "usa cruze EMA20/50 como sinal primario com filtro RSI delta (momentum). "
        "Backtest 180d (16862 candles): 150 trades, WR 50.0%, PF 1.13, "
        "PnL +5.97%, DD 3.53%. Supera B&H (-5.36%) em +11.33pp. "
        "Walk-Forward 4 passos: train +1.28%, test +0.50%. "
        "IMPORTANTE: Lucrativo SOMENTE com limit orders (maker fee 0.016% + spread 2bps + slip 2bps). "
        "Com custos de market order (0.325%/lado), a estrategia nao e lucrativa. "
        "Otimizado via optimize_15m_v8.py (grid 654 combos, 4 estrategias testadas). "
        "30m nao validado — usa mesmos params por extrapolacao."
    ),
    # NOTA: Esta estrategia usa logica diferente (EMA Cross) — os parametros
    # abaixo sao usados apenas para SL/TP. A logica de sinal real precisa ser
    # implementada via evaluate_long/evaluate_short ou estrategia separada.
    adx_min=15.0,           # ADX relaxado — foco em momentum, nao forca de tendencia
    allow_transition=True,   # Permite entradas em transicao de regime
    rsi_long_min=30.0,
    rsi_long_max=80.0,     # RSI largo para longs no cruzamento
    rsi_short_min=20.0,
    rsi_short_max=70.0,     # RSI largo para shorts no cruzamento
    fib_tolerance_pct=0.025,
    ema50_slope_min=-1.0,
    volume_confirm=False,
    volume_sma_ratio=0.30,

    # Volatilidade
    atr_pct_min=0.10,
    atr_pct_max=0.90,

    # Gestao de risco — Otimizado v8 EMA Cross
    sl_atr_mult=2.00,       # 2.0x ATR — mais espaco para ruido
    tp_atr_mult=2.50,       # 2.5x ATR — R:R 1.25:1 (WR compensa)
    max_bars_held=48,        # 48 candles = 12h (15m)
)

PROFILE_STANDARD = StrategyProfile(
    name="STANDARD",
    timeframes=("1h",),
    description=(
        "[VALIDADO P2+WF] Tempo grafico de 1h. Otimizado via P2 grid search "
        "ao redor de SL 2.5x/TP 12.0x + Walk-Forward 6 janelas. "
        "Backtest 730d (17408 candles): 39 trades, WR 35.9%, PF 1.53, "
        "PnL +20.37%, DD 13.35%, Sharpe 2.95, R:R 4.0:1. "
        "Supera Buy & Hold (+18.60%) com +20.37%. WF OOS: +12.49% vs B&H -30.79%, "
        "4/6 janelas superam B&H. SL/TP mais frequente nos WF steps: 2.25/8.0 (6/6)."
    ),
    # Filtros de entrada — mantidos do P1 v5.0
    adx_min=30.0,
    allow_transition=True,
    rsi_long_min=28.0,
    rsi_long_max=48.0,
    rsi_short_min=55.0,
    rsi_short_max=75.0,
    fib_tolerance_pct=0.025,
    ema50_slope_min=-1.0,
    volume_confirm=False,
    volume_sma_ratio=0.30,

    # Volatilidade
    atr_pct_min=0.10,
    atr_pct_max=0.90,

    # Gestao de risco — Otimizado P2+WF: SL 2.12x ATR / TP 8.5x ATR (R:R 4.0:1)
    sl_atr_mult=2.12,
    tp_atr_mult=8.50,
    max_bars_held=72,        # 3 dias
)

PROFILE_SWING = StrategyProfile(
    name="SWING",
    timeframes=("2h", "4h"),
    description=(
        "[NAO VALIDADO] Timeframes de 2-4h. Backtest 730d: 21 trades, "
        "WR 28.6%, PF 0.60. Teste com params identicos ao 1h: 11 trades, "
        "WR 18.2%, PF 0.35. A edge nao transfere para TF maiores — "
        "tendencias sao mais limpas mas poucas e a taxa de acerto cai. "
        "Perfil disponivel para experimentacao e otimizacao futura."
    ),
    # Filtros de entrada — proximo do STANDARD
    adx_min=30.0,           # Igual ao 1h — qualidade > quantidade
    allow_transition=True,
    rsi_long_min=26.0,     # Ligeiramente alargado vs 1h (28)
    rsi_long_max=50.0,     # +2 vs 1h (48)
    rsi_short_min=53.0,    # -2 vs 1h (55)
    rsi_short_max=75.0,
    fib_tolerance_pct=0.030,  # 3% — mais tolerante
    ema50_slope_min=-1.0,   # Igual ao 1h
    volume_confirm=False,
    volume_sma_ratio=0.30,

    # Volatilidade
    atr_pct_min=0.10,
    atr_pct_max=0.90,

    # Gestao de risco — R:R maior para swings
    sl_atr_mult=1.8,        # Um pouco mais largo que 1h
    tp_atr_mult=4.5,        # TP agressivo — movimentos maiores
    max_bars_held=60,        # 60 candles: 5 dias(2h), 10 dias(4h)
)

PROFILE_POSITION = StrategyProfile(
    name="POSITION",
    timeframes=("1d",),
    description=(
        "[NAO VALIDADO] Timeframe diario. Backtest 730d: apenas 2 trades "
        "(amostra insuficiente). O CTEV gera muito poucos sinais em 1d "
        "porque requer multiplas condicoes de regime+pullback+RSI. "
        "Necessita estrategia diferente (ex: breakout de range diario)."
    ),
    # Filtros de entrada
    adx_min=20.0,           # Tendencias desenvolvem lentamente
    allow_transition=True,  # Necessario para gerar sinais com poucos candles
    rsi_long_min=28.0,     # Igual ao 1h
    rsi_long_max=48.0,     # Igual ao 1h
    rsi_short_min=55.0,    # Igual ao 1h
    rsi_short_max=75.0,    # Igual ao 1h
    fib_tolerance_pct=0.030,  # 3% — mais tolerante (poucos niveis fib em 1d)
    ema50_slope_min=-1.0,   # Igual ao 1h
    volume_confirm=False,   # Desabilitado — muito restritivo em 1d
    volume_sma_ratio=0.40,

    # Volatilidade
    atr_pct_min=0.05,
    atr_pct_max=0.95,

    # Gestao de risco
    sl_atr_mult=2.0,        # SL largo — sobrevive ruido diario
    tp_atr_mult=5.0,        # TP agressivo — captura movimentos de varios dias
    max_bars_held=30,        # 30 dias
)


# ==================================================================
# REGISTRO GLOBAL
# ==================================================================

_ALL_PROFILES: Dict[str, StrategyProfile] = {
    p.name: p for p in [
        PROFILE_SCALP,
        PROFILE_INTRADAY,
        PROFILE_STANDARD,
        PROFILE_SWING,
        PROFILE_POSITION,
    ]
}

# Mapa timeframe -> profile name
_TF_TO_PROFILE: Dict[str, str] = {}
for _profile in _ALL_PROFILES.values():
    for _tf in _profile.timeframes:
        _TF_TO_PROFILE[_tf] = _profile.name


def get_profile(timeframe: str) -> StrategyProfile:
    """
    Retorna o StrategyProfile adequado para o timeframe dado.

    Examples:
        >>> get_profile("15m").name
        'INTRADAY'
        >>> get_profile("4h").name
        'SWING'
        >>> get_profile("1h").name
        'STANDARD'
    """
    if timeframe in _TF_TO_PROFILE:
        return _ALL_PROFILES[_TF_TO_PROFILE[timeframe]]
    # Fallback: assume STANDARD (1h parameters)
    import logging
    logging.getLogger(__name__).warning(
        "Timeframe '%s' nao mapeado a perfil, usando STANDARD (1h).", timeframe
    )
    return PROFILE_STANDARD


def list_profiles() -> Dict[str, StrategyProfile]:
    """Retorna copia do dict de todos os perfis disponiveis."""
    return dict(_ALL_PROFILES)
