"""
strategy_liga_crypto.py
-----------------------
Metodologia profissional de análise gráfica hierárquica do canal Liga Crypto.
Motor de análise técnica e sinalização automática para BTC/USDT.

Hierarquia OBRIGATÓRIA de tempos gráficos:
  1. Semanal (1W) — Contexto macro, tendência principal
  2. Diário (1D) — Filtro de tendência, regra de ferro MA200
  3. 4 Horas (4H) — Definição do setup e zonas de interesse
  4. 1 Hora (1H) — Pré-execução e leitura de momentum local
  5. 15 Minutos (15M) — Execução e timing de entrada

Indicadores técnicos:
  - Médias: EMA(9), EMA(20), SMA(50), EMA(200), SMA(200)
  - Estocástico Slow (14, 3, 3)
  - RSI(14) com divergências
  - Bollinger Bands(20, 2.0) + BBWP
  - ATR(14) para SL/TP
  - ADX(14) + DI+/DI-
  - Fibonacci (retrações e projeções)

Regras de entrada:
  - LONG: 6 pré-condições obrigatórias (contexto W+D+4H + gatilho 1H/15M)
  - SHORT: 6 pré-condições obrigatórias
  - R:R mínimo 1:2
  - Gestão de risco: TP parcial 50/30/20, trailing stop

"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from indicators import compute_indicators

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# CONSTANTS — Liga Crypto Parameters
# ═══════════════════════════════════════════════════════════════════

# Timeframe hierarchy
TF_HIERARCHY = ["1W", "1D", "4H", "1H", "15M"]

# Context timeframes (filters)
TF_CONTEXT = {"1W", "1D"}
# Setup timeframe
TF_SETUP = {"4H"}
# Execution timeframes
TF_EXECUTION = {"1H", "15M"}

# MA parameters
EMA9_PERIOD = 9
EMA20_PERIOD = 20
SMA50_PERIOD = 50
EMA200_PERIOD = 200
SMA200_PERIOD = 200

# Stochastic (14, 3, 3)
STOCH_OVERBOUGHT = 80.0
STOCH_OVERSOLD = 20.0

# RSI
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70.0
RSI_OVERSOLD = 30.0
RSI_RESET_LOW = 37.0
RSI_RESET_HIGH = 40.0

# BBWP volatility states
BBWP_CONTRACTING_THRESHOLD = 50.0  # Below = contracting
BBWP_EXTREME_CONTRACTION = 5.0     # Historically extreme

# Risk management
RR_MIN = 2.0          # Minimum risk:reward
SL_MARGIN_PCT = 0.4   # 0.3-0.5% margin below/above structural SL
TP1_WEIGHT = 0.50     # 50% of position at TP1
TP2_WEIGHT = 0.30     # 30% at TP2
TP3_WEIGHT = 0.20     # 20% runner

# Structural analysis
SWING_LOOKBACK_MAP = {
    "1W": 5, "1D": 20, "4H": 24, "1H": 48, "15M": 96,
}

# Minimum candles required per timeframe
MIN_CANDLES_MAP = {
    "1W": 52, "1D": 250, "4H": 250, "1H": 250, "15M": 250,
}


# ═══════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════

class Trend(str, Enum):
    ALTA = "ALTA"
    BAIXA = "BAIXA"
    LATERAL = "LATERAL"


class Decision(str, Enum):
    COMPRA = "COMPRA"
    VENDA = "VENDA"
    AGUARDAR = "AGUARDAR"
    SEM_SINAL = "SEM SINAL"


class Confidence(str, Enum):
    ALTA = "ALTA"
    MEDIA = "MEDIA"
    BAIXA = "BAIXA"


@dataclass
class TimeframeAnalysis:
    """Resultado da análise de um timeframe individual."""
    timeframe: str = ""
    trend: Trend = Trend.LATERAL

    # Preço vs MAs
    price: float = 0.0
    ema9: float = 0.0
    ema20: float = 0.0
    sma50: float = 0.0
    ema200: float = 0.0
    sma200: float = 0.0
    price_vs_ema200: str = "desconhecido"
    price_vs_sma200: str = "desconhecido"

    # Médias ordering (9 → 20 → 50 → 200)
    ma_order: str = ""  # "bullish" = 9>20>50>200, "bearish" = reverse
    ma_order_score: int = 0  # +N for bullish ordering, -N for bearish

    # Volatilidade
    bbwp: float = 50.0
    bbwp_state: str = "desconhecido"  # "contraindo", "expandindo"
    bbwp_percentile: float = 50.0
    bb_lower: float = 0.0
    bb_upper: float = 0.0

    # Osciladores
    rsi: float = 50.0
    rsi_delta: float = 0.0
    stoch_k: float = 50.0
    stoch_d: float = 50.0

    # ADX
    adx: float = 0.0
    plus_di: float = 0.0
    minus_di: float = 0.0

    # Volume
    volume: float = 0.0
    volume_sma20: float = 0.0
    volume_above_avg: bool = False

    # ATR
    atr: float = 0.0

    # Estrutura semanal específica
    weekly_support: float = 0.0
    weekly_resistance: float = 0.0
    weeks_at_ma9: int = 0
    weekly_divergence: str = "nenhuma"

    # 4H específico
    pattern: str = "nenhum"  # canal, cunha_asc, cunha_desc, OCO, triangulo, range
    zone_type: str = "nenhuma"  # suporte, resistencia, meio_range, polaridade
    zone_price: float = 0.0
    cross_recent: str = "nenhum"  # ouro, morte, nenhum

    # 1H/15M específico
    local_structure: str = "desconhecido"
    exhaustion: str = "nenhuma"  # compradores, vendedores, nenhuma
    trigger_signal: str = "nenhum"  # estocastico, rsi_divergencia, rsi_reset, pivo
    trigger_detail: str = ""
    vol_expanding: bool = False

    # Divergência RSI
    rsi_div_bearish: bool = False
    rsi_div_bullish: bool = False

    # Blocking conditions
    surubadas: bool = False
    mid_range: bool = False

    # Fib projections (4H)
    fib_tp1: float = 0.0
    fib_tp2: float = 0.0
    fib_tp3: float = 0.0


@dataclass
class LigaCryptoResult:
    """Resultado completo da análise Liga Crypto."""
    decision: Decision = Decision.AGUARDAR
    confidence: Confidence = Confidence.BAIXA
    justification: str = ""
    entry_price: float = 0.0
    stop_loss: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0
    tp1_pct: float = 0.0
    tp2_pct: float = 0.0
    tp3_pct: float = 0.0
    rr_tp1: float = 0.0
    rr_tp2: float = 0.0
    sl_distance_pct: float = 0.0
    invalidation_reasons: list = field(default_factory=list)
    alerts: list = field(default_factory=list)
    timestamp: str = ""

    # Timeframe analyses
    weekly: TimeframeAnalysis = field(default_factory=TimeframeAnalysis)
    daily: TimeframeAnalysis = field(default_factory=TimeframeAnalysis)
    h4: TimeframeAnalysis = field(default_factory=TimeframeAnalysis)
    h1: TimeframeAnalysis = field(default_factory=TimeframeAnalysis)
    m15: Optional[TimeframeAnalysis] = None

    # Seasonal context
    seasonal_context: str = ""

    def to_report(self) -> str:
        """Gera o relatório no formato Bloco 8 da metodologia."""
        lines = []
        sep = "═" * 45

        lines.append(sep)
        lines.append("RELATÓRIO DE SINALIZAÇÃO — BTC/USDT (Liga Crypto)")
        lines.append(f"Data/Hora: {self.timestamp} UTC")
        lines.append(sep)

        # CONTEXTO MACRO (SEMANAL)
        lines.append("")
        lines.append("■ CONTEXTO MACRO (SEMANAL)")
        lines.append(f"  Tendência: {self.weekly.trend.value}")
        lines.append(f"  Posição vs SMA200 semanal: {self.weekly.price_vs_sma200}")
        if self.weekly.weekly_support > 0:
            lines.append(
                f"  Canaleta de preço: Suporte {self.weekly.weekly_support:,.2f} | "
                f"Resistência {self.weekly.weekly_resistance:,.2f}"
            )
        lines.append(
            f"  Volatilidade (BBWP): {self.weekly.bbwp_state} — {self.weekly.bbwp:.1f}%"
        )
        lines.append(f"  Divergência de topos: {self.weekly.weekly_divergence}")
        if self.weekly.weeks_at_ma9 > 0:
            lines.append(f"  Semanas de interação com MA9: {self.weekly.weeks_at_ma9}")

        # FILTRO DIÁRIO
        lines.append("")
        lines.append("■ FILTRO DIÁRIO")
        lines.append(
            f"  Posição vs MA200 diária: {self.daily.price_vs_sma200} → "
            f"Viés: {self._daily_bias()}"
        )
        lines.append(f"  Posição vs SMA50 diária: {self.daily.price_vs_ema200}")
        lines.append(f"  Padrão: {self._daily_pattern()}")
        lines.append(f"  RSI(14): {self.daily.rsi:.1f}")
        lines.append(
            f"  Volatilidade (BBWP): {self.daily.bbwp_state} — {self.daily.bbwp:.1f}%"
        )
        if self.daily.sma50 > 0:
            lines.append(f"  Suporte diário (SMA50): {self.daily.sma50:,.2f}")
        if self.daily.ema200 > 0:
            lines.append(f"  Resistência diária (EMA200): {self.daily.ema200:,.2f}")

        # SETUP (4H)
        lines.append("")
        lines.append("■ SETUP (4H)")
        lines.append(f"  Padrão identificado: {self.h4.pattern}")
        if self.h4.zone_price > 0:
            lines.append(
                f"  Zona de interesse: {self.h4.zone_type} em {self.h4.zone_price:,.2f}"
            )
        lines.append(
            f"  Médias: MA9 {self.h4.ema9:,.2f} | MA20 {self.h4.ema20:,.2f} | "
            f"SMA50 {self.h4.sma50:,.2f} | MA200 {self.h4.ema200:,.2f}"
        )
        lines.append(f"  Cruzamento recente: {self.h4.cross_recent}")
        lines.append(f"  Volatilidade: {self.h4.bbwp_state}")
        if self.h4.fib_tp1 > 0:
            lines.append(
                f"  Projeção Fibonacci: TP1 {self.h4.fib_tp1:,.2f} | "
                f"TP2 {self.h4.fib_tp2:,.2f} | TP3 {self.h4.fib_tp3:,.2f}"
            )
        lines.append(f"  ADX: {self.h4.adx:.1f} (+DI {self.h4.plus_di:.1f} / -DI {self.h4.minus_di:.1f})")

        # GATILHO DE EXECUÇÃO (1H / 15M)
        exec_tf = self.m15 if self.m15 else self.h1
        lines.append("")
        lines.append(f"■ GATILHO DE EXECUÇÃO ({exec_tf.timeframe})")
        lines.append(f"  Sinal: {exec_tf.trigger_signal}")
        if exec_tf.trigger_detail:
            lines.append(f"  Detalhe: {exec_tf.trigger_detail}")
        lines.append(
            f"  Volume: {'acima da média' if exec_tf.volume_above_avg else 'abaixo da média'}"
        )
        lines.append(
            f"  Confirmação de volatilidade: {'sim' if exec_tf.vol_expanding else 'não'}"
        )
        lines.append(f"  Estocástico: %K={exec_tf.stoch_k:.1f} %D={exec_tf.stoch_d:.1f}")
        lines.append(f"  RSI(14): {exec_tf.rsi:.1f} (delta {exec_tf.rsi_delta:+.1f})")

        # DECISÃO FINAL
        lines.append("")
        lines.append("■ DECISÃO FINAL")
        lines.append(f"  Sinal: {self.decision.value}")
        lines.append(f"  Confiança: {self.confidence.value}")
        lines.append(f"  Justificativa: {self.justification}")

        if self.decision in (Decision.COMPRA, Decision.VENDA):
            lines.append("")
            lines.append(f"  Preço de entrada: {self.entry_price:,.2f}")
            lines.append(f"  Stop Loss: {self.stop_loss:,.2f} ({self.sl_distance_pct:.2f}%)")
            lines.append(f"  TP1: {self.tp1:,.2f} ({self.tp1_pct:.2f}%) — realizar 50%")
            lines.append(f"  TP2: {self.tp2:,.2f} ({self.tp2_pct:.2f}%) — realizar 30%")
            lines.append(f"  TP3: {self.tp3:,.2f} ({self.tp3_pct:.2f}%) — runner 20%")
            if self.rr_tp1 > 0:
                lines.append(f"  Relação R:R até TP1: {self.rr_tp1:.2f}")
            if self.rr_tp2 > 0:
                lines.append(f"  Relação R:R até TP2: {self.rr_tp2:.2f}")

        # CONDIÇÕES DE INVALIDAÇÃO
        if self.invalidation_reasons:
            lines.append("")
            lines.append("■ CONDIÇÕES DE INVALIDAÇÃO")
            for reason in self.invalidation_reasons:
                lines.append(f"  - {reason}")

        # ALERTAS E OBSERVAÇÕES
        if self.alerts:
            lines.append("")
            lines.append("■ ALERTAS E OBSERVAÇÕES")
            for alert in self.alerts:
                lines.append(f"  - {alert}")

        if self.seasonal_context:
            lines.append("")
            lines.append(f"  Contexto sazonal: {self.seasonal_context}")

        lines.append("")
        lines.append(sep)

        return "\n".join(lines)

    # ── Report helpers ──
    def _daily_bias(self) -> str:
        if self.daily.price_vs_sma200 == "acima":
            return "COMPRA PERMITIDA"
        elif self.daily.price_vs_sma200 == "abaixo":
            return "VENDA PERMITIDA"
        return "NEUTRO"

    def _daily_pattern(self) -> str:
        if self.daily.surubadas:
            return "lateralização (médias surubadas)"
        if self.daily.trend == Trend.ALTA:
            return "tendência de alta"
        if self.daily.trend == Trend.BAIXA:
            return "tendência de baixa"
        return "lateralização"


# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS — Technical Analysis
# ═══════════════════════════════════════════════════════════════════

def _safe_float(val, default=0.0) -> float:
    """Retorna float ou default se NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return float(val)


def _determine_trend(
    close: float, ema20: float, ema50: float, ema200: float,
    sma50: float, adx: float, plus_di: float, minus_di: float,
) -> Trend:
    """Determina a tendência baseado nas médias e ADX."""
    if adx > 25 and plus_di > minus_di and close > ema50:
        return Trend.ALTA
    if adx > 25 and minus_di > plus_di and close < ema50:
        return Trend.BAIXA
    if close > ema200 and ema50 > ema200:
        return Trend.ALTA
    if close < ema200 and ema50 < ema200:
        return Trend.BAIXA
    return Trend.LATERAL


def _ma_ordering_score(
    ema9: float, ema20: float, sma50: float, ema200: float,
) -> tuple:
    """Calcula o score de ordenação das médias.
    Returns (label, score). Score: +4 = perfeito bullish, -4 = perfeito bearish."""
    score = 0
    if ema9 > ema20:
        score += 1
    if ema20 > sma50:
        score += 1
    if sma50 > ema200:
        score += 1
    if ema9 > ema200:
        score += 1

    if score >= 3:
        return "bullish", score
    elif score <= 1:
        return "bearish", score - 4  # -3 to -4
    return "misto", 0


def _detect_bbwp_state(bbwp: float, bbwp_prev: float) -> str:
    """Determina se BBWP está contraindo ou expandindo."""
    if bbwp < BBWP_CONTRACTING_THRESHOLD:
        return "contraindo"
    return "expandindo"


def _detect_surubadas(
    ema9: float, ema20: float, sma50: float, ema200: float,
    price: float, tolerance_pct: float = 0.3,
) -> bool:
    """Detecta se as médias estão 'surubadas' (sobrepostas sem direção)."""
    ref = price
    tol = ref * tolerance_pct / 100
    # Se todas as MAs estão dentro de 0.3% do preço, estão sobrepostas
    mas = [ema9, ema20, sma50, ema200]
    mas_valid = [m for m in mas if m > 0]
    if len(mas_valid) < 3:
        return False
    ma_range = max(mas_valid) - min(mas_valid)
    return ma_range < tol * 3  # Todas dentro de ~0.9% umas das outras


def _detect_cross_recent(
    ema50: float, sma200: float,
    ema50_prev: float, sma200_prev: float,
) -> str:
    """Detecta cruzamento recente de ouro/morte (últimos 3 candles)."""
    if ema50_prev <= sma200_prev and ema50 > sma200:
        return "ouro"
    if ema50_prev >= sma200_prev and ema50 < sma200:
        return "morte"
    return "nenhum"


def _detect_rsi_divergence(
    rsi_div_bearish: bool, rsi_div_bullish: bool,
) -> str:
    """Retorna tipo de divergência RSI."""
    if rsi_div_bearish:
        return "bearish_regular"
    if rsi_div_bullish:
        return "bullish_regular"
    return "nenhuma"


def _count_weeks_at_ma9(df_weekly: pd.DataFrame, lookback: int = 12) -> int:
    """Conta semanas de interação com MA9 semanal."""
    if len(df_weekly) < lookback:
        return 0
    recent = df_weekly.tail(lookback)
    interaction = 0
    tolerance = recent["close"].mean() * 0.02  # 2% tolerance
    for _, row in recent.iterrows():
        low = _safe_float(row.get("low"))
        high = _safe_float(row.get("high"))
        ma9 = _safe_float(row.get("ema9"))
        if ma9 > 0 and low <= ma9 * (1 + tolerance / ma9) and high >= ma9 * (1 - tolerance / ma9):
            interaction += 1
    return interaction


def _detect_support_resistance(
    df: pd.DataFrame, lookback: int = 50,
) -> tuple:
    """Detecta zonas de suporte e resistência baseado em swing points.
    Returns (support, resistance)."""
    if len(df) < lookback:
        return (0.0, 0.0)
    recent = df.tail(lookback)
    # Suporte = média dos swing lows recentes
    # Resistência = média dos swing highs recentes
    lows = recent["low"].values
    highs = recent["high"].values
    # Simple approach: use rolling min/max clusters
    support = float(np.percentile(lows, 15))
    resistance = float(np.percentile(highs, 85))
    return (support, resistance)


def _detect_4h_pattern(df: pd.DataFrame, lookback: int = 50) -> str:
    """Detecta padrões gráficos básicos no 4H.
    Returns: canal, cunha_asc, cunha_desc, triangulo, range, nenhum
    """
    if len(df) < lookback:
        return "nenhum"
    recent = df.tail(lookback)
    highs = recent["high"].values
    lows = recent["low"].values
    closes = recent["close"].values

    # Linear regression on highs and lows
    x = np.arange(len(highs))
    if len(x) < 10:
        return "nenhum"

    # Highs trend
    h_slope, h_intercept = np.polyfit(x, highs, 1)
    # Lows trend
    l_slope, l_intercept = np.polyfit(x, lows, 1)
    # Close trend
    c_slope, _ = np.polyfit(x, closes, 1)

    # Channel: parallel highs and lows
    h_r = np.corrcoef(x, highs)[0, 1]
    l_r = np.corrcoef(x, lows)[0, 1]

    if abs(h_r) > 0.8 and abs(l_r) > 0.8:
        # Both trending in same direction = channel
        if h_slope > 0 and l_slope > 0:
            return "canal_ascendente"
        if h_slope < 0 and l_slope < 0:
            return "canal_descendente"

    # Wedge: converging highs and lows
    if h_r > 0.6 and l_r < -0.6:
        return "cunha_descendente"
    if h_r < -0.6 and l_r > 0.6:
        return "cunha_ascendente"

    # Triangle: converging with one flat side
    if abs(h_slope) < 0.0001 and l_slope > 0:
        return "triangulo_ascendente"
    if abs(l_slope) < 0.0001 and h_slope < 0:
        return "triangulo_descendente"
    if abs(h_r) > 0.5 and abs(l_r) > 0.5 and h_slope * l_slope < 0:
        return "triangulo_simétrico"

    # Range: very low slope
    if abs(c_slope) < 0.0001:
        return "range"

    return "tendência"


def _detect_zone_type(
    price: float, support: float, resistance: float,
    bbwp: float, bb_lower: float, bb_upper: float,
) -> tuple:
    """Detecta se o preço está em zona de suporte, resistência ou meio do range.
    Returns (zone_type, zone_price).
    """
    if support <= 0 or resistance <= 0:
        return ("nenhuma", 0.0)
    range_size = resistance - support
    if range_size <= 0:
        return ("nenhuma", 0.0)

    # Se preço nos 20% inferiores = zona de suporte
    position_pct = (price - support) / range_size
    if position_pct < 0.20:
        return ("suporte", support)
    # Se preço nos 20% superiores = zona de resistência
    if position_pct > 0.80:
        return ("resistência", resistance)
    return ("meio_range", 0.0)


def _calc_fib_projections(
    swing_low: float, swing_high: float, direction: int,
) -> tuple:
    """Calcula projeções Fibonacci para alvos.
    direction=1 (alta): projeta para cima a partir do último fundo
    direction=-1 (baixa): projeta para baixo a partir do último topo
    Returns (tp1, tp2, tp3) usando extensões 0.618, 1.0, 1.618
    """
    if swing_low <= 0 or swing_high <= 0:
        return (0.0, 0.0, 0.0)
    move = swing_high - swing_low
    if direction == 1:
        tp1 = swing_low + move * 0.618
        tp2 = swing_low + move * 1.0
        tp3 = swing_low + move * 1.618
    else:
        tp1 = swing_high - move * 0.618
        tp2 = swing_high - move * 1.0
        tp3 = swing_high - move * 1.618
    return (tp1, tp2, tp3)


def _detect_local_structure(df: pd.DataFrame, lookback: int = 24) -> str:
    """Detecta estrutura local de topos e fundos.
    Returns: topos_ascendentes, fundos_ascendentes, lateral, etc.
    """
    if len(df) < lookback:
        return "desconhecido"
    recent = df.tail(lookback)
    # Find local peaks and troughs
    highs = recent["high"].values
    lows = recent["low"].values
    n = len(highs)
    if n < 8:
        return "desconhecido"

    peaks = []
    troughs = []
    for i in range(2, n - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            peaks.append(highs[i])
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            troughs.append(lows[i])

    if len(peaks) < 2 or len(troughs) < 2:
        return "insuficiente"

    # Check trend of peaks and troughs
    peaks_trending_up = all(peaks[i] > peaks[i-1] for i in range(1, len(peaks)))
    peaks_trending_down = all(peaks[i] < peaks[i-1] for i in range(1, len(peaks)))
    troughs_trending_up = all(troughs[i] > troughs[i-1] for i in range(1, len(troughs)))
    troughs_trending_down = all(troughs[i] < troughs[i-1] for i in range(1, len(troughs)))

    if peaks_trending_up and troughs_trending_up:
        return "topos_ascendentes_fundos_ascendentes"
    if peaks_trending_down and troughs_trending_down:
        return "topos_descendentes_fundos_descendentes"
    if not peaks_trending_up and not peaks_trending_down and not troughs_trending_up and not troughs_trending_down:
        return "lateral"
    return "misto"


def _detect_exhaustion(
    stoch_1h: float, stoch_4h: float, stoch_1d: float,
    zone: str,
) -> str:
    """Detecta 'exaustão tripla' — extremos em múltiplos TFs.
    Returns: compradores, vendedores, nenhuma
    """
    # All three in overbought = buyer exhaustion
    if stoch_1h > 80 and stoch_4h > 80 and stoch_1d > 80:
        return "compradores"
    # All three in oversold = seller exhaustion
    if stoch_1h < 20 and stoch_4h < 20 and stoch_1d < 20:
        return "vendedores"
    return "nenhuma"


def _detect_trigger_1h(row: pd.Series, prev_row: pd.Series) -> tuple:
    """Detecta gatilho de execução no 1H.
    Returns (signal_type, detail).
    Possible: estocastico, rsi_divergencia, rsi_reset, pivo, nenhum
    """
    stoch_k = _safe_float(row.get("stoch_k"))
    stoch_d = _safe_float(row.get("stoch_d"))
    stoch_k_prev = _safe_float(prev_row.get("stoch_k"))
    stoch_d_prev = _safe_float(prev_row.get("stoch_d"))
    rsi = _safe_float(row.get("rsi"))
    rsi_delta = _safe_float(row.get("rsi_delta"))
    rsi_div_bullish = bool(row.get("rsi_div_bullish", False))
    rsi_div_bearish = bool(row.get("rsi_div_bearish", False))
    close = _safe_float(row.get("close"))
    high = _safe_float(row.get("high"))
    low = _safe_float(row.get("low"))
    ema50 = _safe_float(row.get("ema50"))
    ema200 = _safe_float(row.get("ema200"))

    # 1. Estocástico: %K cruza acima de %D saindo de sobrevenda
    if (stoch_k_prev < stoch_d_prev and stoch_k > stoch_d
            and stoch_d_prev < STOCH_OVERSOLD):
        return "estocastico", f"%K({stoch_k:.1f}) cruzou acima de %D({stoch_d:.1f}) saindo de sobrevenda"

    # 2. Estocástico: %K cruza abaixo de %D saindo de sobrecompra
    if (stoch_k_prev > stoch_d_prev and stoch_k < stoch_d
            and stoch_d_prev > STOCH_OVERBOUGHT):
        return "estocastico", f"%K({stoch_k:.1f}) cruzou abaixo de %D({stoch_d:.1f}) saindo de sobrecompra"

    # 3. RSI divergência bullish
    if rsi_div_bullish:
        return "rsi_divergência", "Divergência bullish RSI — preço fundo mais baixo, RSI fundo mais alto"

    # 4. RSI divergência bearish
    if rsi_div_bearish:
        return "rsi_divergência", "Divergência bearish RSI — preço topo mais alto, RSI topo mais baixo"

    # 5. RSI reset na zona 37-40 com reversão
    if RSI_RESET_LOW <= rsi <= RSI_RESET_HIGH and rsi_delta > 0.5:
        return "rsi_reset", f"RSI({rsi:.1f}) na zona de reset 37-40 com virada (delta {rsi_delta:+.1f})"

    # 6. Pivô de alta: rompeu topo anterior
    # (simplified: close above recent high)
    recent_high = _safe_float(row.get("bb_upper"))  # proxy
    if close > recent_high and ema50 > ema200:
        return "pivo", f"Rompeu resistência local em {recent_high:,.2f}"

    return "nenhum", ""


def _detect_trigger_15m(row: pd.Series, prev_row: pd.Series) -> tuple:
    """Detecta gatilho de execução no 15M (divergências escondidas, reset).
    Returns (signal_type, detail).
    """
    stoch_k = _safe_float(row.get("stoch_k"))
    stoch_d = _safe_float(row.get("stoch_d"))
    stoch_k_prev = _safe_float(prev_row.get("stoch_k"))
    stoch_d_prev = _safe_float(prev_row.get("stoch_d"))
    rsi = _safe_float(row.get("rsi"))
    rsi_delta = _safe_float(row.get("rsi_delta"))
    close = _safe_float(row.get("close"))
    ema50 = _safe_float(row.get("ema50"))

    # Divergência escondida de alta: preço topo mais baixo, RSI topo mais alto
    if rsi_delta > 1.0 and rsi < RSI_OVERBOUGHT and ema50 > 0 and close > ema50:
        return "divergência_escondida_alta", f"RSI({rsi:.1f}) com delta positivo ({rsi_delta:+.1f}) em tendência de alta"

    # Reset de RSI na zona 37-40
    if RSI_RESET_LOW <= rsi <= RSI_RESET_HIGH and rsi_delta > 0.3:
        return "rsi_reset", f"RSI({rsi:.1f}) na zona de reset com virada"

    # Estocástico saindo de sobrevenda
    if (stoch_k_prev < stoch_d_prev and stoch_k > stoch_d
            and stoch_d_prev < STOCH_OVERSOLD):
        return "estocastico", f"%K({stoch_k:.1f}) cruzou %D({stoch_d:.1f}) saindo de sobrevenda no 15M"

    return "nenhum", ""


def _get_seasonal_context() -> str:
    """Retorna contexto sazonal/cíclico atual."""
    now = datetime.now(timezone.utc)
    month = now.month
    year = now.year

    # Bitcoin 4-year cycle (approximate)
    # Halvings: 2012, 2016, 2020, 2024
    years_since_halving = (year - 2024) % 4
    if years_since_halving <= 1:
        cycle_phase = "Ano 1 pós-halving (tendência de alta esperada)"
    elif years_since_halving == 2:
        cycle_phase = "Ano 2 pós-halving (possível topo de ciclo)"
    else:
        cycle_phase = "Ano 3-4 pós-halving (tendência de baixa possível)"

    # Monthly seasonality
    if month == 7:
        monthly = "Julho: historicamente positivo em anos de queda"
    elif month == 8:
        monthly = "Agosto: até 2 semanas de sustentação antes de movimento maior"
    elif month in (9, 10):
        monthly = "Set-Out: movimentos direcionais mais fortes"
    elif month in (11, 12):
        monthly = "Nov-Dez: possível fundo ou início de recuperação"
    elif month in (1, 2, 3):
        monthly = "Jan-Mar: início de ano, volume tende a aumentar"
    elif month in (4, 5, 6):
        monthly = "Apr-Jun: meio do ano, movimentos moderados"
    else:
        monthly = ""

    parts = [cycle_phase]
    if monthly:
        parts.append(monthly)

    # Day of week (weekend warning)
    dow = now.weekday()
    if dow >= 5:  # Saturday=5, Sunday=6
        parts.append("FIM DE SEMANA: volume extremamente baixo, cautela extra")

    return " | ".join(parts)


def _is_weekend() -> bool:
    """Verifica se é fim de semana (UTC)."""
    dow = datetime.now(timezone.utc).weekday()
    return dow >= 5


def _is_macro_event_window(dt=None) -> bool:
    """Verifica se estamos em janela de eventos macro (aproximação).
    Sem API de calendario: retorna False.
    Em producao, integrar com API de calendario economico.
    
    Parameters:
        dt: datetime para backtest. Se None, usa agora.
    """
    return False


def _get_seasonal_context_quant(dt=None) -> int:
    """Retorna score sazonal quantitativo para filtro de sinal.
    
    Score:
        +2 = ambiente muito favoravel
        +1 = levemente favoravel
         0 = neutro
        -1 = levemente desfavoravel
        -2 = bloquear sinal (nao entrar)
    
    Fatores:
        - Ciclo BTC 4 anos (halving 2012/2016/2020/2024)
        - Mes do ano (sazonalidade historica BTC)
        - Dia da semana (volume)
        
    Parameters:
        dt: datetime-aware ou naive timestamp. Se None, usa agora.
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    
    score = 0
    month = dt.month if hasattr(dt, 'month') else dt.month
    year = dt.year if hasattr(dt, 'year') else dt.year
    dow = dt.weekday() if hasattr(dt, 'weekday') else 0  # 0=Monday, 6=Sunday
    
    # ── Ciclo BTC 4 anos (halvings: 2012, 2016, 2020, 2024) ──
    years_since_halving = (year - 2024) % 4
    if years_since_halving == 0 or years_since_halving == 1:
        score += 2  # Ano 0-1 pos-halving: tendencia historica de alta
    elif years_since_halving == 2:
        score += 0  # Possivel topo de ciclo
    else:  # years_since_halving == 3
        score -= 1  # Pre-halving bear market tipico
    
    # ── Sazonalidade mensal BTC (baseado em dados historicos 2015-2024) ──
    favorable_months = {9, 10, 11}  # Set-Out-Nov: Q4 rally
    unfavorable_months = {1, 2, 5, 6}  # Jan-Fev (dump), Mai-Jun (summer lull)
    
    if month in favorable_months:
        score += 1
    elif month in unfavorable_months:
        score -= 1
    
    # ── Dia da semana ──
    if dow >= 5:  # Weekend
        score -= 2  # BLOQUEAR
    elif dow == 6:  # Sunday
        score -= 1  # Virada semanal
    
    return score


# ═══════════════════════════════════════════════════════════════════
# PER-TIMEFRAME ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def _analyze_single_tf(
    df_ind: pd.DataFrame, timeframe: str,
) -> TimeframeAnalysis:
    """Analisa um timeframe individual e retorna TimeframeAnalysis."""
    if df_ind.empty:
        return TimeframeAnalysis(timeframe=timeframe)

    row = df_ind.iloc[-1]
    prev = df_ind.iloc[-2] if len(df_ind) >= 2 else row
    analysis = TimeframeAnalysis(timeframe=timeframe)

    # Basic values
    analysis.price = _safe_float(row.get("close"))
    analysis.ema9 = _safe_float(row.get("ema9"))
    analysis.ema20 = _safe_float(row.get("ema20"))
    analysis.sma50 = _safe_float(row.get("sma50"))
    analysis.ema200 = _safe_float(row.get("ema200"))
    analysis.sma200 = _safe_float(row.get("sma200"))

    # Price vs MAs
    if analysis.ema200 > 0:
        analysis.price_vs_ema200 = "acima" if analysis.price > analysis.ema200 else "abaixo"
    if analysis.sma200 > 0:
        analysis.price_vs_sma200 = "acima" if analysis.price > analysis.sma200 else "abaixo"

    # MA ordering
    analysis.ma_order, analysis.ma_order_score = _ma_ordering_score(
        analysis.ema9, analysis.ema20, analysis.sma50, analysis.ema200,
    )

    # Trend
    analysis.trend = _determine_trend(
        analysis.price, analysis.ema20, analysis.sma50 if analysis.sma50 > 0 else analysis.ema20,
        analysis.ema200, analysis.sma50,
        _safe_float(row.get("adx")),
        _safe_float(row.get("plus_di")),
        _safe_float(row.get("minus_di")),
    )

    # Volatility
    analysis.bbwp = _safe_float(row.get("bbwp"))
    analysis.bbwp_percentile = _safe_float(row.get("bb_squeeze_pct"))
    bbwp_prev = _safe_float(prev.get("bbwp"))
    analysis.bbwp_state = _detect_bbwp_state(analysis.bbwp, bbwp_prev)
    analysis.bb_lower = _safe_float(row.get("bb_lower"))
    analysis.bb_upper = _safe_float(row.get("bb_upper"))

    # Oscillators
    analysis.rsi = _safe_float(row.get("rsi"))
    analysis.rsi_delta = _safe_float(row.get("rsi_delta"))
    analysis.stoch_k = _safe_float(row.get("stoch_k"))
    analysis.stoch_d = _safe_float(row.get("stoch_d"))

    # ADX
    analysis.adx = _safe_float(row.get("adx"))
    analysis.plus_di = _safe_float(row.get("plus_di"))
    analysis.minus_di = _safe_float(row.get("minus_di"))

    # Volume
    analysis.volume = _safe_float(row.get("volume"))
    analysis.volume_sma20 = _safe_float(row.get("volume_sma20"))
    if analysis.volume_sma20 > 0:
        analysis.volume_above_avg = analysis.volume > analysis.volume_sma20

    # ATR
    analysis.atr = _safe_float(row.get("atr"))

    # Divergences
    analysis.rsi_div_bearish = bool(row.get("rsi_div_bearish", False))
    analysis.rsi_div_bullish = bool(row.get("rsi_div_bullish", False))

    # Blocking
    analysis.surubadas = _detect_surubadas(
        analysis.ema9, analysis.ema20, analysis.sma50, analysis.ema200,
        analysis.price,
    )

    return analysis


def analyze_weekly(df_ind: pd.DataFrame) -> TimeframeAnalysis:
    """Análise do timeframe semanal — contexto macro."""
    analysis = _analyze_single_tf(df_ind, "1W")

    # Suporte e resistência semanais
    support, resistance = _detect_support_resistance(df_ind, lookback=26)
    analysis.weekly_support = support
    analysis.weekly_resistance = resistance

    # Divergência semanal
    analysis.weekly_divergence = _detect_rsi_divergence(
        analysis.rsi_div_bearish, analysis.rsi_div_bullish,
    )

    # Semanas na MA9
    analysis.weeks_at_ma9 = _count_weeks_at_ma9(df_ind)

    return analysis


def analyze_daily(df_ind: pd.DataFrame) -> TimeframeAnalysis:
    """Análise do timeframe diário — filtro de tendência."""
    analysis = _analyze_single_tf(df_ind, "1D")
    return analysis


def analyze_4h(df_ind: pd.DataFrame) -> TimeframeAnalysis:
    """Análise do 4H — definição do setup e zonas."""
    analysis = _analyze_single_tf(df_ind, "4H")

    if len(df_ind) >= 3:
        row = df_ind.iloc[-1]
        prev = df_ind.iloc[-2]
        prev2 = df_ind.iloc[-3] if len(df_ind) >= 3 else prev

        # Cruzamento de ouro/morte
        analysis.cross_recent = _detect_cross_recent(
            _safe_float(row.get("ema50")), _safe_float(row.get("sma200")),
            _safe_float(prev.get("ema50")), _safe_float(prev.get("sma200")),
        )

    # Padrão gráfico
    analysis.pattern = _detect_4h_pattern(df_ind, lookback=50)

    # Zona de suporte/resistência
    support, resistance = _detect_support_resistance(df_ind, lookback=50)
    analysis.zone_type, analysis.zone_price = _detect_zone_type(
        analysis.price, support, resistance,
        analysis.bbwp, analysis.bb_lower, analysis.bb_upper,
    )

    # Fib projections
    # Find recent swing low and high
    if len(df_ind) >= 30:
        recent = df_ind.tail(50)
        swing_low = float(recent["low"].min())
        swing_high = float(recent["high"].max())
        direction = 1 if analysis.trend == Trend.ALTA else -1
        analysis.fib_tp1, analysis.fib_tp2, analysis.fib_tp3 = _calc_fib_projections(
            swing_low, swing_high, direction,
        )

    return analysis


def analyze_1h(
    df_ind: pd.DataFrame,
    h4_stoch_k: float, h4_stoch_d: float,
    d1_stoch_k: float, d1_stoch_d: float,
) -> TimeframeAnalysis:
    """Análise do 1H — pré-execução e momentum local."""
    analysis = _analyze_single_tf(df_ind, "1H")

    # Estrutura local
    analysis.local_structure = _detect_local_structure(df_ind, lookback=48)

    # Exaustão tripla
    analysis.exhaustion = _detect_exhaustion(
        analysis.stoch_k, h4_stoch_k, d1_stoch_k, analysis.zone_type,
    )

    # Volatilidade expandindo (BBWP atual > anterior)
    if len(df_ind) >= 2:
        bbwp_now = _safe_float(df_ind.iloc[-1].get("bbwp"))
        bbwp_prev = _safe_float(df_ind.iloc[-2].get("bbwp"))
        analysis.vol_expanding = bbwp_now > bbwp_prev

    # Gatilho de execução
    if len(df_ind) >= 2:
        row = df_ind.iloc[-1]
        prev_row = df_ind.iloc[-2]
        analysis.trigger_signal, analysis.trigger_detail = _detect_trigger_1h(row, prev_row)

    return analysis


def analyze_15m(df_ind: pd.DataFrame) -> TimeframeAnalysis:
    """Análise do 15M — execução e timing."""
    analysis = _analyze_single_tf(df_ind, "15M")

    # Estrutura local
    analysis.local_structure = _detect_local_structure(df_ind, lookback=48)

    # Volatilidade expandindo
    if len(df_ind) >= 2:
        bbwp_now = _safe_float(df_ind.iloc[-1].get("bbwp"))
        bbwp_prev = _safe_float(df_ind.iloc[-2].get("bbwp"))
        analysis.vol_expanding = bbwp_now > bbwp_prev

    # Gatilho 15M
    if len(df_ind) >= 2:
        row = df_ind.iloc[-1]
        prev_row = df_ind.iloc[-2]
        analysis.trigger_signal, analysis.trigger_detail = _detect_trigger_15m(row, prev_row)

    return analysis


# ═══════════════════════════════════════════════════════════════════
# ENTRY LOGIC — LONG & SHORT
# ═══════════════════════════════════════════════════════════════════

def _check_long_preconditions(
    w: TimeframeAnalysis, d: TimeframeAnalysis,
    h4: TimeframeAnalysis, h1: TimeframeAnalysis,
    m15: Optional[TimeframeAnalysis],
) -> tuple:
    """Verifica as 6 pré-condições de entrada LONG.
    Returns (passed: bool, reasons: list[str]).
    """
    reasons = []

    # 1. Contexto semanal: preço acima ou testando suporte semanal
    #    Ausência de divergência escondida de baixa
    if w.trend == Trend.BAIXA and w.price < w.weekly_support and w.weekly_support > 0:
        reasons.append("Semanal em tendência de baixa abaixo do suporte")
    if w.rsi_div_bearish:
        reasons.append("Divergência bearish RSI no semanal")

    # 2. Contexto diário: preço ACIMA da MA200 diária
    #    "Regra de ferro": preço abaixo = COMPRA BLOQUEADA
    if d.price < d.sma200 and d.sma200 > 0:
        reasons.append(f"BLOCKED: Preço ({d.price:,.2f}) abaixo da MA200 diária ({d.sma200:,.2f}) — regra de ferro")

    # 3. Setup no 4H: zona de suporte OU polaridade OU pullback para média
    #    com contração de volatilidade
    if h4.zone_type not in ("suporte", "resistência") and h4.pattern not in ("canal_ascendente", "tendência"):
        reasons.append("4H sem setup identificável (sem zona ou padrão de interesse)")
    if h4.surubadas:
        reasons.append("4H: Médias surubadas — sem direção clara")
    if h4.mid_range:
        reasons.append("4H: Preço no meio do range — não operar")

    # 4. Sinal no 1H ou 15M
    exec_tf = m15 if m15 and m15.trigger_signal != "nenhum" else h1
    if exec_tf.trigger_signal == "nenhum":
        reasons.append(f"Sem gatilho de execução no 1H/15M")

    # 5. Volatilidade: BBWP contraindo no pullback e expandindo no sinal
    if not exec_tf.vol_expanding:
        reasons.append("Volatilidade não expandindo no sinal — movimento frágil")

    # 6. Volume
    if not exec_tf.volume_above_avg:
        reasons.append("Volume abaixo da média — confirmação fraca")

    passed = len(reasons) == 0
    return passed, reasons


def _check_short_preconditions(
    w: TimeframeAnalysis, d: TimeframeAnalysis,
    h4: TimeframeAnalysis, h1: TimeframeAnalysis,
    m15: Optional[TimeframeAnalysis],
) -> tuple:
    """Verifica as 6 pré-condições de entrada SHORT.
    Returns (passed: bool, reasons: list[str]).
    """
    reasons = []

    # 1. Contexto semanal: tendência de baixa OU rejeitando resistência
    if w.trend == Trend.ALTA and w.price > w.weekly_resistance and w.weekly_resistance > 0:
        reasons.append("Semanal em tendência de alta acima da resistência")
    if w.rsi_div_bullish:
        reasons.append("Divergência bullish RSI no semanal")

    # 2. Contexto diário: preço ABAIXO da MA200 diária
    if d.price > d.sma200 and d.sma200 > 0:
        reasons.append(f"BLOCKED: Preço ({d.price:,.2f}) acima da MA200 diária ({d.sma200:,.2f}) — viés altista")

    # 3. Setup no 4H: zona de resistência OU polaridade OU pullback
    if h4.zone_type not in ("resistência", "suporte") and h4.pattern not in ("canal_descendente", "tendência"):
        reasons.append("4H sem setup identificável para venda")
    if h4.surubadas:
        reasons.append("4H: Médias surubadas — sem direção clara")
    if h4.mid_range:
        reasons.append("4H: Preço no meio do range — não operar")

    # 4. Sinal no 1H ou 15M
    exec_tf = m15 if m15 and m15.trigger_signal != "nenhum" else h1
    if exec_tf.trigger_signal == "nenhum":
        reasons.append("Sem gatilho de execução no 1H/15M")

    # 5. Volatilidade expandindo
    if not exec_tf.vol_expanding:
        reasons.append("Volatilidade não expandindo no sinal — movimento frágil")

    # 6. Volume
    if not exec_tf.volume_above_avg:
        reasons.append("Volume abaixo da média — confirmação fraca")

    passed = len(reasons) == 0
    return passed, reasons


def _check_blocking_conditions(
    w: TimeframeAnalysis, d: TimeframeAnalysis,
    h4: TimeframeAnalysis, h1: TimeframeAnalysis,
) -> list:
    """Verifica condições de bloqueio gerais (para qualquer direção)."""
    blocks = []

    # Preço no meio de range lateral
    if h4.mid_range:
        blocks.append("Preço no meio de range lateral — aguardar borda")

    # Médias surubadas no 4H
    if h4.surubadas:
        blocks.append("Médias 4H sobrepostas sem direção — não operar")

    # Volatilidade em contração sem expansão
    if h1.bbwp_state == "contraindo" and not h1.vol_expanding:
        blocks.append("Volatilidade em contração sem sinal de expansão")

    # Fim de semana
    if _is_weekend():
        blocks.append("Fim de semana — volume baixo, estruturas frágeis")

    # Filtro sazonal quantitativo (usado em backtest via sim_liga_crypto,
    # mas adicionado aqui como camada extra para consistencia)
    _now = datetime.now(timezone.utc)
    if _get_seasonal_context_quant(_now) <= -2:
        blocks.append("Contexto sazonal desfavoravel (score <= -2) — sinal bloqueado")

    # Divergência contrária no TF superior
    # (se 1H mostra sinal de compra mas 4H tem divergência bearish)
    if h4.rsi_div_bearish and h1.trigger_signal in ("estocastico", "rsi_reset", "pivo"):
        blocks.append("Divergência RSI bearish no 4H contradiz sinal de compra no 1H")
    if h4.rsi_div_bullish and h1.trigger_signal in ("estocastico", "rsi_reset", "pivo"):
        blocks.append("Divergência RSI bullish no 4H contradiz sinal de venda no 1H")

    # Semana semanal ainda aberta com preço em zona de dúvida
    # (simplificado: segunda/quarta com BBWP contraindo)
    dow = datetime.now(timezone.utc).weekday()
    if dow <= 2 and w.bbwp_state == "contraindo" and w.bbwp < 20:
        blocks.append("Semana semanal aberta com BBWP extremamente contraído — aguardar definição")

    return blocks


def _calculate_sl_tp_long(
    h4: TimeframeAnalysis, h1: TimeframeAnalysis,
) -> tuple:
    """Calcula SL e TPs para entrada LONG.
    SL: abaixo do último swing low do 4H + margem 0.3-0.5%
    TPs: Fibonacci projections ou zonas de resistência
    Returns (sl, tp1, tp2, tp3).
    """
    entry = h1.price
    atr = h4.atr if h4.atr > 0 else h1.atr

    # SL: abaixo do swing low do 4H + margem
    if h4.atr > 0:
        # Use 1.8x ATR como base (consistente com V13)
        sl_distance = 1.8 * h4.atr
        sl = entry - sl_distance
    else:
        sl = entry * (1 - SL_MARGIN_PCT / 100)

    # TPs: Fibonacci do 4H ou distância fixa
    if h4.fib_tp1 > 0:
        tp1 = h4.fib_tp1
        tp2 = h4.fib_tp2
        tp3 = h4.fib_tp3
    elif atr > 0:
        tp1 = entry + 3.0 * atr  # 1:3 RR to TP1 (approx)
        tp2 = entry + 5.0 * atr
        tp3 = entry + 8.0 * atr
    else:
        risk = entry - sl
        tp1 = entry + risk * 3.0
        tp2 = entry + risk * 5.0
        tp3 = entry + risk * 8.0

    return sl, tp1, tp2, tp3


def _calculate_sl_tp_short(
    h4: TimeframeAnalysis, h1: TimeframeAnalysis,
) -> tuple:
    """Calcula SL e TPs para entrada SHORT.
    Returns (sl, tp1, tp2, tp3).
    """
    entry = h1.price
    atr = h4.atr if h4.atr > 0 else h1.atr

    # SL: acima do swing high + margem
    if h4.atr > 0:
        sl_distance = 1.8 * h4.atr
        sl = entry + sl_distance
    else:
        sl = entry * (1 + SL_MARGIN_PCT / 100)

    # TPs
    if h4.fib_tp1 > 0:
        tp1 = h4.fib_tp1
        tp2 = h4.fib_tp2
        tp3 = h4.fib_tp3
    elif atr > 0:
        tp1 = entry - 3.0 * atr
        tp2 = entry - 5.0 * atr
        tp3 = entry - 8.0 * atr
    else:
        risk = sl - entry
        tp1 = entry - risk * 3.0
        tp2 = entry - risk * 5.0
        tp3 = entry - risk * 8.0

    return sl, tp1, tp2, tp3


def _assess_confidence(
    w: TimeframeAnalysis, d: TimeframeAnalysis,
    h4: TimeframeAnalysis, h1: TimeframeAnalysis,
    m15: Optional[TimeframeAnalysis], confluences: int,
) -> Confidence:
    """Avalia a confiança do sinal baseado em confluências."""
    if confluences >= 4:
        return Confidence.ALTA
    if confluences >= 2:
        return Confidence.MEDIA
    return Confidence.BAIXA


def _count_confluences(
    w: TimeframeAnalysis, d: TimeframeAnalysis,
    h4: TimeframeAnalysis, h1: TimeframeAnalysis,
    is_long: bool,
) -> int:
    """Conta o número de confluências alinhadas."""
    count = 0

    # 1. Semanal e diário na mesma direção
    if is_long and w.trend != Trend.BAIXA and d.trend == Trend.ALTA:
        count += 1
    if not is_long and w.trend != Trend.ALTA and d.trend == Trend.BAIXA:
        count += 1

    # 2. 4H em zona de interesse
    if h4.zone_type in ("suporte", "resistência"):
        count += 1

    # 3. Volatilidade expandindo
    if h1.vol_expanding:
        count += 1

    # 4. Volume acima da média
    if h1.volume_above_avg:
        count += 1

    # 5. RSI na zona correta
    if is_long and h1.rsi < 50:
        count += 1
    if not is_long and h1.rsi > 50:
        count += 1

    # 6. Estocástico confirmando
    if is_long and h1.stoch_k > h1.stoch_d and h1.stoch_d < 40:
        count += 1
    if not is_long and h1.stoch_k < h1.stoch_d and h1.stoch_d > 60:
        count += 1

    # 7. Sem divergência contrária
    if is_long and not h4.rsi_div_bearish:
        count += 1
    if not is_long and not h4.rsi_div_bullish:
        count += 1

    return count


# ═══════════════════════════════════════════════════════════════════
# MAIN ANALYSIS FUNCTION
# ═══════════════════════════════════════════════════════════════════

def analyze_liga_crypto(
    dfs: dict[str, pd.DataFrame],
) -> LigaCryptoResult:
    """
    Análise completa Liga Crypto — hierarquia 1W → 1D → 4H → 1H → 15M.

    Parameters:
        dfs: Dicionário com DataFrames OHLCV por timeframe.
              Chaves obrigatórias: "1W", "1D", "4H", "1H".
              Opcional: "15M".

    Returns:
        LigaCryptoResult com análise completa e possível sinal.
    """
    result = LigaCryptoResult()
    result.timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # ── Validar dados mínimos ──
    for tf in ["1W", "1D", "4H", "1H"]:
        if tf not in dfs or dfs[tf].empty:
            result.justification = f"Dados insuficientes para {tf}"
            result.decision = Decision.SEM_SINAL
            return result

    # ── Computar indicadores para cada timeframe ──
    # Mapear nomes dos TFs para o formato do indicators.py
    _tf_name_map = {"1W": "1W", "1D": "1d", "4H": "4h", "1H": "1h", "15M": "15m"}
    dfs_ind = {}
    for tf, df in dfs.items():
        min_candles = MIN_CANDLES_MAP.get(tf, 250)
        if len(df) >= 50:  # Mínimo absoluto para qualquer indicador
            indicator_tf = _tf_name_map.get(tf, tf.lower())
            dfs_ind[tf] = compute_indicators(df, timeframe=indicator_tf)
        else:
            dfs_ind[tf] = df  # Sem indicadores suficientes
            logger.warning("Liga Crypto: dados insuficientes para %s (%d candles)", tf, len(df))

    # ══════════════════════════════════════════════════════════════
    # PASSO 1: ANÁLISE HIERÁRQUICA (top-down obrigatório)
    # ══════════════════════════════════════════════════════════════

    # 1. SEMANAL — Contexto macro
    result.weekly = analyze_weekly(dfs_ind["1W"])
    logger.info(
        "Liga Crypto [1W]: trend=%s bbwp=%.1f (%s) support=%.0f resistance=%.0f",
        result.weekly.trend.value, result.weekly.bbwp, result.weekly.bbwp_state,
        result.weekly.weekly_support, result.weekly.weekly_resistance,
    )

    # 2. DIÁRIO — Filtro de tendência
    result.daily = analyze_daily(dfs_ind["1D"])
    logger.info(
        "Liga Crypto [1D]: trend=%s price_vs_ma200=%s rsi=%.1f bbwp=%.1f",
        result.daily.trend.value, result.daily.price_vs_sma200,
        result.daily.rsi, result.daily.bbwp,
    )

    # 3. 4H — Setup
    result.h4 = analyze_4h(dfs_ind["4H"])
    logger.info(
        "Liga Crypto [4H]: pattern=%s zone=%s cross=%s adx=%.1f",
        result.h4.pattern, result.h4.zone_type, result.h4.cross_recent, result.h4.adx,
    )

    # 4. 1H — Pré-execução
    result.h1 = analyze_1h(
        dfs_ind["1H"],
        result.h4.stoch_k, result.h4.stoch_d,
        result.daily.stoch_k, result.daily.stoch_d,
    )
    logger.info(
        "Liga Crypto [1H]: trigger=%s stoch_k=%.1f vol_exp=%s",
        result.h1.trigger_signal, result.h1.stoch_k, result.h1.vol_expanding,
    )

    # 5. 15M — Execução (opcional)
    if "15M" in dfs_ind and len(dfs_ind["15M"]) >= 50:
        result.m15 = analyze_15m(dfs_ind["15M"])
        logger.info(
            "Liga Crypto [15M]: trigger=%s stoch_k=%.1f",
            result.m15.trigger_signal, result.m15.stoch_k,
        )

    # ══════════════════════════════════════════════════════════════
    # PASSO 2: VERIFICAR BLOQUEIOS GERAIS
    # ══════════════════════════════════════════════════════════════
    blocks = _check_blocking_conditions(result.weekly, result.daily, result.h4, result.h1)
    if blocks:
        result.decision = Decision.AGUARDAR
        result.justification = "; ".join(blocks)
        result.invalidation_reasons = blocks
        result.alerts.extend(blocks)
        logger.info("Liga Crypto: BLOQUEADO — %s", blocks[0])

    # ══════════════════════════════════════════════════════════════
    # PASSO 3: AVALIAR ENTRADA LONG
    # ══════════════════════════════════════════════════════════════
    long_passed, long_reasons = _check_long_preconditions(
        result.weekly, result.daily, result.h4, result.h1, result.m15,
    )

    if long_passed and not blocks:
        # Calcular SL/TP
        sl, tp1, tp2, tp3 = _calculate_sl_tp_long(result.h4, result.h1)
        entry = result.h1.price

        # Verificar R:R mínimo
        risk = entry - sl
        reward_tp1 = tp1 - entry
        rr = reward_tp1 / risk if risk > 0 else 0

        if rr >= RR_MIN:
            confluences = _count_confluences(
                result.weekly, result.daily, result.h4, result.h1, result.m15, is_long=True,
            )
            exec_tf = result.m15 if result.m15 and result.m15.trigger_signal != "nenhum" else result.h1

            result.decision = Decision.COMPRA
            result.entry_price = entry
            result.stop_loss = sl
            result.tp1 = tp1
            result.tp2 = tp2
            result.tp3 = tp3
            result.sl_distance_pct = (risk / entry) * 100
            result.tp1_pct = (reward_tp1 / entry) * 100
            result.tp2_pct = ((tp2 - entry) / entry) * 100
            result.tp3_pct = ((tp3 - entry) / entry) * 100
            result.rr_tp1 = rr
            result.rr_tp2 = (tp2 - entry) / risk if risk > 0 else 0
            result.confidence = _assess_confidence(
                result.weekly, result.daily, result.h4, result.h1, result.m15, confluences,
            )
            result.justification = (
                f"Confluência de {confluences} fatores: "
                f"semanal/diário alinhados, 4H em {result.h4.zone_type}, "
                f"gatilho {exec_tf.trigger_signal} no {exec_tf.timeframe}, "
                f"volatilidade expandindo"
            )

            # Condições de invalidação
            result.invalidation_reasons = [
                f"Preço abaixo do SL ({sl:,.2f}) — tese invalidada",
                f"Volatilidade contraindo após entrada — sinal frágil",
                f"Divergência RSI contrária aparecendo no 4H",
            ]

            logger.info(
                "Liga Crypto: SINAL LONG | entry=%.2f SL=%.2f TP1=%.2f RR=%.2f conf=%s",
                entry, sl, tp1, rr, result.confidence.value,
            )
        else:
            result.decision = Decision.AGUARDAR
            result.justification = f"R:R ({rr:.2f}) abaixo do mínimo ({RR_MIN}) — entrada descartada"
    elif long_reasons:
        # Store long blocking reasons for context
        pass

    # ══════════════════════════════════════════════════════════════
    # PASSO 4: AVALIAR ENTRADA SHORT
    # ══════════════════════════════════════════════════════════════
    if result.decision not in (Decision.COMPRA,):
        short_passed, short_reasons = _check_short_preconditions(
            result.weekly, result.daily, result.h4, result.h1, result.m15,
        )

        if short_passed and not blocks:
            sl, tp1, tp2, tp3 = _calculate_sl_tp_short(result.h4, result.h1)
            entry = result.h1.price

            risk = sl - entry
            reward_tp1 = entry - tp1
            rr = reward_tp1 / risk if risk > 0 else 0

            if rr >= RR_MIN:
                confluences = _count_confluences(
                    result.weekly, result.daily, result.h4, result.h1, result.m15, is_long=False,
                )
                exec_tf = result.m15 if result.m15 and result.m15.trigger_signal != "nenhum" else result.h1

                result.decision = Decision.VENDA
                result.entry_price = entry
                result.stop_loss = sl
                result.tp1 = tp1
                result.tp2 = tp2
                result.tp3 = tp3
                result.sl_distance_pct = (risk / entry) * 100
                result.tp1_pct = (reward_tp1 / entry) * 100
                result.tp2_pct = ((entry - tp2) / entry) * 100
                result.tp3_pct = ((entry - tp3) / entry) * 100
                result.rr_tp1 = rr
                result.rr_tp2 = (entry - tp2) / risk if risk > 0 else 0
                result.confidence = _assess_confidence(
                    result.weekly, result.daily, result.h4, result.h1, result.m15, confluences,
                )
                result.justification = (
                    f"Confluência de {confluences} fatores: "
                    f"semanal/diário alinhados, 4H em {result.h4.zone_type}, "
                    f"gatilho {exec_tf.trigger_signal} no {exec_tf.timeframe}, "
                    f"volatilidade expandindo"
                )

                result.invalidation_reasons = [
                    f"Preço acima do SL ({sl:,.2f}) — tese invalidada",
                    f"Volatilidade contraindo após entrada — sinal frágil",
                    f"Divergência RSI contrária aparecendo no 4H",
                ]

                logger.info(
                    "Liga Crypto: SINAL SHORT | entry=%.2f SL=%.2f TP1=%.2f RR=%.2f conf=%s",
                    entry, sl, tp1, rr, result.confidence.value,
                )
            else:
                if result.decision not in (Decision.AGUARDAR,):
                    result.decision = Decision.AGUARDAR
                result.justification = f"R:R ({rr:.2f}) abaixo do mínimo ({RR_MIN}) — entrada descartada"

    # ══════════════════════════════════════════════════════════════
    # PASSO 5: AGUARDAR se nenhum sinal
    # ══════════════════════════════════════════════════════════════
    if result.decision not in (Decision.COMPRA, Decision.VENDA):
        if not result.justification:
            # Determinar motivo do AGUARDAR
            exec_tf = result.m15 if result.m15 else result.h1
            if exec_tf.trigger_signal == "nenhum":
                result.justification = "Aguardando gatilho de execução (1H/15M)"
            elif blocks:
                result.justification = "Condições de bloqueio ativas"
            else:
                result.justification = "Ausência de confluência suficiente para entrada"
        result.decision = Decision.AGUARDAR
        result.confidence = Confidence.BAIXA

    # ══════════════════════════════════════════════════════════════
    # PASSO 6: ALERTAS E CONTEXTO SAZONAL
    # ══════════════════════════════════════════════════════════════
    result.seasonal_context = _get_seasonal_context()

    # Adicionar alertas relevantes
    if result.h1.exhaustion != "nenhuma":
        result.alerts.append(
            f"Exaustão tripla detectada ({result.h1.exhaustion}) — alta probabilidade de reversão"
        )

    if result.h4.cross_recent != "nenhum":
        result.alerts.append(
            f"Cruzamento de {result.h4.cross_recent} recente no 4H — possível topo/fundo local antes de continuação"
        )

    if result.weekly.bbwp < BBWP_EXTREME_CONTRACTION:
        result.alerts.append(
            f"BBWP semanal em contração extrema ({result.weekly.bbwp:.1f}%) — grande movimento iminente"
        )

    if result.daily.surubadas:
        result.alerts.append("Médias diárias surubadas — tendência indefinida no curto prazo")

    # Manipulação das 10h (Brasília = UTC-3 = 13h UTC)
    hour_utc = datetime.now(timezone.utc).hour
    if 12 <= hour_utc <= 14 and datetime.now(timezone.utc).weekday() < 5:
        result.alerts.append("Janela de manipulação das 10h (Brasília) — possível armadilha de liquidez")

    if _is_macro_event_window():
        result.alerts.append("Janela de evento macro — não operar 30min antes/depois")

    # "Virada do semanal" (domingo meia-noite UTC)
    dow = datetime.now(timezone.utc).weekday()
    if dow == 6:  # Sunday
        result.alerts.append("Virada do semanal — estruturas de curto prazo podem ser invalidadas")

    return result


# ═══════════════════════════════════════════════════════════════════
# INTEGRATION: Convert to standard Signal for pipeline
# ═══════════════════════════════════════════════════════════════════

def liga_crypto_to_signal(result: LigaCryptoResult):
    """Converte LigaCryptoResult para o Signal padrão do pipeline.
    Isso permite que o resultado da análise Liga Crypto seja processado
    pelo position_tracker, order_executor, etc.

    Returns Signal ou None se não há sinal de entrada.
    """
    if result.decision not in (Decision.COMPRA, Decision.VENDA):
        return None

    from strategy import Signal, SignalType, _make_signal

    # Build a synthetic row for _make_signal
    row_data = {
        "close": result.entry_price,
        "low": result.entry_price - (result.stop_loss - result.entry_price) * 0.1,
        "high": result.entry_price + abs(result.tp1 - result.entry_price) * 0.1,
        "rsi": result.h1.rsi,
        "rsi_delta": result.h1.rsi_delta,
        "macd_hist": 0.0,
        "ema20": result.h1.ema20,
        "ema50": result.h1.ema50 if result.h1.ema50 > 0 else result.h1.sma50,
        "ema200": result.h1.ema200 if result.h1.ema200 > 0 else result.h1.sma200,
        "adx": result.h4.adx,
        "plus_di": result.h4.plus_di,
        "minus_di": result.h4.minus_di,
        "regime": "trending_up" if result.decision == Decision.COMPRA else "trending_down",
        "bb_lower": result.h1.bb_lower,
        "bb_upper": result.h1.bb_upper,
        "bb_width": abs(result.h1.bb_upper - result.h1.bb_lower) / max(result.h1.price, 1) * 100,
        "bb_squeeze_pct": result.h1.bbwp_percentile,
        "bbwp": result.h1.bbwp,
        "volume": result.h1.volume,
        "volume_sma20": result.h1.volume_sma20,
        "volume_sma50": result.h1.volume_sma20,  # proxy
        "atr_percentile": result.h1.bbwp_percentile,  # proxy
        "atr": result.h4.atr if result.h4.atr > 0 else result.h1.atr,
        "ema50_slope": 0.0,
    }

    row = pd.Series(row_data)
    sig_type = SignalType.LONG if result.decision == Decision.COMPRA else SignalType.SHORT

    atr = row_data["atr"]
    sl_mult = abs(result.stop_loss - result.entry_price) / atr if atr > 0 else 1.8
    tp_mult = abs(result.tp1 - result.entry_price) / atr if atr > 0 else 3.0

    return _make_signal(
        sig_type, result.entry_price, result.stop_loss, result.tp1,
        atr, row,
        pullback_type="liga_crypto",
        entry_type="liga_crypto",
        max_bars=168 if result.decision == Decision.COMPRA else 168,
    )


def evaluate_liga_crypto_signal(
    df: pd.DataFrame, timeframe: str = "1h",
    multi_tf_dfs: Optional[dict] = None,
) -> Optional[object]:
    """Entry point compatível com o strategy_router.

    Se multi_tf_dfs for fornecido, usa a análise completa Liga Crypto.
    Caso contrário, retorna None (requer dados multi-timeframe).
    """
    if multi_tf_dfs is None:
        logger.debug("Liga Crypto: sem dados multi-timeframe — pulando")
        return None

    result = analyze_liga_crypto(multi_tf_dfs)
    return liga_crypto_to_signal(result)


def evaluate_liga_crypto_signal_row(
    row: pd.Series, prev_row: pd.Series,
    bar_index: int, timeframe: str = "1h",
    multi_tf_dfs: Optional[dict] = None,
) -> Optional[object]:
    """Entry point compatível com o strategy_router (row-based, para backtest)."""
    if multi_tf_dfs is None:
        return None

    result = analyze_liga_crypto(multi_tf_dfs)
    return liga_crypto_to_signal(result)
