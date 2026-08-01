"""
regime_engine.py
----------------
Motor de deteccao de regime adaptativo v2 com histerese e confidence scoring.

Evolution from the basic _classify_regime() in indicators.py:
  - v1 (indicators.py): 5 regimes simples, sem histerese, sem confianca
  - v2 (este modulo): 9 regimes granulares, histerese anti-whipsaw,
    scoring multi-indicador, transicoes suaves

Regimes detectados:
  STRONG_UPTREND   - Tendencia de alta bem estabelecida (todos indicadores alinham)
  WEAK_UPTREND     - Tendencia de alta enfraquecendo
  STRONG_DOWNTREND - Tendencia de baixa bem estabelecida
  WEAK_DOWNTREND   - Tendencia de baixa enfraquecendo
  RANGING          - Mercado lateral (baixa volatilidade direcional)
  SQUEEZE          - Compressao de volatilidade (BB squeeze + ADX baixo)
  BREAKOUT_BULL    - Expansao de volatilidade para cima
  BREAKOUT_BEAR    - Expansao de volatilidade para baixo
  HIGH_VOLATILITY  - Volatilidade extrema (risco muito alto)

Estrategia recomendada por regime:
  STRONG_UPTREND   -> trend_follow LONG (TP agressivo)
  WEAK_UPTREND     -> trend_follow LONG (TP conservador)
  STRONG_DOWNTREND -> trend_follow SHORT (TP agressivo)
  WEAK_DOWNTREND   -> trend_follow SHORT (TP conservador)
  RANGING          -> mean_reversion (BB bounce)
  SQUEEZE          -> NEUTRO (aguardar breakout)
  BREAKOUT_BULL    -> breakout LONG
  BREAKOUT_BEAR    -> breakout SHORT
  HIGH_VOLATILITY  -> NEUTRO (risco)

Design principles:
  1. Hysteresis: requer N barras consecutivas para confirmar mudanca de regime
     (evita whipsaw em transicoes rapidas)
  2. Confidence: score 0.0-1.0 baseado em quantos indicadores concordam
  3. Multi-timeframe: usa EMA20/50/200 + ADX + DI + BB width + EMA slope
  4. Backward compatible: pode coexistir com o regime v1

Usage:
    from regime_engine import classify_regimes_v2
    df = classify_regimes_v2(df_ind)
    # df agora tem colunas: regime_v2, regime_confidence, regime_bars, regime_strategy
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ==================================================================
# REGIME DEFINITIONS
# ==================================================================

# Numeric codes for efficient numpy operations
REGIME_CODES = {
    "STRONG_UPTREND":   1,
    "WEAK_UPTREND":     2,
    "RANGING":          3,
    "SQUEEZE":          4,
    "BREAKOUT_BULL":    5,
    "BREAKOUT_BEAR":    6,
    "WEAK_DOWNTREND":   7,
    "STRONG_DOWNTREND": 8,
    "HIGH_VOLATILITY":  9,
}

# Inverse mapping
_CODE_TO_REGIME = {v: k for k, v in REGIME_CODES.items()}

# Strategy mapping: which strategy to use per regime
# NOTE: Breakout disabled (no validated edge in testing)
REGIME_STRATEGY = {
    "STRONG_UPTREND":   "trend_follow_long",
    "WEAK_UPTREND":     "trend_follow_long",
    "RANGING":          "mean_reversion",
    "SQUEEZE":          "neutral",
    "BREAKOUT_BULL":    "neutral",      # DISABLED: no edge
    "BREAKOUT_BEAR":    "neutral",      # DISABLED: no edge
    "WEAK_DOWNTREND":   "trend_follow_short",
    "STRONG_DOWNTREND": "trend_follow_short",
    "HIGH_VOLATILITY":  "neutral",
}
# Which regimes allow LONG trades
LONG_REGIMES = {"STRONG_UPTREND", "WEAK_UPTREND", "RANGING", "BREAKOUT_BULL"}
# Which regimes allow SHORT trades
SHORT_REGIMES = {"STRONG_DOWNTREND", "WEAK_DOWNTREND", "RANGING", "BREAKOUT_BEAR"}
# Which regimes should be skipped entirely
NEUTRAL_REGIMES = {"SQUEEZE", "HIGH_VOLATILITY"}


# ==================================================================
# HYSTERESIS CONFIG
# ==================================================================

# Bars of consecutive same-raw-regime required to confirm switch
# Higher = less whipsaw but slower reaction
DEFAULT_HYSTERESIS_BARS = 3


# ==================================================================
# THRESHOLDS
# ==================================================================

# ADX thresholds
ADX_TRENDING_MIN = 22.0       # Below this = not trending
ADX_STRONG_MIN = 35.0          # Above this = strong trend
ADX_EXTREME_MIN = 50.0         # Above this = extreme (potential reversal zone)

# BB Width percentile thresholds
BB_SQUEEZE_MAX = 0.20          # Below 20th percentile = squeeze
BB_NORMAL_MAX = 0.80           # Below 80th = normal volatility
# Above 80th = high volatility

# EMA slope thresholds (from _ema_slope: % change over 20 bars)
EMA_SLOPE_FLAT = 0.3           # |slope| < this = flat (ranging)
EMA_SLOPE_MODERATE = 1.0       # |slope| > this = moderate trend
EMA_SLOPE_STRONG = 2.0         # |slope| > this = strong trend

# DI spread thresholds
DI_SPREAD_TRENDING = 8.0      # |DI+ - DI-| > this = clear directional bias
DI_SPREAD_STRONG = 15.0       # |DI+ - DI-| > this = strong directional bias


# ==================================================================
# CORE CLASSIFICATION
# ==================================================================

def _compute_raw_regimes(df: pd.DataFrame) -> pd.Series:
    """
    Classifica regime raw (sem histerese) para cada candle.

    Usa 5 indicadores-chave:
      1. ADX: forca da tendencia
      2. DI spread: direcao
      3. EMA alignment: EMA20 vs EMA50 vs EMA200
      4. EMA50 slope: velocidade da tendencia
      5. BB width percentile: volatilidade relativa

    Returns pd.Series of regime strings.
    """
    n = len(df)
    regimes = np.full(n, "RANGING", dtype=object)

    # Extract arrays for vectorized computation
    adx = df["adx"].values.astype(float)
    plus_di = df["plus_di"].values.astype(float)
    minus_di = df["minus_di"].values.astype(float)
    close = df["close"].values.astype(float)
    ema20 = df["ema20"].values.astype(float)
    ema50 = df["ema50"].values.astype(float)
    ema200 = df["ema200"].values.astype(float)
    ema50_slope = df["ema50_slope"].values.astype(float)
    bb_width_pct = df["bb_squeeze_pct"].values.astype(float)
    bb_width = df["bb_width"].values.astype(float)

    # Pre-compute derived signals
    di_spread = plus_di - minus_di          # > 0 = bullish, < 0 = bearish
    abs_di_spread = np.abs(di_spread)
    abs_slope = np.abs(ema50_slope)

    # EMA alignment score: +1 per bullish alignment, -1 per bearish
    # Bullish: EMA20 > EMA50 > EMA200
    # Bearish: EMA20 < EMA50 < EMA200
    ema_bull_align = (ema20 > ema50).astype(int) + (ema50 > ema200).astype(int)
    ema_bear_align = (ema20 < ema50).astype(int) + (ema50 < ema200).astype(int)

    # Price vs EMAs
    price_above_ema50 = close > ema50
    price_above_ema200 = close > ema200
    price_below_ema50 = close < ema50
    price_below_ema200 = close < ema200

    for i in range(n):
        if np.isnan(adx[i]) or np.isnan(bb_width_pct[i]):
            continue

        _adx = adx[i]
        _di_spread = di_spread[i]
        _abs_di = abs_di_spread[i]
        _slope = ema50_slope[i]
        _abs_slope = abs_slope[i]
        _bb_pct = bb_width_pct[i]
        _bull_ema = ema_bull_align[i]  # 0, 1, or 2
        _bear_ema = ema_bear_align[i]  # 0, 1, or 2

        # ── HIGH VOLATILITY ──
        # Extreme volatility: BB width > 80th percentile AND (ADX > 50 OR slope extreme)
        if _bb_pct > 0.85 and (_adx > ADX_EXTREME_MIN or _abs_slope > EMA_SLOPE_STRONG * 1.5):
            regimes[i] = "HIGH_VOLATILITY"
            continue

        # ── SQUEEZE ──
        # Low BB width percentile + low ADX = compression (pre-breakout)
        if _bb_pct < BB_SQUEEZE_MAX and _adx < ADX_TRENDING_MIN:
            regimes[i] = "SQUEEZE"
            continue

        # ── BREAKOUT DETECTION ──
        # BB width expanding rapidly (was in low percentile, now mid/high)
        # AND price breaking out of EMA structure
        if _bb_pct > 0.50 and i >= 5:
            # Check if BB width was compressed recently (last 10 bars)
            recent_bb = bb_width_pct[max(0, i-10):i]
            was_compressed = np.nansum(recent_bb < BB_SQUEEZE_MAX) >= 3

            if was_compressed and _abs_di > DI_SPREAD_TRENDING:
                if _di_spread > 5.0 and price_above_ema50[i] and _slope > EMA_SLOPE_FLAT:
                    regimes[i] = "BREAKOUT_BULL"
                    continue
                elif _di_spread < -5.0 and price_below_ema50[i] and _slope < -EMA_SLOPE_FLAT:
                    regimes[i] = "BREAKOUT_BEAR"
                    continue

        # ── RANGING ──
        # Low ADX + flat slope + no clear DI spread = sideways market
        if _adx < ADX_TRENDING_MIN and _abs_slope < EMA_SLOPE_FLAT and _abs_di < DI_SPREAD_TRENDING:
            regimes[i] = "RANGING"
            continue

        # ── TRENDING REGIMES ──
        # Need: ADX >= trending min + directional bias
        if _adx >= ADX_TRENDING_MIN and _abs_di >= DI_SPREAD_TRENDING * 0.5:
            # Determine direction
            is_bullish = (_di_spread > 0 and price_above_ema50[i] and _slope > -EMA_SLOPE_FLAT)
            is_bearish = (_di_spread < 0 and price_below_ema50[i] and _slope < EMA_SLOPE_FLAT)

            if is_bullish:
                # Strong uptrend: high ADX + strong DI + EMA aligned + good slope
                is_strong = (
                    _adx >= ADX_STRONG_MIN and
                    _abs_di >= DI_SPREAD_STRONG and
                    _bull_ema >= 1 and
                    _abs_slope >= EMA_SLOPE_MODERATE
                )
                regimes[i] = "STRONG_UPTREND" if is_strong else "WEAK_UPTREND"
                continue

            elif is_bearish:
                # Strong downtrend: high ADX + strong DI + EMA aligned + good slope
                is_strong = (
                    _adx >= ADX_STRONG_MIN and
                    _abs_di >= DI_SPREAD_STRONG and
                    _bear_ema >= 1 and
                    _abs_slope >= EMA_SLOPE_MODERATE
                )
                regimes[i] = "STRONG_DOWNTREND" if is_strong else "WEAK_DOWNTREND"
                continue

        # ── Default fallback: RANGING ──
        regimes[i] = "RANGING"

    return pd.Series(regimes, index=df.index, dtype=object)


def _compute_confidence(
    df: pd.DataFrame, raw_regimes: pd.Series
) -> pd.Series:
    """
    Computa score de confianca (0.0-1.0) para cada regime.

    Quanto mais indicadores concordam, maior a confianca.
    """
    n = len(df)
    confidence = np.zeros(n, dtype=float)

    adx = df["adx"].values.astype(float)
    plus_di = df["plus_di"].values.astype(float)
    minus_di = df["minus_di"].values.astype(float)
    ema50_slope = df["ema50_slope"].values.astype(float)
    bb_width_pct = df["bb_squeeze_pct"].values.astype(float)
    close = df["close"].values.astype(float)
    ema50 = df["ema50"].values.astype(float)
    ema200 = df["ema200"].values.astype(float)

    for i in range(n):
        regime = raw_regimes.iloc[i]
        score = 0.0
        max_score = 0.0

        # ADX contribution (max 0.25)
        max_score += 0.25
        _adx = adx[i]
        if not np.isnan(_adx):
            if _adx >= ADX_STRONG_MIN:
                score += 0.25
            elif _adx >= ADX_TRENDING_MIN:
                score += 0.15 + 0.10 * ((_adx - ADX_TRENDING_MIN) / (ADX_STRONG_MIN - ADX_TRENDING_MIN))
            else:
                # For ranging/squeeze, low ADX is GOOD
                if regime in ("RANGING", "SQUEEZE"):
                    score += 0.25 * (1.0 - _adx / ADX_TRENDING_MIN)

        # DI spread contribution (max 0.25)
        max_score += 0.25
        di_spread = abs(plus_di[i] - minus_di[i])
        if not np.isnan(di_spread):
            if regime in ("RANGING", "SQUEEZE"):
                score += 0.25 * max(0, 1.0 - di_spread / DI_SPREAD_TRENDING)
            else:
                score += min(0.25, 0.25 * di_spread / DI_SPREAD_STRONG)

        # EMA slope contribution (max 0.25)
        max_score += 0.25
        _slope = abs(ema50_slope[i])
        if not np.isnan(_slope):
            if regime in ("RANGING", "SQUEEZE"):
                score += 0.25 * max(0, 1.0 - _slope / EMA_SLOPE_FLAT)
            else:
                score += min(0.25, 0.25 * _slope / EMA_SLOPE_STRONG)

        # EMA alignment contribution (max 0.25)
        max_score += 0.25
        if regime in ("STRONG_UPTREND", "WEAK_UPTREND"):
            if close[i] > ema50[i] > ema200[i]:
                score += 0.25
            elif close[i] > ema50[i]:
                score += 0.15
        elif regime in ("STRONG_DOWNTREND", "WEAK_DOWNTREND"):
            if close[i] < ema50[i] < ema200[i]:
                score += 0.25
            elif close[i] < ema50[i]:
                score += 0.15
        elif regime in ("RANGING", "SQUEEZE"):
            # For ranging, misaligned EMAs = good (price near middle)
            dist_from_50 = abs(close[i] - ema50[i]) / close[i] * 100
            score += 0.25 * max(0, 1.0 - dist_from_50 / 2.0)

        # Normalize by max_score
        confidence[i] = score / max_score if max_score > 0 else 0.5

    return pd.Series(confidence, index=df.index)


def _apply_hysteresis(
    raw_regimes: pd.Series,
    hysteresis_bars: int = DEFAULT_HYSTERESIS_BARS,
) -> Tuple[pd.Series, pd.Series]:
    """
    Aplica histerese: requer N barras consecutivas do mesmo regime raw
    antes de confirmar a mudanca. Evita whipsaw em transicoes rapidas.

    Returns:
        (smoothed_regimes, regime_bars)
        - smoothed_regimes: Series com regimes pos-histerese
        - regime_bars: Series com contagem de barras no regime atual
    """
    n = len(raw_regimes)
    smoothed = np.empty(n, dtype=object)
    bars = np.zeros(n, dtype=int)

    # Initialize
    smoothed[0] = raw_regimes.iloc[0]
    bars[0] = 1

    current_regime = raw_regimes.iloc[0]
    consecutive = 1
    pending_regime = None
    pending_count = 0

    for i in range(1, n):
        raw = raw_regimes.iloc[i]

        if raw == current_regime:
            # Same regime: reset any pending switch
            consecutive += 1
            pending_regime = None
            pending_count = 0
        else:
            # Different regime detected
            if pending_regime is None:
                # Start tracking potential switch
                pending_regime = raw
                pending_count = 1
            elif raw == pending_regime:
                # Same pending regime, increment counter
                pending_count += 1
                if pending_count >= hysteresis_bars:
                    # Confirmed: switch regime
                    current_regime = pending_regime
                    consecutive = pending_count
                    pending_regime = None
                    pending_count = 0
            else:
                # Different regime than pending: reset
                # (whipsaw protection)
                pending_regime = raw
                pending_count = 1

            consecutive += 1

        smoothed[i] = current_regime
        bars[i] = consecutive

    return pd.Series(smoothed, index=raw_regimes.index), pd.Series(bars, index=raw_regimes.index)


def classify_regimes_v2(
    df: pd.DataFrame,
    hysteresis_bars: int = DEFAULT_HYSTERESIS_BARS,
) -> pd.DataFrame:
    """
    Classifica regimes de mercado v2 com histerese e confidence scoring.

    Requer que o DataFrame ja tenha os indicadores calculados via
    compute_indicators(). Colunas obrigatorias:
      adx, plus_di, minus_di, ema50_slope, bb_squeeze_pct, bb_width,
      close, ema20, ema50, ema200

    Adiciona ao DataFrame:
      - regime_v2: regime classificado (string)
      - regime_v2_raw: regime antes da histerese
      - regime_confidence: score 0.0-1.0
      - regime_bars: numero de barras consecutivas no regime atual
      - regime_strategy: estrategia recomendada (string)

    Parameters:
        df: DataFrame com indicadores (output de compute_indicators)
        hysteresis_bars: barras consecutivas para confirmar mudanca (default 3)

    Returns:
        DataFrame com as novas colunas de regime v2 adicionadas.

    Example:
        df_ind = compute_indicators(df, "1h")
        df_ind = classify_regimes_v2(df_ind)
        last = df_ind.iloc[-1]
        print(last['regime_v2'], last['regime_strategy'], last['regime_confidence'])
    """
    required = {"adx", "plus_di", "minus_di", "ema50_slope", "bb_squeeze_pct",
                "bb_width", "close", "ema20", "ema50", "ema200"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame sem colunas obrigatorias: {missing}")

    out = df.copy()

    # Step 1: Raw classification (no hysteresis)
    out["regime_v2_raw"] = _compute_raw_regimes(out)

    # Step 2: Apply hysteresis
    smoothed, bars = _apply_hysteresis(out["regime_v2_raw"], hysteresis_bars)
    out["regime_v2"] = smoothed
    out["regime_bars"] = bars

    # Step 3: Compute confidence
    out["regime_confidence"] = _compute_confidence(out, out["regime_v2"])

    # Step 4: Map to strategy
    out["regime_strategy"] = out["regime_v2"].map(REGIME_STRATEGY).fillna("neutral")

    # Log regime distribution
    dist = out["regime_v2"].value_counts()
    total = len(out)
    logger.info(
        "Regime v2 distribution (%d bars):\n  %s",
        total,
        "\n  ".join(f"{k}: {v} ({100*v/total:.1f}%)" for k, v in dist.items()),
    )

    return out


def get_regime_params(
    regime: str, confidence: float, base_profile=None
) -> dict:
    """
    Retorna parametros de estrategia adaptados ao regime.

    Cada regime recebe ajustes especificos nos parametros:
    - SL/TP multipliers (mais largo em high vol, mais justo em strong trend)
    - RSI zones (mais extremo em ranging, mais largo em trend)
    - Slope requirements
    - Volume confirmation

    Parameters:
        regime: regime classificado (string)
        confidence: score de confianca 0.0-1.0
        base_profile: StrategyProfile base (opcional, para pegar defaults)

    Returns:
        dict com chaves: sl_mult, tp_mult, rsi_long_range, rsi_short_range,
              require_volume, min_confidence, strategy_type, allow_long, allow_short
    """
    # Base params from profile or defaults
    if base_profile:
        base_sl = base_profile.sl_atr_mult
        base_tp = base_profile.tp_atr_mult
        base_rsi_l = (base_profile.rsi_long_min, base_profile.rsi_long_max)
        base_rsi_s = (base_profile.rsi_short_min, base_profile.rsi_short_max)
    else:
        base_sl = 2.12
        base_tp = 8.50
        base_rsi_l = (28.0, 48.0)
        base_rsi_s = (55.0, 75.0)

    # Regime-specific adjustments
    # DESIGN PRINCIPLE: "Don't fix what isn't broken"
    # Trend-following params were optimized via P2+WF grid search.
    # ONLY change strategy type for non-trending regimes (RANGING, BREAKOUT).
    # For all trend regimes, use EXACT profile params.
    if regime == "STRONG_UPTREND":
        return {
            "strategy_type": "trend_follow",
            "allow_long": True, "allow_short": False,
            "sl_mult": base_sl,              # EXACT profile params (P2+WF optimized)
            "tp_mult": base_tp,              # EXACT profile params
            "rsi_long_range": base_rsi_l,    # EXACT profile params
            "rsi_short_range": (99.0, 99.0), # Block shorts
            "require_volume": False,
            "min_confidence": 0.3,
            "description": "Tendencia de alta forte — trend-following LONG (params otimizados)",
        }

    elif regime == "WEAK_UPTREND":
        return {
            "strategy_type": "trend_follow",
            "allow_long": True, "allow_short": False,
            "sl_mult": base_sl,              # EXACT profile params
            "tp_mult": base_tp,              # EXACT profile params
            "rsi_long_range": base_rsi_l,    # EXACT profile params
            "rsi_short_range": (99.0, 99.0),
            "require_volume": False,
            "min_confidence": 0.4,
            "description": "Tendencia de alta fraca — trend-following LONG (params otimizados)",
        }

    elif regime == "STRONG_DOWNTREND":
        return {
            "strategy_type": "trend_follow",
            "allow_long": False, "allow_short": True,
            "sl_mult": base_sl,              # EXACT profile params
            "tp_mult": base_tp,              # EXACT profile params
            "rsi_long_range": (0.0, 1.0),
            "rsi_short_range": base_rsi_s,    # EXACT profile params
            "require_volume": False,
            "min_confidence": 0.3,
            "description": "Tendencia de baixa forte — trend-following SHORT (params otimizados)",
        }

    elif regime == "WEAK_DOWNTREND":
        return {
            "strategy_type": "trend_follow",
            "allow_long": False, "allow_short": True,
            "sl_mult": base_sl,              # EXACT profile params
            "tp_mult": base_tp,              # EXACT profile params
            "rsi_long_range": (0.0, 1.0),
            "rsi_short_range": base_rsi_s,    # EXACT profile params
            "require_volume": False,
            "min_confidence": 0.4,
            "description": "Tendencia de baixa fraca — trend-following SHORT (params otimizados)",
        }

    elif regime == "RANGING":
        return {
            "strategy_type": "mean_reversion",
            "allow_long": True, "allow_short": True,
            "sl_mult": base_sl * 1.5,          # SL largo — ranging e ruidoso
            "tp_mult": base_sl * 3.0,          # TP = 3x SL (R:R fixo 3:1)
            "rsi_long_range": (20.0, 42.0),    # RSI oversold — comprar (relaxado)
            "rsi_short_range": (58.0, 80.0),   # RSI overbought — vender (relaxado)
            "require_volume": False,
            "min_confidence": 0.2,
            "description": "Mercado lateral — mean-reversion (BB bounce)",
        }

    elif regime == "SQUEEZE":
        return {
            "strategy_type": "neutral",
            "allow_long": False, "allow_short": False,
            "sl_mult": 0, "tp_mult": 0,
            "rsi_long_range": (0, 0),
            "rsi_short_range": (0, 0),
            "require_volume": False,
            "min_confidence": 0.9,             # Very high threshold
            "description": "Compressao de volatilidade — aguardar breakout",
        }

    elif regime == "BREAKOUT_BULL":
        return {
            "strategy_type": "neutral",     # DISABLED: no validated edge
            "allow_long": False, "allow_short": False,
            "sl_mult": base_sl, "tp_mult": base_tp,
            "rsi_long_range": (0, 0),
            "rsi_short_range": (0, 0),
            "require_volume": False,
            "min_confidence": 0.99,
            "description": "Breakout bullish — DISABLED (no validated edge)",
        }

    elif regime == "BREAKOUT_BEAR":
        return {
            "strategy_type": "neutral",     # DISABLED: no validated edge
            "allow_long": False, "allow_short": False,
            "sl_mult": base_sl, "tp_mult": base_tp,
            "rsi_long_range": (0, 0),
            "rsi_short_range": (0, 0),
            "require_volume": False,
            "min_confidence": 0.99,
            "description": "Breakout bearish — DISABLED (no validated edge)",
        }

    elif regime == "HIGH_VOLATILITY":
        return {
            "strategy_type": "neutral",
            "allow_long": False, "allow_short": False,
            "sl_mult": 0, "tp_mult": 0,
            "rsi_long_range": (0, 0),
            "rsi_short_range": (0, 0),
            "require_volume": False,
            "min_confidence": 0.99,
            "description": "Volatilidade extrema — sem trades (risco)",
        }

    # Fallback
    return {
        "strategy_type": "neutral",
        "allow_long": False, "allow_short": False,
        "sl_mult": base_sl, "tp_mult": base_tp,
        "rsi_long_range": base_rsi_l,
        "rsi_short_range": base_rsi_s,
        "require_volume": False,
        "min_confidence": 0.5,
        "description": "Regime desconhecido — neutro",
    }


def get_regime_summary(df: pd.DataFrame) -> str:
    """
    Retorna resumo textual da distribuicao de regimes no DataFrame.
    """
    if "regime_v2" not in df.columns:
        return "regime_v2 nao calculado. Use classify_regimes_v2() primeiro."

    dist = df["regime_v2"].value_counts()
    total = len(df)
    lines = [f"Regime v2 Distribution ({total:,} bars):"]
    for regime, count in dist.items():
        pct = 100 * count / total
        strategy = REGIME_STRATEGY.get(regime, "?")
        lines.append(f"  {regime:20s}: {count:6,} ({pct:5.1f}%) -> {strategy}")
    return "\n".join(lines)
