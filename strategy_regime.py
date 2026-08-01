"""
strategy_regime.py
-----------------
Avaliacao de sinais regime-aware: trend-following, mean-reversion e breakout.

Este modulo e o roteador principal do sistema regime-switching.
Para cada candle, ele:
  1. Identifica o regime atual (via regime_engine)
  2. Obtem parametros adaptados ao regime
  3. Chama a estrategia apropriada
  4. Retorna o sinal com metadados de regime

Estrategias implementadas:
  - trend_follow: CTEV classico (pullback em tendencia) - ja validado
  - mean_reversion: BB bounce em mercados laterais
  - breakout: Entrada em expansao de volatilidade
  - neutral: Sem trade (squeeze, high vol)

Integration:
  O modulo strategy.py continua funcionando isoladamente (backward compat).
  Este modulo e usado quando regime-switching esta ativo.
"""
from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from strategy_profiles import StrategyProfile

from strategy import (
    Signal, SignalType,
    evaluate_long as _evaluate_long,
    evaluate_short as _evaluate_short,
    _price_near_fib,
    _in_fib_zone,
)
from regime_engine import (
    get_regime_params,
    REGIME_STRATEGY,
    LONG_REGIMES,
    SHORT_REGIMES,
    NEUTRAL_REGIMES,
)

logger = logging.getLogger(__name__)


# ==================================================================
# MEAN-REVERSION STRATEGY (RANGING markets)
# ==================================================================

def evaluate_mean_reversion_long(
    row: pd.Series, params: dict, base_profile: Optional[StrategyProfile] = None
) -> Optional[Signal]:
    """
    Mean-reversion LONG para mercados laterais.

    Logica: Preco tocou/quase tocou a BB inferior, RSI em zona de sobrevenda,
    expectativa de retorno a media (BB middle).

    Condicoes:
      1. RSI em zona oversold (params["rsi_long_range"])
      2. Preco proximo a BB inferior (low <= bb_lower * 1.01 ou close < bb_middle)
      3. BB width moderada (nao muito apertada nem muito larga)
      4. ATR nao extremo
      5. Nenhuma tendencia forte (EMA50 slope flat)

    SL: ABAIXO da BB inferior (protege contra breakout falso)
    TP: BB middle band (retorno a media)
    """
    close = float(row["close"])
    low = float(row["low"])
    high = float(row["high"])
    bb_lower = float(row["bb_lower"])
    bb_middle = float(row["bb_middle"])
    bb_upper = float(row["bb_upper"])
    bb_w = float(row.get("bb_width", 0.0))
    bb_sq = float(row.get("bb_squeeze_pct", 0.5))
    rsi = float(row["rsi"])
    rsi_delta = float(row.get("rsi_delta", 0.0))
    macd_hist = float(row.get("macd_hist", 0.0))
    macd_val = float(row.get("macd", 0.0))
    macd_sig = float(row.get("macd_signal", 0.0))
    atr = float(row["atr"])
    atr_pct = float(row.get("atr_percentile", 0.5))
    ema20 = float(row["ema20"])
    ema50 = float(row["ema50"])
    ema200 = float(row["ema200"])
    adx = float(row.get("adx", 0.0))
    plus_di = float(row.get("plus_di", 0.0))
    minus_di = float(row.get("minus_di", 0.0))
    ema50_slope = float(row.get("ema50_slope", 0.0))
    volume = float(row["volume"])
    volume_sma20 = float(row["volume_sma20"])
    volume_sma50 = float(row.get("volume_sma50", 0.0))
    regime = str(row.get("regime_v2", row.get("regime", "")))
    fib_0382 = float(row.get("fib_0382", float("nan")))
    fib_0500 = float(row.get("fib_0500", float("nan")))
    fib_0618 = float(row.get("fib_0618", float("nan")))
    fib_dir = int(row.get("fib_direction", 0))
    fib_prox = float(row.get("fib_proximity", float("nan")))
    ts = row.name

    rsi_min, rsi_max = params["rsi_long_range"]
    sl_mult = params["sl_mult"]
    tp_mult = params["tp_mult"]

    # 1. RSI em zona de sobrevenda
    if not (rsi_min <= rsi <= rsi_max):
        return None

    # 2. Preco proximo a BB inferior
    # Aceita se: low tocou BB lower, OU close esta no quartil inferior do BB
    bb_range = bb_upper - bb_lower
    if bb_range <= 0:
        return None
    bb_position = (close - bb_lower) / bb_range  # 0 = at lower, 1 = at upper

    if not (low <= bb_lower * 1.01 or bb_position < 0.45):
        return None

    # 3. BB width nao muito apertada (nao squeeze) nem extrema
    if bb_sq < 0.10 or bb_sq > 0.90:
        return None

    # 4. ATR nao extremo (evita entrar em spikes)
    if atr_pct < 0.10 or atr_pct > 0.85:
        return None

    # 5. Nenhuma tendencia forte (EMA50 slope flat)
    if abs(ema50_slope) > 2.5:
        return None

    # 6. MACD: preferencia por MACD flat ou com sinal de reversao
    # Nao e obrigatorio mas adiciona qualidade
    # (sem filtro aqui — ranging ja garante contexto)

    # ── Gestao de risco: SL/TP ──
    # SL: abaixo da BB inferior, com ATR buffer
    sl_level = min(close - sl_mult * atr, bb_lower - 0.5 * atr)
    # TP: BB middle (retorno a media) — se muito perto, usa ATR multiplier
    tp_bb = bb_middle
    tp_atr = close + tp_mult * atr
    tp_level = max(tp_bb, close + 1.0 * atr)  # minimo 1x ATR de ganho
    tp_level = min(tp_level, tp_atr)  # mas nao mais que tp_mult * ATR

    if sl_level <= 0:
        return None

    # Verifica R:R minimo de 1.5:1
    risk = close - sl_level
    reward = tp_level - close
    if risk <= 0 or reward / risk < 1.2:  # Relaxed to 1.2:1 for ranging
        return None

    logger.info(
        "SINAL MR LONG | entry=%.2f SL=%.2f TP=%.2f ATR=%.2f "
        "RSI=%.1f BB_pos=%.2f regime=%s",
        close, sl_level, tp_level, atr, rsi, bb_position, regime,
    )

    return Signal(
        type=SignalType.LONG,
        entry_price=close,
        stop_loss=sl_level,
        take_profit=tp_level,
        atr=atr, rsi=rsi, rsi_delta=rsi_delta, macd_hist=macd_hist,
        ema20=ema20, ema50=ema50, ema200=ema200,
        adx=adx, plus_di=plus_di, minus_di=minus_di, regime=regime,
        bb_lower=bb_lower, bb_upper=bb_upper, bb_width=bb_w,
        bb_squeeze_pct=bb_sq,
        volume=volume, volume_sma20=volume_sma20, volume_sma50=volume_sma50,
        atr_percentile=atr_pct,
        fib_0382=fib_0382 if not np.isnan(fib_0382) else 0.0,
        fib_0500=fib_0500 if not np.isnan(fib_0500) else 0.0,
        fib_0618=fib_0618 if not np.isnan(fib_0618) else 0.0,
        fib_direction=fib_dir, fib_proximity=fib_prox,
        pullback_type="bb_bounce_long",
        ema50_slope=ema50_slope, timestamp=ts,
    )


def evaluate_mean_reversion_short(
    row: pd.Series, params: dict, base_profile: Optional[StrategyProfile] = None
) -> Optional[Signal]:
    """
    Mean-reversion SHORT para mercados laterais.

    Logica: Preco tocou a BB superior, RSI em zona de sobrecompra,
    expectativa de retorno a media (BB middle).
    """
    close = float(row["close"])
    low = float(row["low"])
    high = float(row["high"])
    bb_lower = float(row["bb_lower"])
    bb_middle = float(row["bb_middle"])
    bb_upper = float(row["bb_upper"])
    bb_w = float(row.get("bb_width", 0.0))
    bb_sq = float(row.get("bb_squeeze_pct", 0.5))
    rsi = float(row["rsi"])
    rsi_delta = float(row.get("rsi_delta", 0.0))
    macd_hist = float(row.get("macd_hist", 0.0))
    macd_val = float(row.get("macd", 0.0))
    macd_sig = float(row.get("macd_signal", 0.0))
    atr = float(row["atr"])
    atr_pct = float(row.get("atr_percentile", 0.5))
    ema20 = float(row["ema20"])
    ema50 = float(row["ema50"])
    ema200 = float(row["ema200"])
    adx = float(row.get("adx", 0.0))
    plus_di = float(row.get("plus_di", 0.0))
    minus_di = float(row.get("minus_di", 0.0))
    ema50_slope = float(row.get("ema50_slope", 0.0))
    volume = float(row["volume"])
    volume_sma20 = float(row["volume_sma20"])
    volume_sma50 = float(row.get("volume_sma50", 0.0))
    regime = str(row.get("regime_v2", row.get("regime", "")))
    fib_0382 = float(row.get("fib_0382", float("nan")))
    fib_0500 = float(row.get("fib_0500", float("nan")))
    fib_0618 = float(row.get("fib_0618", float("nan")))
    fib_dir = int(row.get("fib_direction", 0))
    fib_prox = float(row.get("fib_proximity", float("nan")))
    ts = row.name

    rsi_min, rsi_max = params["rsi_short_range"]
    sl_mult = params["sl_mult"]
    tp_mult = params["tp_mult"]

    # 1. RSI em zona de sobrecompra
    if not (rsi_min <= rsi <= rsi_max):
        return None

    # 2. Preco proximo a BB superior
    bb_range = bb_upper - bb_lower
    if bb_range <= 0:
        return None
    bb_position = (close - bb_lower) / bb_range

    if not (high >= bb_upper * 0.99 or bb_position > 0.55):
        return None

    # 3. BB width
    if bb_sq < 0.10 or bb_sq > 0.90:
        return None

    # 4. ATR
    if atr_pct < 0.10 or atr_pct > 0.85:
        return None

    # 5. Nenhuma tendencia forte
    if abs(ema50_slope) > 2.5:
        return None

    # ── Gestao de risco ──
    sl_level = max(close + sl_mult * atr, bb_upper + 0.5 * atr)
    tp_bb = bb_middle
    tp_atr = close - tp_mult * atr
    tp_level = min(tp_bb, close - 1.0 * atr)
    tp_level = max(tp_level, tp_atr)

    risk = sl_level - close
    reward = close - tp_level
    if risk <= 0 or reward / risk < 1.2:  # Relaxed to 1.2:1 for ranging
        return None

    logger.info(
        "SINAL MR SHORT | entry=%.2f SL=%.2f TP=%.2f ATR=%.2f "
        "RSI=%.1f BB_pos=%.2f regime=%s",
        close, sl_level, tp_level, atr, rsi, bb_position, regime,
    )

    return Signal(
        type=SignalType.SHORT,
        entry_price=close,
        stop_loss=sl_level,
        take_profit=tp_level,
        atr=atr, rsi=rsi, rsi_delta=rsi_delta, macd_hist=macd_hist,
        ema20=ema20, ema50=ema50, ema200=ema200,
        adx=adx, plus_di=plus_di, minus_di=minus_di, regime=regime,
        bb_lower=bb_lower, bb_upper=bb_upper, bb_width=bb_w,
        bb_squeeze_pct=bb_sq,
        volume=volume, volume_sma20=volume_sma20, volume_sma50=volume_sma50,
        atr_percentile=atr_pct,
        fib_0382=fib_0382 if not np.isnan(fib_0382) else 0.0,
        fib_0500=fib_0500 if not np.isnan(fib_0500) else 0.0,
        fib_0618=fib_0618 if not np.isnan(fib_0618) else 0.0,
        fib_direction=fib_dir, fib_proximity=fib_prox,
        pullback_type="bb_bounce_short",
        ema50_slope=ema50_slope, timestamp=ts,
    )


# ==================================================================
# BREAKOUT STRATEGY (volatility expansion)
# ==================================================================

def evaluate_breakout_long(
    row: pd.Series, params: dict, base_profile: Optional[StrategyProfile] = None
) -> Optional[Signal]:
    """
    Breakout LONG: preco rompe BB superior com volume e momentum.

    Condicoes:
      1. Close acima da BB superior (breakout confirmado)
      2. Volume acima da media (confirmacao)
      3. RSI na faixa de momentum (nao sobrevenda)
      4. MACD histogram positivo ou cruzando para cima
      5. ATR percentil > 0.40 (volatilidade expandindo)
    """
    close = float(row["close"])
    low = float(row["low"])
    high = float(row["high"])
    bb_lower = float(row["bb_lower"])
    bb_middle = float(row["bb_middle"])
    bb_upper = float(row["bb_upper"])
    bb_w = float(row.get("bb_width", 0.0))
    bb_sq = float(row.get("bb_squeeze_pct", 0.5))
    rsi = float(row["rsi"])
    rsi_delta = float(row.get("rsi_delta", 0.0))
    macd_hist = float(row.get("macd_hist", 0.0))
    macd_val = float(row.get("macd", 0.0))
    macd_sig = float(row.get("macd_signal", 0.0))
    atr = float(row["atr"])
    atr_pct = float(row.get("atr_percentile", 0.5))
    ema20 = float(row["ema20"])
    ema50 = float(row["ema50"])
    ema200 = float(row["ema200"])
    adx = float(row.get("adx", 0.0))
    plus_di = float(row.get("plus_di", 0.0))
    minus_di = float(row.get("minus_di", 0.0))
    ema50_slope = float(row.get("ema50_slope", 0.0))
    volume = float(row["volume"])
    volume_sma20 = float(row["volume_sma20"])
    volume_sma50 = float(row.get("volume_sma50", 0.0))
    regime = str(row.get("regime_v2", row.get("regime", "")))
    fib_0382 = float(row.get("fib_0382", float("nan")))
    fib_0500 = float(row.get("fib_0500", float("nan")))
    fib_0618 = float(row.get("fib_0618", float("nan")))
    fib_dir = int(row.get("fib_direction", 0))
    fib_prox = float(row.get("fib_proximity", float("nan")))
    ts = row.name

    rsi_min, rsi_max = params["rsi_long_range"]
    sl_mult = params["sl_mult"]
    tp_mult = params["tp_mult"]

    # 1. Breakout: close acima da BB superior
    if close <= bb_upper:
        return None

    # 2. Volume de confirmacao
    if volume_sma20 > 0 and volume < volume_sma20 * 1.0:
        return None

    # 3. RSI em zona de momentum
    if not (rsi_min <= rsi <= rsi_max):
        return None

    # 4. MACD positivo ou melhorando
    if macd_hist < 0 and macd_val < macd_sig:
        return None

    # 5. Volatilidade expandindo
    if atr_pct < 0.40:
        return None

    # ── Gestao de risco ──
    # SL: abaixo do candle ou BB middle
    sl_level = min(low - 0.1 * atr, bb_middle)
    tp_level = close + tp_mult * atr

    if sl_level <= 0:
        return None

    risk = close - sl_level
    reward = tp_level - close
    if risk <= 0 or reward / risk < 2.0:  # Minimo R:R 2:1 para breakout
        return None

    logger.info(
        "SINAL BK LONG | entry=%.2f SL=%.2f TP=%.2f ATR=%.2f "
        "RSI=%.1f VOL_ratio=%.2f regime=%s",
        close, sl_level, tp_level, atr, rsi,
        volume / max(volume_sma20, 1), regime,
    )

    return Signal(
        type=SignalType.LONG,
        entry_price=close,
        stop_loss=sl_level,
        take_profit=tp_level,
        atr=atr, rsi=rsi, rsi_delta=rsi_delta, macd_hist=macd_hist,
        ema20=ema20, ema50=ema50, ema200=ema200,
        adx=adx, plus_di=plus_di, minus_di=minus_di, regime=regime,
        bb_lower=bb_lower, bb_upper=bb_upper, bb_width=bb_w,
        bb_squeeze_pct=bb_sq,
        volume=volume, volume_sma20=volume_sma20, volume_sma50=volume_sma50,
        atr_percentile=atr_pct,
        fib_0382=fib_0382 if not np.isnan(fib_0382) else 0.0,
        fib_0500=fib_0500 if not np.isnan(fib_0500) else 0.0,
        fib_0618=fib_0618 if not np.isnan(fib_0618) else 0.0,
        fib_direction=fib_dir, fib_proximity=fib_prox,
        pullback_type="breakout_long",
        ema50_slope=ema50_slope, timestamp=ts,
    )


def evaluate_breakout_short(
    row: pd.Series, params: dict, base_profile: Optional[StrategyProfile] = None
) -> Optional[Signal]:
    """
    Breakout SHORT: preco rompe BB inferior com volume e momentum.
    """
    close = float(row["close"])
    low = float(row["low"])
    high = float(row["high"])
    bb_lower = float(row["bb_lower"])
    bb_middle = float(row["bb_middle"])
    bb_upper = float(row["bb_upper"])
    bb_w = float(row.get("bb_width", 0.0))
    bb_sq = float(row.get("bb_squeeze_pct", 0.5))
    rsi = float(row["rsi"])
    rsi_delta = float(row.get("rsi_delta", 0.0))
    macd_hist = float(row.get("macd_hist", 0.0))
    macd_val = float(row.get("macd", 0.0))
    macd_sig = float(row.get("macd_signal", 0.0))
    atr = float(row["atr"])
    atr_pct = float(row.get("atr_percentile", 0.5))
    ema20 = float(row["ema20"])
    ema50 = float(row["ema50"])
    ema200 = float(row["ema200"])
    adx = float(row.get("adx", 0.0))
    plus_di = float(row.get("plus_di", 0.0))
    minus_di = float(row.get("minus_di", 0.0))
    ema50_slope = float(row.get("ema50_slope", 0.0))
    volume = float(row["volume"])
    volume_sma20 = float(row["volume_sma20"])
    volume_sma50 = float(row.get("volume_sma50", 0.0))
    regime = str(row.get("regime_v2", row.get("regime", "")))
    fib_0382 = float(row.get("fib_0382", float("nan")))
    fib_0500 = float(row.get("fib_0500", float("nan")))
    fib_0618 = float(row.get("fib_0618", float("nan")))
    fib_dir = int(row.get("fib_direction", 0))
    fib_prox = float(row.get("fib_proximity", float("nan")))
    ts = row.name

    rsi_min, rsi_max = params["rsi_short_range"]
    sl_mult = params["sl_mult"]
    tp_mult = params["tp_mult"]

    # 1. Breakout: close abaixo da BB inferior
    if close >= bb_lower:
        return None

    # 2. Volume
    if volume_sma20 > 0 and volume < volume_sma20 * 1.0:
        return None

    # 3. RSI
    if not (rsi_min <= rsi <= rsi_max):
        return None

    # 4. MACD negativo ou piorando
    if macd_hist > 0 and macd_val > macd_sig:
        return None

    # 5. Volatilidade expandindo
    if atr_pct < 0.40:
        return None

    # ── Gestao de risco ──
    sl_level = max(high + 0.1 * atr, bb_middle)
    tp_level = close - tp_mult * atr

    risk = sl_level - close
    reward = close - tp_level
    if risk <= 0 or reward / risk < 2.0:
        return None

    logger.info(
        "SINAL BK SHORT | entry=%.2f SL=%.2f TP=%.2f ATR=%.2f "
        "RSI=%.1f VOL_ratio=%.2f regime=%s",
        close, sl_level, tp_level, atr, rsi,
        volume / max(volume_sma20, 1), regime,
    )

    return Signal(
        type=SignalType.SHORT,
        entry_price=close,
        stop_loss=sl_level,
        take_profit=tp_level,
        atr=atr, rsi=rsi, rsi_delta=rsi_delta, macd_hist=macd_hist,
        ema20=ema20, ema50=ema50, ema200=ema200,
        adx=adx, plus_di=plus_di, minus_di=minus_di, regime=regime,
        bb_lower=bb_lower, bb_upper=bb_upper, bb_width=bb_w,
        bb_squeeze_pct=bb_sq,
        volume=volume, volume_sma20=volume_sma20, volume_sma50=volume_sma50,
        atr_percentile=atr_pct,
        fib_0382=fib_0382 if not np.isnan(fib_0382) else 0.0,
        fib_0500=fib_0500 if not np.isnan(fib_0500) else 0.0,
        fib_0618=fib_0618 if not np.isnan(fib_0618) else 0.0,
        fib_direction=fib_dir, fib_proximity=fib_prox,
        pullback_type="breakout_short",
        ema50_slope=ema50_slope, timestamp=ts,
    )


# ==================================================================
# REGIME-AWARE SIGNAL ROUTER
# ==================================================================

def evaluate_signal_regime_aware(
    df_ind: pd.DataFrame,
    profile: Optional[StrategyProfile] = None,
    hysteresis_bars: int = 3,
) -> Optional[Signal]:
    """
    Ponto de entrada principal do regime-switching.

    1. Classifica regime v2 com histerese
    2. Obtem parametros adaptados ao regime
    3. Roteia para a estrategia correta

    Parameters:
        df_ind: DataFrame com indicadores (output de compute_indicators)
        profile: StrategyProfile base (opcional)
        hysteresis_bars: barras para histerese (default 3)

    Returns:
        Signal ou None (se neutro ou sem sinal)
    """
    if df_ind.empty:
        return None

    # Ensure regime v2 is classified
    if "regime_v2" not in df_ind.columns:
        from regime_engine import classify_regimes_v2
        df_ind = classify_regimes_v2(df_ind, hysteresis_bars=hysteresis_bars)

    last = df_ind.iloc[-1]
    regime = str(last.get("regime_v2", ""))
    confidence = float(last.get("regime_confidence", 0.5))

    # Get regime-specific params
    params = get_regime_params(regime, confidence, base_profile=profile)

    # Check if regime allows trading
    strategy_type = params["strategy_type"]
    if strategy_type == "neutral":
        return None

    # Check minimum confidence
    if confidence < params["min_confidence"]:
        logger.debug(
            "Regime %s confidence %.2f < min %.2f — sem sinal",
            regime, confidence, params["min_confidence"],
        )
        return None

    # Route to appropriate strategy
    if strategy_type == "mean_reversion":
        if params["allow_long"]:
            sig = evaluate_mean_reversion_long(last, params, base_profile=profile)
            if sig is not None:
                return sig
        if params["allow_short"]:
            sig = evaluate_mean_reversion_short(last, params, base_profile=profile)
            return sig

    elif strategy_type == "breakout":
        if params["allow_long"]:
            sig = evaluate_breakout_long(last, params, base_profile=profile)
            if sig is not None:
                return sig
        if params["allow_short"]:
            sig = evaluate_breakout_short(last, params, base_profile=profile)
            return sig

    elif strategy_type == "trend_follow":
        # Use existing CTEV trend-following with regime-adapted RSI/SL/TP
        #
        # v7.1 FIX: WEAK_UPTREND LONG ADX floor.
        # _evaluate_trend_long_adapted lacks ADX filter (unlike original
        # evaluate_long). In WEAK_UPTREND, low-ADX entries (21-25) have
        # 71% loss rate. Require ADX >= 22.0 (regime_engine's trending min).
        if params["allow_long"] and regime == "WEAK_UPTREND":
            _adx_val = float(last.get("adx", 0))
            if not np.isnan(_adx_val) and _adx_val < 22.0:
                logger.debug(
                    "WEAK_UPTREND LONG blocked: ADX %.1f < 22.0", _adx_val,
                )
                return None

        if params["allow_long"]:
            sig = _evaluate_trend_long_adapted(last, params, base_profile=profile)
            if sig is not None:
                return sig
        if params["allow_short"]:
            sig = _evaluate_trend_short_adapted(last, params, base_profile=profile)
            return sig

    return None


def _evaluate_trend_long_adapted(
    row: pd.Series, params: dict, base_profile: Optional[StrategyProfile] = None
) -> Optional[Signal]:
    """
    Trend-following LONG com parametros adaptados pelo regime.

    Reusa a logica core do CTEV (EMA alignment, pullback, etc.)
    mas sobrescreve SL/TP e RSI zones conforme o regime.
    """
    close = float(row["close"])
    low = float(row["low"])
    high = float(row["high"])
    ema20 = float(row["ema20"])
    ema50 = float(row["ema50"])
    ema200 = float(row["ema200"])
    bb_lower = float(row["bb_lower"])
    bb_upper = float(row["bb_upper"])
    rsi = float(row["rsi"])
    rsi_delta = float(row.get("rsi_delta", 0.0))
    macd_hist = float(row.get("macd_hist", 0.0))
    macd_val = float(row.get("macd", 0.0))
    macd_sig = float(row.get("macd_signal", 0.0))
    volume = float(row["volume"])
    volume_sma20 = float(row["volume_sma20"])
    volume_sma50 = float(row.get("volume_sma50", 0.0))
    atr = float(row["atr"])
    atr_pct = float(row.get("atr_percentile", 0.5))
    bb_w = float(row.get("bb_width", 0.0))
    bb_sq = float(row.get("bb_squeeze_pct", 0.5))
    adx = float(row.get("adx", 0.0))
    plus_di = float(row.get("plus_di", 0.0))
    minus_di = float(row.get("minus_di", 0.0))
    regime = str(row.get("regime_v2", row.get("regime", "")))
    ema50_slope = float(row.get("ema50_slope", 0.0))
    fib_0382 = float(row.get("fib_0382", float("nan")))
    fib_0500 = float(row.get("fib_0500", float("nan")))
    fib_0618 = float(row.get("fib_0618", float("nan")))
    fib_dir = int(row.get("fib_direction", 0))
    fib_prox = float(row.get("fib_proximity", float("nan")))
    ts = row.name

    rsi_min, rsi_max = params["rsi_long_range"]
    sl_mult = params["sl_mult"]
    tp_mult = params["tp_mult"]

    # Get base profile for non-regime params
    if base_profile:
        _adx_min = base_profile.adx_min
        _fib_tol = base_profile.fib_tolerance_pct
        _slope_min = base_profile.ema50_slope_min
        _atr_pct_min = base_profile.atr_pct_min
        _atr_pct_max = base_profile.atr_pct_max
    else:
        _adx_min = 30.0
        _fib_tol = 0.025
        _slope_min = -1.0
        _atr_pct_min = 0.10
        _atr_pct_max = 0.90

    # 1. TENDENCIA: Dual EMA — uptrend
    if not (close > ema50 and ema50 > ema200):
        return None

    # 2. SLOPE
    if pd.isna(ema50_slope) or ema50_slope <= _slope_min:
        return None

    # 3. PULLBACK: Fibonacci OU EMA touch
    pullback_type = None
    in_fib = _in_fib_zone(close, fib_0382, fib_0618, fib_dir)
    if in_fib and fib_dir == 1:
        pullback_type = "fibonacci"
    else:
        if fib_dir == 1:
            if (_price_near_fib(low, fib_0382, _fib_tol) or
                    _price_near_fib(low, fib_0500, _fib_tol) or
                    _price_near_fib(low, fib_0618, _fib_tol)):
                pullback_type = "fibonacci"

    if pullback_type is None:
        if bool(row.get("ema20_touched", False)) and close > ema20:
            pullback_type = "ema20_touch"

    if pullback_type is None:
        if bool(row.get("ema50_touched", False)) and close > ema50:
            pullback_type = "ema50_touch"

    if pullback_type is None:
        return None

    # 4. RSI (adapted by regime)
    if not (rsi_min <= rsi <= rsi_max):
        return None

    # 5. ATR
    if not (_atr_pct_min <= atr_pct <= _atr_pct_max):
        return None

    # 6. Volume (if required by regime)
    if params["require_volume"] and volume_sma50 > 0:
        if volume < volume_sma50 * 0.3:
            return None

    # ── SL/TP (adapted by regime) ──
    stop_loss = close - sl_mult * atr
    take_profit = close + tp_mult * atr

    if stop_loss <= 0:
        return None

    logger.info(
        "SIGNAL TF LONG (regime=%s) | entry=%.2f SL=%.2f TP=%.2f ATR=%.2f "
        "RSI=%.1f pullback=%s SL_mult=%.2f TP_mult=%.2f",
        regime, close, stop_loss, take_profit, atr, rsi, pullback_type, sl_mult, tp_mult,
    )

    return Signal(
        type=SignalType.LONG,
        entry_price=close, stop_loss=stop_loss, take_profit=take_profit,
        atr=atr, rsi=rsi, rsi_delta=rsi_delta, macd_hist=macd_hist,
        ema20=ema20, ema50=ema50, ema200=ema200,
        adx=adx, plus_di=plus_di, minus_di=minus_di, regime=regime,
        bb_lower=bb_lower, bb_upper=bb_upper, bb_width=bb_w, bb_squeeze_pct=bb_sq,
        volume=volume, volume_sma20=volume_sma20, volume_sma50=volume_sma50,
        atr_percentile=atr_pct,
        fib_0382=fib_0382 if not np.isnan(fib_0382) else 0.0,
        fib_0500=fib_0500 if not np.isnan(fib_0500) else 0.0,
        fib_0618=fib_0618 if not np.isnan(fib_0618) else 0.0,
        fib_direction=fib_dir, fib_proximity=fib_prox,
        pullback_type=pullback_type, ema50_slope=ema50_slope, timestamp=ts,
    )


def _evaluate_trend_short_adapted(
    row: pd.Series, params: dict, base_profile: Optional[StrategyProfile] = None
) -> Optional[Signal]:
    """
    Trend-following SHORT com parametros adaptados pelo regime.
    """
    close = float(row["close"])
    low = float(row["low"])
    high = float(row["high"])
    ema20 = float(row["ema20"])
    ema50 = float(row["ema50"])
    ema200 = float(row["ema200"])
    bb_lower = float(row["bb_lower"])
    bb_upper = float(row["bb_upper"])
    rsi = float(row["rsi"])
    rsi_delta = float(row.get("rsi_delta", 0.0))
    macd_hist = float(row.get("macd_hist", 0.0))
    macd_val = float(row.get("macd", 0.0))
    macd_sig = float(row.get("macd_signal", 0.0))
    volume = float(row["volume"])
    volume_sma20 = float(row["volume_sma20"])
    volume_sma50 = float(row.get("volume_sma50", 0.0))
    atr = float(row["atr"])
    atr_pct = float(row.get("atr_percentile", 0.5))
    bb_w = float(row.get("bb_width", 0.0))
    bb_sq = float(row.get("bb_squeeze_pct", 0.5))
    adx = float(row.get("adx", 0.0))
    plus_di = float(row.get("plus_di", 0.0))
    minus_di = float(row.get("minus_di", 0.0))
    regime = str(row.get("regime_v2", row.get("regime", "")))
    ema50_slope = float(row.get("ema50_slope", 0.0))
    fib_0382 = float(row.get("fib_0382", float("nan")))
    fib_0500 = float(row.get("fib_0500", float("nan")))
    fib_0618 = float(row.get("fib_0618", float("nan")))
    fib_dir = int(row.get("fib_direction", 0))
    fib_prox = float(row.get("fib_proximity", float("nan")))
    ts = row.name

    rsi_min, rsi_max = params["rsi_short_range"]
    sl_mult = params["sl_mult"]
    tp_mult = params["tp_mult"]

    if base_profile:
        _adx_min = base_profile.adx_min
        _fib_tol = base_profile.fib_tolerance_pct
        _slope_min = base_profile.ema50_slope_min
        _atr_pct_min = base_profile.atr_pct_min
        _atr_pct_max = base_profile.atr_pct_max
    else:
        _adx_min = 30.0
        _fib_tol = 0.025
        _slope_min = -1.0
        _atr_pct_min = 0.10
        _atr_pct_max = 0.90

    # 1. TENDENCIA: Dual EMA — downtrend
    if not (close < ema50 and ema50 < ema200):
        return None

    # 2. SLOPE
    if pd.isna(ema50_slope) or ema50_slope >= -_slope_min:
        return None

    # 3. PULLBACK
    pullback_type = None
    in_fib = _in_fib_zone(close, fib_0382, fib_0618, fib_dir)
    if in_fib and fib_dir == -1:
        pullback_type = "fibonacci"
    else:
        if fib_dir == -1:
            if (_price_near_fib(high, fib_0382, _fib_tol) or
                    _price_near_fib(high, fib_0500, _fib_tol) or
                    _price_near_fib(high, fib_0618, _fib_tol)):
                pullback_type = "fibonacci"

    if pullback_type is None:
        if bool(row.get("ema20_touched", False)) and close < ema20 and high >= ema20:
            pullback_type = "ema20_touch"

    if pullback_type is None:
        if bool(row.get("ema50_touched_up", False)) and close < ema50 and high >= ema50:
            pullback_type = "ema50_touch"

    if pullback_type is None:
        return None

    # 4. RSI
    if not (rsi_min <= rsi <= rsi_max):
        return None

    # 5. ATR
    if not (_atr_pct_min <= atr_pct <= _atr_pct_max):
        return None

    # 6. Volume
    if params["require_volume"] and volume_sma50 > 0:
        if volume < volume_sma50 * 0.3:
            return None

    # ── SL/TP ──
    stop_loss = close + sl_mult * atr
    take_profit = close - tp_mult * atr

    logger.info(
        "SIGNAL TF SHORT (regime=%s) | entry=%.2f SL=%.2f TP=%.2f ATR=%.2f "
        "RSI=%.1f pullback=%s SL_mult=%.2f TP_mult=%.2f",
        regime, close, stop_loss, take_profit, atr, rsi, pullback_type, sl_mult, tp_mult,
    )

    return Signal(
        type=SignalType.SHORT,
        entry_price=close, stop_loss=stop_loss, take_profit=take_profit,
        atr=atr, rsi=rsi, rsi_delta=rsi_delta, macd_hist=macd_hist,
        ema20=ema20, ema50=ema50, ema200=ema200,
        adx=adx, plus_di=plus_di, minus_di=minus_di, regime=regime,
        bb_lower=bb_lower, bb_upper=bb_upper, bb_width=bb_w, bb_squeeze_pct=bb_sq,
        volume=volume, volume_sma20=volume_sma20, volume_sma50=volume_sma50,
        atr_percentile=atr_pct,
        fib_0382=fib_0382 if not np.isnan(fib_0382) else 0.0,
        fib_0500=fib_0500 if not np.isnan(fib_0500) else 0.0,
        fib_0618=fib_0618 if not np.isnan(fib_0618) else 0.0,
        fib_direction=fib_dir, fib_proximity=fib_prox,
        pullback_type=pullback_type, ema50_slope=ema50_slope, timestamp=ts,
    )
