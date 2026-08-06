r"""
strategy_bbwp_squeeze.py
------------------------
Squeeze Momentum Breakout Strategy v7 para BTC/USDT (1h).

v7 - QUALIDADE + R:R ALTO + POST-TP1 SL INTELIGENTE:

  Problemas do v6 (730d: 132T, WR=43.2%, PF=1.11, PnL=+10.83%, DD=13.99%):
  1. Entradas muito relaxadas (BBWP<15, sem buffer, vol 0.4) => 180d DESTRUIDO (-2.43%)
  2. R:R 1.20 => floor do post-TP1 SL (TP1-trail*ATR) limita trailing
  3. Trailing 1.5x justo demais => nao deixa ganhadores correrem
  4. Avg win caiu +2.12% -> +1.95% (v5 tinha melhor)
  5. 75 SL losses em 730d, 25% em 1-3 barras (whipsaws de baixa qualidade)

  Mudancas v6 -> v7:

  QUALIDADE (reverter relaxacao excessiva):
  1. BBWP_THRESHOLD: 15 -> 12 (meio-termo entre v5=10 e v6=15)
  2. BB_BREAKOUT_BUFFER: 0 -> 0.10 (filtra breakouts fracos)
  3. VOLUME_MULT: 0.4 -> 0.45
  4. STOCH_RSI OB/OS: 55/45 -> 58/42
  5. ADX_MIN: 18 -> 20 (filtro de tendencia mais forte)
  6. COOLDOWN: 2 -> 3
  7. SQUEEZE_RECENT_BARS: 12 -> 10

  R:R ALTO (otimizar saidas):
  8. TP1: 3.5x -> 3.0x ATR (mais alcancavel => maior hit rate)
  9. TP1_PCT: 40% -> 50% (lock in METADE no TP1)
  10. TRAILING: 1.5x -> 1.5x ATR (mantem, bom para racheta)
  11. NOVO: POST_TP1_SL_BUFFER: 0.5 (SL = TP1 - 0.5*ATR, NAO TP1 - trail*ATR)
      => Floor de 2.5*ATR lucro na porcao trailing (vs 2.0*ATR no v6)
      => R:R floor: (0.50*3.0 + 0.50*2.5)/2.0 = 1.375
  12. SL: 2.2x -> 2.0x ATR
  13. MAX_BARS: 120 (mantem)

  Matematica v7:
    TP1 = entry + 3.0*ATR
    Post-TP1 SL = TP1 - 0.5*ATR = entry + 2.5*ATR
    Win floor = 0.50*3.0*ATR% + 0.50*2.5*ATR% = 2.75*ATR%
    SL = 2.0*ATR%
    R:R floor = 2.75/2.0 = 1.375
    Com trailing rachetando acima: R:R real ~1.50-1.60

  Logica central:
  1. BBWP < 12 nos ultimos 10 bars
  2. BBWP expandindo (delta > 0)
  3. BB breakout com corpo (close > upper + 10% BB width)
  4. ADX > 20 (confirma tendencia)
  5. Stoch RSI confirma momentum (K>=58 LONG, K<=42 SHORT)
  6. Volume >= 0.45 * SMA(Volume, 20)
  7. EMA50 alinha com tendencia
  8. EMA200 confirma tendencia macro

Gestao de risco v7:
  - SL: 2.0x ATR
  - TP1: 3.0x ATR (partial 50%)
  - Pos-TP1 SL: TP1 - 0.5*ATR (floor de 2.5*ATR no trailing)
  - Trailing: 1.5x ATR (racheta acima do floor)
  - Max bars: 120 (5 dias em 1h)
  - Cooldown: 3 bars (2 apos trailing exit)

Custos: maker fee 0.016% + spread 2bps + slip 2bps (limit orders).
"""
from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from strategy import Signal, SignalType

if TYPE_CHECKING:
    from strategy_profiles import StrategyProfile

logger = logging.getLogger(__name__)


# ==================================================================
# PARAMETROS CONFIGURAVEIS
# ==================================================================

BBWP_SQUEEZE_PARAMS = {
    # ---- Squeeze Detection (v7: meio-termo v5/v6) ----
    "bbwp_threshold": 12,        # v7: 12 (v5=10, v6=15)
    "squeeze_recent_bars": 10,    # v7: 10 (v5=10, v6=12)
    "require_bbwp_expansion": True,  # BBWP deve estar expandindo (delta > 0)

    # ---- Entry Conditions (v7: qualidade > quantidade) ----
    "volume_mult": 0.45,         # v7: 0.45 (v5=0.5, v6=0.4)
    "stoch_rsi_ob": 58,           # v7: 58 (v5=60, v6=55)
    "stoch_rsi_os": 42,           # v7: 42 (v5=40, v6=45)
    "stoch_rsi_cross_enable": True,
    "stoch_rsi_min_delta": 0,     # Minimo delta K para confirmar momentum (0=any)
    "bb_breakout_buffer": 0.10,   # v7: 0.10 (v5=0.15, v6=0)
    "adx_min": 20.0,              # v7: 20 (v6=18, v5=none)

    # ---- Stop Loss (v7: volta ao 2.0x) ----
    "sl_atr_mult": 2.0,           # v7: 2.0x ATR (v5=2.0, v6=2.2)
    "sl_atr_mult_high_vol": 2.0, # Mesmo em alta volatilidade
    "sl_atr_mult_low_vol": 2.0,  # Mesmo em baixa volatilidade

    # ---- Take Profit (v7: TP1 mais alcancavel, mais partial) ----
    "tp_atr_mult": 3.0,           # v7: 3.0x ATR (v5=4.0, v6=3.5)
    "tp1_pct": 0.50,              # v7: 50% no TP1 (v5=30%, v6=40%)

    # ---- Trailing Stop (v7: trailing e post-TP1 SL separados) ----
    "use_trailing": True,         # REATIVADO
    "be_trigger_atr_mult": 1.0,  # BE trigger (referencia)
    "trailing_atr_mult": 1.5,     # v7: 1.5x ATR (racheta acima do floor)
    "post_tp1_sl_buffer": 0.5,    # v7: NOVO — SL = TP1 - 0.5*ATR (floor alto)

    # ---- RSI Divergence (OFF) ----
    "use_divergence_exit": False, # Desativado
    "divergence_min_bars": 3,

    # ---- General (v7: cooldown medio) ----
    "max_bars_held": 120,         # v7: 120 (5 dias em 1h)
    "cooldown": 3,                # v7: 3 (v5=4, v6=2)
    "cooldown_trailing": 2,

    # ---- Filters ----
    "atr_pct_min": 0.10,
    "atr_pct_max": 0.90,
    "ema200_filter": True,        # LONG: close > EMA200, SHORT: close < EMA200
    "min_bbwp_bars": 1,           # Minimo de bars com BBWP < threshold na janela
}


# ---- Cooldown State ----
_last_signal_bar: int = -999
_last_exit_was_trailing: bool = False


def reset_cooldown() -> None:
    global _last_signal_bar, _last_exit_was_trailing
    _last_signal_bar = -999
    _last_exit_was_trailing = False


def _check_cooldown(current_idx: int) -> bool:
    global _last_signal_bar, _last_exit_was_trailing
    cd = BBWP_SQUEEZE_PARAMS["cooldown_trailing"] if _last_exit_was_trailing else BBWP_SQUEEZE_PARAMS["cooldown"]
    return (current_idx - _last_signal_bar) >= cd


def _register_signal(current_idx: int, was_trailing: bool = False) -> None:
    global _last_signal_bar, _last_exit_was_trailing
    _last_signal_bar = current_idx
    _last_exit_was_trailing = was_trailing


# ==================================================================
# SQUEEZE + EXPANSION DETECTION
# ==================================================================

def _is_squeeze_breakout(row, prev_row, idx: int = 0, df=None) -> bool:
    """
    Detecta squeeze seguido de expansao (breakout de volatilidade).

    Condicoes:
    1. BBWP < threshold em pelo menos min_bbwp_bars dos ultimos squeeze_recent_bars
    2. Se require_bbwp_expansion=True: BBWP atual > BBWP anterior (expandindo)
    """
    p = BBWP_SQUEEZE_PARAMS
    bbwp = float(row.get("bbwp", 100))

    if pd.isna(bbwp):
        return False

    recent = p["squeeze_recent_bars"]
    min_bars = p["min_bbwp_bars"]

    # Verificar squeeze recente
    if df is not None and idx >= recent and recent > 0:
        bbwp_col = df["bbwp"].values
        window = bbwp_col[max(0, idx - recent + 1):idx + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < min_bars:
            return False
        squeeze_count = np.sum(valid < p["bbwp_threshold"])
        if squeeze_count < min_bars:
            return False
    else:
        # Fallback: apenas BBWP atual
        if bbwp >= p["bbwp_threshold"]:
            return False

    # Verificar expansao
    if p["require_bbwp_expansion"]:
        prev_bbwp = float(prev_row.get("bbwp", 100))
        if pd.isna(prev_bbwp) or bbwp <= prev_bbwp:
            return False

    return True


# ==================================================================
# ADX TREND FILTER
# ==================================================================

def _adx_confirms_trend(row) -> bool:
    """
    Verifica se ADX confirma tendencia real.
    ADX > 20 indica tendencia ativa, filtrando mercados laterais.
    """
    p = BBWP_SQUEEZE_PARAMS
    adx_min = p.get("adx_min", 0)
    if adx_min <= 0:
        return True  # Filtro desativado
    adx = float(row.get("adx", 0))
    if pd.isna(adx):
        return False
    return adx >= adx_min


# ==================================================================
# STOCH RSI CONFIRMATION
# ==================================================================

def _stoch_rsi_confirms(row, prev_row, direction: str) -> bool:
    """
    Verifica se Stoch RSI confirma a direcao do breakout.

    LONG: K > ob OU (K cruza acima de D) OU (K > 50 e delta > 0)
    SHORT: K < os OU (K cruza abaixo de D) OU (K < 50 e delta < 0)
    """
    p = BBWP_SQUEEZE_PARAMS
    k = float(row.get("stoch_rsi_k", 50))
    d = float(row.get("stoch_rsi_d", 50))
    prev_k = float(prev_row.get("stoch_rsi_k", 50))
    prev_d = float(prev_row.get("stoch_rsi_d", 50))

    # Minimo delta
    min_delta = p.get("stoch_rsi_min_delta", 0)
    k_delta = k - prev_k

    if direction == "long":
        if k >= p["stoch_rsi_ob"]:
            return True
        if k_delta < min_delta:
            return False
        if k > d and prev_k <= prev_d and k > 40:
            return True
        if k > 50 and k_delta > 2:
            return True
        return False
    else:
        if k <= p["stoch_rsi_os"]:
            return True
        if k_delta > -min_delta:
            return False
        if k < d and prev_k >= prev_d and k < 60:
            return True
        if k < 50 and k_delta < -2:
            return True
        return False


# ==================================================================
# VOLUME CONFIRMATION
# ==================================================================

def _volume_confirms(row) -> bool:
    """Volume > mult * SMA(Volume, 20)."""
    p = BBWP_SQUEEZE_PARAMS
    vol = float(row.get("volume", 0))
    vol_sma = float(row.get("volume_sma20", 0))
    if vol_sma <= 0:
        return False
    return vol >= vol_sma * p["volume_mult"]


# ==================================================================
# ADAPTIVE SL
# ==================================================================

def _get_sl_mult(row) -> float:
    """SL adaptativo ao ATR percentile."""
    p = BBWP_SQUEEZE_PARAMS
    atr_pct = float(row.get("atr_percentile", 0.5))
    mult = p["sl_atr_mult"]
    if atr_pct > 0.70:
        mult = p["sl_atr_mult_high_vol"]
    elif atr_pct < 0.30:
        mult = p["sl_atr_mult_low_vol"]
    return mult


# ==================================================================
# SIGNAL BUILDER
# ==================================================================

def _build_signal(entry, sl, tp, row, is_long: bool) -> Signal:
    ts = row.name if hasattr(row, "name") else row.index
    close = float(row["close"]) if hasattr(row, "__getitem__") else float(entry)
    return Signal(
        type=SignalType.LONG if is_long else SignalType.SHORT,
        entry_price=close if hasattr(row, "__getitem__") else entry,
        stop_loss=sl, take_profit=tp,
        atr=float(row.get("atr", 0)),
        rsi=float(row.get("rsi", 0)),
        rsi_delta=float(row.get("rsi_delta", 0)),
        macd_hist=float(row.get("macd_hist", 0)),
        ema20=float(row.get("ema20", 0)),
        ema50=float(row.get("ema50", 0)),
        ema200=float(row.get("ema200", 0)),
        adx=float(row.get("adx", 0)),
        plus_di=float(row.get("plus_di", 0)),
        minus_di=float(row.get("minus_di", 0)),
        regime="bbwp_squeeze_v7",
        bb_lower=float(row.get("bb_lower", 0)),
        bb_upper=float(row.get("bb_upper", 0)),
        bb_width=float(row.get("bb_width", 0)),
        bb_squeeze_pct=float(row.get("bb_squeeze_pct", 0.5)),
        volume=float(row.get("volume", 0)),
        volume_sma20=float(row.get("volume_sma20", 0)),
        volume_sma50=float(row.get("volume_sma50", 0)),
        atr_percentile=float(row.get("atr_percentile", 0.5)),
        fib_0382=0.0, fib_0500=0.0, fib_0618=0.0,
        fib_direction=0, fib_proximity=0.0,
        pullback_type="bbwp_squeeze_v7",
        ema50_slope=float(row.get("ema50_slope", 0)),
        timestamp=ts,
    )


# ==================================================================
# CORE EVALUATION
# ==================================================================

def _evaluate_direction(row, prev_row, direction: str, idx: int = 0, df=None, profile=None) -> Optional[tuple]:
    """
    Avalia condicoes de entrada para uma direcao.
    Retorna (sl, tp, atr, bbwp) ou None.
    """
    p = BBWP_SQUEEZE_PARAMS

    # 1. Squeeze + expansion detection
    if not _is_squeeze_breakout(row, prev_row, idx=idx, df=df):
        return None

    # 2. ADX trend filter
    if not _adx_confirms_trend(row):
        return None

    # 3. ATR percentile filter
    atr_pct = float(row.get("atr_percentile", 0.5))
    if not (p["atr_pct_min"] <= atr_pct <= p["atr_pct_max"]):
        return None

    # 4. Volume confirmation
    if not _volume_confirms(row):
        return None

    close = float(row["close"])
    bb_upper = float(row.get("bb_upper", 0))
    bb_lower = float(row.get("bb_lower", 0))
    bb_middle = float(row.get("bb_middle", 0))
    bb_width = float(row.get("bb_width", 0))
    ema50 = float(row.get("ema50", 0))
    ema200 = float(row.get("ema200", 0))
    atr = float(row.get("atr", 0))
    bbwp = float(row.get("bbwp", 100))

    if atr <= 0 or close <= 0 or bb_upper <= 0 or bb_lower <= 0:
        return None

    sl_mult = _get_sl_mult(row)

    # BB breakout with buffer
    bb_buffer = p.get("bb_breakout_buffer", 0)

    if direction == "long":
        # 5. BB breakout (close > upper + buffer * BB width)
        breakout_level = bb_upper + bb_buffer * (bb_upper - bb_lower)
        if close <= breakout_level:
            return None

        # 6. Stoch RSI confirmation
        if not _stoch_rsi_confirms(row, prev_row, "long"):
            return None

        # 7. EMA50 trend filter
        if close <= ema50:
            return None

        # 8. EMA200 macro filter
        if p.get("ema200_filter", False) and ema200 > 0 and close <= ema200:
            return None

        # SL e TP
        sl = close - sl_mult * atr
        if sl <= 0:
            return None
        tp = close + p["tp_atr_mult"] * atr

        return (sl, tp, atr, bbwp)
    else:
        # 5. BB breakout (close < lower - buffer * BB width)
        breakout_level = bb_lower - bb_buffer * (bb_upper - bb_lower)
        if close >= breakout_level:
            return None

        # 6. Stoch RSI confirmation
        if not _stoch_rsi_confirms(row, prev_row, "short"):
            return None

        # 7. EMA50 trend filter
        if close >= ema50:
            return None

        # 8. EMA200 macro filter
        if p.get("ema200_filter", False) and ema200 > 0 and close >= ema200:
            return None

        # SL e TP
        sl = close + sl_mult * atr
        tp = close - p["tp_atr_mult"] * atr

        return (sl, tp, atr, bbwp)


def evaluate_bbwp_squeeze(df, profile=None) -> Optional[Signal]:
    """Avalia sinal para o ultimo candle do DataFrame."""
    if len(df) < 2:
        return None
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    idx = len(df) - 1
    if not _check_cooldown(idx):
        return None

    result = _evaluate_direction(curr, prev, "long", idx=idx, df=df, profile=profile)
    if result is not None:
        sl, tp, atr, bbwp = result
        _register_signal(idx)
        logger.info(
            "SIGNAL BBWP Squeeze v7 LONG | entry=%.2f SL=%.2f TP=%.2f BBWP=%.1f ADX=%.1f",
            float(curr["close"]), sl, tp, bbwp, float(curr.get("adx", 0)),
        )
        return _build_signal(float(curr["close"]), sl, tp, curr, True)

    result = _evaluate_direction(curr, prev, "short", idx=idx, df=df, profile=profile)
    if result is not None:
        sl, tp, atr, bbwp = result
        _register_signal(idx)
        logger.info(
            "SIGNAL BBWP Squeeze v7 SHORT | entry=%.2f SL=%.2f TP=%.2f BBWP=%.1f ADX=%.1f",
            float(curr["close"]), sl, tp, bbwp, float(curr.get("adx", 0)),
        )
        return _build_signal(float(curr["close"]), sl, tp, curr, False)

    return None


def evaluate_bbwp_squeeze_row(row, prev_row, bar_index: int, df=None, profile=None) -> Optional[tuple]:
    """
    Avalia sinal para uma linha individual (para backtest loop).
    Returns: (Signal, bbwp, trigger_direction) ou None
    """
    if not _check_cooldown(bar_index):
        return None

    result = _evaluate_direction(row, prev_row, "long", idx=bar_index, df=df, profile=profile)
    if result is not None:
        sl, tp, atr, bbwp = result
        _register_signal(bar_index)
        signal = _build_signal(float(row["close"]), sl, tp, row, True)
        return (signal, bbwp, "long")

    result = _evaluate_direction(row, prev_row, "short", idx=bar_index, df=df, profile=profile)
    if result is not None:
        sl, tp, atr, bbwp = result
        _register_signal(bar_index)
        signal = _build_signal(float(row["close"]), sl, tp, row, False)
        return (signal, bbwp, "short")

    return None
