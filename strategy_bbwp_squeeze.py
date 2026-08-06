r"""
strategy_bbwp_squeeze.py
------------------------
Squeeze Momentum Breakout Strategy v9 para BTC/USDT (1h).

v9 - EQUILIBRIO OTIMIZADO (sweet spot v7-v8):

  Problemas do v8 (730d: 142T, WR=48.6%, PF=1.22, PnL=+21.57%, DD=10.80%):
  1. PF caiu 1.33->1.22 (trades marginais de baixa qualidade)
  2. DD quase dobrou 6.97%->10.80% (mais exposicao)
  3. 90d PF=1.02 (perigosamente perto de breakeven)
  4. Pior trade -3.94% (SL 2.2x nao reduziu whipsaws)
  5. 56 trades novos contribuiram apenas +3.87% (qualidade 33% dos originais)

  Mudancas v8 -> v9 (8 ajustes para sweet spot, alvo ~110-120 trades):

  ENTRADAS (reverter metade da relaxacao):
  1. BBWP_THRESHOLD: 15 -> 14 (leve aperto)
  2. SQUEEZE_RECENT_BARS: 12 -> 11 (leve aperto)
  3. ADX_MIN: 16 -> 18 (reverter metade — filtro mais impactante)
  4. VOLUME_MULT: 0.35 -> 0.40 (reverter metade)
  5. STOCH_RSI OB/OS: 56/44 -> 57/43 (leve aperto)
  6. COOLDOWN_TRAILING: 1 -> 2 (reverter — evita re-entrada rapida)

  SAIDAS (reverter SL que piorou resultados):
  7. SL: 2.2x -> 2.0x ATR (avg loss -1.34%->-1.22% no v7)
  8. MANTIDO: MAX_BARS=96 (funcionou bem)

  MANTIDOS do v8 (funcionaram):
  - BB_BREAKOUT_BUFFER: 0.05
  - COOLDOWN: 2
  - TP1: 3.0x ATR (partial 50%)
  - Post-TP1 SL: TP1 - 0.5*ATR (floor 2.5*ATR)
  - Trailing: 1.5x ATR (racheta acima do floor)

  Matematica v9:
    TP1 = entry + 3.0*ATR
    Post-TP1 SL = TP1 - 0.5*ATR = entry + 2.5*ATR
    Win floor = 0.50*3.0*ATR% + 0.50*2.5*ATR% = 2.75*ATR%
    SL = 2.0*ATR%
    R:R floor = 2.75/2.0 = 1.375
    Com trailing rachetando acima: R:R real ~1.45-1.60

  Logica central:
  1. BBWP < 14 nos ultimos 11 bars
  2. BBWP expandindo (delta > 0)
  3. BB breakout com corpo (close > upper + 5% BB width)
  4. ADX > 18 (tendencias moderadas-fortes)
  5. Stoch RSI confirma momentum (K>=57 LONG, K<=43 SHORT)
  6. Volume >= 0.40 * SMA(Volume, 20)
  7. EMA50 alinha com tendencia
  8. EMA200 confirma tendencia macro

Gestao de risco v9:
  - SL: 2.0x ATR
  - TP1: 3.0x ATR (partial 50%)
  - Pos-TP1 SL: TP1 - 0.5*ATR (floor de 2.5*ATR no trailing)
  - Trailing: 1.5x ATR (racheta acima do floor)
  - Max bars: 96 (4 dias em 1h)
  - Cooldown: 2 bars (2 apos trailing exit)

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
    # ---- Squeeze Detection (v9: sweet spot) ----
    "bbwp_threshold": 14,        # v9: 14 (v8=15, v7=12, v5=10)
    "squeeze_recent_bars": 11,    # v9: 11 (v8=12, v7=10, v5=10)
    "require_bbwp_expansion": True,  # BBWP deve estar expandindo (delta > 0)

    # ---- Entry Conditions (v9: meio-termo v7-v8) ----
    "volume_mult": 0.40,         # v9: 0.40 (v8=0.35, v7=0.45, v5=0.5)
    "stoch_rsi_ob": 57,           # v9: 57 (v8=56, v7=58, v5=60)
    "stoch_rsi_os": 43,           # v9: 43 (v8=44, v7=42, v5=40)
    "stoch_rsi_cross_enable": True,
    "stoch_rsi_min_delta": 0,     # Minimo delta K para confirmar momentum (0=any)
    "bb_breakout_buffer": 0.05,   # v9: 0.05 (igual v8)
    "adx_min": 18.0,              # v9: 18 (v8=16, v7=20, v6=18, v5=none)

    # ---- Stop Loss (v9: volta ao 2.0x — melhor avg loss) ----
    "sl_atr_mult": 2.0,           # v9: 2.0x ATR (v8=2.2, v7=2.0)
    "sl_atr_mult_high_vol": 2.0, # Mesmo em alta volatilidade
    "sl_atr_mult_low_vol": 2.0,  # Mesmo em baixa volatilidade

    # ---- Take Profit (mantido) ----
    "tp_atr_mult": 3.0,           # v9: 3.0x ATR (igual v7/v8)
    "tp1_pct": 0.50,              # v9: 50% no TP1 (igual v7/v8)

    # ---- Trailing Stop (mantido) ----
    "use_trailing": True,         # REATIVADO
    "be_trigger_atr_mult": 1.0,  # BE trigger (referencia)
    "trailing_atr_mult": 1.5,     # v9: 1.5x ATR (igual v7/v8)
    "post_tp1_sl_buffer": 0.5,    # v9: 0.5 ATR (igual v7/v8)

    # ---- RSI Divergence (OFF) ----
    "use_divergence_exit": False, # Desativado
    "divergence_min_bars": 3,

    # ---- General (v9: cooldown equilibrado) ----
    "max_bars_held": 96,          # v9: 96 (igual v8)
    "cooldown": 2,                # v9: 2 (igual v8)
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
    ADX > 18 indica tendencia ativa, filtrando mercados laterais.
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
        regime="bbwp_squeeze_v9",
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
        pullback_type="bbwp_squeeze_v9",
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
            "SIGNAL BBWP Squeeze v9 LONG | entry=%.2f SL=%.2f TP=%.2f BBWP=%.1f ADX=%.1f",
            float(curr["close"]), sl, tp, bbwp, float(curr.get("adx", 0)),
        )
        return _build_signal(float(curr["close"]), sl, tp, curr, True)

    result = _evaluate_direction(curr, prev, "short", idx=idx, df=df, profile=profile)
    if result is not None:
        sl, tp, atr, bbwp = result
        _register_signal(idx)
        logger.info(
            "SIGNAL BBWP Squeeze v9 SHORT | entry=%.2f SL=%.2f TP=%.2f BBWP=%.1f ADX=%.1f",
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
