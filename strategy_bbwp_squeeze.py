r"""
strategy_bbwp_squeeze.py
------------------------
Squeeze Momentum Breakout Strategy v13 para BTC/USDT (1h).

v13 - COOLDOWN DIRECIONAL + FILTRO ADX PARA REVERSOES:

  Resultados v12 (730d): 140T, WR=49.3%, PF=1.23, PnL=+21.53%, DD=11.34%
  Resultados v8  (730d): 122T, WR=48.4%, PF=1.33, PnL=+25.53%, DD=8.73%

  v12 adicionou 18 trades vs v8, mas PF caiu 1.33->1.23. Os trades extras
  (reversoes com cooldown=1) entram sem filtro adicional — algumas sao
  falsas viradas. 365d do v12 foi excelente (PF=1.48), mas 730d arrastou.

  Mudancas v12 -> v13:
  1. ADX_MIN_REVERSAL: 20 (reversoes rapidas requerem tendencia forte)
     - Entrada normal (mesma dir, cooldown=2): ADX>16 (v8)
     - Reversao rapida (dir oposta, cooldown=1): ADX>20 (mais seletivo)
     - Logica: se vai entrar rapido na direcao oposta, exige mais confianca
  2. Tudo mais mantido do v12

  MANTIDOS do v8 (saidas otimizadas — NAO mexer):
  - TP1: 3.0x ATR, partial 50%
  - TRAILING: 1.5x ATR (racheta acima do floor)
  - SL: 2.2x ATR
  - POST-TP1 SL BUFFER: 0.5 ATR (floor 2.5*ATR)
  - SQUEEZE_RECENT_BARS: 12, BBWP_THRESHOLD: 15
  - VOLUME_MULT: 0.35, STOCH_RSI OB/OS: 56/44
  - BB_BREAKOUT_BUFFER: 0.05, TP1_PCT: 0.50, MAX_BARS: 96

  Matematica v13 (identica ao v8):
    TP1 = entry + 3.0*ATR
    Post-TP1 SL = TP1 - 0.5*ATR = entry + 2.5*ATR
    R:R floor = 2.75/2.2 = 1.25

  Logica central:
  1-6. Mesmo do v8 (BBWP<15, expansao, breakout, StochRSI, volume, EMA)
  7. ADX>16 para entradas normais, ADX>20 para reversoes rapidas
  8. Cooldown direcional: oposto=1 (com ADX>20), mesmo=2

Gestao de risco v13:
  - SL: 2.2x ATR | TP1: 3.0x ATR (50%) | Trailing: 1.5x ATR
  - Cooldown: mesma dir=2, oposta=1 (com ADX>20)

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
    # ---- Squeeze Detection (v8 — nao mexer) ----
    "bbwp_threshold": 15,        # v12: 15 (revertido do v10=16, v8=15)
    "squeeze_recent_bars": 12,    # v8: 12 (v7=10, v5=10)
    "require_bbwp_expansion": True,  # BBWP deve estar expandindo (delta > 0)

    # ---- Entry Conditions (v8 — nao mexer) ----
    "volume_mult": 0.35,         # v8: 0.35 (v7=0.45, v5=0.5)
    "stoch_rsi_ob": 56,           # v8: 56 (v7=58, v5=60)
    "stoch_rsi_os": 44,           # v8: 44 (v7=42, v5=40)
    "stoch_rsi_cross_enable": True,
    "stoch_rsi_min_delta": 0,     # Minimo delta K para confirmar momentum (0=any)
    "bb_breakout_buffer": 0.05,   # v8: 0.05 (v7=0.10)
    "adx_min": 16.0,              # v8: 16 (v7=20, v6=18, v5=none)
    "adx_min_reversal": 20.0,      # v13: 20 — ADX minimo para reversoes rapidas (cooldown_opp)

    # ---- Stop Loss (v8 — nao mexer) ----
    "sl_atr_mult": 2.2,           # v8: 2.2x ATR (v7=2.0)
    "sl_atr_mult_high_vol": 2.2, # Mesmo em alta volatilidade
    "sl_atr_mult_low_vol": 2.2,  # Mesmo em baixa volatilidade

    # ---- Take Profit (v10: revertido ao v8) ----
    "tp_atr_mult": 3.0,           # v10: 3.0x ATR (revertido do v9=2.8)
    "tp1_pct": 0.50,              # v10: 50% no TP1 (igual v8)

    # ---- Trailing Stop (v8 — nao mexer) ----
    "use_trailing": True,         # REATIVADO
    "be_trigger_atr_mult": 1.0,  # BE trigger (referencia)
    "trailing_atr_mult": 1.5,     # v10: 1.5x ATR (revertido do v9=1.7)
    "post_tp1_sl_buffer": 0.5,    # v8: 0.5 ATR (sweet spot)

    # ---- RSI Divergence (OFF) ----
    "use_divergence_exit": False, # Desativado
    "divergence_min_bars": 3,

    # ---- General (v13: cooldown direcional + ADX reversao) ----
    "max_bars_held": 96,          # v8: 96 (v7=120)
    "cooldown": 2,                # v13: 2 (mesma direcao — v8)
    "cooldown_trailing": 2,       # v13: 2 (mesma direcao apos trailing)
    "cooldown_opp_dir": 1,        # v13: 1 (direcao oposta — com ADX>20)
    "use_directional_cooldown": True,  # v13: ativa cooldown direcional

    # ---- Filters ----
    "atr_pct_min": 0.10,
    "atr_pct_max": 0.90,
    "ema200_filter": True,        # LONG: close > EMA200, SHORT: close < EMA200
    "min_bbwp_bars": 1,           # Minimo de bars com BBWP < threshold na janela
}


# ---- Cooldown State ----
_last_signal_bar: int = -999
_last_exit_was_trailing: bool = False
_last_signal_direction: str = ""  # "long" or "short" — v12


def reset_cooldown() -> None:
    global _last_signal_bar, _last_exit_was_trailing, _last_signal_direction
    _last_signal_bar = -999
    _last_exit_was_trailing = False
    _last_signal_direction = ""


def _check_cooldown(current_idx: int, direction: str = "") -> bool:
    """
    v13: Cooldown direcional.
    - Direcao oposta ao ultimo trade: cooldown_opp_dir (1 bar)
    - Mesma direcao: cooldown ou cooldown_trailing (2 bars)
    - Sem trade anterior: sem cooldown
    """
    global _last_signal_bar, _last_exit_was_trailing, _last_signal_direction
    p = BBWP_SQUEEZE_PARAMS

    if _last_signal_bar < 0:
        return True

    # v13: cooldown direcional
    if p.get("use_directional_cooldown", False) and direction and _last_signal_direction:
        is_opposite = (direction != _last_signal_direction)
        if is_opposite:
            cd = p.get("cooldown_opp_dir", p["cooldown"])
        else:
            cd = p["cooldown_trailing"] if _last_exit_was_trailing else p["cooldown"]
    else:
        cd = p["cooldown_trailing"] if _last_exit_was_trailing else p["cooldown"]

    return (current_idx - _last_signal_bar) >= cd


def _is_reversal(direction: str) -> bool:
    """v13: verifica se a direcao e uma reversao (oposta ao ultimo trade)."""
    global _last_signal_direction
    return bool(_last_signal_direction and direction and direction != _last_signal_direction)


def _register_signal(current_idx: int, was_trailing: bool = False, direction: str = "") -> None:
    global _last_signal_bar, _last_exit_was_trailing, _last_signal_direction
    _last_signal_bar = current_idx
    _last_exit_was_trailing = was_trailing
    if direction:
        _last_signal_direction = direction


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

def _adx_confirms_trend(row, is_reversal: bool = False) -> bool:
    """
    v13: ADX com threshold adaptativo.
    Normal: ADX > 16 (v8). Reversao rapida: ADX > 20 (mais seletivo).
    """
    p = BBWP_SQUEEZE_PARAMS
    if is_reversal:
        adx_min = p.get("adx_min_reversal", p.get("adx_min", 0))
    else:
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
        regime="bbwp_squeeze_v13",
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
        pullback_type="bbwp_squeeze_v13",
        ema50_slope=float(row.get("ema50_slope", 0)),
        timestamp=ts,
    )


# ==================================================================
# CORE EVALUATION
# ==================================================================

def _evaluate_direction(row, prev_row, direction: str, idx: int = 0, df=None, profile=None, is_reversal: bool = False) -> Optional[tuple]:
    """
    Avalia condicoes de entrada para uma direcao.
    Retorna (sl, tp, atr, bbwp) ou None.
    v13: is_reversal usa ADX>20 para filtrar falsas viradas.
    """
    p = BBWP_SQUEEZE_PARAMS

    # 1. Squeeze + expansion detection
    if not _is_squeeze_breakout(row, prev_row, idx=idx, df=df):
        return None

    # 2. ADX trend filter (v13: threshold mais alto para reversoes)
    if not _adx_confirms_trend(row, is_reversal=is_reversal):
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
    if not _check_cooldown(idx, direction="long"):
        pass  # nao bloqueia short se long em cooldown direcional
    else:
        result = _evaluate_direction(curr, prev, "long", idx=idx, df=df, profile=profile,
                                       is_reversal=_is_reversal("long"))
        if result is not None:
            sl, tp, atr, bbwp = result
            _register_signal(idx, direction="long")
            logger.info(
                "SIGNAL BBWP Squeeze v13 LONG%s | entry=%.2f SL=%.2f TP=%.2f BBWP=%.1f ADX=%.1f",
                " [REV]" if _is_reversal("long") else "",
                float(curr["close"]), sl, tp, bbwp, float(curr.get("adx", 0)),
            )
            return _build_signal(float(curr["close"]), sl, tp, curr, True)

    if not _check_cooldown(idx, direction="short"):
        return None

    result = _evaluate_direction(curr, prev, "short", idx=idx, df=df, profile=profile,
                                   is_reversal=_is_reversal("short"))
    if result is not None:
        sl, tp, atr, bbwp = result
        _register_signal(idx, direction="short")
        logger.info(
            "SIGNAL BBWP Squeeze v13 SHORT%s | entry=%.2f SL=%.2f TP=%.2f BBWP=%.1f ADX=%.1f",
            " [REV]" if _is_reversal("short") else "",
            float(curr["close"]), sl, tp, bbwp, float(curr.get("adx", 0)),
        )
        return _build_signal(float(curr["close"]), sl, tp, curr, False)

    return None


def evaluate_bbwp_squeeze_row(row, prev_row, bar_index: int, df=None, profile=None) -> Optional[tuple]:
    """
    Avalia sinal para uma linha individual (para backtest loop).
    Returns: (Signal, bbwp, trigger_direction) ou None
    """
    # v13: checa cada direcao com seu proprio cooldown + ADX reversao
    long_ok = _check_cooldown(bar_index, direction="long")
    short_ok = _check_cooldown(bar_index, direction="short")

    if long_ok:
        result = _evaluate_direction(row, prev_row, "long", idx=bar_index, df=df, profile=profile,
                                       is_reversal=_is_reversal("long"))
        if result is not None:
            sl, tp, atr, bbwp = result
            _register_signal(bar_index, direction="long")
            signal = _build_signal(float(row["close"]), sl, tp, row, True)
            return (signal, bbwp, "long")

    if short_ok:
        result = _evaluate_direction(row, prev_row, "short", idx=bar_index, df=df, profile=profile,
                                       is_reversal=_is_reversal("short"))
        if result is not None:
            sl, tp, atr, bbwp = result
            _register_signal(bar_index, direction="short")
            signal = _build_signal(float(row["close"]), sl, tp, row, False)
            return (signal, bbwp, "short")

    return None
