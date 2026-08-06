r"""
strategy_bbwp_squeeze.py
------------------------
Squeeze Momentum Breakout Strategy v3 para BTC/USDT (15m e 1h).

v3 - Otimizado via grid search com 2160 combinacoes (1h, 6 meses):
  Resultado validado: WR=67.7%, PnL=+16.41% (+25.64pp vs B&H), PF=2.56, 31 trades

  Mudancas vs v2:
  1. BBWP_THRESHOLD: 20 -> 5 (mais sensivel ao squeeze)
  2. VOLUME_MULT: 1.3 -> 0.8 (menos exigente, captura mais breakouts)
  3. TP_ATR_MULT: 5.0 -> 2.5 (R:R mais realista, SL=2.0x => R:R 1.25:1)
  4. TRAILING: DESATIVADO (reduzia WR significativamente)
  5. EMA200_FILTER: ON (manter tendencia macro)
  6. DIVERGENCE_FILTER: OFF na entrada (era muito restritivo)
  7. STOCH_RSI_OB: 65 -> 70 (zona de sobrecompra mais tradicional)
  8. STOCH_RSI_OS: 35 -> 30 (zona de sobrevenda mais tradicional)
  9. SQUEEZE_RECENT_BARS: 5 -> 8 (janela maior para detectar squeeze)
  10. SL_ADAPTIVO: removido (SL fixo 2.0x ATR)

Logica central:
  1. BBWP < 5 (percentil) nos ultimos 8 bars
  2. BBWP esta expandindo (delta > 0)
  3. BB breakout (close > upper ou close < lower)
  4. Stoch RSI confirma momentum (K>=70 para LONG, K<=30 para SHORT)
  5. Volume >= 0.8 * SMA(Volume, 20)
  6. EMA50 alinha com tendencia
  7. EMA200 confirma tendencia macro

Gestao de risco:
  - SL: 2.0x ATR (fixo)
  - TP: 2.5x ATR
  - Cooldown: 6 bars

Custos: taker fee 0.031% + spread 1bp + slip 1bp.
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
    # ---- Squeeze Detection (v3: otimizado) ----
    "bbwp_threshold": 5,         # BBWP abaixo deste valor = squeeze (v2: 20)
    "squeeze_recent_bars": 8,     # Ultimos N bars para verificar squeeze (v2: 5)
    "require_bbwp_expansion": True,  # BBWP deve estar expandindo (delta > 0)

    # ---- Entry Conditions (v3: otimizado) ----
    "volume_mult": 0.8,          # Volume > mult * SMA(Volume, 20) (v2: 1.3)
    "stoch_rsi_ob": 70,           # Stoch RSI sobrecompra LONG (v2: 65)
    "stoch_rsi_os": 30,           # Stoch RSI sobrevenda SHORT (v2: 35)
    "stoch_rsi_cross_enable": True,
    "stoch_rsi_min_delta": 0,     # Minimo delta K para confirmar momentum (0=any)

    # ---- Stop Loss (v3: fixo, sem adaptacao) ----
    "sl_atr_mult": 2.0,           # SL base = mult * ATR (fixo)
    "sl_atr_mult_high_vol": 2.0, # SL em alta volatilidade (mesmo)
    "sl_atr_mult_low_vol": 2.0,  # SL em baixa volatilidade (mesmo)

    # ---- Take Profit (v3: reduzido para R:R realista) ----
    "tp_atr_mult": 2.5,           # TP = entry + mult * ATR (v2: 5.0)

    # ---- Trailing Stop (v3: DESATIVADO) ----
    "use_trailing": False,        # Desativado (reduzia WR significativamente)
    "be_trigger_atr_mult": 1.0,  # Move SL para entry apos este ATR em profit
    "trailing_atr_mult": 1.5,     # Trailing distance = mult * ATR

    # ---- RSI Divergence (v3: OFF na entrada) ----
    "use_divergence_exit": False, # Desativado (era muito restritivo)
    "divergence_min_bars": 3,

    # ---- General ----
    "max_bars_held": 72,
    "cooldown": 6,
    "cooldown_trailing": 3,

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
        regime="bbwp_squeeze_v3",
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
        pullback_type="bbwp_squeeze_v2",
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
    
    # 2. ATR percentile filter
    atr_pct = float(row.get("atr_percentile", 0.5))
    if not (p["atr_pct_min"] <= atr_pct <= p["atr_pct_max"]):
        return None
    
    # 3. Volume confirmation
    if not _volume_confirms(row):
        return None
    
    close = float(row["close"])
    bb_upper = float(row.get("bb_upper", 0))
    bb_lower = float(row.get("bb_lower", 0))
    ema50 = float(row.get("ema50", 0))
    ema200 = float(row.get("ema200", 0))
    atr = float(row.get("atr", 0))
    bbwp = float(row.get("bbwp", 100))
    
    if atr <= 0 or close <= 0 or bb_upper <= 0 or bb_lower <= 0:
        return None
    
    sl_mult = _get_sl_mult(row)
    
    if direction == "long":
        # 4. BB breakout (close above upper band)
        if close <= bb_upper:
            return None
        
        # 5. Stoch RSI confirmation
        if not _stoch_rsi_confirms(row, prev_row, "long"):
            return None
        
        # 6. EMA50 trend filter
        if close <= ema50:
            return None
        
        # 7. EMA200 macro filter
        if p.get("ema200_filter", False) and ema200 > 0 and close <= ema200:
            return None
        
        # SL e TP
        sl = close - sl_mult * atr
        if sl <= 0:
            return None
        tp = close + p["tp_atr_mult"] * atr
        
        return (sl, tp, atr, bbwp)
    else:
        # 4. BB breakout (close below lower band)
        if close >= bb_lower:
            return None
        
        # 5. Stoch RSI confirmation
        if not _stoch_rsi_confirms(row, prev_row, "short"):
            return None
        
        # 6. EMA50 trend filter
        if close >= ema50:
            return None
        
        # 7. EMA200 macro filter
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
            "SIGNAL BBWP Squeeze v2 LONG | entry=%.2f SL=%.2f TP=%.2f BBWP=%.1f",
            float(curr["close"]), sl, tp, bbwp,
        )
        return _build_signal(float(curr["close"]), sl, tp, curr, True)
    
    result = _evaluate_direction(curr, prev, "short", idx=idx, df=df, profile=profile)
    if result is not None:
        sl, tp, atr, bbwp = result
        _register_signal(idx)
        logger.info(
            "SIGNAL BBWP Squeeze v2 SHORT | entry=%.2f SL=%.2f TP=%.2f BBWP=%.1f",
            float(curr["close"]), sl, tp, bbwp,
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
