r"""
strategy_bbwp_squeeze.py
------------------------
Squeeze Momentum Breakout Strategy v5 para BTC/USDT (15m e 1h).

v5 - MAIORES RETORNOS + MAIS TRADES:

  Problemas do v4 (730d: 72 trades, WR=43.1%, PF=1.35, PnL=+14.65%):
  1. Trailing 1.2x ATR MUITO JUSTO => 100% dos wins sao tp1_then_sl (trailing sai no BE)
  2. TP1=3.0x com 40% partial => avg win so +1.80% (trailing nao contribui)
  3. BBWP<5 ultra-raro => apenas 72 trades em 730 dias (1 a cada 10 dias)
  4. SL 1.8x aperta demais => 8 trades com SL em 1-2 barras (fake breakouts)
  5. R:R medio 1.25 => trailing nao cumpre seu papel

  Mudancas v4 -> v5:
  MAIORES RETORNOS POR TRADE:
  1. TRAILING: 1.2x -> 2.0x ATR (deixa ganhadores correrem de verdade)
  2. TP1: 3.0x -> 4.0x ATR (primeira saida maior)
  3. TP1_PCT: 40% -> 30% (deixa 70% para trailing capturar o movimento)
  4. BE_TRIGGER: 1.0x -> 1.5x ATR (espera mais profit antes de mover para BE)

  MAIS TRADES:
  5. BBWP_THRESHOLD: 5 -> 10 (3x mais squeeze events detectados)
  6. SQUEEZE_RECENT_BARS: 6 -> 10 (janela maior para detectar squeeze)
  7. VOLUME_MULT: 0.6 -> 0.5 (ainda menos restritivo)
  8. STOCH_RSI OB/OS: 65/35 -> 60/40 (mais sensivel ao momentum)
  9. COOLDOWN: 4 -> 3 (mais oportunidades)

  MENOS WHIPSAWS:
  10. SL: 1.8x -> 2.0x ATR (volta ao v3, evita fake breakouts 1-2 bars)
  11. BB_BREAKOUT_BUFFER: NOVO - close deve estar 15% da largura BB alem da banda
      (filtra wicks/toques vs breakouts reais com corpo)
  12. MAX_BARS: 120 -> 168 (7 dias em 1h - mais tempo para tendencias)

  Logica central:
  1. BBWP < 10 nos ultimos 10 bars (v4: < 5 em 6 bars)
  2. BBWP expandindo (delta > 0)
  3. BB breakout com corpo (close > upper + 15% BB width)
  4. Stoch RSI confirma momentum (K>=60 LONG, K<=40 SHORT)
  5. Volume >= 0.5 * SMA(Volume, 20)
  6. EMA50 alinha com tendencia
  7. EMA200 confirma tendencia macro

Gestao de risco v5:
  - SL: 2.0x ATR (volta ao v3 - evita whipsaws)
  - TP1: 4.0x ATR (partial 30%)
  - Trailing: 2.0x ATR apos BE trigger (1.5x ATR)
  - Max bars: 168 (7 dias em 1h)
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
    # ---- Squeeze Detection (v5: muito mais sensivel) ----
    "bbwp_threshold": 10,        # v5: 10 (de 5) — 3x mais squeeze events
    "squeeze_recent_bars": 10,    # v5: 10 (de 6) — janela maior
    "require_bbwp_expansion": True,  # BBWP deve estar expandindo (delta > 0)

    # ---- Entry Conditions (v5: significativamente mais sensivel) ----
    "volume_mult": 0.5,          # v5: 0.5 (de 0.6) — menos restritivo
    "stoch_rsi_ob": 60,           # v5: 60 (de 65) — mais sensivel ao momentum
    "stoch_rsi_os": 40,           # v5: 40 (de 35) — mais sensivel ao momentum
    "stoch_rsi_cross_enable": True,
    "stoch_rsi_min_delta": 0,     # Minimo delta K para confirmar momentum (0=any)
    "bb_breakout_buffer": 0.15,   # v5: NOVO — close deve estar 15% da BB width alem da banda

    # ---- Stop Loss (v5: volta ao 2.0x para evitar whipsaws) ----
    "sl_atr_mult": 2.0,           # v5: 2.0x ATR (de 1.8x) — evita fake breakouts
    "sl_atr_mult_high_vol": 2.0, # Mesmo em alta volatilidade
    "sl_atr_mult_low_vol": 2.0,  # Mesmo em baixa volatilidade

    # ---- Take Profit (v5: TP1 maior, menos partial) ----
    "tp_atr_mult": 4.0,           # v5: 4.0x ATR (de 3.0x) — TP1 maior
    "tp1_pct": 0.30,              # v5: Saida 30% no TP1 (de 40%), 70% para trailing

    # ---- Trailing Stop (v5: muito mais largo) ----
    "use_trailing": True,         # REATIVADO
    "be_trigger_atr_mult": 1.5,  # v5: 1.5x ATR (de 1.0x) — espera mais antes de BE
    "trailing_atr_mult": 2.0,     # v5: 2.0x ATR (de 1.2x) — DEIXA GANHADORES CORREREM

    # ---- RSI Divergence (OFF) ----
    "use_divergence_exit": False, # Desativado
    "divergence_min_bars": 3,

    # ---- General (v5: mais tempo, menos cooldown) ----
    "max_bars_held": 168,         # v5: 168 (de 120) — 7 dias em 1h
    "cooldown": 3,                # v5: 3 (de 4) — mais oportunidades
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
    bb_middle = float(row.get("bb_middle", 0))
    bb_width = float(row.get("bb_width", 0))
    ema50 = float(row.get("ema50", 0))
    ema200 = float(row.get("ema200", 0))
    atr = float(row.get("atr", 0))
    bbwp = float(row.get("bbwp", 100))
    
    if atr <= 0 or close <= 0 or bb_upper <= 0 or bb_lower <= 0:
        return None
    
    sl_mult = _get_sl_mult(row)
    
    # v5: BB breakout buffer — close deve estar alem da banda + buffer
    bb_buffer = p.get("bb_breakout_buffer", 0)
    bb_range = bb_upper - bb_lower
    
    if direction == "long":
        # 4. BB breakout com corpo (close > upper + buffer * BB width)
        breakout_level = bb_upper + bb_buffer * bb_range
        if close <= breakout_level:
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
        # 4. BB breakout com corpo (close < lower - buffer * BB width)
        breakout_level = bb_lower - bb_buffer * bb_range
        if close >= breakout_level:
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
