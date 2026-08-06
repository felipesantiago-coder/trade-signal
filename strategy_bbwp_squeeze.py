r"""
strategy_bbwp_squeeze.py
------------------------
Squeeze Momentum Breakout Strategy v6 para BTC/USDT (1h).

v6 - MAIORES RETORNOS POR TRADE + MAIS TRADES + FILTRO DE TENDENCIA:

  Problemas do v5 (730d: 88 trades, WR=40.9%, PF=1.12, PnL=+8.03%, DD=19.36%):
  1. Trailing 2.0x ATR LARGO DEMAI => 100% wins sao tp1_then_sl (60% sai no BE)
  2. TP1=4.0x com 30% partial => avg win so +2.12% (trailing nao contribui nada)
  3. BBWP<10 + buffer 15% => 88 trades em 730 dias (1 a cada 8.3 dias)
  4. Sem filtro ADX => entra em mercados sem tendencia (whipsaws)
  5. SL 2.0x => varios whipsaws de 1-2 barras (#21, #23, #38)
  6. R:R 1.32 com WR 40.9% => PF apenas 1.12
  7. 5268 sinais ATR filtrados => apenas 88 trades (1.7% de aproveitamento)

  Mudancas v5 -> v6:

  MAIS TRADES (relaxar filtros de entrada):
  1. BBWP_THRESHOLD: 10 -> 15 (mais squeeze events)
  2. BB_BREAKOUT_BUFFER: 0.15 -> 0 (remove buffer, so exige close alem da banda)
  3. VOLUME_MULT: 0.5 -> 0.4 (ainda menos restritivo)
  4. STOCH_RSI OB/OS: 60/40 -> 55/45 (mais sensivel ao momentum)
  5. COOLDOWN: 3 -> 2 (mais oportunidades)
  6. SQUEEZE_RECENT_BARS: 10 -> 12 (janela maior)

  MAIORES RETORNOS POR TRADE (fix trailing + TP):
  7. TP1: 4.0x -> 3.5x ATR (mais alcançavel, mais hits)
  8. TP1_PCT: 30% -> 40% (lock in mais no TP1)
  9. TRAILING: 2.0x -> 1.5x ATR (mais justo, racheta mais rapido)
  10. POST-TP1 SL: NOVO - SL = TP1 - trail*ATR (garante lucro na porcao trailing)
      (antes: SL = entry = BE, trailing contribuia 0%)

  MENOS WHIPSAWS:
  11. NOVO: ADX_MIN = 18 (so entra quando ha tendencia real)
  12. SL: 2.0x -> 2.2x ATR (evita fake breakouts de 1-2 barras)
  13. MAX_BARS: 168 -> 120 (5 dias em 1h, evita trades estagnados)

  Logica central:
  1. BBWP < 15 nos ultimos 12 bars (v5: < 10 em 10 bars)
  2. BBWP expandindo (delta > 0)
  3. BB breakout (close alem da banda, sem buffer)
  4. ADX > 18 (NOVO - confirma tendencia)
  5. Stoch RSI confirma momentum (K>=55 LONG, K<=45 SHORT)
  6. Volume >= 0.4 * SMA(Volume, 20)
  7. EMA50 alinha com tendencia
  8. EMA200 confirma tendencia macro

Gestao de risco v6:
  - SL: 2.2x ATR (mais largo, evita whipsaws)
  - TP1: 3.5x ATR (partial 40%)
  - Pos-TP1 SL: TP1 - 1.5x ATR (garante lucro no trailing)
  - Trailing: 1.5x ATR apos TP1 (racheta mais rapido)
  - Max bars: 120 (5 dias em 1h)
  - Cooldown: 2 bars (1 apos trailing exit)

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
    # ---- Squeeze Detection (v6: mais sensivel) ----
    "bbwp_threshold": 15,        # v6: 15 (de 10) — mais squeeze events
    "squeeze_recent_bars": 12,    # v6: 12 (de 10) — janela maior
    "require_bbwp_expansion": True,  # BBWP deve estar expandindo (delta > 0)

    # ---- Entry Conditions (v6: mais sensivel + ADX filter) ----
    "volume_mult": 0.4,          # v6: 0.4 (de 0.5) — menos restritivo
    "stoch_rsi_ob": 55,           # v6: 55 (de 60) — mais sensivel ao momentum
    "stoch_rsi_os": 45,           # v6: 45 (de 40) — mais sensivel ao momentum
    "stoch_rsi_cross_enable": True,
    "stoch_rsi_min_delta": 0,     # Minimo delta K para confirmar momentum (0=any)
    "bb_breakout_buffer": 0.0,    # v6: 0.0 (de 0.15) — remove buffer, so exige close alem da banda
    "adx_min": 18.0,              # v6: NOVO — ADX > 18 para confirmar tendencia

    # ---- Stop Loss (v6: mais largo para evitar whipsaws) ----
    "sl_atr_mult": 2.2,           # v6: 2.2x ATR (de 2.0x) — evita fake breakouts
    "sl_atr_mult_high_vol": 2.2, # Mesmo em alta volatilidade
    "sl_atr_mult_low_vol": 2.2,  # Mesmo em baixa volatilidade

    # ---- Take Profit (v6: TP1 mais realista, mais partial) ----
    "tp_atr_mult": 3.5,           # v6: 3.5x ATR (de 4.0x) — mais alcançavel
    "tp1_pct": 0.40,              # v6: Saida 40% no TP1 (de 30%), 60% para trailing

    # ---- Trailing Stop (v6: mais justo, racheta mais rapido) ----
    "use_trailing": True,         # REATIVADO
    "be_trigger_atr_mult": 1.0,  # v6: 1.0x ATR (de 1.5x) — BE mais cedo
    "trailing_atr_mult": 1.5,     # v6: 1.5x ATR (de 2.0x) — racheta mais rapido

    # ---- RSI Divergence (OFF) ----
    "use_divergence_exit": False, # Desativado
    "divergence_min_bars": 3,

    # ---- General (v6: menos cooldown, menos max bars) ----
    "max_bars_held": 120,         # v6: 120 (de 168) — 5 dias em 1h
    "cooldown": 2,                # v6: 2 (de 3) — mais oportunidades
    "cooldown_trailing": 1,

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
# ADX TREND FILTER (v6 NEW)
# ==================================================================

def _adx_confirms_trend(row) -> bool:
    """
    v6: Verifica se ADX confirma tendencia real.
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
        regime="bbwp_squeeze_v6",
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
        pullback_type="bbwp_squeeze_v6",
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

    # 2. v6: ADX trend filter (NOVO)
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

    # v6: BB breakout (sem buffer — so exige close alem da banda)
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
            "SIGNAL BBWP Squeeze v6 LONG | entry=%.2f SL=%.2f TP=%.2f BBWP=%.1f ADX=%.1f",
            float(curr["close"]), sl, tp, bbwp, float(curr.get("adx", 0)),
        )
        return _build_signal(float(curr["close"]), sl, tp, curr, True)

    result = _evaluate_direction(curr, prev, "short", idx=idx, df=df, profile=profile)
    if result is not None:
        sl, tp, atr, bbwp = result
        _register_signal(idx)
        logger.info(
            "SIGNAL BBWP Squeeze v6 SHORT | entry=%.2f SL=%.2f TP=%.2f BBWP=%.1f ADX=%.1f",
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
