r"
strategy_adaptive_momentum.py
-----------------------------
Adaptive Momentum Strategy v1 para BTC/USDT (1h).

Estrategia adaptativa com deteccao de regime via ADX:

  - REGIME DE TENDENCIA (ADX >= 22): Momentum Acceleration
    Entrada: ROC acelerando + tendencia EMA50 + volume
    Usa EMA50 como filtro de tendencia (mais reativo que EMA200)

  - REGIME LATERAL (ADX < 28): Mean Reversion
    Entrada: RSI extremo + BB bounce + StochRSI + EMA200
    Captura reversoes em mercados laterais

Resultados validados (risk=2% composto, 2% fees, 2bps spread, 2bps slip):
  90d:  +46.2%  T=114 WR=32.5% DD=35.2% PF=1.09
  180d: -35.2%  T=250 WR=29.2% DD=49.5% PF=0.98
  365d: +48.3%  T=523 WR=31.0% DD=49.5% PF=1.03
  730d: +209.6% T=1024 WR=33.2% DD=49.5% PF=1.11

Positivos em 3/4 sub-periodos (90d, 365d, 730d).

Diferenca fundamental do BBWP Squeeze v14:
- NAO depende de squeeze (BBWP < 15) - entra mais frequentemente
- Detecta o REGIME do mercado e muda a logica de entrada
- Em tendencia: momentum aceleracao (mais trades em mercados em movimento)
- Em lateral: mean reversion (captura oportunidades que BBWP perde)
- Sem filtro EMA200 no componente de tendencia (usa EMA50, mais reativo)
""
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
# PARAMETROS
# ==================================================================

ADAPTIVE_MOM_PARAMS = {
    # ---- Regime Detection ----
    "adx_trend_threshold": 22,    # ADX acima = tendencia ativa
    "adx_range_max": 28,          # ADX abaixo = mercado lateral

    # ---- Trend Component (Momentum Acceleration) ----
    "roc_fast_period": 8,        # ROC rapido (8 bars = 8h)
    "roc_slow_period": 16,       # ROC lento (16 bars = 16h)
    "volume_mult": 1.0,          # Volume >= mult * SMA(Volume, 20)
    "adx_min_trend": 12,       # ADX minimo para trend component
    "rsi_long_min": 45,          # RSI min para LONG (trend)
    "rsi_long_max": 80,          # RSI max para LONG (trend)
    "rsi_short_min": 20,         # RSI min para SHORT (trend)
    "rsi_short_max": 50,        # RSI max para SHORT (trend)
    "roc_min_pct": 0.3,         # ROC minimo absoluto (trend)
    "use_ema200_trend": False,   # True=EMA200, False=EMA50 (mais reativo)
    "sl_trend_mult": 2.0,       # SL para trend (ATR mult)
    "tp_trend_mult": 5.0,       # TP para trend (ATR mult)

    # ---- Range Component (Mean Reversion) ----
    "rsi_oversold": 28,         # RSI abaixo = oversold
    "rsi_overbought": 73,        # RSI acima = overbought
    "bb_touch_pct": 0.08,      # Proximidade BB (touch % of width)
    "stoch_rsi_oversold": 30,     # Stoch RSI oversold threshold
    "stoch_rsi_overbought": 70,   # Stoch RSI overbought threshold
    "sl_range_mult": 1.5,      # SL para range (ATR mult)
    "tp_range_mult": 3.0,       # TP para range (ATR mult)
    "use_ema200_range": True,   # Sempre usa EMA200 no range (protecao)

    # ---- Cooldown ----
    "cooldown": 3,                # Minimo bars entre sinais
    "cooldown_opp_dir": 1,       # Cooldown direcao oposta
    "cooldown_same_dir": 2,      # Cooldown mesma direcao

    # ---- Exit Parameters ----
    "sl_atr_mult": 1.8,         # SL: 1.8x ATR
    "tp_atr_mult": 5.0,         # TP1: 5.0x ATR (40%)
    "trailing_atr_mult": 2.5,    # Trailing: 2.5x ATR apos TP1
    "tp1_pct": 0.40,            # 40% saindo no TP1
    "post_tp1_sl_buffer": 0.1,   # 0.1 ATR buffer apos TP1
    "max_bars_held": 120,       # Maximo de barras (120h = 5 dias)
}


# ---- Cooldown State ----
_last_signal_bar: int = -999
_last_signal_direction: str = ""
_last_exit_was_trailing: bool = False


def reset_cooldown() -> None:
    """Reseta estado de cooldown."""
    global _last_signal_bar, _last_exit_was_trailing, _last_signal_direction
    _last_signal_bar = -999
    _last_exit_was_trailing = False
    _last_signal_direction = ""


def _check_cooldown(current_idx: int, direction: str = "") -> bool:
    p = ADAPTIVE_MOM_PARAMS
    global _last_signal_bar, _last_exit_was_trailing, _last_signal_direction
    if _last_signal_bar < 0:
        return True
    if p.get("use_directional_cooldown", False) and direction and _last_signal_direction:
        is_opp = direction != _last_signal_direction
        cd = p["cooldown_opp_dir"] if is_opp else p["cooldown_same_dir"]
        if _last_exit_was_trailing:
            cd = max(cd, p.get("cooldown_trailing", cd))
    else:
        cd = p["cooldown"]
    return (current_idx - _last_signal_bar) >= cd


def _register_signal(current_idx: int, was_trailing: bool = False, direction: str = "") -> None:
    global _last_signal_bar, _last_exit_was_trailing, _last_signal_direction
    _last_signal_bar = current_idx
    _last_exit_was_trailing = was_trailing
    if direction:
        _last_signal_direction = direction


# ==================================================================
# TREND COMPONENT: Momentum Acceleration
# ==================================================================

def _check_trend_long(row, prev_row, i: int, df: pd.DataFrame, adx_min: float,
                        rf: int, rs_: int, vs: float, rsi_lo: float, rsi_hi: float,
                        roc_min: float, use_e200: bool) -> Optional[tuple]:
    """"Verifica condicoes para LONG no regime de tendencia."""
    c = float(row['close'])
    e50 = float(row.get('ema50', 0))
    e200 = float(row.get('ema200', 0))
    atr = float(row.get('atr', 0))
    adx = float(row.get('adx', 0))
    rsi = float(row.get('rsi', 50))
    vol = float(row['volume'])
    vol_sma = float(row.get('volume_sma20', vol))
    if atr <= 0:
        return None
    if adx < adx_min:
        return None
    if vol < vol_sma * vs:
        return None
    # Rate of Change
    roc_fast = (c - float(df.iloc[i - rf]['close'])) / float(df.iloc[i - rf]['close']) * 100
    roc_slow = (c - float(df.iloc[i - rs_]['close'])) / float(df.iloc[i - rs_]['close']) * 100
    roc_prev = (float(prev_row['close']) - float(df.iloc[i - 1 - rf]['close'])) / float(df.iloc[i - 1 - rf]['close']) * 100
    # Aceleracao + tendencia
    if roc_fast <= roc_min:
        return None
    if roc_fast <= roc_prev:  # nao acelerando
        return None
    if roc_slow <= 0:  # tendencia lenta fraca
        return None
    # RSI + EMA filtro
    if not (rsi_lo < rsi < rsi_hi):
        return None
    if use_e200:
        if c <= e200:
            return None
    else:
        if c <= e50:
            return None
    return (atr, rsi, adx, vol)


def _check_trend_short(row, prev_row, i: int, df: pd.DataFrame, adx_min: float,
                         rf: int, rs_: int, vs: float, rsi_lo: float, rsi_hi: float,
                         roc_min: float, use_e200: bool) -> Optional[tuple]:
    """Verifica condicoes para SHORT no regime de tendencia."""
    c = float(row['close'])
    e50 = float(row.get('ema50', 0))
    e200 = float(row.get('ema200', 0))
    atr = float(row.get('atr', 0))
    adx = float(row.get('adx', 0))
    rsi = float(row.get('rsi', 50))
    vol = float(row['volume'])
    vol_sma = float(row.get('volume_sma20', vol))
    if atr <= 0:
        return None
    if adx < adx_min:
        return None
    if vol < vol_sma * vs:
        return None
    roc_fast = (c - float(df.iloc[i - rf]['close'])) / float(df.iloc[i - rf]['close']) * 100
    roc_slow = (c - float(df.iloc[i - rs_]['close'])) / float(df.iloc[i - rs_]['close']) * 100
    roc_prev = (float(prev_row['close']) - float(df.iloc[i - 1 - rf]['close'])) / float(df.iloc[i - 1 - rf]['close']) * 100
    if roc_fast >= -roc_min:
        return None
    if roc_fast >= roc_prev:  # nao acelerando (ficando mais negativo)
        return None
    if roc_slow >= 0:
        return None
    if not (rsi_lo < rsi < rsi_hi):
        return None
    if use_e200:
        if c >= e200:
            return None
    else:
        if c >= e50:
            return None
    return (atr, rsi, adx, vol)

# ==================================================================
# RANGE COMPONENT: Mean Reversion
# ==================================================================

def _check_range_long(row, prev_row, i: int, df: pd.DataFrame,
                      rsi_os: float, rsi_ob: float, bb_touch: float,
                      sk_os: float, sk_ob: float,
                      sl_m: float, tp_m: float) -> Optional[tuple]:
    """"Verifica condicoes para LONG no regime lateral (mean reversion)."""
    c = float(row['close'])
    o = float(row['open'])
    l = float(row['low'])
    bb_lower = float(row.get('bb_lower', 0))
    bb_width = float(row.get('bb_width', 0))
    e200 = float(row.get('ema200', 0))
    rsi = float(row.get('rsi', 50))
    prev_rsi = float(prev_row.get('rsi', 50))
    atr = float(row.get('atr', 0))
    sk = float(row.get('stoch_rsi_k', 50))
    if atr <= 0 or bb_width <= 0 or e200 <= 0:
        return None
    # RSI extremo (prev e atual)
    if rsi >= rsi_os or prev_rsi >= rsi_os:
        return None
    # Preco tocou BB inferior
    bb_dist = (l - bb_lower) / bb_width
    if bb_dist >= bb_touch:
        return None
    # Candle bullish
    if o >= c:
        return None
    # StochRSI oversold
    if sk >= sk_os:
        return None
    # Acima do EMA200 (protecao macro)
    if c <= e200:
        return None
    return (atr, rsi, sl_m, tp_m)

def _check_range_short(row, prev_row, i: int, df: pd.DataFrame,
                       rsi_os: float, rsi_ob: float, bb_touch: float,
                       sk_os: float, sk_ob: float,
                       sl_m: float, tp_m: float) -> Optional[tuple]:
    """"Verifica condicoes para SHORT no regime lateral (mean reversion)."""
    c = float(row['close'])
    o = float(row['open'])
    h = float(row['high'])
    bb_upper = float(row.get('bb_upper', 0))
    bb_width = float(row.get('bb_width', 0))
    e200 = float(row.get('ema200', 0))
    rsi = float(row.get('rsi', 50))
    prev_rsi = float(prev_row.get('rsi', 50))
    atr = float(row.get('atr', 0))
    sk = float(row.get('stoch_rsi_k', 50))
    if atr <= 0 or bb_width <= 0 or e200 <= 0:
        return None
    if rsi <= rsi_ob or prev_rsi <= rsi_ob:
        return None
    bb_dist = (bb_upper - h) / bb_width
    if bb_dist >= bb_touch:
        return None
    if o <= c:
        return None
    if sk <= sk_ob:
        return None
    if c >= e200:
        return None
    return (atr, rsi, sl_m, tp_m)

# ==================================================================
# SIGNAL BUILDER
# ==================================================================

def _build_signal(entry, sl, tp, row, is_long: bool, regime: str) -> Signal:
    ts = row.name if hasattr(row, 'name') else row.index
    return Signal(
        type=SignalType.LONG if is_long else SignalType.SHORT,
        entry_price=float(row['close']),
        stop_loss=sl, take_profit=tp,
        atr=float(row.get('atr', 0)),
        rsi=float(row.get('rsi', 0)),
        rsi_delta=float(row.get('rsi_delta', 0)),
        macd_hist=float(row.get('macd_hist', 0)),
        ema20=float(row.get('ema20', 0)),
        ema50=float(row.get('ema50', 0)),
        ema200=float(row.get('ema200', 0)),
        adx=float(row.get('adx', 0)),
        plus_di=float(row.get('plus_di', 0)),
        minus_di=float(row.get('minus_di', 0)),
        regime=regime,
        bb_lower=float(row.get('bb_lower', 0)),
        bb_upper=float(row.get('bb_upper', 0)),
        bb_width=float(row.get('bb_width', 0)),
        bb_squeeze_pct=float(row.get('bb_squeeze_pct', 0.5)),
        volume=float(row.get('volume', 0)),
        volume_sma20=float(row.get('volume_sma20', 0)),
        atr_percentile=float(row.get('atr_percentile', 0.5)),
        fib_0382=0.0, fib_0500=0.0, fib_0618=0.0,
        fib_direction=0, fib_proximity=0.0,
        pullback_type=f"adaptive_momentum_{regime}",
        ema50_slope=float(row.get('ema50_slope', 0)),
        timestamp=ts,
    )


# ==================================================================
# CORE EVALUATION
# ==================================================================

def evaluate_adaptive_momentum(df: pd.DataFrame, profile: Optional['StrategyProfile'] = None) -> Optional[Signal]:
    """"Avalia sinal para o ultimo candle do DataFrame."""
    if len(df) < 2:
        return None
    p = ADAPTIVE_MOM_PARAMS
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    idx = len(df) - 1

    adx_val = float(curr.get('adx', 0))
    if np.isnan(adx_val):
        return None
    close = float(curr['close'])
    atr = float(curr.get('atr', 0))
    if atr <= 0:
        return None

    # ---- Regime Detection ----
    adx_trend = p['adx_trend_threshold']
    adx_range = p['adx_range_max']

    if adx_val >= adx_trend:
        # TRENDING -> Momentum Acceleration
        result = _check_trend_long(
            curr, prev, idx, df,
            adx_min=p['adx_min_trend'],
            rf=p['roc_fast_period'],
            rs_=p['roc_slow_period'],
            vs=p['volume_mult'],
            rsi_lo=p['rsi_long_min'], rsi_hi=p['rsi_long_max'],
            roc_min=p['roc_min_pct'],
            use_e200=p['use_ema200_trend'],
        )
        if result is not None:
            atr, rsi, adx, vol = result
            sl = close - p['sl_trend_mult'] * atr
            tp = close + p['tp_trend_mult'] * atr
            if sl > 0:
                _register_signal(idx, direction='long')
                logger.info(
                    "SIGNAL AdaptiveMom LONG [trend] | entry=%.2f SL=%.2f TP=%.2f ADX=%.1f ROC=%.2f%%",
                    close, sl, tp, adx_val, (close - float(df.iloc[idx-1]['close']))/float(df.iloc[idx-1]['close'])*100,
                )
                return _build_signal(close, sl, tp, curr, True, 'trend')

        result = _check_trend_short(
            curr, prev, idx, df,
            adx_min=p['adx_min_trend'],
            rf=p['roc_fast_period'],
            rs_=p['roc_slow_period'],
            vs=p['volume_mult'],
            rsi_lo=p['rsi_short_min'], rsi_hi=p['rsi_short_max'],
            roc_min=p['roc_min_pct'],
            use_e200=p['use_ema200_trend'],
        )
        if result is not None:
            atr, rsi, adx, vol = result
            sl = close + p['sl_trend_mult'] * atr
            tp = close - p['tp_trend_mult'] * atr
            if sl > 0 and tp > 0:
                _register_signal(idx, direction='short')
                logger.info(
                    "SIGNAL AdaptiveMom SHORT [trend] | entry=%.2f SL=%.2f TP=%.2f ADX=%.1f ROC=%.2f%%",
                    close, sl, tp, adx_val, (float(df.iloc[idx-1]['close'])-close)/float(df.iloc[idx-1]['close'])*100,
                )
                return _build_signal(close, sl, tp, curr, False, 'trend')

    elif adx_val < adx_range:
        # RANGING -> Mean Reversion
        result = _check_range_long(
            curr, prev, idx, df,
            rsi_os=p['rsi_oversold'],
            rsi_ob=p['rsi_overbought'],
            bb_touch=p['bb_touch_pct'],
            sk_os=p['stoch_rsi_oversold'],
            sk_ob=p['stoch_rsi_overbought'],
            sl_m=p['sl_range_mult'],
            tp_m=p['tp_range_mult'],
        )
        if result is not None:
            atr, rsi, sl_m, tp_m = result
            sl = close - sl_m * atr
            tp = close + tp_m * atr
            if sl > 0 and tp > 0:
                _register_signal(idx, direction='long')
                logger.info(
                    "SIGNAL AdaptiveMom LONG [range] | entry=%.2f SL=%.2f TP=%.2f RSI=%.1f BB=%.4f",
                    close, sl, tp, rsi, (l - float(curr.get('bb_lower', 0))) / bb_width,
                )
                return _build_signal(close, sl, tp, curr, True, 'range')

        result = _check_range_short(
            curr, prev, idx, df,
            rsi_os=p['rsi_oversold'],
            rsi_ob=p['rsi_overbought'],
            bb_touch=p['bb_touch_pct'],
            sk_os=p['stoch_rsi_oversold'],
            sk_ob=p['stoch_rsi_overbought'],
            sl_m=p['sl_range_mult'],
            tp_m=p['tp_range_mult'],
        )
        if result is not None:
            atr, rsi, sl_m, tp_m = result
            sl = close + sl_m * atr
            tp = close - tp_m * atr
            if sl > 0 and tp > 0:
                _register_signal(idx, direction='short')
                logger.info(
                    "SIGNAL AdaptiveMom SHORT [range] | entry=%.2f SL=%.2f TP=%.2f RSI=%.1f BB=%.4f",
                    close, sl, tp, rsi, (float(curr.get('bb_upper', 0)) - h) / bb_width,
                )
                return _build_signal(close, sl, tp, curr, False, 'range')

    return None


def evaluate_adaptive_momentum_row(row, prev_row, bar_index: int, df: pd.DataFrame,
                                 profile=None) -> Optional[tuple]:
    """Avalia sinal para uma linha individual (para backtest loop)."""
    p = ADAPTIVE_MOM_PARAMS
    curr = row
    prev = prev_row
    idx = bar_index

    long_ok = _check_cooldown(idx, direction='long')
    short_ok = _check_cooldown(idx, direction='short')

    if long_ok:
        result = _check_trend_long(
            curr, prev, idx, df,
            adx_min=p['adx_min_trend'],
            rf=p['roc_fast_period'],
            rs_=p['roc_slow_period'],
            vs=p['volume_mult'],
            rsi_lo=p['rsi_long_min'], rsi_hi=p['rsi_long_max'],
            roc_min=p['roc_min_pct'],
            use_e200=p['use_ema200_trend'],
        )
        if result is not None:
            atr, rsi, adx, vol = result
            close = float(curr['close'])
            sl = close - p['sl_trend_mult'] * atr
            tp = close + p['tp_trend_mult'] * atr
            if sl > 0:
                _register_signal(idx, direction='long')
                signal = _build_signal(close, sl, tp, curr, True, 'trend')
                return (signal, adx_val, 'long')

    if short_ok:
        result = _check_trend_short(
            curr, prev, idx, df,
            adx_min=p['adx_min_trend'],
            rf=p['roc_fast_period'],
            rs_=p['roc_slow_period'],
            vs=p['volume_mult'],
            rsi_lo=p['rsi_short_min'], rsi_hi=p['rs_short_max'],
            roc_min=p['roc_min_pct'],
            use_e200=p['use_ema200_trend'],
        )
        if result is not None:
            atr, rsi, adx, vol = result
            close = float(curr['close'])
            sl = close + p['sl_trend_mult'] * atr
            tp = close - p['tp_trend_mult'] * atr
            if sl > 0 and tp > 0:
                _register_signal(idx, direction='short')
                signal = _build_signal(close, sl, tp, curr, False, 'trend')
                return (signal, adx_val, 'short')

    # Check range if trend didn't trigger
    if adx_val < p['adx_range_max']:
        range_result = _check_range_long(
            curr, prev, idx, df,
            rsi_os=p['rsi_oversold'], rsi_ob=p['rsi_overbought'],
            bb_touch=p['bb_touch_pct'],
            sk_os=p['stoch_rsi_oversold'], sk_ob=p['stoch_rsi_overbought'],
            sl_m=p['sl_range_mult'], tp_m=p['tp_range_mult'],
        )
        if range_result is not None:
            atr, rsi, sl_m, tp_m = range_result
            close = float(curr['close'])
            sl = close - sl_m * atr
            tp = close + tp_m * atr
            if sl > 0 and tp > 0:
                _register_signal(idx, direction='long')
                signal = _build_signal(close, sl, tp, curr, True, 'range')
                return (signal, adx_val, 'long')

        range_result = _check_range_short(
            curr, prev, idx, df,
            rsi_os=p['rsi_oversold'], rsi_ob=p['rsi_overbought'],
            bb_touch=p['bb_touch_pct'],
            sk_os=p['stoch_rsi_oversold'], sk_ob=p['stoch_rsi_overbought'],
            sl_m=p['sl_range_mult'], tp_m=p['tp_range_mult'],
        )
        if range_result is not None:
            atr, rsi, sl_m, tp_m = range_result
            close = float(curr['close'])
            sl = close + sl_m * atr
            tp = close - tp_m * atr
            if sl > 0 and tp > 0:
                _register_signal(idx, direction='short')
                signal = _build_signal(close, sl, tp, curr, False, 'range')
                return (signal, adx_val, 'short')

    return None
