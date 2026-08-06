r"""
strategy_router.py
------------------
Router inteligente que seleciona a estrategia correta para cada timeframe.

Este e o cerebro do sistema de multi-estrategia. Em vez de usar a mesma
logica (CTEV trend-following) para todos os timeframes, o router escolhe
a estrategia otimizada para cada escala temporal:

  Timeframe  |  Estrategia         |  Engine              |  Validacao
  -----------+---------------------+---------------------+--------------
  15m, 30m   |  ATF v2              |  strategy_atf_v2     |  StochRSI+BBWP
  1h          |  BBWP Squeeze v2    |  strategy_bbwp_squeeze| squeeze+breakout
  2h, 4h     |  CTEV (wider)        |  regime-switching    |  NAO validado
  1d          |  CTEV (position)     |  regime-switching    |  NAO validado
  1m,3m,5m   |  DESATIVADO          |  N/A                 |  Sem edge

Principio de design:
  1. Cada timeframe tem uma estrategia DIFERENTE, nao apenas parametros
     diferentes. A logica de entrada muda completamente.
  2. O router e transparente: bot_worker e backtest nao precisam saber
     qual estrategia esta rodando. Apenas chamam evaluate_signal().
  3. MTF filter se adapta ao timeframe ativo (timeframes menores usam
     TFs de confirmacao mais proximos).

Integration:
  - bot_worker.py: substitui evaluate_signal_regime_aware direto
  - backtest.py: substitui _simulate_regime_switching para INTRADAY
"""
from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from strategy_profiles import StrategyProfile

from strategy_profiles import get_profile, PROFILE_INTRADAY, PROFILE_STANDARD

logger = logging.getLogger(__name__)


# ==================================================================
# TIMEFRAME -> STRATEGY TYPE MAPPING
# ==================================================================

# Timeframes que usam ATF v2 (intraday momentum)
ATF_TIMEFRAMES = {"15m", "30m"}

# Timeframes que usam BBWP Squeeze v2 (squeeze + breakout)
BBWP_SQUEEZE_TIMEFRAMES = {"1h"}

# Timeframes que usam CTEV Regime-Switching (trend-following + mean-reversion)
REGIME_SWITCHING_TIMEFRAMES = {"2h", "4h", "1d"}

# Timeframes desativados (sem edge valida)
DISABLED_TIMEFRAMES = {"1m", "3m", "5m"}


def get_strategy_type(timeframe: str) -> str:
    """
    Retorna o tipo de estrategia para o timeframe dado.

    Returns:
        "atf" — Para 15m/30m (ATF v2 StochRSI + BBWP)
        "bbwp_squeeze" — Para 1h (BBWP Squeeze v2)
        "regime_switching" — Para 2h/4h/1d (CTEV v7.1)
        "disabled" — Para 1m/3m/5m (sem edge)
    """
    if timeframe in ATF_TIMEFRAMES:
        return "atf"
    elif timeframe in BBWP_SQUEEZE_TIMEFRAMES:
        return "bbwp_squeeze"
    elif timeframe in DISABLED_TIMEFRAMES:
        return "disabled"
    else:
        return "regime_switching"


def get_strategy_label(timeframe: str) -> str:
    """Retorna label descritivo da estrategia ativa."""
    st = get_strategy_type(timeframe)
    labels = {
        "atf": "ATF v2 StochRSI + BBWP",
        "bbwp_squeeze": "BBWP Squeeze v6",
        "regime_switching": "CTEV v7.1 Regime-Switching",
        "disabled": "DESATIVADO (sem edge valida)",
    }
    return labels.get(st, "Unknown")


def get_mtf_timeframes(timeframe: str):
    """
    Retorna os timeframes de confirmacao MTF adequados para cada TF ativo.

    Logica: o MTF filter deve olhar 2-3 niveis acima do timeframe ativo.
      - 15m ativo -> confirma em 1h + 4h
      - 30m ativo -> confirma em 2h + 4h (ou 1h + 4h)
      - 1h ativo  -> confirma em 4h + 1d (original)
      - 2h ativo  -> confirma em 4h + 1d
      - 4h ativo  -> confirma em 1d (apenas macro)

    Returns:
        tuple (tf_confirm_1, tf_confirm_2) ou (tf_confirm,) para um nivel
    """
    mtf_map = {
        "15m": ("1h", "4h"),
        "30m": ("1h", "4h"),
        "1h":  ("4h", "1d"),
        "2h":  ("4h", "1d"),
        "4h":  ("1d",),
        "1d":  (None,),       # Sem MTF para diario
        "1m":  (None,),       # Desativado
        "3m":  (None,),       # Desativado
        "5m":  (None,),       # Desativado
    }
    return mtf_map.get(timeframe, ("4h", "1d"))


def evaluate_signal(
    df: pd.DataFrame,
    timeframe: str,
    profile: Optional[StrategyProfile] = None,
) -> Optional[object]:
    """
    Avalia sinal usando a estrategia correta para o timeframe.

    Esta e a funcao principal do router. Bot e backtest chamam esta
    funcao em vez de chamar evaluate_signal_regime_aware diretamente.

    Parameters:
        df: DataFrame com indicadores calculados
        timeframe: timeframe ativo ("15m", "1h", etc.)
        profile: StrategyProfile (se None, resolve automaticamente)

    Returns:
        Signal ou None
    """
    if profile is None:
        profile = get_profile(timeframe)

    strategy_type = get_strategy_type(timeframe)

    # ---- DESATIVADO ----
    if strategy_type == "disabled":
        logger.debug("Timeframe %s desativado (sem edge valida)", timeframe)
        return None

    # ---- ATF v2 (15m/30m) ----
    if strategy_type == "atf":
        from strategy_atf_v2 import evaluate_atf_v2
        logger.debug("Router [%s] -> ATF v2 StochRSI + BBWP", timeframe)
        return evaluate_atf_v2(df, profile=profile)

    # ---- BBWP SQUEEZE v2 (1h) ----
    if strategy_type == "bbwp_squeeze":
        from strategy_bbwp_squeeze import evaluate_bbwp_squeeze
        logger.debug("Router [%s] -> BBWP Squeeze v2", timeframe)
        return evaluate_bbwp_squeeze(df, profile=profile)

    # ---- REGIME SWITCHING (2h/4h/1d) ----
    if strategy_type == "regime_switching":
        from regime_engine import classify_regimes_v2
        from strategy_regime import evaluate_signal_regime_aware
        logger.debug("Router [%s] -> CTEV v7.1 Regime-Switching", timeframe)
        df_classified = classify_regimes_v2(df, hysteresis_bars=3)
        return evaluate_signal_regime_aware(df_classified, profile=profile, hysteresis_bars=3)

    return None


def evaluate_signal_row(
    row: pd.Series,
    prev_row: pd.Series,
    bar_index: int,
    timeframe: str,
    profile: Optional[StrategyProfile] = None,
) -> Optional[object]:
    """
    Avalia sinal em uma linha individual (para backtest loop).

    Parameters:
        row: linha atual do DataFrame
        prev_row: linha anterior
        bar_index: indice absoluto (para cooldown no EMA Cross)
        timeframe: timeframe ativo
        profile: StrategyProfile

    Returns:
        Signal ou None
    """
    if profile is None:
        profile = get_profile(timeframe)

    strategy_type = get_strategy_type(timeframe)

    if strategy_type == "disabled":
        return None

    if strategy_type == "atf":
        from strategy_atf_v2 import evaluate_atf_v2_row
        return evaluate_atf_v2_row(row, prev_row, bar_index, profile=profile)

    if strategy_type == "bbwp_squeeze":
        from strategy_bbwp_squeeze import evaluate_bbwp_squeeze_row
        return evaluate_bbwp_squeeze_row(row, prev_row, bar_index, df=None, profile=profile)

    # Para regime_switching, usa o fluxo padrao do backtest
    # (a logica de regime e aplicada em _simulate_regime_switching)
    return None
