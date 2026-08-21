r"""
strategy_router.py
------------------
Router inteligente que seleciona a estrategia correta para cada timeframe.

V13-ROBUSTA: 1h usa Squeeze Breakout + RSI Reversal (WFO validado, 17 janelas).
LIGA CRYPTO: 1h usa analise hierarquica multi-timeframe (1W->1D->4H->1H->15M).

  Timeframe  |  Estrategia              |  Engine              |  Validacao
  -----------+-------------------------+---------------------+------------------
  15m, 30m   |  ATF v2                  |  strategy_atf_v2     |  StochRSI+BBWP
  1h (LIGA)  |  Liga Crypto             |  strategy_liga_crypto |  Backtest/WFO ✅
  2h, 4h     |  CTEV (wider)            |  regime-switching    |  NAO validado
  1d          |  CTEV (position)         |  regime-switching    |  NAO validado
  1m,3m,5m   |  DESATIVADO              |  N/A                 |  Sem edge

Liga Crypto backtest/WFO:
  - backtest.py: run_backtest(liga_crypto=True) ou profile.name=="LIGA_CRYPTO"
  - sim_liga_crypto.py: simulate_liga_crypto() — simulador dedicado MTF
  - walk_forward_oos.py: run_walk_forward_oos_liga_crypto() — WFO dedicado
  - Busca 5 TFs (1W,1D,4H,1H,15M), computa indicadores, slice por timestamp
  - Filtro sazonal quantitativo (_get_seasonal_context_quant)

V13 1h estrategias ativas (quando LIGA_CRYPTO desativado):
  - Squeeze Breakout: SL 1.8x, TP 6.5x, max 144 bars, risk 3.0%
  - RSI Reversal:     SL 1.8x, TP 5.5x, max 120 bars, risk 1.5%
  - CTEV/EMA Bounce:  DESATIVADAS

Liga Crypto metodologia:
  - Hierarquia: 1W -> 1D -> 4H -> 1H -> 15M
  - Regra de ferro: MA200 diaria
  - Estocastico (14,3,3) + RSI divergencias + BBWP
  - R:R minimo 1:2, TP parcial 50/30/20
  - Relatorio formato Bloco 8
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

# Timeframes que usam V13 Multi-Strategy (Squeeze + RSI Reversal)
V13_MULTI_STRATEGY_TIMEFRAMES = set()  # Vazio quando LIGA_CRYPTO ativo

# Timeframes que usam Liga Crypto (analise hierarquica multi-TF)
LIGA_CRYPTO_TIMEFRAMES = {"1h"}

# DESATIVADOS
CONFLUENCE_V15_TIMEFRAMES = set()
BBWP_SQUEEZE_TIMEFRAMES = set()
ADAPTIVE_MOM_TIMEFRAMES = set()

# Timeframes que usam CTEV Regime-Switching (trend-following + mean-reversion)
REGIME_SWITCHING_TIMEFRAMES = {"2h", "4h", "1d"}

# Timeframes desativados (sem edge valida)
DISABLED_TIMEFRAMES = {"1m", "3m", "5m"}


def get_strategy_type(timeframe: str) -> str:
    """
    Retorna o tipo de estrategia para o timeframe dado.

    Returns:
        "atf" — Para 15m/30m (ATF v2 StochRSI + BBWP)
        "v13_multi_strategy" — Para 1h (Squeeze Breakout + RSI Reversal, WFO validado)
        "regime_switching" — Para 2h/4h/1d (CTEV v7.1, NAO validado)
        "disabled" — Para 1m/3m/5m (sem edge)
    """
    if timeframe in ATF_TIMEFRAMES:
        return "atf"
    elif timeframe in LIGA_CRYPTO_TIMEFRAMES:
        return "liga_crypto"
    elif timeframe in V13_MULTI_STRATEGY_TIMEFRAMES:
        return "v13_multi_strategy"
    elif timeframe in DISABLED_TIMEFRAMES:
        return "disabled"
    elif timeframe in CONFLUENCE_V15_TIMEFRAMES:
        return "confluence_v15"
    elif timeframe in BBWP_SQUEEZE_TIMEFRAMES:
        return "bbwp_squeeze"
    elif timeframe in ADAPTIVE_MOM_TIMEFRAMES:
        return "adaptive_momentum"
    else:
        # 2h, 4h, 1d = regime_switching (NAO validado)
        return "regime_switching"


def get_strategy_label(timeframe: str) -> str:
    """Retorna label descritivo da estrategia ativa."""
    st = get_strategy_type(timeframe)
    labels = {
        "atf": "ATF v2 StochRSI + BBWP",
        "liga_crypto": "Liga Crypto Multi-TF (1W→1D→4H→1H→15M)",
        "v13_multi_strategy": "V13-ROBUSTA Squeeze + RSI Reversal (WFO ✅)",
        "confluence_v15": "Confluence v15 (DESATIVADO)",
        "bbwp_squeeze": "BBWP Squeeze v14 (DESATIVADO)",
        "adaptive_momentum": "Adaptive Momentum v1 (DESATIVADO)",
        "regime_switching": "CTEV v7.1 Regime-Switching (NAO validado)",
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

    # ---- LIGA CRYPTO (1h) ----
    if strategy_type == "liga_crypto":
        from strategy_liga_crypto import evaluate_liga_crypto_signal
        logger.debug("Router [%s] -> Liga Crypto Multi-TF (1W->1D->4H->1H->15M)", timeframe)
        # Liga Crypto requer dados multi-timeframe passados pelo bot_worker
        return evaluate_liga_crypto_signal(df, timeframe=timeframe)

    # ---- V13 MULTI-STRATEGY (1h) ----
    if strategy_type == "v13_multi_strategy":
        from strategy import evaluate_signal as evaluate_v13_signal
        logger.debug("Router [%s] -> V13-ROBUSTA Multi-Strategy (Squeeze + RSI Reversal)", timeframe)
        return evaluate_v13_signal(df, profile=profile)

    # ---- CONFLUENCE v15 (DESATIVADO) ----
    if strategy_type == "confluence_v15":
        from strategy_confluence_v15 import evaluate_confluence_v15
        logger.debug("Router [%s] -> Confluence v15 Multi-Signal (DESATIVADO)", timeframe)
        return evaluate_confluence_v15(df, profile=profile)

    # ---- BBWP SQUEEZE (DESATIVADO) ----
    if strategy_type == "bbwp_squeeze":
        from strategy_bbwp_squeeze import evaluate_bbwp_squeeze
        logger.debug("Router [%s] -> BBWP Squeeze (DESATIVADO)", timeframe)
        return evaluate_bbwp_squeeze(df, profile=profile)

    # ---- ADAPTIVE MOMENTUM (DESATIVADO) ----
    if strategy_type == "adaptive_momentum":
        from strategy_adaptive_momentum import evaluate_adaptive_momentum
        logger.debug("Router [%s] -> Adaptive Momentum (DESATIVADO)", timeframe)
        return evaluate_adaptive_momentum(df, profile=profile)

    # ---- REGIME SWITCHING (2h/4h/1d, NAO validado) ----
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

    # ---- LIGA CRYPTO (1h) ----
    if strategy_type == "liga_crypto":
        from strategy_liga_crypto import evaluate_liga_crypto_signal_row
        logger.debug("Router [%s] -> Liga Crypto Multi-TF (1W->1D->4H->1H->15M)", timeframe)
        return evaluate_liga_crypto_signal_row(row, prev_row, bar_index, timeframe=timeframe)

    # ---- V13 MULTI-STRATEGY (1h) ----
    if strategy_type == "v13_multi_strategy":
        from strategy import evaluate_row_signals
        logger.debug("Router [%s] -> V13-ROBUSTA Multi-Strategy (Squeeze + RSI Reversal)", timeframe)
        return evaluate_row_signals(row, profile=profile)

    # ---- ATF v2 (15m/30m) ----
    if strategy_type == "atf":
        from strategy_atf_v2 import evaluate_atf_v2_row
        return evaluate_atf_v2_row(row, prev_row, bar_index, profile=profile)

    # ---- DESATIVADOS (confluence_v15, bbwp_squeeze, adaptive_momentum) ----
    if strategy_type == "confluence_v15":
        from strategy_confluence_v15 import evaluate_confluence_v15_row
        return evaluate_confluence_v15_row(row, prev_row, bar_index, df=None, profile=profile)

    if strategy_type == "bbwp_squeeze":
        from strategy_bbwp_squeeze import evaluate_bbwp_squeeze_row
        return evaluate_bbwp_squeeze_row(row, prev_row, bar_index, df=None, profile=profile)

    if strategy_type == "adaptive_momentum":
        from strategy_adaptive_momentum import evaluate_adaptive_momentum_row
        return evaluate_adaptive_momentum_row(row, prev_row, bar_index, df=df, profile=profile)

    # Para regime_switching (2h/4h/1d), usa o fluxo padrao do backtest
    # (a logica de regime e aplicada em _simulate_regime_switching)
    return None
