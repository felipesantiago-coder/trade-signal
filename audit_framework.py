"""audit_framework.py
-------------------
Framework de Auditoria Avancada para Backtest.

Funcoes de analise importadas por report_auditor_v2.py:
  - Monte Carlo (5000+ paths) com percentis P5/P25/P50/P75/P95
  - Decomposicao por estrategia (contribution, WR, PnL por tipo)
  - Analise LONG vs SHORT (assimetria, edge direcional)
  - Auditoria de drawdown (timeline, recovery time, severity)
  - Score multi-objetivo (Edge, Risco, Robustez, Validacao, Overfitting, Amostra, MC)
  - Veredito 5 niveis: ROBUSTA / PROMISSORA / FRAGIL / NAO VALIDADA / REJEITADA
  - Analise de overfitting (parametro count, MC robustez, temporal decay)
  - Outlier analysis (top trades impacto, tail risk)
  - Regime analysis (performance por regime de mercado)
  - Estabilidade temporal (rolling window performance)
  - Equity curve completa (por trade e por timestamp)
  - 19 recomendacoes automaticas

Principio: AUDITABILIDADE > ROBUSTEZ > RISCO > CONSISTENCIA > RETORNO
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("ctev.audit_framework")


# ======================================================================
# DATA STRUCTURES
# ======================================================================

@dataclass
class MonteCarloResult:
    """Resultado do simulacao Monte Carlo."""
    n_simulations: int
    n_trades: int
    return_dist: Dict[str, float]  # P5/P25/P50/P75/P95
    dd_dist: Dict[str, float]     # P5/P25/P50/P75/P95 de max drawdown
    prob_loss: float              # % de paths com retorno negativo
    prob_profitable: float        # % de paths com retorno positivo
    obs_return_rank_pct: float    # ranking percentil do retorno observado
    obs_return_vs_p50: float      # retorno observado vs mediana MC (pp)


@dataclass
class OverfittingScore:
    """Score de risco de overfitting (0-100, menor = melhor)."""
    score: float      # 0-100
    level: str       # LOW / MEDIUM / HIGH / CRITICAL
    factors: List[str]
    details: str


@dataclass
class StrategyDecomposition:
    """Decomposicao de performance por estrategia."""
    name: str
    n_trades: int
    win_rate: float
    total_pnl_pct: float
    avg_pnl_pct: float
    contribution_pct: float
    best_trade: float
    worst_trade: float
    avg_r_multiple: float


@dataclass
class LongShortAnalysis:
    """Analise LONG vs SHORT."""
    long_trades: int
    short_trades: int
    long_wr: float
    short_wr: float
    long_pnl: float
    short_pnl: float
    long_pf: float
    short_pf: float
    long_avg_win: float
    long_avg_loss: float
    short_avg_win: float
    short_avg_loss: float
    asymmetry_ratio: float  # |long_pnl / short_pnl|, 1.0 = balanced


@dataclass
class DrawdownAuditResult:
    """Auditoria detalhada de drawdowns."""
    max_drawdown_pct: float
    max_dd_duration_bars: int
    max_dd_recovery_bars: int
    avg_drawdown_pct: float
    n_drawdowns_above_10pct: int
    n_drawdowns_above_20pct: int
    worst_recovery_time_bars: int
    drawdown_timeline: List[Dict[str, Any]]
    current_drawdown_pct: float


@dataclass
class MultiObjectiveScore:
    """Score multi-objetivo composto (0-100)."""
    edge_score: float         # 0-100
    risk_mgmt_score: float    # 0-100
    robustness_score: float   # 0-100
    validation_score: float   # 0-100
    overfitting_score: float  # 0-100 (inverted: 100 = sem overfitting)
    sample_score: float       # 0-100
    mc_score: float           # 0-100
    composite_score: float    # 0-100
    grade: str                # A/B/C/D/F


@dataclass
class RegimeAnalysisResult:
    """Analise de performance por regime de mercado."""
    regimes: Dict[str, Dict[str, Any]]


@dataclass
class OutlierResult:
    """Analise de outliers."""
    top_5_trades_pnl: List[Dict[str, Any]]
    top_5_trades_contribution_pct: float
    bottom_5_trades_pnl: List[Dict[str, Any]]
    tail_ratio: float
    kurtosis: float
    skewness: float
    max_single_trade_risk_pct: float


@dataclass
class TemporalStabilityResult:
    """Analise de estabilidade temporal."""
    window_days: int
    n_windows: int
    window_results: List[Dict[str, Any]]
    wr_std: float
    pnl_std: float
    consistency_score: float  # 0-100, higher = more consistent
    trend: str  # IMPROVING / STABLE / DEGRADING


# ======================================================================
# 1. MONTE CARLO SIMULATION (5000+ paths)
# ======================================================================

def run_monte_carlo(
    pnls: List[float],
    n_simulations: int = 5000,
    seed: Optional[int] = 42,
) -> MonteCarloResult:
    """
    Simulacao Monte Carlo com reamostragem bootstrap dos PnLs.
    Gera n_simulations paths, cada um com a mesma sequencia de trades
    reordenada aleatoriamente.
    Retorna distribuicao de retornos e drawdowns.

    FORMULA: Para cada simulacao s:
        1. Amostrar com reposicao len(pnls) PnLs
        2. Calcular retorno cumulativo: R_s = prod(1 + pnl_i/100) - 1
        3. Calcular max drawdown da curva de equity
    """
    if len(pnls) < 2:
        return MonteCarloResult(
            n_simulations=0, n_trades=len(pnls), return_dist={}, dd_dist={},
            prob_loss=100, prob_profitable=0, obs_return_rank_pct=0, obs_return_vs_p50=0,
        )

    rng = np.random.default_rng(seed)
    n = len(pnls)
    pnls_arr = np.array(pnls, dtype=np.float64)

    # Retorno observado
    obs_equity = np.cumprod(1 + pnls_arr / 100)
    obs_return = float(obs_equity[-1] - 1) * 100

    sim_returns = np.empty(n_simulations)
    sim_max_dds = np.empty(n_simulations)

    for s in range(n_simulations):
        # Bootstrap resampling
        sample_idx = rng.integers(0, n, size=n)
        sampled = pnls_arr[sample_idx]

        # Retorno cumulativo (composto)
        equity_path = np.cumprod(1 + sampled / 100)
        sim_returns[s] = float(equity_path[-1] - 1) * 100

        # Max drawdown do path
        peak = np.maximum.accumulate(equity_path)
        dd = (peak - equity_path) / peak * 100
        sim_max_dds[s] = float(np.max(dd)) if len(dd) > 0 else 0.0

    # Distribuicoes de percentis
    percentiles = [5, 25, 50, 75, 95]
    return_dist = {f"P{p}": float(np.percentile(sim_returns, p)) for p in percentiles}
    dd_dist = {f"P{p}": float(np.percentile(sim_max_dds, p)) for p in percentiles}

    # Probabilidades
    prob_loss = float(np.mean(sim_returns < 0)) * 100
    prob_profitable = float(np.mean(sim_returns > 0)) * 100

    # Ranking do retorno observado
    obs_rank = float(np.mean(sim_returns <= obs_return)) * 100
    obs_vs_p50 = obs_return - float(np.median(sim_returns))

    return MonteCarloResult(
        n_simulations=n_simulations,
        n_trades=n,
        return_dist=return_dist,
        dd_dist=dd_dist,
        prob_loss=round(prob_loss, 2),
        prob_profitable=round(prob_profitable, 2),
        obs_return_rank_pct=round(obs_rank, 1),
        obs_return_vs_p50=round(obs_vs_p50, 2),
    )


# ======================================================================
# 2. STRATEGY DECOMPOSITION
# ======================================================================

def decompose_strategies(
    trades: List[Dict[str, Any]],
    total_pnl: float,
) -> List[StrategyDecomposition]:
    """
    Decompoe os trades por entry_type e calcula metricas por estrategia.

    FORMULA:
      contribution_pct = (sum(pnl_strategy) / total_pnl_abs_ref) * 100
      total_pnl_abs_ref = max(|total_pnl|, 0.01) para evitar div/0
    """
    if not trades:
        return []

    # Group by entry_type
    strat_groups: Dict[str, List[Dict]] = {}
    for t in trades:
        et = t.get("entry_type", "unknown")
        if et not in strat_groups:
            strat_groups[et] = []
        strat_groups[et].append(t)

    total_pnl_abs_ref = max(abs(total_pnl), 0.01)

    results = []
    for name, group in strat_groups.items():
        n = len(group)
        pnls = [t.get("pnl_pct", 0) for t in group]
        wins = [p for p in pnls if p > 0]
        total_strat_pnl = sum(pnls)
        avg_pnl = np.mean(pnls) if pnls else 0.0
        wr = len(wins) / n * 100 if n > 0 else 0.0
        contribution = total_strat_pnl / total_pnl_abs_ref * 100

        # R-multiple
        r_mults = [t.get("r_multiple", 0) for t in group]
        avg_r = np.mean(r_mults) if r_mults else 0.0

        results.append(StrategyDecomposition(
            name=name,
            n_trades=n,
            win_rate=round(wr, 2),
            total_pnl_pct=round(total_strat_pnl, 4),
            avg_pnl_pct=round(avg_pnl, 4),
            contribution_pct=round(contribution, 2),
            best_trade=round(max(pnls), 4) if pnls else 0,
            worst_trade=round(min(pnls), 4) if pnls else 0,
            avg_r_multiple=round(avg_r, 2),
        ))

    # Sort by absolute contribution (most impactful first)
    results.sort(key=lambda x: abs(x.total_pnl_pct), reverse=True)
    return results


def analyze_portfolio_contribution(
    decomposition: List[StrategyDecomposition],
    total_pnl: float,
) -> Dict[str, Any]:
    """
    Analisa a contribuicao de cada estrategia para o portfolio.

    Retorna:
      - dominant_strategy: estrategia com maior contribuicao absoluta
      - concentration_hhi: indice Herfindahl-Hirschman de concentracao
      - diversification_score: 0-100 (100 = perfeitamente diversificado)
    """
    if not decomposition:
        return {"dominant_strategy": "N/A", "concentration_hhi": 0, "diversification_score": 0}

    # Calcular peso de contribuicao (share absoluto)
    contributions = [abs(d.total_pnl_pct) for d in decomposition]
    total_contrib = sum(contributions)
    if total_contrib == 0:
        return {"dominant_strategy": "N/A", "concentration_hhi": 0, "diversification_score": 0}

    shares = [c / total_contrib for c in contributions]

    # HHI: sum(s_i^2). Range [1/n, 1]. 1 = monopoly
    hhi = sum(s ** 2 for s in shares)

    # Dominant strategy
    dominant_idx = int(np.argmax(contributions))
    dominant = decomposition[dominant_idx].name

    # Diversification score: (1 - HHI) * 100, normalizado para n estrategias
    n = max(len(decomposition), 1)
    ideal_hhi = 1.0 / n  # HHI perfeito
    div_score = max(0, min(100, (1 - hhi) / (1 - ideal_hhi) * 100)) if ideal_hhi < 1 else 0

    return {
        "dominant_strategy": dominant,
        "concentration_hhi": round(hhi, 4),
        "diversification_score": round(div_score, 1),
        "n_active": n,
        "shares": {d.name: round(shares[i] * 100, 1) for i, d in enumerate(decomposition)},
    }


# ======================================================================
# 3. LONG vs SHORT ANALYSIS
# ======================================================================

def analyze_long_short(trades: List[Dict[str, Any]]) -> LongShortAnalysis:
    """
    Compara performance LONG vs SHORT.

    FORMULA:
      asymmetry_ratio = |long_pnl / short_pnl| (clipado a 10.0)
    """
    longs = [t for t in trades if t.get("type") == "LONG"]
    shorts = [t for t in trades if t.get("type") == "SHORT"]

    def _calc(group: List[Dict]) -> Tuple[int, float, float, float, float, float, float]:
        if not group:
            return 0, 0, 0, 0, 0, 0, 0
        pnls = [t.get("pnl_pct", 0) for t in group]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        wr = len(wins) / len(group) * 100
        total = sum(pnls)
        gross_win = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0.001
        pf = gross_win / gross_loss
        avg_w = np.mean(wins) if wins else 0
        avg_l = np.mean(losses) if losses else 0
        return len(group), wr, total, pf, avg_w, avg_l, 0

    l_n, l_wr, l_pnl, l_pf, l_aw, l_al, _ = _calc(longs)
    s_n, s_wr, s_pnl, s_pf, s_aw, s_al, _ = _calc(shorts)

    # Asymmetry ratio: |long_pnl / short_pnl|
    if abs(s_pnl) > 0.01:
        asym = min(abs(l_pnl / s_pnl), 10.0)
    else:
        asym = 10.0 if abs(l_pnl) > 0.01 else 1.0

    return LongShortAnalysis(
        long_trades=l_n, short_trades=s_n,
        long_wr=round(l_wr, 2), short_wr=round(s_wr, 2),
        long_pnl=round(l_pnl, 4), short_pnl=round(s_pnl, 4),
        long_pf=round(l_pf, 4), short_pf=round(s_pf, 4),
        long_avg_win=round(l_aw, 4), long_avg_loss=round(l_al, 4),
        short_avg_win=round(s_aw, 4), short_avg_loss=round(s_al, 4),
        asymmetry_ratio=round(asym, 2),
    )


# ======================================================================
# 4. DRAWDOWN AUDIT
# ======================================================================

def audit_drawdown_management(trades: List[Dict[str, Any]]) -> DrawdownAuditResult:
    """
    Analise detalhada dos drawdowns.

    FORMULA:
      Drawdown = (Peak - Equity) / Peak * 100
      Recovery time = bars desde o inicio do DD ate novo high
    """
    if not trades:
        return DrawdownAuditResult(
            max_drawdown_pct=0, max_dd_duration_bars=0, max_dd_recovery_bars=0,
            avg_drawdown_pct=0, n_drawdowns_above_10pct=0, n_drawdowns_above_20pct=0,
            worst_recovery_time_bars=0, drawdown_timeline=[], current_drawdown_pct=0,
        )

    # Build equity curve
    pnls = [t.get("pnl_pct", 0) for t in trades]
    equity = [10000.0]  # initial balance
    for p in pnls:
        equity.append(equity[-1] * (1 + p / 100))
    equity = np.array(equity)

    peak = np.maximum.accumulate(equity)
    dd_series = (peak - equity) / peak * 100

    # Find drawdown periods
    in_dd = False
    dd_start = 0
    dd_timeline = []
    max_dd = 0
    max_dd_duration = 0
    max_dd_recovery = 0
    worst_recovery = 0
    current_dd = 0

    for i in range(1, len(equity)):
        if dd_series[i] > 0.01:  # In drawdown (> 0.01%)
            if not in_dd:
                dd_start = i
                in_dd = True
            current_dd = dd_series[i]
            duration = i - dd_start
            if duration > max_dd_duration:
                max_dd_duration = duration
        else:
            if in_dd:
                # DD ended
                depth = float(np.max(dd_series[dd_start:i+1]))
                recovery = i - dd_start
                dd_timeline.append({
                    "start": dd_start,
                    "end": i,
                    "depth_pct": round(depth, 2),
                    "duration_bars": i - dd_start,
                    "recovery_bars": recovery,
                })
                if recovery > worst_recovery:
                    worst_recovery = recovery
                if recovery > max_dd_recovery:
                    max_dd_recovery = recovery
                in_dd = False
                current_dd = 0

    # If still in drawdown at end
    if in_dd:
        depth = float(np.max(dd_series[dd_start:]))
        dd_timeline.append({
            "start": dd_start,
            "end": len(equity) - 1,
            "depth_pct": round(depth, 2),
            "duration_bars": len(equity) - 1 - dd_start,
            "recovery_bars": -1,  # Not yet recovered
        })
        current_dd = depth

    max_dd = float(np.max(dd_series))
    avg_dd = float(np.mean(dd_series[dd_series > 1.0])) if np.any(dd_series > 1.0) else 0.0
    n_above_10 = sum(1 for d in dd_timeline if d["depth_pct"] >= 10)
    n_above_20 = sum(1 for d in dd_timeline if d["depth_pct"] >= 20)

    return DrawdownAuditResult(
        max_drawdown_pct=round(max_dd, 2),
        max_dd_duration_bars=max_dd_duration,
        max_dd_recovery_bars=max_dd_recovery,
        avg_drawdown_pct=round(avg_dd, 2),
        n_drawdowns_above_10pct=n_above_10,
        n_drawdowns_above_20pct=n_above_20,
        worst_recovery_time_bars=worst_recovery,
        drawdown_timeline=dd_timeline[:10],  # Top 10
        current_drawdown_pct=round(current_dd, 2),
    )


# ======================================================================
# 5. MULTI-OBJECTIVE SCORE
# ======================================================================

def compute_multi_objective_score(
    metrics: Dict[str, Any],
    trades: List[Dict[str, Any]],
    mc: Optional[MonteCarloResult] = None,
    overfitting: Optional[OverfittingScore] = None,
    temporal: Optional[TemporalStabilityResult] = None,
) -> MultiObjectiveScore:
    """
    Score multi-objetivo composto com 7 dimensoes.
    Pesos: Edge=15%, Risco=20%, Robustez=25%, Validacao=15%, Overfitting=10%, Amostra=10%, MC=5%

    FORMULA:
      composite = sum(score_i * weight_i)
      Grade: A>=80, B>=65, C>=50, D>=35, F<35
    """
    # --- Edge Score (0-100) ---
    pf = metrics.get("profit_factor", 0)
    expectancy = metrics.get("expectancy", 0)
    wr = metrics.get("win_rate", 0)

    # PF component (0-40): PF 1.0=0, PF 2.0+=40
    pf_score = min(40, max(0, (pf - 0.8) / 1.2 * 40))
    # Expectancy component (0-30): exp>0.1=30, exp<0=0
    exp_score = min(30, max(0, expectancy / 0.3 * 30))
    # WR component (0-30): wr 40%=10, 50%=20, 60%+=30
    wr_score = min(30, max(0, (wr - 30) / 30 * 30))
    edge_score = pf_score + exp_score + wr_score

    # --- Risk Management Score (0-100) ---
    dd = metrics.get("max_drawdown_pct", 100)
    recovery = metrics.get("recovery_factor", 0)
    calmar = metrics.get("calmar_ratio", 0)
    es = metrics.get("expected_shortfall", 0)

    # DD component (0-30): DD 10%=30, DD 50%=10, DD 100%=0
    dd_score = max(0, 30 - dd * 0.3)
    # Recovery component (0-25): RF 1.0=10, RF 5.0+=25
    rec_score = min(25, max(0, recovery / 5.0 * 25))
    # Calmar component (0-25): Calmar 0.5=10, Calmar 3.0+=25
    cal_score = min(25, max(0, calmar / 3.0 * 25))
    # ES component (0-20): ES -0.5%=20, ES -5%=0
    es_score = max(0, 20 - abs(es) * 4)
    risk_score = dd_score + rec_score + cal_score + es_score

    # --- Robustness Score (0-100) ---
    sharpe = metrics.get("sharpe_ratio", 0)
    sortino = metrics.get("sortino_ratio", 0)
    omega = metrics.get("omega_ratio", 0)

    # Sharpe component (0-35): Sharpe 0.5=10, 1.0=20, 2.0+=35
    sharpe_score = min(35, max(0, sharpe / 2.0 * 35))
    # Sortino component (0-35): Sortino 0.5=10, 1.0=20, 2.0+=35
    sortino_score = min(35, max(0, sortino / 2.0 * 35))
    # Omega component (0-30): Omega 1.0=0, 1.5=15, 2.0+=30
    omega_score = min(30, max(0, (omega - 0.8) / 1.2 * 30))
    robustness_score = sharpe_score + sortino_score + omega_score

    # --- Validation Score (0-100) ---
    total_trades = metrics.get("total_trades", 0)
    avg_rr = metrics.get("avg_r_r", 0)

    # Trade count (0-50): 20=20, 50=35, 100+=50
    trade_score = min(50, max(0, total_trades / 100 * 50))
    # R:R component (0-30): R:R 1.0=10, 2.0=20, 3.0+=30
    rr_score = min(30, max(0, avg_rr / 3.0 * 30))
    # Consistency (0-20): based on PF stability (PF>1 = 20)
    pf_stability = 20 if pf > 1.0 else max(0, 20 * (pf - 0.5) * 2)
    validation_score = trade_score + rr_score + pf_stability

    # --- Overfitting Score (inverted: 100 = no overfitting) ---
    if overfitting:
        of_score = max(0, 100 - overfitting.score)
    else:
        of_score = 50.0  # Neutral

    # --- Sample Score (0-100) ---
    if total_trades >= 100:
        sample_score = 100
    elif total_trades >= 50:
        sample_score = 70 + (total_trades - 50) / 50 * 30
    elif total_trades >= 20:
        sample_score = 40 + (total_trades - 20) / 30 * 30
    else:
        sample_score = total_trades / 20 * 40

    # --- Monte Carlo Score (0-100) ---
    if mc and mc.n_simulations > 0:
        # Based on: prob_profitable, obs_rank, obs_vs_p50
        mc_profit = mc.prob_profitable  # 0-100
        mc_rank = mc.obs_return_rank_pct  # 0-100
        mc_delta = max(0, min(50, 25 + mc.obs_return_vs_p50 * 2))
        mc_score = mc_profit * 0.4 + mc_rank * 0.4 + mc_delta * 0.2
    else:
        mc_score = 30.0  # Low confidence

    # --- Composite ---
    weights = {
        "edge": 0.15,
        "risk": 0.20,
        "robustness": 0.25,
        "validation": 0.15,
        "overfitting": 0.10,
        "sample": 0.10,
        "mc": 0.05,
    }

    composite = (
        edge_score * weights["edge"] +
        risk_score * weights["risk"] +
        robustness_score * weights["robustness"] +
        validation_score * weights["validation"] +
        of_score * weights["overfitting"] +
        sample_score * weights["sample"] +
        mc_score * weights["mc"]
    )

    # Grade
    if composite >= 80:
        grade = "A"
    elif composite >= 65:
        grade = "B"
    elif composite >= 50:
        grade = "C"
    elif composite >= 35:
        grade = "D"
    else:
        grade = "F"

    return MultiObjectiveScore(
        edge_score=round(edge_score, 1),
        risk_mgmt_score=round(risk_score, 1),
        robustness_score=round(robustness_score, 1),
        validation_score=round(validation_score, 1),
        overfitting_score=round(of_score, 1),
        sample_score=round(sample_score, 1),
        mc_score=round(mc_score, 1),
        composite_score=round(composite, 1),
        grade=grade,
    )


# ======================================================================
# 6. VEREDICTO 5 NIVEIS
# ======================================================================

def compute_verdict(
    metrics: Dict[str, Any],
    trades: List[Dict[str, Any]],
    mo: MultiObjectiveScore,
    mc: Optional[MonteCarloResult] = None,
    overfitting: Optional[OverfittingScore] = None,
    temporal: Optional[TemporalStabilityResult] = None,
    has_oos: bool = False,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Classifica a estrategia em 5 niveis:
    - ROBUSTA: Edge consistente, risco controlado, validacao forte (veredito do sistema de scoring)
    - PROMISSORA: Edge presente mas com gaps de validacao
    - FRAGIL: Edge fraco ou dependente de poucos trades
    - NAO VALIDADA: Dados insuficientes
    - REJEITADA: Sem edge ou risco inaceitavel

    FORMULA:
      Veredito baseado em composite_score + condicoes obrigatorias (gate checks)
    """
    total_trades = metrics.get("total_trades", 0)
    pf = metrics.get("profit_factor", 0)
    wr = metrics.get("win_rate", 0)
    dd = metrics.get("max_drawdown_pct", 100)
    pnl = metrics.get("total_pnl_pct", 0)
    composite = mo.composite_score

    details = {
        "edge_score": mo.edge_score,
        "risk_mgmt_score": mo.risk_mgmt_score,
        "robustness_score": mo.robustness_score,
        "validation_score": mo.validation_score,
        "overfitting_score": mo.overfitting_score,
        "sample_score": mo.sample_score,
        "mc_score": mo.mc_score,
        "composite_score": composite,
    }

    # Gate checks (condicoes obrigatorias)
    gates_passed = []
    gates_failed = []

    # Gate 1: Minimum trades
    if total_trades >= 30:
        gates_passed.append("min_trades (30+)")
    else:
        gates_failed.append(f"min_trades ({total_trades} < 30)")

    # Gate 2: Positive edge
    if pnl > 0:
        gates_passed.append("positive_return")
    else:
        gates_failed.append("negative_return")

    # Gate 3: Profitable after costs
    if pf > 1.0:
        gates_passed.append("pf_above_1")
    else:
        gates_failed.append(f"pf_below_1 ({pf:.2f})")

    # Gate 4: Drawdown not catastrophic
    if dd < 80:
        gates_passed.append("dd_acceptable (<80%)")
    else:
        gates_failed.append(f"dd_extreme ({dd:.1f}%)")

    details["gates_passed"] = gates_passed
    details["gates_failed"] = gates_failed

    # Verdict classification
    if composite >= 70 and len(gates_failed) == 0 and total_trades >= 50:
        if has_oos or (mc and mc.prob_profitable > 60):
            verdict = "\U0001f7e2 ROBUSTA"
            justification = (
                f"Score composto {composite:.1f}/100 (Grade {mo.grade}). "
                f"Todos os gates passaram. Edge consistente com {total_trades} trades, "
                f"PF={pf:.2f}, WR={wr:.1f}%, DD={dd:.1f}%. "
                f"{'Validacao OOS confirmada.' if has_oos else 'Monte Carlo robusto.'}"
            )
        else:
            verdict = "\U0001f7e1 PROMISSORA"
            justification = (
                f"Score composto {composite:.1f}/100 (Grade {mo.grade}). "
                f"Edge presente ({total_trades} trades, PF={pf:.2f}). "
                f"Gates OK, mas falta validacao OOS ou MC mais robusto."
            )
    elif composite >= 55 and len(gates_failed) <= 1:
        verdict = "\U0001f7e1 PROMISSORA"
        justification = (
            f"Score composto {composite:.1f}/100 (Grade {mo.grade}). "
            f"Edge presente mas com gaps: {', '.join(gates_failed) if gates_failed else 'nenhum'}. "
            f"Requer validacao adicional."
        )
    elif composite >= 35 and pnl > 0 and len(gates_failed) <= 2:
        verdict = "\U0001f7e0 FRAGIL"
        justification = (
            f"Score composto {composite:.1f}/100. Edge dependente de condicoes especificas. "
            f"Gates falhados: {', '.join(gates_failed)}. "
            f"Nao recomendado para capital real sem melhorias."
        )
    elif total_trades < 20:
        verdict = "\U0001f534 NAO VALIDADA"
        justification = (
            f"Amostra insuficiente: {total_trades} trades (minimo 20). "
            f"Impossivel determinar se a edge e real ou ruido estatistico."
        )
    else:
        verdict = "\u26ab REJEITADA"
        justification = (
            f"Score composto {composite:.1f}/100. Sem edge confiavel. "
            f"Gates falhados: {', '.join(gates_failed)}. "
            f"Requer redesenho completo da estrategia."
        )

    details["verdict"] = verdict
    return verdict, justification, details


# ======================================================================
# 7. OVERFITTING RISK ASSESSMENT
# ======================================================================

def assess_overfitting_risk(
    metrics: Dict[str, Any],
    trades: List[Dict[str, Any]],
    mc: Optional[MonteCarloResult] = None,
    temporal: Optional[TemporalStabilityResult] = None,
    n_active_strategies: int = 3,
    n_parameters: int = 15,
) -> OverfittingScore:
    """
    Avalia risco de overfitting baseado em multiplos fatores.

    Fatores:
      1. Ratio trades/parametros (>10 = LOW risk)
      2. Monte Carlo robustez (obs_return vs P50)
      3. Estabilidade temporal (consistency score)
      4. Numero de estrategias ativas (mais = menos overfit por estrategia)
      5. Concentracao de retornos (top 5 trades)

    FORMULA:
      score = f1*0.25 + f2*0.20 + f3*0.20 + f4*0.15 + f5*0.20
      Level: LOW(<30), MEDIUM(30-50), HIGH(50-70), CRITICAL(>70)
    """
    factors = []
    total_trades = metrics.get("total_trades", 0)
    pnls = [t.get("pnl_pct", 0) for t in trades]

    # Factor 1: Trades/Parameters ratio (0-100, higher = less overfitting)
    if n_parameters > 0:
        ratio = total_trades / n_parameters
        if ratio >= 10:
            f1 = 20  # Very safe
        elif ratio >= 5:
            f1 = 40
        elif ratio >= 3:
            f1 = 60
        else:
            f1 = 80
        factors.append(f"Trades/Params={ratio:.1f} ({'safe' if ratio >= 10 else 'concern'})")
    else:
        f1 = 50

    # Factor 2: Monte Carlo robustez (0-100)
    if mc and mc.n_simulations > 0:
        # obs_return should be above P25 to be robust
        if mc.obs_return_rank_pct > 75:
            f2 = 10  # Very robust
        elif mc.obs_return_rank_pct > 50:
            f2 = 30
        elif mc.obs_return_rank_pct > 25:
            f2 = 50
        else:
            f2 = 80
        factors.append(f"MC rank={mc.obs_return_rank_pct:.0f}%")
    else:
        f2 = 50
        factors.append("MC: sem dados")

    # Factor 3: Temporal stability (0-100)
    if temporal and temporal.n_windows >= 3:
        f3 = max(0, 100 - temporal.consistency_score)
        factors.append(f"Temporal consistency={temporal.consistency_score:.0f}%")
    else:
        f3 = 40
        factors.append("Temporal: janelas insuficientes")

    # Factor 4: Strategy diversification (0-100)
    if n_active_strategies >= 4:
        f4 = 20
    elif n_active_strategies >= 3:
        f4 = 30
    elif n_active_strategies >= 2:
        f4 = 50
    else:
        f4 = 70
    factors.append(f"{n_active_strategies} estrategias ativas")

    # Factor 5: Return concentration (0-100, higher = more concentrated = more risk)
    if len(pnls) >= 5:
        sorted_pnls = sorted(pnls, reverse=True)
        top5_share = sum(sorted_pnls[:5]) / max(sum(abs(p) for p in pnls), 0.01)
        if top5_share < 0.3:
            f5 = 20
        elif top5_share < 0.5:
            f5 = 40
        elif top5_share < 0.7:
            f5 = 60
        else:
            f5 = 80
        factors.append(f"Top-5 trades = {top5_share*100:.0f}% dos ganhos")
    else:
        f5 = 50

    # Composite overfitting risk
    score = f1 * 0.25 + f2 * 0.20 + f3 * 0.20 + f4 * 0.15 + f5 * 0.20

    if score < 30:
        level = "LOW"
    elif score < 50:
        level = "MEDIUM"
    elif score < 70:
        level = "HIGH"
    else:
        level = "CRITICAL"

    details = (
        f"Score={score:.1f}/100. Fatores: "
        f"F1(trades/params)={f1:.0f}, F2(MC)={f2:.0f}, "
        f"F3(temporal)={f3:.0f}, F4(diversification)={f4:.0f}, "
        f"F5(concentration)={f5:.0f}. Level: {level}."
    )

    return OverfittingScore(score=round(score, 1), level=level, factors=factors, details=details)


# ======================================================================
# 8. OUTLIER ANALYSIS
# ======================================================================

def run_outlier_analysis(trades: List[Dict[str, Any]]) -> OutlierResult:
    """
    Identifica outliers e avalia dependencia de poucos trades.

    FORMULA:
      tail_ratio = P90(|PnL|) / P50(|PnL|)
      kurtosis = momento 4 central normalizado
      skewness = momento 3 central normalizado
    """
    if not trades:
        return OutlierResult(
            top_5_trades_pnl=[], top_5_trades_contribution_pct=0,
            tail_ratio=0, kurtosis=0, skewness=0, max_single_trade_risk_pct=0,
        )

    pnls = [t.get("pnl_pct", 0) for t in trades]
    sorted_pnls = sorted(pnls, reverse=True)
    total_gains = sum(p for p in pnls if p > 0)

    # Top 5 trades
    top5 = sorted_pnls[:5]
    top5_contribution = sum(top5) / max(total_gains, 0.01) * 100

    top5_details = []
    for t in trades:
        if t.get("pnl_pct", 0) in top5 and len(top5_details) < 5:
            top5_details.append({
                "entry_ts": str(t.get("entry_ts", ""))[:16],
                "type": t.get("type", ""),
                "entry_type": t.get("entry_type", ""),
                "pnl_pct": round(t.get("pnl_pct", 0), 4),
                "r_multiple": t.get("r_multiple", 0),
            })

    # Bottom 5 trades
    bottom5 = sorted_pnls[-5:]
    bottom5_details = []
    for t in trades:
        if t.get("pnl_pct", 0) in bottom5 and len(bottom5_details) < 5:
            bottom5_details.append({
                "entry_ts": str(t.get("entry_ts", ""))[:16],
                "type": t.get("type", ""),
                "entry_type": t.get("entry_type", ""),
                "pnl_pct": round(t.get("pnl_pct", 0), 4),
                "r_multiple": t.get("r_multiple", 0),
            })

    # Statistical measures
    pnls_arr = np.array(pnls)
    abs_pnls = np.abs(pnls_arr)

    if len(pnls_arr) > 1:
        p90 = float(np.percentile(abs_pnls, 90))
        p50 = float(np.percentile(abs_pnls, 50))
        tail_ratio = p90 / p50 if p50 > 0 else 0
        kurtosis = float(_kurtosis(pnls_arr))
        skewness = float(_skewness(pnls_arr))
    else:
        tail_ratio = 0
        kurtosis = 0
        skewness = 0

    # Max single trade risk (worst trade as % of initial balance)
    worst_pnl = min(pnls) if pnls else 0
    max_risk = abs(worst_pnl)  # As % of balance (since pnl_pct is already %)

    return OutlierResult(
        top_5_trades_pnl=top5_details,
        top_5_trades_contribution_pct=round(top5_contribution, 1),
        bottom_5_trades_pnl=bottom5_details,
        tail_ratio=round(tail_ratio, 2),
        kurtosis=round(kurtosis, 2),
        skewness=round(skewness, 2),
        max_single_trade_risk_pct=round(max_risk, 2),
    )


def _kurtosis(arr: np.ndarray) -> float:
    if len(arr) < 4:
        return 0.0
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        return 0.0
    return float(np.mean(((arr - mean) / std) ** 4) - 3)  # excess kurtosis


def _skewness(arr: np.ndarray) -> float:
    if len(arr) < 3:
        return 0.0
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        return 0.0
    return float(np.mean(((arr - mean) / std) ** 3))


# ======================================================================
# 9. REGIME ANALYSIS
# ======================================================================

def run_regime_analysis(trades: List[Dict[str, Any]]) -> RegimeAnalysisResult:
    """
    Analisa performance por regime de mercado.
    Regimes extraidos do campo 'regime_at_entry' dos trades.

    FORMULA:
      Por regime: WR = wins/trades, PnL = sum(pnl_pct), Avg = mean(pnl_pct)
    """
    if not trades:
        return RegimeAnalysisResult(regimes={})

    regime_groups: Dict[str, List[Dict]] = {}
    for t in trades:
        regime = t.get("regime_at_entry", "unknown")
        if not regime:
            regime = "unknown"
        if regime not in regime_groups:
            regime_groups[regime] = []
        regime_groups[regime].append(t)

    regimes = {}
    for name, group in regime_groups.items():
        pnls = [t.get("pnl_pct", 0) for t in group]
        wins = [p for p in pnls if p > 0]
        n = len(group)
        wr = len(wins) / n * 100 if n > 0 else 0
        total = sum(pnls)
        avg = np.mean(pnls) if pnls else 0
        regimes[name] = {
            "n_trades": n,
            "win_rate": round(wr, 2),
            "total_pnl_pct": round(total, 4),
            "avg_pnl_pct": round(avg, 4),
            "share_pct": round(n / len(trades) * 100, 1),
        }

    return RegimeAnalysisResult(regimes=regimes)


# ======================================================================
# 10. TEMPORAL STABILITY
# ======================================================================

def run_temporal_stability(
    trades: List[Dict[str, Any]],
    window_days: int = 30,
) -> Optional[TemporalStabilityResult]:
    """
    Analisa estabilidade temporal usando janelas rolantes.

    FORMULA:
      Para cada janela de window_days:
        WR, PnL, PF, n_trades
      consistency_score = 100 - (std(WR) + std(PnL)) / 2 normalizado
    """
    if not trades or len(trades) < 5:
        return None

    # Parse timestamps and sort
    dated_trades = []
    for t in trades:
        ts = t.get("entry_ts", "")
        if isinstance(ts, str) and len(ts) >= 10:
            try:
                dt = pd.Timestamp(ts)
                dated_trades.append((dt, t))
            except Exception:
                continue

    if len(dated_trades) < 5:
        return None

    dated_trades.sort(key=lambda x: x[0])
    start_date = dated_trades[0][0]
    end_date = dated_trades[-1][0]
    total_days = max((end_date - start_date).total_seconds() / 86400, 1)
    n_windows = max(1, int(total_days / window_days))

    if n_windows < 2:
        return None

    window_results = []
    for w in range(n_windows):
        w_start = start_date + pd.Timedelta(days=w * window_days)
        w_end = w_start + pd.Timedelta(days=window_days)
        w_trades = [t for dt, t in dated_trades if w_start <= dt < w_end]

        if not w_trades:
            window_results.append({
                "start": str(w_start)[:10],
                "end": str(w_end)[:10],
                "n_trades": 0,
                "win_rate": 0,
                "pnl_pct": 0,
                "profit_factor": 0,
            })
            continue

        pnls = [t.get("pnl_pct", 0) for t in w_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        wr = len(wins) / len(w_trades) * 100
        total_pnl = sum(pnls)
        pf = sum(wins) / max(abs(sum(losses)), 0.001)

        window_results.append({
            "start": str(w_start)[:10],
            "end": str(w_end)[:10],
            "n_trades": len(w_trades),
            "win_rate": round(wr, 2),
            "pnl_pct": round(total_pnl, 4),
            "profit_factor": round(pf, 4),
        })

    # Compute consistency metrics
    valid_windows = [w for w in window_results if w["n_trades"] > 0]
    if len(valid_windows) < 2:
        return None

    wrs = [w["win_rate"] for w in valid_windows]
    pnls_w = [w["pnl_pct"] for w in valid_windows]

    wr_std = float(np.std(wrs))
    pnl_std = float(np.std(pnls_w))

    # Consistency score: lower std = higher consistency
    # WR std: 0=100, 30+=0
    wr_consistency = max(0, 100 - wr_std * 3.33)
    pnl_consistency = max(0, 100 - min(abs(pnl_std) * 2, 100))
    consistency = wr_consistency * 0.5 + pnl_consistency * 0.5

    # Trend detection
    if len(pnls_w) >= 3:
        first_half = np.mean(pnls_w[:len(pnls_w)//2])
        second_half = np.mean(pnls_w[len(pnls_w)//2:])
        diff = second_half - first_half
        if diff > pnl_std * 0.5:
            trend = "IMPROVING"
        elif diff < -pnl_std * 0.5:
            trend = "DEGRADING"
        else:
            trend = "STABLE"
    else:
        trend = "STABLE"

    return TemporalStabilityResult(
        window_days=window_days,
        n_windows=len(window_results),
        window_results=window_results,
        wr_std=round(wr_std, 2),
        pnl_std=round(pnl_std, 2),
        consistency_score=round(consistency, 1),
        trend=trend,
    )


# ======================================================================
# 11. FULL EQUITY CURVE
# ======================================================================

def build_full_equity_curve(
    trades: List[Dict[str, Any]],
    initial_balance: float = 10000.0,
) -> List[Dict[str, Any]]:
    """
    Constroi equity curve completa com dados por trade.

    FORMULA:
      equity[i] = equity[i-1] * (1 + pnl_pct / 100)
      drawdown[i] = (peak - equity[i]) / peak * 100
    """
    if not trades:
        return []

    curve = []
    balance = initial_balance
    peak = initial_balance

    for i, t in enumerate(trades):
        pnl = t.get("pnl_pct", 0)
        pos_usd = t.get("position_usd", 0)
        if pos_usd > 0:
            pnl_usd = pos_usd * (pnl / 100)
        else:
            pnl_usd = balance * (pnl / 100)  # Fallback

        balance += pnl_usd
        peak = max(peak, balance)
        dd = (peak - balance) / peak * 100

        curve.append({
            "trade_num": i + 1,
            "timestamp": str(t.get("entry_ts", ""))[:16],
            "type": t.get("type", ""),
            "entry_type": t.get("entry_type", ""),
            "pnl_pct": round(pnl, 4),
            "pnl_usd": round(pnl_usd, 2),
            "balance": round(balance, 2),
            "peak": round(peak, 2),
            "drawdown_pct": round(dd, 2),
            "return_cumulative_pct": round((balance / initial_balance - 1) * 100, 2),
            "r_multiple": t.get("r_multiple", 0),
            "concurrent_count": t.get("concurrent_count", 0),
        })

    return curve


# ======================================================================
# 12. 19-POINT RECOMMENDATION ENGINE
# ======================================================================

def generate_19_point_recommendation(
    metrics: Dict[str, Any],
    mo: MultiObjectiveScore,
    mc: Optional[MonteCarloResult] = None,
    overfitting: Optional[OverfittingScore] = None,
    temporal: Optional[TemporalStabilityResult] = None,
    regime: Optional[RegimeAnalysisResult] = None,
    has_oos: bool = False,
) -> List[Dict[str, str]]:
    """
    Gera ate 19 recomendacoes automaticas baseadas nas analises.
    Cada recomendacao: {id, area, severity, action, rationale}
    Severity: CRITICAL / HIGH / MEDIUM / LOW / INFO
    """
    recs = []

    total_trades = metrics.get("total_trades", 0)
    pf = metrics.get("profit_factor", 0)
    dd = metrics.get("max_drawdown_pct", 0)
    wr = metrics.get("win_rate", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    sortino = metrics.get("sortino_ratio", 0)
    cagr = metrics.get("cagr", 0)
    recovery = metrics.get("recovery_factor", 0)
    es = metrics.get("expected_shortfall", 0)
    avg_rr = metrics.get("avg_r_r", 0)

    # R1: Drawdown
    if dd > 50:
        recs.append({"id": "R1", "area": "Risco", "severity": "CRITICAL",
            "action": f"Reduzir Max Drawdown de {dd:.1f}% para <30%",
            "rationale": f"DD={dd:.1f}% expoe capital a risco extremo. Reduzir position sizing, adicionar circuit breakers."})
    elif dd > 30:
        recs.append({"id": "R1", "area": "Risco", "severity": "HIGH",
            "action": f"Monitorar e reduzir Max Drawdown de {dd:.1f}%",
            "rationale": f"DD acima de 30% e preocupante mas gerenciavel com controles adequados."})
    else:
        recs.append({"id": "R1", "area": "Risco", "severity": "INFO",
            "action": f"Max Drawdown {dd:.1f}% dentro de limites aceitaveis",
            "rationale": "Drawdown controlado indica boa gestao de risco."})

    # R2: Profit Factor
    if pf < 1.0:
        recs.append({"id": "R2", "area": "Edge", "severity": "CRITICAL",
            "action": f"Profit Factor {pf:.2f} < 1.0 — estrategia nao e lucrativa apos custos",
            "rationale": "PF<1 indica que a estrategia perde dinheiro. Requer redesenho de entrada/saida."})
    elif pf < 1.2:
        recs.append({"id": "R2", "area": "Edge", "severity": "HIGH",
            "action": f"Profit Factor {pf:.2f} marginal — otimizar filtros de entrada",
            "rationale": "PF entre 1.0 e 1.2 pode ser consumido por custos em live trading."})
    else:
        recs.append({"id": "R2", "area": "Edge", "severity": "INFO",
            "action": f"Profit Factor {pf:.2f} saudavel",
            "rationale": "Edge positiva e sustentavel."})

    # R3: Win Rate
    if wr < 30:
        recs.append({"id": "R3", "area": "Edge", "severity": "HIGH",
            "action": f"Win Rate {wr:.1f}% muito baixo — risco psicologico alto",
            "rationale": "WR<30% gera sequencias longas de perdas. Considerar filtros mais seletivos."})
    elif wr > 70:
        recs.append({"id": "R3", "area": "Robustez", "severity": "MEDIUM",
            "action": f"Win Rate {wr:.1f}% suspeito — possivel overfitting",
            "rationale": "WR muito alto em backtest pode indicar sobre-otimizacao. Validar OOS."})

    # R4: Sample Size
    if total_trades < 30:
        recs.append({"id": "R4", "area": "Amostra", "severity": "CRITICAL",
            "action": f"Apenas {total_trades} trades — amostra estatisticamente insignificante",
            "rationale": "Minimo 30 trades para confiabilidade. Ideal: 100+. Estender periodo ou reduzir filtros."})
    elif total_trades < 50:
        recs.append({"id": "R4", "area": "Amostra", "severity": "HIGH",
            "action": f"{total_trades} trades — ampliar amostra para 50+",
            "rationale": "50+ trades fornecem confianca estatistica razoavel."})

    # R5: Sharpe
    if sharpe < 0.5:
        recs.append({"id": "R5", "area": "Robustez", "severity": "HIGH",
            "action": f"Sharpe {sharpe:.2f} abaixo do limiar de 0.5",
            "rationale": "Sharpe<0.5 indica retorno insuficiente por unidade de risco."})
    elif sharpe >= 1.5:
        recs.append({"id": "R5", "area": "Robustez", "severity": "INFO",
            "action": f"Sharpe {sharpe:.2f} excelente",
            "rationale": "Retorno ajustado ao risco muito bom."})

    # R6: Sortino
    if sortino < 0.5 and sharpe > 0.5:
        recs.append({"id": "R6", "area": "Risco", "severity": "MEDIUM",
            "action": f"Sortino {sortino:.2f} << Sharpe {sharpe:.2f} — downside desproporcional",
            "rationale": "Gap Sharpe-Sortino indica perdas grandes esporadicas. Investigar outliers."})

    # R7: Recovery Factor
    if recovery < 1.0 and dd > 10:
        recs.append({"id": "R7", "area": "Risco", "severity": "HIGH",
            "action": f"Recovery Factor {recovery:.2f} < 1.0 — recuperacao lenta apos drawdowns",
            "rationale": "RF<1 significa que o retorno total nao cobre o max drawdown."})

    # R8: Expected Shortfall
    if es < -2.0:
        recs.append({"id": "R8", "area": "Risco", "severity": "HIGH",
            "action": f"Expected Shortfall {es:.2f}% — tail risk elevado",
            "rationale": f"No pior 5% dos casos, perda media de {abs(es):.2f}% por trade."})

    # R9: R:R Ratio
    if avg_rr < 1.0:
        recs.append({"id": "R9", "area": "Edge", "severity": "MEDIUM",
            "action": f"R:R medio {avg_rr:.2f} < 1.0 — risco supera recompensa",
            "rationale": "R:R<1 exige WR>50% para ser lucrativo."})
    elif avg_rr >= 2.0:
        recs.append({"id": "R9", "area": "Edge", "severity": "INFO",
            "action": f"R:R medio {avg_rr:.2f} favoravel",
            "rationale": "Boa relacao risco/recompensa."})

    # R10: Monte Carlo
    if mc and mc.n_simulations > 0:
        if mc.prob_loss > 30:
            recs.append({"id": "R10", "area": "Validacao", "severity": "CRITICAL",
                "action": f"MC: {mc.prob_loss:.0f}% dos paths com perda — estrategia nao e robusta",
                "rationale": f"Com 5000 simulacoes, {mc.prob_loss:.0f}% dos cenarios resultam em perda."})
        elif mc.prob_loss > 10:
            recs.append({"id": "R10", "area": "Validacao", "severity": "HIGH",
                "action": f"MC: {mc.prob_loss:.0f}% dos paths com perda",
                "rationale": "Risco significativo de perda mesmo com bootstrap."})
        if mc.obs_return_vs_p50 < -20:
            recs.append({"id": "R10b", "area": "Validacao", "severity": "HIGH",
                "action": f"Retorno observado {mc.obs_return_vs_p50:.1f}pp abaixo da mediana MC",
                "rationale": "O backtest superestima significativamente o retorno esperado."})

    # R11: Overfitting
    if overfitting:
        if overfitting.level == "CRITICAL":
            recs.append({"id": "R11", "area": "Overfitting", "severity": "CRITICAL",
                "action": f"Risco de overfitting CRITICO (score={overfitting.score:.0f})",
                "rationale": overfitting.details})
        elif overfitting.level == "HIGH":
            recs.append({"id": "R11", "area": "Overfitting", "severity": "HIGH",
                "action": f"Risco de overfitting ALTO (score={overfitting.score:.0f})",
                "rationale": overfitting.details})
        elif overfitting.level == "MEDIUM":
            recs.append({"id": "R11", "area": "Overfitting", "severity": "MEDIUM",
                "action": f"Risco de overfitting MODERADO (score={overfitting.score:.0f})",
                "rationale": overfitting.details})

    # R12: Temporal Stability
    if temporal and temporal.n_windows >= 3:
        if temporal.trend == "DEGRADING":
            recs.append({"id": "R12", "area": "Validacao", "severity": "HIGH",
                "action": "Performance degradando ao longo do tempo — investigar regime shift",
                "rationale": f"Consistencia={temporal.consistency_score:.0f}%, tendencia=DEGRADING."})
        elif temporal.consistency_score < 30:
            recs.append({"id": "R12", "area": "Validacao", "severity": "MEDIUM",
                "action": f"Consistencia temporal baixa ({temporal.consistency_score:.0f}%)",
                "rationale": "Alta variabilidade entre janelas temporais."})

    # R13: OOS Validation
    if not has_oos:
        recs.append({"id": "R13", "area": "Validacao", "severity": "HIGH",
            "action": "Realizar validacao Out-of-Sample (Walk-Forward)",
            "rationale": "Sem validacao OOS, nao e possivel confirmar que a edge persiste fora da amostra."})

    # R14: Costs
    recs.append({"id": "R14", "area": "Custos", "severity": "INFO",
        "action": "Verificar sensibilidade a custos (fees + spread + slippage)",
        "rationale": "Custos podem consumir 30-50% da edge em live. Testar com custos 2x e 5x."})

    # R15: Regime dependency
    if regime and regime.regimes:
        regime_contrib = {k: v["total_pnl_pct"] for k, v in regime.regimes.items()}
        total = sum(abs(v) for v in regime_contrib.values())
        if total > 0:
            for r_name, r_pnl in sorted(regime_contrib.items(), key=lambda x: abs(x[1]), reverse=True):
                share = abs(r_pnl) / total * 100
                if share > 80:
                    recs.append({"id": "R15", "area": "Robustez", "severity": "HIGH",
                        "action": f"Dependencia excessiva do regime '{r_name}' ({share:.0f}% do PnL)",
                        "rationale": "Estrategia muito concentrada em um regime. Diversificar logica."})
                    break

    # R16: Position Sizing
    recs.append({"id": "R16", "area": "Risco", "severity": "INFO",
        "action": "Revisar alocacao de risco por estrategia",
        "rationale": "Squeeze Breakout recebe 8% enquanto CTEV recebe 0.05%. Verificar concentracao."})

    # R17: Concurrency
    recs.append({"id": "R17", "area": "Risco", "severity": "INFO",
        "action": "Max 3 posicoes concurrentes — monitorar correlacao entre positions",
        "rationale": "Posicoes correlacionadas amplificam drawdowns. Correlation guard esta desativado."})

    # R18: Cooldown
    recs.append({"id": "R18", "area": "Risco", "severity": "INFO",
        "action": "Cooldown de 3 bars apos 2 SLs — avaliar eficacia",
        "rationale": "Cooldown curto pode nao proteger contra sequencias rapidas de perda."})

    # R19: Next steps
    recs.append({"id": "R19", "area": "Proximos Passos", "severity": "INFO",
        "action": "Executar: (1) Walk-Forward OOS, (2) Cost sensitivity 2x/5x, (3) Degradation test",
        "rationale": "Validacao completa requer OOS, teste de custos adversos e teste de degradacao temporal."})

    # Sort by severity
    sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
    recs.sort(key=lambda r: sev_order.get(r["severity"], 5))
    return recs


# ======================================================================
# 13. PARAMETER SENSITIVITY ANALYSIS
# ======================================================================

@dataclass
class SensitivityResult:
    """Resultado da analise de sensibilidade de um parametro."""
    param_name: str
    base_value: float
    tested_values: List[float]
    impact_on_pf: List[float]      # PF para cada valor testado
    impact_on_dd: List[float]      # Max DD para cada valor testado
    impact_on_return: List[float]  # Total return para cada valor testado
    sensitivity_score: float       # 0-100 (100 = muito sensivel = ruim)
    robust_range: Tuple[float, float]  # (min, max) range onde PF > 1.0
    recommendation: str


@dataclass
class SensitivityHeatMap:
    """Mapa de calor de sensibilidade para multiplos parametros."""
    parameters: List[SensitivityResult]
    overall_robustness_score: float  # 0-100 (100 = muito robusto)
    critical_parameters: List[str]  # Params com sensitivity > 70
    stable_parameters: List[str]     # Params com sensitivity < 30


def analyze_parameter_sensitivity(
    param_name: str,
    base_value: float,
    test_range: List[float],
    simulate_func: callable,
    metric_key: str = "profit_factor",
) -> SensitivityResult:
    """
    Analisa sensibilidade de um parametro individual.

    FORMULA:
      sensitivity_score = (max_impact - min_impact) / base_metric * 100
        onde impact = |metric(value) - metric(base)| / metric(base)
      robust_range = [min_value, max_value] onde PF > 1.0
    """
    results_pf = []
    results_dd = []
    results_return = []

    for val in test_range:
        try:
            m = simulate_func(val)
            results_pf.append(m.get(metric_key, 0))
            results_dd.append(m.get("max_drawdown_pct", 100))
            results_return.append(m.get("total_pnl_pct", 0))
        except Exception:
            results_pf.append(0.0)
            results_dd.append(100.0)
            results_return.append(0.0)

    # Base metric (usar o valor medio do range como referencia)
    base_idx = len(test_range) // 2
    base_metric = results_pf[base_idx] if results_pf[base_idx] > 0 else 1.0

    # Sensitivity: quanto o PF varia com mudancas no parametro
    deviations = []
    for pf in results_pf:
        if base_metric > 0:
            dev = abs(pf - results_pf[base_idx]) / base_metric * 100
            deviations.append(dev)

    sensitivity = float(np.mean(deviations)) if deviations else 0

    # Robust range: valores onde PF > 1.0
    robust_vals = [test_range[i] for i, pf in enumerate(results_pf) if pf > 1.0]
    if robust_vals:
        robust_range = (min(robust_vals), max(robust_vals))
    else:
        robust_range = (base_value, base_value)

    # Recommendation
    if sensitivity < 20:
        recommendation = f"{param_name} e ROBUSTO — variacoes ate {max(test_range)-min(test_range):.2f} nao degradam PF"
    elif sensitivity < 50:
        recommendation = f"{param_name} e MODERADO — evitar valores extremos fora do range robusto"
    else:
        recommendation = f"{param_name} e SENSIVEL — pequenas mudancas degradam performance. Risco de overfitting."

    return SensitivityResult(
        param_name=param_name,
        base_value=base_value,
        tested_values=test_range,
        impact_on_pf=results_pf,
        impact_on_dd=results_dd,
        impact_on_return=results_return,
        sensitivity_score=round(sensitivity, 1),
        robust_range=robust_range,
        recommendation=recommendation,
    )


def build_sensitivity_heatmap(
    results: List[SensitivityResult],
) -> SensitivityHeatMap:
    """
    Constroi mapa de calor de sensibilidade agregado.

    FORMULA:
      overall_robustness = 100 - mean(sensitivity_scores)
      critical: sensitivity > 70
      stable: sensitivity < 30
    """
    if not results:
        return SensitivityHeatMap(
            parameters=[], overall_robustness_score=0,
            critical_parameters=[], stable_parameters=[],
        )

    scores = [r.sensitivity_score for r in results]
    overall = max(0, 100 - float(np.mean(scores)))

    critical = [r.param_name for r in results if r.sensitivity_score > 70]
    stable = [r.param_name for r in results if r.sensitivity_score < 30]

    return SensitivityHeatMap(
        parameters=results,
        overall_robustness_score=round(overall, 1),
        critical_parameters=critical,
        stable_parameters=stable,
    )


# ======================================================================
# 14. COST SENSITIVITY ANALYSIS (4-Layer Cost Tiers)
# ======================================================================

@dataclass
class CostTierResult:
    """Resultado de teste com camada de custo especifica."""
    tier_name: str
    fee_pct: float
    spread_bps: float
    slippage_bps: float
    total_cost_per_trade_pct: float
    profit_factor: float
    win_rate: float
    total_pnl_pct: float
    max_drawdown_pct: float
    pf_degradation_vs_base: float  # pp degradacao vs base
    return_degradation_vs_base: float


@dataclass
class CostSensitivityResult:
    """Analise completa de sensibilidade a custos."""
    tiers: List[CostTierResult]
    base_pf: float
    base_return: float
    max_acceptable_cost_pct: float  # custo maximo antes de PF < 1.0
    cost_margin_pct: float  # margem de seguranca: (max_cost - actual_cost) / actual_cost
    is_cost_resilient: bool  # True se PF > 1.0 em todos os tiers
    recommendation: str


# 4-Layer Cost Tiers (especificacao 40-partes)
COST_TIERS = {
    "TIER_0_IDEAL": {
        "fee_pct": 0.016, "spread_bps": 2.0, "slippage_bps": 2.0,
        "description": "Maker order, exchange de baixo custo, mercado liquido",
    },
    "TIER_1_REALISTIC": {
        "fee_pct": 0.040, "spread_bps": 5.0, "slippage_bps": 5.0,
        "description": "Taker order, custo medio de mercado",
    },
    "TIER_2_ADVERSE": {
        "fee_pct": 0.060, "spread_bps": 10.0, "slippage_bps": 10.0,
        "description": "Taker order + mercado estressado + slippage elevado",
    },
    "TIER_3_EXTREME": {
        "fee_pct": 0.100, "spread_bps": 20.0, "slippage_bps": 20.0,
        "description": "Pior cenario: alta volatilidade, baixa liquidez, taker",
    },
}


def compute_cost_per_trade(fee_pct: float, spread_bps: float, slippage_bps: float) -> float:
    """Custo total por trade (round-trip) em %.

    FORMULA:
      cost = (fee_pct * 2) + (spread_bps / 10000 * 2) + (slippage_bps / 10000 * 2)
      Cada trade tem entrada + saida, logo multiplica por 2.
    """
    return round(fee_pct * 200 + spread_bps * 2 / 100 + slippage_bps * 2 / 100, 6)


def run_cost_sensitivity(
    trades: List[Dict[str, Any]],
    base_metrics: Dict[str, Any],
    simulate_with_costs: Optional[callable] = None,
) -> CostSensitivityResult:
    """
    Testa a estrategia em 4 camadas de custo.

    Se simulate_with_costs fornece uma funcao que roda backtest com custos
    customizados, usa-a. Caso contrario, estima o impacto pela subtracao
    linear dos custos adicionais do PnL de cada trade.

    FORMULA:
      estimated_pnl(i) = original_pnl(i) - (extra_cost_per_trade * position_exposure)
      degradation = base_metric - estimated_metric
    """
    base_pf = base_metrics.get("profit_factor", 0)
    base_return = base_metrics.get("total_pnl_pct", 0)
    base_fee = base_metrics.get("_fee_pct", 0.016)
    base_spread = base_metrics.get("_spread_bps", 2.0)
    base_slip = base_metrics.get("_slippage_bps", 2.0)
    base_cost = compute_cost_per_trade(base_fee, base_spread, base_slip)

    pnls = [t.get("pnl_pct", 0) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    tier_results = []

    for tier_name, tier_cfg in COST_TIERS.items():
        t_fee = tier_cfg["fee_pct"]
        t_spread = tier_cfg["spread_bps"]
        t_slip = tier_cfg["slippage_bps"]
        t_cost = compute_cost_per_trade(t_fee, t_spread, t_slip)

        extra_cost = t_cost - base_cost  # custo adicional por trade (round-trip %)

        # Estimar PnL ajustado: subtrair custo extra de cada trade
        adjusted_pnls = [p - extra_cost for p in pnls]
        adj_wins = [p for p in adjusted_pnls if p > 0]
        adj_losses = [p for p in adjusted_pnls if p <= 0]

        # Estimated metrics
        adj_gross_profit = sum(adj_wins) if adj_wins else 0
        adj_gross_loss = abs(sum(adj_losses)) if adj_losses else 0.001
        adj_pf = adj_gross_profit / adj_gross_loss if adj_gross_loss > 0 else 0
        adj_wr = len(adj_wins) / len(adjusted_pnls) * 100 if adjusted_pnls else 0

        # Estimate equity curve for return and DD
        eq = [10000.0]
        for p in adjusted_pnls:
            eq.append(eq[-1] * (1 + p / 100))
        eq_arr = np.array(eq)
        adj_return = float((eq_arr[-1] / eq_arr[0] - 1) * 100)
        peak = np.maximum.accumulate(eq_arr)
        dd_series = (peak - eq_arr) / peak * 100
        adj_dd = float(np.max(dd_series)) if len(dd_series) > 0 else 0

        pf_deg = base_pf - adj_pf
        ret_deg = base_return - adj_return

        tier_results.append(CostTierResult(
            tier_name=tier_name,
            fee_pct=t_fee,
            spread_bps=t_spread,
            slippage_bps=t_slip,
            total_cost_per_trade_pct=t_cost,
            profit_factor=round(adj_pf, 4),
            win_rate=round(adj_wr, 2),
            total_pnl_pct=round(adj_return, 4),
            max_drawdown_pct=round(adj_dd, 2),
            pf_degradation_vs_base=round(pf_deg, 4),
            return_degradation_vs_base=round(ret_deg, 4),
        ))

    # Max acceptable cost: maior custo onde PF > 1.0
    max_acceptable = 0
    for tr in tier_results:
        if tr.profit_factor > 1.0:
            max_acceptable = max(max_acceptable, tr.total_cost_per_trade_pct)

    cost_margin = (max_acceptable - base_cost) / base_cost * 100 if base_cost > 0 else 0
    is_resilient = all(tr.profit_factor > 1.0 for tr in tier_results)

    # Recommendation
    if is_resilient:
        rec = (f"Estrategia RESILIENTE a custos. PF > 1.0 em todos os 4 tiers. "
               f"Margem de seguranca: {cost_margin:.0f}%. Edge sobrevive mesmo em cenarios adversos.")
    elif tier_results[0].profit_factor > 1.0 and tier_results[1].profit_factor > 1.0:
        failed = [tr.tier_name for tr in tier_results if tr.profit_factor <= 1.0]
        rec = (f"Estrategia tolera custos realistas (Tier 0-1) mas falha em: {', '.join(failed)}. "
               f"Evitar taker orders em mercados illiquidos. Margem: {cost_margin:.0f}%.")
    else:
        rec = (f"ALERTA: estrategia CONSOMIDA por custos. PF < 1.0 ja em custos realistas. "
               f"Edge insuficiente para cobrir transacoes reais. Requer otimizacao de R:R ou filtros.")

    return CostSensitivityResult(
        tiers=tier_results,
        base_pf=base_pf,
        base_return=base_return,
        max_acceptable_cost_pct=round(max_acceptable, 6),
        cost_margin_pct=round(cost_margin, 1),
        is_cost_resilient=is_resilient,
        recommendation=rec,
    )


# ======================================================================
# 15. DEGRADATION TESTS (4-Level)
# ======================================================================

@dataclass
class DegradationTestResult:
    """Resultado de um teste de degradacao individual."""
    test_name: str
    description: str
    degradation_pct: float        # % de degradacao vs baseline
    pf_after: float
    return_after: float
    dd_after: float
    still_profitable: bool       # PF > 1.0
    verdict: str                 # PASS / MARGINAL / FAIL


@dataclass
class DegradationSuiteResult:
    """Suite completa de testes de degradacao."""
    tests: List[DegradationTestResult]
    n_pass: int
    n_marginal: int
    n_fail: int
    overall_verdict: str  # ROBUST / ACCEPTABLE / FRAGILE / BRITTLE
    worst_degradation_pct: float
    recommendation: str


def _apply_slippage_degradation(trades: List[Dict[str, Any]], extra_slip_bps: float) -> List[Dict[str, Any]]:
    """Aplica slippage extra a todos os trades.

    FORMULA:
      adjusted_exit = exit * (1 - sign * extra_slip_bps / 10000)
      Para LONG: adjusted_exit = exit * (1 - extra_slip / 10000)
      Para SHORT: adjusted_exit = exit * (1 + extra_slip / 10000)
    """
    degraded = []
    for t in trades:
        t_copy = dict(t)
        exit_p = t.get("exit_price", 0)
        entry_p = t.get("entry_price", 0)
        if entry_p > 0 and exit_p > 0:
            sign = 1.0 if t.get("type") == "LONG" else -1.0
            adj_exit = exit_p * (1 - sign * extra_slip_bps / 10000)
            # Recalcular PnL
            if t.get("type") == "LONG":
                new_pnl = (adj_exit - entry_p) / entry_p * 100
            else:
                new_pnl = (entry_p - adj_exit) / entry_p * 100
            t_copy["pnl_pct"] = round(new_pnl, 4)
            t_copy["exit_price"] = round(adj_exit, 2)
        degraded.append(t_copy)
    return degraded


def _apply_random_skip(trades: List[Dict[str, Any]], skip_pct: float, seed: int = 42) -> List[Dict[str, Any]]:
    """Remove trades aleatoriamente (simula sinais perdidos).

    FORMULA:
      n_skip = int(len(trades) * skip_pct)
      Remove n_skip trades aleatorios
    """
    if skip_pct <= 0 or len(trades) <= 1:
        return list(trades)
    rng = np.random.default_rng(seed)
    n_skip = int(len(trades) * skip_pct)
    if n_skip >= len(trades):
        return []
    keep_indices = sorted(rng.choice(len(trades), size=len(trades) - n_skip, replace=False))
    return [trades[i] for i in keep_indices]


def _apply_worst_period_isolation(trades: List[Dict[str, Any]], pct: float = 0.25) -> List[Dict[str, Any]]:
    """Isola o pior periodo de X% dos trades.

    FORMULA:
      worst_n = int(len(trades) * pct)
      Retorna apenas os pior_n trades por PnL
    """
    if not trades:
        return []
    sorted_trades = sorted(trades, key=lambda t: t.get("pnl_pct", 0))
    worst_n = max(1, int(len(trades) * pct))
    return sorted_trades[:worst_n]


def _compute_quick_metrics(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    """Metricas rapidas para teste de degradacao."""
    if not trades:
        return {"profit_factor": 0, "total_pnl_pct": 0, "max_drawdown_pct": 0, "total_trades": 0}
    pnls = [t.get("pnl_pct", 0) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    pf = sum(wins) / max(abs(sum(losses)), 0.001)
    eq = [10000.0]
    for p in pnls:
        eq.append(eq[-1] * (1 + p / 100))
    eq_arr = np.array(eq)
    ret = float((eq_arr[-1] / eq_arr[0] - 1) * 100)
    peak = np.maximum.accumulate(eq_arr)
    dd = float(np.max((peak - eq_arr) / peak * 100)) if len(eq_arr) > 1 else 0
    return {"profit_factor": round(pf, 4), "total_pnl_pct": round(ret, 4),
            "max_drawdown_pct": round(dd, 2), "total_trades": len(trades)}


def run_degradation_suite(
    trades: List[Dict[str, Any]],
    base_metrics: Dict[str, Any],
) -> DegradationSuiteResult:
    """
    Executa 4 testes de degradacao (especificacao 40-partes):

    Level 1: Slippage adverso (+10bps extra por trade)
    Level 2: Slippage extremo (+25bps extra por trade)
    Level 3: Sinais perdidos (20% dos trades removidos aleatoriamente)
    Level 4: Pior quartil isolado (25% piores trades)

    FORMULA:
      degradation_pct = (base_metric - degraded_metric) / base_metric * 100
      Verdict: PASS (<20% deg), MARGINAL (20-50%), FAIL (>50%)
    """
    base_pf = base_metrics.get("profit_factor", 0)
    base_return = base_metrics.get("total_pnl_pct", 0)
    base_dd = base_metrics.get("max_drawdown_pct", 0)

    tests = []

    # Test 1: Slippage adverso (+10bps)
    degraded_1 = _apply_slippage_degradation(trades, 10.0)
    m1 = _compute_quick_metrics(degraded_1)
    deg_1 = (base_pf - m1["profit_factor"]) / max(base_pf, 0.01) * 100
    v1 = "PASS" if deg_1 < 20 else ("MARGINAL" if deg_1 < 50 else "FAIL")
    tests.append(DegradationTestResult(
        test_name="LEVEL_1_SLIPPAGE_ADVERSE",
        description="Slippage extra +10bps por trade (round-trip)",
        degradation_pct=round(deg_1, 1),
        pf_after=m1["profit_factor"],
        return_after=m1["total_pnl_pct"],
        dd_after=m1["max_drawdown_pct"],
        still_profitable=m1["profit_factor"] > 1.0,
        verdict=v1,
    ))

    # Test 2: Slippage extremo (+25bps)
    degraded_2 = _apply_slippage_degradation(trades, 25.0)
    m2 = _compute_quick_metrics(degraded_2)
    deg_2 = (base_pf - m2["profit_factor"]) / max(base_pf, 0.01) * 100
    v2 = "PASS" if deg_2 < 20 else ("MARGINAL" if deg_2 < 50 else "FAIL")
    tests.append(DegradationTestResult(
        test_name="LEVEL_2_SLIPPAGE_EXTREME",
        description="Slippage extra +25bps por trade (round-trip)",
        degradation_pct=round(deg_2, 1),
        pf_after=m2["profit_factor"],
        return_after=m2["total_pnl_pct"],
        dd_after=m2["max_drawdown_pct"],
        still_profitable=m2["profit_factor"] > 1.0,
        verdict=v2,
    ))

    # Test 3: 20% sinais perdidos
    degraded_3 = _apply_random_skip(trades, 0.20)
    m3 = _compute_quick_metrics(degraded_3)
    # Para sinal perdido, medimos degradacao de retorno (PF pode ser instavel com menos trades)
    deg_3 = abs(base_return - m3["total_pnl_pct"]) / max(abs(base_return), 0.01) * 100
    v3 = "PASS" if deg_3 < 30 else ("MARGINAL" if deg_3 < 60 else "FAIL")
    tests.append(DegradationTestResult(
        test_name="LEVEL_3_SIGNAL_LOSS_20PCT",
        description="20% dos sinais perdidos (executor falha, latencia, etc.)",
        degradation_pct=round(deg_3, 1),
        pf_after=m3["profit_factor"],
        return_after=m3["total_pnl_pct"],
        dd_after=m3["max_drawdown_pct"],
        still_profitable=m3["profit_factor"] > 1.0,
        verdict=v3,
    ))

    # Test 4: Pior quartil isolado
    degraded_4 = _apply_worst_period_isolation(trades, 0.25)
    m4 = _compute_quick_metrics(degraded_4)
    # Pior quartil: esperamos perda. Verificamos se PF ainda > 0.5 (nao desmorona)
    worst_pf = m4["profit_factor"]
    # Degradation: PF caiu de base para worst_quartil
    deg_4 = (base_pf - worst_pf) / max(base_pf, 0.01) * 100
    v4 = "PASS" if worst_pf > 0.7 else ("MARGINAL" if worst_pf > 0.4 else "FAIL")
    tests.append(DegradationTestResult(
        test_name="LEVEL_4_WORST_QUARTILE",
        description="Pior quartil (25% piores trades) isolado",
        degradation_pct=round(min(deg_4, 100), 1),
        pf_after=worst_pf,
        return_after=m4["total_pnl_pct"],
        dd_after=m4["max_drawdown_pct"],
        still_profitable=worst_pf > 0.5,
        verdict=v4,
    ))

    # Aggregate
    n_pass = sum(1 for t in tests if t.verdict == "PASS")
    n_marginal = sum(1 for t in tests if t.verdict == "MARGINAL")
    n_fail = sum(1 for t in tests if t.verdict == "FAIL")
    worst_deg = max(t.degradation_pct for t in tests)

    if n_fail == 0 and n_marginal <= 1:
        overall = "ROBUST"
        rec = "Estrategia ROBUSTA: tolera perturbacoes sem perder edge. Pronta para validacao OOS."
    elif n_fail == 0:
        overall = "ACCEPTABLE"
        rec = "Estrategia ACEITAVEL: tolera perturbacoes com alguma degradacao. Monitorar em live."
    elif n_fail <= 2:
        overall = "FRAGILE"
        rec = "Estrategia FRAGIL: falha em cenarios adversos. Requer melhoria de robustez antes de live."
    else:
        overall = "BRITTLE"
        rec = "Estrategia FRAGIL/QUEBRADICA: falha na maioria dos cenarios. Nao recomendada para capital real."

    return DegradationSuiteResult(
        tests=tests,
        n_pass=n_pass,
        n_marginal=n_marginal,
        n_fail=n_fail,
        overall_verdict=overall,
        worst_degradation_pct=round(worst_deg, 1),
        recommendation=rec,
    )


# ======================================================================
# 16. WALK-FORWARD AUDIT INTEGRATION
# ======================================================================

@dataclass
class WalkForwardAuditResult:
    """Resultado auditado do Walk-Forward Analysis."""
    n_windows: int
    train_avg_pf: float
    test_avg_pf: float
    train_avg_return: float
    test_avg_return: float
    avg_degradation_pct: float   # media da degradacao train->test
    max_degradation_pct: float
    min_degradation_pct: float
    degradation_std: float
    oos_win_rate: float          # % de janelas com PF_test > 1.0
    consistency_score: float     # 0-100
    has_oos_edge: bool          # True se test_avg_pf > 1.0
    verdict: str                # VALIDATED / PARTIAL / FAILED


def audit_walk_forward_results(
    wf_results: list,
) -> WalkForwardAuditResult:
    """
    Audita resultados do Walk-Forward Analysis.

    Recebe lista de WalkForwardResult do backtest.py e calcula:
    - Degradação media train -> test
    - Consistencia (quantas janelas OOS são lucrativas)
    - Veredito de validação

    FORMULA:
      degradation_i = (train_metric_i - test_metric_i) / train_metric_i * 100
      consistency_score = (n_profitable_windows / n_total_windows) * 100
    """
    if not wf_results:
        return WalkForwardAuditResult(
            n_windows=0, train_avg_pf=0, test_avg_pf=0,
            train_avg_return=0, test_avg_return=0,
            avg_degradation_pct=0, max_degradation_pct=0,
            min_degradation_pct=0, degradation_std=0,
            oos_win_rate=0, consistency_score=0,
            has_oos_edge=False, verdict="FAILED",
        )

    train_pfs = []
    test_pfs = []
    train_rets = []
    test_rets = []
    degradations = []
    profitable_oos = 0

    for wf in wf_results:
        t_pf = wf.train_metrics.profit_factor
        e_pf = wf.test_metrics.profit_factor
        t_ret = wf.train_metrics.total_pnl_pct
        e_ret = wf.test_metrics.total_pnl_pct

        train_pfs.append(t_pf)
        test_pfs.append(e_pf)
        train_rets.append(t_ret)
        test_rets.append(e_ret)

        if t_pf > 0:
            deg = (t_pf - e_pf) / t_pf * 100
            degradations.append(deg)

        if e_pf > 1.0:
            profitable_oos += 1

    n = len(wf_results)
    avg_deg = float(np.mean(degradations)) if degradations else 0
    oos_wr = profitable_oos / n * 100
    consistency = oos_wr  # Simplicidade: % de janelas lucrativas

    # Verdict
    test_avg_pf_val = float(np.mean(test_pfs)) if test_pfs else 0
    if test_avg_pf_val > 1.2 and oos_wr >= 70:
        verdict = "VALIDATED"
    elif test_avg_pf_val > 1.0 and oos_wr >= 50:
        verdict = "PARTIAL"
    else:
        verdict = "FAILED"

    return WalkForwardAuditResult(
        n_windows=n,
        train_avg_pf=round(float(np.mean(train_pfs)), 4) if train_pfs else 0,
        test_avg_pf=round(test_avg_pf_val, 4),
        train_avg_return=round(float(np.mean(train_rets)), 4) if train_rets else 0,
        test_avg_return=round(float(np.mean(test_rets)), 4) if test_rets else 0,
        avg_degradation_pct=round(avg_deg, 1),
        max_degradation_pct=round(max(degradations) if degradations else 0, 1),
        min_degradation_pct=round(min(degradations) if degradations else 0, 1),
        degradation_std=round(float(np.std(degradations)) if degradations else 0, 1),
        oos_win_rate=round(oos_wr, 1),
        consistency_score=round(consistency, 1),
        has_oos_edge=test_avg_pf_val > 1.0,
        verdict=verdict,
    )


# ======================================================================
# 17. VERSION ATTRACTIVENESS SCORE (V2-V5 Selection)
# ======================================================================

@dataclass
class VersionCandidate:
    """Candidato de versao para comparacao."""
    version_label: str
    metrics: Dict[str, Any]
    trades: List[Dict[str, Any]]
    mc: Optional[MonteCarloResult] = None
    overfitting: Optional[OverfittingScore] = None
    temporal: Optional[TemporalStabilityResult] = None
    degradation: Optional[DegradationSuiteResult] = None
    cost_sensitivity: Optional[CostSensitivityResult] = None
    wf_audit: Optional[WalkForwardAuditResult] = None


@dataclass
class AttractivenessScore:
    """Score de atratividade para selecao de versao.

    Principio: AUDITABILIDADE > ROBUSTEZ > RISCO > CONSISTENCIA > RETORNO
    Pesos refletem esta hierarquia.
    """
    version_label: str
    auditability_score: float   # 0-100 (peso 25%)
    robustness_score: float     # 0-100 (peso 25%)
    risk_score: float          # 0-100 (peso 20%)
    consistency_score: float   # 0-100 (peso 15%)
    return_score: float        # 0-100 (peso 15%)
    composite_score: float     # 0-100
    rank: int                  # 1 = melhor
    recommendation: str        # DEPLOY / IMPROVE / HOLD / REJECT


def compute_attractiveness_score(
    candidate: VersionCandidate,
    all_candidates: Optional[List[VersionCandidate]] = None,
) -> AttractivenessScore:
    """
    Calcula score de atratividade para uma versao candidata.

    Principio de ordenacao:
      AUDITABILIDADE (25%) > ROBUSTEZ (25%) > RISCO (20%) > CONSISTENCIA (15%) > RETORNO (15%)

    AUDITABILIDADE (0-100):
      - Amostra (trades): 0-30 pontos
      - Walk-Forward OOS: 0-30 pontos
      - Monte Carlo: 0-20 pontos
      - Sensibilidade/Overfitting: 0-20 pontos

    ROBUSTEZ (0-100):
      - Multi-obj composite: 0-30 pontos
      - Degradation suite: 0-30 pontos
      - Cost sensitivity: 0-20 pontos
      - Parameter sensitivity: 0-20 pontos

    RISCO (0-100):
      - Max DD: 0-25 pontos (DD 10%=25, DD 100%=0)
      - Recovery Factor: 0-25 pontos
      - Expected Shortfall: 0-25 pontos
      - VaR 95: 0-25 pontos

    CONSISTENCIA (0-100):
      - Temporal stability: 0-40 pontos
      - Regime balance: 0-30 pontos
      - LONG/SHORT balance: 0-30 pontos

    RETORNO (0-100):
      - Profit Factor: 0-30 pontos
      - CAGR: 0-30 pontos
      - Omega: 0-20 pontos
      - Calmar: 0-20 pontos
    """
    m = candidate.metrics
    trades = candidate.trades

    # === AUDITABILIDADE (0-100) ===
    n_trades = m.get("total_trades", 0)
    if n_trades >= 100:
        sample_pts = 30
    elif n_trades >= 50:
        sample_pts = 20
    elif n_trades >= 30:
        sample_pts = 12
    else:
        sample_pts = max(0, n_trades / 30 * 12)

    if candidate.wf_audit and candidate.wf_audit.n_windows > 0:
        if candidate.wf_audit.verdict == "VALIDATED":
            wfo_pts = 30
        elif candidate.wf_audit.verdict == "PARTIAL":
            wfo_pts = 18
        else:
            wfo_pts = 5
    else:
        wfo_pts = 0

    if candidate.mc and candidate.mc.n_simulations > 0:
        mc_pts = min(20, candidate.mc.prob_profitable / 100 * 20)
    else:
        mc_pts = 0

    if candidate.overfitting:
        of_pts = max(0, 20 - candidate.overfitting.score / 5)
    else:
        of_pts = 10

    auditability = sample_pts + wfo_pts + mc_pts + of_pts

    # === ROBUSTEZ (0-100) ===
    pf = m.get("profit_factor", 0)
    sharpe = m.get("sharpe_ratio", 0)
    sortino = m.get("sortino_ratio", 0)
    omega = m.get("omega_ratio", 0)

    mo_composite = min(30, max(0, sharpe / 2.0 * 15 + sortino / 2.0 * 15))

    if candidate.degradation:
        n_p = candidate.degradation.n_pass
        n_t = len(candidate.degradation.tests)
        deg_pts = n_p / max(n_t, 1) * 30
    else:
        deg_pts = 10

    if candidate.cost_sensitivity:
        if candidate.cost_sensitivity.is_cost_resilient:
            cost_pts = 20
        elif len(candidate.cost_sensitivity.tiers) > 1 and candidate.cost_sensitivity.tiers[1].profit_factor > 1.0:
            cost_pts = 12
        else:
            cost_pts = 4
    else:
        cost_pts = 8

    # Parameter sensitivity (usar overfitting como proxy se nao disponivel)
    if candidate.overfitting:
        param_pts = max(0, 20 - candidate.overfitting.score / 5)
    else:
        param_pts = 10

    robustness = mo_composite + deg_pts + cost_pts + param_pts

    # === RISCO (0-100) ===
    dd = m.get("max_drawdown_pct", 100)
    dd_pts = max(0, 25 - dd * 0.25)

    recovery = m.get("recovery_factor", 0)
    rec_pts = min(25, max(0, recovery / 5.0 * 25))

    es = m.get("expected_shortfall", 0)
    es_pts = max(0, 25 - abs(es) * 5)

    var95 = m.get("var_95", 0)
    var_pts = max(0, 25 - abs(var95) * 5)

    risk = dd_pts + rec_pts + es_pts + var_pts

    # === CONSISTENCIA (0-100) ===
    if candidate.temporal and candidate.temporal.n_windows >= 3:
        temp_pts = candidate.temporal.consistency_score * 0.4
    else:
        temp_pts = 10

    # Regime balance (se temos trades com regime)
    regime_trades = [t for t in trades if t.get("regime_at_entry", "")]
    if len(regime_trades) > 5:
        regime_groups = {}
        for t in regime_trades:
            r = t.get("regime_at_entry", "unknown")
            regime_groups.setdefault(r, []).append(t)
        regime_pnls = {k: sum(t.get("pnl_pct", 0) for t in v) for k, v in regime_groups.items()}
        pos_regimes = sum(1 for v in regime_pnls.values() if v > 0)
        regime_balance = pos_regimes / max(len(regime_groups), 1) * 30
    else:
        regime_balance = 10

    # LONG/SHORT balance
    longs = [t for t in trades if t.get("type") == "LONG"]
    shorts = [t for t in trades if t.get("type") == "SHORT"]
    if longs and shorts:
        l_pnl = sum(t.get("pnl_pct", 0) for t in longs)
        s_pnl = sum(t.get("pnl_pct", 0) for t in shorts)
        if abs(l_pnl) + abs(s_pnl) > 0:
            ls_balance = (1 - abs(l_pnl - s_pnl) / (abs(l_pnl) + abs(s_pnl))) * 30
        else:
            ls_balance = 15
    else:
        ls_balance = 5  # Penalidade para unidirecional

    consistency = temp_pts + regime_balance + ls_balance

    # === RETORNO (0-100) ===
    pf_pts = min(30, max(0, (pf - 0.8) / 1.2 * 30))
    cagr = m.get("cagr", 0)
    cagr_pts = min(30, max(0, cagr / 200 * 30))  # 200% CAGR = max
    omega_pts = min(20, max(0, (omega - 0.8) / 1.2 * 20))
    calmar = m.get("calmar_ratio", 0)
    calmar_pts = min(20, max(0, calmar / 3.0 * 20))
    return_score = pf_pts + cagr_pts + omega_pts + calmar_pts

    # === COMPOSITE ===
    composite = (
        auditability * 0.25 +
        robustness * 0.25 +
        risk * 0.20 +
        consistency * 0.15 +
        return_score * 0.15
    )

    # Recommendation
    if composite >= 70 and auditability >= 50 and robustness >= 40:
        rec = "DEPLOY"
    elif composite >= 50 and risk >= 30:
        rec = "IMPROVE"
    elif composite >= 30:
        rec = "HOLD"
    else:
        rec = "REJECT"

    return AttractivenessScore(
        version_label=candidate.version_label,
        auditability_score=round(auditability, 1),
        robustness_score=round(robustness, 1),
        risk_score=round(risk, 1),
        consistency_score=round(consistency, 1),
        return_score=round(return_score, 1),
        composite_score=round(composite, 1),
        rank=0,  # Filled externally after ranking all candidates
        recommendation=rec,
    )


def rank_versions(
    candidates: List[VersionCandidate],
) -> List[AttractivenessScore]:
    """
    Rankeia multiplas versoes por score de atratividade.

    FORMULA:
      rank_i = posicao ordenada por composite_score (desc)
    """
    scores = [compute_attractiveness_score(c, candidates) for c in candidates]
    scores.sort(key=lambda s: s.composite_score, reverse=True)
    ranked = []
    for i, s in enumerate(scores):
        ranked.append(AttractivenessScore(
            version_label=s.version_label,
            auditability_score=s.auditability_score,
            robustness_score=s.robustness_score,
            risk_score=s.risk_score,
            consistency_score=s.consistency_score,
            return_score=s.return_score,
            composite_score=s.composite_score,
            rank=i + 1,
            recommendation=s.recommendation,
        ))
    return ranked


# ======================================================================
# 18. REGIME CLASSIFICATION (7 Regimes)
# ======================================================================

@dataclass
class RegimeClassificationResult:
    """Classificacao detalhada de regimes com performance."""
    n_regimes_observed: int
    regimes: Dict[str, Dict[str, Any]]
    dominant_regime: str
    regime_diversification_score: float  # 0-100
    regime_risk_concentration: float     # 0-100 (100 = toda edge em 1 regime)


def classify_regime_performance(
    trades: List[Dict[str, Any]],
) -> RegimeClassificationResult:
    """
    Classifica e analise performance por regime de mercado.

    7 regimes esperados: trending_up, trending_down, transition,
    ranging, volatile, squeeze, unknown.

    FORMULA:
      regime_contribution = sum(PnL_regime) / sum(|PnL_all|)
      diversification = 1 - HHI(contribuicoes)
      concentration = HHI * 100
    """
    EXPECTED_REGIMES = [
        "trending_up", "trending_down", "transition",
        "ranging", "volatile", "squeeze", "unknown",
    ]

    if not trades:
        return RegimeClassificationResult(
            n_regimes_observed=0, regimes={},
            dominant_regime="N/A", regime_diversification_score=0,
            regime_risk_concentration=100,
        )

    regime_groups: Dict[str, List[Dict]] = {}
    for t in trades:
        r = t.get("regime_at_entry", "unknown") or "unknown"
        regime_groups.setdefault(r, []).append(t)

    # Calcular metricas por regime
    regimes = {}
    total_abs_pnl = 0
    for name in EXPECTED_REGIMES:
        group = regime_groups.get(name, [])
        if not group:
            continue

        pnls = [t.get("pnl_pct", 0) for t in group]
        wins = [p for p in pnls if p > 0]
        n = len(group)
        wr = len(wins) / n * 100 if n > 0 else 0
        total = sum(pnls)
        avg = np.mean(pnls) if pnls else 0
        gross_w = sum(wins) if wins else 0
        gross_l = abs(sum(p for p in pnls if p <= 0))
        pf = gross_w / gross_l if gross_l > 0 else 0
        total_abs_pnl += abs(total)

        r_mults = [t.get("r_multiple", 0) for t in group]
        avg_r = float(np.mean(r_mults)) if r_mults else 0

        # R:R e expectancy
        avg_win = float(np.mean(wins)) if wins else 0
        avg_loss = float(np.mean([p for p in pnls if p <= 0])) if any(p <= 0 for p in pnls) else 0
        exp_val = wr / 100 * avg_win + (1 - wr / 100) * avg_loss

        regimes[name] = {
            "n_trades": n,
            "share_pct": round(n / len(trades) * 100, 1),
            "win_rate": round(wr, 2),
            "total_pnl_pct": round(total, 4),
            "avg_pnl_pct": round(avg, 4),
            "profit_factor": round(pf, 4),
            "avg_r_multiple": round(avg_r, 2),
            "expectancy": round(exp_val, 4),
            "best_trade": round(max(pnls), 4) if pnls else 0,
            "worst_trade": round(min(pnls), 4) if pnls else 0,
            "contribution_to_total_pct": 0,  # Preenchido abaixo
        }

    # Contribution percentages
    for name, data in regimes.items():
        if total_abs_pnl > 0:
            data["contribution_to_total_pct"] = round(
                abs(data["total_pnl_pct"]) / total_abs_pnl * 100, 1
            )
        else:
            data["contribution_to_total_pct"] = 0

    # Dominant regime
    if regimes:
        dominant = max(regimes.keys(), key=lambda k: abs(regimes[k]["total_pnl_pct"]))
    else:
        dominant = "N/A"

    # Diversification (HHI)
    contributions = [abs(d["total_pnl_pct"]) for d in regimes.values()]
    total_contrib = sum(contributions)
    if total_contrib > 0:
        shares = [c / total_contrib for c in contributions]
        hhi = sum(s ** 2 for s in shares)
        n_regimes = max(len(regimes), 1)
        div_score = max(0, min(100, (1 - hhi) / (1 - 1 / n_regimes) * 100)) if n_regimes > 1 else 0
        concentration = hhi * 100
    else:
        div_score = 0
        concentration = 100

    return RegimeClassificationResult(
        n_regimes_observed=len(regimes),
        regimes=regimes,
        dominant_regime=dominant,
        regime_diversification_score=round(div_score, 1),
        regime_risk_concentration=round(concentration, 1),
    )


# ======================================================================
# 19. CONCURRENT POSITION AUDIT
# ======================================================================

@dataclass
class ConcurrentPositionAudit:
    """Auditoria de posicoes concurrentes."""
    max_concurrent_seen: int
    avg_concurrent: float
    trades_with_concurrent: int
    pct_trades_with_concurrent: float
    pnl_by_concurrency: Dict[int, Dict[str, Any]]
    correlation_risk: str  # LOW / MEDIUM / HIGH


def audit_concurrent_positions(
    trades: List[Dict[str, Any]],
    max_allowed: int = 3,
) -> ConcurrentPositionAudit:
    """
    Audita impacto de posicoes concurrentes na performance.

    FORMULA:
      Para cada nivel de concorrencia N:
        WR_N = wins_N / total_N
        PnL_N = sum(pnl) para trades com N posicoes abertas
      correlation_risk: baseado na relacao entre concorrencia e drawdown
    """
    if not trades:
        return ConcurrentPositionAudit(
            max_concurrent_seen=0, avg_concurrent=0,
            trades_with_concurrent=0, pct_trades_with_concurrent=0,
            pnl_by_concurrency={}, correlation_risk="LOW",
        )

    concurrencies = [t.get("concurrent_count", 1) for t in trades]
    max_conc = max(concurrencies) if concurrencies else 0
    avg_conc = float(np.mean(concurrencies))
    with_conc = sum(1 for c in concurrencies if c > 1)
    pct_conc = with_conc / len(trades) * 100

    # Group by concurrency level
    conc_groups: Dict[int, List[Dict]] = {}
    for t in trades:
        c = t.get("concurrent_count", 1)
        conc_groups.setdefault(c, []).append(t)

    pnl_by_conc = {}
    for level, group in sorted(conc_groups.items()):
        pnls = [t.get("pnl_pct", 0) for t in group]
        wins = [p for p in pnls if p > 0]
        n = len(group)
        wr = len(wins) / n * 100 if n > 0 else 0
        total = sum(pnls)
        avg = np.mean(pnls) if pnls else 0

        # Drawdown within this group
        eq = [10000.0]
        for p in pnls:
            eq.append(eq[-1] * (1 + p / 100))
        eq_arr = np.array(eq)
        peak = np.maximum.accumulate(eq_arr)
        dd = float(np.max((peak - eq_arr) / peak * 100)) if len(eq_arr) > 1 else 0

        pnl_by_conc[level] = {
            "n_trades": n,
            "share_pct": round(n / len(trades) * 100, 1),
            "win_rate": round(wr, 2),
            "total_pnl_pct": round(total, 4),
            "avg_pnl_pct": round(avg, 4),
            "max_drawdown_pct": round(dd, 2),
        }

    # Correlation risk: se WR cai com mais posicoes, ha correlacao
    wrs_by_level = {k: v["win_rate"] for k, v in pnl_by_conc.items() if k > 1}
    if len(wrs_by_level) >= 2:
        wr_vals = list(wrs_by_level.values())
        if wr_vals[0] > wr_vals[-1] * 1.3:  # WR cai >30% com mais posicoes
            corr_risk = "HIGH"
        elif wr_vals[0] > wr_vals[-1] * 1.1:
            corr_risk = "MEDIUM"
        else:
            corr_risk = "LOW"
    else:
        corr_risk = "LOW"

    return ConcurrentPositionAudit(
        max_concurrent_seen=max_conc,
        avg_concurrent=round(avg_conc, 2),
        trades_with_concurrent=with_conc,
        pct_trades_with_concurrent=round(pct_conc, 1),
        pnl_by_concurrency=pnl_by_conc,
        correlation_risk=corr_risk,
    )


# ======================================================================
# 20. COMPREHENSIVE AUDIT ORCHESTRATOR
# ======================================================================

@dataclass
class ComprehensiveAuditResult:
    """Resultado completo da auditoria — todos os modulos."""
    # Core
    monte_carlo: MonteCarloResult
    strategy_decomposition: List[StrategyDecomposition]
    portfolio_contribution: Dict[str, Any]
    long_short: LongShortAnalysis
    drawdown_audit: DrawdownAuditResult
    multi_objective: MultiObjectiveScore
    verdict: str
    verdict_justification: str
    verdict_details: Dict[str, Any]
    overfitting: OverfittingScore
    outlier: OutlierResult
    regime: RegimeClassificationResult
    temporal: Optional[TemporalStabilityResult]
    equity_curve: List[Dict[str, Any]]
    recommendations: List[Dict[str, str]]
    # New modules (Part 13-19)
    cost_sensitivity: Optional[CostSensitivityResult]
    degradation: Optional[DegradationSuiteResult]
    wf_audit: Optional[WalkForwardAuditResult]
    concurrent_audit: ConcurrentPositionAudit
    # Metadata
    version_label: str
    n_analyses_run: int


def run_comprehensive_audit(
    metrics: Dict[str, Any],
    trades: List[Dict[str, Any]],
    version_label: str = "LIGA_CRYPTO",
    mc_simulations: int = 5000,
    has_oos: bool = False,
    wf_results: Optional[list] = None,
) -> ComprehensiveAuditResult:
    """
    Orquestra TODOS os modulos de auditoria em uma unica chamada.

    Executa na ordem:
    1. Monte Carlo
    2. Strategy Decomposition + Portfolio Contribution
    3. LONG vs SHORT
    4. Drawdown Audit
    5. Multi-Objective Score
    6. Overfitting Assessment
    7. Outlier Analysis
    8. Regime Classification
    9. Temporal Stability
    10. Cost Sensitivity
    11. Degradation Suite
    12. Walk-Forward Audit
    13. Concurrent Position Audit
    14. Full Equity Curve
    15. 19-Point Recommendations
    16. Verdict

    Returns ComprehensiveAuditResult com todos os resultados.
    """
    logger.info("Iniciando auditoria compreensiva para %s (%d trades)", version_label, len(trades))

    pnls = [t.get("pnl_pct", 0) for t in trades]

    # 1. Monte Carlo
    mc = run_monte_carlo(pnls, n_simulations=mc_simulations) if len(pnls) >= 5 else MonteCarloResult(
        n_simulations=0, n_trades=0, return_dist={}, dd_dist={},
        prob_loss=100, prob_profitable=0, obs_return_rank_pct=0, obs_return_vs_p50=0,
    )

    # 2. Strategy Decomposition
    decomposition = decompose_strategies(trades, metrics.get("total_pnl_pct", 0))
    portfolio_contrib = analyze_portfolio_contribution(decomposition, metrics.get("total_pnl_pct", 0))

    # 3. LONG vs SHORT
    ls_analysis = analyze_long_short(trades)

    # 4. Drawdown Audit
    dd_audit = audit_drawdown_management(trades)

    # 5. Temporal Stability (needed for overfitting and MO)
    temporal = run_temporal_stability(trades, window_days=30) if len(trades) >= 10 else None

    # 6. Overfitting Assessment
    n_active = len(set(t.get("entry_type", "") for t in trades))
    overfitting = assess_overfitting_risk(metrics, trades, mc, temporal, n_active_strategies=n_active)

    # 7. Outlier Analysis
    outlier = run_outlier_analysis(trades)

    # 8. Regime Classification
    regime = classify_regime_performance(trades)

    # 9. Multi-Objective Score
    mo = compute_multi_objective_score(metrics, trades, mc, overfitting, temporal)

    # 10. Cost Sensitivity
    cost_sens = run_cost_sensitivity(trades, metrics)

    # 11. Degradation Suite
    degradation = run_degradation_suite(trades, metrics)

    # 12. Walk-Forward Audit
    wf_audit = None
    if wf_results:
        wf_audit = audit_walk_forward_results(wf_results)

    # 13. Concurrent Position Audit
    conc_audit = audit_concurrent_positions(trades)

    # 14. Full Equity Curve
    equity = build_full_equity_curve(trades)

    # 15. Recommendations
    regime_for_rec = RegimeAnalysisResult(regimes={k: v for k, v in regime.regimes.items()})
    recs = generate_19_point_recommendation(
        metrics, mo, mc, overfitting, temporal, regime_for_rec, has_oos or (wf_audit and wf_audit.has_oos_edge),
    )

    # 16. Verdict
    verdict, justification, details = compute_verdict(
        metrics, trades, mo, mc, overfitting, temporal,
        has_oos=has_oos or (wf_audit and wf_audit.has_oos_edge),
    )

    n_analyses = 16  # Total modules executed

    logger.info(
        "Auditoria completa: %s | Veredito: %s | Composite: %.1f | %d analises executadas",
        version_label, verdict, mo.composite_score, n_analyses,
    )

    return ComprehensiveAuditResult(
        monte_carlo=mc,
        strategy_decomposition=decomposition,
        portfolio_contribution=portfolio_contrib,
        long_short=ls_analysis,
        drawdown_audit=dd_audit,
        multi_objective=mo,
        verdict=verdict,
        verdict_justification=justification,
        verdict_details=details,
        overfitting=overfitting,
        outlier=outlier,
        regime=regime,
        temporal=temporal,
        equity_curve=equity,
        recommendations=recs,
        cost_sensitivity=cost_sens,
        degradation=degradation,
        wf_audit=wf_audit,
        concurrent_audit=conc_audit,
        version_label=version_label,
        n_analyses_run=n_analyses,
    )
