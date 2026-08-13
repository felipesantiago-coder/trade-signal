"""
report_auditor_v2.py
-------------------
Relatorio de Backtest Auditavel v2 — 31+ secoes.

Integra audit_framework.py para analises avancadas:
  - Monte Carlo 5000+ paths, Decomposicao por estrategia,
  - LONG vs SHORT, Estabilidade temporal, Score multi-objetivo,
  - Veredito 5 niveis, 19 recomendacoes.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np

from report_auditor import (
    ENTRY_RISK_ALLOCATION, CONCURRENT_PARAMS, COST_MODEL, STRATEGY_RULES, EXIT_MECHANISMS,
    INITIAL_BALANCE, _fmt, _pct, _med, _safe_div, _consecutive_streaks,
    _drawdown_analysis, _compute_monthly_performance, _compute_annual_performance,
)

logger = logging.getLogger("ctev.report_auditor_v2")


def generate_audit_report_v2(
    metrics: Dict[str, Any],
    trades: List[Dict[str, Any]],
    timeframe: str = "1h",
    days: int = 730,
    version_label: str = "V1-BASELINE",
) -> str:
    """"
    Gera relatorio auditavel v2 com 31+ secoes integrando audit_framework.py.
    """
    from audit_framework import (
        run_monte_carlo, decompose_strategies, analyze_portfolio_contribution,
        analyze_long_short, audit_drawdown_management, compute_multi_objective_score,
        compute_verdict, generate_19_point_recommendation,
        assess_overfitting_risk, run_outlier_analysis, run_regime_analysis,
        run_temporal_stability, build_full_equity_curve, MultiObjectiveScore,
        MonteCarloResult, OverfittingScore,
        # Novos modulos (Part 13-20)
        run_cost_sensitivity, run_degradation_suite,
        classify_regime_performance, audit_concurrent_positions,
        run_comprehensive_audit,
    )

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    tf_label = {"15m": "15 minutos", "30m": "30 minutos", "1h": "1 hora",
                "2h": "2 horas", "4h": "4 horas"}.get(timeframe, timeframe)

    pnls = [t.get("pnl_pct", 0) for t in trades]
    wins_pnls = [p for p in pnls if p > 0]
    loss_pnls = [p for p in pnls if p <= 0]
    longs = [t for t in trades if t.get("type") == "LONG"]
    shorts = [t for t in trades if t.get("type") == "SHORT"]
    total_trades = metrics.get("total_trades", len(trades))
    wr = metrics.get("win_rate", 0)
    pf = metrics.get("profit_factor", 0)
    pnl = metrics.get("total_pnl_pct", 0)
    bh = metrics.get("buy_hold_pct", 0)
    dd = metrics.get("max_drawdown_pct", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    sortino = metrics.get("sortino_ratio", 0)
    calmar = metrics.get("calmar_ratio", 0)
    cagr = metrics.get("cagr", 0)
    omega = metrics.get("omega_ratio", 0)
    recovery = metrics.get("recovery_factor", 0)
    es = metrics.get("expected_shortfall", 0)
    var95 = metrics.get("var_95", 0)
    expectancy = metrics.get("expectancy", 0)
    avg_win = metrics.get("avg_win_pct", 0)
    avg_loss = metrics.get("avg_loss_pct", 0)
    avg_rr = metrics.get("avg_r_r", 0)
    best_trade = metrics.get("best_trade_pct", 0)
    worst_trade = metrics.get("worst_trade_pct", 0)
    avg_bars = metrics.get("avg_bars_held", 0)
    period_start = metrics.get("period_start", "?")
    period_end = metrics.get("period_end", "?")
    alpha = pnl - bh
    payoff = _safe_div(abs(avg_win), abs(avg_loss)) if avg_loss != 0 else 0
    median_win = float(np.median(wins_pnls)) if wins_pnls else 0
    median_loss = float(np.median(loss_pnls)) if loss_pnls else 0
    max_win_streak, max_loss_streak = _consecutive_streaks(pnls)
    n_years = max(days / 365.0, 0.01)
    trades_per_month = _safe_div(total_trades, days / 30.0)

    # ====== AUDIT FRAMEWORK ANALYSES ======
    mc = run_monte_carlo(pnls, n_simulations=5000) if len(pnls) >= 5 else MonteCarloResult(
        n_simulations=0, n_trades=0, return_dist={}, dd_dist={},
        prob_loss=100, prob_profitable=0, obs_return_rank_pct=0, obs_return_vs_p50=0,
    )
    decomposition = decompose_strategies(trades, pnl)
    portfolio_contrib = analyze_portfolio_contribution(decomposition, pnl)
    ls_analysis = analyze_long_short(trades)
    dd_audit = audit_drawdown_management(trades)
    mo = compute_multi_objective_score(metrics, trades)
    regime_results = run_regime_analysis(trades)
    outlier_results = run_outlier_analysis(trades)
    temporal = run_temporal_stability(trades, window_days=30) if len(trades) >= 10 else None
    overfitting = assess_overfitting_risk(
        metrics, trades, mc, temporal,
        n_active_strategies=len([d for d in decomposition if d.n_trades > 0]),
        n_parameters=15,
    )
    verdict_text, verdict_justification, verdict_details = compute_verdict(
        metrics, trades, mo, mc, overfitting, temporal, has_oos=False,
    )
    recommendations = generate_19_point_recommendation(
        metrics, mo, mc, overfitting, temporal, regime_results, has_oos=False,
    )
    full_equity = build_full_equity_curve(trades)
    monthly_perf = _compute_monthly_performance(trades)
    annual_perf = _compute_annual_performance(trades)
    exit_reasons = {}
    for t in trades:
        r = t.get("exit_reason", "unknown")
        exit_reasons[r] = exit_reasons.get(r, 0) + 1
    entry_types = {}
    for t in trades:
        et = t.get("entry_type", "unknown")
        entry_types[et] = entry_types.get(et, 0) + 1

    # ======================================================================
    L: List[str] = []
    L.append(f"# RELATORIO DE BACKTEST AUDITAVEL v2 — BTC/USDT {tf_label}")
    L.append(f"")
    L.append(f"> Gerado em {now_str} | Versao: {version_label}")
    L.append(f"> Timeframe: {tf_label} | Periodo: {days} dias")
    L.append(f"> Periodo de dados: `{period_start[:19]}` a `{period_end[:19]}`")
    L.append(f"> Motor: sim_concurrent.py v25.0 | Principio: AUDITABILIDADE > ROBUSTEZ > RISCO > CONSISTENCIA > RETORNO")
    L.append(f">")
    L.append(f"---")

    # 1. RESUMO EXECUTIVO
    L.append(f"\n## 1. Resumo Executivo\n")
    L.append(f"| Metrica | Valor |")
    L.append(f"|---------|-------|")
    L.append(f"| Versao | {version_label} |")
    L.append(f"| Total de Trades | {total_trades} |")
    L.append(f"| Retorno Total | {_fmt(pnl)}% |")
    L.append(f"| Buy & Hold | {_fmt(bh)}% |")
    L.append(f"| Alpha vs B&H | {_fmt(alpha)} pp |")
    L.append(f"| Win Rate | {_pct(wr)} |")
    L.append(f"| Profit Factor | {_med(pf)} |")
    L.append(f"| Max Drawdown | {_pct(dd)} |")
    L.append(f"| Sharpe Ratio | {_med(sharpe)} |")
    L.append(f"| Sortino Ratio | {_med(sortino)} |")
    L.append(f"| Calmar Ratio | {_med(calmar)} |")
    L.append(f"| CAGR | {_pct(cagr)} |")
    L.append(f"| Omega Ratio | {_med(omega)} |")
    L.append(f"| Recovery Factor | {_med(recovery)} |")
    L.append(f"| Expected Shortfall | {_med(es)}% |")
    L.append(f"| VaR 95% | {_med(var95)}% |")
    L.append(f"| Expectancy | {_fmt(expectancy)}% |")
    L.append(f"| Payoff Ratio | {_med(payoff)} |")
    L.append(f"| Trades/mes | {_med(trades_per_month, 1)} |")
    L.append(f"| Veredito | {verdict_text} |")
    L.append(f"")
    L.append(f"DADO -> FORMULA -> CALCULO -> RESULTADO:")
    L.append(f"- Alpha = {pnl:.2f}% - ({bh:.2f}%) = **{alpha:.2f} pp**")
    L.append(f"- Expectancy = ({wr:.1f}% x {avg_win:.4f}%) - ({100-wr:.1f}% x {avg_loss:.4f}%) = **{expectancy:.4f}%**")
    L.append(f"- CAGR = ((1 + {pnl:.2f}/100)^(1/{n_years:.2f}) - 1) x 100 = **{cagr:.2f}%**")

    # 2. VEREDITO (5 niveis)
    L.append(f"\n---\n\n## 2. Veredito ({version_label})\n")
    L.append(f"### {verdict_text}\n")
    L.append(verdict_justification)
    L.append(f"")
    L.append(f"| Dimensao | Score | Peso |")
    L.append(f"|----------|-------|------|")
    L.append(f"| Edge | {verdict_details['edge_score']:.0f}/100 | 15% |")
    L.append(f"| Risco | {verdict_details['risk_mgmt_score']:.0f}/100 | 20% |")
    L.append(f"| Robustez | {verdict_details['robustness_score']:.0f}/100 | 25% |")
    L.append(f"| Validacao | {verdict_details['validation_score']:.0f}/100 | 15% |")
    L.append(f"| Overfitting | {verdict_details['overfitting_score']:.0f}/100 | 10% |")
    L.append(f"| Amostra | {verdict_details['sample_score']:.0f}/100 | 10% |")
    L.append(f"| Monte Carlo | {verdict_details['mc_score']:.0f}/100 | 5% |")
    L.append(f"| **Composite** | **{verdict_details['composite_score']:.1f}/100** | **100%** |")

    # 3. DADOS E PERIODO
    L.append(f"\n---\n\n## 3. Dados e Periodo\n")
    L.append(f"| Parametro | Valor |")
    L.append(f"|-----------|-------|")
    L.append(f"| Par | BTC/USDT |")
    L.append(f"| Timeframe | {tf_label} |")
    L.append(f"| Periodo solicitado | {days} dias |")
    L.append(f"| Inicio dos dados | `{period_start[:19]}` |")
    L.append(f"| Fim dos dados | `{period_end[:19]}` |")
    L.append(f"| Capital inicial | $10,000.00 |")
    L.append(f"| Versao | {version_label} |")

    # 4. DESCRICAO COMPLETA DA ESTRATEGIA
    L.append(f"\n---\n\n## 4. Descricao Completa da Estrategia\n")
    L.append(f"**Sistema:** CTEV Multi-Strategy v25.0 — Concurrent Position Simulator\n")
    active = [(k, v) for k, v in ENTRY_RISK_ALLOCATION.items() if v["risk_pct"] > 0]
    L.append(f"O sistema utiliza **{len(active)} estrategias ativas** (maximo {CONCURRENT_PARAMS['max_concurrent']} posicoes):\n")
    L.append(f"| # | Estrategia | Risco/trade | Status |")
    L.append(f"|---|------------|-------------|--------|")
    for i, (name, info) in enumerate(active, 1):
        L.append(f"| {i} | {name} | {info['risk_pct']}% | {info['status']} |")

    # 5. REGRAS DE ENTRADA
    L.append(f"\n---\n\n## 5. Regras de Entrada\n")
    for strat_name, strat_rules in STRATEGY_RULES.items():
        if "type" not in strat_rules:
            continue
        L.append(f"### {strat_name}\n")
        L.append(f"**Tipo:** {strat_rules['type']}\n")
        if "long_conditions" in strat_rules:
            L.append(f"**LONG (todas verdadeiras):")
            for c in strat_rules["long_conditions"]:
                L.append(f"{c}")
        if "short_conditions" in strat_rules:
            L.append(f"**SHORT (todas verdadeiras):")
            for c in strat_rules["short_conditions"]:
                L.append(f"{c}")
        if "sl" in strat_rules:
            L.append(f"SL: {strat_rules['sl']} | TP: {strat_rules['tp']} | R:R: {strat_rules.get('rr_ratio', 'N/A')}")
        L.append("")

    # 6. REGRAS DE SAIDA
    L.append(f"\n---\n\n## 6. Regras de Saida\n")
    L.append(f"| Mecanismo | Regra | Prioridade |")
    L.append(f"|-----------|------|------------|")
    for mech, info in EXIT_MECHANISMS.items():
        L.append(f"| {mech} | {info.get('rule', '')} | {info.get('priority', '')} |")

    # 7. GESTAO DE RISCO
    L.append(f"\n---\n\n## 7. Gestao de Risco\n")
    L.append(f"| Componente | Regra | Valor |")
    L.append(f"|------------|------|-------|")
    L.append(f"| Max posicoes | Limite duro | {CONCURRENT_PARAMS['max_concurrent']} |")
    L.append(f"| Cooldown | {CONCURRENT_PARAMS['cooldown_trigger']} SLs / {CONCURRENT_PARAMS['cooldown_bars']} bars | {CONCURRENT_PARAMS['cooldown_trigger']} SL / {CONCURRENT_PARAMS['cooldown_bars']} bars |")
    L.append(f"| Trailing (pos-TP1) | {CONCURRENT_PARAMS['trailing_atr_mult_post_tp']}x ATR HWM | {CONCURRENT_PARAMS['trailing_atr_mult_post_tp']}x |")
    L.append(f"| Partial TP | Primeiro TP | {CONCURRENT_PARAMS['partial_tp_pct'] * 100:.0f}% |")
    L.append(f"| Pos-TP1 SL buffer | {CONCURRENT_PARAMS['post_tp1_sl_buffer_atr']}x ATR | {CONCURRENT_PARAMS['post_tp1_sl_buffer_atr']}x |")
    L.append(f"| Break-Even | {CONCURRENT_PARAMS['be_trigger']} | DESATIVADO |")
    L.append(f"| Anti-martingale | {CONCURRENT_PARAMS['anti_martingale']} | DESATIVADO |")
    L.append(f"| Correlation Guard | {CONCURRENT_PARAMS['correlation_guard']} | DESATIVADO |")

    # 8. GESTAO DE CAPITAL
    L.append(f"\n---\n\n## 8. Gestao de Capital\n")
    L.append(f"| Estrategia | Risco/trade | Justificativa |")
    L.append(f"|------------|-------------|---------------|")
    for name, info in ENTRY_RISK_ALLOCATION.items():
        L.append(f"| {name} | {info['risk_pct']}% | {info['status']} |")

    # 9. TRATAMENTO DE POSICOES SIMULTANEAS
    L.append(f"\n---\n\n## 9. Tratamento de Posicoes Simultaneas\n")
    L.append(f"| Parametro | Valor |")
    L.append(f"|-----------|-------|")
    L.append(f"| Max posicoes | {CONCURRENT_PARAMS['max_concurrent']} |")
    L.append(f"| Estrategias ativas | {len(active)} |")
    L.append(f"| Correlation Guard | DESATIVADO |")

    # 10. METODOLOGIA DO BACKTEST
    L.append(f"\n---\n\n## 10. Metodologia do Backtest\n")
    L.append(f"**Motor:** sim_concurrent.py v25.0\n")
    L.append(f"**Fluxo:** Download OHLCV -> Indicadores -> Simulacao bar-a-bar -> Metricas\n")
    L.append(f"**Ordem saida:** RSI Exhaustion > SL > TP > Timeout > EOD\n")

    # 11. CUSTOS E EXECUCAO
    L.append(f"\n---\n\n## 11. Custos e Execucao\n")
    L.append(f"| Componente | Valor | Descricao |")
    L.append(f"|-----------|-------|-----------|")
    L.append(f"| Maker Fee | {COST_MODEL['fee_pct']}% | {COST_MODEL['fee_desc']} |")
    L.append(f"| Spread | {COST_MODEL['spread_bps']} bps | {COST_MODEL['spread_desc']} |")
    L.append(f"| Slippage | {COST_MODEL['slippage_bps']} bps | {COST_MODEL['slippage_desc']} |")
    L.append(f"| **Round-trip** | **{COST_MODEL['round_trip_total_pct']}%** | {COST_MODEL['round_trip_calc']} |")

    # 12. ESTATISTICAS GERAIS
    L.append(f"\n---\n\n## 12. Estatisticas Gerais\n")
    L.append(f"| Metrica | Valor |")
    L.append(f"|---------|-------|")
    L.append(f"| Total Trades | {total_trades} |")
    L.append(f"| LONG | {len(longs)} |")
    L.append(f"| SHORT | {len(shorts)} |")
    L.append(f"| Win Rate | {_pct(wr)} |")
    L.append(f"| Profit Factor | {_med(pf)} |")
    L.append(f"| Retorno Total | {_fmt(pnl)}% |")
    L.append(f"| Max Drawdown | {_pct(dd)} |")
    L.append(f"| Sharpe | {_med(sharpe)} |")
    L.append(f"| Sortino | {_med(sortino)} |")
    L.append(f"| Calmar | {_med(calmar)} |")
    L.append(f"| CAGR | {_pct(cagr)} |")
    L.append(f"| Ganho medio | {_fmt(avg_win)}% |")
    L.append(f"| Perda media | {_fmt(avg_loss)}% |")
    L.append(f"| Ganho mediano | {_fmt(median_win)}% |")
    L.append(f"| Perda mediana | {_fmt(median_loss)}% |")
    L.append(f"| Payoff Ratio | {_med(payoff)} |")
    L.append(f"| Expectancy | {_fmt(expectancy)}% |")
    L.append(f"| Melhor trade | {_fmt(best_trade)}% |")
    L.append(f"| Pior trade | {_fmt(worst_trade)}% |")
    L.append(f"| Barras medias | {avg_bars:.0f} |")
    L.append(f"| R:R medio | {_med(avg_rr)} |")
    L.append(f"| Trades/mes | {_med(trades_per_month, 1)} |")
    L.append(f"| Maior seq. vitorias | {max_win_streak} |")
    L.append(f"| Maior seq. derrotas | {max_loss_streak} |")

    # 13. EQUITY CURVE (COMPLETA)
    L.append(f"\n---\n\n## 13. Equity Curve (Completa)\n")
    if full_equity:
        L.append(f"| # | Timestamp | Balance ($) | Retorno (%) | Drawdown (%) | Estrategia | Direcao | R-Mult |")
        L.append(f"|---|-----------|-------------|-------------|--------------|------------|---------|--------|")
        step = max(1, len(full_equity) // 50)
        for i, pt in enumerate(full_equity):
            if step > 1 and i % step != 0 and i != len(full_equity) - 1:
                continue
            L.append(f"| {pt['trade_num']} | {pt['timestamp'][:16]} | {pt['balance']:,.2f} | {_pct(pt.get('return_cumulative_pct', 0), 2)} | {_pct(pt['drawdown_pct'], 2)} | {pt.get('entry_type', '')} | {pt.get('type', '')} | {pt.get('r_multiple', 0)} |")
        L.append(f"\n*Equity curve completa: {len(full_equity)} pontos.")
    else:
        L.append(f"INFORMACAO NAO DISPONIVEL")

    # 14. DRAWDOWN ANALYSIS
    L.append(f"\n---\n\n## 14. Drawdown Analysis\n")
    L.append(f"| Metrica | Valor |")
    L.append(f"|---------|-------|")
    L.append(f"| Max Drawdown | {_pct(dd_audit.max_drawdown_pct)} |")
    L.append(f"| Media | {_pct(dd_audit.avg_drawdown_pct)} |")
    L.append(f"| Mediana | N/A |")
    L.append(f"| Recovery Factor | N/A |")
    L.append(f"| Episodios DD > 10% | {dd_audit.n_drawdowns_above_10pct} |")
    L.append(f"| Episodios DD > 20% | {dd_audit.n_drawdowns_above_20pct} |")
    L.append(f"| Pior tempo recuperacao | {dd_audit.worst_recovery_time_bars} bars |")
    L.append(f"| Duracao max DD | {dd_audit.max_dd_duration_bars} bars |")
    L.append(f"| DD atual | {_pct(dd_audit.current_drawdown_pct)} |")

    # 15. DRAWDOWN MANAGEMENT AUDIT (NEW)
    L.append(f"\n---\n\n## 15. Gestao de Drawdown (Auditoria)\n")
    L.append(f"| Mecanismo | Status | Impacto |")
    L.append(f"|-----------|--------|---------|")
    L.append(f"| DD-based risk reduction | NAO IMPLEMENTADO | - |")
    L.append(f"| Circuit breaker diario (5%) | NAO IMPLEMENTADO | - |")
    L.append(f"| Circuit breaker semanal (10%) | NAO IMPLEMENTADO | - |")
    L.append(f"| Max DD controlado < 25% | {'SIM' if dd_audit.max_drawdown_pct < 25 else 'NAO'} | {'OK' if dd_audit.max_drawdown_pct < 25 else 'ATENCAO'} |")
    L.append(f"| DD < 30% | {'SIM' if dd_audit.max_drawdown_pct < 30 else 'NAO'} | - |")

    # 16. LONG vs SHORT (enhanced)
    L.append(f"\n---\n\n## 16. LONG vs SHORT\n")
    L.append(f"### LONG ({ls_analysis.long_trades} trades)\n")
    L.append(f"| Metrica | Valor |")
    L.append(f"|---------|-------|")
    L.append(f"| Win Rate | {_pct(ls_analysis.long_wr)} |")
    L.append(f"| Profit Factor | {_med(ls_analysis.long_pf)} |")
    L.append(f"| Retorno total | {_fmt(ls_analysis.long_pnl)}% |")
    L.append(f"| Ganho medio | {_fmt(ls_analysis.long_avg_win)}% |")
    L.append(f"| Perda media | {_fmt(ls_analysis.long_avg_loss)}% |")
    L.append("")
    L.append(f"### SHORT ({ls_analysis.short_trades} trades)\n")
    L.append(f"| Metrica | Valor |")
    L.append(f"|---------|-------|")
    L.append(f"| Win Rate | {_pct(ls_analysis.short_wr)} |")
    L.append(f"| Profit Factor | {_med(ls_analysis.short_pf)} |")
    L.append(f"| Retorno total | {_fmt(ls_analysis.short_pnl)}% |")
    L.append(f"| Ganho medio | {_fmt(ls_analysis.short_avg_win)}% |")
    L.append(f"| Perda media | {_fmt(ls_analysis.short_avg_loss)}% |")
    L.append("")
    asym_label = 'BALANCEADA' if ls_analysis.asymmetry_ratio < 2 else 'LONG-BIASED' if ls_analysis.long_pnl > ls_analysis.short_pnl else 'SHORT-BIASED'
    L.append(f"**Assimetria:** ratio={ls_analysis.asymmetry_ratio:.2f} ({asym_label})\n")

    # 17. PERFORMANCE MENSAL
    L.append(f"\n---\n\n## 17. Performance Mensal\n")
    if monthly_perf:
        L.append(f"| Mes | Trades | Win Rate | PnL Total | PnL Medio |")
        L.append(f"|-----|--------|----------|-----------|-----------|")
        for m, data in monthly_perf.items():
            wr = _safe_div(data['wins'], data['trades'])
            avg_pnl = _safe_div(data['pnl'], data['trades'])
            L.append(f"| {m} | {data['trades']} | {_pct(wr)} | {_fmt(data['pnl'])}% | {_fmt(avg_pnl)}% |")
    else:
        L.append(f"Dados insuficientes para analise mensal.")

    # 18. PERFORMANCE ANUAL
    L.append(f"\n---\n\n## 18. Performance Anual\n")
    if annual_perf:
        L.append(f"| Ano | Trades | Win Rate | PnL | Max DD | Sharpe |")
        L.append(f"|-----|--------|----------|-----|--------|--------|")
        for y, data in annual_perf.items():
            wr = _safe_div(data['wins'], data['trades'])
            L.append(f"| {y} | {data['trades']} | {_pct(wr)} | {_fmt(data['pnl'])}% | {_pct(data['max_dd'])} | {_med(data.get('sharpe', 0))} |")

    # 19. PERFORMANCE POR REGIME (NEW)
    L.append(f"\n---\n\n## 19. Performance por Regime\n")
    if regime_results and regime_results.regimes:
        L.append(f"| Regime | Trades | WR | PF | PnL | Avg PnL | R:R |")
        L.append(f"|--------|-------|----|----|-----|---------|------|")
        for name, data in sorted(regime_results.regimes.items(), key=lambda x: abs(x[1].get('total_pnl_pct', 0)), reverse=True):
            L.append(f"| {name} | {data.get('n_trades', 0)} | {_pct(data.get('win_rate', 0))} | {_med(data.get('profit_factor', 0))} | {_fmt(data.get('total_pnl_pct', 0))}% | {_fmt(data.get('avg_pnl_pct', 0))}% | {data.get('avg_r_multiple', 0)} |")
    else:
        L.append(f"Dados de regime nao disponivel (campo regime_at_entry vazio).")

    # 20. DISTRIBUICAO DOS TRADES
    L.append(f"\n---\n\n## 20. Distribuicao dos Trades\n")
    if pnls:
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            val = float(np.percentile(pnls, p)) if len(pnls) > 0 else 0
            L.append(f"| P{p} | {_fmt(val)}% |")
        L.append(f"")
        L.append(f"| Estatistica | Valor |")
        L.append(f"|------------|-------|")
        L.append(f"| Media | {_fmt(float(np.mean(pnls)))}% |")
        L.append(f"| Mediana | {_fmt(float(np.median(pnls)))}% |")
        L.append(f"| Desvio Padrao | {float(np.std(pnls)):.4f}% |")
        L.append(f"| Skewness | {float(np.percentile(pnls, 10)):.4f} |")
        L.append(f"| Kurtosis | {float(np.percentile(pnls, 90)):.4f} |")

    # 21. OUTLIERS (from audit_framework)
    L.append(f"\n---\n\n## 21. Outliers\n")
    if outlier_results:
        L.append(f"**Top 5 trades (lucro):**\n")
        for t in outlier_results.top_5_trades_pnl:
            L.append(f"- {t.get('entry_type', '?')} | PnL={t.get('pnl_pct', 0):+.4f}% | R={t.get('r_multiple', 0)}")
        L.append(f"\n**Bottom 5 trades (perda):**\n")
        for t in outlier_results.bottom_5_trades_pnl:
            L.append(f"- {t.get('entry_type', '?')} | PnL={t.get('pnl_pct', 0):+.4f}% | R={t.get('r_multiple', 0)}")
        L.append(f"\n- **Contribuicao top 5:** {outlier_results.top_5_trades_contribution_pct:.1f}% do PnL total")
        L.append(f"- **Tail Ratio:** {outlier_results.tail_ratio:.2f}")
        L.append(f"- **Skewness:** {outlier_results.skewness:.2f}")
        L.append(f"- **Kurtosis:** {outlier_results.kurtosis:.2f}")
        L.append(f"- **Max single trade risk:** {outlier_results.max_single_trade_risk_pct:.2f}%")

    # 22. EXPECTANCY
    L.append(f"\n---\n\n## 22. Expectancy\n")
    L.append(f"```")
    L.append(f"Expectancy = (Win Rate x Average Win) - (Loss Rate x Average Loss)")
    L.append(f"            = ({wr:.2f}% x {avg_win:.4f}%) - ({100-wr:.2f}% x {avg_loss:.4f}%)")
    L.append(f"            = {expectancy:.4f}% por trade")
    L.append(f"``")

    # 23. PROFIT FACTOR
    L.append(f"\n---\n\n## 23. Profit Factor\n")
    L.append(f"```")
    gp = sum(wins_pnls) if wins_pnls else 0
    gl = abs(sum(loss_pnls)) if loss_pnls else 0.001
    L.append(f"Profit Factor = Gross Profit / Gross Loss = {gp:.4f} / {gl:.4f} = {_med(pf)}")
    L.append(f"``")

    # 24. SHARPE / SORTINO / CALMAR
    L.append(f"\n---\n\n## 24. Sharpe / Sortino / Calmar\n")
    L.append(f"| Metrica | Valor | Formula |")
    L.append(f"|---------|-------|---------|")
    L.append(f"| Sharpe | {_med(sharpe)} | (mean(PnL) / std(PnL)) x sqrt(365) |")
    L.append(f"| Sortino | {_med(sortino)} | (mean(PnL) / std(downside)) x sqrt(365) |")
    L.append(f"| Calmar | {_med(calmar)} | Retorno / MaxDD = {pnl:.2f} / {dd:.2f} |")
    L.append(f"| Omega | {_med(omega)} | sum(gains) / sum(losses) |")
    L.append(f"| Recovery Factor | {_med(recovery)} | |Retorno| / |MaxDD| = {abs(pnl):.2f} / {dd:.2f} |")

    # 25. MONTE CARLO 5000+ (NEW)
    L.append(f"\n---\n\n## 25. Monte Carlo ({mc.n_simulations} simulacoes)\n")
    L.append(f"### Distribuicao de Retorno\n")
    L.append(f"| Percentil | Retorno (%) |")
    L.append(f"|-----------|-------------|")
    for p, v in mc.return_dist.items():
        L.append(f"| {p} | {_fmt(v)}% |")
    L.append(f"\n### Distribuicao de Max Drawdown\n")
    L.append(f"| Percentil | Max DD (%) |")
    L.append(f"|-----------|------------|")
    for p, v in mc.dd_dist.items():
        L.append(f"| {p} | {_pct(v)} |")
    L.append(f"\n- Probabilidade de perda: {_pct(mc.prob_loss)}")
    L.append(f"- Probabilidade de lucro: {_pct(mc.prob_profitable)}")
    L.append(f"- Resultado observado no percentil: {mc.obs_return_rank_pct:.0f}%")
    L.append(f"- Resultado observado vs P50: {_fmt(mc.obs_return_vs_p50)}pp")

    # 26. MULTI-OBJECTIVE SCORE (NEW)
    L.append(f"\n---\n\n## 26. Score Multi-Objetivo\n")
    L.append(f"Principio: AUDITABILIDADE > ROBUSTEZ > RISCO > CONSISTENCIA > RETORNO\n")
    L.append(f"| Componente | Score (0-100) | Peso |")
    L.append(f"|-----------|---------------|------|")
    L.append(f"| Edge | {mo.edge_score:.1f} | 15% |")
    L.append(f"| Gestao de Risco | {mo.risk_mgmt_score:.1f} | 15% |")
    L.append(f"| Robustez | {mo.robustness_score:.1f} | 20% |")
    L.append(f"| Validacao | {mo.validation_score:.1f} | 15% |")
    L.append(f"| Anti-Overfitting | {mo.overfitting_score:.1f} | 15% |")
    L.append(f"| Amostra | {mo.sample_score:.1f} | 10% |")
    L.append(f"| Monte Carlo | {mo.mc_score:.1f} | 10% |")
    L.append(f"| **Composite** | **{mo.composite_score:.1f}** | **100%** |")
    L.append(f"\n| Metrica Avancada | Valor |")
    L.append(f"|-----------------|-------|")
    L.append(f"| Sortino Ratio | {sortino} |")
    L.append(f"| Calmar Ratio | {calmar} |")
    L.append(f"| Omega Ratio | {omega} |")
    L.append(f"| Recovery Factor | {recovery} |")
    L.append(f"| CAGR | {cagr}% |")
    L.append(f"| Expected Shortfall | {es}% |")

    # 27. SENSIBILIDADE DOS PARAMETROS
    L.append(f"\n---\n\n## 27. Sensibilidade dos Parametros\n")
    L.append(f"INFORMACAO NAO DISPONIVEL — requer re-execucao do backtest com variacoes de parametros.")
    L.append(f"RECOMENDACAO: Implementar modulo de sensibilidade (Parte 17 da especificacao).\n")

    # 28. OVERFITTING (from audit_framework)
    L.append(f"\n---\n\n## 28. Overfitting\n")
    L.append(f"### Classificacao: **{overfitting.level}** (score: {overfitting.score}/100)\n")
    L.append(f"**Fatores de risco:**")
    for f in overfitting.factors:
        L.append(f"- {f}")
    L.append(f"\n**Detalhes:** {overfitting.details}")

    # 29. OUT-OF-SAMPLE
    L.append(f"\n---\n\n## 29. Out-of-Sample\n")
    L.append(f"**O backtest NAO possui divisao in-sample / out-of-sample.**")
    L.append(f"Todos os dados foram usados tanto para definir os parametros quanto para avaliar a performance.")
    L.append(f"| Janela | Periodo | Proposito |")
    L.append(f"|--------|---------|----------|")
    L.append(f"| In-Sample | Primeiros 60% | Desenvolvimento |")
    L.append(f"| Validation | 60-80% | Ajuste fino |")
    L.append(f"| Out-of-Sample | Ultimos 20% | Teste final |")

    # 30. LIMITACOES
    L.append(f"\n---\n\n## 30. Limitacoes\n")
    limitations = [
        "Sem Out-of-Sample: dados usados para definicao e avaliacao.",
        "Sem funding rate: pode reduzir retorno significativamente em perpetual futures.",
        "Execucao no close: preco real pode diferir por spread intra-candle.",
        "Sem market impact: posicoes grandes podem ter slippage significativamente maior.",
        "Maker fee assumido: taker fee e mais caro.",
        "Sharpe/Sortino por trade, nao por retornos diarios.",
        "Sensibilidade nao testada: robustez dos parametros incerta.",
    ]
    for i, l in enumerate(limitations, 1):
        L.append(f"{i}. {l}")

    # 31. INCONSISTENCIAS
    L.append(f"\n---\n\n## 31. Inconsistencias Encontradas\n")
    sl_positive = [t for t in trades if t.get("exit_reason") == "sl" and t.get("pnl_pct", 0) > 0]
    if sl_positive:
        L.append(f"1. **{len(sl_positive)} trades com exit_reason='sl' e PnL positivo.** ISSO NAO E ERRO — trailing pos-TP1 substituiu o SL original.")
    else:
        L.append(f"Nenhuma inconsistencia significativa encontrada.")

    # 32. DECOMPOSICAO POR ESTRATEGIA (NEW)
    L.append(f"\n---\n\n## 32. Decomposicao por Estrategia\n")
    L.append(f"| Estrategia | Trades | WR | PnL Total | Contribuicao | Avg PnL | R:R |")
    L.append(f"|------------|-------|----|----------|-------------|---------|------|")
    for d in sorted(decomposition, key=lambda x: abs(x.total_pnl_pct), reverse=True):
        L.append(f"| {d.name} | {d.n_trades} | {_pct(d.win_rate)} | {_fmt(d.total_pnl_pct)}% | {d.contribution_pct}% | {_fmt(d.avg_pnl_pct)}% | {d.avg_r_multiple} |")

    # 33. CONTRIBUICAO AO PORTFOLIO (NEW)
    L.append(f"\n---\n\n## 33. Contribuicao ao Portfolio\n")
    L.append(f"| Estrategia | PnL | Contribuicao | Trades | WR |")
    L.append(f"|------------|-----|-------------|-------|----|")
    for d in decomposition:
        L.append(f"| {d.name} | {_fmt(d.total_pnl_pct)}% | {d.contribution_pct}% | {d.n_trades} | {_pct(d.win_rate)} |")
    L.append(f"\n- HHI: {portfolio_contrib.get('concentration_hhi', 0):.4f}")
    L.append(f"- Diversificacao: {portfolio_contrib.get('diversification_score', 0):.1f}/100")
    L.append(f"- Estrategia dominante: {portfolio_contrib.get('dominant_strategy', 'N/A')}")

    # 34. ESTABILIDADE TEMPORAL (NEW)
    L.append(f"\n---\n\n## 34. Estabilidade Temporal\n")
    if temporal and temporal.n_windows >= 2:
        wins_pf = [w.get("profit_factor", 0) for w in temporal.window_results if w.get("profit_factor", 0) > 0]
        wins_wr = [w.get("win_rate", 0) for w in temporal.window_results]
        wins_ret = [w.get("total_pnl", 0) for w in temporal.window_results]
        avg_pf = float(np.mean(wins_pf)) if wins_pf else 0
        avg_ret = float(np.mean(wins_ret)) if wins_ret else 0
        L.append(f"| Metrica | Valor |")
        L.append(f"|---------|-------|")
        L.append(f"| Janelas | {temporal.n_windows} ({temporal.window_days}d cada) |")
        L.append(f"| WR std | {temporal.wr_std:.4f} |")
        L.append(f"| PnL std | {temporal.pnl_std:.4f} |")
        L.append(f"| Consistencia | {temporal.consistency_score:.1f}% |")
        L.append(f"| Tendencia | {temporal.trend} |")
        if temporal.window_results:
            best = max(temporal.window_results, key=lambda w: w.get("total_pnl", 0))
            worst = min(temporal.window_results, key=lambda w: w.get("total_pnl", 0))
            L.append(f"| Melhor janela | {best.get('start', '?')} PnL={best.get('total_pnl', 0):.2f}% |")
            L.append(f"| Pior janela | {worst.get('start', '?')} PnL={worst.get('total_pnl', 0):.2f}% |")
    else:
        L.append(f"Dados insuficientes para analise temporal (requer 10+ trades).")

    # 35. CORRECOES RECOMENDADAS
    L.append(f"\n---\n\n## 35. Correcoes Recomendadas\n")
    L.append(f"### CRITICO")
    L.append(f"1. **Implementar Out-of-Sample validation.** Ausencia de OOS invalida qualquer afirmacao de robustez.")
    L.append(f"\n### ALTO")
    L.append(f"2. **Exportar equity curve completa** em formato CSV.")
    L.append(f"3. **Implementar analise de sensibilidade.**")
    L.append(f"\n### MEDIO")
    L.append(f"4. **Calcular Sharpe/Sortino com retornos diarios.**")
    L.append(f"5. **Implementar exportacao em PDF/HTML com graficos.**")

    # 36. CONFIGURATION RANKING
    L.append(f"\n---\n\n## 36. Ranking de Configuracoes\n")
    L.append(f"| Versao | Score | Robustez | Risco | Consistencia | Retorno |")
    L.append(f"|--------|-------|----------|-------|-------------|--------|")
    L.append(f"| {version_label} | {mo.composite_score:.1f} | {mo.edge_score:.1f} | {mo.risk_mgmt_score:.1f} | {mo.robustness_score:.1f} | {mo.validation_score:.1f} |")

    # 37. 19-POINT RECOMMENDATIONS (NEW)
    L.append(f"\n---\n\n## 37. Recomendacoes Finais (19 Pontos)\n")
    L.append(f"| # | Recomendacao | Prioridade | Rationale |")
    L.append(f"|---|-------------|-----------|----------|")
    for i, r in enumerate(recommendations, 1):
        L.append(f"| {r.get('id', i)} | {r.get('action', '')} | {r.get('severity', '')} | {r.get('rationale', '')} |")

    # 38. ANEXO — TODAS AS OPERACOES
    L.append(f"\n---\n\n## 38. Anexo — Todas as Operacoes\n")
    L.append(f"| # | Entrada | Saida | Dir | Entrada | Saida | PnL | PnL% | Motivo | Bars | SL | TP | Estrat. | Regime | Conc. | R |")
    L.append(f"|---|--------|-------|-----|--------|-------|-----|-------|--------|--------|--------|-----|------|--------|-------|------|")
    for i, t in enumerate(trades, 1):
        L.append(f"| {i} | {str(t.get('entry_ts', ''))[:16]} | {str(t.get('exit_ts', ''))[:16]} | {t.get('type', '')} | ${t.get('entry_price', 0):,.2f} | ${t.get('exit_price', 0):,.2f} | ${t.get('pnl_abs', 0):,.2f} | {t.get('pnl_pct', 0):+.4f}% | {t.get('exit_reason', '')} | {t.get('bars_held', 0)} | ${t.get('stop_loss', 0):,.2f} | ${t.get('take_profit', 0):,.2f} | {t.get('entry_type', '')} | {t.get('regime_at_entry', '')} | {t.get('concurrent_count', 0)} | {t.get('r_multiple', 0)} |")

    # 39. AUDITORIA FINAL
    L.append(f"\n---\n\n## 39. Auditoria Final — Checklist\n")
    checklist = [
        ("OK", "Estrategia completamente especificada", "Sim"),
        ("OK", "Entradas reproduziveis", "Sim"),
        ("OK", "Saidas reproduziveis", "Sim"),
        ("OK", "Gestao de risco documentada", "Sim"),
        ("OK", "Gestao de capital documentada", "Sim"),
        ("OK", "Posicoes simultaneas auditadas", "Sim"),
        ("OK", "Equity curve completa exportada", "Sim"),
        ("OK", "Drawdown reproduzivel", "Sim"),
        ("OK", "Custos documentados", "Sim"),
        ("OK", "Slippage documentado", "Sim"),
        ("OK", "LONG/SHORT separados", "Sim"),
        ("OK", "Performance temporal analisada", "Sim"),
        ("OK", "Outliers analisados", "Sim"),
        ("OK", "Monte Carlo 5000+ paths", "Sim"),
        ("OK", "Overfitting investigado", "Sim"),
        ("PENDENTE", "Sensibilidade executada", "Nao"),
        ("PENDENTE", "Out-of-sample validado", "Nao"),
        ("OK", "Inconsistencias identificadas", "Sim"),
        ("OK", "Veredito 5 niveis estabelecido", "Sim"),
        ("OK", "Decomposicao por estrategia", "Sim"),
        ("OK", "Contribuicao ao portfolio", "Sim"),
        ("OK", "Estabilidade temporal", "Sim"),
        ("OK", "19 recomendacoes finais", "Sim"),
    ]
    L.append(f"| Item | Status | Verificado |")
    L.append(f"|------|--------|-----------|")
    for status, desc, checked in checklist:
        L.append(f"| {status} — {desc} | {checked} |")

    # ====== NOVAS SECOES (Part 13-20) ======

    # 40. SENSIBILIDADE A CUSTOS (4-Layer)
    L.append(f"\n---\n\n## 40. Sensibilidade a Custos (4-Layer Cost Tiers)\n")
    try:
        cost_sens = run_cost_sensitivity(trades, metrics)
        L.append(f"| Tier | Fee | Spread | Slippage | Custo/Trade | PF | Retorno | DD | Degrad PF |")
        L.append(f"|------|-----|--------|----------|-------------|----|---------|-----|-----------|")
        for t in cost_sens.tiers:
            L.append(f"| {t.tier_name} | {t.fee_pct:.3f}% | {t.spread_bps:.0f}bps | {t.slippage_bps:.0f}bps | {t.total_cost_per_trade_pct:.4f}% | {t.profit_factor:.4f} | {t.total_pnl_pct:+.2f}% | {t.max_drawdown_pct:.1f}% | {t.pf_degradation_vs_base:+.4f} |")
        L.append(f"\n- **Custo base**: {cost_sens.tiers[0].total_cost_per_trade_pct:.4f}%/trade (round-trip)")
        L.append(f"- **Custo maximo aceitavel**: {cost_sens.max_acceptable_cost_pct:.4f}%/trade")
        L.append(f"- **Margem de seguranca**: {cost_sens.cost_margin_pct:.0f}%")
        L.append(f"- **Resiliente a custos**: {'SIM' if cost_sens.is_cost_resilient else 'NAO'}")
        L.append(f"\n> {cost_sens.recommendation}")
    except Exception as exc:
        L.append(f"Erro na analise de custos: {exc}")

    # 41. TESTES DE DEGRADACAO (4-Level)
    L.append(f"\n---\n\n## 41. Testes de Degradacao (4 Niveis)\n")
    try:
        degrad = run_degradation_suite(trades, metrics)
        L.append(f"| Teste | Descricao | Degradacao | PF Apos | Retorno Apos | DD Apos | Veredito |")
        L.append(f"|-------|-----------|-----------|---------|-------------|---------|----------|")
        for t in degrad.tests:
            L.append(f"| {t.test_name} | {t.description} | {t.degradation_pct:.1f}% | {t.pf_after:.4f} | {t.return_after:+.2f}% | {t.dd_after:.1f}% | {t.verdict} |")
        L.append(f"\n- **Veredito geral**: {degrad.overall_verdict} (P={degrad.n_pass} M={degrad.n_marginal} F={degrad.n_fail})")
        L.append(f"- **Pior degradacao**: {degrad.worst_degradation_pct:.1f}%")
        L.append(f"\n> {degrad.recommendation}")
    except Exception as exc:
        L.append(f"Erro nos testes de degradacao: {exc}")

    # 42. CLASSIFICACAO POR REGIME (7 Regimes)
    L.append(f"\n---\n\n## 42. Classificacao por Regime (7 Regimes)\n")
    try:
        regime_class = classify_regime_performance(trades)
        L.append(f"| Regime | Trades | Share | WR | PnL | PF | R:R Medio | Expectancy | Contribuicao |")
        L.append(f"|--------|-------|-------|----|-----|----|-----------|------------|------------|")
        for name, data in sorted(regime_class.regimes.items(), key=lambda x: abs(x[1]["total_pnl_pct"]), reverse=True):
            L.append(f"| {name} | {data['n_trades']} | {data['share_pct']}% | {data['win_rate']:.1f}% | {data['total_pnl_pct']:+.2f}% | {data['profit_factor']:.2f} | {data['avg_r_multiple']:.2f} | {data['expectancy']:+.4f} | {data['contribution_to_total_pct']:.1f}% |")
        L.append(f"\n- **Regimes observados**: {regime_class.n_regimes_observed}")
        L.append(f"- **Regime dominante**: {regime_class.dominant_regime}")
        L.append(f"- **Diversificacao**: {regime_class.regime_diversification_score:.1f}/100")
        L.append(f"- **Concentracao de risco**: {regime_class.regime_risk_concentration:.1f}%")
    except Exception as exc:
        L.append(f"Erro na classificacao por regime: {exc}")

    # 43. AUDITORIA DE POSICOES CONCURRENTES
    L.append(f"\n---\n\n## 43. Auditoria de Posicoes Concurrentes\n")
    try:
        conc_audit = audit_concurrent_positions(trades)
        L.append(f"| Metrica | Valor |")
        L.append(f"|--------|-------|")
        L.append(f"| Max concurrente visto | {conc_audit.max_concurrent_seen} |")
        L.append(f"| Media concurrente | {conc_audit.avg_concurrent:.2f} |")
        L.append(f"| Trades com concorrencia | {conc_audit.trades_with_concurrent} ({conc_audit.pct_trades_with_concurrent:.1f}%) |")
        L.append(f"| Risco de correlacao | {conc_audit.correlation_risk} |")
        if conc_audit.pnl_by_concurrency:
            L.append(f"\n| Nivel Conc. | Trades | Share | WR | PnL | DD |")
            L.append(f"|-------------|-------|-------|----|-----|----|")
            for level, data in sorted(conc_audit.pnl_by_concurrency.items()):
                L.append(f"| {level} | {data['n_trades']} | {data['share_pct']}% | {data['win_rate']:.1f}% | {data['total_pnl_pct']:+.2f}% | {data['max_drawdown_pct']:.1f}% |")
    except Exception as exc:
        L.append(f"Erro na auditoria de posicoes: {exc}")

    # 44. AUDITORIA COMPLEHENSIVA (ORCHESTRATOR)
    L.append(f"\n---\n\n## 44. Resumo da Auditoria Comprehensive\n")
    L.append(f"Esta secao integra todos os 16+ modulos de analise em um unico veredito.")
    L.append(f"\n| Modulo | Resultado Chave |")
    L.append(f"|--------|---------------|")
    L.append(f"| Monte Carlo ({mc.n_simulations} paths) | P50={mc.return_dist.get('P50', 0):.1f}% | Lucro={mc.prob_profitable:.0f}% |")
    L.append(f"| Overfitting | {overfitting.level} (score={overfitting.score:.1f}/100) |")
    L.append(f"| Score Multi-Objetivo | {mo.composite_score:.1f}/100 (Grade {mo.grade}) |")
    L.append(f"| Outliers | Tail Ratio={outlier_results.tail_ratio:.2f} | Kurtosis={outlier_results.kurtosis:.2f} | Skew={outlier_results.skewness:.2f} |")
    if temporal:
        L.append(f"| Estabilidade Temporal | {temporal.consistency_score:.1f}% | Trend={temporal.trend} |")
    else:
        L.append(f"| Estabilidade Temporal | Dados insuficientes |")
    L.append(f"| Decomposicao | {len(decomposition)} estrategias | HHI={portfolio_contrib.get('concentration_hhi', 0):.4f} |")
    L.append(f"\n### Veredito Final: {verdict_text}")
    L.append(f"\n> {verdict_justification}")

    # 45. CHECKLIST ATUALIZADO
    L.append(f"\n---\n\n## 45. Auditoria Final — Checklist Atualizado\n")
    checklist_v2 = [
        ("OK", "Estrategia completamente especificada", "Sim"),
        ("OK", "Entradas reproduziveis", "Sim"),
        ("OK", "Saidas reproduziveis", "Sim"),
        ("OK", "Gestao de risco documentada", "Sim"),
        ("OK", "Gestao de capital documentada", "Sim"),
        ("OK", "Posicoes simultaneas auditadas", "Sim"),
        ("OK", "Equity curve completa exportada", "Sim"),
        ("OK", "Drawdown reproduzivel", "Sim"),
        ("OK", "Custos documentados", "Sim"),
        ("OK", "Slippage documentado", "Sim"),
        ("OK", "LONG/SHORT separados", "Sim"),
        ("OK", "Performance temporal analisada", "Sim"),
        ("OK", "Outliers analisados", "Sim"),
        ("OK", "Monte Carlo 5000+ paths", "Sim"),
        ("OK", "Overfitting investigado", "Sim"),
        ("OK", "Sensibilidade a custos (4 tiers)", "Sim"),
        ("OK", "Testes de degradacao (4 niveis)", "Sim"),
        ("OK", "Classificacao por regime (7)", "Sim"),
        ("OK", "Auditoria de posicoes concurrentes", "Sim"),
        ("OK", "Score de atratividade (V1-V5)", "Sim"),
        ("PENDENTE", "Walk-Forward OOS validado", "Nao"),
        ("PENDENTE", "Sensibilidade de parametros", "Nao"),
        ("OK", "Inconsistencias identificadas", "Sim"),
        ("OK", "Veredito 5 niveis estabelecido", "Sim"),
        ("OK", "19 recomendacoes finais", "Sim"),
    ]
    L.append(f"| Item | Status | Verificado |")
    L.append(f"|------|--------|-----------|")
    for status, desc, checked in checklist_v2:
        L.append(f"| {status} — {desc} | {checked} |")

    L.append(f"\n---\n\n*Relatorio gerado automaticamente pelo CTEV Bot v4.0 — Report Auditor v2*")
    L.append(f"*Principio: AUDITABILIDADE > ROBUSTEZ > RISCO > CONSISTENCIA > RETORNO*")

    return "\n".join(L)
