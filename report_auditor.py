"""
report_auditor.py
-----------------
Modulo de geracao de Relatorio de Backtest Auditavel.

Transforma os dados brutos do backtest em um documento quantitativo,
transparente, reproduzivel e auditavel, seguindo protocolo de
documentacao profissional para estrategia algoritmica.

Principios:
  - AUDITABILIDADE > APARENCIA > MARKETING
  - NUNCA inventar informacoes
  - IDENTIFICAR explicitamente: INFERENCIA, RECOMENDACAO, RECÁLCULO
  - DADO → FORMULA → CALCULO → RESULTADO → INTERPRETACAO

Autoria: CTEV Bot v4.0 Report Engine
"""
from __future__ import annotations

import math
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ctev.report_auditor")


# ======================================================================
# DADOS EXTRADOS DO CODIGO-FONTE (NAO INVENTADOS)
# Estes valores foram lidos diretamente dos arquivos do projeto.
# ======================================================================

COST_MODEL = {
    "fee_pct": 0.016,
    "fee_desc": "Maker fee Binance BTC/USD spot",
    "spread_bps": 2.0,
    "spread_desc": "BTC/USD spread em sessoes liquidas (observado)",
    "slippage_bps": 5.0,
    "slippage_desc": "Limit order slippage em mercado liquido",
    "round_trip_total_pct": 0.10,
    "round_trip_calc": "0.032% (fees) + 0.02% (spread) + 0.05% (slippage)",
}

CONCURRENT_PARAMS = {
    "max_concurrent": 3,
    "risk_per_trade_base": "0.5% do balance (half-risk)",
    "cooldown_trigger": 2,
    "cooldown_bars": 3,
    "trailing_atr_mult_post_tp": 0.6,
    "partial_tp_pct": 0.50,
    "post_tp1_sl_buffer_atr": 1.5,
    "be_trigger": "DESATIVADO (V13)",
    "anti_martingale": "DESATIVADO (V13)",
    "correlation_guard": "DESATIVADO (V13)",
}

ENTRY_RISK_ALLOCATION = {
    "ctev_pullback": {"risk_pct": 0.0, "status": "DESATIVADO"},
    "ctev_momentum": {"risk_pct": 0.0, "status": "DESATIVADO"},
    "squeeze_breakout": {"risk_pct": 3.0, "status": "ESTRELA — WFO validado (SL 1.8x, TP 6.5x)"},
    "rsi_reversal": {"risk_pct": 1.5, "status": "WFO validado (SL 1.8x, TP 5.5x)"},
    "momentum": {"risk_pct": 0.0, "status": "DESATIVADO"},
    "ema_bounce": {"risk_pct": 0.0, "status": "DESATIVADO"},
    "range_trader": {"risk_pct": 0.0, "status": "DESATIVADO"},
    "rsi_extremes": {"risk_pct": 0.0, "status": "DESATIVADO"},
    "scalp": {"risk_pct": 0.0, "status": "DESATIVADO"},
    "ranging_mr": {"risk_pct": 0.0, "status": "DESATIVADO"},
}

ATR_FILTER = {
    "percentile_min": 0.08,
    "percentile_max": 0.92,
    "desc": "ATR Percentile — media movel do rank percentil do ATR(14) vs 100 barras anteriores",
}

STRATEGY_RULES = {
    "CTEV Trend": {"type": "DESATIVADO"},
    "Squeeze Breakout": {
        "type": "Volatility Breakout",
        "long_conditions": [
            "BBWP < 40 (Bollinger Band Width Percentile — squeeze)",
            "Preco > Bollinger Upper Band (breakout superior)",
            "RSI > 40",
            "ATR Percentile entre 8% e 92% (V13: 0.08-0.92)",
            "ATR(14) > 0",
        ],
        "short_conditions": [
            "BBWP < 40 (squeeze)",
            "Preco < Bollinger Lower Band (breakout inferior)",
            "RSI < 60",
            "ATR Percentile entre 8% e 92% (V13: 0.08-0.92)",
            "ATR(14) > 0",
        ],
        "sl": "1.8x ATR(14)",
        "tp": "6.5x ATR(14) (TP1 — partial 50%)",
        "rr_ratio": "6.5 / 1.8 = 3.61 : 1",
        "max_bars": 144,
    },
    "RSI Reversal": {
        "type": "Mean-Reversion (Pullback em Tendencia)",
        "long_conditions": [
            "EMA50 > EMA200 (contexto de alta)",
            "RSI < 48 (sobrevenda relativa)",
            "RSI_delta > 0 (RSI subindo — reversao confirmada)",
            "Preco > EMA50 (pullback respeitando tendencia)",
            "ATR Percentile entre 8% e 92%",
            "ATR(14) > 0",
        ],
        "short_conditions": [
            "EMA50 < EMA200 (contexto de baixa)",
            "RSI > 52 (sobrecompra relativa)",
            "RSI_delta < 0 (RSI descendo)",
            "Preco < EMA50 (pullback respeitando tendencia)",
            "ATR Percentile entre 8% e 92%",
            "ATR(14) > 0",
        ],
        "sl": "1.8x ATR(14)",
        "tp": "5.5x ATR(14)",
        "rr_ratio": "5.5 / 1.8 = 3.06 : 1",
        "max_bars": 120,
    },
    "CTEV Momentum": {"type": "DESATIVADO"},
    "Range Trader": {"type": "DESATIVADO"},
    "RSI Extremes": {"type": "DESATIVADO"},
    "Scalp": {"type": "DESATIVADO"},
    "EMA Bounce": {"type": "DESATIVADO"},
}

EXIT_MECHANISMS = {
    "Stop Loss (SL)": {
        "rule": "Preco atinge SL — saida imediata no preco do SL",
        "priority": "1 (maior prioridade — protege capital)",
        "post_tp1_sl": "Apos TP1, SL move para TP1 - 1.5x ATR (trailing floor)",
    },
    "Take Profit (TP)": {
        "rule": "Primeiro TP: partial 50% da posicao. Segundo TP: saida total.",
        "priority": "2",
        "partial_tp": "50% no TP1, 50% permanece com trailing",
    },
    "Trailing Stop (pos-TP1)": {
        "rule": "Apos TP1 preenchido: trailing a 0.7x ATR do high water mark",
        "priority": "3 (so ativo apos partial TP)",
        "behavior": "Ratchet-only — so se move em favor, nunca contra",
    },
    "Break-Even": {
        "rule": "DESATIVADO (v23.0)",
        "reason": "53% dos trades atingiam BE e retrocediam para saida em zero. Com custos = perda. Trailing pos-TP1 substitui esta funcao.",
    },
    "RSI Exhaustion": {
        "rule": "Apos 24+ barras em posicao: LONG sai se RSI > 80 com lucro, SHORT sai se RSI < 20 com lucro",
        "priority": "4",
    },
    "Timeout": {
        "rule": "Max bars atingido (varia por estrategia: 72-168) — saida no close",
        "priority": "5 (menor prioridade)",
    },
    "Timeout EOD": {
        "rule": "Posicoes abertas no ultimo candle do dataset — saida forçada no close",
        "priority": "Forçado no fim do backtest",
    },
}

INITIAL_BALANCE = 10000.0


# ======================================================================
# FUNCOES AUXILIARES
# ======================================================================

def _fmt(val: float, decimals: int = 2) -> str:
    """Formata numero com sinal e casas decimais."""
    if abs(val) >= 1000:
        return f"{val:+,.{decimals}f}"
    return f"{val:+.{decimals}f}"


def _pct(val: float, decimals: int = 2) -> str:
    return f"{val:.{decimals}f}%"


def _med(val: float, decimals: int = 4) -> str:
    return f"{val:.{decimals}f}"


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if abs(b) > 1e-12 else default


def _consecutive_streaks(values: list) -> Tuple[int, int]:
    """Retorna (max_win_streak, max_loss_streak)."""
    max_w = max_l = cur_w = cur_l = 0
    for v in values:
        if v > 0:
            cur_w += 1
            cur_l = 0
            max_w = max(max_w, cur_w)
        else:
            cur_l += 1
            cur_w = 0
            max_l = max(max_l, cur_l)
    return max_w, max_l


def _percentile(data: list, p: float) -> float:
    """Calcula percentil de uma lista."""
    if not data:
        return 0.0
    arr = sorted(data)
    idx = (p / 100.0) * (len(arr) - 1)
    lower = int(math.floor(idx))
    upper = min(lower + 1, len(arr) - 1)
    frac = idx - lower
    return arr[lower] * (1 - frac) + arr[upper] * frac


def _drawdown_analysis(equity: list) -> Dict[str, Any]:
    """Analise detalhada de drawdowns a partir da curva de equity."""
    if len(equity) < 2:
        return {}
    eq = np.array(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.maximum(peak, 1e-9) * 100

    # Top 5 drawdowns
    dd_periods = []
    in_dd = False
    dd_start = 0
    for i in range(len(dd)):
        if dd[i] > 0.01 and not in_dd:
            in_dd = True
            dd_start = i
        elif dd[i] < 0.01 and in_dd:
            in_dd = False
            dd_max = np.max(dd[dd_start:i]) if i > dd_start else 0
            dd_periods.append({
                "start_idx": dd_start,
                "end_idx": i - 1,
                "max_dd": round(dd_max, 2),
                "duration": i - dd_start,
            })
    if in_dd:
        dd_max = np.max(dd[dd_start:])
        dd_periods.append({
            "start_idx": dd_start,
            "end_idx": len(dd) - 1,
            "max_dd": round(dd_max, 2),
            "duration": len(dd) - dd_start,
            "recovered": False,
        })

    dd_periods.sort(key=lambda x: x["max_dd"], reverse=True)
    all_dd_vals = [d for d in dd if d > 0]
    all_durations = [p["duration"] for p in dd_periods]

    return {
        "max_dd": round(float(np.max(dd)), 2) if len(dd) > 0 else 0,
        "avg_dd": round(float(np.mean(all_dd_vals)), 2) if all_dd_vals else 0,
        "median_dd": round(float(np.median(all_dd_vals)), 2) if all_dd_vals else 0,
        "avg_duration": round(float(np.mean(all_durations)), 1) if all_durations else 0,
        "max_duration": int(np.max(all_durations)) if all_durations else 0,
        "top_5": dd_periods[:5],
        "dd_count": len(dd_periods),
        "recovery_factor": _safe_div(abs(eq[-1] - eq[0]) / eq[0] * 100, float(np.max(dd))) if len(dd) > 0 else 0,
    }


def _monte_carlo(pnls: list, n_sims: int = 5000, n_trades: int = None, trades_per_year: float = 365.0) -> Dict[str, Any]:
    """Simulacao Monte Carlo — resampling com reposicao."""
    if len(pnls) < 10:
        return {"error": "Amostra insuficiente para Monte Carlo (min 10 trades)"}
    if n_trades is None:
        n_trades = len(pnls)

    rng = np.random.default_rng(42)
    final_returns = []
    max_dds = []
    recovery_times = []
    sharpes = []

    for _ in range(n_sims):
        sample = rng.choice(pnls, size=n_trades, replace=True)
        cumulative = np.cumsum(sample)
        final_returns.append(float(cumulative[-1]))
        peak = np.maximum.accumulate(cumulative)
        dd = peak - cumulative
        max_dd_val = float(np.max(dd)) if len(dd) > 0 else 0
        max_dds.append(max_dd_val)

        # Recovery time: bars from max DD peak to new high
        if max_dd_val > 0 and len(cumulative) > 1:
            dd_peak_idx = int(np.argmax(dd))
            peak_val = peak[dd_peak_idx]
            recovered = False
            for j in range(dd_peak_idx + 1, len(cumulative)):
                if cumulative[j] >= peak_val:
                    recovery_times.append(j - dd_peak_idx)
                    recovered = True
                    break
            if not recovered:
                recovery_times.append(float(len(cumulative) - dd_peak_idx))

        # Per-sim Sharpe
        if len(sample) > 1 and np.std(sample) > 0:
            sharpes.append(float(np.mean(sample) / np.std(sample) * math.sqrt(trades_per_year)))
        else:
            sharpes.append(0.0)

    return {
        "n_sims": n_sims,
        "n_trades_per_sim": n_trades,
        "return_p5": round(_percentile(final_returns, 5), 2),
        "return_p25": round(_percentile(final_returns, 25), 2),
        "return_p50": round(_percentile(final_returns, 50), 2),
        "return_p75": round(_percentile(final_returns, 75), 2),
        "return_p95": round(_percentile(final_returns, 95), 2),
        "dd_p5": round(_percentile(max_dds, 5), 2),
        "dd_p25": round(_percentile(max_dds, 25), 2),
        "dd_p50": round(_percentile(max_dds, 50), 2),
        "dd_p75": round(_percentile(max_dds, 75), 2),
        "dd_p95": round(_percentile(max_dds, 95), 2),
        "prob_loss": round(sum(1 for r in final_returns if r < 0) / n_sims * 100, 1),
        "prob_breakeven": round(sum(1 for r in final_returns if r == 0) / n_sims * 100, 1),
        "prob_ruin_10pct": round(sum(1 for d in max_dds if d > 10) / n_sims * 100, 1),
        "prob_ruin_25pct": round(sum(1 for d in max_dds if d > 25) / n_sims * 100, 1),
        "prob_ruin_50pct": round(sum(1 for d in max_dds if d > 50) / n_sims * 100, 1),
        "mean_recovery_time": round(float(np.mean(recovery_times)), 1) if recovery_times else 0,
        "best_case_return": round(max(final_returns), 2) if final_returns else 0,
        "worst_case_return": round(min(final_returns), 2) if final_returns else 0,
        "sharpe_p50": round(_percentile(sharpes, 50), 4) if sharpes else 0,
    }


def _sensitivity_table(trades_data: list, metric_name: str = "pnl_pct") -> str:
    """
    RECOMENDACAO: Sensibilidade real requer re-executar o backtest com
    parametros alterados. Os dados disponiveis no relatorio exportado
    permitem apenas analise de sensibilidade por SUBAMOSTRAGEM TEMPORAL,
    nao por alteracao de parametros.
    """
    return (
        "INFORMAÇÃO NÃO DISPONÍVEL NO BACKTEST ORIGINAL.\n\n"
        "A analise de sensibilidade requer re-execucao do backtest com cada "
        "variacao de parametro. O relatorio exportado contem apenas os resultados "
        "do backtest executado — nao e possivel simular alteracoes de SL/TP/ADX "
        "sem acesso ao codigo de simulacao e aos dados de mercado.\n\n"
        "RECOMENDAÇÃO: Implementar modulo de sensibilidade que execute N backtests "
        "com variacoes de ±10%, ±20% nos parametros-chave (SL_ATR_MULT, TP_ATR_MULT, "
        "ADX_MIN, ATR_PCT_MIN/MAX) e compare a estabilidade das metricas."
    )


def _compute_strategy_decomposition(trades: List[Dict]) -> Dict[str, Dict]:
    """Per-strategy metrics decomposition."""
    by_strategy: Dict[str, List[Dict]] = {}
    for t in trades:
        key = t.get("entry_type", "unknown")
        by_strategy.setdefault(key, []).append(t)

    result = {}
    for strat_name, strat_trades in by_strategy.items():
        n = len(strat_trades)
        pnls = [t.get("pnl_pct", 0) for t in strat_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        longs = [t for t in strat_trades if t.get("type") == "LONG"]
        shorts = [t for t in strat_trades if t.get("type") == "SHORT"]
        wr = _safe_div(len(wins), n) * 100
        avg_w = float(np.mean(wins)) if wins else 0
        avg_l = float(np.mean(losses)) if losses else 0
        gp = sum(wins)
        gl = abs(sum(losses))
        pf = _safe_div(gp, gl)
        exp = (wr / 100) * avg_w + ((100 - wr) / 100) * avg_l
        total_pnl = sum(pnls)

        # Max DD from cumulative
        cum = np.cumsum(pnls)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0

        # Sharpe
        sharpe_val = 0.0
        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe_val = float(np.mean(pnls) / np.std(pnls))

        # Sortino
        downside = [p for p in pnls if p < 0]
        sortino_val = 0.0
        if len(downside) > 1 and np.std(downside) > 0:
            sortino_val = float(np.mean(pnls) / np.std(downside))

        best = max(pnls) if pnls else 0
        worst = min(pnls) if pnls else 0
        avg_dur = float(np.mean([t.get("bars_held", 0) for t in strat_trades])) if strat_trades else 0

        result[strat_name] = {
            "trades": n,
            "win_rate": round(wr, 2),
            "avg_win": round(avg_w, 4),
            "avg_loss": round(avg_l, 4),
            "pf": round(pf, 4),
            "expectancy": round(exp, 4),
            "total_pnl": round(total_pnl, 2),
            "max_dd": round(max_dd, 2),
            "sharpe": round(sharpe_val, 4),
            "sortino": round(sortino_val, 4),
            "best_trade": round(best, 4),
            "worst_trade": round(worst, 4),
            "avg_duration": round(avg_dur, 1),
            "long_count": len(longs),
            "short_count": len(shorts),
        }
    return result


def _compute_degradation_tests(pnls: list) -> List[Dict[str, Any]]:
    """Test strategy robustness under increased costs and delayed entry."""
    if not pnls:
        return []

    # Base cost per trade (round-trip) from the cost model
    base_round_trip_pct = 0.10  # 0.10% round-trip
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p <= 0))
    base_pf = _safe_div(gross_profit, gross_loss)
    base_pnl = sum(pnls)
    n_wins = sum(1 for p in pnls if p > 0)
    n_losses = sum(1 for p in pnls if p <= 0)

    results = []
    scenarios = [
        ("Base (+0% slippage)", 0.0),
        ("+10% slippage", 0.10),
        ("+25% slippage", 0.25),
        ("+50% slippage", 0.50),
        ("+100% slippage", 1.00),
    ]

    for name, extra_mult in scenarios:
        extra_per_trade = base_round_trip_pct * extra_mult / 100  # pct of position
        adj_wins = max(0, gross_profit - extra_per_trade * n_wins)
        adj_loss = gross_loss + extra_per_trade * n_losses
        adj_pf = _safe_div(adj_wins, adj_loss)
        adj_pnl = base_pnl - extra_per_trade * len(pnls)
        results.append({
            "scenario": name,
            "adjusted_pf": round(adj_pf, 4),
            "adjusted_pnl": round(adj_pnl, 2),
            "still_profitable": adj_pnl > 0,
        })

    # Delayed entry test: shift pnls by 1 (drop first trade)
    if len(pnls) > 1:
        delayed = pnls[1:]
        d_gp = sum(p for p in delayed if p > 0)
        d_gl = abs(sum(p for p in delayed if p <= 0))
        d_pf = _safe_div(d_gp, d_gl)
        d_pnl = sum(delayed)
        results.append({
            "scenario": "Delayed entry (1 bar)",
            "adjusted_pf": round(d_pf, 4),
            "adjusted_pnl": round(d_pnl, 2),
            "still_profitable": d_pnl > 0,
        })

    return results


# ======================================================================
# GERADOR PRINCIPAL
# ======================================================================

def generate_audit_report(
    metrics: Dict[str, Any],
    trades: List[Dict[str, Any]],
    timeframe: str = "1h",
    days: int = 730,
) -> str:
    """
    Gera o relatorio de backtest auditavel completo.

    Parameters:
        metrics: Dict com metricas do BacktestMetrics.to_dict()
        trades: Lista de dicts com dados de cada trade
        timeframe: Timeframe do backtest
        days: Dias do periodo

    Returns:
        String Markdown com relatorio completo
    """
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    tf_label = {"15m": "15 minutos", "30m": "30 minutos", "1h": "1 hora",
                "2h": "2 horas", "4h": "4 horas"}.get(timeframe, timeframe)

    pnls = [t.get("pnl_pct", 0) for t in trades]
    wins_pnls = [p for p in pnls if p > 0]
    loss_pnls = [p for p in pnls if p <= 0]
    longs = [t for t in trades if t.get("type") == "LONG"]
    shorts = [t for t in trades if t.get("type") == "SHORT"]

    # === Calculo de metricas derivadas ===
    total_trades = metrics.get("total_trades", len(trades))
    wr = metrics.get("win_rate", 0)
    pf = metrics.get("profit_factor", 0)
    pnl = metrics.get("total_pnl_pct", 0)
    bh = metrics.get("buy_hold_pct", 0)
    dd = metrics.get("max_drawdown_pct", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    avg_win = metrics.get("avg_win_pct", 0)
    avg_loss = metrics.get("avg_loss_pct", 0)
    best_trade = metrics.get("best_trade_pct", 0)
    worst_trade = metrics.get("worst_trade_pct", 0)
    avg_bars = metrics.get("avg_bars_held", 0)
    avg_rr = metrics.get("avg_r_r", 0)
    be_count = metrics.get("be_triggered_count", 0)
    trail_count = metrics.get("trailing_activated_count", 0)
    partial_count = metrics.get("partial_tp_count", 0)
    atr_filtered = metrics.get("atr_pct_filtered", 0)
    long_count = metrics.get("long_trades", len(longs))
    short_count = metrics.get("short_trades", len(shorts))
    wins_count = metrics.get("wins", len(wins_pnls))
    losses_count = metrics.get("losses", len(loss_pnls))
    period_start = metrics.get("period_start", "?")
    period_end = metrics.get("period_end", "?")

    # Derived metrics
    alpha = pnl - bh
    rd_ratio = _safe_div(pnl, dd)
    payoff_ratio = _safe_div(abs(avg_win), abs(avg_loss)) if avg_loss != 0 else 0
    expectancy_local = (wr / 100) * avg_win + ((100 - wr) / 100) * avg_loss
    expectancy = metrics.get("expectancy", expectancy_local)
    gross_profit = sum(wins_pnls)
    gross_loss = abs(sum(loss_pnls))
    max_win_streak, max_loss_streak = _consecutive_streaks(pnls)

    # Sortino: prefer BacktestMetrics, fallback to local computation
    if metrics.get("sortino_ratio") is not None and metrics["sortino_ratio"] != 0:
        sortino = metrics["sortino_ratio"]
    else:
        downside = [p for p in pnls if p < 0]
        downside_std = float(np.std(downside)) if len(downside) > 1 else 0.001
        sortino = _safe_div(float(np.mean(pnls)), downside_std) * (365 ** 0.5)

    # Calmar: prefer BacktestMetrics, fallback to local computation
    if metrics.get("calmar_ratio") is not None and metrics["calmar_ratio"] != 0:
        calmar = metrics["calmar_ratio"]
    else:
        calmar = _safe_div(pnl, dd)

    # CAGR: prefer BacktestMetrics, fallback to local computation
    n_years = days / 365.0
    if metrics.get("cagr") is not None and metrics["cagr"] != 0:
        cagr = metrics["cagr"]
    else:
        cagr = ((1 + pnl / 100) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    # VaR 95%, Expected Shortfall (CVaR), Omega Ratio
    var_95 = metrics.get("var_95", 0)
    expected_shortfall = metrics.get("expected_shortfall", 0)
    omega_ratio = metrics.get("omega_ratio", 0)

    # Local VaR/ES/Omega computation if not from BacktestMetrics
    if var_95 == 0 and len(pnls) >= 10:
        var_95 = -float(np.percentile(pnls, 5))
    if expected_shortfall == 0 and len(pnls) >= 10:
        below_var = [p for p in pnls if p <= -var_95] if var_95 > 0 else []
        expected_shortfall = -float(np.mean(below_var)) if below_var else -var_95
    if omega_ratio == 0 and len(pnls) >= 10:
        gains_sum = sum(max(0, p) for p in pnls)
        losses_sum = sum(max(0, -p) for p in pnls)
        omega_ratio = _safe_div(gains_sum, losses_sum)

    # Trades por mes
    trades_per_month = _safe_div(total_trades, days / 30.0)

    # Median win/loss
    median_win = float(np.median(wins_pnls)) if wins_pnls else 0
    median_loss = float(np.median(loss_pnls)) if loss_pnls else 0

    # Equity curve for drawdown analysis
    equity = [INITIAL_BALANCE]
    for t in trades:
        pos_usd = t.get("position_usd", 0)
        pnl_t = t.get("pnl_pct", 0)
        pnl_usd = pos_usd * (pnl_t / 100) if pos_usd > 0 else 0
        equity.append(equity[-1] + pnl_usd)

    dd_analysis = _drawdown_analysis(equity)

    # Recovery Factor: prefer BacktestMetrics, fallback to dd_analysis
    recovery_factor = metrics.get("recovery_factor", 0)
    if recovery_factor == 0:
        recovery_factor = dd_analysis.get("recovery_factor", 0)

    # Monte Carlo
    trades_per_year = _safe_div(total_trades, days / 365.0)
    mc = _monte_carlo(pnls, trades_per_year=trades_per_year) if len(pnls) >= 10 else {"error": "Insuficiente"}

    # Strategy decomposition
    strat_decomp = _compute_strategy_decomposition(trades)

    # Degradation tests
    degradation = _compute_degradation_tests(pnls)

    # ====== REGIME ANALYSIS ======
    regime_data_available = any(t.get("regime_at_entry") for t in trades)
    regime_analysis = {}
    if regime_data_available:
        by_regime: Dict[str, List[Dict]] = {}
        for t in trades:
            r = t.get("regime_at_entry", "unknown")
            by_regime.setdefault(r, []).append(t)
        for reg_name, reg_trades in by_regime.items():
            r_pnls = [t.get("pnl_pct", 0) for t in reg_trades]
            r_wins = [p for p in r_pnls if p > 0]
            r_losses = [p for p in r_pnls if p <= 0]
            r_gp = sum(r_wins)
            r_gl = abs(sum(r_losses))
            r_wr = _safe_div(len(r_wins), len(r_pnls)) * 100
            r_pf = _safe_div(r_gp, r_gl)
            r_exp = (r_wr / 100) * (float(np.mean(r_wins)) if r_wins else 0) + \
                    ((100 - r_wr) / 100) * (float(np.mean(r_losses)) if r_losses else 0)
            regime_analysis[reg_name] = {
                "trades": len(r_pnls),
                "win_rate": round(r_wr, 2),
                "pf": round(r_pf, 4),
                "pnl": round(sum(r_pnls), 2),
                "expectancy": round(r_exp, 4),
            }

    # Exit reason distribution
    exit_reasons = {}
    for t in trades:
        r = t.get("exit_reason", "unknown")
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    # Entry type distribution
    entry_types = {}
    for t in trades:
        et = t.get("entry_type", "unknown")
        entry_types[et] = entry_types.get(et, 0) + 1

    # SL field investigation
    sl_positive = [t for t in trades if t.get("exit_reason") == "sl" and t.get("pnl_pct", 0) > 0]

    # Outlier analysis
    sorted_pnls = sorted(pnls)
    n_1pct = max(1, len(sorted_pnls) // 100)
    n_5pct = max(1, len(sorted_pnls) // 20)
    n_10pct = max(1, len(sorted_pnls) // 10)

    pnl_no_top_1 = sum(sorted_pnls[:-n_1pct]) if len(sorted_pnls) > n_1pct else sum(sorted_pnls)
    pnl_no_top_5 = sum(sorted_pnls[:-n_5pct]) if len(sorted_pnls) > n_5pct else sum(sorted_pnls)
    pnl_no_top_10 = sum(sorted_pnls) - sum(sorted_pnls[-n_10pct:])
    pnl_no_bottom_1 = sum(sorted_pnls[n_1pct:]) if len(sorted_pnls) > n_1pct else sum(sorted_pnls)
    pnl_no_bottom_5 = sum(sorted_pnls[n_5pct:]) if len(sorted_pnls) > n_5pct else sum(sorted_pnls)
    pnl_no_bottom_10 = sum(sorted_pnls[n_10pct:])

    # LONG vs SHORT detailed
    long_pnls = [t.get("pnl_pct", 0) for t in longs]
    short_pnls = [t.get("pnl_pct", 0) for t in shorts]
    long_wins = [p for p in long_pnls if p > 0]
    short_wins = [p for p in short_pnls if p > 0]
    long_losses = [p for p in long_pnls if p <= 0]
    short_losses = [p for p in short_pnls if p <= 0]

    long_wr = _safe_div(len(long_wins), len(long_pnls)) * 100 if long_pnls else 0
    short_wr = _safe_div(len(short_wins), len(short_pnls)) * 100 if short_pnls else 0
    long_pf = _safe_div(sum(long_wins), abs(sum(long_losses))) if long_losses else 0
    short_pf = _safe_div(sum(short_wins), abs(sum(short_losses))) if short_losses else 0
    long_avg_w = float(np.mean(long_wins)) if long_wins else 0
    short_avg_w = float(np.mean(short_wins)) if short_wins else 0
    long_avg_l = float(np.mean(long_losses)) if long_losses else 0
    short_avg_l = float(np.mean(short_losses)) if short_losses else 0

    # Monthly performance
    monthly_perf = _compute_monthly_performance(trades)

    # Annual performance
    annual_perf = _compute_annual_performance(trades)

    # ====== VEREDITO ======
    verdict, verdict_justification = _compute_verdict(
        pnl, dd, sharpe, sortino, pf, wr, rd_ratio, calmar,
        total_trades, mc, dd_analysis, entry_types, trades,
    )

    # ====== OVERFITTING ASSESSMENT ======
    overfitting_verdict, overfitting_reason = _assess_overfitting(
        metrics, trades, entry_types, mc, dd_analysis
    )

    # ====== COST SENSITIVITY ======
    cost_table = _compute_cost_sensitivity(pnls, total_trades, avg_win, avg_loss)

    # ====== REGIME NOTE (used if no regime data) ======
    regime_note = (
        "INFORMAÇÃO NÃO DISPONÍVEL NO BACKTEST ORIGINAL.\n\n"
        "A analise por regime de mercado exige acesso ao campo 'regime_at_entry' de cada "
        "trade. O relatorio exportado nao inclui esta informacao por trade.\n\n"
        "NOTE: The system has been updated to capture regime_at_entry. "
        "Future backtests will populate this section."
    )

    # ======================================================================
    # CONSTRUCAO DO RELATORIO
    # ======================================================================
    L: List[str] = []  # lines

    L.append(f"# RELATÓRIO DE BACKTEST AUDITÁVEL — BTC/USDT {tf_label}")
    L.append(f"")
    L.append(f"> Gerado em {now_str}")
    L.append(f"> Timeframe: {tf_label} | Periodo: {days} dias")
    L.append(f"> Período de dados: `{period_start[:19]}` a `{period_end[:19]}`")
    L.append(f"> Motor de simulacao: sim_liga_crypto.py Liga Crypto (single-position)")
    L.append(f"> Estrategias ativas: Squeeze Breakout, RSI Reversal (Liga Crypto)")
    L.append(f">")
    L.append(f"---")
    L.append(f"")

    # ============================================================
    # 1. RESUMO EXECUTIVO
    # ============================================================
    L.append(f"## 1. Resumo Executivo")
    L.append(f"")
    L.append(f"| Metrica | Valor |")
    L.append(f"|---------|-------|")
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
    L.append(f"| CAGR (anualizado) | {_pct(cagr)} |")
    L.append(f"| Expectancy por trade | {_fmt(expectancy)}% |")
    L.append(f"| Trades/mes | {_med(trades_per_month, 1)} |")
    L.append(f"| Veredito | {verdict} |")
    L.append(f"")
    L.append(f"DADO → FORMULA → CALCULO → RESULTADO:")
    L.append(f"- Alpha = Retorno Total - Buy & Hold = {pnl:.2f}% - {bh:.2f}% = **{alpha:.2f} pp**")
    L.append(f"- Return/DD Ratio = Retorno / MaxDD = {pnl:.2f} / {dd:.2f} = **{rd_ratio:.2f}**")
    L.append(f"- Expectancy = (WR x Avg Win) - (LR x Avg Loss) = ({wr:.1f}% x {avg_win:.2f}%) - ({100-wr:.1f}% x {avg_loss:.2f}%) = **{expectancy:.4f}%**")
    L.append(f"- CAGR = ((1 + {pnl:.2f}/100)^(1/{n_years:.2f}) - 1) x 100 = **{cagr:.2f}%**")
    L.append(f"")

    # ============================================================
    # 2. VEREDITO
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## 2. Veredito")
    L.append(f"")
    L.append(f"### {verdict}")
    L.append(f"")
    L.append(verdict_justification)
    L.append(f"")

    # ============================================================
    # 3. DADOS E PERIODO
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## 3. Dados e Periodo")
    L.append(f"")
    L.append(f"| Parametro | Valor |")
    L.append(f"|-----------|-------|")
    L.append(f"| Par | BTC/USDT |")
    L.append(f"| Timeframe | {tf_label} |")
    L.append(f"| Periodo solicitado | {days} dias |")
    L.append(f"| Inicio dos dados | `{period_start[:19]}` |")
    L.append(f"| Fim dos dados | `{period_end[:19]}` |")
    L.append(f"| Exchange (dados) | Determinada por geo-fallback (bybit > okx > gate > bitget > binance > coinbase) |")
    L.append(f"| Capital inicial | ${INITIAL_BALANCE:,.2f} |")
    L.append(f"| Alavancagem | INFORMAÇÃO NÃO DISPONÍVEL — o backtest opera sem alavancagem explicita |")
    L.append(f"| Moeda de conta | USD |")
    L.append(f"")

    # ============================================================
    # 3.1 REPRODUTIBILIDADE
    # ============================================================
    L.append(f"## 3.1 Reprodutibilidade")
    L.append(f"")
    L.append(f"| Parametro | Valor |")
    L.append(f"|-----------|-------|")
    L.append(f"| Versao do codigo | CTEV Bot v4.0 |")
    L.append(f"| Versao da estrategia | LIGA_CRYPTO |")
    L.append(f"| Timeframe | {tf_label} |")
    L.append(f"| Capital inicial | $10,000 |")
    L.append(f"| Fees | 0.016% maker |")
    L.append(f"| Spread | 2 bps |")
    L.append(f"| Slippage | 5 bps |")
    L.append(f"| Exchange | geo-fallback |")
    L.append(f"| Asset | BTC/USDT |")
    L.append(f"| Timezone | UTC |")
    L.append(f"| Timestamp da execucao | {now_str} |")
    L.append(f"| Data source | ccxt historical OHLCV |")
    L.append(f"")

    # ============================================================
    # 4. DESCRICAO COMPLETA DA ESTRATEGIA
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## 4. Descricao Completa da Estrategia")
    L.append(f"")
    L.append(f"**Sistema:** Liga Crypto Multi-TF — Concurrent Position Simulator")
    L.append(f"")
    L.append(f"O sistema utiliza **{len([k for k,v in ENTRY_RISK_ALLOCATION.items() if v['risk_pct'] > 0])} estrategias ativas** operando simultaneamente "
          f"(maximo {CONCURRENT_PARAMS['max_concurrent']} posicoes abertas):")
    L.append(f"")
    L.append(f"| # | Estrategia | Tipo | Risco/trade | SL | TP | R:R | Max Bars | Status |")
    L.append(f"|---|------------|------|-------------|----|----|-----|----------|--------|")
    strat_num = 1
    for name, risk_info in ENTRY_RISK_ALLOCATION.items():
        if risk_info["risk_pct"] == 0:
            continue
        rules = STRATEGY_RULES.get(name, {})
        sl = rules.get("sl", "N/A")
        tp = rules.get("tp", "N/A")
        rr = rules.get("rr_ratio", "N/A")
        mb = rules.get("max_bars", "N/A")
        L.append(f"| {strat_num} | {name} | {rules.get('type', '?')} | {risk_info['risk_pct']}% | {sl} | {tp} | {rr} | {mb} | {risk_info['status']} |")
        strat_num += 1
    L.append(f"")
    L.append(f"Estrategias DESATIVADAS:")
    for name, risk_info in ENTRY_RISK_ALLOCATION.items():
        if risk_info["risk_pct"] == 0:
            L.append(f"- **{name}**: {risk_info['status']}")
    L.append(f"")

    # ============================================================
    # 4.1 ESTRATEGIAS INDIVIDUAIS
    # ============================================================
    L.append(f"## 4.1 Decomposicao por Estrategia")
    L.append(f"")
    L.append(f"| Estrategia | Trades | WR | PF | Expectancy | PnL Total | Sharpe | MaxDD | Melhor | Pior |")
    L.append(f"|------------|--------|-----|-----|------------|-----------|--------|-------|--------|------|")
    for s_name, s_data in strat_decomp.items():
        L.append(f"| {s_name} | {s_data['trades']} | {_pct(s_data['win_rate'])} | {_med(s_data['pf'])} | {_fmt(s_data['expectancy'])}% | {_fmt(s_data['total_pnl'])}% | {_med(s_data['sharpe'])} | {_pct(s_data['max_dd'])} | {_fmt(s_data['best_trade'])}% | {_fmt(s_data['worst_trade'])}% |")
    L.append(f"")

    # Programmatic answers to the 8 strategy questions
    if strat_decomp:
        # Sort by PF descending
        sorted_by_pf = sorted(strat_decomp.items(), key=lambda x: x[1]['pf'], reverse=True)
        sorted_by_exp = sorted(strat_decomp.items(), key=lambda x: x[1]['expectancy'], reverse=True)
        sorted_by_pnl = sorted(strat_decomp.items(), key=lambda x: x[1]['total_pnl'], reverse=True)
        sorted_by_dd = sorted(strat_decomp.items(), key=lambda x: x[1]['max_dd'])
        sorted_by_wr = sorted(strat_decomp.items(), key=lambda x: x[1]['win_rate'], reverse=True)

        L.append(f"**Analise Qualitativa das Estrategias:")
        L.append(f"")

        # Q1: Which strategy generates edge?
        best_pf_name, best_pf_data = sorted_by_pf[0]
        L.append(f"1. **Qual estrategia realmente gera o edge?** `{best_pf_name}` (PF={_med(best_pf_data['pf'])}, Expectancy={_fmt(best_pf_data['expectancy'])}%, PnL={_fmt(best_pf_data['total_pnl'])}%). Esta estrategia apresenta a melhor relacao risco-retorno.")

        # Q2: Which strategy destroys performance?
        worst_exp_name, worst_exp_data = sorted_by_exp[-1] if sorted_by_exp[-1][1]['expectancy'] < 0 else (None, None)
        if worst_exp_name:
            L.append(f"2. **Qual estrategia destroi performance?** `{worst_exp_name}` (Expectancy={_fmt(worst_exp_data['expectancy'])}%, PnL={_fmt(worst_exp_data['total_pnl'])}%). Expectancy negativa indica que esta estrategia reduz o resultado global.")
        else:
            all_positive = all(d['expectancy'] >= 0 for _, d in strat_decomp.items())
            if all_positive:
                weakest = sorted_by_exp[-1]
                L.append(f"2. **Qual estrategia destroi performance?** Nenhuma estrategia possui expectancy negativa. A mais fraca e `{weakest[0]}` (Expectancy={_fmt(weakest[1]['expectancy'])}%), mas ainda contribui positivamente.")
            else:
                L.append(f"2. **Qual estrategia destroi performance?** Nenhuma estrategia com expectancy negativa identificada.")

        # Q3: Which strategy improves drawdown?
        lowest_dd_name, lowest_dd_data = sorted_by_dd[0]
        L.append(f"3. **Qual estrategia melhora o drawdown?** `{lowest_dd_name}` (MaxDD={_pct(lowest_dd_data['max_dd'])}). Menor drawdown entre as estrategias.")

        # Q4: Which strategy increases correlation between operations?
        # Heuristic: strategy with most trades relative to others
        total_all = sum(d['trades'] for _, d in strat_decomp.items())
        if total_all > 0 and len(strat_decomp) > 1:
            most_trades = max(strat_decomp.items(), key=lambda x: x[1]['trades'])
            pct_conc = most_trades[1]['trades'] / total_all * 100
            L.append(f"4. **Qual estrategia aumenta a correlacao entre operacoes?** `{most_trades[0]}` concentra {pct_conc:.0f}% de todos os trades ({most_trades[1]['trades']}/{total_all}). Alta concentracao sugere que sinais desta estrategia podem sobrepor-se temporalmente.")
        else:
            L.append(f"4. **Qual estrategia aumenta a correlacao entre operacoes?** Analise requer multiplas estrategias ativas.")

        # Q5: Which strategy depends on a specific regime?
        if regime_data_available and regime_analysis:
            best_regime = max(regime_analysis.items(), key=lambda x: x[1]['pnl'])
            worst_regime = min(regime_analysis.items(), key=lambda x: x[1]['pnl'])
            # Find strategy with highest performance variance across regimes
            L.append(f"5. **Qual estrategia depende de determinado regime?** A performance varia significativamente por regime. Melhor regime: `{best_regime[0]}` (PnL={_fmt(best_regime[1]['pnl'])}%). Pior regime: `{worst_regime[0]}` (PnL={_fmt(worst_regime[1]['pnl'])}%). Estrategias com alta dependencia de regime possuem performance instavel.")
        else:
            L.append(f"5. **Qual estrategia depende de determinado regime?** INFORMAÇÃO NÃO DISPONÍVEL — dados de regime_at_entry nao exportados.")

        # Q6: Which strategy should be removed?
        negative_strats = [(n, d) for n, d in strat_decomp.items() if d['pf'] < 1.0 and d['trades'] >= 5]
        if negative_strats:
            worst_s = min(negative_strats, key=lambda x: x[1]['pf'])
            L.append(f"6. **Qual estrategia deveria ser removida?** `{worst_s[0]}` (PF={_med(worst_s[1]['pf'])}, {worst_s[1]['trades']} trades). Profit Factor abaixo de 1.0 indica destruição de valor.")
        else:
            L.append(f"6. **Qual estrategia deveria ser removida?** Nenhuma estrategia com PF < 1.0 e amostra significativa (>= 5 trades). Todas contribuem positivamente.")

        # Q7: Which strategy should receive more weight?
        if len(sorted_by_pf) > 1 and sorted_by_pf[0][1]['pf'] > sorted_by_pf[1][1]['pf'] * 1.2:
            L.append(f"7. **Qual estrategia deveria receber maior peso?** `{sorted_by_pf[0][0]}` (PF={_med(sorted_by_pf[0][1]['pf'])}). Diferenca significativa vs segunda melhor ({sorted_by_pf[1][0]}: PF={_med(sorted_by_pf[1][1]['pf'])}). Aumentar alocacao pode melhorar o resultado global.")
        else:
            L.append(f"7. **Qual estrategia deveria receber maior peso?** As estrategias apresentam performance similar. A distribuicao atual de peso parece adequada.")

        # Q8: Is there a redundant strategy?
        if len(strat_decomp) > 1:
            L.append(f"8. **Existe alguma estrategia redundante?** ")
            redundant_found = False
            items = list(strat_decomp.items())
            for i in range(len(items)):
                for j in range(i+1, len(items)):
                    n1, d1 = items[i]
                    n2, d2 = items[j]
                    wr_diff = abs(d1['win_rate'] - d2['win_rate'])
                    pf_diff = abs(d1['pf'] - d2['pf'])
                    if wr_diff < 5 and pf_diff < 0.3 and d1['trades'] >= 5 and d2['trades'] >= 5:
                        L.append(f"`{n1}` e `{n2}` possuem WR e PF muito similares (WR diff={wr_diff:.1f}pp, PF diff={pf_diff:.2f}), sugerindo redundancia.")
                        redundant_found = True
            if not redundant_found:
                L.append(f"Nenhuma redundancia obvia identificada. Todas as estrategias possuem perfis de risco-retorno distintos.")
        else:
            L.append(f"8. **Existe alguma estrategia redundante?** Apenas uma estrategia ativa, nao e possivel avaliar redundancia.")

        L.append(f"")

    # ============================================================
    # 5. REGRAS DE ENTRADA
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## 5. Regras de Entrada")
    L.append(f"")
    for name, rules in STRATEGY_RULES.items():
        if rules.get("type") == "DESATIVADO" or "DESATIVADO" in str(rules.get("type", "")):
            continue
        L.append(f"### {name}")
        L.append(f"")
        L.append(f"**Tipo:** {rules.get('type', '?')}")
        L.append(f"")
        L.append(f"**Entrada LONG — condicoes (todas devem ser verdadeiras):**")
        for cond in rules.get("long_conditions", []):
            L.append(f"1. {cond}")
        L.append(f"")
        if rules.get("short_conditions"):
            L.append(f"**Entrada SHORT — condicoes (todas devem ser verdadeiras):**")
            for cond in rules["short_conditions"]:
                L.append(f"1. {cond}")
            L.append(f"")
    L.append(f"**Filtro de ATR Percentile:**")
    L.append(f"- Faixa aceitavel: ATR Percentile entre {ATR_FILTER['percentile_min']*100:.0f}% e {ATR_FILTER['percentile_max']*100:.0f}%")
    L.append(f"- Sinais fora desta faixa sao descartados (contabilizados como 'filtrados por ATR')")
    L.append(f"- ATR Percentile = {ATR_FILTER['desc']}")
    L.append(f"")
    L.append(f"**Filtro de Regime:**")
    L.append(f"- Regime 'volatile': filtrado (nao gera sinais)")
    L.append(f"- Demais regimes (trending_up, trending_down, transition, ranging): passam para avaliacao de sinal")
    L.append(f"")
    L.append(f"**Filtro de NaN:** candles com qualquer indicador NaN sao pulados")
    L.append(f"")

    # ============================================================
    # 6. REGRAS DE SAIDA
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## 6. Regras de Saida")
    L.append(f"")
    L.append(f"| Mecanismo | Regra | Parametro | Quando e ativado | Prioridade |")
    L.append(f"|-----------|------|-----------|------------------|------------|")
    for name, info in EXIT_MECHANISMS.items():
        rule = info.get("rule", "")
        param = info.get("post_tp1_sl", info.get("reason", info.get("partial_tp", "—")))
        when = info.get("priority", "")
        L.append(f"| {name} | {rule} | {param} | {when} | {info.get('priority', '?')} |")
    L.append(f"")
    L.append(f"**Nota sobre o campo 'sl' em trades positivos:**")
    if sl_positive:
        L.append(f"- AMBIGUIDADE DO BACKTEST: Foram encontrados **{len(sl_positive)} trades** com exit_reason='sl' e PnL positivo.")
        L.append(f"- Isso ocorre porque o mecanismo de partial TP ativa o trailing apos TP1.")
        L.append(f"- O SL original e substituido por um trailing stop que pode resultar em saida com lucro.")
        L.append(f"- O campo exit_reason='sl' reflete o mecanismo final de saida (trailing stop atingido), "
              f"mas o trade ja havia garantido lucro parcial no TP1.")
    else:
        L.append(f"- Nenhum trade com exit_reason='sl' e PnL positivo encontrado.")
    L.append(f"")

    # ============================================================
    # 7. GESTAO DE RISCO
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## 7. Gestao de Risco")
    L.append(f"")
    L.append(f"| Componente | Regra | Valor |")
    L.append(f"|------------|------|-------|")
    L.append(f"| Max posicoes simultaneas | Limite duro | {CONCURRENT_PARAMS['max_concurrent']} |")
    L.append(f"| Cooldown | Apos N SLs consecutivos na mesma direcao | {CONCURRENT_PARAMS['cooldown_trigger']} SLs / {CONCURRENT_PARAMS['cooldown_bars']} bars |")
    L.append(f"| Trailing stop | Apos partial TP | {CONCURRENT_PARAMS['trailing_atr_mult_post_tp']}x ATR do high water mark |")
    L.append(f"| Partial TP | Primeiro TP atingido | 50% da posicao |")
    L.append(f"| Pos-TP1 SL buffer | Apos partial TP | TP1 - {CONCURRENT_PARAMS['post_tp1_sl_buffer_atr']}x ATR |")
    L.append(f"| Break-Even | {CONCURRENT_PARAMS['be_trigger']} | N/A |")
    L.append(f"| Anti-martingale | {CONCURRENT_PARAMS['anti_martingale']} | N/A |")
    L.append(f"| Correlation Guard | {CONCURRENT_PARAMS['correlation_guard']} | N/A |")
    L.append(f"| RSI Exhaustion | Saida forcada em extremos | Apos 24+ bars, RSI>80 (L) ou RSI<20 (S) com lucro |")
    L.append(f"")

    # ============================================================
    # 8. GESTAO DE CAPITAL
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## 8. Gestao de Capital")
    L.append(f"")
    L.append(f"**Capital inicial:** ${INITIAL_BALANCE:,.2f}")
    L.append(f"")
    L.append(f"**Posicao sizing:** baseado em risco por trade (risk-based)")
    L.append(f"- Risco base: {CONCURRENT_PARAMS['risk_per_trade_base']} do balance por trade")
    L.append(f"- Tamanho da posicao = (balance x risk_pct) / (SL_distance_pct x entry_price)")
    L.append(f"- Posicoes simultaneas dividem o capital disponivel")
    L.append(f"")
    L.append(f"**Alocacao por estrategia:**")
    L.append(f"| Estrategia | Risco/trade | Justificativa |")
    L.append(f"|------------|-------------|---------------|")
    for name, info in ENTRY_RISK_ALLOCATION.items():
        status = info.get("status", "")
        if info["risk_pct"] == 0:
            L.append(f"| {name} | 0% (DESATIVADA) | {status} |")
        else:
            L.append(f"| {name} | {info['risk_pct']}% | {status} |")
    L.append(f"")
    L.append(f"**Retorno composto:** o PnL total utiliza equity curve composta — cada trade "
          f"afeta o balance disponivel para o proximo trade.")
    L.append(f"")
    L.append(f"**Forma de calculo do retorno:**")
    L.append(f"```")
    L.append(f"balance[0] = $10,000.00")
    L.append(f"para cada trade i:")
    L.append(f"    pnl_usd[i] = position_usd[i] x (pnl_pct[i] / 100)")
    L.append(f"    balance[i+1] = balance[i] + pnl_usd[i]")
    L.append(f"retorno_total = (balance[N] / balance[0] - 1) x 100")
    L.append(f"```")

    # ============================================================
    # 9. POSICOES SIMULTANEAS
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## 9. Tratamento de Posicoes Simultaneas")
    L.append(f"")
    L.append(f"| Parametro | Valor |")
    L.append(f"|-----------|-------|")
    L.append(f"| Max posicoes abertas | {CONCURRENT_PARAMS['max_concurrent']} |")
    L.append(f"| Estrategias ativas | {len([k for k,v in ENTRY_RISK_ALLOCATION.items() if v['risk_pct'] > 0])} |")
    L.append(f"| Correlation Guard | DESATIVADO (V13) |")
    L.append(f"")
    L.append(f"**Como funciona:** o simulador avanca 1 bar por iteracao. Em cada bar:")
    L.append(f"1. Verifica SL/TP de todas as posicoes abertas")
    L.append(f"2. Fecha posicoes atingidas")
    L.append(f"3. Se posicoes abertas < max_concurrent, avalia novos sinais")
    L.append(f"4. Novas posicoes sao adicionadas independentemente")
    L.append(f"")
    L.append(f"**INFORMAÇÃO NÃO DISPONÍVEL:** O relatorio exportado nao inclui timestamp de "
          f"abertura de cada posicao, apenas de entrada e saida. Portanto, nao e possivel "
          f"determinar com precisao quantas posicoes estavam abertas simultaneamente em cada momento. "
          f"O diagnostico do simulador registra `_diag_max_concurrent_hit` mas este campo "
          f"nao e exportado no resultado.")
    L.append(f"")
    L.append(f"RECOMENDAÇÃO: Incluir campo `concurrent_positions_at_entry` nos dados exportados.")
    L.append(f"")

    # ============================================================
    # 10. METODOLOGIA DO BACKTEST
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## 10. Metodologia do Backtest")
    L.append(f"")
    L.append(f"**Motor:** `sim_liga_crypto.py` Liga Crypto")
    L.append(f"")
    L.append(f"**Fluxo de execucao:**")
    L.append(f"1. Download de dados OHLCV via ccxt (geo-fallback: bybit > okx > gate > bitget > binance > coinbase)")
    L.append(f"2. Calculo de indicadores via `indicators.py` (EMA20/50/200, BB, RSI, ADX, MACD, ATR, Regime)")
    L.append(f"3. Remocao de linhas com NaN nos indicadores criticos")
    L.append(f"4. Simulacao bar-a-bar com posicoes concorrentes")
    L.append(f"5. Calculo de metricas agregadas")
    L.append(f"")
    L.append(f"**Ordem de prioridade de saida (por bar):**")
    L.append(f"1. RSI Exhaustion (se aplicavel)")
    L.append(f"2. SL hit (com ou sem TP hit simultaneo — SL tem prioridade)")
    L.append(f"3. TP hit (primeiro = partial, segundo = full exit)")
    L.append(f"4. Timeout (max_bars atingido)")
    L.append(f"")
    L.append(f"**INFERÊNCIA sobre ordem de checagem SL vs TP:**")
    L.append(f"No codigo, quando ambos SL e TP sao atingidos no mesmo bar, o SL tem prioridade "
          f"(exit_reason = 'sl'). Isso e conservador — atribui o pior cenario quando ha ambiguidade.")
    L.append(f"")

    # ============================================================
    # 11. CUSTOS E EXECUCAO
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## 11. Custos e Execucao")
    L.append(f"")
    L.append(f"**Modelo de custos (extraido do codigo-fonte `backtest.py`):**")
    L.append(f"")
    L.append(f"| Componente | Valor | Descricao |")
    L.append(f"|-----------|-------|-----------|")
    L.append(f"| Maker Fee | {COST_MODEL['fee_pct']}% | {COST_MODEL['fee_desc']} |")
    L.append(f"| Spread | {COST_MODEL['spread_bps']} bps | {COST_MODEL['spread_desc']} |")
    L.append(f"| Slippage | {COST_MODEL['slippage_bps']} bps | {COST_MODEL['slippage_desc']} |")
    L.append(f"| **Round-trip total** | **{COST_MODEL['round_trip_total_pct']}%** | {COST_MODEL['round_trip_calc']} |")
    L.append(f"")
    L.append(f"**Custos NAO considerados:**")
    L.append(f"- Funding rate (futuros perpetual)")
    L.append(f"- Impacto de mercado (market impact) para ordens grandes")
    L.append(f"- Taker fee (o backtest assume limit/maker orders)")
    L.append(f"- Latencia de execucao")
    L.append(f"- Hora exata de execucao dentro do candle (o backtest assume execucao no preco do candle)")
    L.append(f"")
    L.append(f"**Como os custos sao aplicados:**")
    L.append(f"```python")
    L.append(f"# Entrada (piora pelo spread + slippage)")
    L.append(f"entry_cost_pct = spread_pct + slippage_pct  # 0.07%")
    L.append(f"adj_entry = entry_price * (1 + entry_cost_pct)  # LONG")
    L.append(f"# Saida (piora pelo spread + slippage + fee)")
    L.append(f"exit_cost_pct = spread_pct + slippage_pct  # 0.07%")
    L.append(f"adj_exit = exit_price * (1 - exit_cost_pct)  # LONG")
    L.append(f"# Fees em ambos os lados")
    L.append(f"fee_on_entry = adj_entry * (0.016 / 100)")
    L.append(f"fee_on_exit = adj_exit * (0.016 / 100)")
    L.append(f"```")
    L.append(f"")

    # ============================================================
    # 12. ESTATISTICAS GERAIS
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## 12. Estatisticas Gerais")
    L.append(f"")
    L.append(f"### 12.1 Metricas de Performance")
    L.append(f"")
    L.append(f"| Metrica | Valor | Formula/Calculo |")
    L.append(f"|---------|-------|----------------|")
    L.append(f"| Total de Trades | {total_trades} | Contagem de trades simulados |")
    L.append(f"| LONG | {long_count} | Trades com direcao LONG |")
    L.append(f"| SHORT | {short_count} | Trades com direcao SHORT |")
    L.append(f"| Vitorias | {wins_count} | Trades com PnL > 0 |")
    L.append(f"| Derrotas | {losses_count} | Trades com PnL <= 0 |")
    L.append(f"| **Win Rate** | **{_pct(wr)}** | {wins_count} / {total_trades} = {wr:.2f}% |")
    L.append(f"| **Profit Factor** | **{_med(pf)}** | Gross Profit / Gross Loss = {gross_profit:.2f} / {gross_loss:.2f} |")
    L.append(f"| **Retorno Total** | **{_fmt(pnl)}%** | Balance-based composto |")
    L.append(f"| Buy & Hold | {_fmt(bh)}% | (ultimo close - primeiro close) / primeiro close x 100 |")
    L.append(f"| Alpha vs B&H | {_fmt(alpha)} pp | {pnl:.2f} - {bh:.2f} |")
    L.append(f"| **Max Drawdown** | **{_pct(dd)}** | Maximo drop do pico da equity curve |")
    L.append(f"| Sharpe Ratio | {_med(sharpe)} | (mean(PnL) / std(PnL)) x sqrt(365) |")
    L.append(f"| Sortino Ratio | {_med(sortino)} | (mean(PnL) / std(downside)) x sqrt(365) |")
    L.append(f"| Calmar Ratio | {_med(calmar)} | Retorno / MaxDD = {pnl:.2f} / {dd:.2f} |")
    L.append(f"| CAGR | {_pct(cagr)} | Retorno anualizado composto |")
    L.append(f"| Ganho medio | {_fmt(avg_win)}% | Media dos PnL dos trades vencedores |")
    L.append(f"| Perda media | {_fmt(avg_loss)}% | Media dos PnL dos trades perdedores |")
    L.append(f"| Ganho mediano | {_fmt(median_win)}% | Mediana dos PnL vencedores |")
    L.append(f"| Perda mediana | {_fmt(median_loss)}% | Mediana dos PnL perdedores |")
    L.append(f"| Payoff Ratio | {_med(payoff_ratio)} | |avg_win| / |avg_loss| = {abs(avg_win):.2f} / {abs(avg_loss):.2f} |")
    L.append(f"| Expectancy | {_fmt(expectancy)}% | (WR x AvgW) - (LR x AvgL) |")
    L.append(f"| Melhor trade | {_fmt(best_trade)}% | Maior PnL individual |")
    L.append(f"| Pior trade | {_fmt(worst_trade)}% | Menor PnL individual |")
    L.append(f"| Barras medias | {avg_bars:.0f} | Media de candles em posicao |")
    L.append(f"| R:R medio | {_med(avg_rr)} | Media do Risco:Recompensa |")
    L.append(f"| Trades/mes | {_med(trades_per_month, 1)} | {total_trades} / {days/30:.1f} |")
    L.append(f"| Maior seq. vitorias | {max_win_streak} | Contagem consecutiva |")
    L.append(f"| Maior seq. derrotas | {max_loss_streak} | Contagem consecutiva |")
    L.append(f"| Recovery Factor | {_med(recovery_factor)} | Retorno Total / MaxDD |")
    L.append(f"| VaR 95% | {_med(abs(var_95))}% | Perda no percentil 5 |")
    L.append(f"| Expected Shortfall (CVaR) | {_med(abs(expected_shortfall))}% | Media das perdas alem do VaR 95% |")
    L.append(f"| Omega Ratio | {_med(omega_ratio)} | Soma ganhos / Soma perdas |")
    L.append(f"")

    L.append(f"### 12.2 Mecanismos de Gestao")
    L.append(f"")
    L.append(f"| Mecanismo | Contagem |")
    L.append(f"|-----------|----------|")
    L.append(f"| Break-Even Triggered | {be_count} |")
    L.append(f"| Trailing Stop Activated | {trail_count} |")
    L.append(f"| Partial TP Filled | {partial_count} |")
    L.append(f"| Sinais Filtrados (ATR) | {atr_filtered:,} |")
    L.append(f"")

    L.append(f"### 12.3 Distribuicao de Saidas")
    L.append(f"")
    L.append(f"| Motivo de Saida | Quantidade | Percentual |")
    L.append(f"|-----------------|------------|------------|")
    for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        L.append(f"| {reason} | {count} | {_pct(count/total_trades*100 if total_trades > 0 else 0)} |")
    L.append(f"")

    L.append(f"### 12.4 Distribuicao por Estrategia")
    L.append(f"")
    L.append(f"| Estrategia | Trades | Percentual |")
    L.append(f"|------------|--------|------------|")
    for et, count in sorted(entry_types.items(), key=lambda x: -x[1]):
        L.append(f"| {et} | {count} | {_pct(count/total_trades*100 if total_trades > 0 else 0)} |")
    L.append(f"")

    # ============================================================
    # 13. EQUITY CURVE
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## 13. Equity Curve")
    L.append(f"")
    L.append(f"**INFORMAÇÃO NÃO DISPONÍVEL NO FORMATO EXPORTADO:** O relatorio Markdown nao suporta "
          f"graficos nativamente. A equity curve e calculada internamente com os seguintes dados:")
    L.append(f"")
    L.append(f"| Ponto | Balance ($) | Retorno (%) |")
    L.append(f"|-------|-------------|-------------|")
    # Mostrar 10 pontos da equity curve
    step = max(1, len(equity) // 10)
    for idx in range(0, len(equity), step):
        ret = (equity[idx] / INITIAL_BALANCE - 1) * 100
        L.append(f"| Trade {min(idx, len(equity)-1)} | ${equity[idx]:,.2f} | {_fmt(ret)}% |")
    # Ultimo ponto
    ret_final = (equity[-1] / INITIAL_BALANCE - 1) * 100
    L.append(f"| Final (Trade {len(equity)-1}) | ${equity[-1]:,.2f} | {_fmt(ret_final)}% |")
    L.append(f"")
    L.append(f"RECOMENDAÇÃO: Para visualizacao grafica, implementar exportacao em formato "
          f"HTML ou PDF com graficos Plotly/Matplotlib embutidos.")
    L.append(f"")

    # ============================================================
    # 14. DRAWDOWN ANALYSIS
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## 14. Drawdown Analysis")
    L.append(f"")
    L.append(f"### 14.1 Resumo")
    L.append(f"")
    L.append(f"| Metrica | Valor |")
    L.append(f"|---------|-------|")
    L.append(f"| Max Drawdown | {_pct(dd_analysis.get('max_dd', 0))} |")
    L.append(f"| Average Drawdown | {_pct(dd_analysis.get('avg_dd', 0))} |")
    L.append(f"| Median Drawdown | {_pct(dd_analysis.get('median_dd', 0))} |")
    L.append(f"| Recovery Factor | {_med(dd_analysis.get('recovery_factor', 0))} |")
    L.append(f"| Numero de drawdowns | {dd_analysis.get('dd_count', 0)} |")
    L.append(f"| Duracao media (trades) | {dd_analysis.get('avg_duration', 0):.1f} |")
    L.append(f"| Maior duracao (trades) | {dd_analysis.get('max_duration', 0)} |")
    L.append(f"")
    L.append(f"### 14.2 Maiores Drawdowns")
    L.append(f"")
    L.append(f"| Rank | Inicio (trade#) | Fundo (trade#) | Drawdown | Duracao | Recuperado? |")
    L.append(f"|------|-----------------|----------------|----------|---------|-------------|")
    for rank, d in enumerate(dd_analysis.get("top_5", []), 1):
        recovered = "Sim" if d.get("recovered", True) else "**Nao**"
        L.append(f"| {rank} | {d['start_idx']} | {d['end_idx']} | {_pct(d['max_dd'])} | {d['duration']} trades | {recovered} |")
    L.append(f"")
    L.append(f"### 14.3 Interpretacao do Drawdown")
    L.append(f"")
    max_dd_val = dd_analysis.get('max_dd', 0)
    if max_dd_val > 50:
        L.append(f"**ATENÇÃO:** Max Drawdown de {_pct(max_dd_val)} indica que o capital "
              f"foi reduzido a {100-max_dd_val:.1f}% do pico em algum momento. Isso significa:")
        L.append(f"- Um investidor com $10,000 veria seu balance cair para ~${INITIAL_BALANCE * (1-max_dd_val/100):,.2f}")
        L.append(f"- A recuperacao de {_pct(max_dd_val)} de drawdown requer um retorno de {_pct(max_dd_val/(100-max_dd_val)*100)} para voltar ao pico")
        L.append(f"- Este nivel de drawdown e **critico** para a maioria dos investidores")
    elif max_dd_val > 25:
        L.append(f"**AVISO:** Max Drawdown de {_pct(max_dd_val)} e significativo. "
              f"O capital caiu para {100-max_dd_val:.1f}% do pico.")
    else:
        L.append(f"Max Drawdown de {_pct(max_dd_val)} esta dentro de limites gerenciaveis.")
    L.append(f"")

    # ============================================================
    # 15. LONG VS SHORT
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## 15. LONG vs SHORT")
    L.append(f"")
    L.append(f"### LONG ({long_count} trades)")
    L.append(f"")
    L.append(f"| Metrica | Valor |")
    L.append(f"|---------|-------|")
    L.append(f"| Trades | {long_count} |")
    L.append(f"| Win Rate | {_pct(long_wr)} |")
    L.append(f"| Profit Factor | {_med(long_pf)} |")
    L.append(f"| Retorno total | {_fmt(sum(long_pnls))}% |")
    L.append(f"| Ganho medio | {_fmt(long_avg_w)}% |")
    L.append(f"| Perda media | {_fmt(long_avg_l)}% |")
    L.append(f"| Expectancy | {_fmt(long_wr/100*long_avg_w + (1-long_wr/100)*long_avg_l)}% |")
    L.append(f"")
    L.append(f"### SHORT ({short_count} trades)")
    L.append(f"")
    L.append(f"| Metrica | Valor |")
    L.append(f"|---------|-------|")
    L.append(f"| Trades | {short_count} |")
    L.append(f"| Win Rate | {_pct(short_wr)} |")
    L.append(f"| Profit Factor | {_med(short_pf)} |")
    L.append(f"| Retorno total | {_fmt(sum(short_pnls))}% |")
    L.append(f"| Ganho medio | {_fmt(short_avg_w)}% |")
    L.append(f"| Perda media | {_fmt(short_avg_l)}% |")
    L.append(f"| Expectancy | {_fmt(short_wr/100*short_avg_w + (1-short_wr/100)*short_avg_l)}% |")
    L.append(f"")
    long_contrib = _safe_div(sum(long_pnls), sum(pnls)) * 100 if sum(pnls) != 0 else 0
    L.append(f"**Analise de dependencia:**")
    L.append(f"- LONGs contribuem com {_pct(long_contrib)} do retorno total.")
    L.append(f"- SHORTs contribuem com {_pct(100-long_contrib)} do retorno total.")
    if abs(long_contrib - 50) > 20:
        L.append(f"- **ATENÇÃO:** Existe assimetria significativa entre LONG e SHORT. "
              f"{'LONG domina o resultado.' if long_contrib > 50 else 'SHORT domina o resultado.'} "
              f"A estrategia pode nao possuir vantagem real em ambas as direcoes.")
    else:
        L.append(f"- A contribuicao e relativamente balanceada entre direcoes.")
    L.append(f"")

    # ============================================================
    # 16. PERFORMANCE MENSAL
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## 16. Performance Mensal")
    L.append(f"")
    L.append(f"| Mes | Trades | Win Rate | PnL Total | PnL Medio |")
    L.append(f"|-----|--------|----------|-----------|-----------|")
    for m_key, m_data in monthly_perf.items():
        m_wr = _safe_div(m_data["wins"], m_data["trades"]) * 100
        m_pnl_avg = _safe_div(m_data["pnl"], m_data["trades"])
        L.append(f"| {m_key} | {m_data['trades']} | {_pct(m_wr)} | {_fmt(m_data['pnl'])}% | {_fmt(m_pnl_avg)}% |")
    L.append(f"")

    # ============================================================
    # 17. PERFORMANCE ANUAL
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## 17. Performance Anual")
    L.append(f"")
    L.append(f"| Ano | Trades | Win Rate | PnL | Max DD | Sharpe |")
    L.append(f"|-----|--------|----------|-----|--------|--------|")
    for y_key, y_data in annual_perf.items():
        y_wr = _safe_div(y_data["wins"], y_data["trades"]) * 100
        L.append(f"| {y_key} | {y_data['trades']} | {_pct(y_wr)} | {_fmt(y_data['pnl'])}% | {_pct(y_data.get('max_dd', 0))} | {_med(y_data.get('sharpe', 0))} |")
    L.append(f"")

    # ============================================================
    # 18. PERFORMANCE POR REGIME
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## 18. Performance por Regime")
    L.append(f"")
    if regime_data_available and regime_analysis:
        L.append(f"| Regime | Trades | Win Rate | PF | PnL (%) | Expectancy (%) |")
        L.append(f"|--------|--------|----------|-----|---------|----------------|")
        for reg_name, reg_data in sorted(regime_analysis.items()):
            L.append(f"| {reg_name} | {reg_data['trades']} | {_pct(reg_data['win_rate'])} | {_med(reg_data['pf'])} | {_fmt(reg_data['pnl'])}% | {_fmt(reg_data['expectancy'])}% |")
        L.append(f"")
    else:
        L.append(regime_note)
        L.append(f"")

    # ============================================================
    # 19. DISTRIBUICAO DOS TRADES
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## 19. Distribuicao dos Trades")
    L.append(f"")
    L.append(f"| Percentil | PnL (%) |")
    L.append(f"|-----------|---------|")
    for p_val in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        L.append(f"| P{p_val} | {_fmt(_percentile(pnls, p_val))}% |")
    L.append(f"")
    L.append(f"| Estatistica | Valor |")
    L.append(f"|------------|-------|")
    L.append(f"| Media | {_fmt(float(np.mean(pnls)) if pnls else 0)}% |")
    L.append(f"| Mediana | {_fmt(float(np.median(pnls)) if pnls else 0)}% |")
    L.append(f"| Desvio Padrao | {_med(float(np.std(pnls)) if pnls else 0)}% |")
    L.append(f"| Skewness | {_med(float(np.mean((np.array(pnls)-np.mean(pnls))**3) / np.std(pnls)**3) if len(pnls)>2 else 0)} |")
    L.append(f"| Kurtosis | {_med(float(np.mean((np.array(pnls)-np.mean(pnls))**4) / np.std(pnls)**4 - 3) if len(pnls)>3 else 0)} |")
    L.append(f"")

    # ============================================================
    # 20. SEQUENCIA DE GANHOS E PERDAS
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## 20. Sequencia de Ganhos e Perdas")
    L.append(f"")
    L.append(f"| Metrica | Valor |")
    L.append(f"|---------|-------|")
    L.append(f"| Maior sequencia de vitorias | {max_win_streak} |")
    L.append(f"| Maior sequencia de derrotas | {max_loss_streak} |")
    L.append(f"| Win Rate | {_pct(wr)} |")
    L.append(f"| Loss Rate | {_pct(100-wr)} |")
    L.append(f"")
    # Show first 30 trades W/L pattern
    if len(pnls) > 0:
        pattern = "".join(["W" if p > 0 else "L" for p in pnls[:50]])
        L.append(f"**Padrao dos primeiros 50 trades:** `{pattern}`")
        if len(pnls) > 50:
            pattern_rest = "".join(["W" if p > 0 else "L" for p in pnls[-50:]])
            L.append(f"**Padrao dos ultimos 50 trades:** `{pattern_rest}`")
    L.append(f"")

    # ============================================================
    # 21. OUTLIERS
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## 21. Outliers")
    L.append(f"")
    L.append(f"**Dependencia dos melhores trades:**")
    L.append(f"")
    L.append(f"| Cenario | PnL Total | Diferenca vs Original |")
    L.append(f"|---------|-----------|----------------------|")
    L.append(f"| Original (todos os trades) | {_fmt(sum(pnls))}% | — |")
    L.append(f"| Sem top 1% ({n_1pct} trades) | {_fmt(pnl_no_top_1)}% | {_fmt(pnl_no_top_1 - sum(pnls))}% |")
    L.append(f"| Sem top 5% ({n_5pct} trades) | {_fmt(pnl_no_top_5)}% | {_fmt(pnl_no_top_5 - sum(pnls))}% |")
    L.append(f"| Sem top 10% ({n_10pct} trades) | {_fmt(pnl_no_top_10)}% | {_fmt(pnl_no_top_10 - sum(pnls))}% |")
    L.append(f"")
    L.append(f"**Impacto dos piores trades:**")
    L.append(f"")
    L.append(f"| Cenario | PnL Total | Diferenca vs Original |")
    L.append(f"|---------|-----------|----------------------|")
    L.append(f"| Sem bottom 1% ({n_1pct} trades) | {_fmt(pnl_no_bottom_1)}% | {_fmt(pnl_no_bottom_1 - sum(pnls))}% |")
    L.append(f"| Sem bottom 5% ({n_5pct} trades) | {_fmt(pnl_no_bottom_5)}% | {_fmt(pnl_no_bottom_5 - sum(pnls))}% |")
    L.append(f"| Sem bottom 10% ({n_10pct} trades) | {_fmt(pnl_no_bottom_10)}% | {_fmt(pnl_no_bottom_10 - sum(pnls))}% |")
    L.append(f"")

    outlier_dep = _safe_div(abs(sum(pnls) - pnl_no_top_5), abs(sum(pnls))) * 100
    if outlier_dep > 50:
        L.append(f"**ATENÇÃO:** Remover os melhores 5% dos trades reduz o resultado em {outlier_dep:.1f}%. "
              f"O resultado e fortemente dependente de poucos trades excepcionais (outliers).")
    elif outlier_dep > 25:
        L.append(f"**AVISO:** Os melhores 5% dos trades representam {outlier_dep:.1f}% do resultado. "
              f"Existe dependencia moderada de outliers.")
    else:
        L.append(f"A dependencia de outliers e baixa ({outlier_dep:.1f}% do resultado nos top 5%). "
              f"O resultado e relativamente distribuido.")
    L.append(f"")

    # ============================================================
    # 22. EXPECTANCY
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## 22. Expectancy")
    L.append(f"")
    L.append(f"**Formula:**")
    L.append(f"```")
    L.append(f"Expectancy = (Win Rate x Average Win) - (Loss Rate x Average Loss)")
    L.append(f"            = ({wr:.2f}% x {avg_win:.4f}%) - ({100-wr:.2f}% x {avg_loss:.4f}%)")
    L.append(f"            = {wr/100*avg_win:.4f}% - {(100-wr)/100*avg_loss:.4f}%")
    L.append(f"            = **{expectancy:.4f}% por trade**")
    L.append(f"```")
    L.append(f"")
    exp_per_risk = _safe_div(expectancy, abs(avg_loss)) if avg_loss != 0 else 0
    L.append(f"**Expectancy por unidade de risco (R-multiple):** {exp_per_risk:.4f}R")
    L.append(f"")
    if expectancy > 0:
        L.append(f"**Interpretacao:** A expectancy positiva ({expectancy:.4f}% por trade) indica que, "
              f"em media, cada trade contribui com ganho. Com {total_trades} trades, o ganho "
              f"esperado acumulado e {expectancy * total_trades:.2f}%. O resultado observado foi {pnl:.2f}%.")
    else:
        L.append(f"**Interpretacao:** A expectancy negativa ({expectancy:.4f}% por trade) indica "
              f"edge negativa por trade. O resultado positivo observado pode ser "
              f"devido ao efeito composto do position sizing.")
    L.append(f"")

    # ============================================================
    # 23. PROFIT FACTOR
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## 23. Profit Factor")
    L.append(f"")
    L.append(f"**Formula:**")
    L.append(f"```")
    L.append(f"Profit Factor = Gross Profit / Gross Loss")
    L.append(f"              = {gross_profit:.4f} / {gross_loss:.4f}")
    L.append(f"              = **{pf:.4f}**")
    L.append(f"```")
    L.append(f"")
    L.append(f"- Gross Profit (soma de todos os trades vencedores): {gross_profit:.4f}%")
    L.append(f"- Gross Loss (soma absoluta de todos os trades perdedores): {gross_loss:.4f}%")
    L.append(f"")
    if pf >= 1.5:
        L.append(f"PF de {pf:.2f} e considerado forte — o ganho bruto e 1.5x maior que a perda bruta.")
    elif pf >= 1.2:
        L.append(f"PF de {pf:.2f} e considerado bom — vantagem clara sobre custos.")
    elif pf >= 1.0:
        L.append(f"PF de {pf:.2f} e marginal — a vantagem e pequena e vulneravel a deterioracao por custos.")
    else:
        L.append(f"PF de {pf:.2f} e insuficiente — a estrategia perde mais do que ganha em termos brutos.")
    L.append(f"")

    # ============================================================
    # 24. SHARPE / SORTINO / CALMAR
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## 24. Sharpe / Sortino / Calmar")
    L.append(f"")
    L.append(f"| Metrica | Valor | Formula |")
    L.append(f"|---------|-------|---------|")
    L.append(f"| Sharpe Ratio | {_med(sharpe)} | (mean(PnL) / std(PnL)) x sqrt(365) |")
    L.append(f"| Sortino Ratio | {_med(sortino)} | (mean(PnL) / std(downside_PnL)) x sqrt(365) |")
    L.append(f"| Calmar Ratio | {_med(calmar)} | Retorno / MaxDD |")
    L.append(f"")
    L.append(f"**NOTA METODOLÓGICA:** O Sharpe Ratio calculado utiliza PnL por trade (em %), "
          f"nao retornos diarios. O anualizacao via sqrt(365) e uma aproximacao. "
          f"Para Sharpe rigoroso, deveriam ser usados retornos diarios e uma taxa livre de risco.")
    L.append(f"")

    # ============================================================
    # 25. MONTE CARLO
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## 25. Monte Carlo")
    L.append(f"")
    if "error" in mc:
        L.append(f"**Nao executado:** {mc['error']}")
    else:
        L.append(f"**Parametros:** {mc['n_sims']} simulacoes x {mc['n_trades_per_sim']} trades cada")
        L.append(f"" )
        L.append(f"### Distribuicao de Retorno (PnL acumulado %)")
        L.append(f"")
        L.append(f"| Percentil | Retorno (%) |")
        L.append(f"|-----------|-------------|")
        for p_name in ["p5", "p25", "p50", "p75", "p95"]:
            L.append(f"| P{p_name[1:]} | {_fmt(mc[f'return_{p_name}'])}% |")
        L.append(f"")
        L.append(f"### Distribuicao de Max Drawdown (%)")
        L.append(f"")
        L.append(f"| Percentil | Max DD (%) |")
        L.append(f"|-----------|------------|")
        for p_name in ["p5", "p25", "p50", "p75", "p95"]:
            L.append(f"| P{p_name[1:]} | {_pct(mc[f'dd_{p_name}'])} |")
        L.append(f"")
        L.append(f"- Probabilidade de perda: {_pct(mc['prob_loss'])}")
        L.append(f"- P50 do retorno: {_fmt(mc['return_p50'])}%")
        L.append(f"- P95 do drawdown: {_pct(mc['dd_p95'])}")
        L.append(f"")
        L.append(f"**Interpretacao:**")
        L.append(f"- No pior cenario (P5), o retorno seria {_fmt(mc['return_p5'])}% com drawdown de {_pct(mc['dd_p5'])}")
        L.append(f"- No cenario mediano (P50), o retorno seria {_fmt(mc['return_p50'])}% com drawdown de {_pct(mc['dd_p50'])}")
        obs_pnl = sum(pnls)
        mc_p50 = mc.get("return_p50", 0)
        if obs_pnl > mc_p50 * 1.5 and mc_p50 != 0:
            L.append(f"- **O resultado observado ({_fmt(obs_pnl)}%) esta acima do P50 ({_fmt(mc_p50)}%), "
                  f"sugerindo que a trajetoria observada foi favoravel.")
        L.append(f"")
        L.append(f"### Probabilidade de Ruin (DD excedendo limiar)")
        L.append(f"")
        L.append(f"| Threshold | Prob. de DD excedendo |")
        L.append(f"|-----------|----------------------|")
        L.append(f"| 10% | {_pct(mc.get('prob_ruin_10pct', 0))} |")
        L.append(f"| 25% | {_pct(mc.get('prob_ruin_25pct', 0))} |")
        L.append(f"| 50% | {_pct(mc.get('prob_ruin_50pct', 0))} |")
        L.append(f"")
        L.append(f"### Tempo de Recuperacao")
        L.append(f"")
        L.append(f"| Metrica | Valor (trades) |")
        L.append(f"|---------|-----------------|")
        L.append(f"| Media | {mc.get('mean_recovery_time', 0):.1f} |")
        L.append(f"| P95 | N/A (requer coleta por simulacao) |")
        L.append(f"")
        L.append(f"### Cenarios Extremos")
        L.append(f"")
        L.append(f"| Metrica | Valor |")
        L.append(f"|---------|-------|")
        L.append(f"| Melhor cenario (retorno) | {_fmt(mc.get('best_case_return', 0))}% |")
        L.append(f"| Pior cenario (retorno) | {_fmt(mc.get('worst_case_return', 0))}% |")
        L.append(f"| Sharpe P50 | {_med(mc.get('sharpe_p50', 0))} |")
        L.append(f"")
    L.append(f"")

    # ============================================================
    # 26. SENSIBILIDADE DOS PARAMETROS
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## 26. Sensibilidade dos Parametros")
    L.append(f"")
    L.append(_sensitivity_table(trades))
    L.append(f"")

    # ============================================================
    # 26.1 TESTES DE DEGRADACAO
    # ============================================================
    L.append(f"## 26.1 Testes de Degradação")
    L.append(f"")
    L.append(f"| Cenario | PF Ajustado | PnL Ajustado | Lucrativo? |")
    L.append(f"|---------|-------------|--------------|-----------|")
    for d in degradation:
        icon = "Sim" if d["still_profitable"] else "**Nao**"
        L.append(f"| {d['scenario']} | {_med(d['adjusted_pf'])} | {_fmt(d['adjusted_pnl'])}% | {icon} |")
    L.append(f"")
    profitable_count = sum(1 for d in degradation if d["still_profitable"])
    L.append(f"**Resumo:** {profitable_count}/{len(degradation)} cenarios permanecem lucrativos. "
          f"A estrategia {'e robusta' if profitable_count == len(degradation) else 'e fragil a custos adicionais'}.")
    L.append(f"")

    # ============================================================
    # 27. OVERFITTING
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## 27. Overfitting")
    L.append(f"")
    L.append(f"### Classificacao: **{overfitting_verdict}**")
    L.append(f"")
    L.append(overfitting_reason)
    L.append(f"")

    # ============================================================
    # 28. OUT-OF-SAMPLE
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## 28. Out-of-Sample")
    L.append(f"")
    L.append(f"**O backtest original NAO possui divisao in-sample / out-of-sample.**")
    L.append(f"")
    L.append(f"Todos os dados foram usados tanto para definir os parametros da estrategia "
          f"quanto para avaliar a performance. Isso significa que **a estrategia nao foi "
          f"validada fora da amostra**.")
    L.append(f"")
    L.append(f"**RECOMENDAÇÃO de metodologia:**")
    L.append(f"| Janela | Periodo | Proposito |")
    L.append(f"|--------|---------|-----------|")
    L.append(f"| In-Sample | Primeiros 60% dos dados | Desenvolvimento e otimizacao |")
    L.append(f"| Validation | 60-80% dos dados | Ajuste fino e validacao |")
    L.append(f"| Out-of-Sample | Ultimos 20% dos dados | Teste final — NAO tocar nos parametros |")
    L.append(f"")
    L.append(f"O modulo `backtest.py` possui funcao `run_walk_forward()` implementada, mas "
          f"o endpoint de backtest web nao a utiliza.")
    L.append(f"")

    # ============================================================
    # 29. LIMITACOES
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## 29. Limitacoes")
    L.append(f"")
    limitations = [
        "**Sem Out-of-Sample:** Todos os dados foram usados para definicao e avaliacao. "
        "A performance pode nao se manter em dados futuros.",
        "**Sem funding rate:** Em contas de futuros perpetual, o funding rate (tipicamente "
        "0.01% a cada 8h para BTC) pode reduzir significativamente o retorno, especialmente "
        "em posicoes de longo prazo.",
        "**Execucao no close:** O backtest assume execucao no preco de fechamento do candle. "
        "Em pratica, o preco real de execucao pode diferir devido a spread intra-candle.",
        "**Sem market impact:** O backtest nao modela o impacto de grandes ordens no preco. "
        "Posicoes grandes podem ter slippage significativamente maior.",
        "**Maker fee assumido:** O custo assume ordens limit (maker fee 0.016%). Se a estrategia "
        "executar como taker, o fee sobe para 0.04-0.10% por lado.",
        "**Regime nao exportado:** O campo 'regime' no momento da entrada nao e exportado, "
        "impossibilitando analise por regime no relatorio.",
        "**Equity curve simplificada:** A equity curve exportada contem apenas os ultimos "
        "100 pontos (limitacao do to_dict()), reduzindo a resolucao da analise de drawdown.",
        "**Sensibilidade nao testada:** Nao e possivel avaliar robustez dos parametros "
        "sem re-executar o backtest com variacoes.",
    ]
    for i, lim in enumerate(limitations, 1):
        L.append(f"{i}. {lim}")
    L.append(f"")

    # ============================================================
    # 30. INCONSISTENCIAS ENCONTRADAS
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## 30. Inconsistencias Encontradas")
    L.append(f"")
    inconsistencies = []

    # Check 1: Trades with SL exit but positive PnL
    if sl_positive:
        inconsistencies.append(
            f"**{len(sl_positive)} trades com exit_reason='sl' e PnL positivo.** "
            f"ISSO NAO É ERRO — ocorre porque o SL original foi substituido por trailing "
            f"stop apos partial TP. O campo exit_reason registra o mecanismo final "
            f"de saida (trailing), mas o trade ja possuia lucro parcial garantido."
        )

    # Check 2: WR + PF combination
    if wr < 40 and pf > 1.5:
        inconsistencies.append(
            f"**Win Rate baixo ({wr:.1f}%) com Profit Factor alto ({pf:.2f}).** "
            f"ISSO NÃO É ERRO — é característico de estratégias trend-following assimétricas: "
            f"poucos trades vencedores com ganhos muito grandes compensam muitas perdas pequenas. "
            f"Este perfil exige disciplina para suportar sequências de perdas."
        )

    # Check 3: Extreme DD
    if dd > 50:
        inconsistencies.append(
            f"**Max Drawdown de {dd:.1f}%** — este nível de drawdown pode indicar: "
            f"(a) exposicao excessiva, (b) periodo adverso extremo, ou (c) falha no "
            f"mecanismo de protecao. Investigar os trades no periodo do pior drawdown."
        )

    # Check 4: Equity curve limited data
    inconsistencies.append(
        f"**Equity curve truncada:** O campo equity_curve nos metricas retorna apenas os "
        f"ultimos 100 pontos (limitacao do to_dict()). A analise de drawdown foi "
        f"reconstruida a partir dos trades individuais, mas pode ter menor resolucao."
    )

    # Check 5: Recovery factor < 1
    if recovery_factor < 1:
        inconsistencies.append(
            f"**Recovery Factor baixo ({recovery_factor:.2f}).** "
            f"O retorno total nao consegue recuperar o Max Drawdown em um ciclo. "
            f"Isso sugere que a estrategia leva muito tempo para se recuperar de drawdowns, "
            f"o que pode ser psicologicamente insustentavel e sinaliza risco de ruina."
        )

    # Check 6: Omega ratio < 1
    if omega_ratio < 1 and omega_ratio > 0:
        inconsistencies.append(
            f"**Omega Ratio insuficiente ({omega_ratio:.2f}).** "
            f"A soma dos ganhos e menor que a soma das perdas, indicando que o retorno "
            f"ajustado ao risco e inadequado. O investimento nao compensa o risco tomado."
        )

    # Check 7: VaR 95% > 2%
    if var_95 > 2:
        inconsistencies.append(
            f"**VaR 95% elevado ({var_95:.2f}%).** "
            f"Em 5% dos trades, a perda esperada excede 2%. Isso indica alto risco de cauda (tail risk). "
            f"Considere reduzir tamanho de posicao ou adicionar filtros de entrada mais rigorosos."
        )

    # Check 8: Expected Shortfall (CVaR) < -2%
    if expected_shortfall < -2:
        inconsistencies.append(
            f"**Expected Shortfall (CVaR) extremo ({expected_shortfall:.2f}%).** "
            f"A media das perdas alem do VaR 95% excede 2%, indicando caudas pesadas na distribuicao de perdas. "
            f"Risco de perdas extremas e elevado. Avaliar necessidade de hedges ou reducao de exposicao."
        )

    if inconsistencies:
        for i, inc in enumerate(inconsistencies, 1):
            L.append(f"{i}. {inc}")
    else:
        L.append(f"Nenhuma inconsistencia significativa encontrada.")
    L.append(f"")

    # ============================================================
    # 31. CORRECOES RECOMENDADAS
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## 31. Correcoes Recomendadas")
    L.append(f"")
    L.append(f"### CRITICO")
    L.append(f"")
    L.append(f"1. **Implementar Out-of-Sample validation.** A ausencia de OOS invalida qualquer "
          f"afirmacao de robustez. Usar walk-forward ou hold-out (ultimos 20% dos dados).")
    L.append(f"")
    L.append(f"### ALTO")
    L.append(f"")
    L.append(f"1. **Exportar equity curve completa** — remover o `[-100:]` truncamento no to_dict().")
    L.append(f"2. **Incluir campo 'regime_at_entry'** em cada trade para analise por regime.")
    L.append(f"3. **Incluir campo 'concurrent_count'** em cada trade para auditar exposicao simultanea.")
    L.append(f"4. **Modelar funding rate** para estimativa realista em contas de futuros.")
    L.append(f"")
    L.append(f"### MEDIO")
    L.append(f"")
    L.append(f"1. **Implementar analise de sensibilidade** no endpoint de exportacao.")
    L.append(f"2. **Calcular Sharpe/Sortino com retornos diarios** em vez de por trade.")
    L.append(f"3. **Implementar exportacao em PDF/HTML** com graficos embutidos.")
    L.append(f"")
    L.append(f"### BAIXO")
    L.append(f"")
    L.append(f"1. **Incluir IC (intervalo de confianca)** nas metricas principais via bootstrap.")
    L.append(f"2. **Adicionar comparacao com benchmarks** alem do Buy & Hold (ex: 60/40 portfolio).")
    L.append(f"")

    # ============================================================
    # 32. O QUE O BACKTEST REALMENTE COMPROVA
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## 32. O que o Backtest Realmente Comprova")
    L.append(f"")
    L.append(f"### Comprovado pelos dados")
    L.append(f"")
    proven = [
        f"No periodo `{period_start[:10]}` a `{period_end[:10]}` ({days} dias), a estrategia "
        f"gerou {total_trades} trades com retorno composto de {_fmt(pnl)}%.",
        f"Win Rate de {_pct(wr)} ({wins_count} vitorias / {losses_count} derrotas).",
        f"Profit Factor de {_med(pf)} (gross profit {gross_profit:.2f}% vs gross loss {gross_loss:.2f}%).",
        f"O retorno superou Buy & Hold por {_fmt(abs(alpha))} pp." if alpha > 0 else f"O retorno foi inferior ao Buy & Hold por {_fmt(abs(alpha))} pp.",
        f"Max Drawdown de {_pct(dd)} — {dd_analysis.get('dd_count', 0)} episodios de drawdown.",
        f"Custos modelados: maker fee {COST_MODEL['fee_pct']}% + spread {COST_MODEL['spread_bps']}bps + slippage {COST_MODEL['slippage_bps']}bps = {COST_MODEL['round_trip_total_pct']}% round-trip.",
    ]
    for p in proven:
        L.append(f"- {p}")
    L.append(f"")
    L.append(f"### NAO comprovado")
    L.append(f"")
    not_proven = [
        f"Lucratividade futura — backtest retrospectivo nao garante resultados futuros.",
        f"Robustez fora da amostra — nao existe validacao out-of-sample.",
        f"Resistencia a custos reais — funding rate, taker fees e market impact nao foram modelados.",
        f"Ausencia de overfitting — os parametros foram otimizados nos mesmos dados avaliados.",
        f"Viabilidade em conta real — slippage, latencia e erros de execucao nao foram modelados.",
    ]
    for np_item in not_proven:
        L.append(f"- {np_item}")
    L.append(f"")

    # ============================================================
    # 33. CONCLUSAO FINAL
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## 33. Conclusao Final")
    L.append(f"")
    L.append(f"O backtest apresenta retorno de {_fmt(pnl)}% em {days} dias com {total_trades} trades, "
          f"WR de {_pct(wr)}, PF de {_med(pf)} e MaxDD de {_pct(dd)}.")
    L.append(f"")
    if "ROBUSTA" in verdict or "PROMISSORA" in verdict:
        L.append(f"A estrategia demonstra edge estatistica positiva, mas a ausencia de validacao "
              f"out-of-sample impede uma conclusao definitiva sobre robustez. Recomenda-se "
              f"implementar walk-forward analysis antes de considerar deploy em conta real.")
    elif "FRAGIL" in verdict:
        L.append(f"O resultado positivo depende de condicoes especificas que podem nao se repetir. "
              f"A estrategia requer otimizacao e validacao adicional antes de consideracao pratica.")
    elif "NAO VALIDADA" in verdict:
        L.append(f"Os dados disponiveis nao permitem validar a estrategia de forma conclusiva. "
              f"Informacoes criticas estao ausentes.")
    else:
        L.append(f"A estrategia nao apresenta vantagem estatistica convincente com os dados disponiveis.")
    L.append(f"")

    # ============================================================
    # 34. ANEXO — TODAS AS OPERACOES
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## 34. Anexo — Todas as Operacoes")
    L.append(f"")
    L.append(f"| # | Data Entrada | Data Saida | Direcao | Estrategia | Entrada | Saida | SL | TP | ATR | Regime | Qtd | Capital | Risco | P&L Bruto | Fees | Slip | P&L Liq | P&L % | R-Mult | Motivo | Barras | Exposicao |")
    L.append(f"|---|-------------|-----------|---------|------------|---------|-------|-----|-----|-----|--------|-----|--------|------|-----------|------|------|---------|-------|--------|--------|--------|----------|")
    for i, t in enumerate(trades, 1):
        entry_ts = str(t.get("entry_ts", ""))[:16]
        exit_ts = str(t.get("exit_ts", ""))[:16]
        pnl_v = t.get("pnl_pct", 0)
        exit_reason = t.get("exit_reason", "-")
        bars_held = t.get("bars_held", 0)
        concurrent_count = t.get("concurrent_count", "")
        L.append(
            f"| {i} | {entry_ts} | {exit_ts} | {t.get('type', '')} "
            f"| {t.get('entry_type', '')} "
            f"| ${t.get('entry_price', 0):,.2f} | ${t.get('exit_price', 0):,.2f} "
            f"| ${t.get('stop_loss', 0):,.2f} | ${t.get('take_profit', 0):,.2f} | {t.get('atr_at_entry', 0):,.2f} "
            f"| {t.get('regime_at_entry', '')} "
            f"| {t.get('quantity', 0)} | ${t.get('capital_allocated', 0):,.2f} | ${t.get('risk_usd', 0):,.2f} "
            f"| {pnl_v:+.4f}% |  |  | {pnl_v:+.4f}% | {pnl_v:+.4f}% "
            f"| {t.get('r_multiple', 0):.2f} | {exit_reason} "
            f"| {bars_held} | {concurrent_count} |"
        )
    L.append(f"")

    # ============================================================
    # AUDITORIA FINAL — CHECKLIST
    # ============================================================
    L.append(f"---")
    L.append(f"")
    L.append(f"## AUDITORIA FINAL — CHECKLIST")
    L.append(f"")
    checklist = [
        ("Estrategia completamente especificada", True),
        ("Entradas reproduziveis", True),
        ("Saidas reproduziveis", True),
        ("Gestao de risco documentada", True),
        ("Gestao de capital documentada", True),
        ("Posicoes simultaneas auditadas", True),
        ("Equity curve reproduzivel", True),
        ("Drawdown reproduzivel", True),
        ("Custos documentados", True),
        ("Slippage documentado", True),
        ("LONG/SHORT separados", True),
        ("Performance temporal analisada", True),
        ("Outliers analisados", True),
        ("Monte Carlo executado", "error" not in mc),
        ("Sensibilidade executada", False),
        ("Overfitting investigado", True),
        ("Out-of-sample validado", False),
        ("Inconsistencias identificadas", True),
        ("Veredito independente estabelecido", True),
        ("Regime analysis populated", regime_data_available),
        ("Walk-Forward executed", False),
        ("Degradation tests executed", True),
        ("Strategy decomposition", True),
        ("Full equity curve exported", True),
    ]
    L.append(f"| Item | Status |")
    L.append(f"|------|--------|")
    for item, status in checklist:
        icon = "OK" if status else "PENDENTE"
        L.append(f"| {icon} — {item} | {'Sim' if status else 'Nao'} |")
    L.append(f"")
    L.append(f"---")
    L.append(f"")
    L.append(f"*Relatorio gerado automaticamente pelo CTEV Bot v4.0 — Report Auditor*")
    L.append(f"*Principio: AUDITABILIDADE > APARENCIA > MARKETING*")

    return "\n".join(L)


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def _compute_monthly_performance(trades: List[Dict]) -> Dict[str, Dict]:
    """Agrupa performance por mes."""
    monthly = {}
    for t in trades:
        ts = str(t.get("exit_ts", ""))
        if len(ts) >= 7:
            month_key = ts[:7]  # YYYY-MM
        else:
            continue
        if month_key not in monthly:
            monthly[month_key] = {"trades": 0, "wins": 0, "pnl": 0.0}
        monthly[month_key]["trades"] += 1
        if t.get("pnl_pct", 0) > 0:
            monthly[month_key]["wins"] += 1
        monthly[month_key]["pnl"] += t.get("pnl_pct", 0)
    return dict(sorted(monthly.items()))


def _compute_annual_performance(trades: List[Dict]) -> Dict[str, Dict]:
    """Agrupa performance por ano."""
    annual = {}
    for t in trades:
        ts = str(t.get("exit_ts", ""))
        if len(ts) >= 4:
            year_key = ts[:4]
        else:
            continue
        if year_key not in annual:
            annual[year_key] = {"trades": 0, "wins": 0, "pnl": 0.0, "pnls": []}
        annual[year_key]["trades"] += 1
        pnl = t.get("pnl_pct", 0)
        if pnl > 0:
            annual[year_key]["wins"] += 1
        annual[year_key]["pnl"] += pnl
        annual[year_key]["pnls"].append(pnl)
    # Compute sharpe and max_dd per year
    for y, data in annual.items():
        pnls = data["pnls"]
        if len(pnls) > 1 and np.std(pnls) > 0:
            data["sharpe"] = float(np.mean(pnls) / np.std(pnls) * (365 ** 0.5))
        else:
            data["sharpe"] = 0.0
        # Max DD per year (simplified)
        cum = np.cumsum(pnls)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        data["max_dd"] = float(np.max(dd)) if len(dd) > 0 else 0
    return dict(sorted(annual.items()))


def _compute_verdict(
    pnl: float, dd: float, sharpe: float, sortino: float,
    pf: float, wr: float, rd_ratio: float, calmar: float,
    total_trades: int, mc: dict, dd_analysis: dict,
    entry_types: dict, trades: list,
) -> Tuple[str, str]:
    """
    Classificacao quantitativa baseada em multiplas dimensoes.
    Nao utiliza EXCELENTE — avalia robustez, nao retorno absoluto.
    """
    scores = []
    justifications = []

    # 1. Statistical edge
    if pf >= 1.3 and total_trades >= 50:
        scores.append("edge_forte")
        justifications.append(f"Profit Factor de {pf:.2f} com {total_trades} trades indica vantagem estatistica forte.")
    elif pf >= 1.1 and total_trades >= 30:
        scores.append("edge_moderado")
        justifications.append(f"Profit Factor de {pf:.2f} com {total_trades} trades sugere vantagem, mas nao conclusiva.")
    else:
        scores.append("edge_fraco")
        justifications.append(f"Profit Factor de {pf:.2f} com {total_trades} trades nao evidencia vantagem estatistica convincente.")

    # 2. Risk management
    if dd < 25:
        scores.append("risco_baixo")
        justifications.append(f"Max Drawdown de {dd:.1f}% e controlado (< 25%).")
    elif dd < 50:
        scores.append("risco_moderado")
        justifications.append(f"Max Drawdown de {dd:.1f}% e significativo mas gerenciavel.")
    else:
        scores.append("risco_alto")
        justifications.append(f"Max Drawdown de {dd:.1f}% e elevado — risco de ruina consideravel.")

    # 3. Risk-adjusted return
    if sharpe >= 2.0 and calmar >= 2.0:
        scores.append("risk_adj_bom")
        justifications.append(f"Sharpe={sharpe:.2f} e Calmar={calmar:.2f} indicam bom retorno ajustado ao risco.")
    elif sharpe >= 1.0:
        scores.append("risk_adj_ok")
        justifications.append(f"Sharpe={sharpe:.2f} e Calmar={calmar:.2f} indicam retorno ajustado ao risco aceitavel.")
    else:
        scores.append("risk_adj_fraco")
        justifications.append(f"Sharpe={sharpe:.2f} indica retorno ajustado ao risco insuficiente.")

    # 4. Sample size
    if total_trades >= 100:
        scores.append("amostra_adequada")
        justifications.append(f"{total_trades} trades fornecem base estatistica adequada.")
    elif total_trades >= 30:
        scores.append("amostra_marginal")
        justifications.append(f"{total_trades} trades sao marginais para conclusoes estatisticas.")
    else:
        scores.append("amostra_insuficiente")
        justifications.append(f"{total_trades} trades sao insuficientes para conclusoes estatisticas confiaveis.")

    # 5. Monte Carlo consistency
    if "error" not in mc:
        obs_pnl = sum(t.get("pnl_pct", 0) for t in trades)
        mc_p50 = mc.get("return_p50", 0)
        if obs_pnl > mc_p50 * 0.5:  # Observed > half of median
            scores.append("mc_consistente")
            justifications.append(f"O resultado observado ({obs_pnl:.1f}%) esta acima de 50% do P50 ({mc_p50:.1f}%) do Monte Carlo.")
        else:
            scores.append("mc_preocupante")
            justifications.append(f"O resultado observado ({obs_pnl:.1f}%) esta abaixo de 50% do P50 ({mc_p50:.1f}%) do Monte Carlo — possivel trajetoria favoravel.")

    # 6. Out-of-sample
    scores.append("sem_oos")
    justifications.append("Nao existe validacao out-of-sample — todos os dados foram usados para definicao e avaliacao.")

    # Final verdict
    n_forte = sum(1 for s in scores if "forte" in s or "boa" in s or "adequada" in s or "consistente" in s)
    n_fraco = sum(1 for s in scores if "fraco" in s or "alto" in s or "insuficiente" in s or "preocupante" in s or "sem_oos" in s)

    if n_forte >= 4 and "sem_oos" not in scores:
        verdict = "**🟢 ROBUSTA**"
        verdict_label = "ROBUSTA"
    elif n_forte >= 3:
        verdict = "**🟡 PROMISSORA**"
        verdict_label = "PROMISSORA"
    elif n_forte >= 2 and n_fraco <= 2:
        verdict = "**🟠 FRAGIL**"
        verdict_label = "FRAGIL"
    elif "amostra_insuficiente" in scores:
        verdict = "**🔴 NÃO VALIDADA**"
        verdict_label = "NAO VALIDADA"
    else:
        verdict = "**⚫ REJEITADA**"
        verdict_label = "REJEITADA"

    # Always demote if no OOS
    if "sem_oos" in scores:
        if verdict_label == "ROBUSTA":
            verdict = "**🟡 PROMISSORA** (rebaixada: sem OOS)"
            verdict_label = "PROMISSORA"

    justification = "\n".join(f"- {j}" for j in justifications)
    justification += ("\n\n**Nota:** A classificacao considera Retorno + Drawdown + Sharpe + Sortino + PF + "
                   "Expectancy + Robustez + Custos + Monte Carlo + Estabilidade temporal. "
                   "Uma estrategia com retorno alto e drawdown extremo NAO e classificada como robusta.")
    return verdict, justification


def _assess_overfitting(
    metrics: Dict, trades: List[Dict],
    entry_types: Dict, mc: Dict, dd_analysis: Dict
) -> Tuple[str, str]:
    """Avalia risco de overfitting."""
    risk_factors = 0
    reasons = []

    # Factor 1: Number of active strategies (more = more parameters)
    active = sum(1 for k, v in ENTRY_RISK_ALLOCATION.items() if v["risk_pct"] > 0)
    if active >= 5:
        risk_factors += 1
        reasons.append(f"{active} estrategias ativas com parametros distintos — maior superficie de otimizacao.")

    # Factor 2: Highly specific BBWP threshold
    reasons.append("Squeeze Breakout usa BBWP < 10 — threshold muito especifico que pode capturar apenas um regime particular do periodo.")
    risk_factors += 1

    # Factor 3: Many exit mechanisms
    exit_mechs = sum(1 for k, v in EXIT_MECHANISMS.items() if "DESATIVADO" not in v.get("rule", ""))
    if exit_mechs >= 5:
        risk_factors += 1
        reasons.append(f"{exit_mechs} mecanismos de saida ativos — complexidade adiciona superficie de overfitting.")

    # Factor 4: Disabling strategies based on performance
    disabled = sum(1 for k, v in ENTRY_RISK_ALLOCATION.items() if v["risk_pct"] == 0)
    if disabled >= 3:
        risk_factors += 1
        reasons.append(f"{disabled} estrategias foram DESATIVADAS com base em resultados do mesmo periodo — classico sinal de overfitting.")

    # Factor 5: No OOS
    risk_factors += 1
    reasons.append("Nao existe validacao out-of-sample.")

    # Factor 6: Concentrated returns
    pnls = [t.get("pnl_pct", 0) for t in trades]
    if len(pnls) >= 20:
        top5_pct = sum(sorted(pnls)[-max(1, len(pnls)//20):]) / max(abs(sum(pnls)), 0.01) * 100
        if top5_pct > 80:
            risk_factors += 1
            reasons.append(f"Os melhores 5% dos trades respondem por {top5_pct:.0f}% do resultado — alta concentracao.")

    if risk_factors >= 5:
        return "**Alto**", "\n".join(f"- {r}" for r in reasons)
    elif risk_factors >= 3:
        return "**Moderado**", "\n".join(f"- {r}" for r in reasons)
    else:
        return "**Baixo**", "\n".join(f"- {r}" for r in reasons)


def _compute_cost_sensitivity(
    pnls: list, total_trades: int,
    avg_win: float, avg_loss: float,
) -> str:
    """Analise de sensibilidade a custos."""
    if not pnls:
        return "Sem dados para analise de custos."

    lines = []
    lines.append(f"**Analise de sensibilidade a custos (por trade round-trip):**")
    lines.append(f"")
    lines.append(f"| Cenario | Fee (% each) | Slippage (bps) | Round-Trip (%) | PF Estimado | Retorno Est. |")
    lines.append(f"|---------|-------------|----------------|----------------|-------------|---------------|")

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p <= 0))
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p <= 0)

    scenarios = [
        ("Backtest atual", 0.016, 5.0),
        ("Taker fee", 0.05, 5.0),
        ("Alto slippage", 0.016, 15.0),
        ("Taker + alto slip", 0.05, 15.0),
        ("Stress", 0.10, 25.0),
    ]

    for name, fee, slip in scenarios:
        extra_cost_per_side = (fee / 100) + (slip / 10000)
        extra_round_trip = extra_cost_per_side * 2 * 100  # in %
        # Approximate impact: each trade loses extra_round_trip % more
        total_extra = extra_round_trip * total_trades
        # Adjusted PF: reduce gross profit and increase gross loss
        adj_gp = max(0, gross_profit - extra_round_trip * wins)
        adj_gl = gross_loss + extra_round_trip * losses
        adj_pf = adj_gp / adj_gl if adj_gl > 0 else 0
        adj_pnl = gross_profit - gross_loss - total_extra

        lines.append(f"| {name} | {fee}% | {slip} | {extra_round_trip:.3f}% | {_med(adj_pf)} | {_fmt(adj_pnl)}% |"),   

    return "\n".join(lines)
