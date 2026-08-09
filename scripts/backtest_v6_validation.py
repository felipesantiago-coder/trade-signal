"""
backtest_v6_validation.py
------------------------
Validacao das mudancas CTEV v6.0 em 730 dias.
Compara v5.0 (antes) vs v6.0 (depois) em modo basic e advanced.
"""

import sys
import os
import logging
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from backtest import (
    run_backtest, simulate_trades, simulate_trades_advanced,
    calculate_metrics, fetch_historical_ohlcv,
)
from indicators import compute_indicators
from strategy import evaluate_long, evaluate_short, SL_ATR_MULT, TP_ATR_MULT, ADX_MIN, ALLOW_TRANSITION
from strategy_profiles import get_profile, PROFILE_STANDARD


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_metrics(m, label=""):
    prefix = f"  [{label}] " if label else "  "
    print(f"{prefix}Total Trades:      {m.total_trades}  (LONG: {m.long_trades}, SHORT: {m.short_trades})")
    print(f"{prefix}Taxa de Acerto:     {m.win_rate:.1f}%")
    print(f"{prefix}Fator de Lucro:     {m.profit_factor:.2f}")
    print(f"{prefix}Resultado Total:   {m.total_pnl_pct:+.2f}%")
    print(f"{prefix}Buy & Hold:        {m.buy_hold_pct:+.2f}%")
    print(f"{prefix}Alpha vs B&H:      {m.total_pnl_pct - m.buy_hold_pct:+.2f} pp")
    print(f"{prefix}Max Drawdown:      {m.max_drawdown_pct:.2f}%")
    print(f"{prefix}Sharpe Ratio:      {m.sharpe_ratio:.2f}")
    print(f"{prefix}Ganho Medio:       {m.avg_win_pct:+.2f}%" if hasattr(m, 'avg_win_pct') else "")
    print(f"{prefix}Perda Media:       {m.avg_loss_pct:+.2f}%" if hasattr(m, 'avg_loss_pct') else "")
    print(f"{prefix}Melhor Trade:      {m.best_trade_pct:+.2f}%")
    print(f"{prefix}Pior Trade:        {m.worst_trade_pct:+.2f}%")
    print(f"{prefix}R:R Medio:         {m.avg_r_r:.2f}")
    print(f"{prefix}Barras Medias:     {m.avg_bars_held:.1f}")
    print(f"{prefix}BE Triggered:      {getattr(m, 'be_triggered_count', 'N/A')}")
    print(f"{prefix}Trailing Activ.:   {getattr(m, 'trailing_activated_count', 'N/A')}")
    print(f"{prefix}Partial TP:        {getattr(m, 'partial_tp_count', 'N/A')}")
    print(f"{prefix}Sinais Filtrados:  {m.atr_pct_filtered}")


def main():
    print_header("CTEV v6.0 — Validacao de Mudancas (730 dias)")
    print(f"\n  Parametros v6.0:")
    print(f"    ADX_MIN:          {ADX_MIN} (era 30)")
    print(f"    ALLOW_TRANSITION: {ALLOW_TRANSITION} (era True)")
    print(f"    SL_ATR_MULT:      {SL_ATR_MULT}x (era 1.50x)")
    print(f"    TP_ATR_MULT:      {TP_ATR_MULT}x (era 3.50x)")
    print(f"    R:R teorico:       {TP_ATR_MULT/SL_ATR_MULT:.1f}:1")
    print(f"    DI_FILTER:        True (NOVO)")
    print(f"    Profile:          {get_profile('1h').name}")
    profile = get_profile('1h')
    print(f"    Profile ADX:      {profile.adx_min}")
    print(f"    Profile TRANS:    {profile.allow_transition}")
    print(f"    Profile SL/TP:     {profile.sl_atr_mult}x/{profile.tp_atr_mult}x")
    print(f"    Profile MAX_BARS:  {profile.max_bars_held}")

    # ==========================================
    # BACKTEST BASICO (SL/TP fixos, sem trailing)
    # ==========================================
    print_header("BACKTEST BASICO (SL/TP fixos, sem trailing/BE)")
    print("\n  Baixando dados e calculando indicadores...")
    m_basic, t_basic = run_backtest(
        symbol="BTC/USDT",
        timeframe="1h",
        days=730,
        atr_pct_min=0.10,  # wider ATR range to match profile
        atr_pct_max=0.90,
        advanced=False,
        regime_switching=False,
    )
    print_metrics(m_basic, "BASICO")

    # Veredicto
    if m_basic.profit_factor >= 1.5 and m_basic.win_rate >= 35:
        verdict = "OTIMO"
    elif m_basic.profit_factor >= 1.0 and m_basic.win_rate >= 30:
        verdict = "ACEITAVEL"
    elif m_basic.total_pnl_pct > m_basic.buy_hold_pct:
        verdict = "SUPERFICIAL"
    else:
        verdict = "FRACO"
    print(f"\n  VEREDICTO: {verdict}")

    # ==========================================
    # BACKTEST AVANCADO (trailing/BE/partial TP)
    # ==========================================
    print_header("BACKTEST AVANCADO (trailing/BE/partial TP)")
    print("\n  Baixando dados e calculando indicadores...")
    m_adv, t_adv = run_backtest(
        symbol="BTC/USDT",
        timeframe="1h",
        days=730,
        atr_pct_min=0.10,
        atr_pct_max=0.90,
        advanced=True,
        regime_switching=False,
    )
    print_metrics(m_adv, "AVANCADO")

    if m_adv.profit_factor >= 1.5 and m_adv.win_rate >= 35:
        verdict_adv = "OTIMO"
    elif m_adv.profit_factor >= 1.0 and m_adv.win_rate >= 30:
        verdict_adv = "ACEITAVEL"
    elif m_adv.total_pnl_pct > m_adv.buy_hold_pct:
        verdict_adv = "SUPERFICIAL"
    else:
        verdict_adv = "FRACO"
    print(f"\n  VEREDICTO: {verdict_adv}")

    # ==========================================
    # COMPARACAO v6.0 vs v5.0 (baseline anterior)
    # ==========================================
    print_header("COMPARACAO: v6.0 vs v5.0 (baseline)")
    v5_baseline = {
        "total_trades": 323,
        "win_rate": 30.0,
        "profit_factor": 0.27,
        "total_pnl_pct": -238.72,
        "buy_hold_pct": 6.06,
        "max_drawdown_pct": 246.79,
        "sharpe_ratio": -9.65,
    }

    best = m_adv if m_adv.profit_factor > m_basic.profit_factor else m_basic
    mode = "AVANCADO" if m_adv.profit_factor > m_basic.profit_factor else "BASICO"

    print(f"\n  Melhor modo: {mode}")
    print(f"  {'Metrica':<22} {'v5.0':>12} {'v6.0':>12} {'Delta':>12}")
    print(f"  {'-'*60}")
    for key, label in [
        ("total_trades", "Total Trades"),
        ("win_rate", "Taxa Acerto (%)"),
        ("profit_factor", "Fator Lucro"),
        ("total_pnl_pct", "PnL Total (%)"),
        ("max_drawdown_pct", "Max DD (%)"),
        ("sharpe_ratio", "Sharpe Ratio"),
    ]:
        old = v5_baseline[key]
        new = getattr(best, key)
        print(f"  {label:<22} {old:>12.2f} {new:>12.2f} {new-old:>+12.2f}")

    # ==========================================
    # DETALHES DOS TRADES (ultimos 20)
    # ==========================================
    print_header(f"ULTIMOS 20 TRADES ({mode})")
    trades = t_adv if mode == "AVANCADO" else t_basic
    for t in trades[-20:]:
        emoji = "+" if t.pnl_pct > 0 else "-"
        print(f"  [{emoji}] {t.type:5s} | {str(t.entry_ts)[:16]} | "
              f"entry={t.entry_price:>10.2f} exit={t.exit_price:>10.2f} | "
              f"PnL={t.pnl_pct:>+7.2f}% | {t.exit_reason:<8s} | bars={t.bars_held:>3d}")

    # ==========================================
    # ANALISE POR DIRECAO
    # =========================================
    print_header("ANALISE POR DIRECAO")
    longs = [t for t in trades if t.type == "LONG"]
    shorts = [t for t in trades if t.type == "SHORT"]
    for label, subset in [("LONG", longs), ("SHORT", shorts)]:
        if not subset:
            print(f"  {label}: 0 trades")
            continue
        wins = [t for t in subset if t.pnl_pct > 0]
        losses = [t for t in subset if t.pnl_pct <= 0]
        wr = len(wins) / len(subset) * 100
        total_pnl = sum(t.pnl_pct for t in subset)
        avg_win = sum(t.pnl_pct for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl_pct for t in losses) / len(losses) if losses else 0
        sl_count = sum(1 for t in subset if t.exit_reason == "sl")
        tp_count = sum(1 for t in subset if t.exit_reason == "tp")
        trailing_count = sum(1 for t in subset if t.exit_reason == "trailing")
        timeout_count = sum(1 for t in subset if t.exit_reason == "timeout")
        print(f"  {label}: {len(subset)} trades | WR {wr:.1f}% | PnL {total_pnl:+.2f}%")
        print(f"    Avg Win: {avg_win:+.2f}% | Avg Loss: {avg_loss:+.2f}%")
        print(f"    SL: {sl_count} | TP: {tp_count} | Trailing: {trailing_count} | Timeout: {timeout_count}")

    # Salva resultados para referencia
    results = {
        "v6_basic": {
            "total_trades": m_basic.total_trades,
            "win_rate": round(m_basic.win_rate, 1),
            "profit_factor": round(m_basic.profit_factor, 2),
            "total_pnl_pct": round(m_basic.total_pnl_pct, 2),
            "buy_hold_pct": round(m_basic.buy_hold_pct, 2),
            "max_drawdown_pct": round(m_basic.max_drawdown_pct, 2),
            "sharpe_ratio": round(m_basic.sharpe_ratio, 2),
        },
        "v6_advanced": {
            "total_trades": m_adv.total_trades,
            "win_rate": round(m_adv.win_rate, 1),
            "profit_factor": round(m_adv.profit_factor, 2),
            "total_pnl_pct": round(m_adv.total_pnl_pct, 2),
            "buy_hold_pct": round(m_adv.buy_hold_pct, 2),
            "max_drawdown_pct": round(m_adv.max_drawdown_pct, 2),
            "sharpe_ratio": round(m_adv.sharpe_ratio, 2),
        },
    }
    out_path = os.path.join(os.path.dirname(__file__), "..", "download", "v6_validation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Resultados salvos em: {out_path}")
    print("\n  Validacao concluida!\n")


if __name__ == "__main__":
    main()
