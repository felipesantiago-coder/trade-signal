# -*- coding: utf-8 -*-
"""validate_router.py - Valida routing inteligente por timeframe"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy_router import get_strategy_type, get_strategy_label, get_mtf_timeframes


def test_router():
    sep = "=" * 70
    print(f"\n{sep}")
    print("  VALIDACAO DO STRATEGY ROUTER")
    print(sep)

    tfs = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"]
    header = f"{'TF':>5} | {'Tipo':>18} | {'Estrategia':>30} | MTF Confirm"
    print(f"\n{header}")
    print("-" * 90)
    for tf in tfs:
        st = get_strategy_type(tf)
        sl = get_strategy_label(tf)
        mtf = get_mtf_timeframes(tf)
        icon = "  " if st != "disabled" else "X "
        print(f"{tf:>5} | {icon}{st:>18} | {sl:>30} | {mtf}")

    print(f"\n{sep}")
    print("  TESTE DE INTEGRACAO: imports")
    print(sep)

    try:
        from strategy_ema_cross import evaluate_ema_cross, EMA_CROSS_PARAMS
        print(f"  [OK] strategy_ema_cross.evaluate_ema_cross")
        p = EMA_CROSS_PARAMS
        print(f"       SL={p['sl_atr_mult']}x TP={p['tp_atr_mult']}x ADX>{p['adx_min']} cooldown={p['cooldown']}")
    except Exception as e:
        print(f"  [ERRO] strategy_ema_cross: {e}")

    try:
        from strategy_router import evaluate_signal
        print(f"  [OK] strategy_router.evaluate_signal")
    except Exception as e:
        print(f"  [ERRO] strategy_router: {e}")

    try:
        from backtest import _simulate_ema_cross
        print(f"  [OK] backtest._simulate_ema_cross")
    except Exception as e:
        print(f"  [ERRO] backtest._simulate_ema_cross: {e}")

    print("\n  Todos os modulos importados com sucesso!")


if __name__ == "__main__":
    test_router()
