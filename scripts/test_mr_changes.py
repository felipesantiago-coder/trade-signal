# test_mr_changes.py
import logging, sys, os
logging.basicConfig(level=logging.ERROR)
import numpy as np
sys.path.insert(0, '/home/z/my-project/trade-signal')
os.environ.setdefault('EXCHANGE_ID', 'bybit')

from backtest import run_backtest
from regime_engine import get_regime_params
from strategy_profiles import get_profile

def test_config(label, rsi_l, rsi_s, sl_f, tp_f):
    """Testa configuracao especifica de RANGING sem modificar arquivos."""
    import regime_engine as re
    import strategy_regime as sr
    
    BASE_SL = 2.12
    base = get_profile('1h')
    base_sl = base.sl_atr_mult
    
    # Patch RANGING params temporariamente
    orig = re.get_regime_params
    def patched(regime, confidence, base_profile=None):
        if regime == 'RANGING':
            return {
                'strategy_type': 'mean_reversion',
                'allow_long': True, 'allow_short': True,
                'sl_mult': base_sl * sl_f,
                'tp_mult': base_sl * tp_f,
                'rsi_long_range': tuple(rsi_l),
                'rsi_short_range': tuple(rsi_s),
                'require_volume': False,
                'min_confidence': 0.2,
            }
        return orig(regime, confidence, base_profile=base_profile)
    
    re.get_regime_params = patched
    m, t = run_backtest(symbol='BTC/USDT', timeframe='1h', days=730, advanced=False, regime_switching=True)
    re.get_regime_params = orig  # restore
    
    longs = [x for x in t if x.type == 'LONG']
    shorts = [x for x in t if x.type == 'SHORT']
    lw = [x for x in longs if x.pnl_pct > 0]
    sw = [x for x in shorts if x.pnl_pct > 0]
    
    lwr = len(lw)/max(len(longs),1)*100
    swr = len(sw)/max(len(shorts),1)*100
    twr = m.win_rate
    
    return {
        'n': m.total_trades, 'wr': twr, 'pnl': m.total_pnl_pct,
        'dd': m.max_drawdown_pct, 'pf': m.profit_factor, 'sharpe': m.sharpe_ratio,
        'longs': len(longs), 'shorts': len(shorts),
        'lwr': lwr, 'swr': swr,
        'l_pnl': sum(x.pnl_pct for x in longs),
        's_pnl': sum(x.pnl_pct for x in shorts),
    }


configs = [
    ('CURRENT',        (20,42), (58,80), 1.5, 3.0),
    # --- LONG improvements ---
    ('L:RSI<40',       (20,40), (58,80), 1.5, 3.0),
    ('L:RSI<38',       (20,38), (58,80), 1.5, 3.0),
    ('L:RSI<35',       (20,35), (58,80), 1.5, 3.0),
    ('L:RSI22-42',     (22,42), (58,80), 1.5, 3.0),
    ('L:SL2.0',        (20,42), (58,80), 2.0, 3.0),
    ('L:SL2.5TP4',     (20,42), (58,80), 2.5, 4.0),
    ('L:SL3.0TP5',     (20,42), (58,80), 3.0, 5.0),
    # --- SHORT frequency ---
    ('S:RSI50-80',     (20,42), (50,80), 1.5, 3.0),
    ('S:RSI55-85',     (20,42), (55,85), 1.5, 3.0),
    ('S:RSI52-82',     (20,42), (52,82), 1.5, 3.0),
    ('S:RSI48-85',     (20,42), (48,85), 1.5, 3.0),
    ('S:RSI55-80',     (20,42), (55,80), 1.5, 3.0),
    # --- Combined ---
    ('L<40+S52-85',   (20,40), (52,85), 1.5, 3.0),
    ('L<38+S52-85',   (20,38), (52,85), 1.5, 3.0),
    ('L22-40+S52-82', (22,40), (52,82), 1.5, 3.0),
    ('L<40+S55-85',   (20,40), (55,85), 1.5, 3.0),
    ('L<40+S50-80SL2',(20,40), (50,80), 2.0, 3.0),
    ('L22-40+S50-85', (22,40), (50,85), 1.5, 3.0),
]

print(f'{"Config":>20s} {"N":>3} {"WR%":>5} {"PnL%":>7} {"DD%":>6} {"PF":>5} {"Sharpe":>6} {"L_n":>4} {"S_n":>4} {"L_WR%":>5} {"S_WR%":>5} {"L_PnL":>7} {"S_PnL":>7}')
print('-' * 115)

results = []
for name, rsi_l, rsi_s, slf, tpf in configs:
    r = test_config(name, rsi_l, rsi_s, slf, tpf)
    results.append((name, r))
    print(f'{name:>20s} {r["n"]:3d} {r["wr"]:5.1f} {r["pnl"]:+7.2f} {r["dd"]:6.2f} {r["pf"]:5.2f} {r["sharpe"]:6.2f} {r["longs"]:4d} {r["shorts"]:4d} {r["lwr"]:5.1f} {r["swr"]:5.1f} {r["l_pnl"]:+7.2f} {r["s_pnl"]:+7.2f}')

# Summary: best by PnL with >= 10 trades
print(f'\nBest by PnL (>=10 trades):')
valid = [(n, r) for n, r in results if r['n'] >= 10]
valid.sort(key=lambda x: x[1]['pnl'], reverse=True)
for n, r in valid[:5]:
    print(f'  {n:>20s}: {r["n"]:3d} trades, WR={r["wr"]:.1f}%, PnL={r["pnl"]:+.2f}%, DD={r["dd"]:.2f}%, Sharpe={r["sharpe"]:.2f}')

print(f'\nBest by Sharpe (>=10 trades):')
valid.sort(key=lambda x: x[1]['sharpe'], reverse=True)
for n, r in valid[:5]:
    print(f'  {n:>20s}: Sharpe={r["sharpe"]:.2f}, PnL={r["pnl"]:+.2f}%, DD={r["dd"]:.2f}%')
