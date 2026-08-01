# test_mr_top.py
import logging, sys, os
logging.basicConfig(level=logging.ERROR)
import numpy as np
sys.path.insert(0, '/home/z/my-project/trade-signal')
os.environ.setdefault('EXCHANGE_ID', 'bybit')

import backtest as bt
import regime_engine as re
from strategy_profiles import get_profile

def test_config(rsi_l, rsi_s, sl_f, tp_f):
    base = get_profile('1h')
    base_sl = base.sl_atr_mult
    orig_fn = re.get_regime_params
    def patched(regime, confidence, base_profile=None):
        if regime == 'RANGING':
            return {'strategy_type':'mean_reversion','allow_long':True,'allow_short':True,
                    'sl_mult':base_sl*sl_f,'tp_mult':base_sl*tp_f,
                    'rsi_long_range':tuple(rsi_l),'rsi_short_range':tuple(rsi_s),
                    'require_volume':False,'min_confidence':0.2}
        return orig_fn(regime, confidence, base_profile=base_profile)
    re.get_regime_params = patched
    try:
        m, t = bt.run_backtest(symbol='BTC/USDT', timeframe='1h', days=730, advanced=False, regime_switching=True)
    finally:
        re.get_regime_params = orig_fn
    longs = [x for x in t if x.type=='LONG']
    shorts = [x for x in t if x.type=='SHORT']
    lw = len([x for x in longs if x.pnl_pct>0])
    sw = len([x for x in shorts if x.pnl_pct>0])
    lwr = lw/max(len(longs),1)*100
    swr = sw/max(len(shorts),1)*100
    lpnl = sum(x.pnl_pct for x in longs)
    spnl = sum(x.pnl_pct for x in shorts)
    return m, len(longs), len(shorts), lwr, swr, lpnl, spnl

configs = [
    ('CURRENT',       (20,42), (58,80), 1.5, 3.0),
    ('S:RSI55-85',    (20,42), (55,85), 1.5, 3.0),
    ('S:RSI52-85',    (20,42), (52,85), 1.5, 3.0),
    ('S:RSI50-85',    (20,42), (50,85), 1.5, 3.0),
    ('L<40+S52-85',  (20,40), (52,85), 1.5, 3.0),
    ('L<38+S52-85',  (20,38), (52,85), 1.5, 3.0),
    ('L<40+S55-85',  (20,40), (55,85), 1.5, 3.0),
]

print(f'{"Config":>20s} {"N":>3} {"WR%":>5} {"PnL%":>7} {"DD%":>6} {"PF":>5} {"Sharpe":>6} {"L_n":>4} {"S_n":>4} {"L_WR%":>5} {"S_WR%":>5} {"L_PnL":>7} {"S_PnL":>7}')
print('-'*115)

for name, rsi_l, rsi_s, slf, tpf in configs:
    m, ln, sn, lwr, swr, lpnl, spnl = test_config(rsi_l, rsi_s, slf, tpf)
    print(f'{name:>20s} {m.total_trades:3d} {m.win_rate:5.1f} {m.total_pnl_pct:+7.2f} {m.max_drawdown_pct:6.2f} {m.profit_factor:5.2f} {m.sharpe_ratio:6.2f} {ln:4d} {sn:4d} {lwr:5.1f} {swr:5.1f} {lpnl:+7.2f} {spnl:+7.2f}')
