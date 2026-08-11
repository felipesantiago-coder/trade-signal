"""
test_v18_diag.py
-----------
Diagnostico detalhado por entry type.
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

from backtest import run_backtest

PERIODS = [30, 90, 180, 365, 730]


def main():
    for days in PERIODS:
        print(f'\n{"=" * 70}')
        print(f'  Periodo: {days} dias')
        print(f'{"=" * 70}')
        
        m, trades = run_backtest(symbol='BTC/USDT', timeframe='1h', days=days, advanced=True)
        
        # Per entry type stats
        by_type = {}
        for t in trades:
            et = t.entry_type
            if et not in by_type:
                by_type[et] = {'count': 0, 'wins': 0, 'pnl': 0.0, 'pnl_pct': 0.0, 'sl': 0, 'tp': 0, 'timeout': 0}
            by_type[et]['count'] += 1
            by_type[et]['pnl_pct'] += t.pnl_pct
            if t.pnl_pct > 0:
                by_type[et]['wins'] += 1
            if t.exit_reason == 'sl':
                by_type[et]['sl'] += 1
            elif t.exit_reason == 'tp':
                by_type[et]['tp'] += 1
            elif t.exit_reason in ('timeout', 'timeout_eod'):
                by_type[et]['timeout'] += 1
        
        print(f'  Total: {m.total_trades} trades, PnL {m.total_pnl_pct:+.2f}%, WR {m.win_rate:.1f}%')
        print(f'  {"Entry Type":<20} {"Count":>6} {"WR":>7} {"PnL%":>9} {"SL":>5} {"TP":>5} {"TO":>5}')
        print(f'  {"-" * 63}')
        for et, d in sorted(by_type.items(), key=lambda x: x[1]['pnl_pct']):
            wr = d['wins'] / d['count'] * 100 if d['count'] > 0 else 0
            print(f'  {et:<20} {d["count"]:>6d} {wr:>6.1f}% {d["pnl_pct"]:>+8.2f}% {d["sl"]:>5d} {d["tp"]:>5d} {d["timeout"]:>5d}')


if __name__ == '__main__':
    main()
