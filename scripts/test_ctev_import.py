import sys, os, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from strategy import evaluate_long
    print('OK')
except Exception as e:
    traceback.print_exc()
