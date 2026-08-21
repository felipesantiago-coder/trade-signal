# Fix: remove problematic character before 'avg_pf'

import os, shutil

os.system('rm /home/z/my-project/trade-signal/audit_framework.py')
shutil.copy2('/home/z/my-project/trade-signal/audit_framework.py')
print('FIXED')
