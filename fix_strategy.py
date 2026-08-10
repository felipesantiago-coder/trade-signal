import re

with open('/home/z/my-project/trade-signal/strategy.py','r') as f:
    c=f.read()

replacements = []

# 1. Docstring
old = 'v13.0 — ACTIVE TRADER MULTI-STRATEGY (4+ trades/semana)'
new = 'v13.2 — CTEV OTIMIZADO FREQUENCIA (sem perda de qualidade)'
replacements.append((old, new))

old = 'Evolucao: v12.0 -> v13.0 (multi-estrategia para frequencia)'
new = 'Evolucao: v10.0 -> v13.2 (EMA proximity amplo para mais sinais)'
replacements.append((old, new))

old = 'v13.0: RELAXED CTEV for more signals'
new = 'v13.2: COMPROMISSO entre v10.0 qualidade e v13.0 frequencia'
replacements.append((old, new))

# 2. ADX
old = 'ADX_MIN = 28.0'
new = 'ADX_MIN = 30.0'
replacements.append((old, new))

# 3. RSI (only need to change if different)
old = 'RSI_LONG_MIN = 68.0'
new = 'RSI_LONG_MAX = 67.0'
replacements.append((old, new))
old = 'RSI_SHORT_MIN = 32.0'
new = 'RSI_SHORT_MIN = 33.0'
replacements.append((old, new))
old = 'RSI_SHORT_MAX = 58.0'
new = 'RSI_SHORT_MAX = 58.0'
replacements.append((old, new))

# 4. EMA proximity
old = 'EMA20_PROXIMITY_PCT = 0.005'
new = 'EMA20_PROXIMITY_PCT = 0.010'
replacements.append((old, new))
old = 'EMA50_PROXIMITY_PCT = 0.008'
new = 'EMA50_PROXIMITY_PCT = 0.015'
replacements.append((old, new))

# 5. Confluence comment
old = '# v13.0: DI Direction ON'
new = '# v13.2: DI Direction ON'
replacements.append((old, new))

# 6. Momentum comment
old = '# v13.1: MOMENTUM CONTINUATION PARAMETERS (usando CTEV SL/TP para consistencia)'
new = '# v13.2: Momentum/MR DESATIVADOS — PROVA: sem edge'
replacements.append((old, new))

c = f.read()
for old, new in replacements:
    c = c.replace(old, new)
with open('/home/z/my-project/trade-signal/strategy.py','w') as f:
    f.write(c)
print('OK, replacements:', len(replacements))
