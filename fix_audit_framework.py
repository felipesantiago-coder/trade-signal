import pathlib

p = pathlib.Path('/home/z/my-project/trade-signal/audit_framework.py')
raw = p.read_bytes()
found = False
for i, line in enumerate(raw.decode('utf-8').split(b'\x0b'), 1):
    if 'avg_pf' in line:
        problem_chars.append(i+1)
        break
if not found:
    print(f'Lines with avg_pf: {problem_chars}')
with open(str(p), 'wb') as f:
    f.write(raw)
    print('Fixed {len(problem_chars)} lines')
