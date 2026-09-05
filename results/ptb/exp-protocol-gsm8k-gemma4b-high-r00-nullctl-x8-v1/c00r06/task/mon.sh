#!/bin/bash
# summarise GRPO reward trend in blocks of 20 steps
tr '\r' '\n' < "$1" | grep -oE "correctness_reward/mean': [0-9.]*" | grep -oE "[0-9.]+$" | \
python3 -c "
import sys
v=[float(x) for x in sys.stdin]
B=20
print('steps:',len(v))
for i in range(0,len(v),B):
    c=v[i:i+B]
    print(f'  {i:4d}-{i+len(c)-1:4d}: {sum(c)/len(c):.4f}')
"
tr '\r' '\n' < "$1" | grep -oE "'entropy': [0-9.]*" | tail -2
tail -c 90 "$1"
