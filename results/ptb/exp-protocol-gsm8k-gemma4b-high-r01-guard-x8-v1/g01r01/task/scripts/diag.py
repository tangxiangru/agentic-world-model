import json, glob, os, sys, re
f = sys.argv[1] if len(sys.argv)>1 else sorted(glob.glob('logs/*gsm8k*.json'), key=os.path.getmtime)[-1]
d = json.load(open(f))
s = d['samples']
c = [x['messages'][-1]['content'] for x in s]
acc = sum(1 for x in s if x['scores']['match']['value']=='C')/len(s)
lastline_ok = sum(1 for x in c if x.strip().split('\n')[-1].startswith('ANSWER:'))
print(f"file={os.path.basename(f)} n={len(s)} accuracy={acc:.4f} "
      f"answer_last_line={lastline_ok}/{len(s)} mean_chars={sum(map(len,c))/len(c):.0f} max={max(map(len,c))}")
if len(sys.argv)>2:
    for x in s:
        if x['scores']['match']['value']!='C':
            print('---- WRONG id', x['id'], 'gold', x['target'])
            print(x['messages'][-1]['content'][-500:])
