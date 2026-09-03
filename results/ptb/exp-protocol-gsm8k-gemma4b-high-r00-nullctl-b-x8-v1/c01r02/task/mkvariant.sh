#!/bin/bash
# mkvariant.sh SRC DST [temperature]
set -e
SRC=$1; DST=$2; T=${3:-0.0}
mkdir -p "$DST"
for f in "$SRC"/*; do
  b=$(basename "$f")
  [ "$b" = "generation_config.json" ] && continue
  ln -sf "$(readlink -f "$f")" "$DST/$b"
done
python3 - "$SRC" "$DST" "$T" <<'PY'
import json,sys
src,dst,t=sys.argv[1],sys.argv[2],float(sys.argv[3])
g=json.load(open(src+'/generation_config.json'))
if t==0.0:
    g.pop('top_k',None); g.pop('top_p',None); g['do_sample']=False; g['temperature']=0.0
else:
    g['do_sample']=True; g['temperature']=t
json.dump(g,open(dst+'/generation_config.json','w'),indent=2)
print(g)
PY
