#!/bin/bash
# mkeval.sh CKPT_DIR DST  -> weights from CKPT, tokenizer+greedy gen config from runs/sft1/final
set -e
CK=$1; DST=$2; REF=runs/sft1/final
rm -rf "$DST"; mkdir -p "$DST"
for f in "$CK"/model*.safetensors "$CK"/model.safetensors.index.json "$CK"/config.json; do
  ln -sf "$(readlink -f "$f")" "$DST/$(basename "$f")"
done
for f in tokenizer.json tokenizer.model tokenizer_config.json special_tokens_map.json added_tokens.json; do
  ln -sf "$(readlink -f "$REF/$f")" "$DST/$f"
done
python3 - "$REF" "$DST" <<'PY'
import json,sys
ref,dst=sys.argv[1],sys.argv[2]
g=json.load(open(ref+'/generation_config.json'))
g.pop('top_k',None); g.pop('top_p',None); g['do_sample']=False; g['temperature']=0.0
json.dump(g,open(dst+'/generation_config.json','w'),indent=2)
PY
