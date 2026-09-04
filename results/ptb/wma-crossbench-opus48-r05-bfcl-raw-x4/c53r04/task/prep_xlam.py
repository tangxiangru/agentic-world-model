import json, ast, random, sys
from datasets import load_dataset

random.seed(0)

def map_type(t):
    if t is None: return None
    t = t.strip().lower()
    # strip trailing qualifiers like ", optional", ", default=..."
    base = t.split(',')[0].strip()
    if base.startswith('str'): return 'string'
    if base.startswith('int'): return 'integer'
    if base.startswith('float') or base.startswith('number'): return 'float'
    if base.startswith('bool'): return 'boolean'
    if base.startswith('list') or base.startswith('tuple') or base.startswith('set'): return 'array'
    if base.startswith('dict'): return 'dict'
    return None  # unsupported (Callable, Union, Any...)

def is_optional(t):
    return 'optional' in (t or '').lower()

def coerce(val, bfcl_type):
    """Coerce a possibly-string value to the proper python type."""
    if bfcl_type == 'string':
        if isinstance(val, str): return val
        return str(val)
    if bfcl_type == 'integer':
        if isinstance(val, bool): raise ValueError
        if isinstance(val, int): return val
        if isinstance(val, float):
            if val.is_integer(): return int(val)
            raise ValueError
        return int(str(val).strip())
    if bfcl_type == 'float':
        if isinstance(val, bool): raise ValueError
        if isinstance(val, (int, float)): return float(val)
        return float(str(val).strip())
    if bfcl_type == 'boolean':
        if isinstance(val, bool): return val
        s = str(val).strip().lower()
        if s in ('true','1','yes'): return True
        if s in ('false','0','no'): return False
        raise ValueError
    if bfcl_type in ('array','dict'):
        if isinstance(val, (list, dict)): return val
        v = ast.literal_eval(str(val))
        if bfcl_type == 'array' and not isinstance(v, (list, tuple)): raise ValueError
        if bfcl_type == 'dict' and not isinstance(v, dict): raise ValueError
        return list(v) if isinstance(v, tuple) else v
    raise ValueError

def build_tool(t):
    """Convert xlam tool -> BFCL raw schema. Returns (tool, ptype_map) or None."""
    params = t.get('parameters') or {}
    props = {}
    required = []
    ptypes = {}
    for pname, sp in params.items():
        bt = map_type(sp.get('type'))
        if bt is None:
            return None  # unsupported type -> drop whole tool
        entry = {'type': bt, 'description': sp.get('description','')}
        if not is_optional(sp.get('type')) and 'default' not in sp:
            required.append(pname)
        if 'default' in sp and sp['default'] not in (None, ''):
            # keep default but coerce best-effort
            try:
                entry['default'] = coerce(sp['default'], bt)
            except Exception:
                pass
        props[pname] = entry
        ptypes[pname] = bt
    tool = {
        'name': t['name'],
        'description': t.get('description',''),
        'parameters': {'type':'dict','properties':props,'required':required},
    }
    return tool, ptypes

def main(out_path, max_examples=30000):
    ds = load_dataset('product-science/xlam-function-calling-60k-raw', split='train')
    idx = list(range(len(ds)))
    random.shuffle(idx)
    n=0
    with open(out_path,'w') as f:
        for i in idx:
            if n>=max_examples: break
            r = ds[i]
            try:
                ans = json.loads(r['answers']); tools = json.loads(r['tools'])
            except Exception:
                continue
            if len(ans)!=1:  # single-call only
                continue
            built = {}
            ptype_all = {}
            ok=True
            for t in tools:
                b = build_tool(t)
                if b is None: ok=False; break
                built[t['name']] = b[0]
                ptype_all[t['name']] = b[1]
            if not ok or not built: continue
            call = ans[0]
            fname = call['name']
            if fname not in ptype_all: continue
            ptypes = ptype_all[fname]
            newargs={}
            good=True
            for k,v in (call.get('arguments') or {}).items():
                if k not in ptypes: good=False; break
                try:
                    newargs[k]=coerce(v, ptypes[k])
                except Exception:
                    good=False; break
            if not good: continue
            # tool list: keep all provided tools (usually 1-3) to teach selection
            toollist=list(built.values())
            rec={'tools':toollist,'query':r['query'],'answer':{'name':fname,'arguments':newargs}}
            f.write(json.dumps(rec)+'\n')
            n+=1
    print('wrote',n,'to',out_path)

if __name__=='__main__':
    out=sys.argv[1] if len(sys.argv)>1 else 'xlam_norm.jsonl'
    mx=int(sys.argv[2]) if len(sys.argv)>2 else 30000
    main(out,mx)
