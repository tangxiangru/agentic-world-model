#!/usr/bin/env python3
"""Extract gsm8k-sourced rows from the locally cached OpenMathInstruct-2 train split."""
import glob, json, os, time
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads

CACHE = "/home/ben/hf_cache/datasets/nvidia___open_math_instruct-2/default/0.0.0/469216e3f46f4dacf476b382e192485ea51a143e"
files = sorted(glob.glob(os.path.join(CACHE, "open_math_instruct-2-train-000*-of-00032.arrow")))
print(len(files), "shards")

out = {"gsm8k": [], "augmented_gsm8k": [], "math": [], "augmented_math": []}
t0 = time.time()
for i, f in enumerate(files):
    with pa.memory_map(f, "rb") as src:
        try:
            tbl = pa.ipc.open_stream(src).read_all()
        except pa.ArrowInvalid:
            src.seek(0)
            tbl = pa.ipc.open_file(src).read_all()
    src_col = tbl.column("problem_source")
    for key in ("gsm8k", "augmented_gsm8k"):
        mask = pc.equal(src_col, key)
        sub = tbl.filter(mask)
        if sub.num_rows:
            out[key].append(sub)
    print(i, tbl.num_rows, f"{time.time()-t0:.0f}s", flush=True)

os.makedirs("data", exist_ok=True)
for key in ("gsm8k", "augmented_gsm8k"):
    if out[key]:
        t = pa.concat_tables(out[key])
        print(key, t.num_rows)
        import pyarrow.parquet as pq
        pq.write_table(t, f"data/omi2_{key}.parquet")
print("done", time.time() - t0)
