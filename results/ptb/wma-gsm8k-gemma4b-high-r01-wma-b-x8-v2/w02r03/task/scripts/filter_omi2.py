"""Filter OpenMathInstruct-2 down to the gsm8k-sourced rows and dump to jsonl.

Keeps problem / generated_solution / expected_answer / problem_source.
Nothing benchmark-derived: OMI2's gsm8k rows are built from the GSM8K *train*
split; the test split is only ever used by ../contamination_check.py.
"""
import glob
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import pyarrow.parquet as pq

KEEP = {"gsm8k", "augmented_gsm8k"}
SHARDS = sorted(
    glob.glob(
        "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
        "snapshots/*/data/*.parquet"
    )
)


def do_shard(args):
    idx, path = args
    out = f"/home/ben/task/data/raw/omi2_{idx:03d}.jsonl"
    n = 0
    with open(out, "w") as fh:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=20000):
            for r in batch.to_pylist():
                if r["problem_source"] in KEEP:
                    fh.write(json.dumps(r) + "\n")
                    n += 1
    return out, n


if __name__ == "__main__":
    import os

    os.makedirs("/home/ben/task/data/raw", exist_ok=True)
    print(len(SHARDS), "shards", flush=True)
    total = 0
    with ProcessPoolExecutor(max_workers=12) as ex:
        for out, n in ex.map(do_shard, list(enumerate(SHARDS))):
            total += n
            print(out, n, "cum", total, flush=True)
    print("TOTAL", total)
