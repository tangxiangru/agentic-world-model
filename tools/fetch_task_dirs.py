"""Fetch the ``task/`` workspace snapshot of every run in a split.

``awm split fetch`` deliberately takes only the trace files (``PTB_RUN_FILES``):
the workspace snapshots are 10.6 GB across the whole release. For experiment-card
extraction the workspace is the point — ``task/train_v3.py`` is the training
script the card must cite, ``task/eval_v2.json`` is the comparator's output —
so this pulls exactly the split's runs' ``task/`` trees, at the split's pinned
revision, into the same ``raw/`` layout the converters read.

Usage:
    python3 tools/fetch_task_dirs.py posttrainbench/gsm8k-gemma-holdout-v1
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from awm import splits  # noqa: E402
from awm.paths import raw_dir  # noqa: E402
from awm.traj.fetch import PTB_DATASET, ptb_list_files  # noqa: E402


def main(split_id: str, workers: int = 12) -> int:
    import huggingface_hub

    s = splits.load(split_id)
    revision = s.dataset["revision"]
    runs = set(s.train + s.test)
    dest = raw_dir("posttrainbench")
    wanted = [
        (p, size) for p, size in ptb_list_files()
        if p.rsplit("/", 2)[0] in runs and "/task/" in p
    ]
    todo = [p for p, _ in wanted if not (dest / p).exists()]
    total_mb = sum(size for _, size in wanted) / 1e6
    print(f"{split_id}: {len(wanted)} task files ({total_mb:.0f} MB) over {len(runs)} runs, "
          f"{len(todo)} to download")

    def one(path: str) -> str:
        return huggingface_hub.hf_hub_download(
            repo_id=PTB_DATASET, repo_type="dataset", filename=path,
            revision=revision, local_dir=str(dest),
        )

    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, p): p for p in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                failed.append((futures[fut], exc))
            if n % 200 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
    print(f"done: {len(todo) - len(failed)} downloaded, {len(failed)} failed")
    for path, exc in failed[:20]:
        print(f"  FAILED {path}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "posttrainbench/gsm8k-gemma-holdout-v1"))
