"""Build the ``raw/.file_list.json`` cache for exactly one split's runs.

``awm.traj.fetch.ptb_list_files`` walks the whole 218k-file dataset tree, and a
single stalled page hangs it with no timeout. For a split we already know the
193 run directories, so list each one (fast, recursive, per-run) and write the
same cache the fetcher reads. The catalogue path is added so the fetcher's
``PTB_CATALOG`` download is covered.

Usage:
    python3 tools/list_split_files.py posttrainbench/gsm8k-gemma-holdout-v1
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from awm import splits  # noqa: E402
from awm.paths import ensure, raw_dir  # noqa: E402
from awm.traj.fetch import PTB_CATALOG, PTB_DATASET  # noqa: E402


def main(split_id: str) -> int:
    from huggingface_hub import HfApi

    s = splits.load(split_id)
    rev = s.dataset["revision"]
    api = HfApi()
    runs = list(s.train) + list(s.test)

    def one(run: str) -> list[tuple[str, int]]:
        for attempt in range(3):
            try:
                return [
                    (f.path, getattr(f, "size", 0) or 0)
                    for f in api.list_repo_tree(PTB_DATASET, path_in_repo=run, revision=rev,
                                                repo_type="dataset", recursive=True)
                    if getattr(f, "size", None) is not None
                ]
            except Exception as exc:  # noqa: BLE001
                last = exc
        raise RuntimeError(f"{run}: {last}")

    out: list[tuple[str, int]] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, files in enumerate(pool.map(one, runs), 1):
            out.extend(files)
            if i % 25 == 0:
                print(f"  {i}/{len(runs)} runs listed", flush=True)
    out.append((PTB_CATALOG, 0))
    cache = ensure(raw_dir("posttrainbench")) / ".file_list.json"
    cache.write_text(json.dumps(out))
    mb = sum(sz for _, sz in out) / 1e6
    print(f"{len(out)} files ({mb:.0f} MB) over {len(runs)} runs -> {cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "posttrainbench/gsm8k-gemma-holdout-v1"))
