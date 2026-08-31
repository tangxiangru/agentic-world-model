"""Pull upstream trajectory releases into ``raw/``.

Each fetcher is idempotent: re-running it resumes rather than re-downloads, so
a batch can be widened without paying for what is already on disk.

Sizes, measured 2026-08-24:

*   PI speedrun — 41 runs, 50 MB. The whole release; no reason to subset.
*   PostTrainBench — 1,842 runs, 28.9 GB in full. The default batch takes four
    agent configurations across the five core benchmarks, and only the files an
    analysis needs (0.59 GB): the ``task/`` workspace snapshots and the
    multi-hundred-MB ``error.log``s stay upstream until something needs them.
    Every batch also takes ``viewer_data/index.json`` (1.1 MB) — see
    ``PTB_CATALOG``.
"""

from __future__ import annotations

import json
import hashlib
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from awm.paths import ensure, raw_dir

PI_REPO = "https://github.com/PrimeIntellect-ai/frontier-automated-speedrun"
PTB_DATASET = "aisa-group/PostTrainBench-Trajectories"

#: The five objectively-scored PostTrainBench benchmarks. The two LLM-judged
#: ones (arenahardwriting, healthbench) are the observation group and are
#: fetched only when asked for.
PTB_CORE_BENCHMARKS = ("aime2025", "gsm8k", "gpqamain", "humaneval", "bfcl")
PTB_OBSERVE_BENCHMARKS = ("arenahardwriting", "healthbench")

#: Per-run files worth having. Deliberately excludes ``task/`` (workspace
#: snapshot) and ``error.log`` (up to 162 MB, mostly duplicate stderr).
PTB_RUN_FILES = (
    "solve_out.txt",  # the native CLI JSONL — the trajectory itself
    "solve_parsed.txt",  # upstream's human-readable rendering, used to cross-check
    "metrics.json",
    "time_taken.txt",
    "judgement_gpt5_4.json",  # contamination / disallowed-model verdict
    "system_monitor.log",  # 60 s samples of GPU util, memory, disk
)

#: First batch: two Claude Code and two Codex configurations, matched roughly in
#: size, covering the top of the leaderboard and the models we already validated
#: the parsers against.
PTB_DEFAULT_CONFIGS = (
    "claude_non_api_max_claude-opus-4-8_10h_run1",
    "claude_non_api_max_claude-fable-5_1m__10h_run1",
    "codex_non_api_high_gpt-5.4_10h_run1",
    "codex_non_api_max_gpt-5.6-sol_10h_run1",
)

#: Sentinel for "every agent configuration in the release". Measured over the
#: published file list: all 62 configurations, all 7 benchmarks, trace files
#: only, is 7.30 GB / 1,842 runs. The other 21.6 GB of the release is workspace
#: snapshots (10.6 GB), pre-parsed viewer JSON (5.3 GB, redundant with the
#: traces — bar ``PTB_CATALOG``) and error logs (1.5 GB).
#:
#: 1,842 is the *run directory* count, not the corpus: 1,786 of those carry a
#: ``solve_out.txt`` and 1,745 convert to events. The 56 with no trace are two
#: opencode configurations holding nothing but ``metrics.json``; the other 41
#: have a trace with no agent event in it (``NoAgentOutput``). Attrition is
#: 20.7% for opencode against under 1% for claude-code and codex, so a
#: per-scaffold count taken off the directory listing compares different
#: sampling rates. Fetch by directory, report by converted run.
ALL_CONFIGS: tuple[str, ...] = ()

#: The one file under ``viewer_data/`` worth having: the catalogue that backs
#: posttrainbench.com/traces. 1.1 MB, and the only place upstream publishes a
#: run's ``accuracy``, ``total_cost_usd``, ``num_turns``, ``duration_ms``,
#: ``trace_format`` and contamination verdict in one table — reconstructing that
#: from the traces means parsing all 7.3 GB. The other ~3,000 files in that
#: directory are per-run pre-parsed renderings (5.3 GB, redundant with the
#: traces) and stay upstream; ``ptb_select`` refuses all of them, so this is
#: fetched alongside a selection rather than through it.
#:
#: It is also a *partial* index: 1,509 of the release's 1,842 run directories.
#: The 333 it omits are the unjudged ones (10% carry ``judgement_gpt5_4.json``
#: against 98% of the catalogued), not the failed ones — 277 have a full trace
#: and a score, and the remaining 56 are the trace-less opencode runs. Treat it
#: as metadata, never as the run population. It is safe in the other direction:
#: every one of the 1,509 catalogued runs does have a trace on disk, so a
#: catalogue-derived split cannot pin a run we are unable to read.
PTB_CATALOG = "viewer_data/index.json"

# Every run in the committed GSM8K study split publishes these three artifacts.
# Other entries in ``PTB_RUN_FILES`` are useful but optional upstream.
PTB_REQUIRED_RUN_FILES = ("solve_out.txt", "metrics.json", "time_taken.txt")


@dataclass
class FetchResult:
    source: str
    path: Path
    n_files: int
    bytes: int

    def __str__(self) -> str:
        return f"{self.source}: {self.n_files} files, {self.bytes / 1e9:.2f} GB at {self.path}"


def _tree_size(path: Path) -> tuple[int, int]:
    n = 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            n += 1
            total += p.stat().st_size
    return n, total


def _local_download_metadata(dest: Path, filename: str) -> Path:
    """The sidecar written by ``hf_hub_download(..., local_dir=...)``."""
    return dest / ".cache" / "huggingface" / "download" / f"{filename}.metadata"


def _file_is_from_revision(dest: Path, filename: str, revision: str) -> bool:
    """Prove that a local file was downloaded at the requested Hub commit.

    Existence alone is insufficient: a resumed fetch may point the same local
    directory at a different dataset revision.  Hugging Face records the
    resolved commit, etag, and download timestamp beside every local-dir file.
    Treat a missing, malformed, stale, or differently pinned sidecar as a cache
    miss so it is downloaded again.
    """
    path = dest / filename
    metadata = _local_download_metadata(dest, filename)
    if not path.is_file() or not metadata.is_file():
        return False
    try:
        lines = metadata.read_text().splitlines()
        commit = lines[0].strip()
        downloaded_at = float(lines[2])
        # HF allows a one-second tolerance for coarse filesystem mtimes.
        unmodified = path.stat().st_mtime - 1 <= downloaded_at
    except (IndexError, OSError, ValueError):
        return False
    # Production split revisions are full 40-character commits.  Prefixes of
    # at least seven characters remain useful for direct/library smoke tests.
    pinned = commit == revision or (len(revision) >= 7 and commit.startswith(revision))
    return pinned and unmodified


def _write_testable_download_metadata(
    dest: Path, filename: str, revision: str, etag: str = "unknown"
) -> None:
    """Write an HF-compatible sidecar for non-Hub importers and test fixtures.

    The production downloader writes this itself.  Keeping the tiny helper here
    makes the provenance format explicit for tools that stage already-verified
    artifacts without reaching the Hub.
    """
    metadata = _local_download_metadata(dest, filename)
    ensure(metadata.parent)
    metadata.write_text(f"{revision}\n{etag}\n{time.time()}\n")


def check_ptb_run_files(
    runs: Iterable[str],
    revision: str,
    dest: Path,
    required_files: tuple[str, ...] = PTB_REQUIRED_RUN_FILES,
) -> list[str]:
    """Return provenance/completeness problems for a declared run corpus."""
    problems: list[str] = []
    for run in dict.fromkeys(runs):
        for name in required_files:
            filename = f"{run}/{name}"
            path = dest / filename
            if not path.is_file():
                problems.append(f"{run}: required {name} is missing")
            elif not _file_is_from_revision(dest, filename, revision):
                problems.append(
                    f"{run}: {name} is not proven to come from revision {revision}"
                )
    return problems


def fetch_pi(dest: Path | None = None) -> FetchResult:
    """Clone the PI speedrun release (shallow). Re-run to update in place."""
    dest = dest or raw_dir("pi_speedrun")
    ensure(dest.parent)
    if (dest / ".git").exists():
        subprocess.run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin"], check=True)
        subprocess.run(["git", "-C", str(dest), "reset", "--hard", "origin/HEAD"], check=True)
    else:
        subprocess.run(
            ["git", "clone", "--depth", "1", PI_REPO, str(dest)],
            check=True,
        )
    n, total = _tree_size(dest)
    return FetchResult("pi_speedrun", dest, n, total)


def ptb_list_files(cache: Path | None = None) -> list[tuple[str, int]]:
    """Every ``(path, size)`` in the dataset, from the paginated tree API.

    ``snapshot_download(allow_patterns=...)`` is unusable here: on a repo with
    218k files it sat for ten minutes without writing anything. Listing the tree
    ourselves takes about 25 s (1000 entries per page) and lets us select exact
    paths, so downloads start immediately. The listing is cached, since it only
    changes when upstream publishes more runs.
    """
    import requests

    cache = cache or (raw_dir("posttrainbench") / ".file_list.json")
    if cache.exists():
        return [(p, s) for p, s in json.loads(cache.read_text())]

    from huggingface_hub import get_token

    token = get_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = f"https://huggingface.co/api/datasets/{PTB_DATASET}/tree/main"
    params: dict[str, Any] = {"recursive": "true", "limit": 1000}
    out: list[tuple[str, int]] = []
    while url:
        r = requests.get(url, headers=headers, params=params, timeout=60)
        r.raise_for_status()
        out.extend((e["path"], e.get("size", 0)) for e in r.json() if e.get("type") == "file")
        # The next page arrives as a fully-formed URL in the Link header.
        link = r.headers.get("Link", "")
        url = link.split(">;")[0].lstrip("<") if 'rel="next"' in link else ""
        params = {}
    ensure(cache.parent)
    cache.write_text(json.dumps(out))
    return out


def ptb_select(
    all_files: list[tuple[str, int]],
    configs: tuple[str, ...] = PTB_DEFAULT_CONFIGS,
    benchmarks: tuple[str, ...] = PTB_CORE_BENCHMARKS,
    files: tuple[str, ...] = PTB_RUN_FILES,
) -> list[tuple[str, int]]:
    """Pick ``<config>/<benchmark>_<org>_<model>_<cluster>/<file>`` paths.

    An empty ``configs`` means every configuration (``ALL_CONFIGS``); an empty
    ``benchmarks`` or ``files`` selects nothing, since those are the filters that
    keep the download from being the whole 28.9 GB release.
    """
    cfg = set(configs)
    bench = set(benchmarks)
    want = set(files)
    picked = []
    for path, size in all_files:
        parts = path.split("/")
        if len(parts) != 3 or parts[2] not in want:
            continue
        if cfg and parts[0] not in cfg:
            continue
        if parts[0] == "viewer_data":
            continue
        if parts[1].split("_")[0] in bench:
            picked.append((path, size))
    return picked


def ptb_select_runs(
    all_files: list[tuple[str, int]],
    runs: Iterable[str],
    files: tuple[str, ...] = PTB_RUN_FILES,
) -> list[tuple[str, int]]:
    """Pick the wanted files of exactly the named ``<config>/<run_name>`` dirs.

    Selection is intersected with the published file list rather than built by
    concatenation: a run that never published one of the wanted files must not
    produce a path that 404s on download.
    """
    wanted_runs = set(runs)
    want = set(files)
    return [
        (path, size)
        for path, size in all_files
        if path.rsplit("/", 1)[0] in wanted_runs and path.rsplit("/", 1)[1] in want
    ]


def ptb_list_run_files(
    runs: Iterable[str],
    revision: str,
    files: tuple[str, ...] = PTB_RUN_FILES,
    workers: int = 12,
) -> list[tuple[str, int]]:
    """List only the direct files of named run directories at ``revision``.

    The dataset-wide recursive tree contains more than 200,000 files and can
    take minutes (or stall behind a proxy) before a split fetch starts.  A
    committed split already names its run directories, so query those small
    trees directly and keep the same explicit file allowlist.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from huggingface_hub import HfApi

    wanted = set(files)
    unique_runs = tuple(dict.fromkeys(runs))

    def one(run: str) -> list[tuple[str, int]]:
        rows = HfApi().list_repo_tree(
            repo_id=PTB_DATASET,
            path_in_repo=run,
            recursive=False,
            expand=False,
            revision=revision,
            repo_type="dataset",
        )
        return [
            (entry.path, int(getattr(entry, "size", 0) or 0))
            for entry in rows
            if entry.path.rsplit("/", 1)[-1] in wanted
            and getattr(entry, "size", None) is not None
        ]

    out: list[tuple[str, int]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed(pool.submit(one, run) for run in unique_runs):
            out.extend(fut.result())
    return sorted(out)


def fetch_ptb_runs(
    runs: Iterable[str],
    revision: str,
    files: tuple[str, ...] = PTB_RUN_FILES,
    dest: Path | None = None,
    workers: int = 12,
) -> FetchResult:
    """Download exactly the named runs' files, pinned to one dataset revision.

    This is the fetcher a committed split uses: the run list and the revision
    both come out of its YAML, so two people running it get the same bytes. The
    pinned catalogue rides along the way it does in every batch.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import huggingface_hub

    dest = dest or raw_dir("posttrainbench")
    ensure(dest)
    runs = tuple(dict.fromkeys(runs))
    # Cache the pinned remote listing, not merely a "mandatory files exist"
    # sentinel.  Some wanted artifacts (for example solve_parsed.txt) are
    # optional upstream.  Remembering the exact listing lets an interrupted
    # fetch restore one of those files without relisting 193 directories, while
    # never inventing a path for a file the run did not publish.
    selection = json.dumps(
        {"revision": revision, "runs": sorted(runs), "files": sorted(files)},
        sort_keys=True,
        separators=(",", ":"),
    )
    cache_key = hashlib.sha256(selection.encode()).hexdigest()
    cache = dest / ".split_file_lists" / f"{cache_key}.json"
    if cache.is_file():
        wanted = [(str(path), int(size)) for path, size in json.loads(cache.read_text())]
    else:
        wanted = ptb_list_run_files(runs, revision, files, workers)
        ensure(cache.parent)
        cache.write_text(json.dumps(wanted, sort_keys=True))
    todo = [p for p, _ in wanted if not _file_is_from_revision(dest, p, revision)]
    if not _file_is_from_revision(dest, PTB_CATALOG, revision):
        todo.append(PTB_CATALOG)
    print(f"posttrainbench@{revision[:12]}: {len(wanted)} files selected, {len(todo)} to download")

    def one(path: str) -> str:
        return huggingface_hub.hf_hub_download(
            repo_id=PTB_DATASET,
            repo_type="dataset",
            filename=path,
            local_dir=str(dest),
            revision=revision,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed({pool.submit(one, p) for p in todo}):
            fut.result()

    unverified = [
        path
        for path in [*(p for p, _ in wanted), PTB_CATALOG]
        if not _file_is_from_revision(dest, path, revision)
    ]
    if unverified:
        preview = ", ".join(unverified[:5])
        suffix = " ..." if len(unverified) > 5 else ""
        raise RuntimeError(
            f"{len(unverified)} downloaded file(s) lack matching revision metadata: "
            f"{preview}{suffix}"
        )

    n, total = _tree_size(dest)
    return FetchResult("posttrainbench", dest, n, total)


def fetch_posttrainbench(
    dest: Path | None = None,
    configs: tuple[str, ...] = PTB_DEFAULT_CONFIGS,
    benchmarks: tuple[str, ...] = PTB_CORE_BENCHMARKS,
    files: tuple[str, ...] = PTB_RUN_FILES,
    workers: int = 12,
    progress_every: int = 50,
    catalog: bool = True,
) -> FetchResult:
    """Download a subset of the PostTrainBench trajectory dataset.

    Apache-2.0 and not gated, so no token is required; one is used if present.

    ``catalog`` adds ``PTB_CATALOG`` to the batch. It is on by default and costs
    1.1 MB: no subset of the traces reproduces it, and the smallest batch wants
    it as much as the largest. Pass ``catalog=False`` for a traces-only mirror.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from huggingface_hub import hf_hub_download

    dest = dest or raw_dir("posttrainbench")
    ensure(dest)
    wanted = ptb_select(ptb_list_files(), configs, benchmarks, files)
    todo = [(p, s) for p, s in wanted if not (dest / p).exists()]
    print(
        f"posttrainbench: {len(wanted)} files selected "
        f"({sum(s for _, s in wanted) / 1e9:.2f} GB), {len(todo)} to download"
    )
    # Outside the selection on purpose: ptb_select's job is to keep viewer_data
    # out, and it must keep doing that whatever configs it is handed.
    if catalog and not (dest / PTB_CATALOG).exists():
        todo.append((PTB_CATALOG, 0))
        print(f"  + {PTB_CATALOG}")

    def one(path: str) -> str:
        return hf_hub_download(
            repo_id=PTB_DATASET, repo_type="dataset", filename=path, local_dir=str(dest)
        )

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, p): p for p, _ in todo}
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if done % progress_every == 0 or done == len(todo):
                print(f"  {done}/{len(todo)}", flush=True)

    n, total = _tree_size(dest)
    return FetchResult("posttrainbench", dest, n, total)


def ptb_catalog(dest: Path | None = None) -> dict[str, Any]:
    """Read the fetched ``PTB_CATALOG``.

    Raises ``FileNotFoundError`` rather than fetching, so an analysis never
    silently turns into a download. ``runs`` is the row list; ``experiments``
    and ``benchmarks`` are the facet lists the site's filters are built from.
    """
    dest = dest or raw_dir("posttrainbench")
    path = dest / PTB_CATALOG
    if not path.exists():
        raise FileNotFoundError(
            f"{path} — run `awm traj fetch posttrainbench` (it is in every batch)"
        )
    return json.loads(path.read_text())


FETCHERS = {"pi_speedrun": fetch_pi, "posttrainbench": fetch_posttrainbench}
