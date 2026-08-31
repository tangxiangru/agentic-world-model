"""Render, per run of a split, what an experiment-card extractor is allowed to read.

Output, under ``<data>/exp-cards/<split-name>/``:

    manifest.json            run_ref -> {run, side, trained_model, benchmark, events, task_files}
    sources.json             run_ref -> run path   (audit only; never handed to a consumer)
    digests/<run_ref>.md     the score-free, identity-free digest of that run
    task/<run_ref>/          symlinks to the run's task/ workspace files

Two properties carried over from ``awm.analysis.recipe``:

* the digest carries **no score** and **no agent identity** — the header names
  the base model and the benchmark, and ``run_ref`` is an opaque id;
* every block is headed ``--- [i] ...`` so a card's evidence can cite the event
  index and be checked against the stream.

The selection is the recipe filter widened with evaluation vocabulary: a card
needs the agent's ``evaluate.py --limit`` runs and their results — the
comparator and the observed problem live there — and the recipe filter drops
them, because ``evaluate.py`` and ``accuracy`` are not recipe words.

Usage:
    python3 tools/render_card_digests.py posttrainbench/gsm8k-gemma-holdout-v1
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from awm import paths, splits  # noqa: E402
from awm.analysis import recipe as R  # noqa: E402

EVAL_SIGNAL = re.compile(
    r"""(?xi)
    (
      # any launch of a script whose name says what it does: train_gsm8k.py,
      # train_v3.py, sft.py, run_grpo.py, merge_lora.py, prep_data2.py, eval_ckpt.py
      \b\w*(train|sft|rft|grpo|dpo|ppo|finetun|fine-tun|merge|distil|prep|data|eval|
             sample|generat|filter|soup|quick|fix|patch|config|export|convert|save|
             setup|pack|submit|final)\w*\.py\b|
      \b(nohup|torchrun|accelerate|deepspeed|python3?\s+-m\s+\S+)\b|
      \b(cp\s+-r|rsync|mv|ln\s+-s)\b.*\b(final_model|submission|ckpt|checkpoint|model)\b|
      \b(fine[-_ ]?tun\w*|evaluate\.py|--limit|accuracy|acc|score|baseline|eval(uation)?|
         dev[-_ ]?set|failures?|wrong|incorrect|mistakes?|error\ rate|regress\w*|
         timer\.sh|remaining|hours?\ left|final_model|submission|
         kill(ed)?|oom|out\ of\ memory|traceback|crash)\b
    )
    """
)
def _body(p: re.Pattern) -> str:
    return re.sub(r"^\s*\(\?[a-z]+\)", "", p.pattern, count=1)


COMBINED = re.compile(f"(?:{_body(R.RECIPE_SIGNAL)})|(?:{_body(EVAL_SIGNAL)})", re.X | re.I)

BUDGET = 260_000


def run_ref(run: str) -> str:
    return "r-" + hashlib.sha256(run.encode()).hexdigest()[:8]


def main(split_id: str, out_name: str | None = None) -> int:
    s = splits.load(split_id)
    name = split_id.split("/", 1)[1]
    out = paths.data_root() / "exp-cards" / (out_name or name)
    (out / "digests").mkdir(parents=True, exist_ok=True)
    (out / "task").mkdir(exist_ok=True)
    raw = paths.raw_dir("posttrainbench")
    events_dir = paths.events_dir("posttrainbench")

    # Widen the filter for this process only.
    R.RECIPE_SIGNAL = COMBINED

    # Sub-agent streams are numbered from 0 separately from ``main`` (schema
    # rule), so a digest that prints bare ``[i]`` collides. Offset each
    # sub-agent's indices by 100000 * k so every ``[i]`` in a digest is unique;
    # the header explains the convention.
    def _events_unique(path: Path):
        agents: dict[str, int] = {}
        with gzip.open(path, "rt") as fh:
            for line in fh:
                if not line.strip():
                    continue
                e = json.loads(line)
                a = e.get("agent_id", "main")
                if a != "main":
                    k = agents.setdefault(a, len(agents) + 1)
                    e["i"] = 100000 * k + e["i"]
                yield e
    R._events = _events_unique

    manifest: dict[str, dict] = {}
    sources: dict[str, str] = {}
    missing = []
    for side in ("train", "test"):
        for run in getattr(s, side):
            ref = run_ref(run)
            sources[ref] = run
            config, run_name = run.split("/", 1)
            run_id = f"{config}__{run_name}"
            ev = events_dir / f"{run_id}.jsonl.gz"
            meta_path = events_dir / f"{run_id}.meta.json"
            if not ev.is_file():
                missing.append(run)
                continue
            meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
            # base model from the run name: gsm8k_<org>_<model>_<cluster id>
            m = re.match(r"^[^_]+_(.+)_\d+$", run_name)
            # meta["model"] is the AGENT model in RunMeta — never use it here.
            trained = m.group(1).replace("_", "/", 1) if m else None
            events = R.select(ev, budget=BUDGET)
            # Elapsed hours since the run's first timestamped event, when the
            # harness recorded timestamps (130 of 193 runs do). The extractor
            # reads it into `elapsed_h`; a run without timestamps leaves it null.
            t0 = None
            hours: dict[int, float] = {}
            with gzip.open(ev, "rt") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    e = json.loads(line)
                    ts = e.get("ts")
                    if ts and e.get("agent_id", "main") == "main":
                        try:
                            t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        except ValueError:
                            continue
                        t0 = t0 or t
                        hours[e["i"]] = (t - t0).total_seconds() / 3600
            for e in events:
                if e["i"] in hours:
                    e["t"] = hours[e["i"]]
            head = [f"# run: {ref}", f"# trained_model: {trained}", f"# benchmark: {s.benchmark}",
                    "# time_budget_h: 10",
                    "# block header: --- [event index] turn=N t=+hours-since-start act ---"
                    if hours else "# block header: --- [event index] turn=N act ---  (no timestamps in this run)",
                    "# sub-agent events (if any) carry indices >= 100000: 100000*k + i for sub-agent k",
                    f"# recipe-bearing events: {len(events)}", ""]
            body = []
            for e in events:
                turn = f" turn={e['turn']}" if e.get("turn") is not None else ""
                t = f" t=+{e['t']:.2f}h" if "t" in e else ""
                body.append(f"--- [{e['i']}]{turn}{t} {e['act']} ---\n{e['text']}")
            digest = "\n".join(head + body)
            (out / "digests" / f"{ref}.md").write_text(digest)

            task_dir = raw / run / "task"
            task_files = []
            if task_dir.is_dir():
                link_dir = out / "task" / ref
                link_dir.mkdir(exist_ok=True)
                for f in sorted(task_dir.rglob("*")):
                    if f.is_file():
                        rel = f.relative_to(task_dir)
                        dst = link_dir / rel
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        if not dst.exists():
                            dst.symlink_to(f.resolve())
                        task_files.append(str(rel))
            with gzip.open(ev, "rt") as fh:
                n_events = sum(1 for line in fh if line.strip())
            manifest[ref] = {
                "side": side,
                "trained_model": trained,
                "benchmark": s.benchmark,
                "events_total": n_events,
                "digest_events": len(events),
                "digest_chars": len(digest),
                "task_files": task_files,
                # manifest only; never in the digest
                "accuracy": (meta.get("final_score") or {}).get("value"),
            }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    (out / "sources.json").write_text(json.dumps(sources, indent=1, sort_keys=True))
    print(f"{len(manifest)} digests -> {out}")
    if missing:
        print(f"  {len(missing)} runs without events (not converted yet?)")
        for r in missing[:10]:
            print("   ", r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(
        sys.argv[1] if len(sys.argv) > 1 else "posttrainbench/gsm8k-gemma-holdout-v1",
        sys.argv[2] if len(sys.argv) > 2 else None,
    ))
