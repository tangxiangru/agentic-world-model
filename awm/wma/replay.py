"""Offline replay over the historical card corpus.

For run R and its k-th card, build a session directory the agent can stand
in as if it were the scientist at that moment: cards 1..k-1 with their
results, card k with sections 0-4 only, an index, and a ``history/`` link to
every *other* run on the same side of the split. The truth — the original
card k with its result and the run's official score — is kept outside every
session directory, under ``<out>/_truth/``. The leakage rules are this code:

* the run's own later cards and the k-th card's result are never copied;
* ``outcome`` (the run's official score) is stripped from every card of the
  run in the session, and only ever read by the ledger, from ``_truth``;
* history contains only the other runs of the requested side — replaying
  ``train`` never touches ``test``.

Verdicts are written once; scoring happens when the ledger reads them.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from awm.exp_protocol import lineage
from awm.exp_protocol.schema import CardError, dump_card, load_card, migrate_v1

from . import schema
from .backends import Backend, BackendError, Budget
from .review import ReviewError, review

CARD_RE = re.compile(r"^exp-(\d+)\.yaml$")
STRIP_ALWAYS = ("outcome",)
STRIP_FROM_K = ("result", "conclusion", "outcome")


@dataclass
class Sample:
    run_ref: str
    card_id: str
    k: int
    n_cards: int
    session_dir: Path
    truth_path: Path

    def to_json(self) -> str:
        d = asdict(self)
        d["session_dir"], d["truth_path"] = str(self.session_dir), str(self.truth_path)
        return json.dumps(d)

    @classmethod
    def from_json(cls, line: str) -> "Sample":
        d = json.loads(line)
        d["session_dir"], d["truth_path"] = Path(d["session_dir"]), Path(d["truth_path"])
        return cls(**d)


def _card_num(p: Path) -> int:
    m = CARD_RE.match(p.name)
    return int(m.group(1)) if m else -1


def load_run(run_dir: Path) -> list[dict[str, Any]]:
    """The run's cards in launch order, migrated to v2 in memory; unreadable files are skipped."""
    out = []
    for p in sorted((q for q in Path(run_dir).glob("exp-*.yaml") if CARD_RE.match(q.name)), key=_card_num):
        try:
            out.append(migrate_v1(load_card(p)))
        except CardError:
            continue
    return out


def _strip(card: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {k: v for k, v in card.items() if k not in keys}


def _history_dir(out: Path, run_ref: str, all_runs: list[Path]) -> Path:
    hist = out / "_history" / run_ref
    hist.mkdir(parents=True, exist_ok=True)
    for other in all_runs:
        if other.name == run_ref:
            continue
        link = hist / other.name
        if not link.is_symlink():
            os.symlink(other.resolve(), link)
    return hist


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def truth_card(cards: list[dict[str, Any]], k: int) -> dict[str, Any]:
    """Card k as the truth: the corpus card, plus the comparator the corpus implied but did not write.

    Corpus cards mostly leave ``evaluation.comparator.value`` empty. When the
    parent is an earlier card of the same run with a measurement of the same
    metric, that measurement is the comparator — the scientist had it, and
    L2 becomes scorable. Marked ``comparator_source: parent_card``. The
    session's copy of the card is not changed: the agent sees the corpus.
    """
    card = json.loads(json.dumps(cards[k - 1]))  # deep copy
    ev = card.setdefault("evaluation", {})
    comp = ev.get("comparator") or {}
    if _is_num(comp.get("value")):
        return card
    origin = ((card.get("setup") or {}).get("parent_checkpoint") or {}).get("origin")
    parent = next((c for c in cards[: k - 1] if c.get("card_id") == origin), None)
    if parent is None:
        return card
    ms = [m for m in ((parent.get("result") or {}).get("measurements") or []) if isinstance(m, dict)]
    own = [m for m in ((card.get("result") or {}).get("measurements") or []) if isinstance(m, dict)]
    metric = own[0].get("metric") if own else None
    match = next((m for m in ms if _is_num(m.get("value")) and (metric is None or m.get("metric") == metric)), None)
    if match is None:
        return card
    ev["comparator"] = {"ref": origin, "value": match["value"], "path": match.get("path")}
    ev["comparator_source"] = "parent_card"
    return card


def _build_session(out: Path, run_ref: str, cards: list[dict[str, Any]], k: int, hist: Path) -> tuple[Path, Path]:
    card_id = cards[k - 1]["card_id"]
    session = out / run_ref / card_id
    truth = out / "_truth" / run_ref / f"{card_id}.yaml"
    dump_card(truth, truth_card(cards, k))  # truth is derived from the corpus; rewriting it is always safe
    cdir = session / "memory" / "cards"
    link = session / "history"
    complete = (cdir / f"{card_id}.yaml").is_file() and (session / "memory" / "index.md").is_file() and link.is_symlink()
    if complete:
        return session, truth  # a verdict may already be there; never touch a finished session
    # Either a fresh session or one an interrupted build left half-written: (re)write the cards
    # and the index, link the history. Verdict files, if any, are not among what is written.
    for card in cards[: k - 1]:
        dump_card(cdir / f"{card['card_id']}.yaml", _strip(card, STRIP_ALWAYS))
    dump_card(cdir / f"{card_id}.yaml", _strip(cards[k - 1], STRIP_FROM_K))
    lineage.write_index(session)
    if not link.is_symlink():
        os.symlink(hist.resolve(), link)
    return session, truth


def run_ref(run_id: str) -> str:
    """The corpus names a run by the first eight hex digits of the sha256 of its catalogue id."""
    return "r-" + hashlib.sha256(run_id.encode()).hexdigest()[:8]


def default_split(corpus: Path) -> Path:
    """``splits/posttrainbench/<corpus basename>.yaml`` — the contract the corpus was cut from."""
    from awm import paths

    return paths.REPO_ROOT / "splits" / "posttrainbench" / (Path(corpus).name + ".yaml")


def runs_by_agent(split: Path, side: str, agents: str) -> tuple[set[str], int]:
    """Which run_refs on ``side`` belong to an agent whose run id matches ``agents``.

    The run id (agent, scaffold, base model) lives only in the split file; the corpus, the sessions and
    the history carry the opaque run_ref, so the filter never tells the WMA who the scientist was.
    """
    import yaml

    split = Path(split)
    if not split.is_file():
        raise FileNotFoundError(f"{split}: the split file that maps run ids to the corpus")
    ids = (yaml.safe_load(split.read_text()).get("splits") or {}).get(side) or []
    pat = re.compile(agents)
    return {run_ref(r) for r in ids if pat.search(r)}, len(ids)


def build_samples(corpus: Path, out: Path, *, side: str = "train", sample: int | None = None,
                  seed: int = 0, agents: str | None = None, split: Path | None = None) -> list[Sample]:
    corpus, out = Path(corpus), Path(out)
    side_dir = corpus / side
    if not side_dir.is_dir():
        raise FileNotFoundError(f"{side_dir} is not a directory")
    runs = sorted(d for d in side_dir.iterdir() if d.is_dir())
    picked = runs
    meta: dict[str, Any] = {}
    if agents:
        wanted, total = runs_by_agent(split or default_split(corpus), side, agents)
        picked = [d for d in runs if d.name in wanted]
        meta = {"agents": agents, "runs_matched": len(picked), "runs_total": total,
                "split": str(split or default_split(corpus))}
    loaded = {d.name: load_run(d) for d in picked}
    pairs = [(d.name, k) for d in picked for k in range(1, len(loaded[d.name]) + 1)]
    if sample is not None and sample < len(pairs):
        pairs = sorted(random.Random(seed).sample(pairs, sample))
    samples: list[Sample] = []
    hist_cache: dict[str, Path] = {}
    for run_ref, k in pairs:
        cards = loaded[run_ref]
        if run_ref not in hist_cache:
            hist_cache[run_ref] = _history_dir(out, run_ref, runs)
        session, truth = _build_session(out, run_ref, cards, k, hist_cache[run_ref])
        samples.append(Sample(run_ref, cards[k - 1]["card_id"], k, len(cards), session, truth))
    out.mkdir(parents=True, exist_ok=True)
    (out / "samples.jsonl").write_text("".join(s.to_json() + "\n" for s in samples))
    (out / "samples.sha").write_text(fingerprint(samples) + "\n")
    if meta:
        (out / "filter.json").write_text(json.dumps(meta, indent=2) + "\n")
    return samples


def fingerprint(samples: list[Sample]) -> str:
    """The identity of a sample set: the (run, card) pairs, not the paths they were built under.

    ``samples.jsonl`` embeds the out directory, so its bytes differ between two builds of the same
    set; this does not. Rounds are comparable only on the same fingerprint.
    """
    lines = sorted(f"{s.run_ref} {s.card_id}" for s in samples)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def read_samples(out: Path) -> list[Sample]:
    path = Path(out) / "samples.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path}: run build first")
    return [Sample.from_json(line) for line in path.read_text().splitlines() if line.strip()]


def run_replay(out: Path, backend: Backend, *, budget: Budget | None = None, model: str | None = None,
               effort: str | None = None, limit: int | None = None, jobs: int = 1) -> dict[str, int]:
    out = Path(out)
    counts = {"reviewed": 0, "skipped": 0, "errors": 0}
    errors = out / "errors.jsonl"
    lock = threading.Lock()
    pending: list[Sample] = []
    for s in read_samples(out):
        card_path = s.session_dir / "memory" / "cards" / f"{s.card_id}.yaml"
        if schema.verdict_path(card_path).is_file():
            counts["skipped"] += 1
        elif limit is None or len(pending) < limit:
            pending.append(s)

    def one(s: Sample) -> None:
        session = s.session_dir
        try:
            review(session, s.card_id, backend, mode="offline", budget=budget or Budget(), model=model,
                   effort=effort, history_dir=session / "history")
            with lock:
                counts["reviewed"] += 1
        except (ReviewError, BackendError, ValueError) as exc:
            with lock:
                counts["errors"] += 1
                with errors.open("a") as fh:
                    fh.write(json.dumps({"run_ref": s.run_ref, "card_id": s.card_id, "error": str(exc)}) + "\n")

    # Sessions are independent directories, so samples can be reviewed concurrently; the only shared
    # state is the skill link (locked in prepare_session), the counters and the error log.
    if jobs > 1 and len(pending) > 1:
        with ThreadPoolExecutor(max_workers=min(jobs, len(pending))) as pool:
            list(pool.map(one, pending))
    else:
        for s in pending:
            one(s)
    return counts

