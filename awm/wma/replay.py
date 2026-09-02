"""Offline replay over the historical card corpus.

For run R and its k-th card, build a session directory the agent can stand
in as if it were the scientist at that moment: cards 1..k-1 with their
results, card k with sections 0-4 only, an index, and a ``history/`` link to
every *other* run on the same side of the split. The truth — the original
card k with its result and the run's official score — is kept outside every
session directory, under ``<out>/_truth/``. The leakage rules are this code:

* the run's own later cards and the k-th card's result are never copied;
* ``outcome`` (the run's official score) is stripped from every card of the
  run in the session, and only ever read by reconcile;
* history contains only the other runs of the requested side — replaying
  ``train`` never touches ``test``.
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from awm.exp_protocol import lineage
from awm.exp_protocol.schema import CardError, dump_card, load_card, migrate_v1

from . import schema
from .backends import Backend, BackendError, Budget
from .reconcile import reconcile
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


def _history_dir(out: Path, side_dir: Path, run_ref: str, all_runs: list[Path]) -> Path:
    hist = out / "_history" / run_ref
    hist.mkdir(parents=True, exist_ok=True)
    for other in all_runs:
        if other.name == run_ref:
            continue
        link = hist / other.name
        if not link.is_symlink():
            os.symlink(other.resolve(), link)
    return hist


def _build_session(out: Path, run_ref: str, cards: list[dict[str, Any]], k: int, hist: Path) -> tuple[Path, Path]:
    card_id = cards[k - 1]["card_id"]
    session = out / run_ref / card_id
    truth = out / "_truth" / run_ref / f"{card_id}.yaml"
    if not truth.is_file():
        dump_card(truth, cards[k - 1])
    if session.is_dir():
        return session, truth  # never rebuild: a verdict may already be there
    cdir = session / "memory" / "cards"
    for card in cards[: k - 1]:
        dump_card(cdir / f"{card['card_id']}.yaml", _strip(card, STRIP_ALWAYS))
    dump_card(cdir / f"{card_id}.yaml", _strip(cards[k - 1], STRIP_FROM_K))
    lineage.write_index(session)
    link = session / "history"
    if not link.is_symlink():
        os.symlink(hist.resolve(), link)
    return session, truth


def build_samples(corpus: Path, out: Path, *, side: str = "train", sample: int | None = None,
                  seed: int = 0) -> list[Sample]:
    corpus, out = Path(corpus), Path(out)
    side_dir = corpus / side
    if not side_dir.is_dir():
        raise FileNotFoundError(f"{side_dir} is not a directory")
    runs = sorted(d for d in side_dir.iterdir() if d.is_dir())
    loaded = {d.name: load_run(d) for d in runs}
    pairs = [(d.name, k) for d in runs for k in range(1, len(loaded[d.name]) + 1)]
    if sample is not None and sample < len(pairs):
        pairs = sorted(random.Random(seed).sample(pairs, sample))
    samples: list[Sample] = []
    hist_cache: dict[str, Path] = {}
    for run_ref, k in pairs:
        cards = loaded[run_ref]
        if run_ref not in hist_cache:
            hist_cache[run_ref] = _history_dir(out, side_dir, run_ref, runs)
        session, truth = _build_session(out, run_ref, cards, k, hist_cache[run_ref])
        samples.append(Sample(run_ref, cards[k - 1]["card_id"], k, len(cards), session, truth))
    out.mkdir(parents=True, exist_ok=True)
    (out / "samples.jsonl").write_text("".join(s.to_json() + "\n" for s in samples))
    return samples


def read_samples(out: Path) -> list[Sample]:
    path = Path(out) / "samples.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path}: run build first")
    return [Sample.from_json(line) for line in path.read_text().splitlines() if line.strip()]


def run_replay(out: Path, backend: Backend, *, budget: Budget | None = None, model: str | None = None,
               limit: int | None = None) -> dict[str, int]:
    out = Path(out)
    counts = {"reviewed": 0, "reconciled": 0, "skipped": 0, "errors": 0}
    errors = out / "errors.jsonl"
    done = 0
    for s in read_samples(out):
        session = s.session_dir
        card_path = session / "memory" / "cards" / f"{s.card_id}.yaml"
        if schema.verdict_path(card_path).is_file():
            counts["skipped"] += 1
            continue
        if limit is not None and done >= limit:
            break
        done += 1
        try:
            review(session, s.card_id, backend, mode="offline", budget=budget or Budget(), model=model,
                   history_dir=session / "history")
            counts["reviewed"] += 1
            reconcile(card_path, s.truth_path)
            counts["reconciled"] += 1
        except (ReviewError, BackendError, ValueError) as exc:
            counts["errors"] += 1
            with errors.open("a") as fh:
                fh.write(json.dumps({"run_ref": s.run_ref, "card_id": s.card_id, "error": str(exc)}) + "\n")
    return counts
