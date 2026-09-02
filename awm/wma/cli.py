"""``awm wma``: review a card, read the ledger, replay the corpus.

``register`` needs only argparse; the package's modules are imported inside
the handlers, like the other ``awm`` command groups.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _budget(spec: str | None):
    from .backends import Budget

    b = Budget()
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        key, sep, val = item.partition("=")
        if not sep or key not in ("cpu", "gpu", "wall"):
            raise ValueError(f"--budget expects cpu=,gpu=,wall= (minutes), got {item!r}")
        setattr(b, f"{key}_min", float(val))
    return b


def _review(args: argparse.Namespace) -> int:
    from .backends import get_backend
    from .review import ReviewError, review

    try:
        budget = _budget(args.budget)
    except ValueError as exc:
        print(exc)
        return 2
    try:
        v = review(Path(args.dir), args.card_id, get_backend(args.backend, args.model), mode=args.mode,
                   budget=budget, model=args.model, force=args.force,
                   history_dir=Path(args.history) if args.history else None)
    except ReviewError as exc:
        print(f"not reviewed: {exc}")
        return 2 if "no such card" in str(exc) else 1
    lv = v["levels"]
    l3 = lv["L3_worth_now"]
    by_id = {e.get("id"): e for e in v.get("evidence") or [] if isinstance(e, dict)}
    why = "; ".join(str(by_id[b].get("note") or by_id[b].get("path")) for b in l3.get("basis") or [] if b in by_id)
    # The scientist's interface is one line; the four levels are the ledger's and stay in the file.
    print(f"{v['card_id']}: worth running now = {l3['answer']} (confidence {l3.get('confidence')})"
          + (f" — {why}" if why else ""))
    if v["suggestions"]["preconditions"]:
        print("  verify first: " + "; ".join(v["suggestions"]["preconditions"]))
    if v["suggestions"]["cheaper_variants"]:
        print("  cheaper: " + "; ".join(v["suggestions"]["cheaper_variants"]))
    l2 = lv["L2_effect"]
    print(f"  (levels: runs={lv['L0_runs']['answer']} valid={lv['L1_valid']['answer']} "
          f"effect={l2.get('interval')} @{l2.get('confidence')}; {v['backend']}, skill {v['wma_skill']}, {v['mode']}; "
          f"full verdict beside the card)")
    return 0


def _ledger(args: argparse.Namespace) -> int:
    from . import ledger

    summary = ledger.summarize(ledger.rows([Path(d) for d in args.dirs]))
    print(ledger.to_csv(summary) if args.csv else ledger.render(summary), end="" if args.csv else "\n")
    return 0


def _replay(args: argparse.Namespace) -> int:
    from . import replay
    from .backends import get_backend

    try:
        budget = _budget(args.budget)
    except ValueError as exc:
        print(exc)
        return 2
    out = Path(args.out)
    samples = replay.build_samples(Path(args.corpus), out, side=args.side, sample=args.sample, seed=args.seed)
    print(f"{len(samples)} samples under {out}")
    if args.build_only:
        return 0
    counts = replay.run_replay(out, get_backend(args.backend, args.model), budget=budget, model=args.model,
                               limit=args.limit)
    print(", ".join(f"{k}={v}" for k, v in counts.items()))
    return 0 if counts.get("errors", 0) == 0 else 1


def register(sub: argparse._SubParsersAction) -> None:
    wp = sub.add_parser("wma", help="the world-model agent: review a card, ledger, replay")
    cmds = wp.add_subparsers(dest="cmd", required=True)

    r = cmds.add_parser("review", help="ask a backend for a verdict on one card (writes exp-NN.verdict.json)")
    r.add_argument("--dir", required=True)
    r.add_argument("card_id")
    r.add_argument("--backend", choices=("heuristic", "claude", "codex"), default="heuristic")
    r.add_argument("--model")
    r.add_argument("--mode", choices=("offline", "online"), default="online")
    r.add_argument("--budget", help="cpu=,gpu=,wall= in minutes")
    r.add_argument("--history", help="read-only directory of other runs' cards")
    r.add_argument("--force", action="store_true", help="allow a verdict on a card that already has a result")
    r.set_defaults(func=_review)

    lg = cmds.add_parser("ledger", help="summarise every verdict under the given directories")
    lg.add_argument("dirs", nargs="+")
    lg.add_argument("--csv", action="store_true")
    lg.set_defaults(func=_ledger)

    rp = cmds.add_parser("replay", help="offline replay over the historical card corpus")
    rp.add_argument("--corpus", required=True, help="results/exp-cards/<split> directory with train/ and test/")
    rp.add_argument("--out", required=True)
    rp.add_argument("--side", choices=("train", "test"), default="train")
    rp.add_argument("--sample", type=int, help="random subset of (run, card) samples")
    rp.add_argument("--seed", type=int, default=0)
    rp.add_argument("--backend", choices=("heuristic", "claude", "codex"), default="heuristic")
    rp.add_argument("--model")
    rp.add_argument("--budget", help="cpu=,gpu=,wall= in minutes")
    rp.add_argument("--limit", type=int, help="review at most this many samples this invocation")
    rp.add_argument("--build-only", action="store_true", help="build the sessions, review nothing")
    rp.set_defaults(func=_replay)
