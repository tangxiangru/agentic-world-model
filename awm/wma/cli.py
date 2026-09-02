"""``awm wma``: review cards, check status, read the ledger, replay the corpus.

``register`` needs only argparse; the package's modules are imported inside
the handlers, like the other ``awm`` command groups.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _budget(spec: str | None):
    from .backends import Budget

    b = Budget()
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        key, sep, val = item.partition("=")
        if not sep or key not in ("cpu", "gpu", "wall", "turns"):
            raise ValueError(f"--budget expects cpu=,gpu=,wall= (minutes) and turns=, got {item!r}")
        if key == "turns":
            b.max_turns = int(val)
        else:
            setattr(b, f"{key}_min", float(val))
    return b


def _one_line(v: dict) -> str:
    """What the scientist reads: the L3 answer, its confidence, the evidence behind it, the suggestions."""
    lv = v["levels"]
    l3 = lv["L3_worth_now"]
    by_id = {e.get("id"): e for e in v.get("evidence") or [] if isinstance(e, dict)}
    why = "; ".join(str(by_id[b].get("note") or by_id[b].get("path")) for b in l3.get("basis") or [] if b in by_id)
    lines = [f"{v['card_id']}: worth running now = {l3['answer']} (confidence {l3.get('confidence')})"
             + (f" — {why}" if why else "")]
    if v["suggestions"]["preconditions"]:
        lines.append("  verify first: " + "; ".join(v["suggestions"]["preconditions"]))
    if v["suggestions"]["cheaper_variants"]:
        lines.append("  cheaper: " + "; ".join(v["suggestions"]["cheaper_variants"]))
    l2 = lv["L2_effect"]
    lines.append(f"  (levels: runs={lv['L0_runs']['answer']} valid={lv['L1_valid']['answer']} "
                 f"effect={l2.get('interval')} @{l2.get('confidence')}; {v['backend']}, skill {v['wma_skill']}, "
                 f"{v['mode']}; full verdict beside the card)")
    return "\n".join(lines)


def _rank_key(v: dict) -> tuple:
    l3 = v["levels"]["L3_worth_now"]
    order = {"yes": 0, "defer": 1, "no": 2}
    return (order.get(l3.get("answer"), 3), -float(l3.get("confidence") or 0))


def _pid_path(session_dir: Path, card_id: str, tag: str | None) -> Path:
    suffix = f".review.{tag}.pid" if tag else ".review.pid"
    return Path(session_dir) / "memory" / "cards" / f"{card_id}{suffix}"


def _alive(pid_path: Path) -> bool:
    try:
        pid = int(pid_path.read_text().strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def _detach(args: argparse.Namespace) -> int:
    """Re-run this very command without --background as a detached process; return at once."""
    from . import schema

    try:
        for cid in args.card_id:
            schema.verdict_path(Path("x.yaml"), tag=args.tag)  # validates the tag before we fork
    except ValueError as exc:
        print(exc)
        return 2
    argv = [sys.executable, "-m", "awm.cli", "wma", "review", "--dir", args.dir, *args.card_id,
            "--backend", args.backend, "--mode", args.mode, "--jobs", str(args.jobs)]
    if args.model:
        argv += ["--model", args.model]
    if args.effort:
        argv += ["--effort", args.effort]
    if args.budget:
        argv += ["--budget", args.budget]
    if args.history:
        argv += ["--history", args.history]
    if args.tag:
        argv += ["--tag", args.tag]
    if args.force:
        argv.append("--force")
    cdir = Path(args.dir) / "memory" / "cards"
    cdir.mkdir(parents=True, exist_ok=True)
    log = cdir / (f"review.{args.tag}.log" if args.tag else "review.log")
    # The child must import the same `awm` this process is running, wherever that is.
    import awm

    env = dict(os.environ)
    pkg_root = str(Path(awm.__file__).resolve().parent.parent)
    env["PYTHONPATH"] = pkg_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    with log.open("a") as fh:
        proc = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True, env=env)
    for cid in args.card_id:
        _pid_path(args.dir, cid, args.tag).write_text(f"{proc.pid}\n")
    print(f"review of {', '.join(args.card_id)} started in the background (pid {proc.pid}); "
          f"keep working. `awm wma status --dir {args.dir}` shows when the verdict is in; log: {log}")
    return 0


def _review(args: argparse.Namespace) -> int:
    from . import schema
    from .backends import get_backend
    from .review import ReviewError, review

    if args.background:
        return _detach(args)
    try:
        budget = _budget(args.budget)
    except ValueError as exc:
        print(exc)
        return 2
    if args.tag is not None and not schema.TAG_RE.match(args.tag):
        print(f"--tag must match {schema.TAG_RE.pattern}")
        return 2
    backend = get_backend(args.backend, args.model, args.effort)
    history = Path(args.history) if args.history else None

    def one(cid: str):
        try:
            v = review(Path(args.dir), cid, backend, mode=args.mode, budget=budget, model=args.model,
                       force=args.force, history_dir=history, tag=args.tag, effort=args.effort)
            return cid, v, None
        except ReviewError as exc:
            return cid, None, str(exc)
        finally:
            pid = _pid_path(args.dir, cid, args.tag)
            if pid.is_file():
                pid.unlink()

    jobs = max(1, min(args.jobs, len(args.card_id)))
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(one, args.card_id))
    rc = 0
    done = []
    for cid, v, err in results:
        if err:
            print(f"{cid}: not reviewed: {err}")
            rc = 2 if "no such card" in err else 1
        else:
            print(_one_line(v))
            done.append(v)
    if len(done) > 1:
        ranked = sorted(done, key=_rank_key)
        print("ranking by worth-now: " + " > ".join(
            f"{v['card_id']} ({v['levels']['L3_worth_now']['answer']} {v['levels']['L3_worth_now'].get('confidence')})"
            for v in ranked))
    return rc


def _status(args: argparse.Namespace) -> int:
    from . import schema

    cdir = Path(args.dir) / "memory" / "cards"
    if not cdir.is_dir():
        print(f"no cards under {args.dir}")
        return 2
    for card in sorted(cdir.glob("exp-*.yaml")):
        cid = card.stem
        verdicts = sorted(cdir.glob(f"{cid}.verdict*.json"))
        pending = [p for p in cdir.glob(f"{cid}.review*.pid") if _alive(p)]
        if verdicts:
            for vp in verdicts:
                try:
                    v = schema.load_verdict(vp)
                except ValueError:
                    print(f"{cid}: unreadable verdict {vp.name}")
                    continue
                l3 = v["levels"]["L3_worth_now"]
                tag = schema.VERDICT_FILE_RE.match(vp.name).group(2)
                print(f"{cid}: worth running now = {l3['answer']} ({l3.get('confidence')})"
                      f"{' [' + tag + ']' if tag else ''} — {v['backend']}, {v['mode']}")
        if pending:
            print(f"{cid}: review running (pid {pending[0].read_text().strip()})")
        if not verdicts and not pending:
            print(f"{cid}: no verdict")
    return 0


def _ledger(args: argparse.Namespace) -> int:
    from . import ledger

    dirs = [Path(d) for d in args.dirs]
    summary = ledger.summarize(ledger.rows(dirs))
    print(ledger.to_csv(summary) if args.csv else ledger.render(summary), end="" if args.csv else "\n")
    rej = ledger.rejected(dirs)
    if rej["n"] and not args.csv:
        print(f"rejected (not verdicts, still paid for): {rej['n']} file(s), usd {rej['cost_usd_sum']}")
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
    print(f"{len(samples)} samples under {out}; set fingerprint {replay.fingerprint(samples)[:16]}")
    if args.build_only:
        return 0
    counts = replay.run_replay(out, get_backend(args.backend, args.model, args.effort), budget=budget,
                               model=args.model, effort=args.effort, limit=args.limit, jobs=args.jobs)
    print(", ".join(f"{k}={v}" for k, v in counts.items()))
    return 0 if counts.get("errors", 0) == 0 else 1


def register(sub: argparse._SubParsersAction) -> None:
    wp = sub.add_parser("wma", help="the world-model agent: review cards, status, ledger, replay")
    cmds = wp.add_subparsers(dest="cmd", required=True)

    r = cmds.add_parser("review", help="ask a backend for a verdict on one or more cards (writes exp-NN.verdict[.tag].json)")
    r.add_argument("--dir", required=True)
    r.add_argument("card_id", nargs="+", help="exp-NN ...; several cards are reviewed in parallel and ranked")
    r.add_argument("--backend", choices=("heuristic", "claude", "codex"), default="heuristic")
    r.add_argument("--model")
    r.add_argument("--effort", default="high",
                   help="reasoning effort passed to the agent CLI (never inherited from its settings)")
    r.add_argument("--mode", choices=("offline", "online"), default="online")
    r.add_argument("--budget", help="cpu=,gpu=,wall= in minutes, turns= for the agent")
    r.add_argument("--history", help="read-only directory of other runs' cards")
    r.add_argument("--tag", help="name this verdict (exp-NN.verdict.<tag>.json) so several agents can review one card")
    r.add_argument("--jobs", type=int, default=4, help="how many cards to review concurrently")
    r.add_argument("--background", action="store_true",
                   help="return at once; the review runs detached and the verdict appears when it is done")
    r.add_argument("--force", action="store_true", help="allow a verdict on a card that already has a result")
    r.set_defaults(func=_review)

    st = cmds.add_parser("status", help="which cards have a verdict, which reviews are still running")
    st.add_argument("--dir", required=True)
    st.set_defaults(func=_status)

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
    rp.add_argument("--effort", default="high",
                    help="reasoning effort passed to the agent CLI (never inherited from its settings)")
    rp.add_argument("--budget", help="cpu=,gpu=,wall= in minutes, turns= for the agent")
    rp.add_argument("--limit", type=int, help="review at most this many samples this invocation")
    rp.add_argument("--jobs", type=int, default=1, help="how many samples to review concurrently")
    rp.add_argument("--build-only", action="store_true", help="build the sessions, review nothing")
    rp.set_defaults(func=_replay)
