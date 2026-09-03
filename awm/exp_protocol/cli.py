"""``awm exp_protocol``: the protocol as commands. Each one reads files, checks, writes a record, exits.

``register`` needs only argparse; the package's modules are imported inside
the handlers, like the other ``awm`` command groups, so building the parser
never pays for — or fails on — code the invoked command does not use.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CARD_FILE_RE = re.compile(r"^exp-(\d+)\.yaml$")


def _cards_dir(args: argparse.Namespace) -> Path:
    from .lineage import cards_dir

    return cards_dir(Path(args.dir))


def _card_path(args: argparse.Namespace) -> Path | None:
    p = _cards_dir(args) / f"{args.card_id}.yaml"
    if not p.is_file():
        print(f"no such card: {p}")
        return None
    return p


def _next_id(cards_directory: Path) -> str:
    nums = [int(m.group(1)) for p in cards_directory.glob("exp-*.yaml") if (m := CARD_FILE_RE.match(p.name))]
    return f"exp-{(max(nums) + 1) if nums else 1:02d}"


def _print_questions(card: dict) -> None:
    from .questions import missing_fields

    missing = missing_fields(card)
    if missing:
        print("fill in, in this order:")
        for field_, q in missing:
            print(f"  {field_}: {q}")


# ----------------------------------------------------------------- commands

def _new(args: argparse.Namespace) -> int:
    from . import schema

    cdir = _cards_dir(args)
    cdir.mkdir(parents=True, exist_ok=True)
    card_id = args.id or _next_id(cdir)
    path = cdir / f"{card_id}.yaml"
    if path.exists():
        print(f"{path} already exists")
        return 2
    card = schema.minimal_card(card_id)
    schema.dump_card(path, card)
    print(f"wrote {path}")
    _print_questions(card)
    return 0


def _check(args: argparse.Namespace) -> int:
    from . import schema

    path = _card_path(args)
    if path is None:
        return 2
    card = schema.load_card(path)
    report = schema.validate_plan(card, Path(args.dir))
    print(report.render())
    _print_questions(card)
    return 0 if report.ok else 1


def _run_preflight(args: argparse.Namespace, card: dict, path: Path) -> dict:
    from . import lock, preflight

    report = preflight.run_preflight(card, Path(args.dir))
    lock.preflight_path(path).write_text(json.dumps(report, indent=2) + "\n")
    print(preflight.render(report))
    return report


def _preflight(args: argparse.Namespace) -> int:
    from . import schema

    path = _card_path(args)
    if path is None:
        return 2
    report = _run_preflight(args, schema.load_card(path), path)
    return 0 if report["summary"]["fail"] == 0 else 1


def _parse_overrides(raw: list[str] | None) -> dict[str, str] | None:
    """``--override check_id=reason`` pairs; None on a malformed or unknown check id (printed)."""
    from .preflight import CHECKS

    out: dict[str, str] = {}
    for item in raw or []:
        check_id, sep, reason = item.partition("=")
        if not sep or not reason.strip():
            print(f"--override needs check_id=reason, got {item!r}")
            return None
        if check_id not in CHECKS:
            print(f"--override names no such check: {check_id!r} (known: {', '.join(CHECKS)})")
            return None
        out[check_id] = reason.strip()
    return out


def _lock(args: argparse.Namespace) -> int:
    from . import lock, schema

    path = _card_path(args)
    if path is None:
        return 2
    overrides = _parse_overrides(args.override)
    if overrides is None:
        return 2
    card = schema.load_card(path)
    plan = schema.validate_plan(card, Path(args.dir))
    if not plan.ok:
        print(plan.render())
        _print_questions(card)
        print("not locked: fix the errors above")
        return 1
    report = _run_preflight(args, card, path)
    failing = [r["check"] for r in report["results"] if r["status"] == "fail"]
    blocking = [c for c in failing if c not in overrides]
    for c in failing:
        if c in overrides:
            print(f"overridden: {c} — {overrides[c]}")
    if blocking:
        print("not locked: preflight failed on " + ", ".join(blocking)
              + " (a wrong check can be overridden with --override check_id=reason; the reason is recorded)")
        return 1
    try:
        info = lock.write_lock(path, card, report["summary"], relock_reason=args.relock, overrides=overrides)
    except lock.LockExists:
        print(f"not locked: {lock.lock_path(path)} already exists. Re-locking after a change needs "
              "--relock \"<reason>\"; the previous hash is kept in the lock file")
        return 1
    print(f"locked {card['card_id']} at {info['locked_at']} (plan {info['plan_sha256'][:12]})")
    if info["relocked_from"]:
        print(f"re-locked {len(info['relocked_from'])} time(s); previous hashes kept")
    _wma_gate(args, path, card["card_id"])
    return 0


def _wma_gate(args: argparse.Namespace, card_path: Path, card_id: str) -> dict:
    """The verdict is part of the lock (2026-09-03): when a world-model agent is attached, ask it
    now and wait for its answer before handing the launch back. The scientist may prepare while the
    wait runs, but the protocol forbids starting the run before this returns. Whatever happened —
    delivered, failed, timed out, not attached, skipped with a reason — is written into the lock file
    under `wma`, so the record shows whether the verdict was in the loop for this card."""
    from .. import wma_client
    from . import lock

    session = Path(args.dir)
    skip_reason = getattr(args, "no_wma_wait", None)
    if skip_reason:
        result = {"state": "skipped", "reason": skip_reason, "waited_s": 0.0, "verdict_path": None,
                  "error": None, "request_id": None, "requested_at": None}
        print(f"WMA review skipped by request: {skip_reason} (recorded in the lock)")
    elif not wma_client.sidecar_attached(session):
        result = {"state": "not_attached", "waited_s": 0.0, "verdict_path": None, "error": None,
                  "request_id": None, "requested_at": None}
    else:
        try:
            result = wma_client.review_and_wait(session, card_id,
                                                timeout_min=getattr(args, "wma_timeout_min", None)
                                                or wma_client.DEFAULT_WAIT_MIN)
        except ValueError as exc:   # the card is not reviewable; the lock stands, the record says why
            result = {"state": "failed", "waited_s": 0.0, "verdict_path": None, "error": str(exc),
                      "request_id": None, "requested_at": None}
            print(f"WMA review not queued: {exc}")
    lock.annotate_lock(card_path, "wma", result)
    return result


def _close(args: argparse.Namespace) -> int:
    from . import lineage, lock, schema

    path = _card_path(args)
    if path is None:
        return 2
    card = schema.load_card(path)
    result = schema.validate_result(card)
    integrity = lock.verify_lock(path, card)
    for r in (result, integrity):
        if r.problems:
            print(r.render())
    if not (result.ok and integrity.ok):
        print("not closed")
        return 1
    info = lock.read_lock(path) or {}
    if info.get("relocked_from"):
        n = len(info["relocked_from"])
        print(f"note: this card was re-locked {n} time{'s' if n != 1 else ''} before the run; reasons are in the lock file")
    out = lineage.write_index(Path(args.dir))
    print(f"closed {card['card_id']}; index at {out}")
    return 0


def _index(args: argparse.Namespace) -> int:
    from . import lineage

    out = lineage.write_index(Path(args.dir))
    print(f"wrote {out}")
    cards = lineage.load_cards(_cards_dir(args))
    for p in lineage.starting_points(cards):
        where = p["checkpoint"] or "(checkpoint gone: rerun " + " <- ".join(p["chain"]) + ")"
        print(f"  start from {p['card_id']} [{p['level']}] {p['measurement']} {where}")
    return 0


def _chain(args: argparse.Namespace) -> int:
    from . import lineage

    cards = lineage.load_cards(_cards_dir(args))
    if args.card_id not in cards:
        print(f"no such card: {args.card_id}")
        return 2
    print(" <- ".join(lineage.chain(cards, args.card_id)))
    return 0


def _collect(args: argparse.Namespace) -> int:
    from . import collect as collect_mod

    rows = collect_mod.collect([Path(d) for d in args.dirs])
    if args.csv:
        print(collect_mod.to_csv(rows), end="")
    else:
        for r in rows:
            print("  ".join(f"{k}={r[k]}" for k in collect_mod.COLUMNS))
    return 0


def _install(args: argparse.Namespace) -> int:
    from . import install as install_mod

    try:
        written = install_mod.install(Path(args.target), args.tool)
    except install_mod.InstallError as exc:
        print(f"not installed: {exc}")
        return 2
    for p in written:
        print(f"wrote {p}")
    return 0


# ----------------------------------------------------------------- parser

def register(sub: argparse._SubParsersAction) -> None:
    ep = sub.add_parser("exp_protocol",
                        help="the experiment protocol: card, check, preflight, lock, close, index")
    cmds = ep.add_subparsers(dest="cmd", required=True)

    def with_dir(name: str, help_: str, card: bool = True):
        c = cmds.add_parser(name, help=help_)
        c.add_argument("--dir", required=True, help="the scientist's task/session directory")
        if card:
            c.add_argument("card_id", help="exp-NN")
        return c

    n = with_dir("new", "write a fresh card with every required field empty, and print the questions",
                 card=False)
    n.add_argument("--id", help="card id; default is the next exp-NN")
    n.set_defaults(func=_new)
    with_dir("check", "validate sections 0-4 and print what is still missing").set_defaults(func=_check)
    with_dir("preflight", "run the pre-flight checks; write exp-NN.preflight.json").set_defaults(func=_preflight)
    lk = with_dir("lock", "check + preflight, pin sections 0-4, the script and the data, then ask the "
                          "world-model agent (if attached) and wait for its verdict")
    lk.add_argument("--relock", metavar="REASON", help="lock again after a change; the previous hash is kept")
    lk.add_argument("--override", action="append", metavar="CHECK=REASON",
                    help="let a failing preflight check through; recorded in the lock (repeatable)")
    lk.add_argument("--no-wma-wait", metavar="REASON", dest="no_wma_wait",
                    help="lock without asking the world-model agent; the reason is recorded in the lock")
    lk.add_argument("--wma-timeout-min", type=float, dest="wma_timeout_min",
                    help="how long lock waits for the verdict (default 20 min)")
    lk.set_defaults(func=_lock)
    with_dir("close", "validate sections 5-6, re-check the lock, rebuild the index").set_defaults(func=_close)
    with_dir("index", "rebuild memory/index.md and list starting points", card=False).set_defaults(func=_index)
    with_dir("chain", "print the parent chain back to the base model").set_defaults(func=_chain)

    c = cmds.add_parser("collect", help="per-session numbers for comparing protocol variants")
    c.add_argument("dirs", nargs="+")
    c.add_argument("--csv", action="store_true")
    c.set_defaults(func=_collect)

    i = cmds.add_parser("install", help="copy the scientist skill into a task dir (never the meta skill)")
    i.add_argument("--target", required=True)
    i.add_argument("--tool", choices=("claude", "codex", "both"), default="both")
    i.set_defaults(func=_install)
