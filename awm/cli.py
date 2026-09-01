"""Command line entry point: ``awm``.

Thin wiring over the library. Everything it prints is meant to be readable in a
terminal and greppable in a log; anything a program should consume comes out of
the library or the parquet index instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from awm import paths


def _fetch(args: argparse.Namespace) -> int:
    from awm.traj import fetch

    sources = list(fetch.FETCHERS) if args.source == "all" else [args.source]
    for source in sources:
        print(f"fetching {source} ...")
        if source == "posttrainbench":
            benchmarks = fetch.PTB_CORE_BENCHMARKS
            if args.observe or args.all:
                benchmarks = benchmarks + fetch.PTB_OBSERVE_BENCHMARKS
            result = fetch.fetch_posttrainbench(
                configs=fetch.ALL_CONFIGS if args.all else fetch.PTB_DEFAULT_CONFIGS,
                benchmarks=benchmarks,
            )
        else:
            result = fetch.FETCHERS[source]()
        print("  " + str(result))
    return 0


def _convert_pi(limit: int | None) -> int:
    from awm.traj import convert_pi

    raw = paths.raw_dir("pi_speedrun")
    if not (raw / "traces").is_dir():
        print(
            f"pi_speedrun not fetched at {raw} — run `awm traj fetch pi_speedrun`", file=sys.stderr
        )
        return 1
    metas = convert_pi.convert_all(raw, paths.events_dir("pi_speedrun"), limit=limit)
    print(f"pi_speedrun: {len(metas)} runs -> {paths.events_dir('pi_speedrun')}")
    return 0


def _convert_ptb(limit: int | None) -> int:
    from awm.traj import posttrainbench as ptb

    raw = paths.raw_dir("posttrainbench")
    if not raw.is_dir():
        print(
            f"posttrainbench not fetched at {raw} — run `awm traj fetch posttrainbench`",
            file=sys.stderr,
        )
        return 1
    out = paths.events_dir("posttrainbench")
    runs = list(ptb.iter_run_dirs(raw))
    if limit:
        runs = runs[:limit]
    failed = []
    skipped = []
    for run in runs:
        try:
            ptb.convert_run_dir(run, out)
        except ptb.NoAgentOutput:
            # The CLI died before emitting anything — `opencode: command not
            # found`, an unknown flag, no CUDA. 41 runs, none with a
            # metrics.json. Nothing was published to convert, so this is not a
            # conversion failure and must not red the exit code.
            skipped.append(run.run_id)
        except Exception as exc:  # one malformed run must not abandon the batch
            # run_id, not the directory name: two agent configurations hold the
            # same 28 run names, so the bare name identifies neither.
            failed.append((run.run_id, exc))
    ok = len(runs) - len(failed) - len(skipped)
    print(f"posttrainbench: {ok}/{len(runs)} runs -> {out}")
    if skipped:
        print(f"  skipped {len(skipped)} run(s) with no agent output")
    for name in skipped:
        print(f"  SKIPPED {name}: no agent output", file=sys.stderr)
    for name, exc in failed:
        print(f"  FAILED {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 1 if failed else 0


def _convert(args: argparse.Namespace) -> int:
    rc = 0
    if args.source in ("all", "pi_speedrun"):
        rc |= _convert_pi(args.limit)
    if args.source in ("all", "posttrainbench"):
        rc |= _convert_ptb(args.limit)
    return rc


def _run(args: argparse.Namespace) -> int:
    """Run a task directory under Harbor with the host paths it needs.

    The generated compose files require absolute host paths (compose resolves a
    relative volume source against its own directory), and those paths differ per
    machine because ``data/`` is a symlink. Rather than committing one layout,
    the task files demand the variables and this fills them in.
    """
    import os
    import subprocess

    task_dir = Path(args.task).resolve()
    if not (task_dir / "task.toml").exists():
        print(f"no task.toml under {task_dir}", file=sys.stderr)
        return 1

    env = dict(os.environ)
    env.setdefault("AWM_AIRS_PREPARED", str(paths.data_root().resolve() / "assets/airs/prepared"))
    env.setdefault("AWM_FINEWEB_DIR", str(paths.data_root().resolve() / "assets/fineweb10B"))

    cmd = [args.harbor, "run", "-p", str(task_dir), "-a", args.agent, "-o", str(args.jobs_dir)]
    if args.model:
        cmd += ["-m", args.model]
    cmd += args.harbor_arg
    print(" ".join(cmd))
    return subprocess.run(cmd, env=env).returncode


def _spans(args: argparse.Namespace) -> int:
    from awm.traj import train_spans

    df = train_spans.build()
    path = train_spans.save(df, args.out)
    print(f"{len(df)} spans over {df['run_id'].nunique()} runs -> {path}")
    timed = df[df["sec"].notna()]
    print(f"  {len(timed)} timed, {len(df) - len(timed)} untimed (source carries no clock)")
    for kind, g in timed.groupby("kind"):
        print(f"  {kind}: {len(g)}, median {g['sec'].median() / 60:.1f} min")
    occ = train_spans.occupancy_by_run(df)
    if len(occ):
        over = (occ["sum_s"] > occ["occupied_s"]).sum()
        print(f"  {over} runs launched overlapping trainings (union, not sum, is the share)")
    return 0


def _index(args: argparse.Namespace) -> int:
    from awm.traj import index

    df = index.build()
    path = index.save(df)
    print(f"{len(df)} runs -> {path}")
    if len(df):
        by_source = df.groupby("source", dropna=False).size()
        for source, n in by_source.items():
            print(f"  {source}: {n}")
    return 0


def _split_list(args: argparse.Namespace) -> int:
    from awm import splits

    for sid in splits.list_ids():
        try:
            s = splits.load(sid)
            print(f"{sid}  [{s.benchmark}]  train={s.counts['train']} test={s.counts['test']}")
        except splits.SplitError:
            sel = splits.load_selection(sid)
            print(f"{sid}  [{sel.benchmark}]  tasks={len(sel.tasks)}")
    return 0


def _split_catalog(args: argparse.Namespace):
    from awm.traj import fetch

    path = paths.raw_dir("posttrainbench") / fetch.PTB_CATALOG
    if not path.exists():
        print(
            f"catalogue not fetched at {path} — run `awm split fetch {args.id}` "
            "or `awm traj fetch posttrainbench`",
            file=sys.stderr,
        )
        return None
    return fetch.ptb_catalog(), path.read_bytes()


def _split_check(args: argparse.Namespace) -> int:
    from awm import splits

    s = splits.load(args.id)
    got = _split_catalog(args)
    if got is None:
        return 1
    issues = splits.check(s, got[0], catalog_bytes=got[1])
    for issue in issues:
        print(f"  - {issue}")
    print(f"{len(issues)} issue(s)")
    return 1 if issues else 0


def _split_fetch(args: argparse.Namespace) -> int:
    from awm import splits
    from awm.traj import fetch

    s = splits.load(args.id)
    result = fetch.fetch_ptb_runs(s.train + s.test, revision=s.dataset["revision"])
    print("  " + str(result))
    return 0


def _ptb(args: argparse.Namespace) -> int:
    from awm import ptb_experiments as ptb

    try:
        manifest = ptb.load_manifest(args.manifest)
        if args.cmd == "check":
            issues = ptb.local_issues(manifest, require_context=not args.before_context_gate)
            if not args.local_only:
                issues += ptb.site_issues()
            for issue in issues:
                print(f"  - {issue}")
            print(f"{len(issues)} issue(s)")
            return 1 if issues else 0
        if args.cmd == "dry-run":
            for cell_id, command in ptb.dry_run(manifest, pilot=args.pilot):
                print(f"{cell_id}: {command}")
            return 0
        if args.cmd == "submit":
            receipt = ptb.submit(manifest, pilot=args.pilot)
            print(receipt)
            return 0
        if args.cmd == "context-smoke":
            for job in ptb.submit_context_smokes(manifest, args.cell):
                print(f"{job['cell_id']}: Slurm job {job['job_id']}")
            return 0
        if args.cmd == "audit":
            issues = ptb.audit_result(args.result_dir.resolve())
            for issue in issues:
                print(f"  - {issue}")
            print(f"{len(issues)} issue(s)")
            return 1 if issues else 0
        if args.cmd == "status":
            receipt = ptb.load_receipt(args.receipt)
            for job in ptb.receipt_status(receipt):
                print(
                    f"{job['cell_id']} job={job['job_id']} state={job['state']} "
                    f"result={job['result_dir'] or '<pending>'}"
                )
            return 0
        if args.cmd == "audit-receipt":
            receipt = ptb.load_receipt(args.receipt)
            all_issues = ptb.audit_receipt(receipt)
            issue_count = 0
            for cell_id, issues in all_issues.items():
                print(f"{cell_id}: {'PASS' if not issues else 'FAIL'}")
                for issue in issues:
                    print(f"  - {issue}")
                    issue_count += 1
            print(f"{issue_count} issue(s)")
            return 1 if issue_count else 0
        if args.cmd == "research-judges":
            output = ptb.submit_research_judges(ptb.load_receipt(args.receipt))
            print(output)
            return 0
    except ptb.ExperimentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    raise AssertionError(args.cmd)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="awm", description=__doc__)
    p.add_argument("--data-root", type=Path, help="override AWM_DATA_ROOT for this call")
    sub = p.add_subparsers(dest="group", required=True)

    traj = sub.add_parser("traj", help="fetch and convert trajectories").add_subparsers(
        dest="cmd", required=True
    )

    f = traj.add_parser("fetch", help="download an upstream release into raw/")
    f.add_argument("source", choices=["all", "pi_speedrun", "posttrainbench"])
    f.add_argument(
        "--all",
        action="store_true",
        help="posttrainbench: every agent configuration and benchmark (7.3 GB, 1842 runs) "
        "instead of the default four configurations on the five core benchmarks (0.6 GB)",
    )
    f.add_argument(
        "--observe",
        action="store_true",
        help="posttrainbench: also take the two LLM-judged benchmarks "
        "(arenahardwriting, healthbench)",
    )
    f.set_defaults(func=_fetch)

    c = traj.add_parser("convert", help="convert raw/ into the unified event schema")
    c.add_argument("--source", default="all", choices=["all", "pi_speedrun", "posttrainbench"])
    c.add_argument("--limit", type=int, help="convert at most N runs per source")
    c.set_defaults(func=_convert)

    i = traj.add_parser("index", help="rebuild the run index")
    i.set_defaults(func=_index)

    sn = traj.add_parser("spans", help="extract how long each training occupied the box")
    sn.add_argument(
        "--out",
        type=Path,
        default=Path("data/traj/derived/cc_train_spans_v1.parquet"),
    )
    sn.set_defaults(func=_spans)

    r = sub.add_parser(
        "run",
        help="run a task directory under Harbor, supplying this machine's host paths",
    )
    r.add_argument("task", help="path to a task directory (tasks/airs/<name>, ...)")
    r.add_argument("-a", "--agent", default="oracle")
    r.add_argument("-m", "--model")
    r.add_argument("-o", "--jobs-dir", type=Path, default=Path("data/smoke/jobs"))
    r.add_argument("--harbor", default="harbor", help="path to the harbor executable")
    r.add_argument(
        "--harbor-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="pass one argument through to harbor; repeat for more",
    )
    r.set_defaults(func=_run)

    sp = sub.add_parser("split", help="query the committed data splits").add_subparsers(
        dest="cmd", required=True
    )
    sl = sp.add_parser("list", help="list the committed splits and selections")
    sl.set_defaults(func=_split_list)

    sk = sp.add_parser("check", help="replay a split's rule over the pinned catalogue and compare")
    sk.add_argument("id", help="e.g. posttrainbench/gsm8k-gemma-holdout-v1")
    sk.set_defaults(func=_split_check)

    sf = sp.add_parser("fetch", help="download exactly a split's runs at its pinned revision")
    sf.add_argument("id")
    sf.set_defaults(func=_split_fetch)

    ep = sub.add_parser("ptb", help="validate, launch, and audit committed PTB batches")
    eps = ep.add_subparsers(dest="cmd", required=True)
    default_manifest = Path("experiments/posttrainbench/gsm8k-aime2025-opus5-4x4x2-batch1.yaml")
    for command_name in ("check", "dry-run", "submit"):
        command = eps.add_parser(command_name)
        command.add_argument("manifest", nargs="?", type=Path, default=default_manifest)
        if command_name in ("dry-run", "submit"):
            command.add_argument("--pilot", action="store_true")
        if command_name == "check":
            command.add_argument("--local-only", action="store_true")
            command.add_argument(
                "--before-context-gate",
                action="store_true",
                help="skip provider-context records while preparing G0-G4",
            )
        command.set_defaults(func=_ptb)
    context_smoke = eps.add_parser(
        "context-smoke", help="submit setup-specific provider/runtime probes for selected cells"
    )
    context_smoke.add_argument("manifest", nargs="?", type=Path, default=default_manifest)
    context_smoke.add_argument("--cell", action="append", required=True)
    context_smoke.set_defaults(func=_ptb)
    audit = eps.add_parser("audit")
    audit.add_argument("result_dir", type=Path)
    audit.add_argument("--manifest", type=Path, default=default_manifest)
    audit.set_defaults(func=_ptb)
    for command_name in ("status", "audit-receipt", "research-judges"):
        receipt_command = eps.add_parser(command_name)
        receipt_command.add_argument("receipt", type=Path)
        receipt_command.add_argument("--manifest", type=Path, default=default_manifest)
        receipt_command.set_defaults(func=_ptb)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.data_root:
        import os

        os.environ["AWM_DATA_ROOT"] = str(args.data_root)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
