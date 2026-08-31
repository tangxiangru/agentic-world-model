"""Command line entry point: ``awm``.

Thin wiring over the library. Everything it prints is meant to be readable in a
terminal and greppable in a log; anything a program should consume comes out of
the library or the parquet index instead.
"""

from __future__ import annotations

import argparse
import json
import os
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
        print(f"pi_speedrun not fetched at {raw} — run `awm traj fetch pi_speedrun`", file=sys.stderr)
        return 1
    metas = convert_pi.convert_all(raw, paths.events_dir("pi_speedrun"), limit=limit)
    print(f"pi_speedrun: {len(metas)} runs -> {paths.events_dir('pi_speedrun')}")
    return 0


def _convert_ptb(limit: int | None) -> int:
    from awm.traj import posttrainbench as ptb

    raw = paths.raw_dir("posttrainbench")
    if not raw.is_dir():
        print(f"posttrainbench not fetched at {raw} — run `awm traj fetch posttrainbench`",
              file=sys.stderr)
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
        except Exception as exc:  # noqa: BLE001  # one malformed run must not abandon batch
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
    return subprocess.run(cmd, env=env, check=False).returncode


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
        print(f"catalogue not fetched at {path} — run `awm split fetch {args.id}` "
              "or `awm traj fetch posttrainbench`", file=sys.stderr)
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
    catalog_path = result.path / fetch.PTB_CATALOG
    catalog_bytes = catalog_path.read_bytes()
    issues = splits.check(
        s,
        json.loads(catalog_bytes),
        catalog_bytes=catalog_bytes,
    )
    issues.extend(
        fetch.check_ptb_run_files(
            s.train + s.test,
            revision=s.dataset["revision"],
            dest=result.path,
        )
    )
    for issue in issues:
        print(f"  - {issue}", file=sys.stderr)
    if issues:
        print(f"split fetch failed validation: {len(issues)} issue(s)", file=sys.stderr)
        return 1
    print(
        f"split fetch validated: {len(s.train)} train + {len(s.test)} test runs "
        f"at {s.dataset['revision']}"
    )
    return 0


def _experiment_scaffold(args: argparse.Namespace) -> int:
    from awm.experiment import ExperimentBundle

    bundle = ExperimentBundle.scaffold(args.directory, args.id, args.title)
    print(bundle.card_path)
    return 0


def _experiment_validate(args: argparse.Namespace) -> int:
    from awm.experiment import ExperimentBundle

    bundle = ExperimentBundle(args.directory)
    card = bundle.card
    print(f"{card['experiment_id']}: valid {card['schema_version']}")
    return 0


def _experiment_freeze(args: argparse.Namespace) -> int:
    from awm.experiment import ExperimentBundle

    bundle = ExperimentBundle(args.directory)
    manifest = bundle.freeze()
    missing_tools = [p["phase_id"] for p in manifest["phases"] if not p["executable"]]
    print(f"{manifest['experiment_id']}: frozen -> {bundle.manifest_path}")
    if missing_tools:
        print("  executables not present on this machine (allowed before VM upload): "
              + ", ".join(missing_tools))
    return 0


def _experiment_run(args: argparse.Namespace) -> int:
    from awm.experiment import ExperimentBundle

    bundle = ExperimentBundle(args.directory)
    if args.detach:
        pid = bundle.run_detached()
        print(f"{bundle.card['experiment_id']}: queued as pid {pid}")
        return 0
    summary = bundle.run(queued_ok=args.worker)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["execution_status"] == "completed" else 1


def _experiment_observe(args: argparse.Namespace) -> int:
    from awm.experiment import ExperimentBundle, ExperimentError

    data = {}
    if args.data:
        try:
            data = json.loads(args.data)
        except json.JSONDecodeError as exc:
            raise ExperimentError(f"--data is not JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ExperimentError("--data must decode to a JSON object")
    ref = ExperimentBundle(args.directory).observe(
        args.kind, args.summary, phase_id=args.phase, data=data, artifact=args.artifact
    )
    print(ref)
    return 0


def _experiment_finalize(args: argparse.Namespace) -> int:
    from awm.experiment import ExperimentBundle

    bundle = ExperimentBundle(args.directory)
    result = bundle.finalize(args.result)
    assessment = result["scientist_assessment"]
    print(
        f"{result['experiment_id']}: closed; "
        f"outcome={assessment['outcome']['verdict']} "
        f"mechanism={assessment['mechanism']['verdict']} "
        f"decision={result['scientist_decision']['action']}"
    )
    return 0


def _experiment_status(args: argparse.Namespace) -> int:
    from awm.experiment import ExperimentBundle

    state = ExperimentBundle(args.directory).state
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


def _experiment_check(args: argparse.Namespace) -> int:
    from awm.experiment import open_experiments

    root = args.root or paths.data_root() / "experiments"
    found = open_experiments(root)
    for item in found:
        print(f"{item['experiment_id']}  {item['status']}  {item['path']}")
    if not found:
        print(f"no open experiments under {root}")
    return 1 if found else 0


# ---------------------------------------------------------------- awm wm

def _wm_session(args: argparse.Namespace):
    from awm.wm.runtime import Session

    return Session(args.dir)


def _wm_init(args: argparse.Namespace) -> int:
    import shlex

    from awm.wm.runtime import Session

    overrides = {
        "submission": str(Path(args.submission).resolve()) if args.submission else None,
        "memory_root": str(Path(args.memory_root).resolve()) if args.memory_root else None,
        "memory_readonly": True if args.memory_readonly else None,
        "official_argv": shlex.split(args.official_argv) if args.official_argv else None,
        "official_cwd": str(Path(args.official_cwd).resolve()) if args.official_cwd else None,
        "custom_argv": shlex.split(args.custom_argv) if args.custom_argv else None,
        "auto_relaunch": False if args.no_auto_relaunch else None,
        "spawn_worker": False if args.no_spawn_worker else None,
        "split_side": args.split_side,
        "submission_mode": args.submission_mode,
        "memory_sides": [x.strip() for x in args.memory_sides.split(",") if x.strip()] if args.memory_sides else None,
        "wma_model": args.wma_model,
        "wma_corpus_kind": args.wma_corpus_kind,
        "wma_corpus_root": str(Path(args.wma_corpus_root).resolve()) if args.wma_corpus_root else None,
        "wma_effort": args.wma_effort,
        "wma_max_budget_usd": args.wma_max_budget_usd,
        "wma_timeout_s": args.wma_timeout_s,
        "wma_validation_attempts": args.wma_validation_attempts,
    }
    s = Session.init(args.dir, arm=args.arm, **overrides)
    print(f"initialised {s.wm} (arm={s.config['arm']}, memory={s.config['memory_root']})")
    print(f"hook example: {s.wm / 'hook_example.py'}")
    return 0


def _wm_propose(args: argparse.Namespace) -> int:
    from awm.wm.runtime import print_ping

    s = _wm_session(args)
    ping = s.propose(args.card)
    print_ping(ping, s)
    return 0


def _wm_reply(args: argparse.Namespace) -> int:
    s = _wm_session(args)
    out = s.reply(args.ping, args.choose, why=args.why, amend=args.amend)
    printed = out.pop("printed", None)
    print(json.dumps(out, indent=2, sort_keys=True, default=str))
    if printed:
        print(printed)
    return 0


def _wm_checkpoint(args: argparse.Namespace) -> int:
    from awm.wm.runtime import print_ping

    s = _wm_session(args)
    code = s.checkpoint(args.card_id, args.checkpoint, step=args.step, final=args.final)
    st = s.state(args.card_id)
    if code == 3:
        py = st.get("pending_yield") or {}
        print(f"[{args.card_id}] YIELD at step {args.step}: exit after this save; the runtime evaluates "
              f"{py.get('standing') or ''} {[r['ping_id'] for r in py.get('requested', [])] or ''} "
              f"and relaunches from {args.checkpoint}".replace("  ", " "))
    elif code == 4:
        print(f"[{args.card_id}] ABORT: stop training; write result.yaml and finalize")
    for ping in s.mailbox(args.card_id).pending():
        print_ping(ping, s)
    return code


def _wm_worker(args: argparse.Namespace) -> int:
    s = _wm_session(args)
    out = s.run_worker(args.card_id)
    print(json.dumps(out, indent=2, sort_keys=True, default=str))
    return 0


def _wm_finalize(args: argparse.Namespace) -> int:
    s = _wm_session(args)
    out = s.finalize(args.card_id, args.result)
    print(json.dumps(out, indent=2, sort_keys=True, default=str))
    return 0


def _wm_status(args: argparse.Namespace) -> int:
    s = _wm_session(args)
    print(json.dumps(s.status(args.card_id), indent=2, sort_keys=True, default=str))
    return 0


def _wm_pending(args: argparse.Namespace) -> int:
    from awm.wm.runtime import print_ping

    s = _wm_session(args)
    pending = s.pending_replies()
    for ping in pending:
        print_ping(ping, s)
    if not pending:
        print("no pings awaiting a reply")
    return 1 if pending else 0


def _wm_memory_seed(args: argparse.Namespace) -> int:
    s = _wm_session(args)
    n = s.memory.seed_from_exp_cards(Path(args.results_dir), side=args.side)
    print(f"seeded {n} reconstructed cards from {args.results_dir} ({args.side}); memory now {s.memory.stats()}")
    return 0


def _wm_memory_stats(args: argparse.Namespace) -> int:
    s = _wm_session(args)
    print(json.dumps(s.memory.stats(), indent=2))
    return 0


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

    sk = sp.add_parser(
        "check", help="replay a split's rule over the pinned catalogue and compare"
    )
    sk.add_argument("id", help="e.g. posttrainbench/gsm8k-gemma-holdout-v1")
    sk.set_defaults(func=_split_check)

    sf = sp.add_parser("fetch", help="download exactly a split's runs at its pinned revision")
    sf.add_argument("id")
    sf.set_defaults(func=_split_fetch)

    exp = sub.add_parser(
        "experiment", aliases=["exp"],
        help="create, run, observe, and review reproducible experiment cards",
    ).add_subparsers(dest="cmd", required=True)

    esc = exp.add_parser("scaffold", help="create a draft card and lifecycle state")
    esc.add_argument("directory", type=Path)
    esc.add_argument("--id", help="experiment id (defaults to the directory name)")
    esc.add_argument("--title")
    esc.set_defaults(func=_experiment_scaffold)

    ev = exp.add_parser("validate", help="validate a draft or frozen card")
    ev.add_argument("directory", type=Path)
    ev.set_defaults(func=_experiment_validate)

    ef = exp.add_parser("freeze", help="freeze a validated plan and snapshot its inputs")
    ef.add_argument("directory", type=Path)
    ef.set_defaults(func=_experiment_freeze)

    er = exp.add_parser("run", help="run every phase and prepare scientist review")
    er.add_argument("directory", type=Path)
    er.add_argument("--detach", action="store_true", help="run under a detached local worker")
    er.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    er.set_defaults(func=_experiment_run)

    eo = exp.add_parser("observe", help="append a grounded intermediate observation")
    eo.add_argument("directory", type=Path)
    eo.add_argument("--kind", required=True)
    eo.add_argument("--summary", required=True)
    eo.add_argument("--phase")
    eo.add_argument("--data", help="JSON object with measurement details")
    eo.add_argument("--artifact", help="path to the observed artifact")
    eo.set_defaults(func=_experiment_observe)

    efin = exp.add_parser("finalize", help="validate and record the scientist's result")
    efin.add_argument("directory", type=Path)
    efin.add_argument("--result", type=Path, help="alternate completed result YAML")
    efin.set_defaults(func=_experiment_finalize)

    es = exp.add_parser("status", help="show the lifecycle state")
    es.add_argument("directory", type=Path)
    es.set_defaults(func=_experiment_status)

    ec = exp.add_parser("check", help="find queued, running, or unreviewed experiments")
    ec.add_argument("--root", type=Path)
    ec.set_defaults(func=_experiment_check)
    wm = sub.add_parser("wm", help="the world-model agent protocol: propose, reply, checkpoint, finalize")
    wm.add_argument("--dir", type=Path,
                    default=Path(os.environ.get("AWM_SESSION_DIR") or Path.cwd()),
                    help="the scientist's session directory (default: $AWM_SESSION_DIR or cwd)")
    wmc = wm.add_subparsers(dest="cmd", required=True)

    wi = wmc.add_parser("init", help="create wm/ with config.yaml, inbox.md, hook_example.py (harness step)")
    wi.add_argument("--arm", default="null", choices=["null", "retrieval", "llm", "predictor"])
    wi.add_argument("--submission", help="path that adopt will point at the sealed checkpoint")
    wi.add_argument("--memory-root", help="WMA memory location (default $AWM_WM_MEMORY or <data>/wm-memory)")
    wi.add_argument("--memory-readonly", action="store_true", help="held-out sessions: read memory, never write")
    wi.add_argument("--split-side", default="train", choices=["train", "test"])
    wi.add_argument("--submission-mode", default=None, choices=["symlink", "copy"],
                    help="adopt links (default) or copies the sealed checkpoint into --submission")
    wi.add_argument("--memory-sides", default=None, help="comma list of split sides the agent may retrieve from (default train)")
    wi.add_argument("--wma-model", default=None,
                    help="explicit Vertex Claude model for the llm sidecar (or set AWM_WMA_MODEL)")
    wi.add_argument("--wma-corpus-kind", choices=["cards", "raw"], default=None,
                    help="complete historical source exposed to the llm arm (default cards)")
    wi.add_argument("--wma-corpus-root", default=None,
                    help="read-only indexed prior-run root for --wma-corpus-kind raw")
    wi.add_argument("--wma-effort", choices=["low", "medium", "high", "xhigh", "max"], default=None)
    wi.add_argument("--wma-max-budget-usd", type=float, default=None,
                    help="logical-call Vertex cap, divided across validation attempts (default 1.0)")
    wi.add_argument("--wma-timeout-s", type=float, default=None,
                    help="logical wall timeout shared across validation attempts (default 900)")
    wi.add_argument("--wma-validation-attempts", type=int, choices=range(1, 6), default=None,
                    help="bounded fresh attempts for invalid model output (default 1)")
    wi.add_argument("--official-argv", help="shell string with {checkpoint} {n} {out}; default runs evaluate.py")
    wi.add_argument("--official-cwd", help="cwd for the official evaluator (default: session dir)")
    wi.add_argument("--custom-argv", help="shell string with {checkpoint} {items} {out} {n}; default awm.wm.score_items")
    wi.add_argument("--no-auto-relaunch", action="store_true", help="the scientist resumes training by hand")
    wi.add_argument("--no-spawn-worker", action="store_true", help="run `awm wm worker` yourself after a yield")
    wi.set_defaults(func=_wm_init)

    wp = wmc.add_parser("propose", help="issue an experiment card; returns the brief")
    wp.add_argument("card", type=Path)
    wp.set_defaults(func=_wm_propose)

    wr = wmc.add_parser("reply", help="answer a ping")
    wr.add_argument("ping", help="<card_id>/<ping_id>, e.g. exp-03/p-2")
    wr.add_argument("--choose", required=True)
    wr.add_argument("--why")
    wr.add_argument("--amend", help="revised card file (brief: amend)")
    wr.set_defaults(func=_wm_reply)

    wk = wmc.add_parser("checkpoint", help="the training hook; exit 0 continue, 3 yield, 4 abort")
    wk.add_argument("card_id")
    wk.add_argument("checkpoint", type=Path)
    wk.add_argument("--step", type=int, required=True)
    wk.add_argument("--final", action="store_true")
    wk.set_defaults(func=_wm_checkpoint)

    ww = wmc.add_parser("worker", help="process a pending yield (spawned by the hook; run by hand with --no-spawn-worker)")
    ww.add_argument("card_id")
    ww.set_defaults(func=_wm_worker)

    wf = wmc.add_parser("finalize", help="validate sections 5-6 of the completed card, close it, adopt if asked")
    wf.add_argument("card_id")
    wf.add_argument("result", type=Path, help="the card file with result + conclusion filled in")
    wf.set_defaults(func=_wm_finalize)

    ws = wmc.add_parser("status", help="session or card state")
    ws.add_argument("card_id", nargs="?")
    ws.set_defaults(func=_wm_status)

    wpend = wmc.add_parser("pending", help="list pings awaiting a reply (exit 1 if any)")
    wpend.set_defaults(func=_wm_pending)

    wmem = wmc.add_parser("memory", help="WMA memory").add_subparsers(dest="memcmd", required=True)
    wseed = wmem.add_parser("seed", help="load reconstructed cards from results/exp-cards/<split> as precedents")
    wseed.add_argument("results_dir", type=Path)
    wseed.add_argument("--side", default="train", choices=["train", "test"])
    wseed.set_defaults(func=_wm_memory_seed)
    wstat = wmem.add_parser("stats")
    wstat.set_defaults(func=_wm_memory_stats)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.data_root:
        import os

        os.environ["AWM_DATA_ROOT"] = str(args.data_root)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
