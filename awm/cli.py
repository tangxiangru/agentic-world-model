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


# ---------------------------------------------------------------- awm wm  (the world-model agent's toolbelt)

def _wm_dir(args: argparse.Namespace) -> Path:
    base = Path(args.dir) if args.dir else Path(os.environ.get("AWM_SESSION_DIR") or Path.cwd())
    return base.resolve() / "wm"


def _wm_config(args: argparse.Namespace) -> dict:
    from awm.wm.schema import WMError, load_json

    path = _wm_dir(args) / "config.json"
    if not path.is_file():
        raise WMError(f"{path} missing; run `awm wm init` first")
    return load_json(path)


def _wm_roots(cfg: dict) -> list[Path]:
    return [Path(r) for r in (cfg.get("session_dir"), cfg.get("prior_runs_root"), cfg.get("memory_root")) if r]


def _wm_memory(cfg: dict):
    from awm.wm.memory import Memory

    root = cfg.get("memory_root")
    if not root:
        return None
    return Memory(Path(root), session=cfg["session_id"], arm=cfg["arm"], split_side=cfg.get("split_side", "train"),
                  readonly=bool(cfg.get("memory_readonly")), visible_sides=tuple(cfg.get("memory_sides") or ["train"]))


def _wm_init(args: argparse.Namespace) -> int:
    import time

    from awm.wm.schema import dump_json

    wm = _wm_dir(args)
    session_dir = wm.parent
    for sub in ("cards", "tmp"):
        (wm / sub).mkdir(parents=True, exist_ok=True)
    (wm / ("records.jsonl" if getattr(args, "mode", "consult") == "record" else "consults.jsonl")).touch()
    mode = getattr(args, "mode", "consult")
    if mode == "record" and args.arm != "null":
        print(f"WARNING: recorder mode ignores past evidence; forcing arm null (was {args.arm})", file=sys.stderr)
        args.arm = "null"
    cfg = {
        "schema_version": "awm-wm-config-v2",
        "session_id": f"{session_dir.name}-{int(time.time())}",
        "session_dir": str(session_dir),
        "mode": mode,
        "arm": args.arm,
        "prior_runs_root": str(Path(args.prior_runs).resolve()) if args.prior_runs else None,
        "memory_root": str(Path(args.memory_root).resolve()) if args.memory_root else None,
        "memory_sides": [x.strip() for x in (args.memory_sides or "train").split(",") if x.strip()],
        "memory_readonly": bool(args.memory_readonly),
        "split_side": args.split_side,
        "wma_model": args.wma_model,
        "base_model": args.base_model,
        "consult_api": ("SendMessage to the wma session; one response shape (awm-record-response-v1); no advice"
                        if mode == "record" else
                        "SendMessage to the wma session; one response shape (awm-consult-response-v1)"),
    }
    if cfg["arm"] in ("traj", "llm") and not cfg["prior_runs_root"]:
        print(f"WARNING: arm {cfg['arm']} without --prior-runs: the WMA has no raw runs to read", file=sys.stderr)
    if cfg["arm"] in ("retrieval", "llm") and not cfg["memory_root"]:
        print(f"WARNING: arm {cfg['arm']} without --memory-root: the WMA has no cards to search", file=sys.stderr)
    dump_json(wm / "config.json", cfg)
    print(f"initialised {wm} (arm={cfg['arm']}, prior_runs={cfg['prior_runs_root']}, memory={cfg['memory_root']})")
    return 0


def _wm_text(args: argparse.Namespace) -> str:
    if getattr(args, "text", None):
        return args.text
    if getattr(args, "plan", None):
        return Path(args.plan).read_text()
    return sys.stdin.read()


def _wm_draft_card(args: argparse.Namespace) -> int:
    import yaml

    from awm.wm import intake

    cfg = _wm_config(args)
    text = _wm_text(args)
    card_id = args.card_id or f"exp-{len(list((_wm_dir(args) / 'cards').glob('exp-*'))) + 1:02d}"
    card, questions = intake.draft_card(card_id, text, {}, Path(cfg["session_dir"]), base_model=cfg.get("base_model"))
    card["gaps"] = [q["question"] for q in questions]
    card.pop("intake", None)
    print(yaml.safe_dump(card, sort_keys=False, allow_unicode=True, width=100))
    return 0


def _wm_search(args: argparse.Namespace) -> int:
    import yaml

    from awm.wm import intake

    cfg = _wm_config(args)
    mem = _wm_memory(cfg)
    if mem is None or cfg["arm"] not in ("retrieval", "llm"):
        print(json.dumps({"precedents": [], "note": f"arm {cfg['arm']} has no memory to search"}, indent=2))
        return 0
    if args.card:
        card = yaml.safe_load(Path(args.card).read_text())
    else:
        card, _ = intake.draft_card("query", _wm_text(args), {}, Path(cfg["session_dir"]), base_model=cfg.get("base_model"))
    pre = mem.precedents(card, k=args.k)
    curves = mem.curves([(p["session"], p["card_id"]) for p in pre]) if pre else {}
    print(json.dumps({"precedents": pre, "curves": curves, "visible_sides": list(mem.visible_sides)}, indent=2, default=str))
    return 0


def _wm_eval_plan(args: argparse.Namespace) -> int:
    from awm.wm.consult import default_eval_plan

    print(json.dumps(default_eval_plan(args.steps, n=args.n, parent_value=args.parent), indent=2))
    return 0


def _wm_read_eval(args: argparse.Namespace) -> int:
    import math

    from awm.wm.schema import WMError, load_json

    cfg = _wm_config(args)
    path = Path(args.path).resolve()
    if not any(str(path).startswith(str(Path(r).resolve())) for r in _wm_roots(cfg)):
        raise WMError(f"{path} is outside the allowed roots")
    raw = load_json(path)
    value = raw.get("accuracy", raw.get("value"))
    n = raw.get("n")
    stderr = raw.get("stderr")
    if stderr is None and isinstance(value, (int, float)) and n:
        stderr = math.sqrt(max(value * (1 - value), 0) / n)
    print(json.dumps({"path": str(path), "metric": "accuracy", "value": value, "n": n, "stderr": stderr, "raw_keys": sorted(raw)}, indent=2))
    return 0


def _wm_log(args: argparse.Namespace) -> int:
    from awm.wm.consult import log_consult

    cfg = _wm_config(args)
    response = json.loads(Path(args.response).read_text())
    request = Path(args.request).read_text() if args.request else ""
    entry = log_consult(_wm_dir(args), response, request=request, roots=_wm_roots(cfg), arm=cfg["arm"], model=cfg.get("wma_model"))
    print(json.dumps(entry, indent=2, default=str))
    return 0


def _wm_record(args: argparse.Namespace) -> int:
    from awm.wm.record import log_record

    cfg = _wm_config(args)
    response = json.loads(Path(args.response).read_text())
    request = Path(args.request).read_text() if args.request else ""
    entry = log_record(_wm_dir(args), response, request=request, model=cfg.get("wma_model"))
    print(json.dumps(entry, indent=2, default=str))
    return 0


def _wm_snapshot(args: argparse.Namespace) -> int:
    import shutil

    from awm.wm.schema import WMError, dump_json, inside, load_json, now, sha256_file

    cfg = _wm_config(args)
    session_dir = Path(cfg["session_dir"]).resolve()
    dest = _wm_dir(args) / "cards" / args.card / "snapshot"
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "MANIFEST.json"
    manifest = load_json(manifest_path, default={"files": []}) if manifest_path.is_file() else {"files": []}
    for raw in args.paths:
        src = Path(raw).resolve()
        if not src.is_file():
            raise WMError(f"{src} is not a file")
        if not inside(src, session_dir):
            raise WMError(f"{src} is outside the session directory {session_dir}")
        rel = src.relative_to(session_dir)
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        manifest["files"] = [f for f in manifest["files"] if f.get("path") != str(rel)]
        manifest["files"].append({"path": str(rel), "sha256": sha256_file(src),
                                  "bytes": src.stat().st_size, "at": now()})
    dump_json(manifest_path, manifest)
    print(json.dumps({"card": args.card, "snapshot": str(dest),
                      "files": [f["path"] for f in manifest["files"]]}, indent=2))
    return 0


def _wm_archive(args: argparse.Namespace) -> int:
    from awm.wm.record import archive_checkpoint

    cfg = _wm_config(args)
    manifest = archive_checkpoint(_wm_dir(args), Path(cfg["session_dir"]), args.card, Path(args.checkpoint))
    print(json.dumps({"card": manifest["card_id"], "archived": str(_wm_dir(args) / "checkpoints" / args.card),
                      "bytes_total": manifest["bytes_total"], "files": len(manifest["files"])}, indent=2))
    return 0


def _wm_outcome(args: argparse.Namespace) -> int:
    cfg = _wm_config(args)
    if cfg.get("mode") == "record":
        from awm.wm.record import record_outcome
    else:
        from awm.wm.consult import record_outcome

    entry = record_outcome(_wm_dir(args), args.card, final_value=args.final, shipped=args.shipped, note=args.note)
    print(json.dumps(entry, indent=2, default=str))
    return 0


def _wm_status(args: argparse.Namespace) -> int:
    from awm.wm.consult import ConsultLedger

    cfg = _wm_config(args)
    if cfg.get("mode") == "record":
        from awm.wm.record import RecordLedger, check_sufficiency
        from awm.wm.schema import load_json

        rows = RecordLedger(_wm_dir(args) / "records.jsonl").rows()
        by_card: dict[str, list] = {}
        for r in rows:
            by_card.setdefault(r.get("card_id", "?"), []).append(r)
        cards = {}
        for k, v in by_card.items():
            card_file = _wm_dir(args) / "cards" / k / "card.json"
            card = load_json(card_file, default={}) if card_file.is_file() else {}
            last_stage = next((x.get("stage") for x in reversed(v) if x.get("stage")), None)
            cards[k] = {"records": len(v), "stage": last_stage,
                        "missing": check_sufficiency(card, last_stage or "plan") if card else None,
                        "outcome": next((x.get("final_value") for x in reversed(v) if x.get("event") == "outcome"), None)}
        print(json.dumps({"mode": "record", "wma_model": cfg.get("wma_model"),
                          "records": len(rows), "cards": cards}, indent=2, default=str))
        return 0
    rows = ConsultLedger(_wm_dir(args) / "consults.jsonl").rows()
    by_card: dict[str, list] = {}
    for r in rows:
        by_card.setdefault(r.get("card_id", "?"), []).append(r)
    print(json.dumps({"arm": cfg["arm"], "wma_model": cfg.get("wma_model"), "consults": len(rows),
                      "cards": {k: {"consults": len(v), "last_verdict": next((x.get("verdict") for x in reversed(v) if x.get("verdict")), None),
                                    "last_suggestion": next((x.get("suggestion") for x in reversed(v) if x.get("suggestion")), None),
                                    "outcome": next((x.get("final_value") for x in reversed(v) if x.get("event") == "outcome"), None)}
                                for k, v in by_card.items()}}, indent=2, default=str))
    return 0


def _wm_memory_seed(args: argparse.Namespace) -> int:
    cfg = _wm_config(args)
    mem = _wm_memory(cfg)
    if mem is None:
        raise SystemExit("no memory_root configured")
    n = mem.seed_from_exp_cards(Path(args.results_dir), side=args.side)
    print(f"seeded {n} reconstructed cards from {args.results_dir} ({args.side}); memory now {mem.stats()}")
    return 0


def _wm_memory_stats(args: argparse.Namespace) -> int:
    cfg = _wm_config(args)
    mem = _wm_memory(cfg)
    print(json.dumps(mem.stats() if mem else {"note": "no memory_root configured"}, indent=2))
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
    wm = sub.add_parser("wm", help="the world-model agent's toolbelt (draft cards, search past experiments, log consults)")
    wm.add_argument("--dir", type=Path, help="the scientist's session directory (default: $AWM_SESSION_DIR or cwd)")
    wmc = wm.add_subparsers(dest="cmd", required=True)

    wi = wmc.add_parser("init", help="write wm/config.json: which evidence the WMA may read")
    wi.add_argument("--mode", default="consult", choices=["consult", "record"],
                    help="consult: the WMA advises; record: the WMA only keeps the reproducible record")
    wi.add_argument("--arm", default="null", choices=["null", "retrieval", "traj", "llm"])
    wi.add_argument("--prior-runs", help="raw prior runs (traj/llm arms)")
    wi.add_argument("--memory-root", help="WMA memory with extracted cards (retrieval/llm arms)")
    wi.add_argument("--memory-sides", default="train", help="comma list of split sides visible in memory")
    wi.add_argument("--memory-readonly", action="store_true")
    wi.add_argument("--split-side", default="train", choices=["train", "test"])
    wi.add_argument("--wma-model", help="the model the WMA session runs on (recorded)")
    wi.add_argument("--base-model", help="hub id being post-trained; default parent when a plan names none")
    wi.add_argument(
        "--wma-validation-attempts",
        type=int,
        choices=range(1, 6),
        default=1,
        help=argparse.SUPPRESS,
    )
    wi.set_defaults(func=_wm_init)

    wd = wmc.add_parser("draft-card", help="deterministic card skeleton + gaps from the scientist's words and the workspace")
    wd.add_argument("--text"); wd.add_argument("--plan", type=Path); wd.add_argument("--card-id")
    wd.set_defaults(func=_wm_draft_card)

    wsr = wmc.add_parser("search", help="nearest past experiments in memory, with outcomes and curves")
    wsr.add_argument("--text"); wsr.add_argument("--plan", type=Path); wsr.add_argument("--card", type=Path)
    wsr.add_argument("--k", type=int, default=5)
    wsr.set_defaults(func=_wm_search)

    we = wmc.add_parser("eval-plan", help="the default evaluation schedule (25/50/75 percent of the planned steps)")
    we.add_argument("--steps", type=int); we.add_argument("--n", type=int, default=150); we.add_argument("--parent", type=float)
    we.set_defaults(func=_wm_eval_plan)

    wre = wmc.add_parser("read-eval", help="parse an evaluate.py --json-output-file result the scientist points at")
    wre.add_argument("path")
    wre.set_defaults(func=_wm_read_eval)

    wl = wmc.add_parser("log", help="validate a consult response, lint its citations, append it to wm/consults.jsonl")
    wl.add_argument("--response", required=True, type=Path); wl.add_argument("--request", type=Path)
    wl.set_defaults(func=_wm_log)

    wr = wmc.add_parser("record", help="validate a record response (recorder mode), persist the card, append it to wm/records.jsonl")
    wr.add_argument("--response", required=True, type=Path); wr.add_argument("--request", type=Path)
    wr.set_defaults(func=_wm_record)

    wsn = wmc.add_parser("snapshot", help="copy the files a card's command names into wm/cards/<card>/snapshot/, with hashes")
    wsn.add_argument("--card", required=True); wsn.add_argument("paths", nargs="+")
    wsn.set_defaults(func=_wm_snapshot)

    wa = wmc.add_parser("archive", help="preserve a card's checkpoint under wm/checkpoints/<card>/ for the post-run official evaluation")
    wa.add_argument("--card", required=True); wa.add_argument("checkpoint")
    wa.set_defaults(func=_wm_archive)

    wo = wmc.add_parser("outcome", help="record what the scientist shipped and scored")
    wo.add_argument("--card", required=True); wo.add_argument("--final", type=float); wo.add_argument("--shipped"); wo.add_argument("--note")
    wo.set_defaults(func=_wm_outcome)

    wst = wmc.add_parser("status", help="consults per card, last verdict and suggestion, outcomes")
    wst.set_defaults(func=_wm_status)

    wmem = wmc.add_parser("memory", help="WMA memory").add_subparsers(dest="memcmd", required=True)
    wseed = wmem.add_parser("seed", help="load reconstructed cards from results/exp-cards/<split> as precedents")
    wseed.add_argument("results_dir", type=Path); wseed.add_argument("--side", default="train", choices=["train", "test"])
    wseed.set_defaults(func=_wm_memory_seed)
    wstat = wmem.add_parser("stats"); wstat.set_defaults(func=_wm_memory_stats)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.data_root:
        import os

        os.environ["AWM_DATA_ROOT"] = str(args.data_root)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
