"""Step 5 of the construction log: normalize the Inspect log behind every labeled evaluation
into a compact per-(question, run) index, and check it against the label's per-question matrix.

    python -m tools.wm_benchmark.normalize_z <mirror_root> <benchmark_dir> [--workers N] [--force] [--only-present]

Reads  <benchmark_dir>/labels.jsonl   (z_log.path names the log for each example)
Writes <benchmark_dir>/z/<example_id>/samples.jsonl.gz  one record per (question, run): question_id,
           run, score, extracted answer, scorer name, stop reason, token counts, error flag,
           and the sha256 of the model's completion text (the text itself stays in the raw log)
       <benchmark_dir>/z/<example_id>/log_ref.json     log path + sha256, Inspect status, eval
           config as logged, sample counts, agreement with the results matrix, and
           `log_is_record_of_label` = every scored sample matches per_problem and nothing is missing.

Z is label-side data: nothing written here is ever a predictor input.

Relationship to `normalize_logs.py` (kept alongside, unchanged): that script does the same job
for the earlier `cc2ac9d884a7` mirror and the rescore10 track only, driven by a list of ids.
This one is driven by `labels.jsonl`, covers both the native and the controlled-matrix tracks at
the `07132f15e3c6` revision, and additionally records the completion hash/length per sample.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def score_of(sample):
    for name, v in (sample.get("scores") or {}).items():
        val = v.get("value") if isinstance(v, dict) else v
        ans = v.get("answer") if isinstance(v, dict) else None
        if val in ("C", 1, 1.0, True):
            return 1.0, ans, name
        if val in ("I", 0, 0.0, False):
            return 0.0, ans, name
    return None, None, None


def completion_text(sample) -> str:
    out = sample.get("output") or {}
    ch = (out.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
    return ""


def one(args):
    row, root, out = args
    eid = row["example_id"]
    log = root / row["z_log"]["path"]
    res = root / row["source_file"]
    zdir = out / "z" / eid
    zdir.mkdir(parents=True, exist_ok=True)
    ref = {"example_id": eid, "track": row["track"], "log_path": row["z_log"]["path"], "log_available": log.exists(),
           "results_path": row["source_file"]}
    if not log.exists():
        (zdir / "log_ref.json").write_text(json.dumps(ref, indent=1))
        return eid, "no_log"
    ref["log_sha256"] = sha256_file(log)
    d = json.load(gzip.open(log))
    pp = json.loads(res.read_text())["per_problem"]
    ev = d.get("eval") or {}
    ref.update({"log_status": d.get("status"), "eval_created": ev.get("created"), "model": ev.get("model"),
                "model_args": ev.get("model_args"), "generate_config": ev.get("model_generate_config"),
                "packages": ev.get("packages") or {}, "epochs_requested": (ev.get("config") or {}).get("epochs"),
                "dataset": (ev.get("dataset") or {}).get("name"), "task": ev.get("task"), "task_version": ev.get("task_version")})
    r = d.get("results") or {}
    ref["total_samples"], ref["completed_samples"] = r.get("total_samples"), r.get("completed_samples")
    n = same = diff = unscored = errors = 0
    stops, seen = {}, set()
    with gzip.open(zdir / "samples.jsonl.gz", "wt") as fh:
        for s in d.get("samples") or []:
            n += 1
            sc, ans, scorer = score_of(s)
            out_ = s.get("output") or {}
            ch = (out_.get("choices") or [{}])[0]
            usage = out_.get("usage") or {}
            stop = ch.get("stop_reason")
            stops[stop] = stops.get(stop, 0) + 1
            err = bool(s.get("error"))
            errors += err
            qid, ep = str(s.get("id")), s.get("epoch")
            text = completion_text(s)
            rec = {"question_id": qid, "run": ep, "score": sc, "answer": ans, "scorer": scorer, "stop_reason": stop,
                   "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                   "completion_sha256": hashlib.sha256(text.encode()).hexdigest(), "completion_chars": len(text),
                   "error": err}
            fh.write(json.dumps(rec) + "\n")
            seen.add((qid, ep))
            if sc is None or qid not in pp or not (isinstance(ep, int) and 1 <= ep <= 10):
                unscored += 1
            elif abs(pp[qid][ep - 1] - sc) < 1e-9:
                same += 1
            else:
                diff += 1
    expected = len(pp) * 10
    ref.update({"samples_in_log": n, "samples_with_error": errors, "stop_reasons": stops,
                "distinct_question_run_pairs": len(seen),
                "agreement_with_results_matrix": {"same": same, "different": diff, "unscored": unscored},
                "log_is_record_of_label": (d.get("status") == "success" and diff == 0 and unscored == 0
                                           and len(seen) == expected and errors == 0)})
    (zdir / "log_ref.json").write_text(json.dumps(ref, indent=1))
    return eid, ("ok" if ref["log_is_record_of_label"] else "mismatch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mirror_root")
    ap.add_argument("benchmark_dir")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only-present", action="store_true", help="skip examples whose log is not downloaded yet")
    a = ap.parse_args()
    root, out = Path(a.mirror_root), Path(a.benchmark_dir)
    rows = [json.loads(l) for l in (out / "labels.jsonl").open()]
    todo = []
    for row in rows:
        if not a.force and (out / "z" / row["example_id"] / "log_ref.json").exists():
            continue
        if a.only_present and not (root / row["z_log"]["path"]).exists():
            continue
        todo.append(row)
    print(f"{len(rows)} examples, {len(todo)} to process", flush=True)
    counts = {}
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(one, (row, root, out)) for row in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            eid, st = fu.result()
            counts[st] = counts.get(st, 0) + 1
            if i % 100 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} {counts}", flush=True)
    print("done", counts, flush=True)


if __name__ == "__main__":
    main()
