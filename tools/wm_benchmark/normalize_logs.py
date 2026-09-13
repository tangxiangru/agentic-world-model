"""Normalize the ten-pass inspect logs (Z) into a compact per-sample index.

For every labeled example, reads rescore10/trajectories/<id>.json.gz from the pinned mirror and writes
  <out>/z/<id>/samples.jsonl.gz   one record per (question, run): question_id, run (epoch), score, extracted answer,
                                  stop reason, input/output token counts, error flag
  <out>/z/<id>/log_ref.json       mirror path, sha256, log status, sample counts, agreement with the results matrix
Responses themselves are not copied (they stay in the raw log, referenced by hash) to keep the dataset small.
Idempotent: examples whose log_ref.json already exists are skipped unless --force.
"""
import argparse, gzip, hashlib, json, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MIRROR = REPO / "data/traj/raw/awm-gsm8k-trajectories-cc2ac9d884a7"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def score_of(sample):
    for name, v in (sample.get("scores") or {}).items():
        val = v.get("value") if isinstance(v, dict) else v
        ans = v.get("answer") if isinstance(v, dict) else None
        if val in ("C", 1, 1.0, True): return 1.0, ans, name
        if val in ("I", 0, 0.0, False): return 0.0, ans, name
    return None, None, None


def one(args):
    eid, out = args
    log = MIRROR / "rescore10/trajectories" / f"{eid}.json.gz"
    res = MIRROR / "rescore10/results" / f"{eid}.json"
    zdir = out / "z" / eid; zdir.mkdir(parents=True, exist_ok=True)
    ref = {"example_id": eid, "log_path": str(log.relative_to(REPO)), "log_available": log.exists(), "results_path": str(res.relative_to(REPO))}
    if not log.exists():
        json.dump(ref, open(zdir / "log_ref.json", "w"), indent=1); return eid, "no_log"
    ref["log_sha256"] = sha256(log)
    d = json.load(gzip.open(log))
    results = json.load(open(res)); pp = results["per_problem"]
    ref["log_status"] = d.get("status"); ev = d.get("eval") or {}
    ref["eval_created"] = ev.get("created"); ref["model"] = ev.get("model"); ref["model_args"] = ev.get("model_args")
    ref["generate_config"] = ev.get("model_generate_config"); ref["packages"] = (ev.get("packages") or {})
    ref["epochs_requested"] = (ev.get("config") or {}).get("epochs")
    r = d.get("results") or {}; ref["total_samples"] = r.get("total_samples"); ref["completed_samples"] = r.get("completed_samples")
    n = same = diff = unscored = errors = 0; stops = {}
    with gzip.open(zdir / "samples.jsonl.gz", "wt") as f:
        for s in d.get("samples") or []:
            n += 1
            sc, ans, scorer = score_of(s)
            out_ = s.get("output") or {}; ch = (out_.get("choices") or [{}])[0]; usage = out_.get("usage") or {}
            stop = ch.get("stop_reason"); stops[stop] = stops.get(stop, 0) + 1
            err = bool(s.get("error")); errors += err
            qid = str(s.get("id")); ep = s.get("epoch")
            rec = {"question_id": qid, "run": ep, "score": sc, "answer": ans, "scorer": scorer, "stop_reason": stop,
                   "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"), "error": err}
            f.write(json.dumps(rec) + "\n")
            if sc is None or qid not in pp or not (isinstance(ep, int) and 1 <= ep <= 10): unscored += 1
            elif abs(pp[qid][ep - 1] - sc) < 1e-9: same += 1
            else: diff += 1
    ref.update({"samples_in_log": n, "samples_with_error": errors, "stop_reasons": stops,
                "agreement_with_results_matrix": {"same": same, "different": diff, "unscored": unscored},
                "log_is_record_of_label": (d.get("status") == "success" and diff == 0 and n == len(pp) * 10)})
    json.dump(ref, open(zdir / "log_ref.json", "w"), indent=1)
    return eid, ref["log_status"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--ids", required=True, help="file with one example id per line")
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--force", action="store_true")
    a = ap.parse_args(); out = Path(a.out)
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    todo = [e for e in ids if a.force or not (out / "z" / e / "log_ref.json").exists()]
    print(f"{len(ids)} ids, {len(todo)} to process", flush=True)
    done = 0
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(one, (e, out)) for e in todo]
        for fu in as_completed(futs):
            eid, st = fu.result(); done += 1
            if done % 25 == 0 or done == len(todo): print(f"{done}/{len(todo)} last={eid} status={st}", flush=True)
    print("done", flush=True)
