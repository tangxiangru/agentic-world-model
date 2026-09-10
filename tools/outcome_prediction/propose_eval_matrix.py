"""Build a review-only, outcome-free checkpoint/serving evaluation proposal.

This does not launch evaluation, change checkpoints, or fit predictors. It consumes
the separately audited candidate selection and emits proposed exp_ids. The output
is deliberately NOT an executable release manifest: weight/tokenizer/launch-time
recipe binding and actual server SamplingParams still require preflight.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "data/analysis/wm_exp_designs/eval_matrix_1k_design"
RAW = ROOT / "data/traj/raw/awm-gsm8k-trajectories-cc2ac9d884a7"
ANCHORS = {
    "gsm8k": ["r0-29-exp-02", "gsm2-r0-26-exp-03"],
    "aime2025": ["aime-r0-22-exp-02", "aime-r0-31-exp-03"],
}
CORE = {"gsm8k": ["G01", "G02"], "aime2025": ["A01", "A02", "A03"]}
EXTRA = ["A04", "A05"]
SALT = "poc-serving-matrix-v1-20260909"
FORBIDDEN = {"accuracy", "avg_pass_rate", "per_run_pass_rate", "label", "output", "scientist"}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def order_key(kind, value):
    return hashlib.sha256(f"{SALT}|{kind}|{value}".encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(v, sort_keys=True) + "\n" for v in values))


def diverse_subset(rows, n, anchors=()):
    """Outcome-free diagnostic/pilot subset: anchors, then session/recipe breadth.

    These metadata strata are sampling aids, not a definition of true similarity.
    Pilot membership is not a declaration that a checkpoint has passed preflight.
    """
    by_id = {r["example_id"]: r for r in rows}
    chosen = [by_id[e] for e in anchors if e in by_id]
    while len(chosen) < n:
        used = {r["example_id"] for r in chosen}
        sessions = Counter(r["cell"] for r in chosen)
        families = Counter(r["recipe"]["family"] for r in chosen)
        loras = Counter(str(r["recipe"].get("lora")) for r in chosen)
        def rank(r):
            return (
                sessions[r["cell"]],
                r["code"].get("training_status") != "reconstructed",
                families[r["recipe"]["family"]],
                loras[str(r["recipe"].get("lora"))],
                order_key("pilot", r["example_id"]),
            )
        remaining = [r for r in rows if r["example_id"] not in used]
        if not remaining:
            raise ValueError("Not enough candidates for the declared subset")
        chosen.append(min(remaining, key=rank))
    return chosen


def session_splits(selected, all_rows, per_benchmark=20):
    """Keep known learned-parent links together; public base models are not links."""
    sessions = {r["cell"] for r in all_rows}
    parent = {s: s for s in sessions}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    by_id = {r["example_id"]: r for r in all_rows}
    for r in all_rows:
        for e in r["parent"].get("parent_ids_all", []):
            if e in by_id:
                union(r["cell"], by_id[e]["cell"])
    components = defaultdict(list)
    for s in sessions:
        components[find(s)].append(s)
    reserved, summary = set(), {}
    for benchmark in CORE:
        selected_sessions = {r["cell"] for r in selected if r["benchmark"] == benchmark}
        anchor_sessions = {by_id[e]["cell"] for e in ANCHORS[benchmark]}
        protected = {find(s) for s in anchor_sessions}
        eligible = {find(s) for s in selected_sessions} - protected
        ranked = sorted(eligible, key=lambda s: order_key("locked-session", s))
        chosen, count = [], 0
        for c in ranked:
            if count >= per_benchmark:
                break
            chosen.append(c)
            count += len(set(components[c]) & selected_sessions)
        if count < per_benchmark:
            raise ValueError("Insufficient independent session components")
        locked = sorted(s for c in chosen for s in components[c])
        reserved.update(locked)
        summary[benchmark] = {"locked_sessions": locked, "n_locked_sessions": count,
                              "known_anchor_sessions_development_only": sorted(anchor_sessions)}
    return reserved, {s: find(s) for s in sessions}, summary


def ensure_no_outcomes(value):
    if isinstance(value, dict):
        bad = set(value) & FORBIDDEN
        if bad:
            raise ValueError(f"Outcome/measurement fields leaked into proposal: {bad}")
        for v in value.values():
            ensure_no_outcomes(v)
    elif isinstance(value, list):
        for v in value:
            ensure_no_outcomes(v)


def build(out, selected_ids):
    table_path = ROOT / "data/analysis/wm_exp_designs/table_v2/experiments.jsonl"
    label_path = ROOT / "data/analysis/wm_exp_designs/prefix_recipes_v6/labels.jsonl"
    input_path = ROOT / "data/analysis/wm_exp_designs/prefix_recipes_v6/inputs.jsonl"
    audit_path = ROOT / "data/analysis/wm_exp_designs/prefix_recipes_v6/audit.jsonl"
    config_path = out / "config_audit.json"
    inventory_path = out / "checkpoint_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    candidates = {r["exp_id"] for r in inventory["checkpoints"] if r["candidate_weight_bool"]}
    assert set(selected_ids) <= candidates, "Decode/selection bundles cannot fill weight-coverage slots"
    # Scores are neither used for selection nor copied into the manifest.
    eligible = {r["exp_id"] for r in read_jsonl(label_path) if r["output"].get("eligible_for_training")}
    if not set(selected_ids) <= eligible:
        raise ValueError("Selection includes an ineligible checkpoint")
    rows = [r for r in read_jsonl(table_path) if r["example_id"] in eligible]
    selected = [r for r in rows if r["example_id"] in set(selected_ids)]
    selected.sort(key=lambda r: r["example_id"])
    assert len(set(selected_ids)) == len(selected) == 400
    assert Counter(r["benchmark"] for r in selected) == {"gsm8k": 240, "aime2025": 160}
    base_ids = {r["example_id"] for r in rows if r["parent"]["parent_kind"] == "base"}
    assert len(base_ids) == 313 and base_ids <= set(selected_ids)
    inputs = {r["exp_id"]: r for r in read_jsonl(input_path)}
    audits = {r["exp_id"]: r for r in read_jsonl(audit_path)}
    config_audit = json.loads(config_path.read_text())
    policy_ids = {p for ids in CORE.values() for p in ids} | set(EXTRA)
    policies = {p["policy_id"]: p for b in config_audit["benchmarks"].values()
                for p in b["policies"] if p["policy_id"] in policy_ids}
    native_policy = {e: p["policy_id"] for b in config_audit["benchmarks"].values()
                     for p in b["policies"] for e in p["source_exp_ids"]}
    reserved, groups, split_summary = session_splits(selected, rows)
    pilot_ids = set()
    for benchmark in CORE:
        pool = [r for r in selected if r["benchmark"] == benchmark and r["cell"] not in reserved]
        # 20 GSM * 2 policies + 12 AIME * 5 policies = 100 pilot cells.
        # Cover all seven policies before using the pilot to forecast cost.
        n_pilot = 20 if benchmark == "gsm8k" else 12
        pilot_ids.update(r["example_id"] for r in diverse_subset(pool, n_pilot, ANCHORS[benchmark]))
    diagnostic_pool = [r for r in selected if r["benchmark"] == "aime2025" and r["cell"] not in reserved]
    diagnostic_ids = {r["example_id"] for r in diverse_subset(diagnostic_pool, 20, ANCHORS["aime2025"])}
    assert {r["example_id"] for r in selected if r["benchmark"] == "aime2025" and r["example_id"] in pilot_ids} <= diagnostic_ids
    assert len(diagnostic_ids) == 20
    policies_out = {}
    for p_id, p in sorted(policies.items()):
        proposed_gen = dict(p["generation_config_json_payload"])
        proposed_gen["do_sample"] = proposed_gen["temperature"] > 0
        write_json(out / "configs" / p_id / "generation_config.json", proposed_gen)
        request = json.loads(json.dumps(p["openai_request_payload"]))
        request.update({"n": 1, "presence_penalty": 0, "frequency_penalty": 0, "stop": []})
        request["extra_body"]["min_tokens"] = 0
        write_json(out / "configs" / p_id / "request_template.json", request)
        policies_out[p_id] = {"benchmark": "gsm8k" if p_id.startswith("G") else "aime2025",
                             "parameters": p["settings"], "stop_token_ids": p["planned_stop_token_ids"],
                             "corpus_records_with_sampling_and_cap_policy": p["n"],
                             "source_representative_exp_id": p["representative"]["exp_id"],
                             "config_sha256": digest(out / "configs" / p_id / "generation_config.json"),
                             "request_template_sha256": digest(out / "configs" / p_id / "request_template.json")}
    jobs, checkpoints = [], []
    for r in selected:
        e, bench, session = r["example_id"], r["benchmark"], r["cell"]
        split = "locked_session_test" if session in reserved else "development"
        quality = inputs[e]["quality"]
        checkpoint = {
            "checkpoint_id": e, "benchmark": bench, "trajectory_id": session,
            "split": split, "known_lineage_group": groups[session],
            "checkpoint_uri": audits[e]["checkpoint_uri"],
            "parent_kind": r["parent"]["parent_kind"],
            "parent_checkpoint_ids": r["parent"]["parent_ids_all"],
            "card_family": r["recipe"]["family"], "lora": r["recipe"].get("lora"),
            "declared_training_script": r["recipe"].get("script"),
            "historical_plan_script_status": r["code"].get("training_status"),
            "v6_missing_declared_code": quality.get("missing_declared_code"),
            "v6_content_review_required": quality.get("content_review_required"),
            "training_only_launch_bundle_status": "not_frozen",
            "weight_shard_hashes_verified": False,
            "tokenizer_template_compatibility_verified": False,
            "selection_basis": "complete_eligible_from_base_cohort" if e in base_ids else "outcome_free_continuation_coverage",
            "pilot": e in pilot_ids,
        }
        checkpoints.append(checkpoint)
        p_ids = CORE[bench] + (EXTRA if e in diagnostic_ids else [])
        for p_id in p_ids:
            is_extra = p_id in EXTRA
            phase = ("4_locked_test" if split == "locked_session_test" else
                     "1_operational_pilot" if e in pilot_ids else
                     "3_diagnostic_extensions" if is_extra else "2_development_core")
            jobs.append({
                "exp_id": f"poc1k-{e}-{p_id}", "checkpoint_id": e,
                "checkpoint_uri": checkpoint["checkpoint_uri"],
                "trajectory_id": session, "known_lineage_group": groups[session],
                "benchmark": bench, "generation_config_id": p_id,
                "generation_config_path": f"configs/{p_id}/generation_config.json",
                "generation_config_sha256": policies_out[p_id]["config_sha256"],
                "request_template_path": f"configs/{p_id}/request_template.json",
                "request_template_sha256": policies_out[p_id]["request_template_sha256"],
                "protocol_path": "protocol.json", "split": split, "phase": phase,
                "primary_evaluation": not is_extra, "n_passes": 10,
                "n_questions_per_pass": 1319 if bench == "gsm8k" else 30,
                "release_status": "proposal_only_preflight_required",
                "historical_sampling_cap_match": native_policy[e] == p_id,
                "historical_reuse_status": "unverified_full_protocol_and_artifact_equivalence",
            })
    jobs.sort(key=lambda j: (j["phase"], j["benchmark"], j["checkpoint_id"], j["generation_config_id"]))
    protocol = {
        "status": "proposal_not_execution_authorization", "n_passes": 10,
        "target_metric": "mean_pass_at_1_over_ten_complete_equal_question_passes",
        "missing_pass_policy": "invalid_until_retried_not_zero_not_dropped",
        "server_generation_config_mode": "vllm",
        "scheduling_recommendation": "Load each checkpoint once and issue all its policy requests against that server; pin cache/concurrency settings and record policy order. Ten seeds do not require ten weight downloads or server restarts.",
        "request_policy": "send_complete_request_template_plus_question_specific_seed_and_messages",
        "seed_formula": "int.from_bytes(sha256(f'poc-serving-v1|{benchmark}|{question_id}|{replicate_id}').digest()[:4], 'big') & 0x7fffffff",
        "replicate_ids": list(range(10)),
        "runtime_reference": {"vllm": "0.11.0", "inspect_ai": "0.3.150", "transformers": "4.57.3", "torch": "2.8.0+cu129"},
        "runtime_release_requirements": [
            "Pin complete container image, hardware/dtype/quantization, concurrency and scheduler settings before execution.",
            "Hash weights, model/tokenizer configs and tokenizer artifacts; validate special-token ID mappings.",
            "Require tokenizer primary EOS to be included in the explicitly supplied benchmark stop-token set.",
            "Record server-resolved SamplingParams, loaded artifact hashes and effective prompt-dependent token caps, not merely request JSON.",
            "Assert every relevant request setting; reject silent generation_config inheritance or request overrides.",
            "Do not silently replace incompatible tokenizers or overwrite archived checkpoint files.",
            "Freeze clean training-launch-only predictor inputs before paying for a checkpoint's evaluations.",
            "Detect duplicate learned weights across all available corpora; regroup/refreeze splits before exposing new labels.",
            "Preserve per-question/run correctness, completion, stop reason, generated-token length, latency and request seed.",
        ],
        "historical_reuse": "Reuse only after full weights/tokenizer/template/effective-policy/scorer/runtime-equivalence audit. Matching sampling/cap fields alone is insufficient. Do not automatically spend savings on more cells.",
        "benchmark_files": {},
        "primary_inference_input": "training launch script prefix + data/config specification + prescribed serving policy; no target/same-session benchmark outcomes",
        "holdout_exposure": "Old recipes/outcomes have been inspected historically; this is not an untouched future-recipe test. Some proposed cells may repeat native policies.",
    }
    for bench in CORE:
        template = "gemma3.jinja" if bench == "gsm8k" else "qwen3.jinja"
        paths = [RAW / "rescore10/eval/tasks" / bench / "test_data.json",
                 RAW / "rescore10/eval/tasks" / bench / "evaluate_epochs.py",
                 RAW / "rescore10/eval/templates" / template]
        protocol["benchmark_files"][bench] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    coverage = []
    by_checkpoint = {r["checkpoint_id"]: r for r in checkpoints}
    for path in sorted((ROOT / "data/analysis/wm_exp_designs/twins_manual/matches").glob("*.json")):
        m = json.loads(path.read_text())
        if m["query"] not in by_checkpoint:
            continue
        q, partner = by_checkpoint[m["query"]], by_checkpoint.get(m["best_match"])
        coverage.append({"query_checkpoint_id": m["query"], "best_match_checkpoint_id": m["best_match"],
                         "historical_manual_grade": m["grade"], "partner_selected": partner is not None,
                         "query_split": q["split"], "partner_split": partner["split"] if partner else None,
                         "locked_query_has_development_match": q["split"] == "locked_session_test" and partner is not None and partner["split"] == "development"})
    assert len(jobs) == len({j["exp_id"] for j in jobs}) == 1000
    assert sum(j["primary_evaluation"] for j in jobs) == 960
    assert sum(j["phase"] == "1_operational_pilot" for j in jobs) == 100
    assert all(j["split"] == "development" for j in jobs if not j["primary_evaluation"])
    for value in [jobs, checkpoints, protocol, policies_out, coverage]:
        ensure_no_outcomes(value)
    summary = {
        "status": "review_only_not_launch_ready", "n_exp_ids": 1000, "n_candidate_weight_sets": 400,
        "n_verified_unique_weight_sets": None, "n_full_benchmark_passes_if_all_run": 10000,
        "question_completions_if_all_run": sum(j["n_passes"] * j["n_questions_per_pass"] for j in jobs),
        "selected_checkpoints_by_benchmark": dict(Counter(r["benchmark"] for r in selected)),
        "cells_by_benchmark": dict(Counter(j["benchmark"] for j in jobs)),
        "cells_by_phase": dict(Counter(j["phase"] for j in jobs)),
        "sessions_by_benchmark": {b: len({r["cell"] for r in selected if r["benchmark"] == b}) for b in CORE},
        "splits": split_summary,
        "selected_from_base": len(base_ids), "selected_continuations": 400 - len(base_ids),
        "historical_sampling_cap_matching_cells_not_verified_replays": sum(j["historical_sampling_cap_match"] for j in jobs),
        "historical_plan_script_status": dict(Counter(c["historical_plan_script_status"] for c in checkpoints)),
        "v6_content_review_required": sum(bool(c["v6_content_review_required"]) for c in checkpoints),
        "v6_missing_declared_code": sum(bool(c["v6_missing_declared_code"]) for c in checkpoints),
        "selection_salt": SALT,
        "selected_ids_semantic_sha256": hashlib.sha256(json.dumps(sorted(selected_ids)).encode()).hexdigest(),
        "manual_query_coverage": len(coverage),
        "all_known_learned_parent_edges_within_session": all(groups[s] == s for s in groups),
        "source_pins": {"trajectories": "cc2ac9d884a7d962a6024ab0d5cd8ed3370070de", "checkpoint_metadata": "446127629d7b271d537390e69bfb2d960a3aa515"},
        "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in [table_path, label_path, input_path, audit_path, config_path, inventory_path, Path(__file__), Path(__file__).with_name("select_eval_matrix_checkpoints.py")]},
        "safety_checks": {"unique_exp_ids": True, "eligible_checkpoints_only": True, "whole_session_splits": True,
                          "all_313_eligible_from_base_retained": True, "no_outcome_fields_in_manifests": True,
                          "no_eval_jobs_launched": True, "user_owned_design_document_unchanged": True},
    }
    write_jsonl(out / "experiment_matrix.jsonl", jobs)
    write_jsonl(out / "selected_checkpoints.jsonl", checkpoints)
    write_json(out / "generation_policies.json", policies_out)
    write_json(out / "protocol.json", protocol)
    write_json(out / "splits.json", split_summary)
    write_json(out / "manual_match_coverage.json", coverage)
    write_json(out / "matrix_summary.json", summary)
    for phase in sorted({j["phase"] for j in jobs}):
        write_jsonl(out / "phases" / f"{phase}.jsonl", [j for j in jobs if j["phase"] == phase])
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-ids", type=Path, required=True, help="JSON array of the audited 400 checkpoint IDs")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    selected_ids = json.loads(args.selected_ids.read_text())
    build(args.out, selected_ids)


if __name__ == "__main__":
    main()
