"""Portable-bundle checks, including corruption tests without external corpus data."""

import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "tools/outcome_prediction/verify_eval_matrix_bundle.py"
SPEC = importlib.util.spec_from_file_location("verify_eval_matrix_bundle", PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True), encoding="utf-8")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def refresh_hashes(root):
    files = {p.relative_to(root).as_posix(): M.sha256(p) for p in root.rglob("*")
             if p.is_file() and p.name not in {"bundle_files.sha256.json", "README.md"}}
    write_json(root / "bundle_files.sha256.json", {
        "schema": "eval-matrix-bundle-files-sha256-v1", "algorithm": "sha256", "files": files,
        "excluded": ["bundle_files.sha256.json", "README.md"],
    })


@pytest.fixture
def bundle(tmp_path):
    # Synthetic protocol-complete fixture: no ignored data, cache, network or GPU.
    configs = {"G01": (1, 64, .95, 1, 4000), "G02": (0, 0, 1, 1, 4000),
               "A01": (1, 0, 1, 1, 16000), "A02": (.6, 20, .95, 1, 16000),
               "A03": (1, 0, 1, 1, 2048), "A04": (.6, 20, .95, 1.05, 16000),
               "A05": (0, 0, 1, 1, 16000)}
    for policy, (temp, k, p, penalty, cap) in configs.items():
        stops = [1, 106] if policy.startswith("G") else [151643, 151645]
        gen = {"temperature": temp, "top_k": k, "top_p": p, "min_p": 0,
               "repetition_penalty": penalty, "max_new_tokens": cap,
               "do_sample": temp > 0, "eos_token_id": stops}
        request = {"temperature": temp, "top_p": p, "max_tokens": cap, "n": 1,
                   "presence_penalty": 0, "frequency_penalty": 0, "stop": [],
                   "extra_body": {"top_k": k, "min_p": 0, "repetition_penalty": penalty,
                                  "stop_token_ids": stops, "ignore_eos": False, "min_tokens": 0}}
        write_json(tmp_path / f"configs/{policy}/generation_config.json", gen)
        write_json(tmp_path / f"configs/{policy}/request_template.json", request)
    cps, jobs, inventory, splits = [], [], [], {}
    for bench, count, locked_n, session_n, base_n, pilot_n in (
        ("gsm8k", 240, 70, 63, 190, 20), ("aime2025", 160, 49, 61, 123, 12)
    ):
        locked_sessions = [f"{bench}-s{i:03d}" for i in range(20)]
        splits[bench] = {"locked_sessions": locked_sessions, "n_locked_sessions": 20,
                         "known_anchor_sessions_development_only": [f"{bench}-s020"]}
        for i in range(count):
            locked = i < locked_n
            session_num = i % 20 if locked else 20 + (i - locked_n) % (session_n - 20)
            session = f"{bench}-s{session_num:03d}"
            checkpoint = f"{session}-exp-{i:03d}"
            pilot = not locked and i - locked_n < pilot_n
            diagnostic = bench == "aime2025" and not locked and i - locked_n < 20
            cp = {"checkpoint_id": checkpoint, "benchmark": bench, "trajectory_id": session,
                  "known_lineage_group": session, "checkpoint_uri": f"gs://fixture/{checkpoint}/",
                  "split": "locked_session_test" if locked else "development",
                  "parent_kind": "base" if i < base_n else "checkpoint",
                  "training_only_launch_bundle_status": "not_frozen",
                  "weight_shard_hashes_verified": False,
                  "tokenizer_template_compatibility_verified": False,
                  "v6_missing_declared_code": False, "v6_content_review_required": False}
            cps.append(cp)
            inventory.append({"exp_id": checkpoint, "candidate_weight_bool": True,
                              "from_base": i < base_n, "eligible": True})
            for policy in sorted(M.CORE[bench] | (M.EXTRA if diagnostic else set())):
                phase = "4_locked_test" if locked else "1_operational_pilot" if pilot else \
                    "3_diagnostic_extensions" if policy in M.EXTRA else "2_development_core"
                job = {k: cp[k] for k in ("checkpoint_id", "benchmark", "trajectory_id",
                                         "known_lineage_group", "checkpoint_uri", "split")}
                job.update({"exp_id": f"poc1k-{checkpoint}-{policy}", "generation_config_id": policy,
                            "phase": phase, "primary_evaluation": policy in M.CORE[bench],
                            "n_passes": 10, "n_questions_per_pass": 1319 if bench == "gsm8k" else 30,
                            "release_status": "proposal_only_preflight_required",
                            "historical_reuse_status": "unverified_full_protocol_and_artifact_equivalence",
                            "historical_sampling_cap_match": False, "protocol_path": "protocol.json"})
                for stem in ("generation_config", "request_template"):
                    relative = f"configs/{policy}/{stem}.json"
                    job[stem + "_path"], job[stem + "_sha256"] = relative, M.sha256(tmp_path / relative)
                jobs.append(job)
    write_rows(tmp_path / "experiment_matrix.jsonl", jobs)
    write_rows(tmp_path / "selected_checkpoints.jsonl", cps)
    write_json(tmp_path / "selected_ids_400.json", [r["checkpoint_id"] for r in cps])
    write_json(tmp_path / "checkpoint_inventory.json", {"checkpoints": inventory})
    write_json(tmp_path / "splits.json", splits)
    write_json(tmp_path / "matrix_summary.json", {})
    write_json(tmp_path / "protocol.json", {
        "status": "proposal_not_execution_authorization", "n_passes": 10,
        "replicate_ids": list(range(10)), "server_generation_config_mode": "vllm",
        "target_metric": "mean_pass_at_1_over_ten_complete_equal_question_passes",
        "missing_pass_policy": "invalid_until_retried_not_zero_not_dropped",
        "runtime_release_requirements": ["weights", "tokenizer", "runtime", "SamplingParams", "launch"],
    })
    for phase in M.PHASE_COUNTS:
        write_rows(tmp_path / f"phases/{phase}.jsonl", [r for r in jobs if r["phase"] == phase])
    refresh_hashes(tmp_path)
    return tmp_path


def test_synthetic_bundle_passes_without_external_sources(bundle):
    report = M.verify_bundle(bundle)
    assert report["ok"], report
    assert report["counts"]["cells"] == 1000
    assert report["counts"]["locked_cells"] == 287
    assert "not_launch_ready" in report["status"]
    assert any("400 checkpoints lack verified weight" in w for w in report["warnings"])
    assert any("not frozen" in w for w in report["warnings"])


@pytest.mark.parametrize(("field", "value", "message"), [
    ("n_passes", 9, "ten passes"),
    ("n_questions_per_pass", 100, "question count"),
    ("release_status", "ready", "preflight-required"),
    ("historical_reuse_status", "verified", "explicitly unverified"),
    ("primary_evaluation", False, "Core/diagnostic"),
    ("split", "development", "Checkpoint/job split disagreement"),
])
def test_semantic_job_corruption_is_rejected_even_with_fresh_hash_map(bundle, field, value, message):
    jobs = M.read_json(bundle, "experiment_matrix.jsonl", lines=True)
    jobs[0][field] = value
    write_rows(bundle / "experiment_matrix.jsonl", jobs)
    refresh_hashes(bundle)
    report = M.verify_bundle(bundle)
    assert not report["ok"]
    assert message in report["errors"][0]


def test_duplicate_combination_with_unique_exp_id_rejected(bundle):
    jobs = M.read_json(bundle, "experiment_matrix.jsonl", lines=True)
    jobs[1]["generation_config_id"] = jobs[0]["generation_config_id"]
    write_rows(bundle / "experiment_matrix.jsonl", jobs)
    refresh_hashes(bundle)
    assert "Duplicate checkpoint-policy" in M.verify_bundle(bundle)["errors"][0]


def test_phase_content_not_only_ids_is_validated(bundle):
    relative = "phases/1_operational_pilot.jsonl"
    rows = M.read_json(bundle, relative, lines=True)
    rows[0]["n_passes"] = 9
    write_rows(bundle / relative, rows)
    refresh_hashes(bundle)
    assert "Phase partition/content mismatch" in M.verify_bundle(bundle)["errors"][0]


def test_hash_tamper_rejected(bundle):
    (bundle / "configs/A01/generation_config.json").write_text("{}", encoding="utf-8")
    assert "SHA256 mismatch" in M.verify_bundle(bundle)["errors"][0]


def test_request_semantics_checked_after_consistent_hash_updates(bundle):
    relative = "configs/G01/request_template.json"
    request = M.read_json(bundle, relative)
    request["temperature"] = .5
    write_json(bundle / relative, request)
    jobs = M.read_json(bundle, "experiment_matrix.jsonl", lines=True)
    for job in jobs:
        if job["generation_config_id"] == "G01":
            job["request_template_sha256"] = M.sha256(bundle / relative)
    write_rows(bundle / "experiment_matrix.jsonl", jobs)
    for phase in M.PHASE_COUNTS:
        write_rows(bundle / f"phases/{phase}.jsonl", [r for r in jobs if r["phase"] == phase])
    refresh_hashes(bundle)
    assert "Config/request temperature mismatch" in M.verify_bundle(bundle)["errors"][0]


def test_malformed_hash_map_fails_without_traceback(bundle):
    write_json(bundle / "bundle_files.sha256.json", [])
    assert "must be a JSON object" in M.verify_bundle(bundle)["errors"][0]


def test_map_cannot_omit_core_asset(bundle):
    manifest = M.read_json(bundle, "bundle_files.sha256.json")
    manifest["files"].pop("protocol.json")
    manifest["excluded"].append("protocol.json")
    write_json(bundle / "bundle_files.sha256.json", manifest)
    assert "Required assets absent" in M.verify_bundle(bundle)["errors"][0]


def test_map_rejects_path_escape(bundle):
    manifest = M.read_json(bundle, "bundle_files.sha256.json")
    manifest["files"]["../outside"] = "0" * 64
    write_json(bundle / "bundle_files.sha256.json", manifest)
    assert "Unsafe/noncanonical asset path" in M.verify_bundle(bundle)["errors"][0]


def test_unlisted_assets_rejected_but_excluded_readme_permitted(bundle):
    (bundle / "README.md").write_text("Instructions are deliberately not hashed.", encoding="utf-8")
    assert M.verify_bundle(bundle)["ok"]
    (bundle / "surprise.json").write_text("{}", encoding="utf-8")
    assert "Unhashed assets" in M.verify_bundle(bundle)["errors"][0]


def test_cli_returns_failure_for_missing_bundle(tmp_path, capsys):
    assert M.main(["--bundle", str(tmp_path / "absent"), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_published_portable_bundle():
    if not M.DEFAULT_BUNDLE.is_dir():
        pytest.skip("Published portable bundle not present")
    report = M.verify_bundle(M.DEFAULT_BUNDLE)
    assert report["ok"], report
