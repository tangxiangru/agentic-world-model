import copy
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "tools/outcome_prediction/propose_eval_matrix.py"
SPEC = importlib.util.spec_from_file_location("propose_eval_matrix", PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def row(exp_id, benchmark="gsm8k", family="sft", parents=()):
    return {
        "example_id": exp_id,
        "cell": exp_id.rsplit("-exp-", 1)[0],
        "benchmark": benchmark,
        "recipe": {"family": family, "lora": 0},
        "code": {"training_status": "reconstructed"},
        "parent": {"parent_ids_all": list(parents)},
    }


def test_selection_does_not_use_scores():
    rows = [row(f"session-{i}-exp-01", family="rft" if i % 3 == 0 else "sft") for i in range(12)]
    selected = [r["example_id"] for r in M.diverse_subset(rows, 6)]
    changed = copy.deepcopy(rows)
    for i, r in enumerate(changed):
        r["label"] = {"acc10": 1 - i / 12}
        r["outcome"] = {"accuracy": i / 12}
    assert selected == [r["example_id"] for r in M.diverse_subset(changed, 6)]


def test_subset_covers_new_sessions_before_repeating():
    rows = [row(f"session-{s}-exp-{i:02d}") for s in range(4) for i in range(3)]
    chosen = M.diverse_subset(rows, 4, anchors=[rows[0]["example_id"]])
    assert len({r["cell"] for r in chosen}) == 4
    assert chosen[0]["example_id"] == rows[0]["example_id"]


def test_outcome_fields_rejected_at_any_depth():
    with pytest.raises(ValueError):
        M.ensure_no_outcomes({"recipes": [{"avg_pass_rate": 0.7}]})
    M.ensure_no_outcomes({"exp_id": "x", "n_passes": 10, "target_metric": "mean pass rate"})


def test_known_cross_session_learned_parents_stay_together():
    rows = [row(e, b) for b, ids in M.ANCHORS.items() for e in ids]
    rows += [row(f"gsm-test-{i}-exp-01") for i in range(4)]
    rows += [row(f"aime-test-{i}-exp-01", "aime2025") for i in range(4)]
    rows += [row("gsm-derived-exp-01", parents=["gsm-test-0-exp-01"])]
    reserved, groups, summary = M.session_splits(rows, rows, per_benchmark=1)
    assert groups["gsm-derived"] == groups["gsm-test-0"]
    assert ("gsm-derived" in reserved) == ("gsm-test-0" in reserved)
    for b, ids in M.ANCHORS.items():
        assert all(e.rsplit("-exp-", 1)[0] not in reserved for e in ids)
        assert summary[b]["n_locked_sessions"] >= 1


@pytest.fixture
def artifacts():
    bundle = M.ROOT / "experiments/eval_matrix_1k"
    if not (bundle / "experiment_matrix.jsonl").exists():
        pytest.skip("Published portable eval-matrix bundle is not present")
    return bundle, M.read_jsonl(bundle / "experiment_matrix.jsonl")


def test_published_budget_and_candidate_identity(artifacts):
    out, jobs = artifacts
    inventory = json.loads((out / "checkpoint_inventory.json").read_text())
    candidates = {r["exp_id"] for r in inventory["checkpoints"] if r["candidate_weight_bool"]}
    base_ids = {r["exp_id"] for r in inventory["checkpoints"] if r["from_base"]}
    selected = {j["checkpoint_id"] for j in jobs}
    assert len(jobs) == len({j["exp_id"] for j in jobs}) == 1000
    assert len({(j["checkpoint_id"], j["generation_config_id"]) for j in jobs}) == 1000
    assert len(selected) == 400
    assert selected <= candidates
    assert len(base_ids) == 313 and base_ids <= selected
    assert Counter(j["benchmark"] for j in jobs) == {"gsm8k": 480, "aime2025": 520}
    assert sum(j["n_passes"] for j in jobs) == 10000
    assert sum(j["n_passes"] * j["n_questions_per_pass"] for j in jobs) == 6487200
    M.ensure_no_outcomes(jobs)


def test_published_splits_and_phase_partition(artifacts):
    out, jobs = artifacts
    session_splits = defaultdict(set)
    for j in jobs:
        session_splits[j["trajectory_id"]].add(j["split"])
        assert j["release_status"] == "proposal_only_preflight_required"
    assert len(session_splits) == 124
    assert all(len(v) == 1 for v in session_splits.values())
    locked = [j for j in jobs if j["split"] == "locked_session_test"]
    assert len(locked) == 287
    assert len({j["trajectory_id"] for j in locked}) == 40
    assert all(j["primary_evaluation"] for j in locked)
    assert sum(j["primary_evaluation"] for j in jobs) == 960
    pilot = [j for j in jobs if j["phase"] == "1_operational_pilot"]
    assert len(pilot) == 100
    assert {j["generation_config_id"] for j in pilot} == {"G01", "G02", "A01", "A02", "A03", "A04", "A05"}
    # The shared directory also contains the additive 2K phase files. This test
    # validates only the original matrix's partition; the extension has its own
    # combined-plan tests, and runners must not glob all files as original work.
    phase_jobs = [j for phase in sorted({row["phase"] for row in jobs})
                  for j in M.read_jsonl(out / "phases" / f"{phase}.jsonl")]
    assert sorted(j["exp_id"] for j in phase_jobs) == sorted(j["exp_id"] for j in jobs)


def test_policy_files_and_hashes_match_request_templates(artifacts):
    out, jobs = artifacts
    for p_id in {j["generation_config_id"] for j in jobs}:
        representative = next(j for j in jobs if j["generation_config_id"] == p_id)
        gen_path = out / representative["generation_config_path"]
        request_path = out / representative["request_template_path"]
        assert M.digest(gen_path) == representative["generation_config_sha256"]
        assert M.digest(request_path) == representative["request_template_sha256"]
        gen, request = json.loads(gen_path.read_text()), json.loads(request_path.read_text())
        effective = request | request["extra_body"]
        for k in ("temperature", "top_k", "top_p", "min_p", "repetition_penalty"):
            assert gen[k] == effective[k]
        assert gen["max_new_tokens"] == effective["max_tokens"]
        assert gen["eos_token_id"] == effective["stop_token_ids"]
        assert gen["do_sample"] == (gen["temperature"] > 0)
        assert effective["ignore_eos"] is False and effective["n"] == 1


@pytest.mark.needs_data
def test_source_hashes_current():
    out = M.DEFAULT_OUT
    if not (out / "matrix_summary.json").exists():
        pytest.skip("Full-corpus source provenance is not present; portable hashes are tested separately")
    summary = json.loads((out / "matrix_summary.json").read_text())
    missing = [p for p in summary["source_hashes"] if not (M.ROOT / p).is_file()]
    if missing:
        pytest.skip("Full-corpus source files are unavailable: " + ", ".join(missing))
    for relative, sha in summary["source_hashes"].items():
        assert M.digest(M.ROOT / relative) == sha, relative


@pytest.mark.needs_data
def test_upstream_selection_reproduces_ids_and_ignores_outcomes(artifacts):
    out, _ = artifacts
    path = PATH.with_name("select_eval_matrix_checkpoints.py")
    spec = importlib.util.spec_from_file_location("select_eval_matrix_checkpoints", path)
    selector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selector)
    sources = [selector.DEFAULT_LABELS, selector.DEFAULT_TABLE, selector.DEFAULT_INPUTS]
    if any(not p.is_file() for p in sources):
        pytest.skip("Upstream selection requires the unbundled full-corpus labels/table/input sources")
    eligible = selector._eligible_ids(selector.DEFAULT_LABELS)
    rows = selector._selection_rows(selector.DEFAULT_TABLE, eligible)
    quality = selector._quality_rows(selector.DEFAULT_INPUTS, eligible)
    expected = json.loads((out / "selected_ids_400.json").read_text())
    assert selector.select(rows, quality) == expected
    assert all(set(r["parent"]) == {"parent_kind"} for r in rows.values())
    M.ensure_no_outcomes(list(rows.values()))
    for i, r in enumerate(rows.values()):
        r["label"] = {"acc10": i / len(rows)}
        r["outcome"] = {"accuracy": 1 - i / len(rows)}
        r["parent"]["parent_acc"] = 1 - i / len(rows)
        r["selection_400"] = False
    assert selector.select(rows, quality) == expected
