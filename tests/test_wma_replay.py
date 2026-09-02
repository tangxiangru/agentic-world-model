"""Offline replay: one session per (run, card k), and the leakage rules are code."""

from __future__ import annotations

import json
import os

import pytest
import yaml

from awm.exp_protocol import schema as cards
from awm.wma import backends, ledger, replay, schema
from exp_protocol_cards import closed_card, plan_card


def v1(card_id: str, decision: str | None = "reject", execution: str = "completed", outcome: float | None = None,
       origin: str = "base_model") -> dict:
    """A six-section corpus card: top-level elapsed_h, no situation, v1 schema_version."""
    c = closed_card() if decision else plan_card()
    c["schema_version"] = "awm-experiment-card-v1"
    c["card_id"] = card_id
    c["elapsed_h"] = c["situation"].pop("elapsed_h")
    del c["situation"]
    c["setup"]["parent_checkpoint"]["origin"] = origin
    if decision:
        c["conclusion"]["decision"] = decision
        c["result"]["execution"] = execution
        c["result"]["wall_h"] = 1.0
        if execution != "completed":
            c["result"]["output_checkpoint"] = None
            c["result"]["measurements"] = []
            c["conclusion"]["verdict"] = "inconclusive"
    if outcome is not None:
        c["outcome"] = {"official_accuracy": outcome}
    c["provenance"] = {"run_ref": "r-x", "launch_i": 1}
    return c


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "corpus"
    runs = {
        "train/r-aaaa": [v1("exp-01"), v1("exp-02", "adopt", outcome=0.61, origin="exp-01"), v1("exp-03", None, origin="exp-02")],
        "train/r-bbbb": [v1("exp-01", "adopt"), v1("exp-02", "abandon_line", execution="failed", origin="exp-01")],
        "train/r-cccc": [v1("exp-01", "reject", execution="killed")],
        "test/r-tttt": [v1("exp-01", "adopt", outcome=0.7)],
    }
    for rel, cs in runs.items():
        d = root / rel
        for c in cs:
            cards.dump_card(d / f"{c['card_id']}.yaml", c)
        (d / "index.md").write_text("# run\n")
    return root


@pytest.fixture
def skill(tmp_path, monkeypatch):
    d = tmp_path / "skill"
    d.mkdir()
    (d / "SKILL.md").write_text("---\nname: wma\n---\n")
    monkeypatch.setenv("AWM_WMA_SKILL_DIR", str(d))
    return d


def test_build_makes_one_session_per_card_with_the_leakage_rules(corpus, tmp_path, skill) -> None:
    out = tmp_path / "replay"
    samples = replay.build_samples(corpus, out, side="train")
    assert len(samples) == 6
    s = next(x for x in samples if x.run_ref == "r-aaaa" and x.card_id == "exp-02")
    cdir = s.session_dir / "memory" / "cards"
    # cards after k are absent; k has no result/conclusion/outcome; earlier cards keep their results
    assert sorted(p.name for p in cdir.glob("exp-*.yaml")) == ["exp-01.yaml", "exp-02.yaml"]
    k = yaml.safe_load((cdir / "exp-02.yaml").read_text())
    assert "result" not in k and "conclusion" not in k and "outcome" not in k
    assert k["schema_version"] == cards.CARD_SCHEMA and "situation" in k
    prev = yaml.safe_load((cdir / "exp-01.yaml").read_text())
    assert prev["conclusion"]["decision"] == "reject" and "outcome" not in prev
    # no card of this run carries the run's official score anywhere in the session
    for p in s.session_dir.rglob("*.yaml"):
        assert "official_accuracy" not in p.read_text(), p
    # index exists; history links to the other train runs only, never to this run or to the test side
    assert (s.session_dir / "memory" / "index.md").is_file()
    hist = s.session_dir / "history"
    assert hist.is_symlink()
    linked = sorted(os.listdir(hist))
    assert linked == ["r-bbbb", "r-cccc"]
    # truth lives outside every session directory and keeps the result and the outcome
    assert s.truth_path.is_file() and not str(s.truth_path).startswith(str(s.session_dir))
    t = yaml.safe_load(s.truth_path.read_text())
    assert t["conclusion"]["decision"] == "adopt" and t["outcome"]["official_accuracy"] == 0.61
    assert (out / "samples.jsonl").is_file()


def test_test_side_is_never_touched_when_replaying_train(corpus, tmp_path, skill) -> None:
    out = tmp_path / "replay"
    replay.build_samples(corpus, out, side="train")
    assert not (out / "r-tttt").exists()
    for p in out.rglob("*"):
        assert "r-tttt" not in p.name


def test_sampling_is_deterministic(corpus, tmp_path, skill) -> None:
    a = replay.build_samples(corpus, tmp_path / "a", side="train", sample=2, seed=7)
    b = replay.build_samples(corpus, tmp_path / "b", side="train", sample=2, seed=7)
    assert [(s.run_ref, s.card_id) for s in a] == [(s.run_ref, s.card_id) for s in b] and len(a) == 2


def test_run_replay_reviews_reconciles_and_is_resumable(corpus, tmp_path, skill) -> None:
    out = tmp_path / "replay"
    replay.build_samples(corpus, out, side="train")
    counts = replay.run_replay(out, backends.HeuristicBackend(), budget=backends.Budget(wall_min=1))
    assert counts == {"reviewed": 6, "skipped": 0, "errors": 0}
    summary = ledger.summarize(ledger.rows([out]))
    # six verdicts; five have an outcome — r-aaaa/exp-03 is an open historical card with none
    assert len(summary) == 1 and summary[0]["n"] == 6 and summary[0]["n_reconciled"] == 5
    # the failed and killed cards scored L0/L1 against the truth kept outside
    rows = {(r["path"].split("/")[-5], r["card_id"]): r for r in ledger.rows([out])}
    assert rows[("r-bbbb", "exp-02")]["scored"]["L0"] == "miss"
    assert rows[("r-cccc", "exp-01")]["scored"] == {"L0": "hit", "L1": "miss", "L2": "unscorable", "L3": "hit"} or \
        rows[("r-cccc", "exp-01")]["scored"]["L1"] == "miss"
    again = replay.run_replay(out, backends.HeuristicBackend(), budget=backends.Budget(wall_min=1))
    assert again == {"reviewed": 0, "skipped": 6, "errors": 0}


def test_backend_errors_are_recorded_not_fatal(corpus, tmp_path, skill) -> None:
    out = tmp_path / "replay"
    replay.build_samples(corpus, out, side="train", sample=2, seed=1)

    class Broken(backends.Backend):
        name = "broken"

        def run(self, brief):
            raise backends.BackendError("boom")

    counts = replay.run_replay(out, Broken(), budget=backends.Budget(wall_min=1))
    assert counts["errors"] == 2 and (out / "errors.jsonl").is_file()
    assert json.loads((out / "errors.jsonl").read_text().splitlines()[0])["error"].endswith("boom")


def test_rebuild_does_not_clobber_existing_verdicts(corpus, tmp_path, skill) -> None:
    out = tmp_path / "replay"
    samples = replay.build_samples(corpus, out, side="train")
    replay.run_replay(out, backends.HeuristicBackend(), budget=backends.Budget(wall_min=1))
    vp = schema.verdict_path(samples[0].session_dir / "memory" / "cards" / f"{samples[0].card_id}.yaml")
    before = vp.read_text()
    replay.build_samples(corpus, out, side="train")
    assert vp.read_text() == before


def test_truth_gets_a_comparator_from_the_parent_card_when_the_corpus_has_none(corpus, tmp_path, skill) -> None:
    """Round-00 finding: 41 % of L2 was unscorable. The parent card's measurement is the comparator the corpus implied."""
    out = tmp_path / "replay"
    samples = replay.build_samples(corpus, out, side="train")
    s = next(x for x in samples if x.run_ref == "r-aaaa" and x.card_id == "exp-02")
    t = yaml.safe_load(s.truth_path.read_text())
    assert t["evaluation"]["comparator"]["value"] == 0.41 and t["evaluation"]["comparator"]["ref"] == "exp-01"
    assert t["evaluation"]["comparator_source"] == "parent_card"
    # the session's copy of the card is untouched: the agent sees exactly what the corpus had
    k = yaml.safe_load((s.session_dir / "memory" / "cards" / "exp-02.yaml").read_text())
    assert (k["evaluation"].get("comparator") or {}).get("value") is None
    # a base_model parent has nothing to fall back to
    s1 = next(x for x in samples if x.run_ref == "r-aaaa" and x.card_id == "exp-01")
    assert "comparator_source" not in yaml.safe_load(s1.truth_path.read_text())["evaluation"]


def test_rewritten_truth_is_picked_up_by_the_ledger_without_any_rescore_step(corpus, tmp_path, skill) -> None:
    out = tmp_path / "replay"
    replay.build_samples(corpus, out, side="train")
    replay.run_replay(out, backends.HeuristicBackend(), budget=backends.Budget(wall_min=1))
    rows = {(r["path"].split("/")[-5], r["card_id"]): r for r in ledger.rows([out])}
    assert rows[("r-aaaa", "exp-02")]["scored"]["L2"] != "unscorable"   # comparator came from the parent card
    assert all(r["truth_path"].split("/")[-3] == "_truth" for r in rows.values() if r["truth_path"])
    assert sum(1 for r in rows.values() if r["truth_path"]) == 5


def test_an_interrupted_session_build_is_completed_on_the_next_build(corpus, tmp_path, skill) -> None:
    """A build killed mid-session left a directory with the earlier cards only; the rerun must finish it."""
    out = tmp_path / "replay"
    samples = replay.build_samples(corpus, out, side="train")
    s = next(x for x in samples if x.run_ref == "r-aaaa" and x.card_id == "exp-03")
    (s.session_dir / "memory" / "cards" / "exp-03.yaml").unlink()
    (s.session_dir / "memory" / "index.md").unlink()
    (s.session_dir / "history").unlink()
    replay.build_samples(corpus, out, side="train")
    assert (s.session_dir / "memory" / "cards" / "exp-03.yaml").is_file()
    assert (s.session_dir / "memory" / "index.md").is_file() and (s.session_dir / "history").is_symlink()
