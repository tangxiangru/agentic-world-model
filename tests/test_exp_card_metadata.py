"""Generated metadata for the published reconstructed-card corpus."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml


REPO = Path(__file__).resolve().parent.parent
AUTHORITATIVE_PROBLEMS_SHA256 = (
    "e86c4879ffd8c8529989a258ec0e520a98327ef95f921b6647977aa94f69d7c2"
)


def _load_generator():
    path = REPO / "tools" / "rebuild_exp_card_metadata.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


metadata = _load_generator()


def _write_split(path: Path, *, train: list[str], test: list[str]) -> None:
    path.write_text(yaml.safe_dump({"splits": {"train": train, "test": test}}))


def _card(
    corpus: Path,
    *,
    side: str,
    run_path: str,
    number: int,
    source: str = "local",
    measurement: float | None = None,
) -> Path:
    ref = metadata.run_ref(run_path)
    path = corpus / side / ref / f"exp-{number:02d}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    measurements = [] if measurement is None else [{"metric": "accuracy", "value": measurement}]
    value = {
        "card_id": path.stem,
        "problem": {"statement": "observed failure", "evidence": [], "failure_examples": []},
        "hypothesis": {
            "claim": "training helps",
            "mechanism": None,
            "expected_effect": {"metric": "accuracy"},
            "falsified_if": None,
        },
        "setup": {
            "base_model": "model|with-pipe",
            "parent_checkpoint": {"path": "model", "origin": "base_model"},
            "method": {"family": "sft"},
            "data": [{"source": source}],
            "command": {"argv": ["python", "train.py"]},
            "budget": {"gpu": "one"},
        },
        "evaluation": {"protocol": {}, "comparator": None, "diagnostic": {}},
        "result": {
            "execution": "completed",
            "measurements": measurements,
            "training_summary": {},
            "diagnostic_result": {},
        },
        "conclusion": {
            "verdict": "inconclusive",
            "mechanism_verdict": "not_tested",
            "summary": "done",
            "decision": "adopt",
            "next_step": None,
        },
        "outcome": {"official_accuracy": 0.0},
        "provenance": {
            "run_ref": ref,
            "launch_i": number,
            "stated_by_agent": {"hypothesis": False},
        },
    }
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    return path


def test_render_excludes_probes_orders_numeric_cards_and_reports_missing(tmp_path: Path) -> None:
    corpus = tmp_path / "cards"
    train_paths = ["agent/train-a", "agent/train-missing"]
    test_paths = ["agent/test-a"]
    split = tmp_path / "split.yaml"
    _write_split(split, train=train_paths, test=test_paths)

    _card(
        corpus,
        side="train",
        run_path=train_paths[0],
        number=10,
        source="a\n| b",
        measurement=0.4,
    )
    _card(corpus, side="train", run_path=train_paths[0], number=2, measurement=0.7)
    _card(corpus, side="test", run_path=test_paths[0], number=1)
    _card(corpus / ".probe-v1", side="train", run_path=train_paths[1], number=1)

    rendered = metadata.render_metadata(corpus, split)
    coverage = json.loads(rendered.coverage)
    missing_ref = metadata.run_ref(train_paths[1])
    assert coverage["cards"] == 3
    assert coverage["cards_by_side"] == {"train": 2, "test": 1}
    assert coverage["runs"] == 2
    assert coverage["expected_runs"] == 3
    assert coverage["runs_without_cards"]["by_side"] == {"train": [missing_ref], "test": []}
    assert coverage["runs_without_cards"]["cause"] == "unknown"
    assert coverage["field_coverage"]["hypothesis.claim"] == 3
    assert coverage["field_coverage"]["result.measurements"] == 2
    assert coverage["field_coverage"]["evaluation.comparator"] == 0
    assert coverage["problems"] == []

    assert rendered.index.index("| exp-02 |") < rendered.index.index("| exp-10 |")
    assert "model\\|with-pipe" in rendered.index
    assert "a \\| b" in rendered.index
    assert "| 0.7 |" in rendered.index
    assert "| 0.0 |" in rendered.index
    assert ".probe-v1" not in rendered.index


def test_rebuild_preserves_unreproducible_source_audit_problems(tmp_path: Path) -> None:
    corpus = tmp_path / "cards"
    split = tmp_path / "split.yaml"
    _write_split(split, train=["agent/train"], test=[])
    _card(corpus, side="train", run_path="agent/train", number=1)
    (corpus / "test").mkdir()
    problems = ["card one: source digest launch check could not be reproduced"]
    (corpus / "coverage.json").write_text(json.dumps({"problems": problems}))

    rendered = metadata.render_metadata(corpus, split)
    assert json.loads(rendered.coverage)["problems"] == problems
    metadata.write_outputs(corpus, rendered)
    assert json.loads((corpus / "coverage.json").read_text())["problems"] == problems


def test_check_mode_never_writes_and_normal_mode_is_byte_stable(tmp_path: Path) -> None:
    corpus = tmp_path / "cards"
    split = tmp_path / "split.yaml"
    _write_split(split, train=["agent/train"], test=[])
    _card(corpus, side="train", run_path="agent/train", number=1)
    (corpus / "test").mkdir()

    assert metadata.main(["--corpus", str(corpus), "--split", str(split), "--check"]) == 1
    assert not (corpus / "coverage.json").exists()
    assert not (corpus / "index.md").exists()

    assert metadata.main(["--corpus", str(corpus), "--split", str(split)]) == 0
    first = ((corpus / "coverage.json").read_bytes(), (corpus / "index.md").read_bytes())
    assert metadata.main(["--corpus", str(corpus), "--split", str(split), "--check"]) == 0
    assert metadata.main(["--corpus", str(corpus), "--split", str(split)]) == 0
    second = ((corpus / "coverage.json").read_bytes(), (corpus / "index.md").read_bytes())
    assert second == first

    (corpus / "index.md").write_text("stale\n")
    assert metadata.main(["--corpus", str(corpus), "--split", str(split), "--check"]) == 1
    assert (corpus / "index.md").read_text() == "stale\n"


def test_card_filename_and_provenance_must_agree(tmp_path: Path) -> None:
    corpus = tmp_path / "cards"
    split = tmp_path / "split.yaml"
    _write_split(split, train=["agent/train"], test=[])
    path = _card(corpus, side="train", run_path="agent/train", number=1)
    (corpus / "test").mkdir()
    value = yaml.safe_load(path.read_text())
    value["provenance"]["run_ref"] = "r-deadbeef"
    path.write_text(yaml.safe_dump(value))

    with pytest.raises(metadata.MetadataError, match="provenance.run_ref"):
        metadata.render_metadata(corpus, split)


def test_committed_corpus_metadata_is_current() -> None:
    corpus = REPO / "results" / "exp-cards" / "gsm8k-gemma-holdout-v1"
    split = REPO / "splits" / "posttrainbench" / "gsm8k-gemma-holdout-v1.yaml"
    rendered = metadata.render_metadata(corpus, split)
    coverage = json.loads(rendered.coverage)
    assert coverage["cards"] == 2030
    assert coverage["cards_by_side"] == {"train": 1580, "test": 450}
    assert coverage["runs"] == 193
    assert coverage["runs_by_side"] == {"train": 143, "test": 50}
    assert coverage["expected_runs_by_side"] == {"train": 143, "test": 50}
    assert {
        side: len(refs)
        for side, refs in coverage["expected_run_refs_by_side"].items()
    } == {"train": 143, "test": 50}
    assert {
        side: set(coverage["expected_run_refs_by_side"][side])
        for side in metadata.SIDES
    } == {
        side: {path.name for path in (corpus / side).iterdir() if path.is_dir()}
        for side in metadata.SIDES
    }
    assert coverage["runs_without_cards"]["count"] == 0
    assert coverage["runs_without_cards"]["by_side"] == {"train": [], "test": []}
    assert len(coverage["problems"]) == 18
    problems_bytes = json.dumps(
        coverage["problems"], ensure_ascii=False, separators=(",", ":")
    ).encode()
    assert hashlib.sha256(problems_bytes).hexdigest() == AUTHORITATIVE_PROBLEMS_SHA256
    assert coverage["source_audit"]["preserved_problem_count"] == 18
    assert rendered.index.count("\n| train |") == 1580
    assert rendered.index.count("\n| test |") == 450
    probe_cards = list(corpus.glob(".probe-*/r-*/exp-*.yaml"))
    production_cards = list(corpus.glob("train/r-*/exp-*.yaml")) + list(
        corpus.glob("test/r-*/exp-*.yaml")
    )
    assert len(probe_cards) == 201
    assert len(production_cards) == 2030
    assert len(list(corpus.rglob("exp-*.yaml"))) == 2231
    assert all(str(path.relative_to(corpus)) not in rendered.index for path in probe_cards)
    assert metadata.stale_outputs(corpus, rendered) == []


def test_authoritative_corpus_seeds_and_validates_both_wma_sides(tmp_path: Path) -> None:
    from awm.wm.memory import Memory

    corpus = REPO / "results" / "exp-cards" / "gsm8k-gemma-holdout-v1"
    memory_root = tmp_path / "memory"
    writer = Memory(
        memory_root,
        session="metadata-integration-seed",
        arm="null",
        visible_sides=("train", "test"),
    )
    assert writer.seed_from_exp_cards(corpus, side="train") == 1580
    assert writer.seed_from_exp_cards(corpus, side="test") == 450

    reader = Memory(
        memory_root,
        session="metadata-integration-read",
        arm="llm",
        readonly=True,
        visible_sides=("train", "test"),
    )
    roots = reader.card_corpus_roots(require=True)
    assert [root.name for root in roots] == ["train", "test"]
    manifests = {
        root.name: json.loads((root / "manifest.json").read_text()) for root in roots
    }
    assert manifests["train"]["card_count"] == 1580
    assert manifests["train"]["expected_run_count"] == 143
    assert manifests["train"]["missing_run_refs"] == []
    assert manifests["test"]["card_count"] == 450
    assert manifests["test"]["expected_run_count"] == 50
    assert manifests["test"]["missing_run_refs"] == []
