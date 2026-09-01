"""Every check here corresponds to a clean-looking wrong answer someone already got."""

from __future__ import annotations

import json

import pytest

from awm.exp_protocol import preflight
from exp_protocol_cards import plan_card


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def session(tmp_path):
    """A session dir with a script, 100 well-formed rows ending in <|im_end|>, and a comparator eval."""
    (tmp_path / "train.py").write_text("pass\n")
    rows = [{"prompt": f"q{i}", "completion": f"reasoning {i} #### {i}<|im_end|>"} for i in range(100)]
    write_jsonl(tmp_path / "data" / "train.jsonl", rows)
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "base.json").write_text(json.dumps({"accuracy": 0.33, "n": 150}))
    (tmp_path / "ckpts").mkdir()
    card = plan_card()
    card["setup"]["command"] = {"argv": ["python", "train.py"], "cwd": str(tmp_path),
                                "script": str(tmp_path / "train.py")}
    card["setup"]["data"] = [{"path": str(tmp_path / "data" / "train.jsonl"),
                              "source": "synthetic:self", "n_examples": 100}]
    card["setup"]["output_dir"] = str(tmp_path / "ckpts" / "exp-01")
    card["setup"]["method"].update({"stop_token": "<|im_end|>", "answer_marker": "#### ",
                                    "hyperparams": {"max_seq_len": 512}})
    card["evaluation"]["comparator"] = {"ref": "base_model", "value": 0.33,
                                        "path": str(tmp_path / "eval" / "base.json")}
    return tmp_path, card


def status(report, check):
    return next(r["status"] for r in report["results"] if r["check"] == check)


def test_a_good_setup_passes_every_mechanised_check(session) -> None:
    root, card = session
    report = preflight.run_preflight(card, root, pitfalls=[])
    assert report["summary"]["fail"] == 0, report["results"]
    for check in ("data_files_exist", "data_n_examples_match", "command_resolves",
                  "stop_token_consistent", "answer_marker_single", "max_seq_len_headroom",
                  "comparator_same_protocol"):
        assert status(report, check) == "pass", check


def test_missing_data_file_fails(session) -> None:
    root, card = session
    card["setup"]["data"][0]["path"] = str(root / "data" / "nope.jsonl")
    report = preflight.run_preflight(card, root, pitfalls=[])
    assert status(report, "data_files_exist") == "fail"


def test_n_examples_mismatch_fails(session) -> None:
    root, card = session
    card["setup"]["data"][0]["n_examples"] = 99
    assert status(preflight.run_preflight(card, root, pitfalls=[]), "data_n_examples_match") == "fail"


def test_eos_mismatch_fails(session) -> None:
    root, card = session
    rows = [{"prompt": "q", "completion": "answer #### 1<|endoftext|>"} for _ in range(100)]
    write_jsonl(root / "data" / "train.jsonl", rows)
    assert status(preflight.run_preflight(card, root, pitfalls=[]), "stop_token_consistent") == "fail"


def test_undeclared_stop_token_is_a_warning_not_a_pass(session) -> None:
    root, card = session
    card["setup"]["method"]["stop_token"] = None
    assert status(preflight.run_preflight(card, root, pitfalls=[]), "stop_token_consistent") == "warn"


def test_double_answer_format_fails(session) -> None:
    root, card = session
    rows = [{"prompt": "q", "completion": "#### 3 then #### 3<|im_end|>"} for _ in range(100)]
    write_jsonl(root / "data" / "train.jsonl", rows)
    assert status(preflight.run_preflight(card, root, pitfalls=[]), "answer_marker_single") == "fail"


def test_rows_longer_than_max_seq_len_fail(session) -> None:
    root, card = session
    card["setup"]["method"]["hyperparams"]["max_seq_len"] = 2   # every row is ~7 estimated tokens
    assert status(preflight.run_preflight(card, root, pitfalls=[]), "max_seq_len_headroom") == "fail"


def test_comparator_measured_under_a_different_n_fails(session) -> None:
    root, card = session
    (root / "eval" / "base.json").write_text(json.dumps({"accuracy": 0.33, "n": 300}))
    assert status(preflight.run_preflight(card, root, pitfalls=[]), "comparator_same_protocol") == "fail"


def test_local_parent_checkpoint_needs_config_json(session) -> None:
    root, card = session
    (root / "ckpts" / "parent").mkdir()
    card["setup"]["parent_checkpoint"] = {"path": str(root / "ckpts" / "parent"), "origin": "base_model"}
    assert status(preflight.run_preflight(card, root, pitfalls=[]), "parent_checkpoint_loadable") == "fail"
    (root / "ckpts" / "parent" / "config.json").write_text("{}")
    assert status(preflight.run_preflight(card, root, pitfalls=[]), "parent_checkpoint_loadable") == "pass"


def test_manual_pitfalls_come_back_as_reminders(session) -> None:
    root, card = session
    pitfalls = [
        {"id": "template_unreachable", "symptom": "s", "cause": "c", "check": None, "guidance": "g", "source": "x"},
        {"id": "eos_mismatch", "symptom": "s", "cause": "c", "check": "stop_token_consistent", "guidance": "g", "source": "x"},
    ]
    report = preflight.run_preflight(card, root, pitfalls=pitfalls)
    assert [r["id"] for r in report["reminders"]] == ["template_unreachable"]


def test_the_shipped_catalogue_loads_and_names_only_real_checks() -> None:
    pitfalls = preflight.load_pitfalls()
    assert len(pitfalls) >= 6
    ids = {p["id"] for p in pitfalls}
    assert len(ids) == len(pitfalls)
    for p in pitfalls:
        assert set(p) >= {"id", "symptom", "cause", "check", "guidance", "source"}
        assert p["check"] is None or p["check"] in preflight.CHECKS, p["id"]
