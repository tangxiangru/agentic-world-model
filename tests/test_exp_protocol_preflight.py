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


# ---- review findings (2026-09-01) --------------------------------------------

def test_chat_rows_with_an_id_field_are_still_measured(session) -> None:
    """C1: {"id", "messages"} rows used to pass with 'longest ~0' because the id string short-circuited."""
    root, card = session
    long = "x" * 20000
    rows = [{"id": f"r{i}", "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": long}]}
            for i in range(20)]
    write_jsonl(root / "data" / "train.jsonl", rows)
    r = next(x for x in preflight.run_preflight(card, root, pitfalls=[])["results"] if x["check"] == "max_seq_len_headroom")
    assert r["status"] == "fail", r
    assert "longest ~0" not in r["detail"]


def test_rows_with_no_measurable_text_never_pass(session) -> None:
    root, card = session
    write_jsonl(root / "data" / "train.jsonl", [{"id": 1, "n": 2} for _ in range(5)])
    r = next(x for x in preflight.run_preflight(card, root, pitfalls=[])["results"] if x["check"] == "max_seq_len_headroom")
    assert r["status"] == "skip"


def test_short_answer_field_does_not_shadow_the_training_text(session) -> None:
    """I5: a row carrying both answer (gold) and text (the target) must be checked on text."""
    root, card = session
    rows = [{"prompt": "q", "answer": "141", "text": "reasoning #### 141<|im_end|>"} for _ in range(50)]
    write_jsonl(root / "data" / "train.jsonl", rows)
    report = preflight.run_preflight(card, root, pitfalls=[])
    assert status(report, "stop_token_consistent") == "pass"
    detail = next(x["detail"] for x in report["results"] if x["check"] == "stop_token_consistent")
    assert "field=text" in detail


def test_details_state_the_sample_and_the_threshold(session) -> None:
    root, card = session
    report = preflight.run_preflight(card, root, pitfalls=[])
    d = {x["check"]: x["detail"] for x in report["results"]}
    assert "first 500 rows" in d["stop_token_consistent"] and ">=95%" in d["stop_token_consistent"]
    assert "first 500 rows" in d["max_seq_len_headroom"] and "chars/4" in d["max_seq_len_headroom"]


def test_missing_catalogue_is_reported_not_raised(session, monkeypatch) -> None:
    """I2: preflight must not traceback because pitfalls.yaml is not where skill_dir() points."""
    root, card = session
    monkeypatch.setenv("AWM_EXP_PROTOCOL_DIR", str(root / "nowhere"))
    report = preflight.run_preflight(card, root)          # pitfalls=None -> loads the catalogue
    assert report["catalogue"] is None and report["reminders"] == []
    assert "pitfalls.yaml" in preflight.render(report)


def test_the_installed_copy_in_the_session_dir_is_preferred(session, monkeypatch) -> None:
    root, card = session
    monkeypatch.setenv("AWM_EXP_PROTOCOL_DIR", str(root / "nowhere"))
    local = root / "skills" / "exp_protocol" / "pitfalls.yaml"
    local.parent.mkdir(parents=True)
    local.write_text("- {id: local_only, symptom: s, cause: c, check: null, guidance: g, source: x}\n")
    report = preflight.run_preflight(card, root)
    assert report["catalogue"] == str(local)
    assert [r["id"] for r in report["reminders"]] == ["local_only"]


def test_relative_parent_path_is_not_mistaken_for_a_hub_id(session) -> None:
    root, card = session
    card["setup"]["parent_checkpoint"] = {"path": "ckpts/exp-02/final", "origin": "exp-02"}
    assert status(preflight.run_preflight(card, root, pitfalls=[]), "parent_checkpoint_loadable") == "fail"
    card["setup"]["parent_checkpoint"] = {"path": "google/gemma-3-4b-pt", "origin": "base_model"}
    assert status(preflight.run_preflight(card, root, pitfalls=[]), "parent_checkpoint_loadable") == "pass"


def test_script_not_named_in_argv_is_a_warning(session) -> None:
    root, card = session
    card["setup"]["command"]["argv"] = ["python", "train2.py"]
    r = next(x for x in preflight.run_preflight(card, root, pitfalls=[])["results"] if x["check"] == "command_resolves")
    assert r["status"] == "warn" and "argv" in r["detail"]
