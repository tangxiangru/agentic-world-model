"""What was written before the run cannot change after it — and neither can the script it named."""

from __future__ import annotations

import json

from awm.exp_protocol import lock, schema
from exp_protocol_cards import closed_card, plan_card


def setup_card(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("print('v1')\n")
    card = plan_card()
    card["setup"]["command"]["script"] = str(script)
    path = tmp_path / "memory" / "cards" / "exp-01.yaml"
    schema.dump_card(path, card)
    return path, card, script


def test_lock_records_plan_and_script_hashes(tmp_path) -> None:
    path, card, script = setup_card(tmp_path)
    info = lock.write_lock(path, card, {"fail": 0, "warn": 1, "pass": 5})
    on_disk = json.loads(lock.lock_path(path).read_text())
    assert on_disk == info
    assert info["plan_sha256"] == schema.plan_hash(card)
    assert info["script"] == {"path": str(script), "sha256": schema.sha256_file(script)}
    assert info["preflight"] == {"fail": 0, "warn": 1, "pass": 5}


def test_verify_passes_when_only_result_sections_were_added(tmp_path) -> None:
    path, card, _ = setup_card(tmp_path)
    lock.write_lock(path, card, {})
    done = closed_card()
    done["setup"]["command"]["script"] = card["setup"]["command"]["script"]
    assert lock.verify_lock(path, done).ok


def test_verify_fails_when_a_plan_field_changed(tmp_path) -> None:
    path, card, _ = setup_card(tmp_path)
    lock.write_lock(path, card, {})
    card["hypothesis"]["claim"] = "a better story"
    r = lock.verify_lock(path, card)
    assert not r.ok and r.errors[0].field == "plan"


def test_verify_fails_when_the_script_changed_after_lock(tmp_path) -> None:
    path, card, script = setup_card(tmp_path)
    lock.write_lock(path, card, {})
    script.write_text("print('v2')\n")
    r = lock.verify_lock(path, card)
    assert not r.ok and r.errors[0].field == "setup.command.script"


def test_verify_warns_when_the_script_is_gone(tmp_path) -> None:
    path, card, script = setup_card(tmp_path)
    lock.write_lock(path, card, {})
    script.unlink()
    r = lock.verify_lock(path, card)
    assert r.ok and r.warnings[0].field == "setup.command.script"


def test_verify_without_a_lock_is_an_error(tmp_path) -> None:
    path, card, _ = setup_card(tmp_path)
    r = lock.verify_lock(path, card)
    assert not r.ok and r.errors[0].field == "lock"


def test_lock_without_a_script_field_records_null(tmp_path) -> None:
    path, card, _ = setup_card(tmp_path)
    card["setup"]["command"]["script"] = None
    info = lock.write_lock(path, card, {})
    assert info["script"] is None


# ---- review findings (2026-09-01) --------------------------------------------

def test_a_second_lock_without_a_reason_is_refused(tmp_path) -> None:
    """I1: lock -> edit -> lock again used to erase the trace."""
    path, card, _ = setup_card(tmp_path)
    lock.write_lock(path, card, {})
    card["hypothesis"]["claim"] = "a story that fits the result"
    import pytest
    with pytest.raises(lock.LockExists):
        lock.write_lock(path, card, {})


def test_a_relock_keeps_the_previous_hash_and_the_reason(tmp_path) -> None:
    path, card, _ = setup_card(tmp_path)
    first = lock.write_lock(path, card, {})
    card["hypothesis"]["claim"] = "a story that fits the result"
    second = lock.write_lock(path, card, {}, relock_reason="parent checkpoint path was wrong")
    assert second["relocked_from"] == [{"plan_sha256": first["plan_sha256"], "locked_at": first["locked_at"],
                                        "overrides": {}, "reason": "parent checkpoint path was wrong"}]
    assert lock.verify_lock(path, card).ok
    third = lock.write_lock(path, card, {}, relock_reason="again")
    assert [r["reason"] for r in third["relocked_from"]] == ["parent checkpoint path was wrong", "again"]


def test_data_files_are_hashed_and_a_change_is_detected(tmp_path) -> None:
    """I12: the data paths were in the hash, the bytes were not."""
    path, card, _ = setup_card(tmp_path)
    data = tmp_path / "train.jsonl"
    data.write_text('{"completion": "a"}\n')
    card["setup"]["data"] = [{"path": str(data), "source": "local", "n_examples": 1}]
    info = lock.write_lock(path, card, {})
    assert info["data"] == [{"path": str(data), "sha256": schema.sha256_file(data), "bytes": data.stat().st_size}]
    data.write_text('{"completion": "b"}\n')
    r = lock.verify_lock(path, card)
    assert not r.ok and r.errors[0].field == "setup.data[0].path"
    data.unlink()
    r = lock.verify_lock(path, card)
    assert r.ok and r.warnings[0].field == "setup.data[0].path"


def test_overrides_are_recorded_in_the_lock(tmp_path) -> None:
    path, card, _ = setup_card(tmp_path)
    info = lock.write_lock(path, card, {"fail": 1}, overrides={"max_seq_len_headroom": "rows are code; chars/4 is wrong"})
    assert info["overrides"] == {"max_seq_len_headroom": "rows are code; chars/4 is wrong"}
