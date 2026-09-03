"""Recorder mode: the record contract, sufficiency, the ledger, snapshot, CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from awm.wm import record
from awm.wm.schema import CARD_SCHEMA, WMError

REPO = Path(__file__).resolve().parent.parent

PLAN = """First run: SFT on worked solutions from the base model.

```bash
python train.py --model_name_or_path google/gemma-3-4b-pt --train_file data/train.jsonl --output_dir runs/exp01 --max_steps 1000
```

Evaluate with `python evaluate.py --limit 150` at each checkpoint.
"""


def full_card(card_id: str = "exp-01") -> dict:
    return {
        "schema_version": CARD_SCHEMA, "card_id": card_id,
        "problem": {"statement": "base model slips on arithmetic"},
        "hypothesis": {"claim": None},
        "setup": {
            "parent_checkpoint": {"path": "google/gemma-3-4b-pt", "origin": "base_model"},
            "data": [{"path": "data/train.jsonl", "source": "HF meta-math/MetaMathQA", "n_examples": 10,
                      "build_command": ["python", "prep.py", "--seed", "42"]}],
            "method": {"family": "sft", "framework": "trl 0.27.0 / transformers 4.57",
                       "hyperparams": {"lr": 2e-5, "epochs": 1, "batch_size": 4, "seed": 42, "precision": "bf16"}},
            "command": {"argv": ["python", "train.py", "--max_steps", "1000"], "script": "train.py"},
        },
        "evaluation": {"protocol": {"command": ["python", "evaluate.py", "--limit", "150"], "n": 150}},
        "result": {"execution": "completed", "output_checkpoint": "runs/exp01/checkpoint-1000",
                   "measurements": [{"step": 1000, "metric": "accuracy", "value": 0.34, "n": 150,
                                     "source": "eval_step1000.json"}]},
    }


def good_response(card: dict, *, stage: str = "plan", questions: list | None = None) -> dict:
    return {"schema_version": record.RESPONSE_SCHEMA, "stage": stage, "card": card,
            "questions": questions or [], "ack": "recorded exp-01 at plan stage"}


def test_validate_rejects_advisory_content() -> None:
    resp = good_response(full_card())
    assert record.validate_response(resp) == []
    for key in ("verdict", "prediction", "eval_plan", "suggestion", "advice"):
        bad = {**good_response(full_card()), key: {"anything": 1}}
        assert any(key in p for p in record.validate_response(bad)), key
    in_card = good_response({**full_card(), "suggestion": {"label": "ADJUST"}})
    assert any("suggestion" in p for p in record.validate_response(in_card))


def test_validate_shape() -> None:
    assert any("schema_version" in p for p in record.validate_response({"stage": "plan"}))
    assert any("stage" in p for p in record.validate_response({**good_response(full_card()), "stage": "shipped"}))
    assert any("questions" in p for p in record.validate_response(
        {**good_response(full_card()), "questions": ["a", "b", "c", "d"]}))
    assert any("ack" in p for p in record.validate_response({**good_response(full_card()), "ack": ""}))


def test_sufficiency_stages() -> None:
    card = full_card()
    assert record.check_sufficiency(card, "plan") == []
    plan_only = full_card(); plan_only.pop("result"); plan_only["evaluation"] = {"protocol": {}}
    # at plan stage, result-phase fields are not yet required
    assert record.check_sufficiency(plan_only, "plan") == []
    missing = record.check_sufficiency(plan_only, "closed")
    assert "result.execution" in missing and "result.measurements" in missing
    assert "evaluation.protocol.command" in missing
    gap = full_card(); gap["setup"]["method"]["hyperparams"] = {}
    assert record.check_sufficiency(gap, "plan") == ["setup.method.hyperparams"]
    # results reported while running pull in the result-phase fields
    running = full_card(); running["evaluation"] = {"protocol": {}}
    assert "evaluation.protocol.n" in record.check_sufficiency(running, "running")


def test_log_record_and_outcome(tmp_path: Path) -> None:
    wm_dir = tmp_path / "wm"
    with pytest.raises(WMError):
        record.log_record(wm_dir, {**good_response(full_card()), "verdict": {}}, request=PLAN, model="m")
    entry = record.log_record(wm_dir, good_response(full_card()), request=PLAN, model="m")
    assert entry["card_id"] == "exp-01" and entry["record_n"] == 1 and entry["missing"] == []
    incomplete = full_card("exp-02"); incomplete["setup"]["command"] = {}
    entry = record.log_record(wm_dir, good_response(incomplete, stage="closed"), request="x", model="m")
    assert set(entry["missing"]) == {"setup.command.argv", "setup.command.script"}
    stored = json.loads((wm_dir / "cards" / "exp-02" / "record-01.json").read_text())
    assert stored["response"]["missing"] == entry["missing"]
    record.record_outcome(wm_dir, "exp-01", final_value=0.71, shipped="runs/exp01")
    card = json.loads((wm_dir / "cards" / "exp-01" / "card.json").read_text())
    assert card["outcome"]["final_value"] == 0.71
    rows = record.RecordLedger(wm_dir / "records.jsonl").rows()
    assert rows[-1]["event"] == "outcome" and len(rows) == 3


def test_cli_recorder_end_to_end(tmp_path: Path) -> None:
    sd = tmp_path / "task"; (sd / "data").mkdir(parents=True); (sd / "wm" / "tmp").mkdir(parents=True)
    (sd / "data" / "train.jsonl").write_text('{"q": 1}\n')
    (sd / "train.py").write_text("print('train')\n")
    env = {**os.environ, "AWM_SESSION_DIR": str(sd), "PYTHONPATH": str(REPO)}

    def wm(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, "-m", "awm.cli", "wm", *args], capture_output=True, text=True, env=env, check=False)

    r = wm("init", "--mode", "record", "--arm", "retrieval", "--wma-model", "claude-opus-5",
           "--base-model", "google/gemma-3-4b-pt")
    assert r.returncode == 0, r.stderr
    assert "forcing arm null" in r.stderr
    cfg = json.loads((sd / "wm" / "config.json").read_text())
    assert cfg["mode"] == "record" and cfg["arm"] == "null"
    assert (sd / "wm" / "records.jsonl").is_file()

    r = wm("draft-card", "--text", PLAN)
    assert r.returncode == 0, r.stderr
    assert yaml.safe_load(r.stdout)["card_id"] == "exp-01"

    r = wm("snapshot", "--card", "exp-01", str(sd / "train.py"))
    assert r.returncode == 0, r.stderr
    assert (sd / "wm" / "cards" / "exp-01" / "snapshot" / "train.py").read_text() == "print('train')\n"
    manifest = json.loads((sd / "wm" / "cards" / "exp-01" / "snapshot" / "MANIFEST.json").read_text())
    assert manifest["files"][0]["path"] == "train.py" and manifest["files"][0]["sha256"]
    assert wm("snapshot", "--card", "exp-01", "/etc/hosts").returncode != 0

    (sd / "wm" / "tmp" / "response.json").write_text(json.dumps(good_response(full_card())))
    (sd / "wm" / "tmp" / "request.txt").write_text(PLAN)
    r = wm("record", "--response", str(sd / "wm/tmp/response.json"), "--request", str(sd / "wm/tmp/request.txt"))
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["missing"] == []

    # a consult-shaped response is refused in recorder mode
    (sd / "wm" / "tmp" / "response.json").write_text(json.dumps(
        {**good_response(full_card()), "verdict": {"label": "SURE_WILL_WORK", "confidence": 0.9}}))
    assert wm("record", "--response", str(sd / "wm/tmp/response.json")).returncode != 0

    r = wm("outcome", "--card", "exp-01", "--final", "0.71", "--shipped", "runs/exp01")
    assert r.returncode == 0, r.stderr
    st = json.loads(wm("status").stdout)
    assert st["mode"] == "record"
    assert st["cards"]["exp-01"] == {"records": 2, "stage": "closed", "missing": [], "outcome": 0.71}


def test_sufficiency_requires_output_checkpoint_when_completed() -> None:
    card = full_card()
    assert record.check_sufficiency(card, "closed") == []
    del card["result"]["output_checkpoint"]
    assert record.check_sufficiency(card, "closed") == ["result.output_checkpoint"]
    crashed = full_card()
    crashed["result"] = {"execution": "failed", "measurements": [{"metric": "n/a", "value": 0, "n": 0, "source": "train.log"}]}
    assert "result.output_checkpoint" not in record.check_sufficiency(crashed, "closed")


def test_archive_checkpoint(tmp_path: Path) -> None:
    sd = tmp_path / "task"
    ckpt = sd / "runs" / "exp01" / "checkpoint-1000"
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text('{"model_type": "gemma"}')
    (ckpt / "model.safetensors").write_bytes(b"\x00" * 128)
    wm_dir = sd / "wm"
    record.log_record(wm_dir, good_response(full_card()), request=PLAN, model="m")

    with pytest.raises(WMError):  # outside the session dir
        record.archive_checkpoint(wm_dir, sd, "exp-01", tmp_path)
    with pytest.raises(WMError):  # not a checkpoint directory
        (sd / "notes").mkdir()
        record.archive_checkpoint(wm_dir, sd, "exp-01", sd / "notes")

    manifest = record.archive_checkpoint(wm_dir, sd, "exp-01", ckpt)
    dest = wm_dir / "checkpoints" / "exp-01"
    assert (dest / "config.json").is_file() and (dest / "model.safetensors").stat().st_size == 128
    assert manifest["source"] == str(ckpt) and manifest["bytes_total"] == 128 + len('{"model_type": "gemma"}')
    by_path = {f["path"]: f for f in manifest["files"]}
    assert by_path["model.safetensors"]["sha256"]
    stored = json.loads((wm_dir / "cards" / "exp-01" / "card.json").read_text())
    assert stored["result"]["archived_checkpoint"] == str(dest)
    # the archive survives the scientist deleting the original
    import shutil
    shutil.rmtree(ckpt)
    assert (dest / "model.safetensors").is_file()
    with pytest.raises(WMError):  # one archive per card
        record.archive_checkpoint(wm_dir, sd, "exp-01", dest)
