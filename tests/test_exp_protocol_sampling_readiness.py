"""CPU orchestration: real cards/locks, inert factories, no model execution."""

import json
import shutil
import subprocess
import sys

import pytest
from test_exp_protocol_sampling import Tokenizer, outputs_for

from awm.cli import main
from awm.exp_protocol import sampling, schema

pytest_plugins = ["test_exp_protocol_sampling"]


@pytest.fixture
def ready_case(case, monkeypatch):
    monkeypatch.setattr(sampling, "_native_runtime", lambda params: {"adapter": "inert_test"})
    return case


def run_factory(case, factory, **kwargs):
    root, card, prompts, _, params = case
    return sampling.record_vllm_from_factory(
        factory,
        prompts,
        params,
        root / "draws",
        tokenizer=Tokenizer(),
        card_path=card,
        required_stop_tokens=["<stop>"],
        **kwargs,
    )


@pytest.mark.parametrize("defect", ["lock", "script", "stop", "tokens", "existing_output"])
def test_cpu_failure_happens_before_factory(ready_case, defect):
    root, card, prompts, engine, params = ready_case
    if defect == "lock":
        card.with_suffix(".lock.json").unlink()
    elif defect == "script":
        (root / "sample.py").write_text("# changed\n")
    elif defect == "stop":
        params.stop_token_ids = []
    elif defect == "tokens":
        prompts = (sampling.PreparedPrompt(0, 0, "<bos> q", (2, 99), "single_at_start"),)
        ready_case = root, card, prompts, engine, params
    else:
        (root / "draws").mkdir()
        (root / "draws/raw.jsonl").write_text("retained\n")
    with pytest.raises((ValueError, FileExistsError)):
        run_factory(ready_case, lambda: pytest.fail("factory must not run"))
    if defect == "existing_output":
        assert (root / "draws/raw.jsonl").read_text() == "retained\n"


@pytest.mark.parametrize("late_create", [False, True])
def test_lock_override_cannot_waive_future_input_hash(ready_case, late_create):
    root, path, *_ = ready_case
    future = root / "future-training.jsonl"
    card = schema.load_card(path)
    card["setup"]["data"] = [{"path": str(future), "source": "derived", "n_examples": 1}]
    schema.dump_card(path, card)
    assert (
        main(
            [
                "exp_protocol",
                "lock",
                "--dir",
                str(root),
                "exp-01",
                "--relock",
                "new phase",
                "--override",
                "data_files_exist=will be generated later",
            ]
        )
        == 0
    )
    if late_create:
        future.write_text('{"completion":"generated"}\n')
    with pytest.raises(ValueError, match="pinned hash"):
        run_factory(ready_case, lambda: pytest.fail("future data cannot start engine"))


def test_staged_sampling_succeeds_and_keeps_raw_on_later_validation_failure(
    ready_case, monkeypatch
):
    root, _, _, engine, _ = ready_case
    cleaned = []

    def incomplete(engine, inputs, params):
        rows = outputs_for(inputs, params)
        rows[0].finished = False
        return rows

    monkeypatch.setattr(sampling, "_call_engine", incomplete)
    with pytest.raises(sampling.SamplingEvidenceError):
        run_factory(ready_case, lambda: engine, close_engine=cleaned.append)
    assert cleaned == [engine]
    raw = (root / "draws/raw.jsonl").read_bytes()
    failure = json.loads((root / "draws/capture-failure.json").read_text())
    assert raw and failure["raw_durable"]
    assert not (root / "draws/capture.json").exists()


def test_factory_cannot_make_stale_card_valid(ready_case):
    root, _, _, engine, _ = ready_case

    def factory():
        (root / "sample.py").write_text("# changed during construction\n")
        return engine

    with pytest.raises(ValueError):
        run_factory(ready_case, factory)
    assert engine.called == 0


def test_persisted_sampling_data_can_be_pinned_by_the_next_card(ready_case):
    root, path, _, engine, _ = ready_case
    run_factory(ready_case, lambda: engine)
    summary = sampling.parse_recording(root / "draws", lambda text, _: {"completion": text})
    # No training: verify the real next-card lock/live-input contract on CPU.
    card = schema.load_card(path)
    card["card_id"] = "exp-02"
    card["setup"]["data"] = [
        {"path": summary["parse_path"], "source": "derived:exp-01", "n_examples": 4}
    ]
    next_path = path.with_name("exp-02.yaml")
    schema.dump_card(next_path, card)
    assert main(["exp_protocol", "lock", "--dir", str(root), "exp-02"]) == 0
    from awm.exp_protocol.execution import _live_plan

    _, locked, _ = _live_plan(next_path, root)
    assert locked["data"][0]["sha256"] == summary["parse_sha256"]
    with open(summary["parse_path"], "a") as stream:
        stream.write("changed\n")
    with pytest.raises(ValueError):
        _live_plan(next_path, root)


def test_cleanup_failure_does_not_destroy_retained_capture(ready_case):
    root, _, _, engine, _ = ready_case

    def bad_cleanup(_):
        raise RuntimeError("cleanup failed")

    with pytest.raises(RuntimeError, match="cleanup failed"):
        run_factory(ready_case, lambda: engine, close_engine=bad_cleanup)
    assert json.loads((root / "draws/capture.json").read_text())["raw_durable"]
    assert (root / "draws/raw.jsonl").is_file()


def test_native_adapter_validation_precedes_factory(case, monkeypatch):
    pytest.importorskip("vllm")
    from vllm import SamplingParams

    root, path, prompts, engine, _ = case
    params = SamplingParams(n=2, max_tokens=20, stop_token_ids=[9])
    native_case = root, path, prompts, engine, params
    monkeypatch.setitem(sampling.VLLM_SOURCES, "outputs.py", "0" * 64)
    with pytest.raises(sampling.SamplingEvidenceError, match="source mismatch"):
        run_factory(native_case, lambda: pytest.fail("native validation precedes factory"))


def test_documented_stage_deadline_preserves_partial_file(tmp_path):
    if shutil.which("timeout") is None:
        pytest.skip("GNU timeout unavailable")
    # An owned synthetic process ignores TERM, requiring the bounded KILL fallback.
    script = tmp_path / "stage.py"
    raw = tmp_path / "raw.jsonl"
    script.write_text(
        "import signal,time\nfrom pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"Path({str(raw)!r}).write_text('partial raw\\n')\n"
        "while True: time.sleep(1)\n"
    )
    result = subprocess.run(
        ["timeout", "--signal=TERM", "--kill-after=0.2s", "0.5s", sys.executable, str(script)],
        timeout=5,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0 and raw.read_text() == "partial raw\n"
