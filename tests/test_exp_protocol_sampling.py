"""Sampling metadata/retention tests. Every inference call is inert, never a model."""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from exp_protocol_cards import plan_card

from awm import paths
from awm.cli import main
from awm.exp_protocol import sampling, schema


class Tokenizer:
    bos_token_id = 2
    vocabulary: ClassVar[dict] = {"<bos>": 2, "q": 3, "r": 4, "<stop>": 9}

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [self.vocabulary.get(part, 0) for part in text.split()]

    def get_vocab(self):
        return dict(self.vocabulary)


@dataclass
class Params:
    __struct_fields__ = (
        "n",
        "max_tokens",
        "stop_token_ids",
        "temperature",
        "_real_n",
        "logits_processors",
    )
    n: int = 2
    max_tokens: int = 20
    stop_token_ids: list = field(default_factory=lambda: [9])
    temperature: float = 1.0
    _real_n: int | None = None
    logits_processors: object = None


def outputs_for(inputs, params):
    return [
        SimpleNamespace(
            request_id=f"r{ordinal}",
            prompt_token_ids=item["prompt_token_ids"],
            finished=True,
            outputs=[
                SimpleNamespace(
                    index=i,
                    text="1e999" if i == 0 else "4",
                    token_ids=[5],
                    finish_reason="stop",
                    stop_reason=params.stop_token_ids[0],
                )
                for i in range(params.n)
            ],
        )
        for ordinal, item in enumerate(inputs)
    ]


class Engine:
    def __init__(self):
        self.called = 0

    def get_tokenizer(self):
        return Tokenizer()

    def generate(self, inputs, params, *, use_tqdm):
        assert use_tqdm is False
        self.called += 1
        self.received = inputs, params
        return outputs_for(inputs, params)


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv("AWM_EXP_PROTOCOL_DIR", str(paths.REPO_ROOT / "skills/exp_protocol"))
    monkeypatch.setattr(
        sampling, "_native_identity", lambda engine, params: {"adapter": "inert_test"}
    )
    (tmp_path / "sample.py").write_text("# synthetic, no inference\n")
    card = plan_card()
    card["setup"]["data"] = []
    card["setup"]["method"] = {"family": "other"}
    card["setup"]["command"] = {
        "argv": ["python", str(tmp_path / "sample.py")],
        "cwd": str(tmp_path),
        "script": str(tmp_path / "sample.py"),
    }
    card["setup"]["output_dir"] = str(tmp_path / "output")
    path = tmp_path / "memory/cards/exp-01.yaml"
    schema.dump_card(path, card)
    assert main(["exp_protocol", "lock", "--dir", str(tmp_path), "exp-01"]) == 0
    prompts = sampling.prepare_prompts(
        ["<bos> q", "<bos> r"], Tokenizer(), item_ids=[1, "1"], bos_policy="single_at_start"
    )
    return tmp_path, path, prompts, Engine(), Params()


def capture(case):
    root, path, prompts, engine, params = case
    return sampling.record_vllm(
        engine, prompts, params, root / "draws", card_path=path, required_stop_tokens=["<stop>"]
    )


def test_raw_is_complete_before_parser_and_nonfinite_errors_are_explicit(case):
    root, _, prompts, engine, params = case
    report = capture(case)
    assert engine.received[0] == [{"prompt_token_ids": list(item.token_ids)} for item in prompts]
    assert engine.received[1] is not params and params.stop_token_ids == [9]
    assert (
        report["returned_requests"] == 2
        and report["returned_completions"] == 4
        and report["raw_durable"]
    )
    assert report["returned_tokens"] == 4 and report["engine_call_seconds"] >= 0
    assert report["engine_call_started_at"] and report["engine_call_returned_at"]
    original = (root / "draws/raw.jsonl").read_bytes()
    seen_ids = []

    def parser(text, metadata):
        rows = [json.loads(line) for line in (root / "draws/raw.jsonl").read_text().splitlines()]
        assert len(rows) == 2 and sum(len(row["completions"]) for row in rows) == 4
        seen_ids.append(metadata["item_id"])
        return sampling.finite_float(text)

    summary = sampling.parse_recording(root / "draws", parser)
    assert summary["parsed"] == summary["parser_errors"] == 2
    assert not summary["all_parsed"] and summary["official_score"] is False
    assert seen_ids == [1, 1, "1", "1"]
    assert (root / "draws/raw.jsonl").read_bytes() == original
    records = [json.loads(line) for line in Path(summary["parse_path"]).read_text().splitlines()]
    assert records[0]["status"] == "parser_error" and "value" not in records[0]
    assert records[1]["value"] == 4.0
    assert summary["parser"]["source_sha256"] and summary["parse_seconds"] >= 0


@pytest.mark.parametrize("value", ["nan", "NaN", "inf", "-inf", "1e999", float("nan")])
def test_finite_conversion_is_not_a_nonfinite_success(value):
    with pytest.raises(ValueError):
        sampling.finite_float(value)
    assert math.isfinite(sampling.finite_float("3.5"))


def test_bos_and_stop_resolution_are_actual_tokenizer_checks():
    assert sampling.resolve_stop_ids(Tokenizer(), ["<stop>"]) == [9]
    with pytest.raises(sampling.SamplingEvidenceError, match="single"):
        sampling.prepare_prompts(["<bos> <bos> q"], Tokenizer(), bos_policy="single_at_start")
    with pytest.raises(sampling.SamplingEvidenceError, match="known"):
        sampling.resolve_stop_ids(Tokenizer(), ["<missing>"])


@pytest.mark.parametrize("defect", ["stop", "count", "best_of", "custom"])
def test_invalid_actual_parameters_do_not_invoke_engine(case, defect):
    _, _, _, engine, params = case
    if defect == "stop":
        params.stop_token_ids = []
    elif defect == "count":
        params.n = True
    elif defect == "best_of":
        params._real_n = 1
    else:
        params.logits_processors = [lambda x: x]
    with pytest.raises(sampling.SamplingEvidenceError):
        capture(case)
    assert engine.called == 0


@pytest.mark.parametrize(
    "defect", ["order", "missing", "unfinished", "duplicate", "indices", "abort"]
)
def test_bad_returned_evidence_is_preserved_before_validation_fails(case, monkeypatch, defect):
    root, _, _, _, _ = case

    def bad_call(engine, inputs, params):
        rows = outputs_for(inputs, params)
        if defect == "order":
            rows.reverse()
        elif defect == "missing":
            rows.pop()
        elif defect == "unfinished":
            rows[0].finished = False
        elif defect == "duplicate":
            rows[1].request_id = rows[0].request_id
        elif defect == "indices":
            rows[0].outputs[1].index = 0
        else:
            rows[0].outputs[0].finish_reason = "abort"
        return rows

    monkeypatch.setattr(sampling, "_call_engine", bad_call)
    with pytest.raises(sampling.SamplingEvidenceError):
        capture(case)
    failure = json.loads((root / "draws/capture-failure.json").read_text())
    assert failure["raw_durable"] and failure["returned_completions"] > 0
    assert (root / "draws/raw.jsonl").is_file() and not (root / "draws/capture.json").exists()


def test_capture_never_overwrites_prior_raw_and_request_changes_reject_parsing(case):
    root, _, _, engine, _ = case
    capture(case)
    raw = (root / "draws/raw.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        capture(case)
    assert engine.called == 1 and (root / "draws/raw.jsonl").read_bytes() == raw
    request = root / "draws/request.json"
    request.write_text(request.read_text() + "\n")
    with pytest.raises(sampling.SamplingEvidenceError):
        sampling.parse_recording(
            root / "draws", lambda *_: pytest.fail("must not parse changed input identity")
        )


def test_parser_interrupt_preserves_raw_and_partial_failure_record(case):
    root, *_ = case
    capture(case)
    original = (root / "draws/raw.jsonl").read_bytes()

    def interrupted(*_):
        raise KeyboardInterrupt("synthetic parser interruption")

    with pytest.raises(KeyboardInterrupt):
        sampling.parse_recording(root / "draws", interrupted)
    assert (root / "draws/raw.jsonl").read_bytes() == original
    failure = next((root / "draws").glob("parse-*.failure.json"))
    assert json.loads(failure.read_text())["status"] == "interrupted_or_failed"
    assert not list((root / "draws").glob("parse-*.summary.json"))


def test_new_recording_can_create_owned_parent_directories(case):
    root, card, prompts, engine, params = case
    report = sampling.record_vllm(
        engine,
        prompts,
        params,
        root / "new-parent/batch/recording",
        card_path=card,
        required_stop_tokens=["<stop>"],
    )
    assert report["status"] == "captured" and engine.called == 1


def test_incomplete_recording_parse_explains_why_it_is_not_certified(tmp_path):
    tmp_path.joinpath("capture-failure.json").write_text('{"status":"capture_failed"}')
    with pytest.raises(sampling.SamplingEvidenceError, match="no completed capture"):
        sampling.parse_recording(tmp_path, lambda *_: pytest.fail("no completed capture"))


def test_raw_fsync_failure_cannot_become_a_completed_capture(case, monkeypatch):
    root, *_ = case
    native_fsync = sampling.os.fsync
    calls = 0

    def failed_second(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("raw fsync failure")
        return native_fsync(fd)

    monkeypatch.setattr(sampling.os, "fsync", failed_second)
    with pytest.raises(OSError, match="raw fsync"):
        capture(case)
    assert not (root / "draws/capture.json").exists()
    assert json.loads((root / "draws/capture-failure.json").read_text())["raw_durable"] is False


def test_missing_lock_prevents_generation(case):
    root, path, _, engine, _ = case
    path.with_suffix(".lock.json").unlink()
    with pytest.raises(ValueError):
        capture(case)
    assert engine.called == 0 and not (root / "draws").exists()


def test_native_parameter_and_output_objects_without_inference(case, monkeypatch):
    pytest.importorskip("vllm")
    from vllm import SamplingParams
    from vllm.outputs import CompletionOutput, RequestOutput

    root, path, prompts, engine, _ = case
    params = SamplingParams(n=2, max_tokens=20, stop_token_ids=[9])

    def native_records(_engine, inputs, actual):
        assert type(actual) is SamplingParams and actual.stop_token_ids == [9]
        return [
            RequestOutput(
                str(i),
                None,
                item["prompt_token_ids"],
                None,
                [CompletionOutput(j, "4", [5], None, None, "stop", 9) for j in range(2)],
                True,
            )
            for i, item in enumerate(inputs)
        ]

    monkeypatch.setattr(sampling, "_call_engine", native_records)
    result = sampling.record_vllm(
        engine, prompts, params, root / "draws", card_path=path, required_stop_tokens=["<stop>"]
    )
    assert result["returned_completions"] == 4


def test_native_entrypoint_validation_without_model_construction(monkeypatch):
    pytest.importorskip("vllm")
    from vllm import LLM, SamplingParams

    llm = LLM.__new__(LLM)  # No constructor, weights, engine, or inference.
    params = SamplingParams(n=2, max_tokens=20, stop_token_ids=[9])
    identity = sampling._native_identity(llm, params)
    assert identity["vllm"] == "0.11.0"
    llm.get_tokenizer = lambda: None
    with pytest.raises(sampling.SamplingEvidenceError, match="entrypoint"):
        sampling._native_identity(llm, params)
    del llm.get_tokenizer
    monkeypatch.setitem(sampling.VLLM_SOURCES, "outputs.py", "0" * 64)
    with pytest.raises(sampling.SamplingEvidenceError, match="source mismatch"):
        sampling._native_identity(llm, params)


def test_offline_gemma_prompt_and_stop_tokens_without_model(case, monkeypatch):
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    assets = (
        paths.REPO_ROOT
        / "data/ptb/hf/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
    )
    if not (assets / "tokenizer.json").is_file():
        pytest.skip("local pinned Gemma tokenizer assets unavailable")
    tokenizer = AutoTokenizer.from_pretrained(
        assets, local_files_only=True, trust_remote_code=False
    )
    template = (
        paths.REPO_ROOT
        / "results/ptb/exp-protocol-gsm8k-gemma4b-high-r01-guard-x8-v1/g01r03/task/templates/gemma3.jinja"
    ).read_text()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Compute 2 plus 3."}],
        chat_template=template,
        tokenize=False,
        add_generation_prompt=True,
    )
    prepared = sampling.prepare_prompts([prompt], tokenizer, bos_policy="single_at_start")
    stops = sampling.resolve_stop_ids(tokenizer, ["<end_of_turn>"])
    assert prepared[0].token_ids[0] == tokenizer.bos_token_id
    assert prepared[0].token_ids[1] != tokenizer.bos_token_id
    assert stops == [tokenizer.convert_tokens_to_ids("<end_of_turn>")]
    root, path, _, engine, params = case
    monkeypatch.setattr(engine, "get_tokenizer", lambda: tokenizer)
    params.stop_token_ids = stops
    report = sampling.record_vllm(
        engine,
        prepared,
        params,
        root / "gemma-draws",
        card_path=path,
        required_stop_tokens=["<end_of_turn>"],
    )
    assert report["returned_requests"] == 1 and report["returned_completions"] == 2


def test_plain_enum_values_are_preserved_without_assuming_intenum():
    from enum import Enum

    class Kind(Enum):
        CUMULATIVE = 0

    result = sampling._plain(Kind.CUMULATIVE)
    assert result["value"] == 0 and result["enum"].endswith(".Kind")


def test_native_style_generic_token_sequences_are_not_rejected_as_nonlists(case, monkeypatch):
    from array import array

    def records(engine, inputs, params):
        rows = outputs_for(inputs, params)
        for row in rows:
            row.prompt_token_ids = array("I", row.prompt_token_ids)
            for completion in row.outputs:
                completion.token_ids = array("I", completion.token_ids)
        return rows

    monkeypatch.setattr(sampling, "_call_engine", records)
    assert capture(case)["returned_tokens"] == 4


def test_parser_nonfinite_return_does_not_leak_value_into_error_record(case):
    root, *_ = case
    capture(case)
    summary = sampling.parse_recording(root / "draws", lambda *_: {"score": float("nan")})
    assert summary["parser_errors"] == 4 and summary["parsed"] == 0
    rows = [json.loads(line) for line in Path(summary["parse_path"]).read_text().splitlines()]
    assert all(row["status"] == "parser_error" and "value" not in row for row in rows)


@pytest.mark.parametrize("filename", ["raw.jsonl", "request.json"])
def test_parser_cannot_certify_files_it_changed_even_with_buffered_reads(case, filename):
    root, *_ = case
    capture(case)
    changed = False

    def parser(text, metadata):
        nonlocal changed
        if not changed:
            path = root / "draws" / filename
            content = path.read_text()
            path.write_text(
                content.replace("1e999", "2e999") if filename == "raw.jsonl" else content + " "
            )
            changed = True
        return {"observed_text": text}

    with pytest.raises(sampling.SamplingEvidenceError, match="changed"):
        sampling.parse_recording(root / "draws", parser)
    assert not list((root / "draws").glob("parse-*.summary.json"))
    assert list((root / "draws").glob("parse-*.failure.json"))
