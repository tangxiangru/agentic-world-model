"""CPU behavioral tests; native cases use real fast tokenizers, never models."""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import pytest
from exp_protocol_cards import plan_card

from awm.exp_protocol import lock, preflight, schema
from awm.exp_protocol.rendered_training import (
    RenderedParts,
    RenderedSettings,
    RenderedTrainingBundle,
    RenderedTrainingError,
    UnsupportedRenderedTraining,
    check_card,
)
from awm.exp_protocol.token_bundle import digest, file_entry, json_bytes, strict_json

TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}{{ message.role }}: "
    "{{ message.content }}<STOP>\n{% endfor %}"
    "{% if add_generation_prompt %}assistant: {% endif %}"
)
TEMPLATE_BYTES = TEMPLATE.encode()


def shared_render(row, *, template, settings, rng):
    """A source-backed stand-in for the g05/g08 shared CPU renderer, not a model."""
    from jinja2 import Template

    body = row.get("completion_body", row.get("body", row.get("completion", "reason ANSWER: 2")))
    messages = row.get("messages", [{"role": "user", "content": row.get("q", "question")}])
    context = {"bos_token": settings["renderer"].get("bos", "<B>")}
    prefix = Template(template).render(messages=messages, add_generation_prompt=True, **context)
    if settings["mode"] == "joint_prefix":
        full_messages = messages + [{"role": "assistant", "content": body}]
        full = Template(template).render(
            messages=full_messages, add_generation_prompt=False, **context
        )
        return RenderedParts(
            prefix=prefix, full=full, messages=messages, full_messages=full_messages
        )
    target = row.get("target", body + settings["stop_token"] + settings["tail_text"])
    return RenderedParts(prefix=prefix, target=target, messages=messages)


def pre_rendered(row, *, template, settings, rng):
    # Intentionally consumes existing prompt bytes, not another template render.
    if settings["mode"] == "joint_prefix":
        return RenderedParts(prefix=row["prompt"], full=row["full"])
    return RenderedParts(prefix=row["prompt"], target=row["completion"])


def stale_renderer(row, *, template, settings, rng):
    # Deliberately simulates a module-global compiled old template.
    return shared_render(row, template=TEMPLATE, settings=settings, rng=rng)


def malformed_renderer(row, *, template, settings, rng):
    return {"passed": True, "n": 100}


def mutating_renderer(row, *, template, settings, rng):
    settings["renderer"]["mutated"] = True
    return RenderedParts(prefix="x", target="y")


def write_rows(path, rows):
    path.write_bytes(b"".join(json_bytes(row) for row in rows))


def token_rows(bundle):
    return [json.loads(line) for line in Path(bundle.data_entry["path"]).read_text().splitlines()]


def make_card(tmp_path, bundle, script):
    card = plan_card()
    card["setup"].update(
        data=[bundle.data_entry],
        rendered_training=bundle.declaration,
        command={"argv": ["python", str(script)], "script": str(script), "cwd": str(tmp_path)},
        output_dir=str(tmp_path / "checkpoints"),
    )
    settings = json.loads(bundle.receipt_path.read_text())["settings"]
    card["setup"]["method"].update(
        stop_token=settings["stop_token"],
        answer_marker=settings["answer_marker"],
        hyperparams={"max_seq_len": settings["max_seq_len"]},
    )
    path = tmp_path / "memory/cards/exp-01.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    schema.dump_card(path, card)
    return path, card


def lock_card(path, card):
    session = path.parents[2]
    assert schema.validate_plan(card, session).ok
    report = preflight.run_preflight(card, session, pitfalls=[])
    assert report["summary"]["fail"] == 0, report
    lock.write_lock(path, card, report["summary"])


@pytest.fixture
def native(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    specials = ["<PAD>", "<UNK>", "<B>", "<STOP>"]
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {word: index for index, word in enumerate(specials + alphabet)}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<UNK>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<PAD>",
        unk_token="<UNK>",
        bos_token="<B>",
        eos_token="<STOP>",
        model_max_length=4096,
    )
    script = tmp_path / "train.py"
    script.write_text("# This test never executes a model.\n")
    raw = tmp_path / "raw.jsonl"
    write_rows(raw, [{"q": "question", "body": "reason ANSWER: 2"}])
    settings = RenderedSettings(
        mode="separate_concat", max_seq_len=512, stop_token="<STOP>", answer_marker="ANSWER: "
    )

    def prepare(
        *,
        rows=None,
        output="bundle",
        render=shared_render,
        settings_=None,
        template=TEMPLATE_BYTES,
        tokenizer=None,
        reuse=False,
    ):
        if rows is not None:
            write_rows(raw, rows)
        return RenderedTrainingBundle.prepare(
            sources=[raw],
            render=render,
            tokenizer=tokenizer or tok,
            template_bytes=template,
            settings=settings_ or settings,
            source_files=[script, Path(__file__).absolute()],
            output=tmp_path / output,
            reuse=reuse,
        )

    return tok, raw, script, settings, prepare


@pytest.mark.parametrize("mode", ["separate_concat", "joint_prefix"])
def test_sampling_and_checked_training_share_actual_prompt_and_stop_boundary(native, tmp_path, mode):
    """E6 x E7 CPU integration: no engine, model, or inference is constructed."""
    from awm.exp_protocol.sampling import prepare_prompts, resolve_stop_ids

    tok, _, script, _, prepare = native
    row = {"q": "question", "body": "reason ANSWER: 2"}
    settings = RenderedSettings(
        mode=mode, max_seq_len=512, stop_token="<STOP>", answer_marker="ANSWER: ",
        tail_text="\n",
    )
    bundle = prepare(rows=[row], settings_=settings)
    path, card = make_card(tmp_path, bundle, script)
    lock_card(path, card)
    assert check_card(card, tmp_path)["verified_preparation"]
    opened = RenderedTrainingBundle.open_for_training(path)
    feature = opened.dataset[0]
    prefix = tok.apply_chat_template(
        [{"role": "user", "content": row["q"]}], chat_template=TEMPLATE,
        tokenize=False, add_generation_prompt=True,
    )
    request = prepare_prompts(
        [prefix], tok, item_ids=["independent-query"], bos_policy="single_at_start",
    )[0]
    boundary = feature["target_start"]
    assert list(request.token_ids) == feature["input_ids"][:boundary]
    assert feature["labels"][:boundary] == [-100] * boundary
    stops = resolve_stop_ids(tok, [settings.stop_token])
    assert stops == [tok.eos_token_id]
    # The raw answer has no stop marker. Actual rendering adds supervised STOP
    # plus the declared tail; prompt-side STOP is masked and not target evidence.
    assert settings.stop_token not in row["body"]
    assert stops[0] in feature["input_ids"][:boundary]
    assert feature["labels"][boundary:].count(stops[0]) == 1
    assert feature["labels"][boundary:] == feature["input_ids"][boundary:]
    opened.flush_consumption()


@pytest.mark.parametrize(
    "change",
    [
        {"mode": "packed"},
        {"length_policy": "truncate"},
        {"label_policy": "shifted"},
        {"add_special_tokens": True},
        {"prompt_mode": "guess"},
        {"max_seq_len": True},
        {"limit": -1},
        {"pad_to_multiple_of": 0},
        {"stop_min_fraction": float("nan")},
        {"marker_max_bad_fraction": 2},
        {"seed": False},
        {"unexpected": "field"},
    ],
)
def test_explicit_settings_reject_unknown_or_invalid_policies(change):
    values = {"mode": "separate_concat", "max_seq_len": 128, "stop_token": "END", **change}
    with pytest.raises(RenderedTrainingError):
        RenderedSettings.parse(values)


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}'])
def test_nonfinite_or_ambiguous_json_is_not_evidence(raw):
    with pytest.raises(ValueError):
        strict_json(raw)


@pytest.mark.parametrize(
    "policy",
    [
        {"stop_min_fraction": 0},
        {"stop_min_fraction": 0.94},
        {"marker_max_bad_fraction": 1},
        {"marker_max_bad_fraction": 0.03},
    ],
)
def test_settings_cannot_relax_frozen_semantic_thresholds(policy):
    with pytest.raises(RenderedTrainingError, match="cannot weaken"):
        RenderedSettings.parse(
            {"mode": "separate_concat", "max_seq_len": 128, "stop_token": "END", **policy}
        )


def test_optional_v2_field_and_aggregate_only_forgery(tmp_path):
    card = plan_card()
    assert schema.validate_plan(card).ok
    assert check_card(card, tmp_path)["status"] == "warn"
    for declaration in (True, {}, {"receipt": "x", "sha256": "wrong"}):
        card["setup"]["rendered_training"] = declaration
        assert not schema.validate_plan(card).ok
    forged = tmp_path / "summary.json"
    forged.write_text('{"schema_version":"awm-rendered-training-v1","passed":true,"n":100}')
    card["setup"]["rendered_training"] = {
        "receipt": str(forged),
        "sha256": file_entry(forged)["sha256"],
    }
    assert schema.validate_plan(card).ok
    assert check_card(card, tmp_path)["status"] == "fail"


def test_no_opt_in_preserves_raw_failure_and_unverified_rendering(tmp_path):
    raw = tmp_path / "raw.jsonl"
    write_rows(raw, [{"completion": "reason ANSWER: 2"}])
    card = plan_card()
    card["setup"]["data"] = [{"path": str(raw), "n_examples": 1, "source": "local"}]
    card["setup"]["method"]["stop_token"] = "<STOP>"
    report = preflight.run_preflight(card, tmp_path, pitfalls=[])
    states = {r["check"]: r["status"] for r in report["results"]}
    assert states["stop_token_consistent"] == "fail"
    assert states["rendered_training_evidence"] == "warn"


def test_native_preparation_lock_and_checked_consumer(native, tmp_path):
    _, _, script, _, prepare = native
    bundle = prepare()
    assert bundle.report["proof"] == "verified_preparation"
    assert bundle.report["model_consumption"] == "unknown"
    with pytest.raises(RenderedTrainingError, match="lock"):
        _ = bundle.dataset
    path, card = make_card(tmp_path, bundle, script)
    with pytest.raises(RenderedTrainingError, match="lock"):
        RenderedTrainingBundle.open_for_training(path)
    lock_card(path, card)
    consumer = RenderedTrainingBundle.open_for_training(path)
    assert consumer.flush_consumption()["proof"] == "verified_loader_binding"
    feature = consumer.dataset[0]
    assert consumer.flush_consumption()["dataset_access_observed"]
    batch = consumer.collator(return_tensors="python")([feature])
    assert batch["input_ids"][0] == token_rows(bundle)[0]["input_ids"]
    observed = consumer.flush_consumption()
    assert observed["proof"] == "observed_collator_consumption"
    assert observed["counts_at_last_flush"]["collator_rows"] == 1
    assert observed["model_consumption"] == "unknown"
    assert "plan_sha256" not in json.loads(bundle.receipt_path.read_text())


def test_native_valid_evidence_supersedes_raw_but_invalid_claim_never_does(native, tmp_path):
    _, _, script, _, prepare = native
    bundle = prepare()
    _, card = make_card(tmp_path, bundle, script)
    report = preflight.run_preflight(card, tmp_path, pitfalls=[])
    for row in report["results"]:
        if row["check"] in (
            "stop_token_consistent",
            "answer_marker_single",
            "max_seq_len_headroom",
        ):
            assert row["status"] == "skip" and "superseded" in row["detail"]
    assert report["rendered_training"]["verified_preparation"]
    card["setup"]["rendered_training"]["sha256"] = "0" * 64
    report = preflight.run_preflight(card, tmp_path, pitfalls=[])
    assert report["rendered_training"]["status"] == "fail"
    assert not any("superseded" in r["detail"] for r in report["results"])


@pytest.mark.parametrize("representation", ["g05", "g08"])
def test_native_old_and_new_raw_representations_have_identical_supervised_tokens(
    native, representation
):
    _, _, _, default, prepare = native
    values = asdict(default)
    if representation == "g05":
        values.update(mode="joint_prefix", tail_text="\n")
        old = prepare(rows=[{"completion": "reason ANSWER: 2"}], settings_=values, output="old")
        # Record the original arrays before deliberately replacing the raw source.
        before = token_rows(old)[0]
        new = prepare(
            rows=[{"completion": "reason ANSWER: 2<STOP>", "completion_body": "reason ANSWER: 2"}],
            settings_=values,
            output="new",
        )
    else:
        old = prepare(rows=[{"body": "reason ANSWER: 2"}], output="old")
        before = token_rows(old)[0]
        new = prepare(rows=[{"target": "reason ANSWER: 2<STOP>"}], output="new")
    after = token_rows(new)[0]
    assert before["input_ids"] == after["input_ids"] and before["labels"] == after["labels"]
    assert new.report["findings"]["stop_fraction"] == 1


def test_native_g03_pre_rendered_inputs_need_no_new_messages_or_double_wrapper(native):
    tok, _, _, settings, prepare = native
    prefix = tok.apply_chat_template(
        [{"role": "user", "content": "already rendered"}],
        chat_template=TEMPLATE,
        tokenize=False,
        add_generation_prompt=True,
    )
    values = {**asdict(settings), "prompt_mode": "pre_rendered"}
    bundle = prepare(
        rows=[{"prompt": prefix, "completion": "reason ANSWER: 2<STOP>"}],
        render=pre_rendered,
        settings_=values,
    )
    row = token_rows(bundle)[0]
    prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    assert row["input_ids"][: row["target_start"]] == prefix_ids
    assert row["input_ids"].count(tok.bos_token_id) == 1
    assert "unverified" in bundle.report["template_coverage"]
    assert row["rendered_sha256"]["prefix"] == digest(prefix.encode())


def test_native_template_snapshot_replay_detects_cached_old_template(native):
    _, _, _, _, prepare = native
    with pytest.raises(RenderedTrainingError, match="template snapshot"):
        prepare(
            render=stale_renderer,
            template=TEMPLATE.replace("assistant:", "assistant response:").encode(),
        )


def test_native_template_tail_and_prompt_markers_are_not_target_failures(native):
    tok, _, _, settings, prepare = native
    values = {**asdict(settings), "mode": "joint_prefix", "tail_text": "\n"}
    bundle = prepare(
        rows=[{"q": "ANSWER: in prompt. ANSWER: in demo.", "body": "reason ANSWER: 2"}],
        settings_=values,
    )
    row = token_rows(bundle)[0]
    assert row["input_ids"][-1] != tok.eos_token_id
    assert row["input_ids"][-2] == tok.eos_token_id
    assert row["labels"][-2] == tok.eos_token_id
    assert bundle.report["findings"]["marker_bad"] == 0


@pytest.mark.parametrize(
    "target",
    [
        "reason ANSWER: 2",
        "reason ANSWER: 2<STOP><STOP>",
        "reason only<STOP>",
        "ANSWER: 1 ANSWER: 2<STOP>",
    ],
)
def test_native_masked_prompt_stop_or_bad_target_is_not_good_supervision(native, target):
    _, _, _, _, prepare = native
    with pytest.raises(RenderedTrainingError):
        prepare(rows=[{"target": target}])


def test_native_semantic_tolerances_are_explicit_and_report_rates(native):
    _, _, _, _, prepare = native
    rows = [{"target": "reason ANSWER: 2<STOP>"} for _ in range(19)] + [
        {"target": "reason ANSWER: 2"}
    ]
    bundle = prepare(rows=rows)
    assert bundle.report["findings"]["stop_fraction"] == 0.95
    with pytest.raises(RenderedTrainingError, match="stop consistency"):
        prepare(
            rows=rows,
            output="stricter",
            settings_={
                **json.loads(bundle.receipt_path.read_text())["settings"],
                "stop_min_fraction": 1.0,
            },
        )


def test_native_whole_source_drop_limit_denominators_and_padding(native, tmp_path):
    _, _, script, settings, prepare = native
    values = {**asdict(settings), "max_seq_len": 65, "pad_to_multiple_of": 8, "limit": 3}
    bundle = prepare(
        rows=[
            {"body": "ANSWER: 2"},
            {"q": "long " * 100},
            {"q": "other", "body": "reason ANSWER: 2"},
            {"body": "not considered"},
        ],
        settings_=values,
    )
    counts = bundle.report["findings"]["counts"]
    assert counts == {
        "source_rows": 4,
        "considered_rows": 3,
        "excluded_by_limit": 1,
        "kept_rows": 2,
        "dropped_overlength": 1,
        "dropped_prefix_drift": 0,
    }
    path, card = make_card(tmp_path, bundle, script)
    lock_card(path, card)
    consumer = RenderedTrainingBundle.open_for_training(path)
    batch = consumer.collator(pad_to_multiple_of=8, return_tensors="python")(
        [consumer.dataset[0], consumer.dataset[1]]
    )
    assert len(batch["input_ids"][0]) % 8 == 0
    for labels, attention in zip(batch["labels"], batch["attention_mask"]):
        assert all(label == -100 for label, mask in zip(labels, attention) if not mask)
        assert sum(attention) <= 65
    with pytest.raises(RenderedTrainingError, match="padding differs"):
        consumer.collator(pad_to_multiple_of=16)


def test_native_template_added_tokens_can_exceed_limit_when_raw_estimate_passes(native, tmp_path):
    _, raw, _, settings, prepare = native
    bundle = prepare(
        rows=[{"q": "", "body": "ANSWER: 2"}, {"q": "five?", "body": "ANSWER: 2"}],
        settings_={**asdict(settings), "max_seq_len": 32},
    )
    raw_card = plan_card()
    raw_card["setup"]["data"] = [{"path": str(raw), "n_examples": 2, "source": "local"}]
    raw_card["setup"]["method"]["hyperparams"]["max_seq_len"] = 32
    assert (
        preflight.max_seq_len_headroom(preflight.Context(raw_card, tmp_path, {})).status == "pass"
    )
    counts = bundle.report["findings"]["counts"]
    assert counts["kept_rows"] == 1 and counts["dropped_overlength"] == 1


def test_native_padding_width_may_exceed_limit_without_truncating_retained_tokens(native, tmp_path):
    _, _, script, settings, prepare = native
    observed = prepare(output="observed")
    length = len(token_rows(observed)[0]["input_ids"])
    assert length % 8
    bundle = prepare(settings_={**asdict(settings), "max_seq_len": length, "pad_to_multiple_of": 8})
    path, card = make_card(tmp_path, bundle, script)
    lock_card(path, card)
    consumer = RenderedTrainingBundle.open_for_training(path)
    batch = consumer.collator(return_tensors="python")([consumer.dataset[0]])
    assert len(batch["input_ids"][0]) > length
    assert sum(batch["attention_mask"][0]) == length
    assert all(label == -100 for label in batch["labels"][0][length:])


def test_native_multiple_token_stop_and_non_106_id(native):
    tok, _, _, settings, prepare = native
    values = {**asdict(settings), "stop_token": "END TURN"}
    bundle = prepare(rows=[{"target": "reason ANSWER: 2END TURN"}], settings_=values)
    assert len(tok("END TURN", add_special_tokens=False)["input_ids"]) > 1
    assert tok.eos_token_id != 106 and bundle.report["findings"]["stop_ok"] == 1


def test_native_prefix_drift_is_a_measured_drop_not_a_mode_substitution(native, tmp_path):
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    _, raw, script, _, _ = native
    backend = Tokenizer(
        models.BPE(
            vocab={"<PAD>": 0, "<UNK>": 1, "a": 2, "b": 3, "ab": 4, "<STOP>": 5},
            merges=[("a", "b")],
            unk_token="<UNK>",
        )
    )
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<PAD>", unk_token="<UNK>", eos_token="<STOP>"
    )
    write_rows(raw, [{"prompt": "a", "full": "ab<STOP>"}, {"prompt": "a", "full": "a<STOP>"}])
    settings = RenderedSettings(
        mode="joint_prefix", prompt_mode="pre_rendered", max_seq_len=16, stop_token="<STOP>"
    )
    bundle = RenderedTrainingBundle.prepare(
        sources=[raw],
        render=pre_rendered,
        tokenizer=tok,
        template_bytes=b"reference only",
        settings=settings,
        source_files=[script, Path(__file__).absolute()],
        output=tmp_path / "joint",
    )
    assert bundle.report["findings"]["counts"]["dropped_prefix_drift"] == 1
    assert bundle.report["findings"]["counts"]["kept_rows"] == 1
    write_rows(raw, [{"prompt": "a", "completion": "b<STOP>"}])
    separate = RenderedTrainingBundle.prepare(
        sources=[raw],
        render=pre_rendered,
        tokenizer=tok,
        template_bytes=b"reference only",
        settings={**asdict(settings), "mode": "separate_concat"},
        source_files=[script, Path(__file__).absolute()],
        output=tmp_path / "separate",
    )
    assert token_rows(separate)[0]["input_ids"] == [2, 3, 5]


def rewrite_token_artifact(bundle, mutate):
    rows = token_rows(bundle)
    mutate(rows)
    token = Path(bundle.data_entry["path"])
    write_rows(token, rows)
    receipt = strict_json(bundle.receipt_path.read_bytes())
    receipt["tokens"]["file"] = file_entry(token)
    bundle.receipt_path.write_bytes(json_bytes(receipt))


@pytest.mark.parametrize(
    "mutation",
    ["masked_stop", "all_ignored", "shifted", "wrong_length", "bool_id", "unaccounted_row"],
)
def test_native_all_row_revalidation_rejects_resealed_bad_arrays(native, mutation):
    _, _, _, _, prepare = native
    bundle = prepare(rows=[{"body": "reason ANSWER: 2"} for _ in range(503)])

    def mutate(rows):
        row = rows[-1]  # Beyond the old raw first-500 heuristic.
        if mutation == "masked_stop":
            row["labels"][-1] = -100
        elif mutation == "all_ignored":
            row["labels"] = [-100] * len(row["labels"])
        elif mutation == "shifted":
            row["labels"] = row["labels"][1:] + [-100]
        elif mutation == "wrong_length":
            row["labels"].pop()
        elif mutation == "bool_id":
            row["input_ids"][0] = True
        else:
            rows.append(copy.deepcopy(row))

    rewrite_token_artifact(bundle, mutate)
    with pytest.raises(RenderedTrainingError):
        RenderedTrainingBundle.verify(bundle.receipt_path)


@pytest.mark.parametrize(
    "change", ["raw", "script", "tokenizer_asset", "template", "settings", "snapshot"]
)
def test_native_stale_bindings_fail_instead_of_becoming_an_advisory_pass(native, tmp_path, change):
    tok, raw, script, _, prepare = native
    assets = tmp_path / "tokenizer-assets"
    tok.save_pretrained(assets)
    tok.name_or_path = str(assets)
    bundle = prepare()
    _, card = make_card(tmp_path, bundle, script)
    if change == "raw":
        raw.write_bytes(raw.read_bytes() + b'{"q":"new"}\n')
    elif change == "script":
        script.write_text("# changed pipeline\n")
    elif change == "tokenizer_asset":
        (assets / "tokenizer_config.json").write_text("{}")
    elif change == "template":
        (bundle.receipt_path.parent / "template.jinja").write_text("different")
    elif change == "snapshot":
        (bundle.receipt_path.parent / "tokenizer-snapshot.json").write_text("{}")
    else:
        card["setup"]["method"]["hyperparams"]["max_seq_len"] += 1
    assert check_card(card, tmp_path)["status"] == "fail"


def test_native_unchanged_cache_reused_without_second_token_artifact(native):
    _, _, _, settings, prepare = native
    first = prepare()
    token = Path(first.data_entry["path"])
    identity = (file_entry(token), token.stat().st_ino)
    reused = prepare(reuse=True)
    assert reused.declaration == first.declaration
    assert (file_entry(token), token.stat().st_ino) == identity
    with pytest.raises(RenderedTrainingError, match="cached bundle"):
        prepare(reuse=True, settings_={**asdict(settings), "seed": 2})


def test_native_effective_added_tokens_and_active_backend_settings_are_bound(native):
    tok, _, _, _, prepare = native
    tok.add_tokens(["special-added-term"])
    tok.backend_tokenizer.enable_truncation(max_length=2)
    tok.backend_tokenizer.enable_padding(length=2, pad_id=tok.pad_token_id, pad_token=tok.pad_token)
    before = tok.backend_tokenizer.to_str()
    bundle = prepare()
    assert tok.backend_tokenizer.to_str() == before
    assert len(token_rows(bundle)[0]["input_ids"]) > 2  # No hidden backend truncation.
    tok.add_tokens(["another-term"])
    with pytest.raises(RenderedTrainingError, match="cached bundle"):
        prepare(reuse=True)


def test_native_malformed_source_and_renderer_declarations_never_make_receipt_pass(native):
    _, raw, _, _, prepare = native
    raw.write_bytes(b'{"q":"fine"}\nnot json\n')
    with pytest.raises(RenderedTrainingError, match="malformed source row"):
        prepare()
    with pytest.raises(UnsupportedRenderedTraining, match="RenderedParts"):
        prepare(rows=[{"q": "fine"}], render=malformed_renderer, output="bad-render")
    with pytest.raises(RenderedTrainingError, match="mutated"):
        prepare(render=mutating_renderer, output="mutating")


def test_native_loader_collator_mutations_and_stale_lock_are_rejected(native, tmp_path):
    _, _, script, _, prepare = native
    bundle = prepare()
    path, card = make_card(tmp_path, bundle, script)
    lock_card(path, card)
    consumer = RenderedTrainingBundle.open_for_training(path)
    feature = consumer.dataset[0]
    feature["labels"][-1] = -100
    with pytest.raises(RenderedTrainingError, match="feature/source"):
        consumer.collator(return_tensors="python")([feature])
    intact = consumer.dataset[0]
    stripped = {k: v for k, v in intact.items() if not k.startswith("_awm")}
    with pytest.raises(RenderedTrainingError, match="feature/source"):
        consumer.collator(return_tensors="python")([stripped])
    card["hypothesis"]["claim"] = "changed after lock"
    schema.dump_card(path, card)
    with pytest.raises(RenderedTrainingError, match="lock"):
        RenderedTrainingBundle.open_for_training(path)
    with pytest.raises(RenderedTrainingError, match="changed"):
        consumer.flush_consumption()


def test_native_later_binding_changes_are_checked_at_explicit_flush_not_every_batch(
    native, tmp_path
):
    _, _, script, _, prepare = native
    bundle = prepare()
    path, card = make_card(tmp_path, bundle, script)
    lock_card(path, card)
    consumer = RenderedTrainingBundle.open_for_training(path)
    collate = consumer.collator(return_tensors="python")
    collate([consumer.dataset[0]])  # Completes the initial per-kind auto-flushes.
    card["hypothesis"]["claim"] = "changed after the first checked batch"
    schema.dump_card(path, card)
    assert collate([consumer.dataset[0]])["labels"]  # Arrays still match their prepared hashes.
    with pytest.raises(RenderedTrainingError, match="changed"):
        consumer.flush_consumption()


def test_native_verifier_does_not_execute_renderer_or_replay_template(
    native, tmp_path, monkeypatch
):
    import awm.exp_protocol.rendered_training as module

    _, _, script, _, prepare = native
    bundle = prepare()
    _, card = make_card(tmp_path, bundle, script)

    def forbidden(*args, **kwargs):
        pytest.fail("preflight invoked preparation renderer/template replay")

    monkeypatch.setattr(module, "_parts", forbidden)
    report = check_card(card, tmp_path)
    assert report["status"] == "pass"
    assert report["template_validation_phase"] == "preparation observation; not rerun by preflight"


@pytest.mark.parametrize(
    "mutation",
    [
        "minimal",
        "schema",
        "timestamp",
        "empty_timestamp",
        "script_missing",
        "script_hash",
        "script_path",
    ],
)
def test_native_matching_plan_and_token_entry_are_not_a_successful_lock(native, tmp_path, mutation):
    _, _, script, _, prepare = native
    bundle = prepare()
    path, card = make_card(tmp_path, bundle, script)
    lock_card(path, card)
    info = lock.read_lock(path)
    if mutation == "minimal":
        info = {k: info[k] for k in ("plan_sha256", "data", "card_id")}
    elif mutation == "schema":
        info["schema_version"] = "unknown-lock"
    elif mutation == "timestamp":
        del info["locked_at"]
    elif mutation == "empty_timestamp":
        info["locked_at"] = " "
    elif mutation == "script_missing":
        info["script"] = None
    elif mutation == "script_hash":
        info["script"]["sha256"] = "0" * 64
    else:
        info["script"]["path"] = str(tmp_path / "different.py")
    lock.lock_path(path).write_bytes(json_bytes(info))
    with pytest.raises(RenderedTrainingError, match="lock|script"):
        RenderedTrainingBundle.open_for_training(path)


@pytest.mark.parametrize("location", ["wrong-directory", "wrong-filename", "symlink"])
def test_native_consumer_requires_canonical_matching_card_path(native, tmp_path, location):
    _, _, script, _, prepare = native
    bundle = prepare()
    path, card = make_card(tmp_path, bundle, script)
    lock_card(path, card)
    if location == "wrong-directory":
        alternate = tmp_path / "elsewhere/exp-01.yaml"
        alternate.parent.mkdir()
        alternate.write_bytes(path.read_bytes())
    elif location == "wrong-filename":
        alternate = path.with_name("exp-02.yaml")
        alternate.write_bytes(path.read_bytes())
    else:
        alternate = tmp_path / "alias.yaml"
        alternate.symlink_to(path)
    with pytest.raises(RenderedTrainingError, match="canonical|filename"):
        RenderedTrainingBundle.open_for_training(alternate)


def test_native_relative_receipt_and_actual_data_binding(native, tmp_path):
    _, raw, script, _, prepare = native
    bundle = prepare()
    _, card = make_card(tmp_path, bundle, script)
    card["setup"]["rendered_training"]["receipt"] = str(bundle.receipt_path.relative_to(tmp_path))
    assert check_card(card, tmp_path)["status"] == "pass"
    card["setup"]["data"][0]["path"] = str(raw)
    assert check_card(card, tmp_path)["status"] == "fail"


def test_native_torch_collation_is_cpu_only(native, tmp_path):
    torch = pytest.importorskip("torch")
    _, _, script, settings, prepare = native
    bundle = prepare(settings_={**asdict(settings), "pad_to_multiple_of": 8})
    path, card = make_card(tmp_path, bundle, script)
    lock_card(path, card)
    consumer = RenderedTrainingBundle.open_for_training(path)
    batch = consumer.collator(pad_to_multiple_of=8)([consumer.dataset[0]])
    assert all(
        tensor.device.type == "cpu" and tensor.dtype == torch.long for tensor in batch.values()
    )


@pytest.mark.parametrize("context", ["fork", "spawn"])
def test_native_cpu_dataloader_workers_consume_same_artifact(native, tmp_path, context):
    torch = pytest.importorskip("torch")
    _, _, script, _, prepare = native
    bundle = prepare(rows=[{"body": "reason ANSWER: 2"} for _ in range(7)])
    path, card = make_card(tmp_path, bundle, script)
    lock_card(path, card)
    consumer = RenderedTrainingBundle.open_for_training(path)
    loader = torch.utils.data.DataLoader(
        consumer.dataset,
        batch_size=2,
        num_workers=2,
        multiprocessing_context=context,
        collate_fn=consumer.collator(),
    )
    batches = list(loader)
    assert sum(batch["input_ids"].shape[0] for batch in batches) == 7
    records = [
        strict_json(p.read_bytes())
        for p in (path.parents[1] / "rendered-consumers/exp-01").glob("*.json")
    ]
    assert any(r["collation_observed"] and r["pid"] != os.getpid() for r in records)
    assert all(r["model_consumption"] == "unknown" for r in records)


def test_native_bound_array_mutation_after_loader_is_detected(native, tmp_path):
    _, _, script, _, prepare = native
    bundle = prepare()
    path, card = make_card(tmp_path, bundle, script)
    lock_card(path, card)
    consumer = RenderedTrainingBundle.open_for_training(path)
    rows = token_rows(bundle)
    rows[0]["labels"][-1] = -100
    write_rows(Path(bundle.data_entry["path"]), rows)
    with pytest.raises(RenderedTrainingError, match="changed after"):
        consumer.dataset[0]


def test_native_drop_ledger_cannot_invent_a_length_or_discard_source_rows(native):
    _, _, _, settings, prepare = native
    bundle = prepare(
        rows=[{"body": "reason ANSWER: 2"}, {"body": "long " * 100}],
        settings_={**asdict(settings), "max_seq_len": 100},
    )
    receipt = strict_json(bundle.receipt_path.read_bytes())
    decision_path = Path(receipt["decisions"]["file"]["path"])
    decisions = [strict_json(line) for line in decision_path.read_bytes().splitlines()]
    decisions[-1]["input_ids"] = decisions[-1]["prefix_ids"] + [3]
    write_rows(decision_path, decisions)
    receipt["decisions"]["file"] = file_entry(decision_path)
    bundle.receipt_path.write_bytes(json_bytes(receipt))
    with pytest.raises(RenderedTrainingError, match="overlength"):
        RenderedTrainingBundle.verify(bundle.receipt_path)


def test_native_template_tail_is_excluded_from_marker_count_even_with_bad_stop(native):
    _, _, _, settings, prepare = native
    values = {**asdict(settings), "tail_text": " footer ANSWER: "}
    rows = [{"target": "reason ANSWER: 2<STOP> footer ANSWER: "} for _ in range(48)]
    # 96% stops satisfies the frozen minimum, but these two rows have no answer
    # marker: the apparent marker is in the template tail, not the answer.
    rows += [{"target": "no answer or stop footer ANSWER: "} for _ in range(2)]
    with pytest.raises(RenderedTrainingError, match="answer-marker"):
        prepare(rows=rows, settings_=values)


def test_native_zero_stop_all_bad_marker_cannot_obtain_superseding_pass(native, tmp_path):
    _, _, script, settings, prepare = native
    relaxed = {**asdict(settings), "stop_min_fraction": 0, "marker_max_bad_fraction": 1}
    with pytest.raises(RenderedTrainingError, match="cannot weaken"):
        prepare(rows=[{"target": "no terminal and no answer marker"}], settings_=relaxed)
    bundle = prepare(rows=[{"body": "reason ANSWER: 2"}])
    receipt = strict_json(bundle.receipt_path.read_bytes())
    receipt["settings"].update(stop_min_fraction=0, marker_max_bad_fraction=1)
    bundle.receipt_path.write_bytes(json_bytes(receipt))
    _, card = make_card(tmp_path, bundle, script)
    card["setup"]["rendered_training"]["sha256"] = file_entry(bundle.receipt_path)["sha256"]
    report = preflight.run_preflight(card, tmp_path, pitfalls=[])
    assert report["rendered_training"]["status"] == "fail"
    assert "cannot weaken" in report["rendered_training"]["detail"]
    assert not any("superseded" in row["detail"] for row in report["results"])


def test_native_offline_gemma_tokenizer_only_replay(tmp_path):
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    original = Path(
        "/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator"
    )
    assets = (
        original
        / "data/ptb/hf/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
    )
    if not (assets / "tokenizer.json").is_file():
        pytest.skip("optional existing local Gemma tokenizer assets unavailable")
    tok = AutoTokenizer.from_pretrained(
        assets, local_files_only=True, trust_remote_code=False, token=False
    )
    template = (
        original
        / "results/ptb/exp-protocol-gsm8k-gemma4b-high-r01-guard-x8-v1/g01r05/task/templates/gemma3.jinja"
    ).read_bytes()
    raw = tmp_path / "raw.jsonl"
    write_rows(raw, [{"q": "What is two plus two?", "body": "Two plus two is four. ANSWER: 4"}])
    script = tmp_path / "train.py"
    script.write_text("# tokenizer-only synthetic fixture; never executed\n")
    settings = RenderedSettings(
        mode="joint_prefix",
        max_seq_len=256,
        stop_token="<end_of_turn>",
        answer_marker="ANSWER: ",
        tail_text="\n",
        renderer={"bos": tok.bos_token},
    )
    bundle = RenderedTrainingBundle.prepare(
        sources=[raw],
        render=shared_render,
        tokenizer=tok,
        template_bytes=template,
        settings=settings,
        source_files=[script, Path(__file__).absolute()],
        output=tmp_path / "gemma-bundle",
    )
    record = token_rows(bundle)[0]
    assert record["labels"][-2] == tok.convert_tokens_to_ids("<end_of_turn>")
    assert bundle.report["findings"]["stop_fraction"] == 1
    assert bundle.report["template_coverage"] == "prompt_and_full_replayed"


def test_native_representative_cpu_cost(native, tmp_path, capsys):
    _, _, script, settings, prepare = native
    n = int(os.environ.get("AWM_RENDERED_BENCHMARK_ROWS", "200"))
    rows = [
        {
            "q": f"ordinary synthetic training question {i}",
            "body": "reason " * (30 + i % 20) + f"ANSWER: {i % 1000}",
        }
        for i in range(n)
    ]
    start = time.perf_counter()
    bundle = prepare(rows=rows, settings_={**asdict(settings), "max_seq_len": 512})
    prepared = time.perf_counter()
    path, card = make_card(tmp_path, bundle, script)
    assert check_card(card, tmp_path)["status"] == "pass"
    checked = time.perf_counter()
    lock.write_lock(path, card, {"pass": 1, "warn": 0, "fail": 0, "skip": 0})
    consumer = RenderedTrainingBundle.open_for_training(path)
    loaded = time.perf_counter()
    batch = consumer.collator(return_tensors="python")(
        [consumer.dataset[i] for i in range(min(n, 32))]
    )
    assert len(batch["input_ids"]) == min(n, 32)
    stats = {
        "rows": n,
        "prepared_tokens": bundle.report["findings"]["post_filter_lengths"]["sum"],
        "prepare_including_verification_s": prepared - start,
        "preflight_s": checked - prepared,
        "loader_s": loaded - checked,
        "first_batch_s": time.perf_counter() - loaded,
        "token_file_bytes": Path(bundle.data_entry["path"]).stat().st_size,
    }
    with capsys.disabled():
        print("RENDERED_CPU_COST " + json.dumps(stats, sort_keys=True))
