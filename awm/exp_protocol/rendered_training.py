"""One prepared token artifact and a checked, opt-in completion-only consumer.

Preparation is CPU-only; importing this module does not load ML libraries. A
preparation report is not proof of model execution or arbitrary script behavior.
"""

from __future__ import annotations

import copy
import hashlib
import os
import random
import threading
import time
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from .token_bundle import (
    SCHEMA,
    RenderedParts,
    RenderedSettings,
    RenderedTrainingError,
    UnsupportedRenderedTraining,
    decoder_from_snapshot,
    digest,
    file_entry,
    findings,
    implementation_identity,
    json_bytes,
    renderer_identity,
    strict_json,
    token_sequences,
    tokenizer_snapshot,
    validate_row,
    verify_file,
)

_COUNTS = (
    "source_rows",
    "considered_rows",
    "excluded_by_limit",
    "kept_rows",
    "dropped_overlength",
    "dropped_prefix_drift",
)


def _write(path, value):
    from .save_contract import _atomic_bytes

    _atomic_bytes(Path(path), json_bytes(value))


def _line(stream, description):
    raw = stream.readline()
    if not raw:
        raise RenderedTrainingError(f"truncated {description}")
    value = strict_json(raw)
    if not isinstance(value, dict):
        raise RenderedTrainingError(f"non-object {description}")
    return value, raw


def _raw_rows(path):
    with Path(path).open("rb") as stream:
        for ordinal, raw in enumerate(stream, 1):
            try:
                value = strict_json(raw)
                if not isinstance(value, dict):
                    raise RenderedTrainingError("row must be a JSON object")
            except (ValueError, UnicodeError) as exc:
                raise RenderedTrainingError(
                    f"malformed source row {path}:{ordinal}: {exc}"
                ) from exc
            yield ordinal, raw, value


def _parts(render, row, template, settings, rng, tokenizer):
    arguments = asdict(settings)
    before = json_bytes(arguments)
    parts = render(row, template=template, settings=arguments, rng=rng)
    if not isinstance(parts, RenderedParts) or not isinstance(parts.prefix, str):
        raise UnsupportedRenderedTraining(
            "renderer must return RenderedParts, not a summary/filter decision"
        )
    if json_bytes(arguments) != before:
        raise RenderedTrainingError("renderer mutated its bound settings")
    text = parts.target if settings.mode == "separate_concat" else parts.full
    if not isinstance(text, str):
        raise RenderedTrainingError("renderer omitted the target/full text required by the mode")
    if settings.prompt_mode == "template_replay":
        if not isinstance(parts.messages, list) or not parts.messages:
            raise RenderedTrainingError("template_replay requires actual prompt messages")
        reference = tokenizer.apply_chat_template(
            parts.messages, chat_template=template, tokenize=False, add_generation_prompt=True
        )
        if parts.prefix != reference:
            raise RenderedTrainingError(
                "rendered prefix differs from the explicit template snapshot"
            )
        if settings.mode == "joint_prefix":
            if not isinstance(parts.full_messages, list) or not parts.full_messages:
                raise RenderedTrainingError(
                    "joint template replay requires full conversation messages"
                )
            full_reference = tokenizer.apply_chat_template(
                parts.full_messages,
                chat_template=template,
                tokenize=False,
                add_generation_prompt=False,
            )
            if parts.full != full_reference:
                raise RenderedTrainingError(
                    "rendered full text differs from the explicit template snapshot"
                )
    return parts.prefix, text


def _new_counts():
    return {name: 0 for name in _COUNTS}


def _rejected_update(hasher, source, row, reason):
    hasher.update(json_bytes({"source": source, "row": row, "reason": reason}))


def _read_and_verify(receipt_path, expected_sha=None):
    """Recompute all token findings, counts and source locators; never run renderer code."""
    receipt_path = Path(receipt_path).absolute()
    raw_receipt = receipt_path.read_bytes()
    if expected_sha is not None and digest(raw_receipt) != expected_sha:
        raise RenderedTrainingError("rendered receipt hash differs from its declared identity")
    receipt = strict_json(raw_receipt)
    required = {
        "schema_version",
        "implementation",
        "producer",
        "settings",
        "raw_sources",
        "tokenizer",
        "template",
        "tokens",
        "decisions",
        "findings",
        "rejected_locator_sha256",
        "template_coverage",
        "prepared_at",
        "prepare_seconds",
    }
    if (
        not isinstance(receipt, dict)
        or set(receipt) != required
        or receipt["schema_version"] != SCHEMA
    ):
        raise RenderedTrainingError(
            "a complete supported token artifact is required, not a summary declaration"
        )
    if receipt["implementation"] != implementation_identity():
        raise UnsupportedRenderedTraining("rendered adapter implementation changed")
    settings = RenderedSettings.parse(receipt["settings"])
    if asdict(settings) != receipt["settings"]:
        raise RenderedTrainingError("receipt settings are not complete/canonical")
    for entry in receipt["producer"]["files"]:
        verify_file(entry)
    if not receipt["producer"]["files"]:
        raise RenderedTrainingError("missing producer source identities")
    snapshot_path = verify_file(receipt["tokenizer"])
    snapshot = strict_json(snapshot_path.read_bytes())
    backend = decoder_from_snapshot(snapshot)
    template_path = verify_file(receipt["template"])
    template_path.read_bytes().decode("utf-8")
    expected_coverage = (
        "prompt_and_full_replayed"
        if settings.mode == "joint_prefix"
        else "prompt_replayed_target_not_template_equivalence"
    )
    if settings.prompt_mode == "pre_rendered":
        expected_coverage = "not_applied_reference_only_equivalence_unverified"
    if receipt["template_coverage"] != expected_coverage:
        raise RenderedTrainingError("template application coverage does not match the actual mode")
    token_path = verify_file(receipt["tokens"]["file"])
    decision_path = verify_file(receipt["decisions"]["file"])
    valid_ids = set(backend.get_vocab(with_added_tokens=True).values())
    stop, tail = token_sequences(backend, settings, snapshot)
    counts, row_findings, pre_lengths = _new_counts(), [], []
    offsets, row_hashes = [], []
    rejected = hashlib.sha256()
    raw_sources = receipt["raw_sources"]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise RenderedTrainingError("missing raw-source identities")
    if len({s["file"]["path"] for s in raw_sources}) != len(raw_sources):
        raise RenderedTrainingError("duplicate raw-source paths")
    with token_path.open("rb") as tokens, decision_path.open("rb") as decisions:
        for source_index, source in enumerate(raw_sources):
            source_path = verify_file(source["file"])
            source_rows = 0
            for ordinal, raw, _ in _raw_rows(source_path):
                source_rows += 1
                counts["source_rows"] += 1
                decision, _ = _line(decisions, "source decision ledger")
                locator = {"source": source_index, "row": ordinal, "raw_sha256": digest(raw)}
                if any(decision.get(k) != v for k, v in locator.items()):
                    raise RenderedTrainingError(
                        "source decision does not match actual source occurrence"
                    )
                excluded = settings.limit is not None and counts["source_rows"] > settings.limit
                if excluded:
                    if decision != {**locator, "decision": "excluded_limit"}:
                        raise RenderedTrainingError("declared limit accounting mismatch")
                    counts["excluded_by_limit"] += 1
                    _rejected_update(rejected, source_index, ordinal, "excluded_limit")
                    continue
                counts["considered_rows"] += 1
                kind = decision.get("decision")
                if kind == "kept":
                    offset = tokens.tell()
                    record, _ = _line(tokens, "training token artifact")
                    observed = validate_row(record, settings, backend, valid_ids, stop, tail)
                    if (record["source"], record["row"]) != (source_index, ordinal):
                        raise RenderedTrainingError(
                            "token record/source decision alignment mismatch"
                        )
                    if decision != {
                        **locator,
                        "decision": "kept",
                        "token_index": counts["kept_rows"],
                        "length_before": observed["length"],
                    }:
                        raise RenderedTrainingError("kept count/length/index declaration mismatch")
                    counts["kept_rows"] += 1
                    offsets.append(offset)
                    row_hashes.append(digest(json_bytes(record)))
                    row_findings.append(observed)
                    pre_lengths.append(observed["length"])
                elif kind in ("overlength", "prefix_drift"):
                    ids = decision.get("input_ids")
                    prefix = decision.get("prefix_ids")
                    if (
                        not isinstance(ids, list)
                        or not isinstance(prefix, list)
                        or not prefix
                        or not ids
                        or any(type(i) is not int or i not in valid_ids for i in ids + prefix)
                        or set(decision) != set(locator) | {"decision", "input_ids", "prefix_ids"}
                    ):
                        raise RenderedTrainingError(
                            "drop requires actual token evidence, not an asserted length"
                        )
                    drift = ids[: len(prefix)] != prefix
                    if kind == "prefix_drift":
                        if settings.mode != "joint_prefix" or not drift:
                            raise RenderedTrainingError("unsubstantiated prefix-drift drop")
                        counts["dropped_prefix_drift"] += 1
                    else:
                        if drift or len(ids) <= settings.max_seq_len:
                            raise RenderedTrainingError("unsubstantiated overlength drop")
                        counts["dropped_overlength"] += 1
                    pre_lengths.append(len(ids))
                    _rejected_update(rejected, source_index, ordinal, kind)
                else:
                    raise UnsupportedRenderedTraining(
                        "unreported or unsupported filtering decision"
                    )
            if source_rows != source["rows"]:
                raise RenderedTrainingError("raw-source row count mismatch")
            verify_file(source["file"])
        if tokens.read(1) or decisions.read(1):
            raise RenderedTrainingError(
                "unaccounted token/decision rows beyond declared source coverage"
            )
    actual = findings(row_findings, pre_lengths, counts, settings)
    if (
        receipt["tokens"]["rows"] != counts["kept_rows"]
        or receipt["decisions"]["rows"] != counts["source_rows"]
        or receipt["findings"] != actual
        or receipt["rejected_locator_sha256"] != rejected.hexdigest()
    ):
        raise RenderedTrainingError("derived findings/counts do not match the actual artifacts")
    if actual["failures"]:
        raise RenderedTrainingError("; ".join(actual["failures"]))
    # Detect changes while the complete verification pass was in progress.
    verify_file(receipt["tokens"]["file"])
    verify_file(receipt["decisions"]["file"])
    verify_file(receipt["tokenizer"])
    verify_file(receipt["template"])
    for entry in receipt["producer"]["files"]:
        verify_file(entry)
    if receipt_path.read_bytes() != raw_receipt:
        raise RenderedTrainingError("receipt changed during verification")
    return receipt, snapshot, backend, offsets, row_hashes, digest(raw_receipt)


def _declaration(card, session_dir):
    from .schema import get

    value = get(card, "setup.rendered_training")
    if (
        not isinstance(value, dict)
        or set(value) != {"receipt", "sha256"}
        or not isinstance(value["receipt"], str)
        or not value["receipt"]
        or not isinstance(value["sha256"], str)
        or len(value["sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in value["sha256"])
    ):
        raise RenderedTrainingError("rendered_training needs only a receipt path and its SHA256")
    path = Path(value["receipt"])
    if not path.is_absolute():
        if session_dir is None:
            raise RenderedTrainingError("relative receipt requires an explicit session directory")
        path = Path(session_dir) / path
    return path.absolute(), value["sha256"]


def _bind_card(card, receipt):
    from .schema import get

    data = get(card, "setup.data")
    token_file = receipt["tokens"]["file"]
    if (
        not isinstance(data, list)
        or len(data) != 1
        or not isinstance(data[0], dict)
        or data[0].get("path") != token_file["path"]
        or data[0].get("n_examples") != receipt["tokens"]["rows"]
    ):
        raise RenderedTrainingError(
            "setup.data must name the actual token file and retained row count"
        )
    settings = receipt["settings"]
    for field, expected in (
        ("stop_token", settings["stop_token"]),
        ("answer_marker", settings["answer_marker"]),
        ("hyperparams.max_seq_len", settings["max_seq_len"]),
    ):
        if get(card, "setup.method." + field) != expected:
            raise RenderedTrainingError(f"card method {field} differs from prepared settings")
    script = get(card, "setup.command.script")
    if script not in {e["path"] for e in receipt["producer"]["files"]}:
        raise RenderedTrainingError(
            "the locked training script must be included in producer sources"
        )


def check_card(card, session_dir=None):
    """Public preflight bridge: absent evidence is unverified, not a failure/PASS."""
    from .schema import get

    if get(card, "setup.rendered_training") is None:
        return {
            "status": "warn",
            "verified_preparation": False,
            "detail": "no rendered token evidence declared; actual rendered supervision is unverified",
        }
    try:
        path, expected = _declaration(card, session_dir)
        receipt, _, _, _, _, _ = _read_and_verify(path, expected)
        _bind_card(card, receipt)
        counts = receipt["findings"]["counts"]
        return {
            "status": "pass",
            "verified_preparation": True,
            "detail": (
                f"all {counts['kept_rows']} retained rows verified from "
                f"{counts['source_rows']} source rows; model consumption remains unknown"
            ),
            "findings": receipt["findings"],
            "template_coverage": receipt["template_coverage"],
            "template_validation_phase": "preparation observation; not rerun by preflight",
            "receipt": str(path),
            "sha256": expected,
        }
    except (OSError, ValueError, TypeError, KeyError, AttributeError, ImportError) as exc:
        return {
            "status": "fail",
            "verified_preparation": False,
            "detail": f"rendered training evidence invalid/unverified: {exc}",
        }


class RenderedTrainingBundle:
    def __init__(self, receipt_path, verified, *, card_path=None):
        from .schema import now

        receipt, snapshot, backend, offsets, row_hashes, verified_sha = verified
        self.receipt_path = Path(receipt_path).absolute()
        self._receipt = receipt
        self._snapshot = snapshot
        self._backend = backend
        self._offsets = offsets
        self._row_hashes = row_hashes
        self._settings = RenderedSettings.parse(receipt["settings"])
        self._receipt_hash = file_entry(self.receipt_path)["sha256"]
        if self._receipt_hash != verified_sha:
            raise RenderedTrainingError(
                "receipt changed between verification and consumer construction"
            )
        self._verified_at = now()
        self._card_path = Path(card_path) if card_path else None
        self._consumer_id = uuid4().hex
        self._observed = {}
        self._dataset = _PreparedDataset(self)

    @property
    def declaration(self):
        return {"receipt": str(self.receipt_path), "sha256": self._receipt_hash}

    @property
    def data_entry(self):
        return {
            "path": self._receipt["tokens"]["file"]["path"],
            "source": "prepared:" + self._receipt_hash,
            "n_examples": self._receipt["tokens"]["rows"],
        }

    @property
    def report(self):
        return {
            "proof": "verified_preparation",
            "verified_at": self._verified_at,
            "verification_scope": "completed verification snapshot, not continuous file monitoring",
            "receipt_sha256": self._receipt_hash,
            "model_consumption": "unknown",
            "template_coverage": self._receipt["template_coverage"],
            "template_validation_phase": "preparation observation; not rerun by verification",
            "findings": copy.deepcopy(self._receipt["findings"]),
        }

    @classmethod
    def verify(cls, receipt_path, *, sha256=None):
        """Revalidate every prepared row without executing renderer/model code."""
        return cls(receipt_path, _read_and_verify(receipt_path, sha256)).report

    @classmethod
    def prepare(
        cls,
        *,
        sources,
        render,
        tokenizer,
        template_bytes,
        settings,
        source_files,
        output,
        reuse=False,
    ):
        from .schema import now

        started = time.monotonic()
        settings = RenderedSettings.parse(settings)
        if not isinstance(template_bytes, bytes):
            raise TypeError("template_bytes must be the frozen UTF-8 snapshot, not a path")
        template = template_bytes.decode("utf-8")
        producer = renderer_identity(render, source_files)
        source_entries = [file_entry(path) for path in sources]
        if not source_entries or len({s["path"] for s in source_entries}) != len(source_entries):
            raise RenderedTrainingError("sources must name distinct, existing raw JSONL files")
        snapshot = tokenizer_snapshot(tokenizer)
        backend = decoder_from_snapshot(snapshot)
        valid_ids = set(backend.get_vocab(with_added_tokens=True).values())
        stop, tail = token_sequences(backend, settings, snapshot)
        output = Path(output).absolute()
        receipt_path = output / "receipt.json"
        if output.exists():
            if not reuse:
                raise RenderedTrainingError(
                    "bundle output already exists; use explicit verified reuse or a new path"
                )
            verified = _read_and_verify(receipt_path)
            old = verified[0]
            if (
                old["settings"] != asdict(settings)
                or old["producer"] != producer
                or [s["file"] for s in old["raw_sources"]] != source_entries
                or strict_json(Path(old["tokenizer"]["path"]).read_bytes()) != snapshot
                or Path(old["template"]["path"]).read_bytes() != template_bytes
            ):
                raise RenderedTrainingError(
                    "cached bundle bindings differ; never reuse stale preparation"
                )
            return cls(receipt_path, verified)
        output.mkdir(parents=True, exist_ok=False)
        _write(output / "tokenizer-snapshot.json", snapshot)
        (output / "template.jinja").write_bytes(template_bytes)
        counts, row_findings, pre_lengths, raw_sources = _new_counts(), [], [], []
        rejected, rng = hashlib.sha256(), random.Random(settings.seed)
        with (
            (output / "tokens.jsonl").open("wb") as tokens,
            (output / "decisions.jsonl").open("wb") as decisions,
        ):
            for source_index, entry in enumerate(source_entries):
                source_rows = 0
                for ordinal, raw, row in _raw_rows(entry["path"]):
                    source_rows += 1
                    counts["source_rows"] += 1
                    locator = {"source": source_index, "row": ordinal, "raw_sha256": digest(raw)}
                    if settings.limit is not None and counts["source_rows"] > settings.limit:
                        decision = {**locator, "decision": "excluded_limit"}
                        counts["excluded_by_limit"] += 1
                        _rejected_update(rejected, source_index, ordinal, "excluded_limit")
                        decisions.write(json_bytes(decision))
                        continue
                    counts["considered_rows"] += 1
                    prefix_text, text = _parts(render, row, template, settings, rng, tokenizer)
                    prefix = backend.encode(prefix_text, add_special_tokens=False).ids
                    encoded = backend.encode(text, add_special_tokens=False).ids
                    ids = prefix + encoded if settings.mode == "separate_concat" else encoded
                    pre_lengths.append(len(ids))
                    kind = "kept"
                    if settings.mode == "joint_prefix" and ids[: len(prefix)] != prefix:
                        kind = "prefix_drift"
                        counts["dropped_prefix_drift"] += 1
                    elif len(ids) > settings.max_seq_len:
                        kind = "overlength"
                        counts["dropped_overlength"] += 1
                    if kind != "kept":
                        decision = {
                            **locator,
                            "decision": kind,
                            "input_ids": ids,
                            "prefix_ids": prefix,
                        }
                        _rejected_update(rejected, source_index, ordinal, kind)
                    else:
                        record = {
                            "source": source_index,
                            "row": ordinal,
                            "input_ids": ids,
                            "labels": [-100] * len(prefix) + ids[len(prefix) :],
                            "target_start": len(prefix),
                            "rendered_sha256": {
                                "prefix": digest(prefix_text.encode()),
                                "target_or_full": digest(text.encode()),
                            },
                        }
                        row_findings.append(
                            validate_row(record, settings, backend, valid_ids, stop, tail)
                        )
                        decision = {
                            **locator,
                            "decision": kind,
                            "token_index": counts["kept_rows"],
                            "length_before": len(ids),
                        }
                        counts["kept_rows"] += 1
                        tokens.write(json_bytes(record))
                    decisions.write(json_bytes(decision))
                raw_sources.append({"file": entry, "rows": source_rows})
                verify_file(entry)
        if (
            tokenizer_snapshot(tokenizer) != snapshot
            or renderer_identity(render, source_files) != producer
        ):
            raise RenderedTrainingError(
                "effective tokenizer or producer sources changed during preparation"
            )
        actual = findings(row_findings, pre_lengths, counts, settings)
        template_coverage = (
            "prompt_and_full_replayed"
            if settings.mode == "joint_prefix"
            else "prompt_replayed_target_not_template_equivalence"
        )
        if settings.prompt_mode == "pre_rendered":
            template_coverage = "not_applied_reference_only_equivalence_unverified"
        receipt = {
            "schema_version": SCHEMA,
            "implementation": implementation_identity(),
            "producer": producer,
            "settings": asdict(settings),
            "raw_sources": raw_sources,
            "tokenizer": file_entry(output / "tokenizer-snapshot.json"),
            "template": file_entry(output / "template.jinja"),
            "tokens": {"file": file_entry(output / "tokens.jsonl"), "rows": counts["kept_rows"]},
            "decisions": {
                "file": file_entry(output / "decisions.jsonl"),
                "rows": counts["source_rows"],
            },
            "findings": actual,
            "rejected_locator_sha256": rejected.hexdigest(),
            "template_coverage": template_coverage,
            "prepared_at": now(),
            "prepare_seconds": time.monotonic() - started,
        }
        # Invalid semantic findings remain inspectable, never a verified bundle.
        _write(receipt_path, receipt)
        return cls(receipt_path, _read_and_verify(receipt_path))

    @classmethod
    def open_for_training(cls, card_path):
        from . import lock, schema

        card_path = Path(card_path).absolute()
        if (
            card_path != card_path.resolve()
            or card_path.parent.name != "cards"
            or card_path.parent.parent.name != "memory"
        ):
            raise RenderedTrainingError(
                "consumer requires the canonical memory/cards/exp-NN.yaml path"
            )
        card = schema.load_card(card_path)
        if card_path.name != f"{card.get('card_id')}.yaml":
            raise RenderedTrainingError("card filename does not match its card_id")
        if schema.get(card, "conclusion.decision") is not None:
            raise RenderedTrainingError("a closed card cannot authorize a new training consumer")
        session_dir = card_path.parents[2]
        plan = schema.validate_plan(card, session_dir)
        if not plan.ok:
            raise RenderedTrainingError(plan.render())
        info = lock.read_lock(card_path)
        if (
            not isinstance(info, dict)
            or info.get("schema_version") != lock.LOCK_SCHEMA
            or not isinstance(info.get("locked_at"), str)
            or not info["locked_at"].strip()
        ):
            raise RenderedTrainingError("missing/malformed successful lock schema or locked_at")
        script = schema.get(card, "setup.command.script")
        if (
            not isinstance(script, str)
            or not Path(script).is_absolute()
            or not Path(script).is_file()
        ):
            raise RenderedTrainingError(
                "consumer needs an existing absolute main script bound by the lock"
            )
        if info.get("script") != {"path": script, "sha256": file_entry(script)["sha256"]}:
            raise RenderedTrainingError(
                "lock does not contain the exact existing main-script hash binding"
            )
        integrity = lock.verify_lock(card_path, card)
        if not info or not integrity.ok or integrity.warnings:
            raise RenderedTrainingError("missing/stale/uncheckable lock: " + integrity.render())
        if info.get("card_id") != card.get("card_id"):
            raise RenderedTrainingError("lock belongs to another card")
        receipt_path, expected = _declaration(card, session_dir)
        verified = _read_and_verify(receipt_path, expected)
        _bind_card(card, verified[0])
        token = verified[0]["tokens"]["file"]
        if not any(e == token for e in info.get("data", [])):
            raise RenderedTrainingError("actual token artifact is not pinned by the matching lock")
        bundle = cls(receipt_path, verified, card_path=card_path)
        bundle._plan_hash = schema.plan_hash(card)
        bundle._lock_hash = file_entry(lock.lock_path(card_path))["sha256"]
        bundle.flush_consumption()
        return bundle

    @property
    def dataset(self):
        if self._card_path is None:
            raise RenderedTrainingError(
                "training dataset requires open_for_training and a successful matching lock"
            )
        return self._dataset

    def _observe(self, kind, count=1):
        pid = os.getpid()
        entry = self._observed.setdefault(
            pid, {"dataset_rows": 0, "collator_rows": 0, "collator_batches": 0}
        )
        first = entry[kind] == 0
        entry[kind] += count
        if first:
            self.flush_consumption()

    def flush_consumption(self):
        from . import lock, schema

        if self._card_path is None:
            raise RenderedTrainingError("no locked consumer is bound")
        card = schema.load_card(self._card_path)
        if (
            schema.plan_hash(card) != self._plan_hash
            or file_entry(lock.lock_path(self._card_path))["sha256"] != self._lock_hash
            or file_entry(self.receipt_path)["sha256"] != self._receipt_hash
        ):
            raise RenderedTrainingError("consumer card/lock/receipt changed after binding")
        if schema.get(card, "conclusion.decision") is not None:
            raise RenderedTrainingError(
                "consumer card was closed; no further data access is authorized"
            )
        counts = self._observed.get(
            os.getpid(), {"dataset_rows": 0, "collator_rows": 0, "collator_batches": 0}
        )
        proof = (
            "observed_collator_consumption"
            if counts["collator_batches"]
            else "observed_dataset_access"
            if counts["dataset_rows"]
            else "verified_loader_binding"
        )
        record = {
            "proof": proof,
            "model_consumption": "unknown",
            "preparation_verified": True,
            "loader_bound": True,
            "dataset_access_observed": counts["dataset_rows"] > 0,
            "collation_observed": counts["collator_batches"] > 0,
            "prepared_receipt_sha256": self._receipt_hash,
            "plan_sha256": self._plan_hash,
            "card": str(self._card_path),
            "pid": os.getpid(),
            "observed_at": schema.now(),
            "attempt_id_hint": os.environ.get("AWM_EXP_ATTEMPT_ID"),
            "counts_at_last_flush": dict(counts),
            "counts_are_lower_bounds": True,
            "meaning": "verified loader binding and observed CPU data access, not model/optimizer execution",
        }
        directory = self._card_path.parents[1] / "rendered-consumers" / self._card_path.stem
        path = directory / f"{self._consumer_id}-{os.getpid()}.json"
        _write(path, record)
        return record

    def collator(self, *, pad_to_multiple_of=None, return_tensors="pt"):
        if self._card_path is None:
            raise RenderedTrainingError("collator requires the locked training loader")
        multiple = self._settings.pad_to_multiple_of
        if pad_to_multiple_of is not None and pad_to_multiple_of != multiple:
            raise RenderedTrainingError("collator padding differs from prepared settings")
        if return_tensors not in ("pt", "python"):
            raise UnsupportedRenderedTraining(
                "only CPU PyTorch or Python-list collation is supported"
            )
        return _CheckedCollator(self, return_tensors)


class _PreparedDataset:
    """Seek-indexed single token artifact; never materialize a second token cache."""

    def __init__(self, bundle):
        self.bundle = bundle
        self._streams = {}
        self._contexts = {}
        self._mutex = threading.Lock()

    def close(self):
        for context in self._contexts.values():
            context.close()
        self._contexts.clear()
        self._streams.clear()

    def __del__(self):
        for context in getattr(self, "_contexts", {}).values():
            context.close()

    def __len__(self):
        return len(self.bundle._offsets)

    def __getstate__(self):
        return {"bundle": self.bundle}

    def __setstate__(self, state):
        self.__init__(state["bundle"])

    def __getitem__(self, index):
        if type(index) is not int:
            raise TypeError("token dataset index must be an integer")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        path = self.bundle._receipt["tokens"]["file"]["path"]
        with self._mutex:
            stream = self._streams.get(os.getpid())
            if stream is None or stream.closed:
                context = self._contexts.setdefault(os.getpid(), ExitStack())
                stream = context.enter_context(open(path, "rb"))  # noqa: SIM115 - worker ExitStack owns lifetime
                self._streams[os.getpid()] = stream
            stream.seek(self.bundle._offsets[index])
            record, _ = _line(stream, "consumed token record")
        if digest(json_bytes(record)) != self.bundle._row_hashes[index]:
            raise RenderedTrainingError("consumed token record changed after full verification")
        self.bundle._observe("dataset_rows")
        return {**record, "_awm_row_index": index, "_awm_bundle_sha256": self.bundle._receipt_hash}


class _CheckedCollator:
    def __init__(self, bundle, return_tensors):
        self.bundle, self.return_tensors = bundle, return_tensors

    def __call__(self, features):
        if not features:
            raise RenderedTrainingError("cannot collate an empty batch")
        bundle = self.bundle
        owned = []
        for original_feature in features:
            feature = copy.deepcopy(original_feature)
            if not isinstance(feature, dict):
                raise RenderedTrainingError(
                    "collator requires unchanged features from the checked dataset"
                )
            index = feature.get("_awm_row_index")
            record = {
                k: v
                for k, v in feature.items()
                if k not in ("_awm_row_index", "_awm_bundle_sha256")
            }
            if (
                type(index) is not int
                or not 0 <= index < len(bundle._row_hashes)
                or feature.get("_awm_bundle_sha256") != bundle._receipt_hash
                or digest(json_bytes(record)) != bundle._row_hashes[index]
            ):
                raise RenderedTrainingError(
                    "feature/source identity changed; use remove_unused_columns=False and no transforms"
                )
            owned.append(feature)
        features = owned
        multiple = bundle._settings.pad_to_multiple_of
        width = max(len(f["input_ids"]) for f in features)
        width = ((width + multiple - 1) // multiple) * multiple
        pad = bundle._snapshot["specials"]["pad_token_id"]
        left = bundle._snapshot["padding_side"] == "left"
        batch = {"input_ids": [], "labels": [], "attention_mask": []}
        for feature in features:
            n = len(feature["input_ids"])
            for key, values, padding in (
                ("input_ids", feature["input_ids"], [pad] * (width - n)),
                ("labels", feature["labels"], [-100] * (width - n)),
                ("attention_mask", [1] * n, [0] * (width - n)),
            ):
                batch[key].append(padding + values if left else values + padding)
        if self.return_tensors == "pt":
            try:
                import torch
            except ImportError as exc:
                raise UnsupportedRenderedTraining(
                    "CPU PyTorch is unavailable for tensor collation"
                ) from exc
            batch = {k: torch.tensor(v, dtype=torch.long, device="cpu") for k, v in batch.items()}
        bundle._observe("collator_rows", len(features))
        bundle._observe("collator_batches")
        return batch
