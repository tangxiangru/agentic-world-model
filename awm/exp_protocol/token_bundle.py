"""Model-free primitives for the rendered-training artifact (no eager ML imports)."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA = "awm-rendered-training-v1"
VERSIONS = {"transformers": "4.57.3", "tokenizers": "0.22.2"}


class RenderedTrainingError(ValueError):
    """Missing, stale, malformed, or invalid rendered-training evidence."""


class UnsupportedRenderedTraining(RenderedTrainingError):
    """An adapter would be needed; no preparation/consumer proof is supplied."""


def json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise RenderedTrainingError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    result = json.loads(data, object_pairs_hook=pairs)
    json_bytes(result)  # Also rejects NaN/Infinity and numeric overflow to infinity.
    return result


def file_entry(path) -> dict:
    from .schema import sha256_file

    path = Path(path).absolute()
    if not path.is_file():
        raise RenderedTrainingError(f"missing bound file: {path}")
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def verify_file(entry) -> Path:
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise RenderedTrainingError("malformed file identity")
    path = Path(entry["path"])
    if not path.is_absolute() or file_entry(path) != entry:
        raise RenderedTrainingError(f"bound file changed: {path}")
    return path


def implementation_identity() -> dict:
    return {
        name: file_entry(Path(__file__).with_name(name))["sha256"]
        for name in ("token_bundle.py", "rendered_training.py", "schema.py", "save_contract.py")
    }


@dataclass(frozen=True)
class RenderedSettings:
    mode: str
    max_seq_len: int
    stop_token: str
    prompt_mode: str = "template_replay"
    answer_marker: str | None = None
    tail_text: str = ""
    seed: int = 0
    limit: int | None = None
    pad_to_multiple_of: int = 1
    stop_min_fraction: float = 0.95
    marker_max_bad_fraction: float = 0.02
    max_drop_fraction: float | None = None
    renderer: dict = field(default_factory=dict)
    add_special_tokens: bool = False
    length_policy: str = "drop_overlength"
    label_policy: str = "completion_only_causal_unshifted"

    @classmethod
    def parse(cls, value):
        if isinstance(value, cls):
            value = asdict(value)
        if not isinstance(value, dict):
            raise RenderedTrainingError("settings must be RenderedSettings or an explicit mapping")
        try:
            settings = cls(**value)
        except TypeError as exc:
            raise RenderedTrainingError(f"invalid settings: {exc}") from exc
        if (
            settings.mode not in ("separate_concat", "joint_prefix")
            or settings.prompt_mode not in ("template_replay", "pre_rendered")
            or settings.add_special_tokens is not False
            or settings.length_policy != "drop_overlength"
            or settings.label_policy != "completion_only_causal_unshifted"
        ):
            raise UnsupportedRenderedTraining(
                "unsupported tokenization, truncation, packing, or label policy"
            )
        for name in ("max_seq_len", "pad_to_multiple_of"):
            if type(getattr(settings, name)) is not int or getattr(settings, name) <= 0:
                raise RenderedTrainingError(f"{name} must be a positive integer")
        if type(settings.seed) is not int:
            raise RenderedTrainingError("seed must be an integer")
        if settings.limit is not None and (type(settings.limit) is not int or settings.limit < 0):
            raise RenderedTrainingError(
                "limit must be null or a nonnegative whole-source-prefix count"
            )
        if not isinstance(settings.stop_token, str) or not settings.stop_token:
            raise RenderedTrainingError("stop_token must be a nonempty string")
        if not isinstance(settings.tail_text, str):
            raise RenderedTrainingError("tail_text must be a string")
        if settings.answer_marker is not None and (
            not isinstance(settings.answer_marker, str) or not settings.answer_marker
        ):
            raise RenderedTrainingError("answer_marker must be null or a nonempty string")
        if not isinstance(settings.renderer, dict):
            raise RenderedTrainingError("renderer settings must be a JSON mapping")
        for name in ("stop_min_fraction", "marker_max_bad_fraction", "max_drop_fraction"):
            number = getattr(settings, name)
            if number is None and name == "max_drop_fraction":
                continue
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(number)
                or not 0 <= number <= 1
            ):
                raise RenderedTrainingError(f"{name} must be finite and between zero and one")
        if settings.stop_min_fraction < 0.95 or settings.marker_max_bad_fraction > 0.02:
            raise RenderedTrainingError(
                "settings cannot weaken the protocol's 95% stop / 2% bad-marker thresholds; "
                "exceptions remain unverified and use the recorded raw-check override route"
            )
        json_bytes(asdict(settings))
        return settings


@dataclass(frozen=True)
class RenderedParts:
    """Actual strings from the shared renderer, with template-reference messages.

    For separate_concat return prefix + target, without rewrapping the prefix.
    For joint_prefix return prefix + full and the full conversation messages.
    The producer checks the supplied strings against the explicit template.
    """

    prefix: str
    messages: list[dict] | None = None
    target: str | None = None
    full: str | None = None
    full_messages: list[dict] | None = None


def _versions():
    try:
        actual = {name: importlib.metadata.version(name) for name in VERSIONS}
    except importlib.metadata.PackageNotFoundError as exc:
        raise UnsupportedRenderedTraining("pinned fast-tokenizer runtime is unavailable") from exc
    if actual != VERSIONS:
        raise UnsupportedRenderedTraining(
            f"unsupported tokenizer runtime: {actual}; expected {VERSIONS}"
        )
    return actual


def tokenizer_snapshot(tokenizer) -> dict:
    _versions()
    from transformers import PreTrainedTokenizerFast
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if (
        not isinstance(tokenizer, PreTrainedTokenizerFast)
        or not type(tokenizer).__module__.startswith("transformers.")
        or getattr(tokenizer.apply_chat_template, "__func__", None)
        is not PreTrainedTokenizerBase.apply_chat_template
        or getattr(tokenizer._decode, "__func__", None) is not PreTrainedTokenizerFast._decode
    ):
        raise UnsupportedRenderedTraining(
            "only native serializable HF fast tokenizers are supported"
        )
    backend = tokenizer.backend_tokenizer.to_str()
    state = strict_json(backend)
    if state.get("model", {}).get("dropout") not in (None, 0, 0.0):
        raise UnsupportedRenderedTraining("stochastic tokenizer dropout is unsupported")
    specials = {}
    for key in ("bos", "eos", "pad", "unk", "sep", "cls", "mask"):
        specials[key + "_token"] = getattr(tokenizer, key + "_token", None)
        specials[key + "_token_id"] = getattr(tokenizer, key + "_token_id", None)
    if type(specials["pad_token_id"]) is not int:
        raise UnsupportedRenderedTraining("an explicit tokenizer pad token is required")
    if tokenizer.padding_side not in ("left", "right") or tokenizer.truncation_side not in (
        "left",
        "right",
    ):
        raise UnsupportedRenderedTraining("unknown tokenizer padding/truncation side")
    assets = []
    local = Path(tokenizer.name_or_path)
    if tokenizer.name_or_path and local.is_dir():
        names = {
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "tokenizer.model",
            "spiece.model",
            "vocab.json",
            "merges.txt",
        }
        names.update(tokenizer.vocab_files_names.values())
        assets = [file_entry(local / name) for name in sorted(names) if (local / name).is_file()]
    return {
        "runtime": dict(VERSIONS),
        "class": type(tokenizer).__module__ + "." + type(tokenizer).__qualname__,
        "backend": state,
        "specials": specials,
        "all_special_ids": list(tokenizer.all_special_ids),
        "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side,
        "model_max_length": tokenizer.model_max_length,
        "chat_template": tokenizer.chat_template,
        "assets": assets,
        "decode": {"skip_special_tokens": False, "clean_up_tokenization_spaces": False},
    }


def decoder_from_snapshot(snapshot):
    _versions()
    from tokenizers import Tokenizer

    required = {
        "runtime",
        "class",
        "backend",
        "specials",
        "all_special_ids",
        "padding_side",
        "truncation_side",
        "model_max_length",
        "chat_template",
        "assets",
        "decode",
    }
    if (
        not isinstance(snapshot, dict)
        or set(snapshot) != required
        or not isinstance(snapshot.get("class"), str)
        or not snapshot["class"].startswith("transformers.")
        or snapshot.get("runtime") != VERSIONS
        or snapshot.get("padding_side") not in ("left", "right")
        or snapshot.get("truncation_side") not in ("left", "right")
        or snapshot.get("decode")
        != {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
    ):
        raise UnsupportedRenderedTraining("incomplete or unsupported tokenizer snapshot")
    for asset in snapshot["assets"]:
        verify_file(asset)
    try:
        backend = Tokenizer.from_str(json.dumps(snapshot["backend"]))
    except Exception as exc:
        raise UnsupportedRenderedTraining(
            f"tokenizer backend snapshot cannot be reconstructed: {exc}"
        ) from exc
    # These are the explicit effective call options; the original active settings
    # remain in the identity, but no silent backend truncation/padding is used.
    backend.no_padding()
    backend.no_truncation()
    valid_ids = set(backend.get_vocab(with_added_tokens=True).values())
    if (
        type(snapshot["specials"].get("pad_token_id")) is not int
        or snapshot["specials"]["pad_token_id"] not in valid_ids
    ):
        raise UnsupportedRenderedTraining("snapshot pad token is invalid")
    if snapshot["backend"].get("model", {}).get("dropout") not in (None, 0, 0.0):
        raise UnsupportedRenderedTraining("stochastic tokenizer snapshot is unsupported")
    return backend


def renderer_identity(render, source_files) -> dict:
    if not inspect.isfunction(render) or render.__closure__:
        raise UnsupportedRenderedTraining(
            "renderer must be a source-backed function, not hidden closure state"
        )
    source = inspect.getsourcefile(render)
    entries = [file_entry(path) for path in source_files]
    if len({e["path"] for e in entries}) != len(entries):
        raise RenderedTrainingError("duplicate producer source paths")
    if source is None or str(Path(source).absolute()) not in {e["path"] for e in entries}:
        raise RenderedTrainingError("source_files must include the shared renderer's actual module")
    return {
        "module": render.__module__,
        "qualname": render.__qualname__,
        "files": entries,
        "source_closure": "explicit_files_not_arbitrary_import_proof",
    }


def token_sequences(backend, settings, snapshot):
    stop = backend.encode(settings.stop_token, add_special_tokens=False).ids
    tail = backend.encode(settings.tail_text, add_special_tokens=False).ids
    if not stop or snapshot["specials"]["unk_token_id"] in stop:
        raise RenderedTrainingError("stop sequence is empty or resolves to an unknown token")
    if backend.decode(stop, skip_special_tokens=False) != settings.stop_token:
        raise UnsupportedRenderedTraining(
            "stop text does not round-trip through the effective tokenizer"
        )
    return stop, tail


def validate_row(row, settings, backend, valid_ids, stop, tail) -> dict:
    if not isinstance(row, dict) or set(row) != {
        "input_ids",
        "labels",
        "target_start",
        "source",
        "row",
        "rendered_sha256",
    }:
        raise RenderedTrainingError("malformed token record or unsupported packed/custom fields")
    rendered = row["rendered_sha256"]
    if (
        not isinstance(rendered, dict)
        or set(rendered) != {"prefix", "target_or_full"}
        or any(
            not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
            for v in rendered.values()
        )
    ):
        raise RenderedTrainingError("missing observed rendered-string identities")
    ids, labels, start = row["input_ids"], row["labels"], row["target_start"]
    if (
        not isinstance(ids, list)
        or not isinstance(labels, list)
        or len(ids) != len(labels)
        or not ids
        or any(type(token) is not int or token not in valid_ids for token in ids)
        or any(type(token) is not int for token in labels)
        or type(start) is not int
        or not 0 < start < len(ids)
        or type(row["source"]) is not int
        or row["source"] < 0
        or type(row["row"]) is not int
        or row["row"] < 1
    ):
        raise RenderedTrainingError(
            "invalid token IDs, lengths, source locator or empty supervision"
        )
    if labels[:start] != [-100] * start or labels[start:] != ids[start:]:
        raise RenderedTrainingError("labels do not match completion-only unshifted supervision")
    if len(ids) > settings.max_seq_len:
        raise RenderedTrainingError("retained unpadded example exceeds max_seq_len")
    target = ids[start:]
    has_tail = not tail or target[-len(tail) :] == tail
    content = target[: -len(tail)] if tail and has_tail else target
    has_terminal = len(content) >= len(stop) and content[-len(stop) :] == stop
    terminal_ok = has_tail and has_terminal
    occurrences = (
        target.count(stop[0])
        if len(stop) == 1
        else sum(target[i : i + len(stop)] == stop for i in range(len(target) - len(stop) + 1))
    )
    terminal_ok = terminal_ok and occurrences == 1
    # Template tail is not answer text even when the terminal token is wrong.
    body = content[: -len(stop)] if has_terminal else content
    text = backend.decode(body, skip_special_tokens=False)
    marker_bad = settings.answer_marker is not None and text.count(settings.answer_marker) != 1
    return {
        "length": len(ids),
        "target_tokens": len(target),
        "stop_ok": terminal_ok,
        "marker_bad": bool(marker_bad),
    }


def findings(rows, pre_lengths, counts, settings):
    kept = counts["kept_rows"]
    if kept == 0:
        raise RenderedTrainingError("no retained training rows")

    def distribution(values):
        values = sorted(values)
        return {
            "n": len(values),
            "sum": sum(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "p50": values[len(values) // 2] if values else None,
            "p99": values[min(len(values) - 1, int(len(values) * 0.99))] if values else None,
        }

    stop_ok = sum(r["stop_ok"] for r in rows)
    marker_bad = sum(r["marker_bad"] for r in rows)
    dropped = counts["dropped_overlength"] + counts["dropped_prefix_drift"]
    drop_fraction = dropped / counts["considered_rows"]
    failures = []
    if stop_ok / kept < settings.stop_min_fraction:
        failures.append("supervised stop consistency below the explicit threshold")
    if marker_bad / kept > settings.marker_max_bad_fraction:
        failures.append("supervised answer-marker failure rate exceeds the explicit threshold")
    if settings.max_drop_fraction is not None and drop_fraction > settings.max_drop_fraction:
        failures.append("drop fraction exceeds the explicit cap")
    return {
        "counts": counts,
        "stop_ok": stop_ok,
        "stop_fraction": stop_ok / kept,
        "marker_bad": marker_bad,
        "marker_bad_fraction": marker_bad / kept,
        "marker_applicable": settings.answer_marker is not None,
        "drop_fraction": drop_fraction,
        "pre_filter_lengths": distribution(pre_lengths),
        "post_filter_lengths": distribution([r["length"] for r in rows]),
        "supervised_tokens": sum(r["target_tokens"] for r in rows),
        "failures": failures,
    }
