"""Retain and check offline sampling evidence before fallible post-processing.

No model is imported or run at module import. Native inference is only invoked
by record_vllm inside the scientist's already-locked command. This is not a grader.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from . import lock, schema

VLLM_VERSION = "0.11.0"
VLLM_SOURCES = {
    "sampling_params.py": "8219442773ae7cca6d3b63507438e8b0654fbcf024bf7f037dd80ed9f860bf01",
    "outputs.py": "ee60aa7563b7db029c5cc352fcdbd63015575e2d5f898933dba294fb69afd389",
    "entrypoints/llm.py": "ee4a1ae908f6e04b822c8ef55bb1ecff89a781cee90944a3d989e85286b6489a",
}
SCHEMA = "awm-sampling-record-v1"


class SamplingEvidenceError(ValueError):
    """Evidence is incomplete, inconsistent, or outside the supported adapter."""


def _bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _plain(value):
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return {"set": sorted((_plain(item) for item in value), key=lambda item: _bytes(item))}
    if isinstance(value, dict):
        if all(type(key) is str for key in value):
            return {key: _plain(item) for key, item in value.items()}
        return {"mapping": [[_plain(key), _plain(item)] for key, item in value.items()]}
    # Native RequestOutputKind is Enum, not IntEnum; preserve its type and value.
    from enum import Enum

    if isinstance(value, Enum):
        return {
            "enum": f"{type(value).__module__}.{type(value).__name__}",
            "value": _plain(value.value),
        }
    raise SamplingEvidenceError(f"unsupported/non-finite parameter value: {type(value).__name__}")


def _write_json(path: Path, value) -> None:
    encoded = _bytes(value)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def finite_float(value) -> float:
    """A finite numeric conversion, not an implementation of any benchmark grader."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("numeric value is not finite")
    return number


def _token_ids(value) -> tuple[int, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or any(type(item) is not int or item < 0 for item in value)
    ):
        raise SamplingEvidenceError("token IDs must be a non-text sequence of nonnegative integers")
    return tuple(value)


@dataclass(frozen=True)
class PreparedPrompt:
    ordinal: int
    item_id: int | str | None
    text: str
    token_ids: tuple[int, ...]
    bos_policy: str


def prepare_prompts(texts, tokenizer, *, item_ids=None, bos_policy="unconstrained"):
    """CPU-only. Do not apply another template to already-rendered strings."""
    if bos_policy not in ("unconstrained", "single_at_start"):
        raise SamplingEvidenceError("unknown BOS policy")
    texts = list(texts)
    identifiers = list(range(len(texts))) if item_ids is None else list(item_ids)
    if not texts or len(texts) != len(identifiers):
        raise SamplingEvidenceError("nonempty prompts and matching item IDs are required")
    result = []
    for ordinal, (text, item_id) in enumerate(zip(texts, identifiers)):
        if not isinstance(text, str) or type(item_id) not in (str, int, type(None)):
            raise SamplingEvidenceError(
                "prompts must be strings; source IDs are typed int/string/null"
            )
        ids = _token_ids(tokenizer.encode(text, add_special_tokens=False))
        if not ids:
            raise SamplingEvidenceError("empty tokenized prompt")
        if bos_policy == "single_at_start":
            bos = getattr(tokenizer, "bos_token_id", None)
            if type(bos) is not int or ids[0] != bos or (len(ids) > 1 and ids[1] == bos):
                raise SamplingEvidenceError("prompt violates declared single-at-start BOS policy")
        result.append(PreparedPrompt(ordinal, item_id, text, ids, bos_policy))
    return tuple(result)


def resolve_stop_ids(tokenizer, stop_tokens) -> list[int]:
    """Resolve explicit token spellings; multi-token string stops need another adapter."""
    tokens = list(stop_tokens)
    if not tokens or any(not isinstance(token, str) or not token for token in tokens):
        raise SamplingEvidenceError("declare required stop-token spellings")
    vocabulary = tokenizer.get_vocab()
    result = []
    for token in tokens:
        identifier = vocabulary.get(token)
        encoded = _token_ids(tokenizer.encode(token, add_special_tokens=False))
        if type(identifier) is not int or encoded != (identifier,):
            raise SamplingEvidenceError(f"stop {token!r} is not a single known tokenizer token")
        if identifier not in result:
            result.append(identifier)
    return result


def _card_evidence(card_path):
    from .execution import _live_plan

    path = Path(card_path).resolve()
    if len(path.parents) < 3 or path.parent.name != "cards" or path.parent.parent.name != "memory":
        raise SamplingEvidenceError("sampling needs a session memory/cards card")
    card, info, _ = _live_plan(path, path.parents[2])
    return {
        "card_path": str(path),
        "card_id": card["card_id"],
        "plan_sha256": info["plan_sha256"],
        "lock_sha256": _sha(lock.lock_path(path).read_bytes()),
    }


def _native_runtime(params):
    """Inspect the pinned adapter without constructing an engine or loading weights."""
    import vllm
    from vllm import LLM, SamplingParams

    if vllm.__version__ != VLLM_VERSION or type(params) is not SamplingParams:
        raise SamplingEvidenceError("requires pinned native vLLM 0.11.0 LLM and SamplingParams")
    root = Path(vllm.__file__).parent
    for relative, digest in VLLM_SOURCES.items():
        if _sha((root / relative).read_bytes()) != digest:
            raise SamplingEvidenceError(f"pinned vLLM source mismatch: {relative}")
    if (
        Path(inspect.getsourcefile(LLM.generate)).resolve()
        != (root / "entrypoints/llm.py").resolve()
    ):
        raise SamplingEvidenceError("native offline generate entrypoint was replaced")
    # Bind the actual supported invocation on CPU; never probe by running a model.
    inspect.signature(LLM.generate).bind(None, [], params, use_tqdm=False)
    return {
        "vllm": VLLM_VERSION,
        "sources": dict(VLLM_SOURCES),
        "order_contract": "native offline generate returns requests in input order",
        "resolved_engine_configuration": "not_independently_verified",
    }


def _native_identity(llm, params):
    from vllm import LLM

    native = _native_runtime(params)
    if type(llm) is not LLM:
        raise SamplingEvidenceError("requires pinned native vLLM 0.11.0 LLM and SamplingParams")
    if (
        getattr(llm.generate, "__func__", None) is not LLM.generate
        or getattr(llm.get_tokenizer, "__func__", None) is not LLM.get_tokenizer
    ):
        raise SamplingEvidenceError("native offline generate/tokenizer entrypoint was replaced")
    return native


def sampling_ready(prepared, params, output_dir, *, tokenizer, card_path, required_stop_tokens):
    """CPU readiness for THIS locked sampling stage, not a model/GPU validation.

    An override of a preflight heuristic does not waive live input hashes.
    Generated training inputs belong in a later card, after they are persisted.
    """
    card = _card_evidence(card_path)
    native = _native_runtime(params)
    prepared = tuple(prepared)
    if not prepared or any(
        not isinstance(item, PreparedPrompt) or item.ordinal != ordinal
        for ordinal, item in enumerate(prepared)
    ):
        raise SamplingEvidenceError("use nonempty ordered prepare_prompts output")
    for item in prepared:
        actual = prepare_prompts(
            [item.text], tokenizer, item_ids=[item.item_id], bos_policy=item.bos_policy
        )[0]
        if actual.token_ids != item.token_ids:
            raise SamplingEvidenceError("CPU tokenizer does not reproduce prepared prompt tokens")
    stops = resolve_stop_ids(tokenizer, required_stop_tokens)
    requested = _parameters(params, stops)
    if Path(output_dir).exists() or Path(output_dir).is_symlink():
        raise FileExistsError("recording destination already exists; retain it and use a new path")
    return {
        "status": "ready_before_engine",
        "checked_at": schema.now(),
        "card": card,
        "native": native,
        "requested_parameters": requested,
        "model_loading": "not_performed",
        "scientific_validation": "not_performed",
    }


def record_vllm_from_factory(
    engine_factory,
    prepared,
    params,
    output_dir,
    *,
    tokenizer,
    card_path,
    required_stop_tokens,
    close_engine=None,
):
    """Reject CPU-detectable defects before calling the scientist's engine factory.

    Factory/cleanup run in the caller's process. This is NOT a timeout manager.
    Run it in a stage with an owned deadline/exit observer. Optional cleanup sees
    only the returned engine; a factory failing midway must manage its own state.
    """
    if not callable(engine_factory) or (close_engine is not None and not callable(close_engine)):
        raise SamplingEvidenceError("engine factory and optional cleanup must be callable")
    prepared = tuple(prepared)
    required_stop_tokens = tuple(required_stop_tokens)
    params = copy.deepcopy(params)
    sampling_ready(
        prepared,
        params,
        output_dir,
        tokenizer=tokenizer,
        card_path=card_path,
        required_stop_tokens=required_stop_tokens,
    )
    engine = engine_factory()
    failure = None
    try:
        # Recheck the live card, tokenizer and native instance after construction.
        return record_vllm(
            engine,
            prepared,
            params,
            output_dir,
            card_path=card_path,
            required_stop_tokens=required_stop_tokens,
        )
    except BaseException as exc:
        failure = exc
        raise
    finally:
        if close_engine is not None:
            try:
                close_engine(engine)
            except BaseException as cleanup_error:
                if failure is None:
                    raise
                if hasattr(failure, "add_note"):
                    failure.add_note(f"engine cleanup also failed: {cleanup_error}")


def _parameters(params, required_stops):
    if (
        type(params.n) is not int
        or params.n < 1
        or type(params.max_tokens) is not int
        or params.max_tokens < 1
    ):
        raise SamplingEvidenceError("explicit positive n and max_tokens are required")
    if getattr(params, "_real_n", None) is not None or getattr(params, "logits_processors", None):
        raise SamplingEvidenceError("best-of and custom logits processors are not supported")
    stops = _token_ids(params.stop_token_ids)
    if not set(required_stops).issubset(stops):
        raise SamplingEvidenceError("actual SamplingParams omit a required stop token")
    fields = getattr(type(params), "__struct_fields__", None)
    if fields is None:
        raise SamplingEvidenceError("native parameter fields are unavailable")
    return {field: _plain(getattr(params, field)) for field in fields}


def _raw_request(output, ordinal):
    return {
        "ordinal": ordinal,
        "request_id": output.request_id,
        "prompt_token_ids": list(_token_ids(output.prompt_token_ids)),
        "finished": output.finished,
        "completions": [
            {
                "index": completion.index,
                "text": completion.text,
                "token_ids": list(_token_ids(completion.token_ids)),
                "finish_reason": completion.finish_reason,
                "stop_reason": completion.stop_reason,
            }
            for completion in output.outputs
        ],
    }


def _validate_request(raw, prepared, expected_n):
    if raw["prompt_token_ids"] != list(prepared.token_ids):
        raise SamplingEvidenceError(
            "returned prompt tokens/order differ from the actual submitted input"
        )
    if raw["finished"] is not True or len(raw["completions"]) != expected_n:
        raise SamplingEvidenceError(
            "request is unfinished or completion count differs from requested n"
        )
    indices = [item["index"] for item in raw["completions"]]
    if any(type(index) is not int for index in indices) or sorted(indices) != list(
        range(expected_n)
    ):
        raise SamplingEvidenceError("completion indices are missing/duplicated")
    for item in raw["completions"]:
        if not isinstance(item["text"], str) or item["finish_reason"] not in ("stop", "length"):
            raise SamplingEvidenceError(
                "completion text or native finish reason is unsupported/incomplete"
            )
        if type(item["stop_reason"]) not in (str, int, type(None)):
            raise SamplingEvidenceError("unsupported stop reason")


def _call_engine(llm, inputs, params):
    return llm.generate(inputs, params, use_tqdm=False)


def record_vllm(llm, prepared, params, output_dir, *, card_path, required_stop_tokens):
    """One native synchronous call. No parser runs until its raw output is durable."""
    prepared = tuple(prepared)
    if not prepared or any(
        not isinstance(item, PreparedPrompt) or item.ordinal != ordinal
        for ordinal, item in enumerate(prepared)
    ):
        raise SamplingEvidenceError("use nonempty ordered prepare_prompts output")
    card = _card_evidence(card_path)
    native = _native_identity(llm, params)
    tokenizer = llm.get_tokenizer()
    for item in prepared:
        actual = prepare_prompts(
            [item.text], tokenizer, item_ids=[item.item_id], bos_policy=item.bos_policy
        )[0]
        if actual.token_ids != item.token_ids:
            raise SamplingEvidenceError(
                "engine tokenizer does not reproduce prepared prompt tokens"
            )
    stops = resolve_stop_ids(tokenizer, required_stop_tokens)
    copied_params = copy.deepcopy(params)
    requested = _parameters(copied_params, stops)
    expected_n = copied_params.n
    target = Path(output_dir).resolve()
    target.mkdir(mode=0o700, parents=True)  # Never overwrite an earlier generation or parser run.
    before = {
        "schema_version": SCHEMA,
        "created_at": schema.now(),
        "card": card,
        "native": native,
        "required_stop_ids": stops,
        "requested_parameters": requested,
        "inputs": [
            {
                "ordinal": item.ordinal,
                "item_id": item.item_id,
                "prompt_text_sha256": _sha(item.text.encode()),
                "prompt_token_ids": list(item.token_ids),
                "bos_policy": item.bos_policy,
            }
            for item in prepared
        ],
    }
    _write_json(target / "request.json", before)
    report = {
        "schema_version": SCHEMA,
        "status": "capture_failed",
        "returned_requests": 0,
        "returned_completions": 0,
        "returned_tokens": 0,
        "raw_durable": False,
        "request_sha256": schema.sha256_file(target / "request.json"),
        "scientific_validation": "not_performed",
    }
    call_started = None
    try:
        report["engine_call_started_at"] = schema.now()
        call_started = time.monotonic()
        outputs = _call_engine(
            llm, [{"prompt_token_ids": list(item.token_ids)} for item in prepared], copied_params
        )
        report["engine_call_seconds"] = time.monotonic() - call_started
        report["engine_call_returned_at"] = schema.now()
        digest = hashlib.sha256()
        with (target / "raw.jsonl").open("xb") as stream:
            for ordinal, output in enumerate(outputs):
                raw = _raw_request(output, ordinal)
                encoded = _bytes(raw)
                stream.write(encoded)
                digest.update(encoded)
                report["returned_requests"] += 1
                report["returned_completions"] += len(raw["completions"])
                report["returned_tokens"] += sum(
                    len(item["token_ids"]) for item in raw["completions"]
                )
            stream.flush()
            os.fsync(stream.fileno())
        report.update({"raw_sha256": digest.hexdigest(), "raw_durable": True})
        del outputs
        # Validate only after preserving ALL returned draw text/token/finish fields.
        if report["returned_requests"] != len(prepared):
            raise SamplingEvidenceError("returned request count differs from input count")
        request_ids = set()
        verified_digest = hashlib.sha256()
        with (target / "raw.jsonl").open("rb") as stream:
            for ordinal, line in enumerate(stream):
                verified_digest.update(line)
                raw = json.loads(line)
                if raw["ordinal"] != ordinal or ordinal >= len(prepared):
                    raise SamplingEvidenceError("raw ordinal/count changed during validation")
                if not isinstance(raw["request_id"], str) or raw["request_id"] in request_ids:
                    raise SamplingEvidenceError("native request IDs are missing/duplicated")
                request_ids.add(raw["request_id"])
                _validate_request(raw, prepared[ordinal], expected_n)
        if verified_digest.hexdigest() != report["raw_sha256"]:
            raise SamplingEvidenceError("raw data changed during validation")
        if _card_evidence(card_path) != card:
            raise SamplingEvidenceError("card/lock changed during generation")
        report["status"] = "captured"
        report["requested_identity_note"] = (
            "parameters are the submitted request, not a full engine-state audit"
        )
        _write_json(target / "capture.json", report)
        return report
    except BaseException as exc:
        if call_started is not None and "engine_call_seconds" not in report:
            report["failed_engine_call_seconds"] = time.monotonic() - call_started
        report["error"] = f"{type(exc).__name__}: {exc}"
        try:
            _write_json(target / "capture-failure.json", report)
        except BaseException as audit_error:  # noqa: BLE001 - preserve the original capture error/interruption
            if hasattr(exc, "add_note"):
                exc.add_note(f"sampling failure record also failed: {audit_error}")
        raise


def parse_recording(directory, parser):
    """Parse retained draws with explicit per-draw errors; never edit the raw file.

    parser receives (completion_text, input_metadata). It must not run inference.
    Its JSON result is developer evidence only; this function computes no accuracy.
    """
    directory = Path(directory).resolve()
    if not (directory / "capture.json").is_file():
        raise SamplingEvidenceError(
            "no completed capture; inspect capture-failure.json/raw evidence before parsing"
        )
    capture = json.loads((directory / "capture.json").read_text())
    request_bytes = (directory / "request.json").read_bytes()
    request = json.loads(request_bytes)
    raw_path = directory / "raw.jsonl"
    if (
        not isinstance(capture, dict)
        or not isinstance(request, dict)
        or capture.get("schema_version") != SCHEMA
        or request.get("schema_version") != SCHEMA
        or capture.get("status") != "captured"
        or capture.get("raw_durable") is not True
        or _sha(request_bytes) != capture.get("request_sha256")
        or schema.sha256_file(raw_path) != capture.get("raw_sha256")
    ):
        raise SamplingEvidenceError("no complete unchanged raw capture to parse")
    output = directory / f"parse-{uuid4().hex}.jsonl"
    parsed = errors = count = 0
    parser_path = (
        inspect.getsourcefile(parser)
        if inspect.isfunction(parser) or inspect.ismethod(parser)
        else None
    )
    parser_identity = {
        "name": getattr(parser, "__qualname__", type(parser).__name__),
        "module": getattr(parser, "__module__", type(parser).__module__),
        "source_path": parser_path,
        "source_sha256": schema.sha256_file(Path(parser_path))
        if parser_path and Path(parser_path).is_file()
        else None,
        "closure_and_dependencies": "not_independently_verified",
    }
    parse_started = time.monotonic()
    try:
        with output.open("xb") as stream, raw_path.open("rb") as raw_stream:
            digest = hashlib.sha256()
            for line in raw_stream:
                digest.update(line)
                raw = json.loads(line)
                metadata = request["inputs"][raw["ordinal"]]
                for completion in raw["completions"]:
                    record = {
                        "ordinal": raw["ordinal"],
                        "item_id": metadata["item_id"],
                        "completion_index": completion["index"],
                    }
                    try:
                        value = parser(completion["text"], copy.deepcopy(metadata))
                        record.update({"status": "parsed", "value": _plain(value)})
                        encoded = _bytes(record)
                        parsed += 1
                    except Exception as exc:  # noqa: BLE001 - preserve explicit per-draw parser failures
                        errors += 1
                        record.pop("value", None)
                        encoded = _bytes(
                            {
                                **record,
                                "status": "parser_error",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                    stream.write(encoded)
                    count += 1
            stream.flush()
            os.fsync(stream.fileno())
        if (
            digest.hexdigest() != capture["raw_sha256"]
            or schema.sha256_file(raw_path) != capture["raw_sha256"]
            or schema.sha256_file(directory / "request.json") != capture["request_sha256"]
            or count != capture["returned_completions"]
        ):
            raise SamplingEvidenceError(
                "raw/request evidence changed or counts differ during parsing; parse is unverified"
            )
    except BaseException as exc:
        try:
            _write_json(
                output.with_suffix(".failure.json"),
                {
                    "schema_version": SCHEMA,
                    "raw_sha256": capture["raw_sha256"],
                    "parse_path": str(output),
                    "status": "interrupted_or_failed",
                    "parser": parser_identity,
                    "count_written": count,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        except BaseException as audit_error:  # noqa: BLE001 - preserve the original parser interruption
            if hasattr(exc, "add_note"):
                exc.add_note(f"parser failure record also failed: {audit_error}")
        raise
    summary = {
        "schema_version": SCHEMA,
        "raw_sha256": capture["raw_sha256"],
        "parse_path": str(output),
        "parse_sha256": schema.sha256_file(output),
        "count": count,
        "parsed": parsed,
        "parser_errors": errors,
        "all_parsed": errors == 0,
        "official_score": False,
        "parser": parser_identity,
        "parse_seconds": time.monotonic() - parse_started,
    }
    _write_json(output.with_suffix(".summary.json"), summary)
    return summary
