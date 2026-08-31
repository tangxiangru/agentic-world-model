#!/usr/bin/env python3
"""Validate and attest a read-only historical input mounted into a study cell.

This file is copied into each study agent's PTB payload by ``rollout/setup.sh``.
It is intentionally standalone: C1 has no AWM runtime.  It prints one compact
JSON attestation and can atomically record the same object under the PTB task
directory, which is copied back with the rest of the cell artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

RAW_SCHEMA = "awm-prior-runs-v1"
CARD_SCHEMA = "awm-exp-card-corpus-v1"
RAW_FILES = ("solve_out.txt", "metrics.json", "time_taken.txt")
RAW_RUN_RE = re.compile(r"^(?P<bench>[^_]+)_(?P<model>.+)_(?P<cid>\d+)$")
RAW_README = (
    "Read-only copy of prior PostTrainBench runs for this task, built by "
    "tools/build_prior_runs.py. Start with INDEX.md.\n"
)
HEX40 = re.compile(r"[0-9a-fA-F]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")

# C1 receives this validator as a single file and intentionally has no AWM
# package.  This credential contract therefore mirrors awm/credential_guard.py
# in-place.  tests/test_credential_guard.py compares both the declaration and
# behaviour so either copy changing alone fails CI.
RULESET_VERSION = "awm-raw-credential-guard-v1"
DIRECT_RULE_SPECS: tuple[tuple[str, bytes, int, bytes], ...] = (
    (
        "private-key-block",
        rb"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----",
        0,
        b"-----BEGIN ",
    ),
    (
        "huggingface-token",
        rb"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{20,}\b",
        0,
        b"hf_",
    ),
    (
        "secret-key-token",
        rb"(?<![A-Za-z0-9])sk-(?:ant-|proj-|svcacct-)?[A-Za-z0-9_-]{20,}\b",
        0,
        b"sk-",
    ),
    (
        "google-oauth-token",
        rb"(?<![A-Za-z0-9])ya29\.[A-Za-z0-9._~+/-]{20,}",
        0,
        b"ya29.",
    ),
    (
        "github-token",
        rb"(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b",
        0,
        b"gh",
    ),
)
PLACEHOLDER_VALUES = frozenset(
    {
        "",
        "undefined",
        "null",
        "none",
        "nil",
        "unset",
        "not-set",
        "not_set",
        "redacted",
        "omitted",
        "<redacted>",
        "<omitted>",
        "<omitted-api-key>",
        "[redacted]",
        "[omitted]",
        "***redacted***",
        "n/a",
        "na",
    }
)
SENSITIVE_KEY_PARTS = frozenset({"token", "secret", "password", "passwd", "authorization"})
SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "access_token",
        "accesstoken",
        "refresh_token",
        "refreshtoken",
        "client_secret",
        "clientsecret",
        "private_key",
        "privatekey",
    }
)
_DIRECT_RULES = tuple(
    (rule_id, re.compile(pattern, flags), marker)
    for rule_id, pattern, flags, marker in DIRECT_RULE_SPECS
)
_SENSITIVE_KEY_PATTERN = (
    rb"(?:[A-Za-z][A-Za-z0-9_.-]{0,70})?"
    rb"(?:token|secret|password|passwd|authorization|api[_-]?key|private[_-]?key)"
)
_SENSITIVE_ENV_KEY_PATTERN = (
    rb"(?:[A-Za-z][A-Za-z0-9_]{0,70})?"
    rb"(?:TOKEN|SECRET|PASSWORD|PASSWD|AUTHORIZATION|API_KEY|PRIVATE_KEY)"
)
_ENV_ASSIGNMENT = re.compile(
    rb"(?m)(?<![A-Za-z0-9_])(?:export[ \t]+)?"
    rb"(?P<key>" + _SENSITIVE_ENV_KEY_PATTERN + rb")[ \t]*=[ \t]*"
    rb"(?P<value>\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s;,]+)",
    re.IGNORECASE,
)
_JSON_ASSIGNMENT = re.compile(
    rb'"(?P<key>' + _SENSITIVE_KEY_PATTERN + rb')"[ \t]*:[ \t]*'
    rb'"(?P<value>(?:\\.|[^"\\\r\n]){1,4096})"',
    re.IGNORECASE,
)
_FIELD_ASSIGNMENT = re.compile(
    rb"^[ \t]*(?P<key>" + _SENSITIVE_KEY_PATTERN + rb")[ \t]*:[ \t]*"
    rb"(?P<value>[^\r\n]{1,4096})$",
    re.IGNORECASE | re.MULTILINE,
)
_BEARER = re.compile(
    rb"(?i)\bauthorization[ \t]*[\"']?[ \t]*[:=][ \t]*[\"']?"
    rb"bearer[ \t]+(?P<value>[A-Za-z0-9._~+/-]{20,}={0,2})"
)


class ValidationError(RuntimeError):
    """The mounted corpus does not equal its declared immutable inventory."""


def credential_ruleset_contract() -> dict[str, object]:
    """Return the standalone scanner's stable, secret-free contract."""
    return {
        "version": RULESET_VERSION,
        "direct": [
            {
                "rule_id": rule_id,
                "pattern": pattern.decode("ascii"),
                "flags": flags,
                "marker": marker.decode("ascii"),
            }
            for rule_id, pattern, flags, marker in DIRECT_RULE_SPECS
        ],
        "placeholders": sorted(PLACEHOLDER_VALUES),
        "sensitive_key_parts": sorted(SENSITIVE_KEY_PARTS),
        "sensitive_keys": sorted(SENSITIVE_KEYS),
        "context_patterns": {
            "env_assignment": _ENV_ASSIGNMENT.pattern.decode("ascii"),
            "json_assignment": _JSON_ASSIGNMENT.pattern.decode("ascii"),
            "field_assignment": _FIELD_ASSIGNMENT.pattern.decode("ascii"),
            "bearer": _BEARER.pattern.decode("ascii"),
            "sensitive_key": _SENSITIVE_KEY_PATTERN.decode("ascii"),
            "sensitive_env_key": _SENSITIVE_ENV_KEY_PATTERN.decode("ascii"),
        },
    }


def _normalise_credential_value(value: bytes) -> str:
    text = value.decode("ascii", errors="ignore").strip()
    for escaped_suffix in (r"\n", r"\r", r"\t"):
        while text.endswith(escaped_suffix):
            text = text[: -len(escaped_suffix)].rstrip()
    changed = True
    while changed:
        changed = False
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"`":
            text = text[1:-1].strip()
            changed = True
        elif len(text) >= 4 and text[:2] in (r"\"", r"\'") and text[-2:] == text[:2]:
            text = text[2:-2].strip()
            changed = True
    return text.rstrip(",;").strip().lower()


def _is_material_secret(value: bytes, *, key: bytes | None = None) -> bool:
    normalised = _normalise_credential_value(value)
    if key is not None:
        normalised_key = key.decode("ascii", errors="ignore").strip().lower()
        if normalised in {
            normalised_key,
            f"${normalised_key}",
            "${" + normalised_key + "}",
        }:
            return False
    return normalised not in PLACEHOLDER_VALUES and len(normalised) >= 8


def _is_sensitive_key(value: bytes) -> bool:
    text = value.decode("ascii", errors="ignore")
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    normalised = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if normalised in SENSITIVE_KEYS or any(
        normalised.endswith(f"_{key}") for key in SENSITIVE_KEYS
    ):
        return True
    return any(part in SENSITIVE_KEY_PARTS for part in normalised.split("_"))


def scan_credential_bytes(data: bytes, *, path: str) -> list[dict[str, object]]:
    """Return secret-free finding records for one file's bytes."""
    counts: Counter[str] = Counter()
    for rule_id, pattern, marker in _DIRECT_RULES:
        if marker in data:
            counts[rule_id] += sum(1 for _match in pattern.finditer(data))
    for match in _ENV_ASSIGNMENT.finditer(data):
        if _is_sensitive_key(match.group("key")) and _is_material_secret(
            match.group("value"), key=match.group("key")
        ):
            counts["secret-env-assignment"] += 1
    for pattern, rule_id in (
        (_JSON_ASSIGNMENT, "secret-json-field"),
        (_FIELD_ASSIGNMENT, "secret-text-field"),
    ):
        for match in pattern.finditer(data):
            if _is_sensitive_key(match.group("key")) and _is_material_secret(match.group("value")):
                counts[rule_id] += 1
    if b"Bearer" in data or b"bearer" in data or b"BEARER" in data:
        for match in _BEARER.finditer(data):
            if _is_material_secret(match.group("value")):
                counts["authorization-bearer"] += 1
    return [
        {"path": path, "rule_id": rule_id, "count": count}
        for rule_id, count in sorted(counts.items())
        if count
    ]


def scan_credential_files(files: Iterable[tuple[Path, str]]) -> list[dict[str, object]]:
    """Scan ``(filesystem path, report path)`` pairs without decoding them."""
    findings: list[dict[str, object]] = []
    for file_path, report_path in files:
        findings.extend(scan_credential_bytes(file_path.read_bytes(), path=report_path))
    return sorted(findings, key=lambda row: (str(row["path"]), str(row["rule_id"])))


def format_credential_rejection(findings: Iterable[dict[str, object]]) -> str:
    """Render findings without ever interpolating matched credential bytes."""
    rows = list(findings)
    details = "; ".join(
        "rule_id={rule} path={path} count={count}".format(
            rule=row["rule_id"],
            path=json.dumps(str(row["path"]), ensure_ascii=True),
            count=row["count"],
        )
        for row in rows
    )
    return f"raw corpus credential guard rejected {len(rows)} finding group(s): {details}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def has_symlink_component(root: Path, relative: PurePosixPath) -> bool:
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def safe_run(value: Any) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"invalid run path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 2 or any(p in ("", ".", "..") for p in path.parts):
        raise ValidationError(f"invalid run path: {value!r}")
    return value


def read_json(path: Path) -> Any:
    if not regular(path):
        raise ValidationError(f"required regular metadata file is missing: {path}")
    try:
        return json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid JSON in {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not regular(path):
        raise ValidationError(f"required regular metadata file is missing: {path}")
    rows: list[dict[str, Any]] = []
    try:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValidationError(f"{path}:{number} is not a JSON object")
            rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid JSONL in {path}: {exc}") from exc
    return rows


def parse_sides(raw: str) -> tuple[str, ...]:
    sides = tuple(raw.split(","))
    if sides not in (("train",), ("train", "test")):
        raise ValidationError("scope must be exactly train or train,test")
    return sides


def require_readonly(root: Path) -> None:
    try:
        readonly = bool(os.statvfs(root).f_flag & os.ST_RDONLY)
    except OSError as exc:
        raise ValidationError(f"cannot inspect mount flags for {root}: {exc}") from exc
    if not readonly:
        raise ValidationError(f"historical corpus mount is not read-only: {root}")


def validate_raw(root: Path, sides: tuple[str, ...], expected_sha256: str) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise ValidationError(f"raw corpus root is missing, linked, or not a directory: {root}")
    if not HEX64.fullmatch(expected_sha256):
        raise ValidationError("expected raw manifest SHA-256 must be 64 lowercase hex characters")
    manifest_path = root / "corpus-manifest.json"
    actual_sha256 = sha256_file(manifest_path) if regular(manifest_path) else ""
    if actual_sha256 != expected_sha256:
        raise ValidationError(
            f"raw corpus manifest SHA-256 is {actual_sha256 or '<missing>'}, expected {expected_sha256}"
        )
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != RAW_SCHEMA:
        raise ValidationError(f"invalid raw corpus schema in {manifest_path}")
    split = manifest.get("split")
    dataset = manifest.get("dataset")
    entries = manifest.get("runs")
    if (
        not isinstance(split, dict)
        or not isinstance(split.get("id"), str)
        or split.get("sides") != list(sides)
        or not isinstance(dataset, dict)
        or not isinstance(dataset.get("revision"), str)
        or not HEX40.fullmatch(dataset["revision"])
        or manifest.get("file_scope") != list(RAW_FILES)
        or not isinstance(entries, list)
        or manifest.get("run_count") != len(entries)
    ):
        raise ValidationError(f"raw corpus manifest provenance/scope is malformed: {manifest_path}")

    attested: dict[str, str] = {}
    credential_files: list[tuple[Path, str]] = []
    for number, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValidationError(f"raw manifest run {number} is not an object")
        run = safe_run(entry.get("run"))
        side = entry.get("side")
        if side not in sides or run in attested or not isinstance(entry.get("files"), dict):
            raise ValidationError(f"raw manifest run is duplicate or out of scope: {run}")
        files = entry["files"]
        if set(files) != set(RAW_FILES):
            raise ValidationError(f"raw manifest file scope mismatch: {run}")
        run_dir = root / run
        if has_symlink_component(root, PurePosixPath(run)) or not run_dir.is_dir():
            raise ValidationError(f"raw run directory is missing or linked: {run_dir}")
        for name in RAW_FILES:
            path = run_dir / name
            record = files[name]
            if (
                not regular(path)
                or not isinstance(record, dict)
                or isinstance(record.get("bytes"), bool)
                or not isinstance(record.get("bytes"), int)
                or not isinstance(record.get("sha256"), str)
                or not HEX64.fullmatch(record["sha256"])
                or path.stat().st_size != record["bytes"]
                or sha256_file(path) != record["sha256"]
            ):
                raise ValidationError(f"raw corpus hash/size mismatch: {path}")
            credential_files.append((path, f"{run}/{name}"))
        attested[run] = str(side)

    findings = scan_credential_files(credential_files)
    if findings:
        raise ValidationError(format_credential_rejection(findings))

    index_rows = read_jsonl(root / "index.jsonl")
    indexed: dict[str, str] = {}
    for row in index_rows:
        run = safe_run(row.get("run"))
        side = row.get("side")
        if side not in sides or run in indexed:
            raise ValidationError(f"raw index run is duplicate or out of scope: {run}")
        indexed[run] = str(side)
    if indexed != attested:
        raise ValidationError("raw corpus manifest and index do not name identical runs/sides")

    # Root metadata is a deterministic view of the manifest-attested run
    # bytes. Compare its complete canonical content, so a changed score, model,
    # path, ordering, or prose cannot bypass the pinned manifest digest.
    expected_rows: list[dict[str, Any]] = []
    for run, side in attested.items():
        config, run_name = run.split("/", 1)
        match = RAW_RUN_RE.fullmatch(run_name)
        if match is None:
            raise ValidationError(f"raw run name is not canonical PostTrainBench form: {run}")
        metrics = read_json(root / run / "metrics.json")
        accuracy = metrics.get("accuracy") if isinstance(metrics, dict) else None
        if (
            isinstance(accuracy, bool)
            or not isinstance(accuracy, (int, float))
            or not math.isfinite(float(accuracy))
        ):
            raise ValidationError(f"raw run has invalid official accuracy: {run}")
        try:
            time_taken = (root / run / "time_taken.txt").read_text().strip()
        except (OSError, UnicodeDecodeError) as exc:
            raise ValidationError(f"raw run has invalid time_taken.txt: {run}: {exc}") from exc
        expected_rows.append(
            {
                "run": run,
                "agent_config": config,
                "run_name": run_name,
                "side": side,
                "base_model": match.group("model").replace("_", "/", 1),
                "accuracy": accuracy,
                "time_taken": time_taken,
                "has_trace": True,
                "trace_bytes": (root / run / "solve_out.txt").stat().st_size,
                "path": f"/home/ben/prior_runs/{run}",
            }
        )
    expected_rows.sort(key=lambda row: (-float(row["accuracy"]), row["run"]))
    expected_index_jsonl = "".join(json.dumps(row) + "\n" for row in expected_rows)
    try:
        actual_index_jsonl = (root / "index.jsonl").read_text()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValidationError(f"invalid raw corpus index.jsonl: {exc}") from exc
    if actual_index_jsonl != expected_index_jsonl:
        raise ValidationError("raw corpus index.jsonl is not the canonical manifest-derived index")
    index_lines = [
        "# Prior runs",
        "",
        (
            f"{len(expected_rows)} previous attempts at this task by autonomous agents, one directory each, "
            "laid out as `<agent config>/<run>/`. Each holds `solve_out.txt` (the agent's complete "
            "session trace), `metrics.json` (official accuracy), and `time_taken.txt`. "
            "No optional run artifacts or `task/` workspace snapshots are exposed."
        ),
        "",
        "Sorted by official accuracy, best first.",
        "",
        "| accuracy | base model | agent config | time | trace | path |",
        "|---:|---|---|---|---:|---|",
    ]
    for row in expected_rows:
        index_lines.append(
            f"| {row['accuracy']:.3f} | {row['base_model']} | {row['agent_config']} | "
            f"{row['time_taken'] or '—'} | {row['trace_bytes'] // 1024} KB | `{row['path']}` |"
        )
    expected_overview = "\n".join(index_lines) + "\n"
    try:
        actual_overview = (root / "INDEX.md").read_text()
        actual_readme = (root / "README.md").read_text()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValidationError(f"invalid raw corpus metadata: {exc}") from exc
    if actual_overview != expected_overview:
        raise ValidationError("raw corpus INDEX.md is not the canonical manifest-derived overview")
    if actual_readme != RAW_README:
        raise ValidationError("raw corpus README.md is not the canonical generated README")
    actual_runs = {
        path.parent.relative_to(root).as_posix()
        for path in root.glob("*/*/solve_out.txt")
        if regular(path)
    }
    if actual_runs != set(attested):
        raise ValidationError("raw corpus filesystem inventory does not equal its manifest")
    allowed_metadata = {"INDEX.md", "README.md", "index.jsonl", "corpus-manifest.json"}
    allowed_run_files = set(RAW_FILES)
    expected_dirs = {root}
    for run in attested:
        run_path = root / run
        expected_dirs.update((run_path.parent, run_path))
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValidationError(f"raw corpus contains a symlink: {path}")
        if path.is_dir() and path not in expected_dirs:
            raise ValidationError(f"raw corpus contains an unexpected directory: {path}")
        if path.is_file():
            rel = path.relative_to(root)
            allowed = (
                (len(rel.parts) == 1 and rel.name in allowed_metadata)
                or (len(rel.parts) == 3 and rel.name in allowed_run_files)
            )
            if not allowed:
                raise ValidationError(f"raw corpus contains an unexpected file: {path}")
    return {
        "kind": "raw",
        "scope": list(sides),
        "manifest_sha256": actual_sha256,
        "schema_version": RAW_SCHEMA,
        "split_id": split["id"],
        "dataset_revision": dataset["revision"].lower(),
        "run_count": len(attested),
        "credential_scan": {
            "ruleset_version": RULESET_VERSION,
            "file_count": len(credential_files),
            "finding_group_count": 0,
        },
        "index_sha256": sha256_file(root / "index.jsonl"),
        "overview_sha256": sha256_file(root / "INDEX.md"),
        "readme_sha256": sha256_file(root / "README.md"),
    }


def safe_card_path(value: Any) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"invalid card path: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or len(path.parts) != 2
        or not path.parts[0].startswith("r-")
        or not path.name.startswith("exp-")
        or path.suffix != ".yaml"
    ):
        raise ValidationError(f"invalid card path: {value!r}")
    return value


def validate_cards(
    root: Path, sides: tuple[str, ...], expected_sha256: str | None = None
) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise ValidationError(f"card memory root is missing, linked, or not a directory: {root}")
    corpus_root = root / "corpus"
    if corpus_root.is_symlink() or not corpus_root.is_dir():
        raise ValidationError(f"card corpus directory is missing or linked: {corpus_root}")
    actual_sides: set[str] = set()
    for path in corpus_root.iterdir():
        if path.is_symlink() or not path.is_dir():
            raise ValidationError(f"card corpus contains an unexpected side entry: {path}")
        actual_sides.add(path.name)
    if actual_sides != set(sides):
        raise ValidationError(
            f"card corpus sides {sorted(actual_sides)} do not exactly match {list(sides)}"
        )
    side_records: list[dict[str, Any]] = []
    total_cards = 0
    for side in sides:
        side_root = root / "corpus" / side
        if has_symlink_component(root, PurePosixPath("corpus") / side) or not side_root.is_dir():
            raise ValidationError(f"card corpus side is missing or linked: {side_root}")
        manifest_path = side_root / "manifest.json"
        cards_manifest_path = side_root / "manifest.jsonl"
        manifest = read_json(manifest_path)
        rows = read_jsonl(cards_manifest_path)
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != CARD_SCHEMA
            or manifest.get("side") != side
            or manifest.get("card_count") != len(rows)
        ):
            raise ValidationError(f"card corpus manifest is malformed: {manifest_path}")
        indexed: dict[str, str] = {}
        for row in rows:
            rel = safe_card_path(row.get("path"))
            digest = row.get("sha256")
            if rel in indexed or not isinstance(digest, str) or not HEX64.fullmatch(digest):
                raise ValidationError(f"duplicate or invalid card manifest entry: {rel}")
            path = side_root / rel
            if has_symlink_component(side_root, PurePosixPath(rel)) or not regular(path) or sha256_file(path) != digest:
                raise ValidationError(f"card corpus hash mismatch: {path}")
            indexed[rel] = digest
        actual: set[str] = set()
        expected_dirs = {side_root}
        expected_dirs.update((side_root / rel).parent for rel in indexed)
        for path in side_root.rglob("*"):
            if path.is_symlink():
                raise ValidationError(f"card corpus contains a symlink: {path}")
            if path.is_dir():
                if path not in expected_dirs:
                    raise ValidationError(
                        f"card corpus filesystem inventory mismatch: unexpected directory {path}"
                    )
                continue
            if not path.is_file():
                raise ValidationError(
                    f"card corpus filesystem inventory mismatch: unexpected entry {path}"
                )
            rel = path.relative_to(side_root).as_posix()
            if rel in ("manifest.json", "manifest.jsonl"):
                continue
            actual.add(rel)
        if actual != set(indexed):
            raise ValidationError(f"card corpus filesystem inventory mismatch: {side_root}")
        total_cards += len(rows)
        side_records.append(
            {
                "side": side,
                "manifest_sha256": sha256_file(manifest_path),
                "cards_manifest_sha256": sha256_file(cards_manifest_path),
                "card_count": len(rows),
                "card_bearing_run_count": manifest.get("card_bearing_run_count"),
                "expected_run_count": manifest.get("expected_run_count"),
                "missing_run_count": manifest.get("missing_run_count"),
            }
        )
    structured_rows = read_jsonl(root / "structured" / "cards.jsonl")
    structured_sides = {
        str((row.get("provenance") or {}).get("split_side")) for row in structured_rows
    }
    if structured_sides != set(sides) or len(structured_rows) != total_cards:
        raise ValidationError(
            "structured card index does not exactly match selected corpus sides/count"
        )
    combined = hashlib.sha256(
        json.dumps(side_records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if expected_sha256 is not None:
        if not HEX64.fullmatch(expected_sha256):
            raise ValidationError(
                "expected card manifest SHA-256 must be 64 lowercase hex characters"
            )
        if combined != expected_sha256:
            raise ValidationError(
                f"combined card manifest SHA-256 is {combined}, expected {expected_sha256}"
            )
    return {
        "kind": "cards",
        "scope": list(sides),
        "manifest_sha256": combined,
        "schema_version": CARD_SCHEMA,
        "card_count": total_cards,
        "side_manifests": side_records,
    }


def atomic_record(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(name, path)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("raw", "cards"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--sides", required=True)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--require-readonly", action="store_true")
    parser.add_argument("--condition", choices=("c1", "c2", "c3"))
    parser.add_argument("--repetition", choices=(1, 2), type=int)
    parser.add_argument("--study-mode", choices=("production", "smoke"))
    parser.add_argument("--num-hours", choices=(1, 10), type=int)
    parser.add_argument("--ptb-commit")
    parser.add_argument("--harness-commit")
    parser.add_argument("--ptb-surface-manifest-sha256")
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    try:
        sides = parse_sides(args.sides)
        root = Path(os.path.abspath(args.root))
        if args.require_readonly:
            require_readonly(root)
        if args.kind == "raw":
            if not args.expected_manifest_sha256:
                raise ValidationError("raw validation requires --expected-manifest-sha256")
            result = validate_raw(root, sides, args.expected_manifest_sha256)
        else:
            result = validate_cards(root, sides, args.expected_manifest_sha256)
        result["validator_sha256"] = sha256_file(Path(__file__).resolve())
        if args.condition:
            result["condition"] = args.condition
        if args.repetition:
            result["repetition"] = args.repetition
        if args.study_mode or args.num_hours:
            if (args.study_mode, args.num_hours) not in (("production", 10), ("smoke", 1)):
                raise ValidationError(
                    "study mode/hours must be exactly production/10 or smoke/1"
                )
            result["study_mode"] = args.study_mode
            result["num_hours"] = args.num_hours
        for label, value in (
            ("ptb_commit", args.ptb_commit),
            ("harness_commit", args.harness_commit),
        ):
            if value is not None:
                if not HEX40.fullmatch(value):
                    raise ValidationError(f"{label.replace('_', ' ')} must be a full 40-hex commit")
                result[label] = value.lower()
        if args.ptb_surface_manifest_sha256 is not None:
            if not HEX64.fullmatch(args.ptb_surface_manifest_sha256):
                raise ValidationError("PTB surface manifest SHA-256 must be 64 lowercase hex")
            result["ptb_surface_manifest_sha256"] = args.ptb_surface_manifest_sha256
        if args.record and (not args.ptb_commit or not args.harness_commit):
            raise ValidationError("recorded cell input requires PTB and harness commits")
        if args.record:
            atomic_record(args.record, result)
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    except (OSError, ValidationError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
