"""High-confidence credential detection for published historical corpora.

The guard deliberately rejects instead of rewriting evidence.  Its public
error formatter exposes only a rule identifier, caller-supplied path, and
match count; matched bytes never enter logs or exceptions.

``rollout/validate_study_corpus.py`` carries a self-contained copy because the
C1 payload intentionally contains no :mod:`awm` package.  Keep the two rule
contracts and behaviours identical; focused tests enforce that invariant.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

RULESET_VERSION = "awm-raw-credential-guard-v1"

# These formats are sufficiently distinctive to reject wherever they occur.
# Patterns are ASCII byte expressions so arbitrary trajectory bytes can be
# inspected without lossy decoding or echoing a decode failure near a secret.
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


def credential_ruleset_contract() -> dict[str, object]:
    """Return a stable, secret-free contract used to prevent copy drift."""
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


def _normalise_value(value: bytes) -> str:
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
    normalised = _normalise_value(value)
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
    # Split CamelCase before normalising separators so accessToken and
    # clientSecret receive the same treatment as access_token/client_secret.
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
