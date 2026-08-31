"""High-confidence credential detection for published historical corpora.

The guard deliberately rejects instead of rewriting evidence.  Its public
error formatter exposes only a rule identifier, caller-supplied path, and
match count; matched bytes never enter logs or exceptions.

``rollout/validate_study_corpus.py`` carries a self-contained copy because the
C1 payload intentionally contains no :mod:`awm` package.  Keep the two rule
contracts and behaviours identical; focused tests enforce that invariant.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

RULESET_VERSION = "awm-raw-credential-guard-v3"

# These formats are sufficiently distinctive to reject wherever they occur.
# Patterns are ASCII byte expressions so arbitrary trajectory bytes can be
# inspected without lossy decoding or echoing a decode failure near a secret.
DIRECT_RULE_SPECS: tuple[tuple[str, bytes, int, bytes], ...] = (
    (
        "private-key-block",
        (
            rb"-----BEGIN (?:(?:RSA|EC|DSA|OPENSSH|ENCRYPTED) PRIVATE KEY|"
            rb"PGP PRIVATE KEY BLOCK|PRIVATE KEY)-----"
        ),
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
        rb"(?<![A-Za-z0-9])sk-(?:ant-|proj-|svcacct-|admin-)?[A-Za-z0-9_-]{20,}\b",
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
        b"g",
    ),
    (
        "aws-access-key-id",
        rb"(?<![A-Z0-9])AKIA[A-Z0-9]{16}(?![A-Z0-9])",
        0,
        b"AKIA",
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
        "auth_token",
        "authtoken",
        "aws_access_key_id",
        "aws_secret_access_key",
        "access_key_id",
        "accesskeyid",
        "access_key",
        "accesskey",
        "api_key_value",
        "private_key_value",
        "access_key_value",
    }
)

_DIRECT_RULES = tuple(
    (rule_id, re.compile(pattern, flags), marker)
    for rule_id, pattern, flags, marker in DIRECT_RULE_SPECS
)
_KEY_IDENTIFIER_PATTERN = (
    rb"(?=[A-Za-z0-9_.-]{0,100}(?:token|secret|password|passwd|"
    rb"authorization|api[_-]?key|private[_-]?key|access[_-]?key)"
    rb"[A-Za-z0-9_.-]{0,100}(?![A-Za-z0-9_.-]))"
    rb"[A-Za-z][A-Za-z0-9_.-]{0,100}(?![A-Za-z0-9_.-])"
)
_ENV_KEY_IDENTIFIER_PATTERN = (
    rb"(?=[A-Z0-9_]{0,100}(?:TOKEN|SECRET|PASSWORD|PASSWD|"
    rb"AUTHORIZATION|API_KEY|PRIVATE_KEY|ACCESS_KEY)"
    rb"[A-Z0-9_]{0,100}(?![A-Z0-9_]))"
    rb"[A-Z][A-Z0-9_]{0,100}(?![A-Z0-9_])"
)
CASE_INSENSITIVE_ENV_KEYS = (
    "anthropic_api_key",
    "claude_code_oauth_token",
    "hf_token",
    "huggingface_token",
    "openai_api_key",
    "sglang_api_key",
    "vllm_api_key",
)
KEY_SPECIFIC_PLACEHOLDERS = {
    "api_key": frozenset({"inspectai"}),
    "vllm_api_key": frozenset({"inspectai"}),
}
ANGLE_PLACEHOLDER_WORDS = frozenset(
    {"dummy", "example", "insert", "omitted", "placeholder", "redacted", "replace", "sample", "your"}
)
NON_CREDENTIAL_TOKEN_KEYS = frozenset(
    {
        "pad_token",
        "eos_token",
        "bos_token",
        "boi_token",
        "eoi_token",
        "image_token",
        "unk_token",
        "sep_token",
        "cls_token",
        "mask_token",
        "space_token",
        "num_token_re",
        "response_stop_token",
        "stop_token",
        "token_budget",
    }
)
LOCAL_API_SENTINELS = frozenset(
    sentinel for sentinels in KEY_SPECIFIC_PLACEHOLDERS.values() for sentinel in sentinels
)
CLI_INLINE_SENTINEL_TERMINAL_PUNCTUATION = b"."
OPENAI_SECRET_PATTERN = r"sk-(?:(?:proj|svcacct|admin)-[a-z0-9_-]{20,}|[a-z0-9]{20,})"
_OPENAI_SECRET = re.compile(OPENAI_SECRET_PATTERN)
_CASE_INSENSITIVE_ENV_KEY_PATTERN = (
    b"(?:"
    + b"|".join(re.escape(key.encode("ascii")) for key in CASE_INSENSITIVE_ENV_KEYS)
    + b")"
)
_ASSIGNMENT_BOUNDARY = rb"(?:(?<![A-Za-z0-9_])|(?<=\\n)|(?<=\\r))"
_ESCAPED_QUOTED_VALUE = (
    rb'(?:\\+"[^"\r\n]*?\\+"|'
    rb"\\+'[^'\r\n]*?\\+')"
)
_CODE_QUOTED_IDENTIFIER = (
    rb'(?:"[A-Za-z_][A-Za-z0-9_]*"|\'[A-Za-z_][A-Za-z0-9_]*\'|'
    rb'\\+"[A-Za-z_][A-Za-z0-9_]*\\+"|\\+\'[A-Za-z_][A-Za-z0-9_]*\\+\')'
)
_EXACT_CODE_REFERENCE_VALUE = (
    rb"(?:os[ \t]*\.[ \t]*environ[ \t]*\[[ \t]*"
    + _CODE_QUOTED_IDENTIFIER
    + rb"[ \t]*\]|os[ \t]*\.[ \t]*getenv[ \t]*\([ \t]*"
    + _CODE_QUOTED_IDENTIFIER
    + rb"[ \t]*\))(?=[ \t]*(?:\r?\n|\\+[nrt]|;|$))"
)
_UNQUOTED_VALUE = rb"[^\s;,\"'\\]+"
_BACKSLASH_UNQUOTED_VALUE = rb"\\+(?![nrt\"'])[^\s;,]*"
_ENCODED_LINE_END = rb"\\+[nrt]"
_JSON_KEY_TOKEN = (
    rb"(?:(?P<raw_key_quote>[\"'])[A-Za-z_][A-Za-z0-9_.-]{0,80}"
    rb"(?P=raw_key_quote)|(?P<escaped_key_slashes>\\+)"
    rb"(?P<escaped_key_quote>[\"'])[A-Za-z_][A-Za-z0-9_.-]{0,80}"
    rb"(?P=escaped_key_slashes)(?P=escaped_key_quote))"
)
_JSON_QUOTED_TOKEN = (
    rb"(?:(?P<raw_value_quote>[\"'])(?P<raw_value>"
    rb"(?:(?!(?P=raw_value_quote))(?:\\[^\r\n]|[^\\\r\n])){0,4096})"
    rb"(?P=raw_value_quote)|(?P<escaped_value_slashes>\\+)"
    rb"(?P<escaped_value_quote>[\"'])(?P<escaped_value>"
    rb"(?:(?!(?<!\\)(?P=escaped_value_slashes)(?P=escaped_value_quote))"
    rb"[^\r\n]){0,4096}?)(?<!\\)(?P=escaped_value_slashes)"
    rb"(?P=escaped_value_quote))"
)
_JSON_PRIMITIVE_TOKEN = (
    rb"(?:" + _JSON_QUOTED_TOKEN + rb"|true|false|null|"
    rb"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?|\{\}|\[\])"
)
_JSON_COMPLETE_MEMBER = (
    rb",[ \t]*" + _JSON_KEY_TOKEN + rb"[ \t]*:[ \t]*" + _JSON_PRIMITIVE_TOKEN
)
_JSON_MEMBER_CONTINUATION = (
    rb"(?:" + _JSON_COMPLETE_MEMBER + rb")+[ \t]*(?:[}\]][ \t]*)*"
    rb"(?=\\+[nrt]|\r?\n|$)"
)
_SERIALIZED_VALUE_END = (
    rb"(?:\\+)?[\"'](?:[}\]](?=\\+[nrt]|\r?\n|$)|"
    + _JSON_MEMBER_CONTINUATION
    + rb")"
)
_JSON_CONTAINER_END = rb"[}\]](?=\\+[nrt]|\r?\n|$)"
_ENV_ASSIGNMENT = re.compile(
    rb"(?m)" + _ASSIGNMENT_BOUNDARY + rb"(?:export[ \t]+)?"
    rb"(?P<key>" + _ENV_KEY_IDENTIFIER_PATTERN + rb")[ \t]*=[ \t]*"
    rb"(?P<value>\"[^\"\r\n]*\"|'[^'\r\n]*'|"
    + _EXACT_CODE_REFERENCE_VALUE
    + rb"|"
    + _ESCAPED_QUOTED_VALUE
    + rb"|"
    + _UNQUOTED_VALUE
    + rb"|"
    + _BACKSLASH_UNQUOTED_VALUE
    + rb")"
)
_CASE_INSENSITIVE_ENV_ASSIGNMENT = re.compile(
    rb"(?m)" + _ASSIGNMENT_BOUNDARY + rb"(?:export[ \t]+)?"
    rb"(?P<key>" + _CASE_INSENSITIVE_ENV_KEY_PATTERN + rb")[ \t]*=[ \t]*"
    rb"(?P<value>\"[^\"\r\n]*\"|'[^'\r\n]*'|"
    + _EXACT_CODE_REFERENCE_VALUE
    + rb"|"
    + _ESCAPED_QUOTED_VALUE
    + rb"|"
    + _UNQUOTED_VALUE
    + rb"|"
    + _BACKSLASH_UNQUOTED_VALUE
    + rb")",
    re.IGNORECASE,
)
_CLI_ASSIGNMENT = re.compile(
    rb"(?<![A-Za-z0-9_-])--(?P<key>" + _KEY_IDENTIFIER_PATTERN + rb")"
    rb"(?:[ \t]*=[ \t]*|[ \t]+)(?P<value>\"[^\"\r\n]*\"|'[^'\r\n]*'|"
    + _EXACT_CODE_REFERENCE_VALUE
    + rb"|"
    + _ESCAPED_QUOTED_VALUE
    + rb"|"
    + _UNQUOTED_VALUE
    + rb"|"
    + _BACKSLASH_UNQUOTED_VALUE
    + rb")",
    re.IGNORECASE,
)
_JSON_ASSIGNMENT = re.compile(
    rb'"(?P<key>' + _KEY_IDENTIFIER_PATTERN + rb')"[ \t]*:[ \t]*'
    rb'"(?P<value>(?:\\.|[^"\\\r\n]){1,4096})"',
    re.IGNORECASE,
)
_ESCAPED_JSON_ASSIGNMENT = re.compile(
    rb'\\+"(?P<key>'
    + _KEY_IDENTIFIER_PATTERN
    + rb')\\+"[ \t]*:[ \t]*\\+"(?P<value>[^"\r\n]{1,4096}?)\\+"',
    re.IGNORECASE,
)
_REPR_ASSIGNMENT = re.compile(
    rb"'(?P<key>"
    + _KEY_IDENTIFIER_PATTERN
    + rb")'[ \t]*:[ \t]*(?P<value_quote>[\"'])(?P<value>"
    rb"(?:(?!(?P=value_quote))(?:\\.|[^\r\n])){1,4096})(?P=value_quote)",
    re.IGNORECASE,
)
_ESCAPED_REPR_ASSIGNMENT = re.compile(
    rb"\\+'(?P<key>"
    + _KEY_IDENTIFIER_PATTERN
    + rb")\\+'[ \t]*:[ \t]*\\+'(?P<value>[^'\r\n]{1,4096}?)\\+'",
    re.IGNORECASE,
)
_FIELD_ASSIGNMENT = re.compile(
    rb"^[ \t]*(?P<key>" + _KEY_IDENTIFIER_PATTERN + rb")[ \t]*:[ \t]*"
    rb"(?P<value>[^\r\n]{1,4096})$",
    re.IGNORECASE | re.MULTILINE,
)
_BEARER = re.compile(
    rb"(?i)\bauthorization[ \t]*[\"']?[ \t]*[:,=][ \t]*[\"']?"
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
        "case_insensitive_env_keys": list(CASE_INSENSITIVE_ENV_KEYS),
        "key_specific_placeholders": {
            key: sorted(values) for key, values in sorted(KEY_SPECIFIC_PLACEHOLDERS.items())
        },
        "angle_placeholder_words": sorted(ANGLE_PLACEHOLDER_WORDS),
        "noncredential_token_keys": sorted(NON_CREDENTIAL_TOKEN_KEYS),
        "local_api_sentinels": sorted(LOCAL_API_SENTINELS),
        "cli_inline_sentinel_terminal_punctuation": (
            CLI_INLINE_SENTINEL_TERMINAL_PUNCTUATION.decode("ascii")
        ),
        "openai_secret_pattern": OPENAI_SECRET_PATTERN,
        "context_patterns": {
            "env_assignment": _ENV_ASSIGNMENT.pattern.decode("ascii"),
            "case_insensitive_env_assignment": (
                _CASE_INSENSITIVE_ENV_ASSIGNMENT.pattern.decode("ascii")
            ),
            "cli_assignment": _CLI_ASSIGNMENT.pattern.decode("ascii"),
            "json_assignment": _JSON_ASSIGNMENT.pattern.decode("ascii"),
            "escaped_json_assignment": _ESCAPED_JSON_ASSIGNMENT.pattern.decode("ascii"),
            "repr_assignment": _REPR_ASSIGNMENT.pattern.decode("ascii"),
            "escaped_repr_assignment": _ESCAPED_REPR_ASSIGNMENT.pattern.decode("ascii"),
            "field_assignment": _FIELD_ASSIGNMENT.pattern.decode("ascii"),
            "bearer": _BEARER.pattern.decode("ascii"),
            "key_identifier": _KEY_IDENTIFIER_PATTERN.decode("ascii"),
            "env_key_identifier": _ENV_KEY_IDENTIFIER_PATTERN.decode("ascii"),
            "encoded_line_end": _ENCODED_LINE_END.decode("ascii"),
            "json_key_token": _JSON_KEY_TOKEN.decode("ascii"),
            "json_quoted_token": _JSON_QUOTED_TOKEN.decode("ascii"),
            "json_primitive_token": _JSON_PRIMITIVE_TOKEN.decode("ascii"),
            "json_complete_member": _JSON_COMPLETE_MEMBER.decode("ascii"),
            "serialized_value_end": _SERIALIZED_VALUE_END.decode("ascii"),
            "json_container_end": _JSON_CONTAINER_END.decode("ascii"),
            "json_member_continuation": _JSON_MEMBER_CONTINUATION.decode("ascii"),
        },
    }


def _normalise_value(value: bytes) -> str:
    # Latin-1 is a one-byte mapping: arbitrary bytes remain present and cannot
    # become a short/empty placeholder through lossy decoding.
    text = value.decode("latin-1").strip()
    # JSON/event streams may spell the provider separator in ``sk-...`` as a
    # Unicode escape.  Decode only this ASCII dash form; direct byte-pattern
    # rules still handle ordinary provider tokens before contextual parsing.
    text = re.sub(r"\\+u002d", "-", text, flags=re.IGNORECASE)
    for escaped_suffix in (r"\n", r"\r", r"\t"):
        while text.endswith(escaped_suffix):
            text = text[: -len(escaped_suffix)].rstrip()
    changed = True
    while changed:
        changed = False
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
            text = text[1:-1].strip()
            changed = True
        elif len(text) >= 4 and text[:2] in (r"\"", r"\'") and text[-2:] == text[:2]:
            text = text[2:-2].strip()
            changed = True
    return text.rstrip(",;").strip().lower()


def _is_direct_material_secret(value: bytes) -> bool:
    normalised = _normalise_value(value)
    return normalised not in PLACEHOLDER_VALUES and len(normalised) >= 8


def _reference_text(value: bytes) -> str:
    text = re.sub(r"\\+([\"'])", r"\1", _normalise_value(value))
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        text = text[1:-1].strip()
    return text


def _reference_names(value: bytes) -> list[str]:
    """Return identifiers from syntactically recognisable self references."""
    text = _reference_text(value)
    names: list[str] = []
    for match in re.finditer(
        r"\$(?:\{(?P<braced>[a-z_][a-z0-9_]*)|(?P<plain>[a-z_][a-z0-9_]*))", text
    ):
        names.append(match.group("braced") or match.group("plain"))
    for pattern in (
        r"os\s*\.\s*environ\s*\[\s*(['\"])(?P<name>[a-z_][a-z0-9_]*)\1\s*\]",
        r"os\s*\.\s*getenv\s*\(\s*(['\"])(?P<name>[a-z_][a-z0-9_]*)\1",
    ):
        names.extend(match.group("name") for match in re.finditer(pattern, text))
    return names


def _is_exact_self_reference(value: bytes, key: bytes) -> bool:
    text = _reference_text(value)
    normalised_key = _normalise_credential_identifier(key.decode("ascii", errors="ignore"))
    if re.fullmatch(r"[a-z_][a-z0-9_]*", text):
        return _normalise_credential_identifier(text) == normalised_key
    match = re.fullmatch(
        r"\$(?:\{(?P<braced>[a-z_][a-z0-9_]*)\}|(?P<plain>[a-z_][a-z0-9_]*))",
        text,
    )
    if match is None:
        match = re.fullmatch(
            r"os\s*\.\s*environ\s*\[\s*(['\"])(?P<name>[a-z_][a-z0-9_]*)\1\s*\]",
            text,
        )
    if match is None:
        match = re.fullmatch(
            r"os\s*\.\s*getenv\s*\(\s*(['\"])(?P<name>[a-z_][a-z0-9_]*)\1\s*\)",
            text,
        )
    if match is None:
        return False
    name = (
        match.groupdict().get("braced")
        or match.groupdict().get("plain")
        or match.groupdict().get("name")
    )
    return _normalise_credential_identifier(name or "") == normalised_key


def _mentions_self_reference(value: bytes, key: bytes) -> bool:
    normalised_key = _normalise_credential_identifier(key.decode("ascii", errors="ignore"))
    return any(
        _normalise_credential_identifier(name) == normalised_key for name in _reference_names(value)
    )


def _is_angle_placeholder(normalised: str, normalised_key: str | None) -> bool:
    match = re.fullmatch(r"<([^<>\r\n]{1,256})>", normalised)
    if match is None:
        return False
    inner = _normalise_credential_identifier(match.group(1))
    if normalised_key is not None and inner == normalised_key:
        return True
    return bool(set(inner.split("_")) & ANGLE_PLACEHOLDER_WORDS)


def _has_material_value_suffix(data: bytes, value_end: int) -> bool:
    """Reject parser-prefix exemptions when the same value continues.

    The corpus contains both shell text and JSON-escaped source text.  A
    recognised placeholder or exact self-reference is safe only if the bytes
    after the captured value are a real/encoded line end, a shell separator,
    a next shell argument, or JSON serialization framing.  Any immediately
    concatenated byte (including an unmatched quote or backslash) makes the
    assignment material.
    """
    tail = data[value_end : value_end + 512]
    if not tail or tail.startswith((b"\r", b"\n")):
        return False
    if re.match(_ENCODED_LINE_END, tail):
        return False
    if re.match(_SERIALIZED_VALUE_END, tail):
        return False
    if re.match(_JSON_CONTAINER_END, tail) or re.match(_JSON_MEMBER_CONTINUATION, tail):
        return False
    if tail.startswith(b";"):
        return False
    horizontal = len(tail) - len(tail.lstrip(b" \t"))
    if horizontal:
        remainder = tail[horizontal:]
        if not remainder or remainder.startswith((b"\r", b"\n", b";", b"#", b"--")):
            return False
        if re.match(_ENCODED_LINE_END, remainder) or re.match(
            _SERIALIZED_VALUE_END, remainder
        ):
            return False
        if re.match(_JSON_CONTAINER_END, remainder) or re.match(
            _JSON_MEMBER_CONTINUATION, remainder
        ):
            return False
        return remainder.startswith((b"+", b"|", b"&"))
    return True


def _key_specific_sentinel(key: bytes, value: bytes) -> str | None:
    """Return an exact, registered sentinel for ``key``, if present."""
    normalised_key = _normalise_credential_identifier(
        key.decode("ascii", errors="ignore")
    )
    normalised_value = _normalise_value(value)
    if normalised_value in KEY_SPECIFIC_PLACEHOLDERS.get(normalised_key, ()):
        return normalised_value
    return None


def _is_balanced_inline_cli_sentinel(data: bytes, match: re.Match[bytes]) -> bool:
    """Recognise only the audited Markdown form for the local CLI sentinel.

    Historical documentation contains ``--api-key inspectai`` in a single-
    backtick code span.  The generic unquoted-value parser consumes the closing
    backtick (and, in prose, a following full stop).  Keep this exception local
    to CLI matches and require both the opening delimiter and a terminal byte
    boundary; a bare/generic backtick suffix must remain material.
    """
    key = match.group("key")
    normalised_key = _normalise_credential_identifier(
        key.decode("ascii", errors="ignore")
    )
    sentinels = KEY_SPECIFIC_PLACEHOLDERS.get(normalised_key, ())
    if not sentinels:
        return False

    raw_value = match.group("value")
    matched_sentinel: bytes | None = None
    for sentinel in sentinels:
        encoded = sentinel.encode("ascii")
        if raw_value.lower().startswith(encoded):
            matched_sentinel = encoded
            break
    if matched_sentinel is None:
        return False

    suffix = raw_value[len(matched_sentinel) :]
    if suffix not in (
        b"`",
        b"`" + CLI_INLINE_SENTINEL_TERMINAL_PUNCTUATION,
    ):
        return False

    line_start = data.rfind(b"\n", 0, match.start()) + 1
    logical_start = max(line_start, match.start() - 4096)
    cursor = logical_start
    while True:
        encoded_newline = data.find(b"\\n", cursor, match.start())
        if encoded_newline < 0:
            break
        if _has_encoded_line_boundary(data, encoded_newline + 2):
            logical_start = encoded_newline + 2
        cursor = encoded_newline + 2

    delimiters: list[int] = []
    for index in range(logical_start, match.start()):
        if data[index] != ord("`"):
            continue
        if index > logical_start and data[index - 1] in (ord("`"), ord("\\")):
            continue
        if index + 1 < match.start() and data[index + 1] == ord("`"):
            continue
        delimiters.append(index)
    if len(delimiters) % 2 != 1:
        return False
    opener = delimiters[-1]
    if opener < 0:
        return False
    # Only a single, unescaped inline-code delimiter is in scope.  Fences and
    # escaped prose are deliberately not inferred to be harmless.
    if opener > 0 and data[opener - 1 : opener] in (b"`", b"\\"):
        return False

    tail = data[match.end() : match.end() + 1]
    return not tail or tail in b" \t\r\n"


def _json_record_bounds(data: bytes, position: int) -> tuple[int, int, int] | None:
    """Return ``(json_start, line_start, line_end)`` for a valid JSONL record."""
    line_start = data.rfind(b"\n", 0, position) + 1
    line_end = data.find(b"\n", position)
    if line_end < 0:
        line_end = len(data)
    json_start = data.find(b"{", line_start, position + 1)
    if json_start < 0:
        return None
    if line_end - json_start > 1_048_576:
        return None
    try:
        decoded = json.loads(data[json_start:line_end])
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, MemoryError):
        return None
    if not isinstance(decoded, (dict, list)):
        return None
    return json_start, line_start, line_end


def _json_string_opener_at(
    data: bytes, *, json_start: int, position: int
) -> tuple[int, bool] | None:
    """Return the active JSON string opener and escape state at ``position``."""
    opener = -1
    escaped = False
    in_string = False
    for index in range(json_start, position):
        byte = data[index]
        if not in_string:
            if byte == ord('"'):
                in_string = True
                opener = index
            continue
        if escaped:
            escaped = False
        elif byte == ord("\\"):
            escaped = True
        elif byte == ord('"'):
            in_string = False
            opener = -1
    if not in_string:
        return None
    return opener, escaped


def _has_encoded_line_boundary(data: bytes, position: int) -> bool:
    """Whether ``position`` follows one JSON-encoded CR/LF escape."""
    for marker in (b"\\n", b"\\r"):
        start = position - len(marker)
        if start < 0 or data[start:position] != marker:
            continue
        preceding_slashes = 0
        cursor = start - 1
        while cursor >= 0 and data[cursor] == ord("\\"):
            preceding_slashes += 1
            cursor -= 1
        if preceding_slashes % 2 == 0:
            return True
    return False


def _is_terminal_json_env_sentinel(data: bytes, match: re.Match[bytes]) -> bool:
    """Recognise the audited terminal env line inside a valid JSON string.

    The raw event stream stores command output in a JSON string.  Its final
    decoded line is ``VLLM_API_KEY=inspectai``; the env parser sees the JSON
    string's closing quote as suspicious suffix material.  Exempt it only when
    the registered sentinel is the complete final logical line, the closer is
    the actual matching JSON-string delimiter, and the full JSONL record parses.
    """
    key = match.group("key")
    if _normalise_credential_identifier(
        key.decode("ascii", errors="ignore")
    ) != "vllm_api_key":
        return False
    if _key_specific_sentinel(key, match.group("value")) is None:
        return False

    value_end = match.end("value")
    if data[value_end : value_end + 1] != b'"':
        return False
    bounds = _json_record_bounds(data, match.start())
    if bounds is None:
        return False
    json_start, _line_start, line_end = bounds
    if value_end >= line_end:
        return False

    state_at_key = _json_string_opener_at(
        data, json_start=json_start, position=match.start()
    )
    state_at_closer = _json_string_opener_at(
        data, json_start=json_start, position=value_end
    )
    if state_at_key is None or state_at_closer is None:
        return False
    opener_at_key, escaped_at_key = state_at_key
    opener_at_closer, escaped_at_closer = state_at_closer
    if escaped_at_key or escaped_at_closer or opener_at_key != opener_at_closer:
        return False
    return match.start() == opener_at_key + 1 or _has_encoded_line_boundary(
        data, match.start()
    )


def _literal_mapping_candidates(
    data: bytes, match: re.Match[bytes]
) -> Iterable[str]:
    """Yield bounded JSON/Python mapping candidates enclosing ``match``."""
    line_start = data.rfind(b"\n", 0, match.start()) + 1
    line_end = data.find(b"\n", match.end())
    if line_end < 0:
        line_end = len(data)
    lower = max(line_start, match.start() - 4096)
    upper = min(line_end, match.end() + 4096)

    opens: list[int] = []
    cursor = match.start()
    while len(opens) < 32:
        cursor = data.rfind(b"{", lower, cursor)
        if cursor < 0:
            break
        opens.append(cursor)
    closes: list[int] = []
    cursor = match.end()
    while len(closes) < 32:
        cursor = data.find(b"}", cursor, upper)
        if cursor < 0:
            break
        closes.append(cursor)
        cursor += 1

    seen: set[str] = set()
    for opener in opens:
        for closer in closes:
            candidate = data[opener : closer + 1].decode("latin-1")
            for variant in (candidate, candidate.replace(r'\"', '"')):
                if variant not in seen:
                    seen.add(variant)
                    yield variant


def _is_valid_local_authorization_wrapper(
    data: bytes, match: re.Match[bytes]
) -> bool:
    """Accept only an exact local bearer sentinel in a parsed mapping."""
    if _normalise_credential_identifier(
        match.group("key").decode("ascii", errors="ignore")
    ) != "authorization":
        return False
    value = _normalise_value(match.group("value"))
    bearer = re.fullmatch(r"bearer[ \t]+([a-z0-9._-]+)", value)
    if bearer is None or bearer.group(1) not in LOCAL_API_SENTINELS:
        return False

    for candidate in _literal_mapping_candidates(data, match):
        for parser in (json.loads, ast.literal_eval):
            try:
                mapping = parser(candidate)
            except (ValueError, SyntaxError, TypeError, RecursionError, MemoryError):
                continue
            if not isinstance(mapping, dict):
                continue
            for key, candidate_value in mapping.items():
                if not isinstance(key, str) or not isinstance(candidate_value, str):
                    continue
                if key.lower() != "authorization":
                    continue
                parsed = re.fullmatch(
                    r"bearer[ \t]+([a-z0-9._-]+)", candidate_value.lower()
                )
                if parsed is not None and parsed.group(1) in LOCAL_API_SENTINELS:
                    return True
    return False


def _is_material_secret(value: bytes, *, key: bytes | None = None) -> bool:
    normalised = _normalise_value(value)
    normalised_key = (
        _normalise_credential_identifier(key.decode("ascii", errors="ignore"))
        if key is not None
        else None
    )
    if _is_angle_placeholder(normalised, normalised_key):
        return False
    if key is not None:
        if _is_exact_self_reference(value, key):
            return False
        if normalised in KEY_SPECIFIC_PLACEHOLDERS.get(normalised_key, ()):
            return False
    if normalised in PLACEHOLDER_VALUES or len(normalised) < 8:
        return False
    if key is not None and normalised_key == "openai_api_key":
        # OPENAI_API_KEY is widely used by local/mock tools.  A literal is
        # credential material only when it has a credible provider shape.
        # Non-exact expressions mentioning the key remain fail-closed.
        reference_expression = (
            re.match(r"(?:[`$]|os\s*\.\s*(?:environ|getenv)\b)", normalised) is not None
        )
        return (
            _mentions_self_reference(value, key)
            or reference_expression
            or "\\" in normalised
            or _OPENAI_SECRET.fullmatch(normalised) is not None
        )
    return True


def _normalise_credential_identifier(text: str) -> str:
    separated = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if separated in CASE_INSENSITIVE_ENV_KEYS:
        return separated
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _is_sensitive_key(value: bytes) -> bool:
    # Split CamelCase before normalising separators so accessToken and
    # clientSecret receive the same treatment as access_token/client_secret.
    normalised = _normalise_credential_identifier(value.decode("ascii", errors="ignore"))
    if normalised in NON_CREDENTIAL_TOKEN_KEYS:
        return False
    if normalised in SENSITIVE_KEYS or any(
        normalised.endswith(f"_{key}") for key in SENSITIVE_KEYS
    ):
        return True
    return any(part in SENSITIVE_KEY_PARTS for part in normalised.split("_"))


def _is_sensitive_env_key(value: bytes) -> bool:
    """Classify a broadly captured uppercase or provider-specific env name."""
    return _is_sensitive_key(value)


def scan_credential_bytes(data: bytes, *, path: str) -> list[dict[str, object]]:
    """Return secret-free finding records for one file's bytes."""
    counts: Counter[str] = Counter()
    for rule_id, pattern, marker in _DIRECT_RULES:
        if marker in data:
            counts[rule_id] += sum(1 for _match in pattern.finditer(data))
    seen_env_spans: set[tuple[int, int]] = set()
    for pattern in (_ENV_ASSIGNMENT, _CASE_INSENSITIVE_ENV_ASSIGNMENT):
        for match in pattern.finditer(data):
            span = match.span()
            if span in seen_env_spans:
                continue
            seen_env_spans.add(span)
            material = _is_material_secret(match.group("value"), key=match.group("key"))
            if _is_sensitive_env_key(match.group("key")) and (
                material or _has_material_value_suffix(data, match.end("value"))
            ) and not _is_terminal_json_env_sentinel(data, match):
                counts["secret-env-assignment"] += 1
    for match in _CLI_ASSIGNMENT.finditer(data):
        material = _is_material_secret(match.group("value"), key=match.group("key"))
        if _is_sensitive_key(match.group("key")) and (
            material or _has_material_value_suffix(data, match.end("value"))
        ) and not _is_balanced_inline_cli_sentinel(data, match):
            counts["secret-cli-argument"] += 1
    seen_context_spans: set[tuple[int, int]] = set()
    for pattern, rule_id in (
        (_JSON_ASSIGNMENT, "secret-json-field"),
        (_ESCAPED_JSON_ASSIGNMENT, "secret-json-field"),
        (_REPR_ASSIGNMENT, "secret-json-field"),
        (_ESCAPED_REPR_ASSIGNMENT, "secret-json-field"),
        (_FIELD_ASSIGNMENT, "secret-text-field"),
    ):
        for match in pattern.finditer(data):
            if match.span() in seen_context_spans:
                continue
            seen_context_spans.add(match.span())
            material = _is_material_secret(match.group("value"), key=match.group("key"))
            if _is_sensitive_key(match.group("key")) and (
                material or _has_material_value_suffix(data, match.end())
            ) and not _is_valid_local_authorization_wrapper(data, match):
                counts[rule_id] += 1
    for match in _BEARER.finditer(data):
        if _is_direct_material_secret(match.group("value")):
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
