#!/usr/bin/env python3
"""Redact credential material from a Claude stream without changing its events.

Claude Code's Bash tool receives a short-lived messaging token, and a scientist
can also print credential files that happen to be present in a shared cache.
The model has already seen a tool result before it is emitted on stdout, so this
filter only changes the retained/published trajectory.  It accepts JSONL stream
events, recursively redacts sensitive fields and strings, and passes malformed
diagnostic lines through the same string scrubber.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any, TextIO

REDACTED = "<redacted>"
SENSITIVE_FIELDS = {
    "access_token",
    "api_key",
    "authorization",
    "client_secret",
    "credential",
    "credentials",
    "hf_token",
    "id_token",
    "oauth_token",
    "password",
    "passwd",
    "private_key",
    "refresh_token",
    "secret",
    "token",
}
ASSIGNMENT = re.compile(
    r"(?im)(?P<prefix>(?<![A-Za-z0-9_])(?:export[ \t]+)?"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*"
    r"(?:TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|AUTHORIZATION))"
    r"[ \t]*=[ \t]*)[^\s,;]+"
)
JSONISH_SECRET = re.compile(
    r"(?i)(?P<prefix>[\"']?(?:access_token|api_key|authorization|client_secret|"
    r"hf_token|id_token|oauth_token|password|passwd|private_key|refresh_token|"
    r"secret|token)[\"']?[ \t]*[:=][ \t]*[\"']?)[^\"'\r\n,}]+"
)
BEARER = re.compile(r"(?i)(?P<prefix>authorization[ \t]*:[ \t]*bearer[ \t]+)\S+")
HF_TOKEN = re.compile(r"\bhf_[A-Za-z0-9]{16,}\b")
OPENAI_STYLE_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")
GOOGLE_ACCESS_TOKEN = re.compile(r"\bya29\.[A-Za-z0-9._~-]{16,}\b")
JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
GOOGLE_API_KEY = re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b")
GITHUB_TOKEN = re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")
SLACK_TOKEN = re.compile(r"\bxox[a-z]-[A-Za-z0-9-]{16,}\b", re.IGNORECASE)
AWS_ACCESS_KEY = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
PRIVATE_KEY = re.compile(
    r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----.*?"
    r"-----END (?:[A-Z ]+ )?PRIVATE KEY-----",
    re.DOTALL,
)


def redact_text(value: str) -> str:
    value = ASSIGNMENT.sub(lambda match: match.group("prefix") + REDACTED, value)
    value = JSONISH_SECRET.sub(lambda match: match.group("prefix") + REDACTED, value)
    value = BEARER.sub(lambda match: match.group("prefix") + REDACTED, value)
    value = HF_TOKEN.sub(REDACTED, value)
    value = OPENAI_STYLE_KEY.sub(REDACTED, value)
    value = GOOGLE_ACCESS_TOKEN.sub(REDACTED, value)
    value = JWT.sub(REDACTED, value)
    value = GOOGLE_API_KEY.sub(REDACTED, value)
    value = GITHUB_TOKEN.sub(REDACTED, value)
    value = SLACK_TOKEN.sub(REDACTED, value)
    value = AWS_ACCESS_KEY.sub(REDACTED, value)
    return PRIVATE_KEY.sub(REDACTED, value)


def sensitive_field(key: object) -> bool:
    if not isinstance(key, str):
        return False
    normalised = key.strip().lower().replace("-", "_")
    return normalised in SENSITIVE_FIELDS or normalised.endswith(
        (
            "_access_token",
            "_api_key",
            "_auth_token",
            "_client_secret",
            "_messaging_token",
            "_oauth_token",
            "_password",
            "_private_key",
            "_refresh_token",
            "_secret",
            "_token",
        )
    )


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        clean: dict[Any, Any] = {}
        for key, child in value.items():
            if sensitive_field(key) and isinstance(child, str):
                clean[key] = REDACTED
            else:
                clean[key] = redact(child)
        return clean
    if isinstance(value, list):
        return [redact(child) for child in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def filter_stream(source: TextIO, destination: TextIO) -> None:
    for raw_line in source:
        line = raw_line.rstrip("\n")
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            rendered = redact_text(line)
        else:
            rendered = json.dumps(redact(event), ensure_ascii=False, separators=(",", ":"))
        destination.write(rendered + "\n")
        destination.flush()


def main() -> int:
    filter_stream(sys.stdin, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
