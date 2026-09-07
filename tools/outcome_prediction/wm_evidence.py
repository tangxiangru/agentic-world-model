"""Read-only, hash-pinned evidence tools for the matched inference/WM experiment.

Baseline CLI::

    python -m tools.outcome_prediction.wm_evidence --config evidence-server.json

The trusted JSON config has exactly ``evidence_root``, ``files`` (a mapping of
POSIX relative paths to SHA256), and ``audit_log``. Relative filesystem paths
resolve against the config's parent directory. Do not put hidden labels in the
config or the evidence allowlist. Unlisted files are never enumerated or read.

A trusted WM wrapper uses the SAME evidence tools and adds only a fixed-ID tool::

    with EvidenceStore(root, files, audit_log=log,
                       candidates=candidate_payloads, predict=predict_by_id) as store:
        serve_stdio(store)

The callback is trusted application code, not an import path or agent-controlled
query. It must load only the train-fitted model and public candidate inputs. This
gateway restricts the model's tool access; it is NOT an OS sandbox for the trusted
Python server. Disable all other filesystem, shell, network, and MCP tools in the
agent harness. The caller is responsible for the allowlist's train/test boundary.

Every file read/search verifies its full bytes before returning any content.
UTF-8 text is preserved, with character (not byte) offsets and resumable pages.
Search scans a bounded number of characters/files per page (a page may have no
matches and a non-null cursor). Full-file hashing can still be expensive for large
files; no stale content cache or lossy truncation is used.

The dependency-free stdio implementation follows MCP 2025-11-25:
https://modelcontextprotocol.io/specification/2025-11-25/basic/transports
https://modelcontextprotocol.io/specification/2025-11-25/server/tools
Only newline-delimited JSON-RPC is written to stdout. Audit records contain tool
names, argument hashes, status and time, never evidence, queries, or predictions.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import stat
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, TextIO

PROTOCOL_VERSION = "2025-11-25"
MAX_READ_CHARS = 16_000
MAX_SEARCH_MATCHES = 50
SEARCH_SCAN_CHARS = 1_000_000
SEARCH_SCAN_FILES = 32
MAX_REQUEST_CHARS = 65_536
MAX_PREDICTION_CHARS = 32_000


class EvidenceError(ValueError):
    """An intentionally safe, model-visible tool error."""


class UnknownTool(EvidenceError):
    """A protocol-level unknown tool name."""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _logical_path(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise EvidenceError("Invalid logical evidence path.")
    if (
        any(ord(c) < 32 or ord(c) == 127 or 0xD800 <= ord(c) <= 0xDFFF for c in value)
        or "\\" in value
        or ":" in value
    ):
        raise EvidenceError("Invalid logical evidence path.")
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in ("", ".", "..") for p in value.split("/")):
        raise EvidenceError("Invalid logical evidence path.")
    if str(path) != value:
        raise EvidenceError("Invalid logical evidence path.")
    return value


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise EvidenceError(f"Invalid {name}.")
    return value


class EvidenceStore:
    """An explicit file allowlist, directory-FD boundary, and optional fixed-ID WM.

    POSIX ``openat``/``O_NOFOLLOW`` protects every component beneath the pinned
    root against symlink traversal and path-replacement races. The root itself
    must not be a symlink; trusted ancestors of that explicit root are resolved.
    Files are checked lazily at access, so listing never reads corpus contents.
    Use ``handle_call`` for audited access; the MCP server does so exclusively.
    """

    def __init__(
        self,
        root: str | Path,
        allowlist: Mapping[str, str],
        *,
        audit_log: str | Path,
        candidates: Iterable[str] = (),
        predict: Callable[[str], Mapping[str, Any]] | None = None,
    ) -> None:
        self._root_fd = -1
        self._audit_fd = -1
        if not hasattr(os, "O_NOFOLLOW") or os.open not in os.supports_dir_fd:
            raise EvidenceError("Secure directory-relative opening is unavailable.")
        if not isinstance(allowlist, Mapping):
            raise EvidenceError("Evidence allowlist must be a path-to-SHA256 mapping.")
        pinned = {}
        for path, digest in allowlist.items():
            path = _logical_path(path)
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                raise EvidenceError("Invalid evidence SHA256.")
            pinned[path] = digest.lower()
        self.allowlist = MappingProxyType(pinned)
        self._paths = tuple(sorted(pinned))
        self._manifest_hash = _sha(_json(sorted(pinned.items())).encode("utf-8"))
        if isinstance(candidates, (str, bytes)):
            raise EvidenceError("Candidate IDs must be a collection of identifiers.")
        candidate_ids = list(candidates)
        if any(
            not isinstance(c, str)
            or not c
            or len(c) > 256
            or any(ord(x) < 32 or ord(x) == 127 for x in c)
            for c in candidate_ids
        ):
            raise EvidenceError("Invalid candidate ID.")
        self._candidates = frozenset(candidate_ids)
        if predict is not None and (not callable(predict) or not self._candidates):
            raise EvidenceError("A prediction callback needs fixed candidate IDs.")
        self._predict = predict
        try:
            root_path = Path(root).absolute()
            if root_path.is_symlink():
                raise EvidenceError("Evidence root must not be a symlink.")
            self.root = root_path.resolve(strict=True)
            self._root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            log_path = Path(audit_log).absolute()
            if log_path.is_symlink():
                raise EvidenceError("Audit log must not be a symlink.")
            log_path = log_path.parent.resolve(strict=True) / log_path.name
            if log_path.is_relative_to(self.root):
                raise EvidenceError("Audit log must be outside the evidence root.")
            self._audit_fd = os.open(
                log_path,
                os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                0o600,
            )
            log_stat = os.fstat(self._audit_fd)
            if not stat.S_ISREG(log_stat.st_mode) or log_stat.st_nlink != 1:
                raise EvidenceError("Audit log must be a regular, non-hardlinked file.")
        except EvidenceError:
            self.close()
            raise
        except (OSError, ValueError, TypeError) as exc:
            self.close()
            raise EvidenceError("Cannot securely open the evidence root or audit log.") from exc

    def close(self) -> None:
        for attr in ("_root_fd", "_audit_fd"):
            fd = getattr(self, attr, -1)
            if fd >= 0:
                os.close(fd)
                setattr(self, attr, -1)

    def __enter__(self) -> EvidenceStore:  # noqa: PYI034 -- Python 3.10 has no typing.Self.
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _verified_text(self, path: str) -> str:
        path = _logical_path(path)
        if path not in self.allowlist:
            raise EvidenceError("Evidence path is not allowlisted.")
        if self._root_fd < 0:
            raise EvidenceError("Evidence store is closed.")
        directory_fd = os.dup(self._root_fd)
        file_fd = -1
        try:
            parts = path.split("/")
            for part in parts[:-1]:
                next_fd = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory_fd
                )
                os.close(directory_fd)
                directory_fd = next_fd
            file_fd = os.open(
                parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd
            )
            if not stat.S_ISREG(os.fstat(file_fd).st_mode):
                raise EvidenceError("Evidence must be a regular non-symlink file.")
            with os.fdopen(file_fd, "rb") as stream:
                file_fd = -1
                data = stream.read()
        except OSError as exc:
            raise EvidenceError("Evidence file is unavailable or traverses a symlink.") from exc
        finally:
            if file_fd >= 0:
                os.close(file_fd)
            os.close(directory_fd)
        if _sha(data) != self.allowlist[path]:
            raise EvidenceError("Evidence integrity check failed; no content was returned.")
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise EvidenceError("Evidence is not valid UTF-8 text.") from exc

    def list_evidence(self, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        offset = _integer(offset, "offset", 0, len(self._paths))
        limit = _integer(limit, "limit", 1, 100)
        end = min(offset + limit, len(self._paths))
        return {
            "files": [{"path": p, "sha256": self.allowlist[p]} for p in self._paths[offset:end]],
            "total_files": len(self._paths),
            "next_offset": end if end < len(self._paths) else None,
        }

    def read_evidence(self, path: str, offset: int = 0, count: int = 8000) -> dict[str, Any]:
        offset = _integer(offset, "offset", 0, 2**63 - 1)
        count = _integer(count, "count", 1, MAX_READ_CHARS)
        content = self._verified_text(path)
        if offset > len(content):
            raise EvidenceError("Read offset exceeds the file's character length.")
        end = min(offset + count, len(content))
        return {
            "path": path,
            "sha256": self.allowlist[path],
            "offset": offset,
            "text": content[offset:end],
            "total_chars": len(content),
            "next_offset": end if end < len(content) else None,
        }

    def _cursor(self, file_index: int, offset: int, query: str) -> str | None:
        if file_index >= len(self._paths):
            return None
        value = [file_index, offset, _sha(query.encode("utf-8")), self._manifest_hash]
        return base64.urlsafe_b64encode(_json(value).encode("ascii")).decode("ascii")

    def search_evidence(
        self, query: str, cursor: str | None = None, limit: int = 20, context_chars: int = 80
    ) -> dict[str, Any]:
        if not isinstance(query, str) or not 1 <= len(query) <= 512:
            raise EvidenceError("Literal query must contain 1 to 512 characters.")
        limit = _integer(limit, "limit", 1, MAX_SEARCH_MATCHES)
        context_chars = _integer(context_chars, "context_chars", 0, 200)
        file_index, offset = 0, 0
        if cursor is not None:
            try:
                if not isinstance(cursor, str) or len(cursor) > 512:
                    raise ValueError
                value = json.loads(base64.b64decode(cursor, altchars=b"-_", validate=True))
                if not isinstance(value, list) or len(value) != 4:
                    raise ValueError
                file_index = _integer(value[0], "cursor", 0, len(self._paths) - 1)
                offset = _integer(value[1], "cursor", 0, 2**63 - 1)
                if value[2:] != [_sha(query.encode("utf-8")), self._manifest_hash]:
                    raise ValueError
            except (ValueError, TypeError, UnicodeError) as exc:
                raise EvidenceError("Invalid cursor for this query and evidence manifest.") from exc
        matches = []
        remaining = SEARCH_SCAN_CHARS
        scanned_files = 0
        while file_index < len(self._paths) and remaining > 0 and scanned_files < SEARCH_SCAN_FILES:
            path = self._paths[file_index]
            content = self._verified_text(path)
            scanned_files += 1
            if offset > len(content):
                raise EvidenceError("Search cursor exceeds the file's character length.")
            boundary = min(len(content), offset + remaining)
            start = offset
            while start < boundary:
                hit = content.find(query, start, min(len(content), boundary + len(query) - 1))
                if hit < 0 or hit >= boundary:
                    break
                left = max(0, hit - context_chars)
                right = min(len(content), hit + len(query) + context_chars)
                matches.append(
                    {
                        "path": path,
                        "offset": hit,
                        "snippet_start": left,
                        "snippet": content[left:right],
                    }
                )
                start = hit + 1  # Include overlapping literal matches without repeating a hit.
                if len(matches) == limit:
                    if start >= len(content):
                        file_index, start = file_index + 1, 0
                    return {
                        "matches": matches,
                        "next_cursor": self._cursor(file_index, start, query),
                    }
            remaining -= boundary - offset
            if boundary >= len(content):
                file_index, offset = file_index + 1, 0
            else:
                offset = boundary
        return {"matches": matches, "next_cursor": self._cursor(file_index, offset, query)}

    def predict_candidate(self, candidate_id: str) -> dict[str, Any]:
        if self._predict is None:
            raise UnknownTool("Unknown tool.")
        if not isinstance(candidate_id, str) or candidate_id not in self._candidates:
            raise EvidenceError("Candidate ID is not registered for this task.")
        try:
            result = self._predict(candidate_id)
            if not isinstance(result, Mapping):
                raise TypeError
            encoded = _json(dict(result))
            if len(encoded) > MAX_PREDICTION_CHARS:
                raise ValueError
            return json.loads(encoded)
        except Exception as exc:
            raise EvidenceError(
                "The registered prediction callback failed or returned invalid data."
            ) from exc

    def tool_specs(self) -> list[dict[str, Any]]:
        def spec(name: str, description: str, properties: dict, required: list[str]) -> dict:
            return {
                "name": name,
                "description": description,
                "inputSchema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
                "annotations": {
                    "readOnlyHint": True,
                    "destructiveHint": False,
                    "idempotentHint": True,
                    "openWorldHint": False,
                },
            }

        def integer(minimum: int, maximum: int, default: int) -> dict:
            return {"type": "integer", "minimum": minimum, "maximum": maximum, "default": default}

        specs = [
            spec(
                "list_evidence",
                "List only allowed evidence files; resume using next_offset.",
                {
                    "offset": integer(0, 2**63 - 1, 0),
                    "limit": integer(1, 100, 50),
                },
                [],
            ),
            spec(
                "read_evidence",
                "Read hash-verified UTF-8 text by character offset. Resume "
                "using next_offset until null to read the entire file.",
                {
                    "path": {"type": "string"},
                    "offset": integer(0, 2**63 - 1, 0),
                    "count": integer(1, MAX_READ_CHARS, 8000),
                },
                ["path"],
            ),
            spec(
                "search_evidence",
                "Search literal, case-sensitive text in allowed files only. "
                "Resume next_cursor even when matches is empty; null means the scan is complete.",
                {
                    "query": {"type": "string", "minLength": 1, "maxLength": 512},
                    "cursor": {"type": ["string", "null"], "maxLength": 512},
                    "limit": integer(1, MAX_SEARCH_MATCHES, 20),
                    "context_chars": integer(0, 200, 80),
                },
                ["query"],
            ),
        ]
        if self._predict is not None:
            specs.append(
                spec(
                    "predict_candidate",
                    "Query the train-fitted world model for one "
                    "registered candidate, without executing its recipe.",
                    {
                        "candidate_id": {"type": "string", "enum": sorted(self._candidates)},
                    },
                    ["candidate_id"],
                )
            )
        return specs

    def _audit(self, name: Any, arguments: Any, status: str, started: float) -> None:
        known = {s["name"] for s in self.tool_specs()}
        try:
            argument_digest = _sha(_json(arguments).encode("utf-8"))
        except (ValueError, TypeError, UnicodeError):
            argument_digest = "invalid-json"
        record = {
            "time_unix_ns": time.time_ns(),
            "tool": name if isinstance(name, str) and name in known else "[unknown]",
            "arguments_sha256": argument_digest,
            "status": status,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
        }
        try:
            data = (_json(record) + "\n").encode("utf-8")
            if self._audit_fd < 0 or os.write(self._audit_fd, data) != len(data):
                raise OSError
        except OSError as exc:
            raise EvidenceError("Audit logging failed; tool output withheld.") from exc

    def handle_call(self, name: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Audited dispatch. Never return raw OS paths, callback errors, or stack traces."""
        started = time.monotonic()
        status = "error"
        try:
            specs = {s["name"]: s for s in self.tool_specs()}
            if not isinstance(name, str) or name not in specs:
                raise UnknownTool("Unknown tool.")
            arguments = {} if arguments is None else arguments
            if not isinstance(arguments, Mapping):
                raise EvidenceError("Tool arguments must be an object.")
            schema = specs[name]["inputSchema"]
            if set(arguments) - set(schema["properties"]) or set(schema["required"]) - set(
                arguments
            ):
                raise EvidenceError("Missing required or unrecognized tool arguments.")
            result = getattr(self, name)(**arguments)
            status = "ok"
            return result
        except EvidenceError:
            raise
        except Exception as exc:
            raise EvidenceError("Evidence tool failed; no output was returned.") from exc
        finally:
            self._audit(name, arguments, status, started)


class MCPServer:
    """Small synchronous, static-tools MCP server; one instance per agent session."""

    def __init__(self, store: EvidenceStore) -> None:
        self.store = store
        self.initialized = False

    @staticmethod
    def error(request_id: Any, code: int, message: str) -> dict[str, Any]:
        result = {"jsonrpc": "2.0", "error": {"code": code, "message": message}}
        if type(request_id) in (str, int):
            result["id"] = request_id
        return result

    def handle_message(self, message: Any) -> dict[str, Any] | None:
        if not isinstance(message, dict):
            return self.error(None, -32600, "Invalid request.")
        request_id = message.get("id")
        is_notification = "id" not in message
        if message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
            return self.error(None, -32600, "Invalid request.")
        if not is_notification and type(request_id) not in (str, int):
            return self.error(None, -32600, "Invalid request ID.")
        # Notifications never cause tool execution and never receive responses.
        if is_notification:
            return None
        params = message.get("params", {})
        if not isinstance(params, dict):
            return self.error(request_id, -32602, "Parameters must be an object.")
        method = message["method"]
        if method == "ping":
            result = {}
        elif method == "initialize":
            client = params.get("clientInfo")
            if self.initialized or not (
                isinstance(params.get("protocolVersion"), str)
                and isinstance(params.get("capabilities"), dict)
                and isinstance(client, dict)
                and isinstance(client.get("name"), str)
                and isinstance(client.get("version"), str)
            ):
                return self.error(request_id, -32602, "Invalid or repeated initialization.")
            self.initialized = True
            result = {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "wm-evidence", "version": "1.0.0"},
            }
        elif not self.initialized:
            return self.error(request_id, -32000, "Initialize the server first.")
        elif method == "tools/list":
            if set(params) - {"cursor", "_meta"} or params.get("cursor") not in (None, ""):
                return self.error(request_id, -32602, "Invalid tools/list parameters.")
            result = {"tools": self.store.tool_specs()}
        elif method == "tools/call":
            if not isinstance(params.get("name"), str) or set(params) - {
                "name",
                "arguments",
                "_meta",
            }:
                return self.error(request_id, -32602, "Invalid tools/call parameters.")
            if "arguments" in params and not isinstance(params["arguments"], dict):
                return self.error(request_id, -32602, "Tool arguments must be an object.")
            try:
                value = self.store.handle_call(params["name"], params.get("arguments", {}))
                result = {
                    "content": [{"type": "text", "text": _json(value)}],
                    "structuredContent": value,
                    "isError": False,
                }
            except UnknownTool:
                return self.error(request_id, -32602, "Unknown tool.")
            except EvidenceError as exc:
                result = {"content": [{"type": "text", "text": str(exc)}], "isError": True}
        else:
            return self.error(request_id, -32601, "Method not found.")
        return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _reject_constant(_: str) -> None:
    raise ValueError("Non-finite JSON number")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def serve_stdio(
    store: EvidenceStore, *, stdin: TextIO | None = None, stdout: TextIO | None = None
) -> None:
    """Serve newline-delimited JSON-RPC until EOF; no logs or prompts on stdout."""
    source = sys.stdin if stdin is None else stdin
    sink = sys.stdout if stdout is None else stdout
    server = MCPServer(store)
    while True:
        line = source.readline(MAX_REQUEST_CHARS + 1)
        if not line:
            return
        if len(line) > MAX_REQUEST_CHARS:
            while line and not line.endswith("\n"):
                line = source.readline(MAX_REQUEST_CHARS + 1)
            response = server.error(None, -32600, "Request exceeds the size limit.")
        else:
            try:
                message = json.loads(
                    line, parse_constant=_reject_constant, object_pairs_hook=_unique_object
                )
            except (ValueError, RecursionError):
                response = server.error(None, -32700, "Invalid JSON.")
            else:
                try:
                    response = server.handle_message(message)
                except Exception:  # noqa: BLE001 -- Never expose trusted-process exception details.
                    response = server.error(None, -32603, "Internal server error.")
        if response is not None:
            # JSON permits escaped surrogate code points; never let a request ID
            # or trusted callback string cause the UTF-8 transport to crash.
            sink.write(json.dumps(response, ensure_ascii=True, allow_nan=False) + "\n")
            sink.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        config_path = args.config.resolve(strict=True)
        config = json.loads(
            config_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
        )
        if not isinstance(config, dict) or set(config) != {"evidence_root", "files", "audit_log"}:
            raise EvidenceError("Invalid configuration keys.")
        root = config_path.parent / config["evidence_root"]
        audit_log = config_path.parent / config["audit_log"]
        with EvidenceStore(root, config["files"], audit_log=audit_log) as store:
            serve_stdio(store)
        return 0
    except (EvidenceError, OSError, ValueError, TypeError):
        print("Evidence server configuration or transport failed.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
