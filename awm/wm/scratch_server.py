"""MCP tools for autonomous WMA corpus queries and scratch-local programs.

The server is standard-library-only and newline-delimited JSON-RPC. Corpus
tools are read-only. Scratch writes are confined to one directory, and scratch
programs run in a new user, mount, PID, and network namespace with a chroot that
contains only system executables, the writable scratch directory, and the
allowed corpus mounted read-only.
"""

from __future__ import annotations

import json
import os
import re
import resource
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import ctypes
import contextlib
import http.server
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO
from collections.abc import Iterator

SERVER_NAME = "awm_scratch"
SERVER_VERSION = "1.0.0"
PROTOCOL_VERSION = "2025-06-18"
SCRATCH_ENV = "AWM_WMA_SCRATCH_ROOT"
CORPUS_ENV = "AWM_WMA_CORPUS_ROOTS"
AUDIT_ENV = "AWM_WMA_SCRATCH_AUDIT"
MAX_WRITE_BYTES = 1_000_000
MAX_RESULT_BYTES = 1_000_000
MAX_LIST = 10_000
MAX_SEARCH_MATCHES = 5_000
MAX_SEARCH_BYTES = 20_000_000
MAX_READ_BYTES = 500_000
MAX_TIMEOUT_S = 120
MAX_SCRATCH_BYTES = 10_000_000
MAX_SCRATCH_ENTRIES = 10_000
MAX_SCRATCH_DEPTH = 32

# Linux prctl/capability constants.  Keeping this stdlib-only avoids adding a
# package to the fixed PTB image while still removing the user-namespace root
# process's ability to remount the historical corpus read-write.
PR_CAPBSET_DROP = 24
PR_SET_NO_NEW_PRIVS = 38
PR_SET_SECUREBITS = 28
PR_CAP_AMBIENT = 47
PR_CAP_AMBIENT_CLEAR_ALL = 4
SECBIT_NOROOT = 1 << 0
SECBIT_NOROOT_LOCKED = 1 << 1
SECBIT_NO_SETUID_FIXUP = 1 << 2
SECBIT_NO_SETUID_FIXUP_LOCKED = 1 << 3
LINUX_CAPABILITY_VERSION_3 = 0x20080522

TOOLS: tuple[dict[str, Any], ...] = (
    {
        "name": "list_corpus",
        "description": (
            "List files across the complete allowed corpus using a caller-chosen glob. "
            "No host-ranked shortlist is applied. Root defaults to 0 only when one root exists."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "root": {"type": "integer", "minimum": 0},
                "glob": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIST},
            },
            "required": ["glob"],
        },
    },
    {
        "name": "search_corpus",
        "description": (
            "Literal-search text files selected by a caller-chosen glob over a complete corpus root. "
            "Returns file, line number, and matching line; the response states if the chosen limit "
            "truncates it. Root defaults to 0 only when one root exists."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "root": {"type": "integer", "minimum": 0},
                "glob": {"type": "string"},
                "pattern": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_SEARCH_MATCHES},
            },
            "required": ["glob", "pattern"],
        },
    },
    {
        "name": "read_corpus",
        "description": (
            "Read a byte range from an exact corpus file. Paths are relative to the selected root. "
            "Use successive offsets for large trajectory files. Root defaults to 0 only when one root exists."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "root": {"type": "integer", "minimum": 0},
                "path": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_READ_BYTES},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Create or replace a scratch-local source/tool file. The path must be relative to the "
            "per-call scratch directory; historical corpus and scientist files cannot be written."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "run",
        "description": (
            "Run a scratch-local program with argv (for example ['python3','tool.py']). It executes "
            "offline in an isolated chroot: /work is writable scratch and /corpus/N are read-only "
            "complete corpus roots. No scientist/session directory is mounted."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "argv": {"type": "array", "minItems": 1, "items": {"type": "string"}},
                "timeout_s": {"type": "integer", "minimum": 1, "maximum": MAX_TIMEOUT_S},
            },
            "required": ["argv"],
        },
    },
)
TOOLS_BY_NAME = {tool["name"]: tool for tool in TOOLS}


def _result(text: str, *, is_error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def _roots(environ: dict[str, str] | None = None) -> list[Path]:
    env = environ if environ is not None else os.environ
    try:
        values = json.loads(env.get(CORPUS_ENV, "[]"))
    except json.JSONDecodeError:
        return []
    if not isinstance(values, list):
        return []
    roots = [Path(str(value)).resolve() for value in values]
    return roots if all(root.is_dir() for root in roots) else []


def _scratch(environ: dict[str, str] | None = None) -> Path | None:
    env = environ if environ is not None else os.environ
    raw = env.get(SCRATCH_ENV)
    if not raw:
        return None
    root = Path(raw).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _relative_file(root: Path, raw: Any, *, must_exist: bool = True) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("path must be a non-empty relative path")
    rel = Path(raw)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("path must stay inside its declared root")
    path = (root / rel).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("path resolves outside its declared root") from exc
    if must_exist and (not path.is_file() or path.is_symlink()):
        raise ValueError(f"file is missing or not regular: {rel}")
    return path


def _root_at(roots: list[Path], value: Any) -> tuple[int, Path]:
    try:
        if value is None and len(roots) == 1:
            return 0, roots[0]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError
        index = value
        if index < 0:
            raise IndexError
        root = roots[index]
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"root must select one of 0..{len(roots) - 1}") from exc
    return index, root


def call_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    scratch: Path | None,
    roots: list[Path],
    audit_path: Path | None = None,
) -> dict[str, Any]:
    if scratch is None or not roots:
        return _result("scratch/corpus configuration is missing", is_error=True)
    try:
        if name == "list_corpus":
            result = _list_corpus(arguments, roots)
        elif name == "search_corpus":
            result = _search_corpus(arguments, roots)
        elif name == "read_corpus":
            result = _read_corpus(arguments, roots)
        elif name == "write_file":
            result = _write_file(arguments, scratch)
        elif name == "run":
            result = _run(arguments, scratch, roots)
        else:
            return _result(f"unknown tool: {name}", is_error=True)
    except (OSError, ValueError, re.error) as exc:
        result = _result(str(exc), is_error=True)
    _audit(name, arguments, result, environ=os.environ, audit_path=audit_path)
    return result


def _safe_glob(raw: Any) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("glob must be non-empty")
    pattern = raw.strip()
    if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        raise ValueError("glob must stay inside the selected corpus root")
    return pattern


def _bounded_int(
    value: Any, *, default: int, minimum: int, maximum: int, name: str
) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in {minimum}..{maximum}")
    return value


def _scratch_usage(root: Path) -> int:
    """Count scratch bytes with bounded work and without following symlinks."""
    total = 0
    entry_count = 0
    pending = [(root, 0)]
    while pending:
        directory, depth = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                entry_count += 1
                if entry_count > MAX_SCRATCH_ENTRIES:
                    raise ValueError(
                        f"scratch exceeds {MAX_SCRATCH_ENTRIES} filesystem entries"
                    )
                if entry.is_dir(follow_symlinks=False):
                    if depth >= MAX_SCRATCH_DEPTH:
                        raise ValueError(
                            f"scratch exceeds maximum directory depth {MAX_SCRATCH_DEPTH}"
                        )
                    pending.append((Path(entry.path), depth + 1))
                elif entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
                    if total > MAX_SCRATCH_BYTES:
                        return total
    return total


def _list_corpus(arguments: dict[str, Any], roots: list[Path]) -> dict[str, Any]:
    index, root = _root_at(roots, arguments.get("root"))
    pattern = _safe_glob(arguments.get("glob"))
    limit = _bounded_int(
        arguments.get("limit"), default=MAX_LIST, minimum=1,
        maximum=MAX_LIST, name="limit"
    )
    matches: list[str] = []
    truncated = False
    for path in sorted(root.glob(pattern)):
        if not path.is_file() or path.is_symlink():
            continue
        matches.append(path.relative_to(root).as_posix())
        if len(matches) >= limit:
            truncated = True
            break
    return _result(json.dumps({"root": index, "glob": pattern, "files": matches,
                               "truncated": truncated}, sort_keys=True))


def _search_corpus(arguments: dict[str, Any], roots: list[Path]) -> dict[str, Any]:
    index, root = _root_at(roots, arguments.get("root"))
    glob = _safe_glob(arguments.get("glob"))
    pattern = arguments.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("pattern must be a non-empty regex")
    if len(pattern.encode()) > 4_000:
        raise ValueError("pattern exceeds 4000 bytes")
    limit = _bounded_int(
        arguments.get("limit"), default=500, minimum=1,
        maximum=MAX_SEARCH_MATCHES, name="limit"
    )
    matches: list[dict[str, Any]] = []
    truncated = False
    scanned_bytes = 0
    scanned_files = 0
    for path in sorted(root.glob(glob)):
        if not path.is_file() or path.is_symlink():
            continue
        size = path.stat().st_size
        if scanned_bytes + size > MAX_SEARCH_BYTES:
            truncated = True
            break
        scanned_bytes += size
        scanned_files += 1
        with path.open(errors="replace") as file:
            for lineno, line in enumerate(file, 1):
                if pattern in line:
                    matches.append({"path": path.relative_to(root).as_posix(),
                                    "line": lineno, "text": line.rstrip()[:4000]})
                    if len(matches) >= limit:
                        truncated = True
                        break
        if truncated:
            break
    return _result(json.dumps({"root": index, "glob": glob, "pattern": pattern,
                               "matches": matches, "truncated": truncated,
                               "scanned_bytes": scanned_bytes,
                               "scanned_files": scanned_files}, sort_keys=True))


def _read_corpus(arguments: dict[str, Any], roots: list[Path]) -> dict[str, Any]:
    index, root = _root_at(roots, arguments.get("root"))
    path = _relative_file(root, arguments.get("path"))
    offset = _bounded_int(
        arguments.get("offset"), default=0, minimum=0,
        maximum=(1 << 63) - 1, name="offset"
    )
    limit = _bounded_int(
        arguments.get("limit"), default=200_000, minimum=1,
        maximum=MAX_READ_BYTES, name="limit"
    )
    with path.open("rb") as file:
        file.seek(offset)
        data = file.read(limit)
        more = bool(file.read(1))
    return _result(json.dumps({
        "root": index,
        "path": path.relative_to(root).as_posix(),
        "offset": offset,
        "bytes": len(data),
        "next_offset": offset + len(data) if more else None,
        "content": data.decode(errors="replace"),
    }, sort_keys=True))


def _write_file(arguments: dict[str, Any], scratch: Path) -> dict[str, Any]:
    content = arguments.get("content")
    if not isinstance(content, str):
        raise ValueError("content must be a string")
    if len(content.encode()) > MAX_WRITE_BYTES:
        raise ValueError(f"content exceeds {MAX_WRITE_BYTES} bytes")
    raw_path = arguments.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("path must be a non-empty relative path")
    rel = Path(raw_path)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts or len(rel.parts) > 32:
        raise ValueError("path must stay inside the scratch root")
    existing = scratch / rel
    previous_size = (
        existing.stat().st_size
        if existing.is_file() and not existing.is_symlink()
        else 0
    )
    projected = _scratch_usage(scratch) - previous_size + len(content.encode())
    if projected > MAX_SCRATCH_BYTES:
        raise ValueError(f"scratch aggregate would exceed {MAX_SCRATCH_BYTES} bytes")

    # Resolve each directory through a no-follow dirfd.  A concurrently running
    # scratch program may create symlinks under /work; ordinary Path.resolve +
    # write would otherwise have a TOCTOU escape in the host MCP process.
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    current_fd = os.open(scratch, flags)
    try:
        for part in rel.parts[:-1]:
            try:
                os.mkdir(part, mode=0o700, dir_fd=current_fd)
            except FileExistsError:
                pass
            next_flags = flags | getattr(os, "O_NOFOLLOW", 0)
            next_fd = os.open(part, next_flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        leaf = rel.parts[-1]
        tmp = f".{leaf}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
        write_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(tmp, write_flags, 0o600, dir_fd=current_fd)
        try:
            with os.fdopen(fd, "w") as file:
                file.write(content)
        except BaseException:
            try:
                os.unlink(tmp, dir_fd=current_fd)
            except FileNotFoundError:
                pass
            raise
        os.replace(tmp, leaf, src_dir_fd=current_fd, dst_dir_fd=current_fd)
    finally:
        os.close(current_fd)
    return _result(json.dumps({"path": rel.as_posix(),
                               "bytes": len(content.encode())}, sort_keys=True))


def _run(arguments: dict[str, Any], scratch: Path, roots: list[Path]) -> dict[str, Any]:
    argv = arguments.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or len(argv) > 64
        or not all(isinstance(arg, str) and arg and "\x00" not in arg for arg in argv)
    ):
        raise ValueError("argv must contain 1..64 non-empty strings")
    timeout = _bounded_int(
        arguments.get("timeout_s"), default=30, minimum=1,
        maximum=MAX_TIMEOUT_S, name="timeout_s"
    )
    if _scratch_usage(scratch) > MAX_SCRATCH_BYTES:
        raise ValueError(f"scratch aggregate exceeds {MAX_SCRATCH_BYTES} bytes")
    unshare = shutil.which("unshare")
    python = sys.executable
    if not unshare or not python:
        raise ValueError("offline scratch sandbox requires unshare and Python")
    jail = Path(tempfile.mkdtemp(prefix=".jail-", dir=str(scratch.parent)))
    stdout_path = jail / "stdout"
    stderr_path = jail / "stderr"
    command = [
        unshare,
        "--user",
        "--map-root-user",
        "--net",
        "--mount",
        "--pid",
        "--fork",
        python,
        str(Path(__file__).resolve()),
        "--jail-run",
        str(jail / "root"),
        str(scratch),
        json.dumps([str(root) for root in roots]),
        "--",
        *argv,
    ]
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                return _result(f"scratch program timed out after {timeout}s", is_error=True)
        stdout_data = stdout_path.read_bytes()
        stderr_data = stderr_path.read_bytes()
        stdout = stdout_data[:MAX_RESULT_BYTES].decode(errors="replace")
        stderr = stderr_data[:MAX_RESULT_BYTES].decode(errors="replace")
        scratch_bytes = _scratch_usage(scratch)
        over_limit = scratch_bytes > MAX_SCRATCH_BYTES
        return _result(json.dumps({"argv": argv, "returncode": proc.returncode,
                                   "stdout": stdout, "stderr": stderr,
                                   "stdout_bytes": len(stdout_data),
                                   "stderr_bytes": len(stderr_data),
                                   "stdout_truncated": len(stdout_data) > MAX_RESULT_BYTES,
                                   "stderr_truncated": len(stderr_data) > MAX_RESULT_BYTES,
                                   "scratch_bytes": scratch_bytes,
                                   "scratch_over_limit": over_limit}, sort_keys=True),
                       is_error=proc.returncode != 0 or over_limit)
    finally:
        shutil.rmtree(jail, ignore_errors=True)


def _mount(source: Path, target: Path, *, readonly: bool) -> None:
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(["mount", "--bind", str(source), str(target)], check=True)
    if readonly:
        subprocess.run(["mount", "-o", "remount,ro,bind", str(target)], check=True)


def _mount_file(source: Path, target: Path, *, readonly: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.touch()
    subprocess.run(["mount", "--bind", str(source), str(target)], check=True)
    if readonly:
        subprocess.run(["mount", "-o", "remount,ro,bind", str(target)], check=True)


def jail_run(root: Path, scratch: Path, roots: list[Path], argv: list[str]) -> int:
    """Construct the isolated filesystem after ``unshare`` and exec the tool."""
    subprocess.run(["mount", "--make-rprivate", "/"], check=True)
    root.mkdir(parents=True, exist_ok=True)
    for name in ("usr", "bin", "lib", "lib64"):
        source = Path("/") / name
        if source.exists():
            _mount(source, root / name, readonly=True)
    for name, readonly in (("null", False), ("zero", False), ("urandom", True)):
        source = Path("/dev") / name
        if source.exists():
            _mount_file(source, root / "dev" / name, readonly=readonly)
    _mount(scratch, root / "work", readonly=False)
    for index, corpus in enumerate(roots):
        _mount(corpus, root / "corpus" / str(index), readonly=True)
    (root / "tmp").mkdir(mode=0o1777)
    (root / "tmp").chmod(0o1777)
    (root / "proc").mkdir()
    subprocess.run(["mount", "-t", "proc", "proc", str(root / "proc")], check=True)
    resource.setrlimit(resource.RLIMIT_CPU, (MAX_TIMEOUT_S + 2, MAX_TIMEOUT_S + 2))
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_SCRATCH_BYTES, MAX_SCRATCH_BYTES))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    resource.setrlimit(resource.RLIMIT_NPROC, (128, 128))
    os.chroot(root)
    os.chdir("/work")
    _drop_all_capabilities()
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/work",
        "TMPDIR": "/tmp",
        "LANG": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "WMA_CORPUS_ROOTS": json.dumps([f"/corpus/{i}" for i in range(len(roots))]),
    }
    os.execvpe(argv[0], argv, env)
    return 127


class _CapHeader(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]


class _CapData(ctypes.Structure):
    _fields_ = [
        ("effective", ctypes.c_uint32),
        ("permitted", ctypes.c_uint32),
        ("inheritable", ctypes.c_uint32),
    ]


def _drop_all_capabilities() -> None:
    """Irreversibly remove namespace-root privilege before executing model code."""
    libc = ctypes.CDLL(None, use_errno=True)

    def prctl(option: int, arg2: int = 0) -> None:
        if libc.prctl(option, arg2, 0, 0, 0) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))

    # Make uid 0 non-special across this and subsequent execs, prevent gaining
    # privilege through executable metadata, clear ambient caps, then remove
    # every capability from the bounding and process sets.
    securebits = (
        SECBIT_NOROOT
        | SECBIT_NOROOT_LOCKED
        | SECBIT_NO_SETUID_FIXUP
        | SECBIT_NO_SETUID_FIXUP_LOCKED
    )
    prctl(PR_SET_SECUREBITS, securebits)
    prctl(PR_SET_NO_NEW_PRIVS, 1)
    prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_CLEAR_ALL)
    for capability in range(64):
        if libc.prctl(PR_CAPBSET_DROP, capability, 0, 0, 0) != 0:
            error = ctypes.get_errno()
            # EINVAL means the running kernel has fewer capability numbers.
            if error != 22:
                raise OSError(error, os.strerror(error))

    header = _CapHeader(LINUX_CAPABILITY_VERSION_3, 0)
    data = (_CapData * 2)()
    if libc.capset(ctypes.byref(header), ctypes.byref(data)) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def probe_sandbox(scratch: Path, roots: list[Path]) -> None:
    """Prove the run jail's load-bearing properties or raise fail-closed.

    The probe deliberately attempts a corpus write, a corpus remount, a fresh
    mount, a second chroot, and an external network connection.  It also checks
    the post-exec Linux capability sets instead of assuming that ``unshare``
    implies a safe child.
    """
    probe = r'''
import json, os, pathlib, socket, subprocess

status = {}
for line in pathlib.Path("/proc/self/status").read_text().splitlines():
    if ":" in line:
        key, value = line.split(":", 1)
        status[key] = value.strip()

checks = {
    "cap_eff_zero": int(status.get("CapEff", "1"), 16) == 0,
    "cap_prm_zero": int(status.get("CapPrm", "1"), 16) == 0,
    "cap_bnd_zero": int(status.get("CapBnd", "1"), 16) == 0,
    "cap_amb_zero": int(status.get("CapAmb", "1"), 16) == 0,
    "no_new_privs": status.get("NoNewPrivs") == "1",
    "host_hidden": not pathlib.Path("/rmeng_data").exists() and not pathlib.Path("/home").exists(),
}

work_probe = pathlib.Path("/work/.probe-write")
try:
    work_probe.write_text("ok")
    checks["scratch_writable"] = work_probe.read_text() == "ok"
finally:
    work_probe.unlink(missing_ok=True)

corpus_probe = pathlib.Path("/corpus/0/.awm-forbidden-probe")
try:
    corpus_probe.write_text("forbidden")
    checks["corpus_readonly"] = False
    corpus_probe.unlink(missing_ok=True)
except OSError:
    checks["corpus_readonly"] = True

checks["remount_blocked"] = subprocess.run(
    ["mount", "-o", "remount,rw,bind", "/corpus/0"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
).returncode != 0
mountpoint = pathlib.Path("/work/mount-probe")
mountpoint.mkdir(exist_ok=True)
checks["mount_blocked"] = subprocess.run(
    ["mount", "-t", "tmpfs", "tmpfs", str(mountpoint)],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
).returncode != 0
try:
    os.chroot("/work")
    checks["chroot_blocked"] = False
except OSError:
    checks["chroot_blocked"] = True

try:
    sock = socket.socket()
    sock.settimeout(0.25)
    try:
        sock.connect(("1.1.1.1", 53))
        checks["network_blocked"] = False
    except OSError:
        checks["network_blocked"] = True
    finally:
        sock.close()
except OSError:
    checks["network_blocked"] = True

print(json.dumps(checks, sort_keys=True))
raise SystemExit(0 if all(checks.values()) else 23)
'''
    result = _run({"argv": ["python3", "-c", probe], "timeout_s": 15}, scratch, roots)
    if result.get("isError"):
        detail = result.get("content") or result
        raise RuntimeError(f"WMA scratch sandbox self-test failed: {detail}")
    try:
        outer = json.loads(result["content"][0]["text"])
        checks = json.loads(outer["stdout"].strip().splitlines()[-1])
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"WMA scratch sandbox self-test was malformed: {result}") from exc
    if not checks or not all(checks.values()):
        raise RuntimeError(f"WMA scratch sandbox self-test failed: {checks}")


def _audit(
    name: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
    *,
    environ: dict[str, str],
    audit_path: Path | None = None,
) -> None:
    raw = str(audit_path) if audit_path is not None else environ.get(AUDIT_ENV)
    if not raw:
        return
    row = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "tool": name,
        "arguments": arguments,
        "is_error": bool(result.get("isError")),
        "result": result,
    }
    path = Path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        file.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _audit_server(event: str, *, audit_path: Path | None = None, **fields: Any) -> None:
    raw = str(audit_path) if audit_path is not None else os.environ.get(AUDIT_ENV)
    if not raw:
        return
    row = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "event": event,
        **fields,
    }
    path = Path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        file.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def handle_message(
    message: dict[str, Any],
    *,
    scratch: Path | None,
    roots: list[Path],
    audit_path: Path | None = None,
) -> dict[str, Any] | None:
    if not isinstance(message, dict):
        return _error(None, -32600, "request must be an object")
    method = message.get("method")
    message_id = message.get("id")
    notification = "id" not in message
    if method == "initialize":
        raw_params = message.get("params")
        if raw_params is not None and not isinstance(raw_params, dict):
            return _error(message_id, -32602, "initialize params must be an object")
        params = raw_params or {}
        result: dict[str, Any] = {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        }
    elif method == "tools/list":
        result = {"tools": list(TOOLS)}
    elif method == "tools/call":
        raw_params = message.get("params")
        if raw_params is not None and not isinstance(raw_params, dict):
            return _error(message_id, -32602, "tool params/arguments must be objects")
        params = raw_params or {}
        raw_arguments = params.get("arguments")
        if raw_arguments is not None and not isinstance(raw_arguments, dict):
            return _error(message_id, -32602, "tool params/arguments must be objects")
        name = str(params.get("name") or "")
        if name not in TOOLS_BY_NAME:
            return _error(message_id, -32601, f"unknown tool: {name}")
        result = call_tool(
            name,
            params.get("arguments") or {},
            scratch=scratch,
            roots=roots,
            audit_path=audit_path,
        )
    elif notification:
        return None
    elif method == "ping":
        result = {}
    else:
        return _error(message_id, -32601, f"unknown method: {method}")
    if notification:
        return None
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def _error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id,
            "error": {"code": code, "message": message}}


def serve(stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    scratch = _scratch()
    roots = _roots()
    _audit_server(
        "server_start",
        pid=os.getpid(),
        scratch_configured=scratch is not None,
        root_count=len(roots),
    )
    for line in stdin:
        try:
            message = json.loads(line)
            if not isinstance(message, dict):
                continue
            _audit_server("rpc", method=message.get("method"), has_id="id" in message)
            response = handle_message(message, scratch=scratch, roots=roots)
        except Exception as exc:  # keep the model's server alive and make the failure actionable
            _audit_server("rpc_error", error=f"{type(exc).__name__}: {exc}")
            response = _error(None, -32603, str(exc))
        if response is not None:
            stdout.write(json.dumps(response, sort_keys=True) + "\n")
            stdout.flush()
    _audit_server("server_stop")
    return 0


@contextlib.contextmanager
def http_server(scratch: Path, roots: list[Path], audit_path: Path) -> Iterator[str]:
    """Serve the fixed MCP tools on an ephemeral loopback HTTP endpoint.

    Claude Code 2.1.251 in the PTB image starts stdio MCP commands with closed
    stdin.  Streamable HTTP avoids that runtime defect without exposing the
    service off-host.  The model still has no general web tool, and its own
    analysis programs execute in the separate networkless jail above.
    """
    scratch = scratch.resolve()
    roots = [root.resolve() for root in roots]
    audit_path = audit_path.resolve()
    route = "/mcp/" + secrets.token_urlsafe(32)

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        call_lock = threading.Lock()

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            if self.path != route:
                self.send_error(404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_error(400)
                return
            if size <= 0 or size > 4 * 1024 * 1024:
                self.send_error(413)
                return
            try:
                message = json.loads(self.rfile.read(size))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.send_error(400)
                return
            if not isinstance(message, dict):
                self.send_error(400)
                return
            _audit_server(
                "rpc",
                audit_path=audit_path,
                transport="http",
                method=message.get("method"),
                has_id="id" in message,
            )
            try:
                # Serializing calls makes scratch aggregate accounting and the
                # append-only server audit deterministic even if a client sends
                # concurrent tool requests.
                with self.call_lock:
                    response = handle_message(
                        message,
                        scratch=scratch,
                        roots=roots,
                        audit_path=audit_path,
                    )
            except Exception as exc:
                _audit_server(
                    "rpc_error", audit_path=audit_path, transport="http",
                    error=f"{type(exc).__name__}: {exc}"
                )
                response = _error(message.get("id"), -32603, "internal tool server error")
            if response is None:
                self.send_response(202)
                self.send_header("Content-Length", "0")
                self.send_header("MCP-Protocol-Version", PROTOCOL_VERSION)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return
            body = json.dumps(response, sort_keys=True).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("MCP-Protocol-Version", PROTOCOL_VERSION)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            # This stateless server sends no notifications; the Streamable HTTP
            # specification permits rejecting the optional SSE listener.
            self.send_response(405)
            self.send_header("Allow", "POST")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, _format: str, *args: Any) -> None:
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, name="awm-wma-mcp", daemon=True)
    thread.start()
    host, port = server.server_address
    _audit_server(
        "server_start",
        audit_path=audit_path,
        transport="http",
        host=host,
        port=port,
        scratch_configured=True,
        root_count=len(roots),
    )
    try:
        yield f"http://{host}:{port}{route}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        _audit_server("server_stop", audit_path=audit_path, transport="http")


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if args and args[0] == "--jail-run":
        marker = args.index("--")
        root = Path(args[1])
        scratch = Path(args[2])
        roots = [Path(value) for value in json.loads(args[3])]
        return jail_run(root, scratch, roots, args[marker + 1 :])
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
