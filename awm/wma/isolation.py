"""A fail-closed, file-allowlisted tool broker for WMA reviews.

The model CLI retains its API connection, but has no built-in tools. Its only
MCP server copies explicitly exported inputs and runs probes in children with
Linux Landlock filesystem rules and a seccomp network/process filter. A Python
library, shell child, or changed cache environment cannot escape those rules.
Neither the scientist's whole session nor model/dataset caches are exported.

This is deliberately a CPU/static-probe boundary. GPU devices, online dataset
downloads and the benchmark evaluator are unavailable. Supplying extra files
is a trusted harness operation, never a tool exposed to the model. Previously
contaminated exported text is outside the guarantee of filesystem isolation.
"""

from __future__ import annotations

import contextlib
import ctypes
import errno
import hashlib
import json
import os
import platform
import re
import resource
import shlex
import signal
import stat
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class IsolationError(RuntimeError):
    """The requested isolation boundary cannot be established."""


# Landlock ABI 3 covers filesystem truncation as well as read/write/execute.
# Network socket creation is blocked separately, including Unix sockets.
_FS_READ = (1 << 0) | (1 << 2) | (1 << 3)
_FS_ALL = (1 << 15) - 1
_MAX_INPUT_BYTES = 16 * 1024 * 1024
_MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024
_MAX_OUTPUT_BYTES = 1024 * 1024
_MAX_HISTORY_FILES = 512
_MAX_ADDRESS_SPACE = 512 * 1024 * 1024
_MAX_PROBE_TASKS = 128
_CODE_CONFIG_SUFFIXES = {".py", ".sh", ".bash", ".json", ".yaml", ".yml", ".toml"}


def _checked(result: int, operation: str) -> int:
    if result < 0:
        error = ctypes.get_errno()
        raise IsolationError(f"{operation}: {os.strerror(error)}")
    return result


def _restrict_files(read_paths: list[str], write_paths: list[str], list_paths: list[str] = ()) -> None:
    """Install an inherited Landlock ruleset in the dedicated probe child."""
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "aarch64"):
        raise IsolationError("WMA probes require Linux x86_64/aarch64 with Landlock ABI >= 3")
    libc = ctypes.CDLL(None, use_errno=True)
    abi = _checked(libc.syscall(444, 0, 0, 1), "query Landlock ABI")
    if abi < 3:
        raise IsolationError(f"Landlock ABI {abi} is insufficient; ABI >= 3 is required")

    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class PathBeneath(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    rules = Ruleset(_FS_ALL)
    fd = _checked(libc.syscall(444, ctypes.byref(rules), ctypes.sizeof(rules), 0), "create Landlock ruleset")
    try:
        grants = ([(p, _FS_READ) for p in read_paths] + [(p, _FS_ALL) for p in write_paths]
                  + [(p, 1 << 3) for p in list_paths])
        for path, rights in grants:
            target = Path(path).resolve(strict=True)
            if not target.is_dir():
                rights &= (1 << 0) | (1 << 1) | (1 << 2) | (1 << 14)
            parent_fd = os.open(target, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathBeneath(rights, parent_fd)
                _checked(libc.syscall(445, fd, 1, ctypes.byref(rule), 0), "add Landlock path rule")
            finally:
                os.close(parent_fd)
        _checked(libc.prctl(38, 1, 0, 0, 0), "set no_new_privs")
        _checked(libc.syscall(446, fd, 0), "enforce Landlock")
    finally:
        os.close(fd)


def _restrict_syscalls() -> None:
    """Deny sockets and ambient-process escape routes, including io_uring.

    Verify audit architecture and reject x32 syscalls instead of letting a
    compat syscall bypass the native-number denylist.
    """
    machine = platform.machine()
    if machine == "x86_64":
        arch = 0xC000003E
        clone_number = 56
        denied = [41, 42, 43, 49, 50, 53, 62, 101, 109, 112, 129, 155, 165, 166, 200, 234,
                  248, 249, 250, 272, 288, 297, 298, 304, 308, 310, 311, 312, 321, 323]
    elif machine == "aarch64":
        arch = 0xC00000B7
        clone_number = 220
        denied = [39, 40, 41, 97, 117, 129, 130, 131, 138, 154, 157, 198, 199, 200, 201,
                  202, 203, 217, 218, 219, 240, 241, 242, 265, 268, 270, 271, 272, 280, 282]
    else:
        raise IsolationError(f"unsupported seccomp architecture: {machine}")
    denied += [424, 425, 426, 427, 428, 429, 430, 431, 432, 438, 440, 442]

    class Filter(ctypes.Structure):
        _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte),
                    ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint32)]

    class Program(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(Filter))]

    # LD architecture; JEQ expected; KILL_PROCESS; LD syscall; reject x32.
    instructions = [(0x20, 0, 0, 4), (0x15, 1, 0, arch), (0x06, 0, 0, 0x80000000),
                    (0x20, 0, 0, 0), (0x45, 0, 1, 0x40000000), (0x06, 0, 0, 0x80000000)]
    # clone3 has pointer-valued flags that classic BPF cannot inspect. ENOSYS
    # permits libc's ordinary-process fallback through the checked clone path.
    instructions += [(0x15, 0, 1, 435), (0x06, 0, 0, 0x00050000 | errno.ENOSYS)]
    # Reject namespace creation and CLONE_PARENT. Ordinary children/threads
    # retain this policy and cannot leave the broker-owned process group.
    instructions += [(0x15, 0, 3, clone_number), (0x20, 0, 0, 16),
                     (0x45, 0, 1, 0x7E028000), (0x06, 0, 0, 0x00050000 | errno.EPERM),
                     (0x20, 0, 0, 0)]
    for number in sorted(set(denied)):
        instructions += [(0x15, 0, 1, number), (0x06, 0, 0, 0x00050000 | errno.EPERM)]
    instructions.append((0x06, 0, 0, 0x7FFF0000))
    filters = (Filter * len(instructions))(*(Filter(*row) for row in instructions))
    program = Program(len(filters), filters)
    libc = ctypes.CDLL(None, use_errno=True)
    _checked(libc.prctl(38, 1, 0, 0, 0), "set no_new_privs")
    _checked(libc.prctl(22, 2, ctypes.byref(program), 0, 0), "enforce seccomp")


def runtime_files() -> list[str]:
    """Python stdlib and OS executables/libraries; no site-packages or caches.

    Do not grant an interpreter installation or /usr wholesale: either may
    also contain downloaded datasets or the benchmark's evaluator package.
    """
    paths = [Path(sys.executable).resolve()]
    stdlib = Path(sysconfig.get_path("stdlib")).resolve()
    for item in stdlib.iterdir():
        if item.name not in {"site-packages", "dist-packages", "__pycache__"}:
            paths.append(item)
    for value in ("/bin/sh", "/bin/bash", "/usr/bin/env", "/usr/bin/cat", "/usr/bin/ls",
                  "/usr/bin/head", "/usr/bin/tail", "/usr/bin/grep", "/usr/bin/sed",
                  "/usr/bin/true", "/etc/ld.so.cache", "/dev/null", "/dev/urandom"):
        if Path(value).exists():
            paths.append(Path(value).resolve())
    # Shared objects only, not arbitrary package trees below /usr/lib.
    for directory in (Path("/lib"), Path("/lib64"), Path("/usr/lib"),
                      stdlib.parent, Path("/lib/x86_64-linux-gnu"), Path("/lib/aarch64-linux-gnu")):
        if directory.is_dir():
            for pattern in ("*.so", "*.so.*", "ld-*.so*"):
                paths.extend(p.resolve() for p in directory.glob(pattern) if p.is_file())
    return sorted({str(p) for p in paths})


def _cap_resource(which: int, soft: int, hard: int | None = None) -> None:
    """Never loosen an inherited limit, including a host's lower hard cap."""
    before_soft, before_hard = resource.getrlimit(which)
    hard = soft if hard is None else hard
    if before_hard != resource.RLIM_INFINITY:
        hard = min(hard, before_hard)
    if before_soft != resource.RLIM_INFINITY:
        soft = min(soft, before_soft)
    resource.setrlimit(which, (min(soft, hard), hard))


def _child(policy: dict[str, Any], command: str) -> None:
    scratch = policy["scratch"]
    os.chdir(scratch)
    _cap_resource(resource.RLIMIT_CORE, 0)
    _cap_resource(resource.RLIMIT_FSIZE, _MAX_OUTPUT_BYTES)
    _cap_resource(resource.RLIMIT_CPU, 60, 61)
    _cap_resource(resource.RLIMIT_NOFILE, 128)
    _cap_resource(resource.RLIMIT_AS, _MAX_ADDRESS_SPACE)
    # Preserve inherited NPROC. Linux counts the entire real UID, including
    # other Slurm jobs and tasks hidden by a PID namespace; an absolute 128
    # here prevents a loaded shared UID from even starting the probe shell.
    # The parent supervises this non-escaping process group separately.
    _restrict_files(policy["runtime"] + [policy["inputs"]], [scratch], policy["runtime_dirs"])
    _restrict_syscalls()
    # No model API credentials, home caches, LD_PRELOAD, PYTHONPATH or proxy.
    os.execve("/bin/sh", ["/bin/sh", "-c", command], {
        "PATH": f"{Path(sys.executable).resolve().parent}:/usr/bin:/bin",
        "HOME": scratch, "TMPDIR": scratch, "LANG": "C.UTF-8",
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HOME": f"{scratch}/hf-cache", "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
    })


def _group_tasks(group: int) -> int:
    """Count threads in our probe group; inspect no command lines or environments."""
    count = 0
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group:  # pgrp, after state and ppid
                count += int(fields[17])  # num_threads
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return count


def run_probe(policy: dict[str, Any], command: str, timeout: float = 30) -> dict[str, Any]:
    """Run a probe; never retry it without the OS boundary on failure."""
    if not isinstance(command, str) or len(command) > 64 * 1024:
        raise IsolationError("probe command must be text shorter than 64 KiB")
    timeout = max(0.1, min(float(timeout), 60))
    task_limit = policy.get("limits", {}).get("processes_per_probe", _MAX_PROBE_TASKS)
    if isinstance(task_limit, bool) or not isinstance(task_limit, int) or not 1 <= task_limit <= _MAX_PROBE_TASKS:
        raise IsolationError("invalid supervised probe task limit")
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(
            [sys.executable, "-I", str(Path(__file__).resolve()), "--child", policy["policy_file"]],
            stdin=subprocess.PIPE, stdout=out, stderr=err, text=True,
            close_fds=True, start_new_session=True,
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
        )
        try:
            import time
            deadline = time.monotonic() + timeout
            pending_input = command
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise IsolationError(f"probe exceeded {timeout:g} seconds")
                try:
                    proc.communicate(pending_input, timeout=min(0.05, remaining))
                    break
                except subprocess.TimeoutExpired:
                    pending_input = None
                    tasks = _group_tasks(proc.pid)
                    if tasks > task_limit:
                        raise IsolationError(f"probe exceeded supervised task limit {task_limit}") from None
                    if tasks == 0 and proc.poll() is None:
                        raise IsolationError("cannot account for the probe process group") from None
        finally:
            # A shell may exit while descendants retain its inherited policy.
            # Reap its process group so no probe keeps consuming CPU afterward.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.communicate()
        out.seek(0)
        err.seek(0)
        result = {"returncode": proc.returncode,
                  "stdout": out.read(_MAX_OUTPUT_BYTES).decode("utf-8", errors="replace"),
                  "stderr": err.read(_MAX_OUTPUT_BYTES).decode("utf-8", errors="replace")}
    if proc.returncode == 125:
        raise IsolationError(f"probe isolation unavailable: {result['stderr'].strip()}")
    return result


def _snapshot_file(source: Path, target: Path, expected_sha: str | None = None) -> dict[str, Any]:
    # Reject a symlink in any component, and never follow a raced final link.
    absolute = source.absolute()
    if any(p.is_symlink() for p in [absolute, *absolute.parents]):
        raise IsolationError(f"exported input must not contain symlinks: {source}")
    fd = os.open(absolute, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_INPUT_BYTES:
            raise IsolationError(f"exported input must be a regular file <= 16 MiB: {source}")
        content = stream.read(_MAX_INPUT_BYTES + 1)
        after = os.fstat(stream.fileno())
        if (info.st_size, info.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise IsolationError(f"exported input changed while copying: {source}")
    if len(content) > _MAX_INPUT_BYTES:
        raise IsolationError(f"exported input grew beyond 16 MiB: {source}")
    sha = hashlib.sha256(content).hexdigest()
    if expected_sha is not None and sha != expected_sha:
        raise IsolationError(f"exported input no longer matches its frozen hash: {source}")
    target.write_bytes(content)
    return {"source": str(absolute), "path": str(target),
            "sha256": sha, "bytes": len(content)}


def _confined_file(path: Path, root: Path) -> bool:
    """Allow ordinary files beneath a trusted root without following inner links."""
    path = path.absolute()
    root = root.resolve(strict=True)
    if not path.is_relative_to(root):
        return False
    current = path
    while current != root:
        if current.is_symlink():
            return False
        current = current.parent
    return path.is_file()


def collect_review_inputs(brief: Any) -> list[Path]:
    """Select bounded evidence exports, without traversing evaluation/data trees.

    A frozen current card is authoritative. Earlier *closed* cards may supply
    scientific feedback; live raw logs/evaluation files and later cards cannot.
    An operator-specified history directory explicitly delegates its top-level
    linked run roots, but never their parents or nested symlinks. Text content
    remains supplied evidence, not a claim of semantic decontamination.
    """
    from awm.exp_protocol.schema import get, load_card, plan_hash

    sources = [Path(brief.card_path)]
    expected = brief.extra.setdefault("probe_expected_hashes", {})
    selection = {"history_files_omitted": 0, "prior_cards": 0}
    session = Path(brief.session_dir).resolve(strict=True)
    match = re.fullmatch(r"exp-([0-9]+)", str(getattr(brief, "card_id", "")))
    if match and Path(brief.card_path).suffix in {".yaml", ".yml"}:
        card = load_card(brief.card_path)
        original = session / "memory" / "cards" / f"{brief.card_id}.yaml"
        snapshot_parent = Path(brief.card_path).absolute().parent
        frozen_lock = snapshot_parent / f"{Path(brief.card_path).stem}.lock.json"
        lock_path = frozen_lock if frozen_lock.is_file() else original.with_suffix(".lock.json")
        lock = None
        lock_root = snapshot_parent if lock_path == frozen_lock else session
        if _confined_file(lock_path, lock_root):
            lock = json.loads(lock_path.read_text())
            if lock.get("plan_sha256") != plan_hash(card):
                raise IsolationError("lock no longer matches the frozen review card")
            sources.append(lock_path)
            preflight = lock_path.with_name(lock_path.name.replace(".lock.json", ".preflight.json"))
            if _confined_file(preflight, lock_root):
                sources.append(preflight)
        if lock and isinstance(lock.get("script"), dict):
            entry = lock["script"]
            path = Path(str(entry.get("path") or ""))
            if path.is_absolute() and path.is_relative_to("/home/ben/task"):
                # PTB scientist and private sidecar mount the same task at
                # these two known locations. Never map any other host path.
                path = session / path.relative_to("/home/ben/task")
            elif not path.is_absolute():
                path = session / path
            sha = entry.get("sha256")
            if (path.suffix in _CODE_CONFIG_SUFFIXES and isinstance(sha, str)
                    and re.fullmatch(r"[a-f0-9]{64}", sha) and _confined_file(path, session)):
                sources.append(path)
                expected[str(path.absolute())] = sha
        cards = session / "memory" / "cards"
        for previous in sorted(cards.glob("exp-*.yaml")):
            previous_id = re.fullmatch(r"exp-([0-9]+)", previous.stem)
            if (not previous_id or int(previous_id[1]) >= int(match[1])
                    or not _confined_file(previous, session)):
                continue
            prior = load_card(previous)
            if get(prior, "conclusion.decision") and get(prior, "result.execution"):
                sources.append(previous)
                selection["prior_cards"] += 1
                if selection["prior_cards"] >= 128:
                    break
        # list_inputs supplies a derived prior-card index. A mutable session
        # index.md may already summarize the current or a future card's result.
    history = getattr(brief, "history_dir", None)
    if history is not None:
        root = Path(history).resolve(strict=True)
        roots = [root]
        # The operator creates these top-level links (offline replay's train
        # split or the online curated corpus). No recursive link expansion.
        for entry in sorted(root.iterdir()):
            if entry.is_dir() and entry.name not in {"memory", "cards"}:
                roots.append(entry.resolve(strict=True))
        historical = []
        for run in dict.fromkeys(roots):
            if run == session:
                continue
            for base in (run, run / "memory", run / "memory" / "cards"):
                if not base.is_dir() or not base.resolve().is_relative_to(run):
                    continue
                for path in [base / "index.md", *sorted(base.glob("exp-*.yaml"))]:
                    if _confined_file(path, run):
                        historical.append(path)
        historical = list(dict.fromkeys(historical))
        selection["history_files_omitted"] = max(0, len(historical) - _MAX_HISTORY_FILES)
        sources += historical[:_MAX_HISTORY_FILES]
    # Explicit operator exports are independent of auto-selection. They are
    # still copied, bounded and rejected if a path contains a symlink.
    sources += [Path(p) for p in brief.extra.get("probe_files", [])]
    brief.extra["probe_input_selection"] = selection
    return list(dict.fromkeys(sources))


@dataclass
class IsolatedTools:
    argv: list[str]
    cwd: Path
    prompt_suffix: str
    policy: dict[str, Any]


@contextlib.contextmanager
def isolated_tools(brief: Any, argv: list[str], backend_name: str) -> Iterator[IsolatedTools]:
    """Configure a Claude CLI whose only tool surface is the isolated broker.

    Pass the yielded cwd/argv and append prompt_suffix to the model prompt.
    On normal context exit, publish the broker's JSON result to verdict_path.
    ``probe_files`` is an explicit trusted harness export list; card paths,
    ``allowed_roots`` and history roots never implicitly grant file access.
    """
    if backend_name != "claude":
        raise IsolationError(f"isolated WMA tools are unavailable for backend {backend_name!r}")
    with tempfile.TemporaryDirectory(prefix="awm-wma-isolated-") as temporary:
        root = Path(temporary)
        inputs, scratch, agent = root / "inputs", root / "scratch", root / "agent"
        for directory in (inputs, scratch, agent):
            directory.mkdir()
        sources = collect_review_inputs(brief)
        # The harness resolves its owned skill directory; reject links within it.
        skill = Path(brief.skill_dir).resolve(strict=True)
        sources += [p for p in sorted(skill.rglob("*")) if p.is_file() or p.is_symlink()]
        files = []
        total_bytes = 0
        for i, source in enumerate(dict.fromkeys(sources)):
            expected = brief.extra.get("probe_expected_hashes", {}).get(str(source.absolute()))
            files.append(_snapshot_file(source, inputs / f"{i:03d}-{source.name}", expected))
            total_bytes += files[-1]["bytes"]
            if total_bytes > _MAX_SNAPSHOT_BYTES:
                raise IsolationError("exported input snapshot exceeds 64 MiB")
        policy = {"version": 1, "inputs": str(inputs), "scratch": str(scratch),
                  "result": str(root / "result.json"), "runtime": runtime_files(),
                  "runtime_dirs": [str(Path(sysconfig.get_path("stdlib")).resolve())],
                  "selection": brief.extra["probe_input_selection"],
                  "limits": {"address_space_bytes": _MAX_ADDRESS_SPACE,
                             "processes_per_probe": _MAX_PROBE_TASKS,
                             "process_limit_enforcement": "process-group supervisor (50ms poll; transient overshoot possible)",
                             "uid_task_limit": "inherited unchanged", "wall_seconds": 60,
                             "cpu_seconds": 60, "file_bytes": _MAX_OUTPUT_BYTES},
                  "files": files, "policy_file": str(root / "policy.json")}
        Path(policy["policy_file"]).write_text(json.dumps(policy))
        check = run_probe(policy, shlex.quote(str(Path(sys.executable).resolve()))
                          + " -I -c 'import json, socket; print(1)'", timeout=10)
        if check["returncode"] != 0:
            raise IsolationError(f"isolated probe self-check failed: {check['stderr']}")
        config = root / "mcp.json"
        config.write_text(json.dumps({"mcpServers": {"wma_probe": {
            "command": sys.executable,
            "args": ["-I", str(Path(__file__).resolve()), "--serve", policy["policy_file"]],
        }}}))
        # Drop directory grants inherited from the legacy CLI setup. They are
        # unnecessary with all built-ins disabled and can trigger context reads.
        cleaned = []
        skip = False
        for argument in argv:
            if skip:
                skip = False
                continue
            if argument == "--add-dir":
                skip = True
                continue
            cleaned.append(argument)
        cleaned += ["--bare", "--disable-slash-commands", "--tools", "",
                    "--strict-mcp-config", "--mcp-config", str(config)]
        suffix = (
            "\n\nEnforced WMA tool boundary: only mcp__wma_probe tools are available. "
            "Use list_inputs to see immutable exported inputs, read_file to read an exported "
            "source path, and run for CPU/static probes over the copied input paths. "
            "The full session, raw historical artifacts, installed ML packages, model/dataset caches, GPU devices "
            "and network are unavailable to probes. Do not infer a scheme is invalid from missing "
            "sandbox capability: record the untested claim. Use write_result to submit the final "
            "JSON object; the harness publishes it to the requested output path.\n"
            f"Input selection: {json.dumps(policy['selection'])}. Earlier closed cards and curated "
            "history card/index exports are available only if listed; use list_inputs as their index.\n"
        )
        yield IsolatedTools(cleaned, agent, suffix, policy)
        result = Path(policy["result"])
        if result.is_file():
            brief.verdict_path.write_bytes(result.read_bytes())


def _tools() -> list[dict[str, Any]]:
    return [
        {"name": "list_inputs", "description": "List frozen exported input files with their hashes.",
         "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
        {"name": "read_file", "description": "Read one exported source path or its copied path.",
         "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}},
                         "required": ["path"], "additionalProperties": False}},
        {"name": "run", "description": "Run a CPU/static shell probe with only copied inputs and stdlib; no network.",
         "inputSchema": {"type": "object", "properties": {"command": {"type": "string"},
                         "timeout_s": {"type": "number", "minimum": 0.1, "maximum": 60}},
                         "required": ["command"], "additionalProperties": False}},
        {"name": "write_result", "description": "Submit the final verdict/comparison JSON object to the harness.",
         "inputSchema": {"type": "object", "properties": {"result": {"type": "object"}},
                         "required": ["result"], "additionalProperties": False}},
    ]


def broker_call(policy: dict[str, Any], name: str, arguments: dict[str, Any]) -> Any:
    if name == "list_inputs":
        return policy["files"]
    if name == "read_file":
        path = arguments.get("path")
        entry = next((f for f in policy["files"] if path in (f["source"], f["path"])), None)
        if entry is None:
            raise IsolationError("path is not an exported input")
        return Path(entry["path"]).read_text(errors="replace")
    if name == "run":
        return run_probe(policy, arguments.get("command"), arguments.get("timeout_s", 30))
    if name == "write_result":
        result = arguments.get("result")
        if not isinstance(result, dict):
            raise IsolationError("result must be a JSON object")
        content = json.dumps(result, ensure_ascii=False, allow_nan=False)
        if len(content.encode()) > _MAX_OUTPUT_BYTES:
            raise IsolationError("result exceeds 1 MiB")
        Path(policy["result"]).write_text(content)
        return {"written": True}
    raise IsolationError(f"unknown tool: {name}")


def _serve(policy: dict[str, Any]) -> None:
    """Minimal MCP stdio transport; no extra dependency or ambient tools."""
    for line in sys.stdin:
        try:
            if len(line) > _MAX_INPUT_BYTES:
                raise IsolationError("MCP request exceeds limit")
            request = json.loads(line)
            if "id" not in request:
                continue
            method, params = request.get("method"), request.get("params", {})
            if method == "initialize":
                result = {"protocolVersion": params.get("protocolVersion", "2024-11-05"),
                          "capabilities": {"tools": {}},
                          "serverInfo": {"name": "wma_probe", "version": "1"}}
            elif method == "tools/list":
                result = {"tools": _tools()}
            elif method == "tools/call":
                try:
                    value = broker_call(policy, params.get("name"), params.get("arguments", {}))
                    result = {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}]}
                except (IsolationError, OSError, ValueError) as exc:
                    result = {"isError": True, "content": [{"type": "text", "text": str(exc)}]}
            elif method == "ping":
                result = {}
            else:
                response = {"jsonrpc": "2.0", "id": request["id"],
                            "error": {"code": -32601, "message": "method not found"}}
                print(json.dumps(response), flush=True)
                continue
            print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
        except (ValueError, TypeError, IsolationError) as exc:
            print(json.dumps({"jsonrpc": "2.0", "id": None,
                              "error": {"code": -32700, "message": str(exc)}}), flush=True)


if __name__ == "__main__":
    try:
        if len(sys.argv) != 3 or sys.argv[1] not in ("--child", "--serve"):
            raise IsolationError("use --child POLICY or --serve POLICY")
        _policy = json.loads(Path(sys.argv[2]).read_text())
        if sys.argv[1] == "--child":
            _child(_policy, sys.stdin.read(64 * 1024 + 1))
        else:
            _serve(_policy)
    except (IsolationError, OSError, ValueError) as exc:
        print(f"WMA isolation failed: {exc}", file=sys.stderr)
        sys.exit(125)
