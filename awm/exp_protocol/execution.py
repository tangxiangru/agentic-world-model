"""Foreground execution evidence for one locked command, not scientific completion.

No scheduler, retries, card mutation, or dependent command execution. A wait result
describes the direct child; its scientific descendants must stay in the foreground.
Unlisted dependencies and uncooperative processes are not certified by this wrapper.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import stat
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from . import lock, preflight, schema

ATTEMPT_SCHEMA = "awm-execution-attempt-v1"
RESERVED_ENV = ("AWM_EXP_ATTEMPT_ID", "AWM_EXP_ATTEMPT_DIR")
MAX_OUTPUT_FILES = 10000
MAX_OUTPUT_BYTES = 1 << 40


class ExecutionError(ValueError):
    """The wrapper could not establish its declared execution evidence."""


class ExecutionInterrupted(KeyboardInterrupt):
    def __init__(self, signum):
        super().__init__(f"execution observer interrupted by signal {signum}")
        self.signum = signum


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _encoded(value) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()


def _process_identity(pid: int) -> dict | None:
    """Read Linux birth identity; PID existence alone cannot identify an old child."""
    if type(pid) is not int or pid < 1:
        return None
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rfind(")") + 2 :].split()
        return {
            "pid": pid,
            "start_ticks": int(fields[19]),
            "state": fields[0],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        }
    except (OSError, ValueError, IndexError):
        return None


def _reject_unresolved(attempts: Path) -> None:
    if not attempts.exists():
        return
    for attempt in attempts.iterdir():
        if not attempt.is_dir() or attempt.is_symlink():
            raise ExecutionError("unexpected attempt entry; investigate before executing")
        try:
            finished = json.loads((attempt / "finish.json").read_text())
        except (OSError, ValueError):
            finished = None
        if (
            isinstance(finished, dict)
            and finished.get("schema_version") == ATTEMPT_SCHEMA
            and finished.get("attempt_id") == attempt.name
        ) and (
            type(finished.get("observed_returncode")) is int
            or finished.get("spawn_attempted") is False
        ):
            continue
        detail = "direct-child identity/exit unknown"
        try:
            prior = json.loads((attempt / "process.json").read_text()).get("identity")
            current = _process_identity(prior["pid"]) if isinstance(prior, dict) else None
            if current is not None and all(
                current.get(key) == prior.get(key) for key in ("pid", "start_ticks", "boot_id")
            ):
                detail = f"same child birth identity still present, state={current['state']}"
            elif prior is not None:
                detail = "old child birth identity is absent/different; descendant outcome remains unknown"
        except (OSError, ValueError, TypeError, KeyError):
            pass
        raise ExecutionError(
            f"unresolved prior attempt {attempt.name}: {detail}; investigate and close honestly"
        )


def _write_once(path: Path, value) -> None:
    encoded = _encoded(value)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def _claim(card_path: Path):
    """The held OS lock, not the sidecar's existence, indicates a live invocation."""
    fd = os.open(
        card_path.with_suffix(".execution.lock"), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
    )
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ExecutionError("this card already has an active guarded invocation") from exc
        yield
    finally:
        os.close(fd)


def _live_plan(card_path: Path, session: Path) -> tuple[dict, dict, dict]:
    card = schema.load_card(card_path)
    report = schema.validate_plan(card, session)
    if not report.ok:
        raise ExecutionError(report.render())
    if schema.get(card, "conclusion.decision"):
        raise ExecutionError("card already has a conclusion; do not silently re-execute it")
    info = lock.read_lock(card_path)
    if (
        not isinstance(info, dict)
        or info.get("schema_version") != lock.LOCK_SCHEMA
        or info.get("card_id") != card.get("card_id")
        or not isinstance(info.get("locked_at"), str)
        or not info["locked_at"]
    ):
        raise ExecutionError("a matching prior lock is required")
    if card_path.stem != card["card_id"]:
        raise ExecutionError("card filename and identity differ")
    integrity = lock.verify_lock(card_path, card)
    # Missing source warnings are useful for historical closure, never live launch proof.
    if integrity.problems:
        raise ExecutionError(integrity.render())
    script = schema.get(card, "setup.command.script")
    entry = info.get("script")
    if (
        not isinstance(script, str)
        or not Path(script).is_absolute()
        or not isinstance(entry, dict)
        or entry.get("path") != script
        or not entry.get("sha256")
        or not Path(script).is_file()
    ):
        raise ExecutionError("run needs an absolute, existing script pinned in the prior lock")
    data = schema.get(card, "setup.data") or []
    pinned_data = info.get("data")
    if not isinstance(pinned_data, list) or len(pinned_data) != len(data):
        raise ExecutionError("lock does not cover every declared data file")
    for declared, pinned in zip(data, pinned_data):
        if not Path(declared["path"]).is_absolute():
            raise ExecutionError(
                "run needs absolute data paths; v2 does not pin a relative path's lock caller cwd"
            )
        if (
            not isinstance(pinned, dict)
            or pinned.get("path") != declared["path"]
            or not pinned.get("sha256")
            or not Path(pinned["path"]).is_file()
        ):
            raise ExecutionError("every declared data file needs an existing pinned hash")
    cwd = Path(schema.get(card, "setup.command.cwd"))
    if not cwd.is_absolute() or not cwd.is_dir():
        raise ExecutionError("run needs an existing absolute command cwd")
    argv = schema.get(card, "setup.command.argv")
    if not any(
        (Path(arg) if Path(arg).is_absolute() else cwd / arg).resolve() == Path(script).resolve()
        for arg in argv
    ):
        raise ExecutionError("the exact declared script must be an argv path, not a substring")
    env = schema.get(card, "setup.command.env")
    env = {} if env is None else env
    if not isinstance(env, dict) or any(
        not isinstance(k, str)
        or not k
        or "=" in k
        or "\0" in k
        or not isinstance(v, str)
        or "\0" in v
        for k, v in env.items()
    ):
        raise ExecutionError("command.env must map valid environment names to strings")
    if any(key in env for key in RESERVED_ENV):
        raise ExecutionError("command.env cannot override the wrapper's attempt identity")
    fresh = preflight.run_preflight(card, session)
    overrides = info.get("overrides") or {}
    failed = [
        item["check"]
        for item in fresh["results"]
        if item["status"] == "fail" and not str(overrides.get(item["check"], "")).strip()
    ]
    if failed:
        raise ExecutionError("fresh preflight failed: " + ", ".join(failed))
    late_integrity = lock.verify_lock(card_path, card)
    if late_integrity.problems:
        raise ExecutionError(late_integrity.render())
    return card, info, fresh


def _stop_child(child, signum=signal.SIGTERM):
    """Signal only the new session created for this exact direct child."""
    if child.poll() is None:
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass
        try:
            return child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            return child.wait()
    return child.returncode


def _snapshot(output: Path, reserved_identity: tuple, check_interrupt) -> dict:
    """Bounded identity snapshot; no symlink traversal or model-format validation."""
    entries = []
    total = 0
    objects = 0
    directories = []
    root_stat = output.lstat()
    if (
        not stat.S_ISDIR(root_stat.st_mode)
        or (root_stat.st_dev, root_stat.st_ino) != reserved_identity
    ):
        raise ExecutionError("fresh output directory was replaced")
    for directory, dirs, files, dir_fd in os.fwalk(output, follow_symlinks=False):
        check_interrupt()
        directory_stat = os.fstat(dir_fd)
        directories.append((Path(directory), directory_stat.st_dev, directory_stat.st_ino))
        objects += len(dirs) + len(files)
        if objects > MAX_OUTPUT_FILES:
            raise ExecutionError("fresh-output inventory limit exceeded; evidence is incomplete")
        for name in dirs + files:
            if stat.S_ISLNK(os.stat(name, dir_fd=dir_fd, follow_symlinks=False).st_mode):
                raise ExecutionError("fresh-output snapshot does not certify symlinks")
        for name in sorted(files):
            check_interrupt()
            path = Path(directory) / name
            before = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise ExecutionError(
                    "fresh-output snapshot requires ordinary, non-hardlinked files"
                )
            total += before.st_size
            if len(entries) >= MAX_OUTPUT_FILES or total > MAX_OUTPUT_BYTES:
                raise ExecutionError(
                    "fresh-output inventory limit exceeded; evidence is incomplete"
                )
            digest = hashlib.sha256()
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dir_fd)
            with os.fdopen(fd, "rb") as stream:
                opened = os.fstat(stream.fileno())
                if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                    raise ExecutionError("output changed while opening it")
                for block in iter(lambda: stream.read(1 << 20), b""):
                    check_interrupt()
                    digest.update(block)
                after = os.fstat(stream.fileno())
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise ExecutionError("output changed while hashing it")
            entries.append(
                {
                    "path": str(path.relative_to(output)),
                    "bytes": before.st_size,
                    "sha256": digest.hexdigest(),
                    "device": before.st_dev,
                    "inode": before.st_ino,
                    "mtime_ns": before.st_mtime_ns,
                    "ctime_ns": before.st_ctime_ns,
                }
            )
    for directory, device, inode in directories:
        current = directory.lstat()
        if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != (device, inode):
            raise ExecutionError("output directory changed during inventory")
    for entry in entries:
        current = (output / entry["path"]).lstat()
        if (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        ) != (
            entry["device"],
            entry["inode"],
            entry["bytes"],
            entry["mtime_ns"],
            entry["ctime_ns"],
        ):
            raise ExecutionError("output file changed during inventory")
    if not entries:
        raise ExecutionError("fresh output contains no regular artifacts")
    return {
        "status": "observed",
        "scope": "fresh_directory_snapshot",
        "files": entries,
        "bytes": total,
        "semantic_validation": "not_performed",
    }


def run_card(card_path: Path, session: Path) -> dict:
    """Run one exact locked foreground command; return its separate evidence record.

    SIGINT/SIGTERM are forwarded to the owned child group. SIGKILL/observer death
    can leave only launch/process records: never infer an exit from missing state.
    Only a main-thread call is supported because signal handlers are process-wide.
    """
    if threading.current_thread() is not threading.main_thread():
        raise ExecutionError("run must execute in the main thread")
    session = Path(session).resolve()
    card_path = Path(card_path).resolve()
    if card_path.parent != session / "memory/cards":
        raise ExecutionError("run needs a card in this session's memory/cards directory")
    with _claim(card_path):
        card, info, checked = _live_plan(card_path, session)
        card_bytes, lock_bytes = card_path.read_bytes(), lock.lock_path(card_path).read_bytes()
        if schema.load_card(card_path) != card or lock.read_lock(card_path) != info:
            raise ExecutionError("card or lock changed during launch preparation")
        attempts = session / "memory/attempts" / card["card_id"]
        _reject_unresolved(attempts)
        attempts.mkdir(parents=True, exist_ok=True, mode=0o700)
        attempt = attempts / uuid4().hex
        attempt.mkdir(mode=0o700)
        print(f"execution record: {attempt}", flush=True)
        command = card["setup"]["command"]
        output = Path(card["setup"]["output_dir"])
        policy = schema.get(card, "setup.execution.output_evidence") or "unverified"
        launch = {
            "schema_version": ATTEMPT_SCHEMA,
            "attempt_id": attempt.name,
            "card_id": card["card_id"],
            "created_at": schema.now(),
            "card_sha256": _digest(card_bytes),
            "lock_sha256": _digest(lock_bytes),
            "plan_sha256": info["plan_sha256"],
            "script": info["script"],
            "data": info["data"],
            "argv": command["argv"],
            "cwd": command["cwd"],
            "output_dir": str(output),
            "output_evidence": policy,
            "preflight": checked,
            "inventory_limits": {"objects": MAX_OUTPUT_FILES, "bytes": MAX_OUTPUT_BYTES},
            "environment_override_names": sorted((command.get("env") or {}).keys()),
            "environment_override_sha256": _digest(_encoded(command.get("env") or {})),
            "injected_environment": {RESERVED_ENV[0]: attempt.name, RESERVED_ENV[1]: str(attempt)},
        }
        _write_once(attempt / "launch.json", launch)
        record = {
            "schema_version": ATTEMPT_SCHEMA,
            "attempt_id": attempt.name,
            "attempt_dir": str(attempt),
            "status": "observer_failed",
            "observed_returncode": None,
            "child_pid": None,
            "spawn_attempted": False,
            "descendant_completion": "not_independently_verified",
            "scientific_validation": "not_performed",
            "artifacts": {"status": "unverified", "output_dir": str(output)},
        }
        child = None
        original_error = None
        interrupted = None
        previous = {}
        reserved_identity = None
        snapshot_started = False

        def capture(signum, _frame):
            nonlocal interrupted
            interrupted = interrupted or signum

        def check_interrupt():
            if interrupted is not None:
                raise ExecutionInterrupted(interrupted)

        started = time.monotonic()
        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.signal(signum, capture)
            if policy == "fresh-directory":
                if not output.is_absolute():
                    raise ExecutionError("fresh output must be an absolute path")
                output.mkdir(mode=0o700)  # exclusive, never reuse/delete an old output
                reserved = output.lstat()
                reserved_identity = (reserved.st_dev, reserved.st_ino)
                record["artifacts"]["fresh_namespace_reserved_at"] = schema.now()
            check_interrupt()
            # Do not overwrite the scientist's log paths; these are per-attempt streams.
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            with (
                os.fdopen(os.open(attempt / "stdout.txt", flags, 0o600), "wb") as out,
                os.fdopen(os.open(attempt / "stderr.txt", flags, 0o600), "wb") as err,
            ):
                record["spawn_attempted"] = True
                child = subprocess.Popen(
                    command["argv"],
                    cwd=command["cwd"],
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                    start_new_session=True,
                    env={
                        **os.environ,
                        **(command.get("env") or {}),
                        **launch["injected_environment"],
                    },
                )
                record["child_pid"] = child.pid
                _write_once(
                    attempt / "process.json",
                    {
                        "schema_version": ATTEMPT_SCHEMA,
                        "attempt_id": attempt.name,
                        "child_pid": child.pid,
                        "process_group": child.pid,
                        "observed_at": schema.now(),
                        "identity": _process_identity(child.pid),
                    },
                )
                while True:
                    check_interrupt()
                    try:
                        record["observed_returncode"] = child.wait(timeout=0.5)
                        break
                    except subprocess.TimeoutExpired:
                        continue
            record["child_exit_observed_at"] = schema.now()
            record["status"] = "command_exited"
            check_interrupt()
            after_card = schema.load_card(card_path)
            integrity = lock.verify_lock(card_path, after_card)
            record["integrity_after"] = [item.message for item in integrity.problems]
            record["card_after_sha256"] = _digest(card_path.read_bytes())
            if any(
                after_card.get(key) != card.get(key)
                for key in ("card_id", "created_at", "schema_version")
            ):
                record["integrity_after"].append("card identity changed during command")
            if lock.lock_path(card_path).read_bytes() != lock_bytes:
                record["integrity_after"].append("lock bytes changed during command")
            if policy == "fresh-directory":
                snapshot_started = True
                record["artifacts"].update(_snapshot(output, reserved_identity, check_interrupt))
            record["wrapper_returncode"] = record["observed_returncode"] or (
                1 if record["integrity_after"] else 0
            )
            return record
        except BaseException as exc:
            original_error = exc
            if child is not None:
                try:
                    record["observed_returncode"] = _stop_child(
                        child,
                        exc.signum if isinstance(exc, ExecutionInterrupted) else signal.SIGTERM,
                    )
                    record.setdefault("child_exit_observed_at", schema.now())
                except BaseException as cleanup_error:  # noqa: BLE001 - preserve the original interruption/error
                    record["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
                    if hasattr(original_error, "add_note"):
                        original_error.add_note(f"child cleanup also failed: {cleanup_error}")
            record["status"] = (
                "interrupted" if isinstance(exc, KeyboardInterrupt) else "observer_failed"
            )
            record["error"] = f"{type(exc).__name__}: {exc}"
            if policy == "fresh-directory":
                record["artifacts"]["snapshot_status"] = (
                    "incomplete" if snapshot_started else "not_attempted"
                )
            record["wrapper_returncode"] = (
                128 + getattr(exc, "signum", 2) if isinstance(exc, KeyboardInterrupt) else 2
            )
            exc.execution_record = record
            raise
        finally:
            if interrupted is not None:
                record["interrupt_signal"] = interrupted
                record["status"] = "interrupted"
                record["wrapper_returncode"] = 128 + interrupted
            record["finished_at"] = schema.now()
            record["observer_elapsed_seconds"] = time.monotonic() - started
            try:
                _write_once(attempt / "finish.json", record)
            except BaseException as audit_error:
                if original_error is None:
                    raise
                if hasattr(original_error, "add_note"):
                    original_error.add_note(f"execution record write also failed: {audit_error}")
            finally:
                for signum, handler in previous.items():
                    signal.signal(signum, handler)
