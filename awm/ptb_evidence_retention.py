"""Preserve official evaluator bytes during harvest, separately from scoring.

Archives contain all supported files from every official attempt, including
failed attempts and the immutable Inspect snapshots referenced by metrics.
They are not a model bundle or an independent scientific validation verdict.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import stat
import uuid
from pathlib import Path

SCHEMA = "ptb-harvest-official-evidence-v1"
CHUNK = 1024 * 1024
SUPPORTED_SUFFIXES = frozenset({".json", ".jsonl", ".txt", ".log", ".eval", ".gz"})


class RetentionError(ValueError):
    pass


def _fingerprint(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _open_beneath(root: Path, relative: Path):
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise RetentionError("unsafe evidence path")
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for name in relative.parts[:-1]:
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise RetentionError("evidence is not a regular file")
    return os.fdopen(fd, "rb")


def _hash_stream(stream):
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: stream.read(CHUNK), b""):
        digest.update(block)
        size += len(block)
    return {"sha256": digest.hexdigest(), "bytes": size}


def _hash_file(root: Path, relative: Path):
    with _open_beneath(root, relative) as stream:
        before = os.fstat(stream.fileno())
        value = _hash_stream(stream)
        if _fingerprint(before) != _fingerprint(os.fstat(stream.fileno())):
            raise RetentionError("file changed while reading")
        if value["bytes"] != before.st_size:
            raise RetentionError("file length changed while reading")
        return value


def _read_checked(root: Path, relative: Path, expected: dict):
    """Parse the same bounded metadata bytes whose fingerprint was verified."""
    with _open_beneath(root, relative) as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read(16 * CHUNK + 1)
        if len(raw) > 16 * CHUNK:
            raise RetentionError("metadata exceeds retention reader limit")
        actual = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        if actual != expected or _fingerprint(before) != _fingerprint(os.fstat(stream.fileno())):
            raise RetentionError("retained metadata changed")
    return json.loads(raw)


def _directory_beneath(root: Path, relative: Path):
    if relative.is_absolute() or ".." in relative.parts:
        raise RetentionError("unsafe output path")
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for name in relative.parts:
            try:
                os.mkdir(name, mode=0o700, dir_fd=directory)
            except FileExistsError:
                pass
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
    except BaseException:
        os.close(directory)
        raise
    return directory


def _archive(root: Path, relative: Path, output: Path):
    archive = Path("raw") / (relative.as_posix() + ".gz")
    directory = _directory_beneath(output, archive.parent)
    temporary = ".retention-" + uuid.uuid4().hex + ".tmp"
    created = False
    try:
        with _open_beneath(root, relative) as source:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=directory)
            created = True
            before = os.fstat(source.fileno())
            digest = hashlib.sha256()
            size = 0
            with os.fdopen(fd, "wb") as target:
                with gzip.GzipFile(filename="", fileobj=target, mode="wb", mtime=0) as compressed:
                    for block in iter(lambda: source.read(CHUNK), b""):
                        digest.update(block)
                        size += len(block)
                        compressed.write(block)
                if _fingerprint(before) != _fingerprint(os.fstat(source.fileno())) or size != before.st_size:
                    raise RetentionError("evidence changed while archiving")
                target.flush()
                os.fsync(target.fileno())
            os.link(temporary, archive.name, src_dir_fd=directory, dst_dir_fd=directory,
                    follow_symlinks=False)
            os.fsync(directory)
    finally:
        if created:
            os.unlink(temporary, dir_fd=directory)
        os.close(directory)
    return {
        "source": relative.as_posix(),
        "sha256": digest.hexdigest(),
        "bytes": size,
        "archive": archive.as_posix(),
        "archive_fingerprint": _hash_file(output, archive),
    }


def _sources(root: Path):
    for base in sorted(root.iterdir()):
        if base.name != "official_eval" and not base.name.startswith(".official-inspect-"):
            continue
        if base.is_symlink() or not base.is_dir():
            yield base.relative_to(root), "official evidence root is not a regular directory"
            continue
        walk_errors = []
        for directory, dirs, files in os.walk(base, followlinks=False, onerror=walk_errors.append):
            directory = Path(directory)
            dirs.sort()
            for name in list(dirs):
                child = directory / name
                if child.is_symlink():
                    dirs.remove(name)
                    yield child.relative_to(root), "symlink evidence directory refused"
            for name in sorted(files):
                path = directory / name
                error = None
                if path.is_symlink():
                    error = "symlink evidence file refused"
                elif path.suffix not in SUPPORTED_SUFFIXES:
                    error = "unsupported evidence file; not copied as a log"
                yield path.relative_to(root), error
        for error in walk_errors:
            yield base.relative_to(root), f"cannot enumerate official evidence: {error}"


def _bound_raw_error(evidence, records):
    if not isinstance(evidence, dict):
        return "malformed official evidence binding"
    raw = next((r for r in records if r["source"] == evidence.get("raw_log")), None)
    if raw is None or (raw["sha256"], raw["bytes"]) != (
        evidence.get("raw_sha256"), evidence.get("raw_bytes")
    ):
        return "referenced raw evidence missing or changed"
    return None


def preserve_official_evidence(result_dir: Path, bundle: Path) -> dict:
    """Archive without the small-text cap; errors never imply complete retention."""
    root = result_dir.resolve(strict=True)
    output = bundle / "official-evidence"
    sources = list(_sources(root))
    metrics = {}
    if (bundle / "metrics.json").is_file():
        try:
            metrics = json.loads((bundle / "metrics.json").read_text())
        except (OSError, ValueError):
            pass
    evidence = metrics.get("official_evidence", {}) if isinstance(metrics, dict) else {}
    if not sources and not evidence:
        return {"state": "absent", "files": 0, "errors": []}
    # A repeated call must not alter an existing archive. harvest_job owns and
    # recreates its bundle; direct callers must supply a fresh destination.
    bundle_fd = os.open(bundle, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.mkdir("official-evidence", mode=0o700, dir_fd=bundle_fd)
    finally:
        os.close(bundle_fd)
    records, errors, bound = [], [], {}
    for name in ("metrics.json", "runtime_provenance.json"):
        if (bundle / name).exists():
            try:
                bound[name] = _hash_file(bundle, Path(name))
            except (OSError, RetentionError) as exc:
                errors.append({"path": name, "error": str(exc)})
    for relative, error in sources:
        if error:
            errors.append({"path": relative.as_posix(), "error": error})
            continue
        try:
            records.append(_archive(root, relative, output))
        except (OSError, RetentionError) as exc:
            errors.append({"path": relative.as_posix(), "error": str(exc)})
    if evidence and (error := _bound_raw_error(evidence, records)):
        errors.append({"path": "metrics.json", "error": error})
    # Terminal state is expected, but still detect a changed file inventory or
    # metadata while harvesting instead of declaring a partial view complete.
    if sources != list(_sources(root)):
        errors.append({"path": ".", "error": "official evidence inventory changed during harvest"})
    for record in records:
        try:
            if _hash_file(root, Path(record["source"])) != {
                "sha256": record["sha256"], "bytes": record["bytes"]
            }:
                raise RetentionError("source evidence changed after archiving")
        except (OSError, RetentionError) as exc:
            errors.append({"path": record["source"], "error": str(exc)})
    for name, expected in bound.items():
        try:
            if _hash_file(root, Path(name)) != expected:
                raise RetentionError("source metadata differs from retained metadata")
        except (OSError, RetentionError) as exc:
            errors.append({"path": name, "error": str(exc)})
    manifest = {
        "schema_version": SCHEMA,
        "source_result_dir": str(root),
        "state": "partial" if errors else "preserved",
        "bound_files": bound,
        "files": records,
        "errors": errors,
        "scope": "byte retention only; scientific validation uses the frozen task validator",
    }
    output_fd = _directory_beneath(bundle, Path("official-evidence"))
    try:
        fd = os.open("index.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=output_fd)
        with os.fdopen(fd, "w") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(output_fd)
    finally:
        os.close(output_fd)
    return {"state": manifest["state"], "files": len(records), "errors": errors,
            "index": "official-evidence/index.json",
            "index_fingerprint": _hash_file(output, Path("index.json"))}


def verify_official_evidence(bundle: Path, *, expected_index_fingerprint: dict) -> dict:
    """Verify against a caller-held harvest fingerprint, never a self-certified index.

    The caller obtains the fingerprint from independently retained harvest
    status (for example the committed operator bundle), not from the index
    currently being verified. This checks retention, not model/task validity.
    """
    output = bundle / "official-evidence"
    try:
        manifest = _read_checked(bundle, Path("official-evidence/index.json"), expected_index_fingerprint)
    except RetentionError as exc:
        raise RetentionError("official evidence index changed") from exc
    if manifest.get("schema_version") != SCHEMA or manifest.get("state") != "preserved":
        raise RetentionError("official evidence retention is not complete")
    if not isinstance(manifest.get("files"), list) or not manifest["files"] or manifest.get("errors") != []:
        raise RetentionError("malformed retained file inventory")
    if not isinstance(manifest.get("bound_files"), dict) or set(manifest["bound_files"]) - {
        "metrics.json", "runtime_provenance.json"
    }:
        raise RetentionError("malformed metadata bindings")
    seen_sources, seen_archives = set(), set()
    metadata = {
        name: _read_checked(bundle, Path(name), expected)
        for name, expected in manifest["bound_files"].items()
    }
    for record in manifest["files"]:
        source = record.get("source") if isinstance(record, dict) else None
        if not isinstance(source, str):
            raise RetentionError("malformed source name")
        parts = Path(source).parts
        if (len(parts) < 2 or Path(source).is_absolute() or ".." in parts
                or Path(source).as_posix() != source
                or (parts[0] != "official_eval" and not parts[0].startswith(".official-inspect-"))
                or record.get("archive") != "raw/" + source + ".gz"):
            raise RetentionError("unsafe or inconsistent archive mapping")
        if source in seen_sources or record["archive"] in seen_archives:
            raise RetentionError("duplicate archive mapping")
        seen_sources.add(source)
        seen_archives.add(record["archive"])
        relative = Path(record["archive"])
        with _open_beneath(output, relative) as stream:
            before = os.fstat(stream.fileno())
            if _hash_stream(stream) != record["archive_fingerprint"]:
                raise RetentionError("archived evidence changed")
            stream.seek(0)
            with gzip.GzipFile(fileobj=stream) as raw:
                if _hash_stream(raw) != {"sha256": record["sha256"], "bytes": record["bytes"]}:
                    raise RetentionError("uncompressed evidence changed")
            if _fingerprint(before) != _fingerprint(os.fstat(stream.fileno())):
                raise RetentionError("archive changed while verifying")
    if (bundle / "metrics.json").exists():
        if "metrics.json" not in manifest["bound_files"]:
            raise RetentionError("metrics exist without an independent binding")
        metrics = metadata["metrics.json"]
        if metrics.get("official_evidence") and (
            error := _bound_raw_error(metrics["official_evidence"], manifest["files"])
        ):
            raise RetentionError(error)
    return manifest
