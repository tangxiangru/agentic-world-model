"""LEGACY v3 packager: NOT APPROVED for the full-recipe/final-target-only goal.

This implementation preserves the superseded ancestor-scored evidence design.
Its output must not be supplied to WM training or selector agents under the new
rule forbidding intermediate/ancestor checkpoint results in full-recipe inputs.
No real corpus was built before that rule changed; only synthetic tests ran.

    python -m tools.outcome_prediction.wm_corpus --dataset-dir DATA --output-dir NEW

``package_corpus(dataset_dir, output_dir)`` returns the completed manifest. Its
``files`` mapping is directly usable by EvidenceStore with root OUTPUT/evidence.
Manifest/provenance stay OUTSIDE that root. Existing outputs are never reused.
An interrupted/failed output has no manifest and must not be served. Source
directories are opened without following symlinks; no whole-source-root globbing
or checkpoint/pickle loading occurs. Only explicitly indexed training runs and
their inventoried cards are considered, even in a mixed-revision source cache.

Historical official TRAIN grades are permitted evidence; private inventory and
test labels themselves are never copied. Historical final cards/snapshots are
explicitly NOT prospective launch code. Missing snapshot members are audited,
not inferred or silently replaced. Available snapshot members must match their
snapshot manifest; unlisted files are not copied. Snapshot manifests themselves
are observed archive metadata, not independently revision-authenticated bytes.

Redaction replaces only known credential prefixes, explicit credential values,
and private-key blocks. It is a conservative heuristic, not a universal secret
detector. Scientific numbers, scores, hashes, URLs, and prose are not filtered.
No matched credentials (or individual credential hashes) enter the audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from tools.outcome_prediction.wm_evidence import _logical_path
from tools.outcome_prediction.wm_model import digest, public_payload

REPLACEMENT = "[REDACTED_CREDENTIAL]"
PREFIXES = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    r"ghs_[0-9]+_[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+|"
    r"hf_[A-Za-z0-9]{20,}|sk-(?:ant-|proj-|svcacct-)?[A-Za-z0-9_-]{20,}|"
    r"gh[pours]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"(?:AKIA|ASIA)[A-Z0-9]{16}|xox[baprs]-[A-Za-z0-9-]{20,}"
    r")(?![A-Za-z0-9_])"
)
SECRET_NAMES = (
    r"(?:[A-Za-z][A-Za-z0-9_]*_)?API_(?:KEY|TOKEN)|HF_TOKEN|HUGGING_FACE_HUB_TOKEN|"
    r"HUGGINGFACE_HUB_TOKEN|GITHUB_TOKEN|GH_TOKEN|AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN|"
    r"apiKey|access_token|refresh_token|auth_token|client_secret|password|passwd"
)
# Quotes may be escaped inside a JSON-serialized trace. Replacements never eat
# the quote, its backslash, delimiters, or line endings.
ASSIGNMENTS = re.compile(
    rf"(?i)(?<![\w])(?:{SECRET_NAMES})(?:\\?[\"'])?\s*[:=]\s*"
    r"(?:(?P<quote>\\?[\"'])(?P<quoted>[^\"'\r\n\\]{1,512})(?P=quote)|"
    r"(?P<bare>[A-Za-z0-9_./+~=-]{8,512}))"
)
AUTHORIZATION = re.compile(
    r"(?i)\bAuthorization(?:\\?[\"'])?\s*[:=]\s*(?:\\?[\"'])?"
    r"(?:Bearer|Basic)\s+(?P<secret>[A-Za-z0-9._~+/-]{8,}={0,2})"
)
PRIVATE_KEY = re.compile(
    r"-----BEGIN (?P<kind>(?:(?:RSA|EC|OPENSSH|DSA) )?PRIVATE KEY)-----"
    r"[\s\S]*?-----END (?P=kind)-----"
)


class CorpusError(ValueError):
    """A corpus boundary, integrity, or schema violation."""


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def _placeholder(value: str) -> bool:
    return (
        value.startswith(
            (
                "$",
                "<",
                "[REDACTED",
                "os.environ",
                "os.getenv",
                "getenv(",
                "args.",
                "config.",
                "settings.",
            )
        )
        or value.lower()
        in {
            "none",
            "null",
            "false",
            "true",
            "redacted",
            "your_api_key",
            "your_token",
            "api_key",
            "hf_token",
            "token",
            "password",
            "passwd",
        }
        or not value.strip(".* ")
    )


def redact_credentials(text: str) -> tuple[str, dict[str, int]]:
    """Replace disjoint secret spans, preserving all unrelated text exactly."""
    spans = [(m.start(), m.end(), "private_key") for m in PRIVATE_KEY.finditer(text)]
    without_keys = PRIVATE_KEY.sub("", text)
    if re.search(r"-----BEGIN (?:(?:RSA|EC|OPENSSH|DSA) )?PRIVATE KEY-----", without_keys):
        raise CorpusError("Unterminated private-key block requires manual review.")
    spans.extend((m.start(), m.end(), "vendor_token") for m in PREFIXES.finditer(text))
    for match in ASSIGNMENTS.finditer(text):
        group = "quoted" if match.group("quoted") is not None else "bare"
        value = match.group(group)
        # Preserve explicit variable/function expressions rather than partially
        # replacing their identifiers. Ambiguous bare values in explicit secret
        # assignments are conservatively redacted, not treated as general prose.
        if _placeholder(value) or (group == "bare" and text[match.end() : match.end() + 1] == "("):
            continue
        spans.append((*match.span(group), "credential_assignment"))
    for match in AUTHORIZATION.finditer(text):
        if not _placeholder(match.group("secret")):
            spans.append((*match.span("secret"), "authorization"))
    # Outer spans win (e.g. assignment of a vendor token); overlapping rules
    # count once, and no fragment of a dotted/bearer credential is left behind.
    merged = []
    for start, end, kind in sorted(spans, key=lambda item: (item[0], -item[1])):
        if merged and start < merged[-1][1]:
            old_start, old_end, old_kind = merged[-1]
            merged[-1] = (old_start, max(old_end, end), old_kind)
        else:
            merged.append((start, end, kind))
    parts, previous, counts = [], 0, Counter()
    for start, end, kind in merged:
        parts.extend((text[previous:start], REPLACEMENT))
        previous = end
        counts[kind] += 1
    parts.append(text[previous:])
    return "".join(parts), dict(counts)


class SourceRoot:
    """Pinned directory descriptor with no-follow opening of every component."""

    def __init__(self, path: str | Path):
        original = Path(path).absolute()
        if original.is_symlink():
            raise CorpusError("Source root must not be a symlink.")
        self.path = original.resolve(strict=True)
        self.fd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)

    def __enter__(self):
        return self

    def __exit__(self, *_: object):
        os.close(self.fd)

    def _open(self, path: str, *, directory: bool = False) -> int:
        path = _logical_path(path)
        fd = os.dup(self.fd)
        try:
            parts = path.split("/")
            for index, part in enumerate(parts):
                flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                if index < len(parts) - 1 or directory:
                    flags |= os.O_DIRECTORY
                next_fd = os.open(part, flags, dir_fd=fd)
                os.close(fd)
                fd = next_fd
            return fd
        except BaseException:
            os.close(fd)
            raise

    def read(
        self, path: str, *, expected: str | None = None, optional: bool = False
    ) -> bytes | None:
        try:
            fd = self._open(path)
        except FileNotFoundError:
            if optional:
                return None
            raise CorpusError("Required source file is missing.") from None
        except OSError as exc:
            raise CorpusError("Unsafe or unavailable source path; symlinks are forbidden.") from exc
        with os.fdopen(fd, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise CorpusError("Source file must be regular.")
            raw = handle.read()
        if expected is not None and (
            not re.fullmatch(r"[0-9a-f]{64}", expected) or sha256(raw) != expected
        ):
            raise CorpusError("Source SHA256 mismatch.")
        return raw

    def names(self, path: str) -> list[str]:
        try:
            fd = self._open(path, directory=True)
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise CorpusError("Unsafe historical directory; symlinks are forbidden.") from exc
        try:
            return sorted(os.listdir(fd))
        finally:
            os.close(fd)


GUIDE = """# LEGACY shared historical training evidence — NOT APPROVED

This corpus follows the superseded v3 ancestor-scored protocol. It is NOT safe
for the new full-base-to-final-recipe setting that allows only final-target
supervision and forbids intermediate/ancestor outcomes in recipe inputs.

Both selector arms receive this exact corpus. Only the treatment arm also gets
the fixed learned-WM query. All included runs belong to the frozen TRAIN split.

training_examples.jsonl gives the exact fitted training examples, their public
inputs, and known official training grades. All training-run raw trajectories,
first plans, recovered pre-proposal code, historical card versions, final cards,
and available manifest-listed final snapshots are also browsable.

Files named historical_final_card or historical_final_snapshot are retrospective
archive artifacts, NOT verified prospective launch code. Recovered pre-proposal
code is separately identified; missing code/snapshot files remain unknown.
Historical grades/results are permitted TRAIN evidence, not candidate outcomes.

Raw text is preserved except narrowly recognized credentials. Numeric scientific
settings, evaluation scores, and hashes are not redacted. No test input file,
test-label file, private inventory, or WM pickle is included. This is a historical
corpus, not a certification of prospective candidate-packet fidelity.
"""


def package_corpus(dataset_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Create a new train-only corpus. Never build into an existing/source path."""
    output = Path(output_dir).absolute()
    if os.path.lexists(output):
        raise FileExistsError("Corpus output already exists; choose a new directory.")
    with ExitStack() as stack:
        dataset = stack.enter_context(SourceRoot(dataset_dir))
        inputs = {}

        def metadata(name):
            raw = dataset.read(name)
            inputs[name] = sha256(raw)
            return json.loads(raw)

        protocol = metadata("protocol.json")
        split = metadata("split.json")
        audits = metadata("source_audits.json")
        index = metadata("train/raw_trajectory_index.json")
        model_manifest = metadata("model/training_manifest.json")
        raw_inventory = dataset.read("private/inventory.jsonl")
        inputs["private/inventory.jsonl"] = sha256(raw_inventory)
        inventory = [json.loads(line) for line in raw_inventory.splitlines() if line.strip()]
        if digest(split) != protocol["split_sha256"]:
            raise CorpusError("Frozen split digest mismatch.")
        train_cells, test_cells = set(split["train_cell_ids"]), set(split["test_cell_ids"])
        if not train_cells or train_cells & test_cells:
            raise CorpusError("Training/test run boundary is empty or overlapping.")
        if train_cells | test_cells != {r["cell_id"] for r in inventory}:
            raise CorpusError("Inventory does not match the complete frozen run partition.")
        for cell in train_cells | test_cells:
            if not re.fullmatch(r"(?:aime-)?r0-\d+", cell):
                raise CorpusError("Only pinned r0 training runs are supported.")
        if len(index) != len(train_cells) or {e["cell_id"] for e in index} != train_cells:
            raise CorpusError("Raw trajectory index must cover each TRAIN run exactly once.")
        sources = {}
        for source in protocol["sources"]:
            key = (source["benchmark"], source["source_revision"])
            if key in sources:
                raise CorpusError("Ambiguous source revision.")
            source_root = stack.enter_context(SourceRoot(source["raw_root"]))
            matching = [
                a
                for a in audits
                if a["source_revision"] == key[1]
                and a["task"]["benchmark"] == key[0]
                and a["manifest_name"] == source["manifest_name"]
            ]
            if len(matching) != 1:
                raise CorpusError("Missing unique pinned source-manifest audit.")
            raw_manifest = source_root.read(
                source["manifest_name"], expected=matching[0]["manifest_sha256"]
            )
            members = {m["cell_id"] for m in json.loads(raw_manifest)}
            sources[key] = (source_root, members)
        for name, expected in protocol["source_code_sha256"].items():
            dataset.read("provenance/source/" + _logical_path(name), expected=expected)
        chosen_sources = {}
        for entry in index:
            key = (entry["benchmark"], entry["source_revision"])
            if key not in sources or entry["cell_id"] not in sources[key][1]:
                raise CorpusError("Training run is absent from its pinned source manifest.")
            source_root = sources[key][0]
            relative = f"cells/{entry['cell_id']}/solve_out_sanitized.txt"
            if Path(entry["path"]).absolute() != source_root.path / relative:
                raise CorpusError("Raw trajectory path is not the exact TRAIN source path.")
            raw = source_root.read(relative, expected=entry["sha256"])
            if len(raw) != entry["bytes"]:
                raise CorpusError("Raw trajectory byte count mismatch.")
            chosen_sources[entry["cell_id"]] = (source_root, key, entry)
        rows = [r for r in inventory if r["cell_id"] in train_cells]
        by_id = {r["example_id"]: r for r in rows}
        if len(by_id) != len(rows):
            raise CorpusError("Duplicate training example identity.")
        if set(by_id) != set(split["train_example_ids"]):
            raise CorpusError("Training inventory identities disagree with frozen split.")
        if (
            set(model_manifest["declared_train_cell_ids"]) != train_cells
            or set(model_manifest["forbidden_cell_ids"]) != test_cells
        ):
            raise CorpusError("Model manifest disagrees with the TRAIN/test run boundary.")
        fitted_ids = {e["example_id"] for e in model_manifest["training_examples"]}
        expected_fit = {
            r["example_id"]
            for r in rows
            if r["audit"]["eligible"] and r["label"]["accuracy"] is not None
        }
        if fitted_ids != expected_fit or len(fitted_ids) != len(
            model_manifest["training_examples"]
        ):
            raise CorpusError("Fitted example identities disagree with eligible TRAIN rows.")
        if digest(model_manifest["training_examples"]) != protocol["training_payload_sha256"]:
            raise CorpusError("Training manifest digest mismatch.")
        for entry in model_manifest["training_examples"]:
            row = by_id[entry["example_id"]]
            if (
                entry["payload_sha256"] != digest(public_payload(row))
                or entry["accuracy"] != row["label"]["accuracy"]
            ):
                raise CorpusError("Training payload/grade differs from the frozen fit.")
        output_parent = output.parent.resolve(strict=True)
        output = output_parent / output.name
        if any(output.is_relative_to(s.path) for s in [dataset, *(v[0] for v in sources.values())]):
            raise CorpusError("Corpus output must be outside every source/dataset root.")
        output.mkdir(mode=0o700)  # Exclusive even if a concurrent caller created it.
        evidence = output / "evidence"
        evidence.mkdir(mode=0o700)
        files, artifacts, missing, redactions = {}, [], [], Counter()

        def emit(
            path, raw, role, source=None, verification="generated from frozen training inventory"
        ):
            path = _logical_path(path)
            if path in files:
                raise CorpusError("Duplicate output evidence path.")
            if redact_credentials(path)[0] != path:
                raise CorpusError("Credential-like filename requires manual review.")
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise CorpusError("Non-UTF-8 historical artifact requires manual review.") from exc
            sanitized, counts = redact_credentials(text)
            encoded = sanitized.encode("utf-8")
            destination = evidence / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                handle.write(encoded)
            destination.chmod(0o444)
            files[path] = sha256(encoded)
            redactions.update(counts)
            artifacts.append(
                {
                    "path": path,
                    "role": role,
                    "source": source,
                    "source_sha256": sha256(raw),
                    "output_sha256": files[path],
                    "source_bytes": len(raw),
                    "output_bytes": len(encoded),
                    "verification": verification,
                    "credential_redactions": counts,
                }
            )

        emit("README.md", GUIDE.encode(), "corpus_guide")
        examples = [
            {
                "example_id": r["example_id"],
                "cell_id": r["cell_id"],
                "benchmark": r["benchmark"],
                "model_input": public_payload(r),
                "official_accuracy": r["label"]["accuracy"],
                "official_evaluation_n": r["label"].get("evaluation_n"),
            }
            for r in sorted(rows, key=lambda r: r["example_id"])
            if r["example_id"] in fitted_ids
        ]
        emit(
            "training_examples.jsonl",
            b"".join(
                (json.dumps(e, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()
                for e in examples
            ),
            "exact_fitted_training_examples_with_known_grades",
        )
        for cell in sorted(chosen_sources):
            source, _, entry = chosen_sources[cell]
            relative = f"cells/{cell}/solve_out_sanitized.txt"
            emit(
                f"runs/{cell}/raw_trajectory.txt",
                source.read(relative, expected=entry["sha256"]),
                "historical_training_raw_trajectory",
                relative,
                "frozen raw index SHA256",
            )
        for row in sorted(rows, key=lambda r: r["example_id"]):
            cell, card = row["cell_id"], row["card_id"]
            if not re.fullmatch(r"exp-\d+", card) or row["example_id"] != f"{cell}/{card}":
                raise CorpusError("Training card identity mismatch.")
            source, key, _ = chosen_sources[cell]
            provenance = row["provenance"]
            if row["benchmark"] != key[0] or provenance["source_revision"] != key[1]:
                raise CorpusError("Training row source revision/benchmark mismatch.")
            base, out = f"cells/{cell}/wm/cards/{card}", f"runs/{cell}/cards/{card}"
            first_plan = {
                "example_id": row["example_id"],
                "first_submitted_at": row.get("first_submitted_at"),
                "first_stage": row.get("first_stage"),
                "task": row["model_input"]["task"],
                "plan": row["model_input"]["plan"],
            }
            emit(out + "/first_plan.json", json_bytes(first_plan), "first_registered_plan")
            code_catalog = []
            for i, code in enumerate(row["model_input"].get("code", [])):
                item = {k: v for k, v in code.items() if k != "content"}
                item["evidence_path"] = None
                if code.get("content") is not None:
                    if code.get("status") != "reconstructed" or not isinstance(
                        code["content"], str
                    ):
                        raise CorpusError(
                            "Non-reconstructed code must not be presented as pre-proposal code."
                        )
                    path = out + f"/recovered_preproposal_code/{i:03d}.txt"
                    emit(path, code["content"].encode(), "reconstructed_before_first_registration")
                    item["evidence_path"] = path
                code_catalog.append(item)
            emit(
                out + "/recovered_code_index.json",
                json_bytes(code_catalog),
                "preproposal_code_availability",
            )
            names = source.names(base)
            required_names = []
            if provenance.get("card_sha256"):
                required_names.append("card.json")
            if provenance.get("first_record_sha256"):
                required_names.append(Path(provenance["first_record_path"]).name)
            if set(required_names) - set(names):
                raise CorpusError("Pinned historical card/first record is missing.")
            for name in names:
                if name != "card.json" and not re.fullmatch(r"record-\d+\.json", name):
                    continue
                relative = base + "/" + name
                expected = provenance.get("card_sha256") if name == "card.json" else None
                first_path = provenance.get("first_record_path")
                if first_path and name == Path(first_path).name:
                    expected = provenance["first_record_sha256"]
                raw = source.read(relative, expected=expected)
                destination = (
                    "historical_final_card.json"
                    if name == "card.json"
                    else "historical_records/" + name
                )
                emit(
                    out + "/" + destination,
                    raw,
                    "historical_final_card_not_launch"
                    if name == "card.json"
                    else "historical_versioned_card",
                    relative,
                    "frozen card SHA256"
                    if expected
                    else "observed archive bytes; raw trajectory separately pinned",
                )
            metric_sha = provenance.get("official_metric_sha256")
            if metric_sha:
                relative = f"cells/{cell}/wm_metrics/{card}.json"
                emit(
                    out + "/historical_official_metric.json",
                    source.read(relative, expected=metric_sha),
                    "historical_training_official_grade",
                    relative,
                    "frozen official-metric SHA256",
                )
            snapshot = base + "/snapshot"
            snapshot_raw = source.read(snapshot + "/MANIFEST.json", optional=True)
            if snapshot_raw is None:
                missing.append(
                    {"cell_id": cell, "card_id": card, "kind": "snapshot_manifest_missing"}
                )
                continue
            snapshot_manifest = json.loads(snapshot_raw)
            emit(
                out + "/historical_final_snapshot/MANIFEST.json",
                snapshot_raw,
                "historical_final_snapshot_manifest_not_launch",
                snapshot + "/MANIFEST.json",
                "observed archive manifest; member hashes checked",
            )
            seen = set()
            for member in snapshot_manifest["files"]:
                name = _logical_path(member["path"])
                if redact_credentials(name)[0] != name:
                    raise CorpusError("Credential-like snapshot filename requires manual review.")
                if name in seen or name == "MANIFEST.json":
                    raise CorpusError("Duplicate/reserved snapshot manifest member.")
                seen.add(name)
                raw = source.read(snapshot + "/" + name, expected=member["sha256"], optional=True)
                if raw is None:
                    missing.append(
                        {
                            "cell_id": cell,
                            "card_id": card,
                            "kind": "snapshot_member_missing",
                            "path": name,
                        }
                    )
                    continue
                if len(raw) != member["bytes"]:
                    raise CorpusError("Snapshot member byte count mismatch.")
                emit(
                    out + "/historical_final_snapshot/" + name,
                    raw,
                    "historical_final_snapshot_not_launch",
                    snapshot + "/" + name,
                    "snapshot manifest member SHA256",
                )
        manifest = {
            "schema": "wm-shared-training-corpus-v1",
            "evidence_root": "evidence",
            "protocol_status": "legacy_ancestor_scored_evidence_not_approved",
            "files": dict(sorted(files.items())),
            "training_run_ids": sorted(train_cells),
            "training_example_count": len(examples),
            "historical_card_count": len(rows),
            "split_sha256": protocol["split_sha256"],
            "same_evidence_for_both_arms": True,
        }
        audit = {
            "schema": "wm-corpus-provenance-v1",
            "dataset_dir": str(dataset.path),
            "input_file_sha256": inputs,
            "training_payload_sha256": protocol["training_payload_sha256"],
            "archived_source_code_sha256": protocol["source_code_sha256"],
            "corpus_builder_sha256": sha256(Path(__file__).read_bytes()),
            "credential_redactions": dict(redactions),
            "missing_historical_artifacts": missing,
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
            "limits": [
                "Credential patterns are heuristic, not comprehensive secret detection.",
                "Historical final snapshots are not verified launch-time code.",
                "Snapshot manifest bytes are observed archive metadata; only available members are copied.",
                "No test evidence, private labels, or model pickle is exposed.",
            ],
        }
        for name, value in [("provenance.json", audit), ("manifest.json", manifest)]:
            with (output / name).open("xb") as handle:
                handle.write(json_bytes(value))
            (output / name).chmod(0o444)
        return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = package_corpus(args.dataset_dir, args.output_dir)
    print(
        json.dumps(
            {"files": len(result["files"]), "training_runs": len(result["training_run_ids"])}
        )
    )


if __name__ == "__main__":
    main()
