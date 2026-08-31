"""WMA memory: sidecar-owned, outside the session directory, grows across sessions.

Layout under ``AWM_WM_MEMORY`` (default ``<data>/wm-memory``)::

    raw/<session>/<card_id>/          copies of card, contract, observations, pings, replies, result
    structured/cards.jsonl            one row per closed card
    structured/observations.jsonl     one row per observation
    structured/interactions.jsonl     ping -> reply pairs
    structured/outcomes.jsonl         official outcomes, imported later (train side only)
    notes/<session>-<card_id>.md      grounded lessons (llm arm)

Every row carries ``provenance: {session, arm, split_side}``. A memory opened
``readonly`` (held-out sessions) answers queries and discards writes.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from .schema import WMError, dump_json, inside, now, read_jsonl, sha256_file

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-_.]{2,}")


def tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(str(text).lower()))


class Memory:
    def __init__(self, root: Path, *, session: str, arm: str, split_side: str = "train",
                 readonly: bool = False, visible_sides: tuple[str, ...] = ("train",)):
        self.root = Path(root)
        self.session = session
        self.arm = arm
        self.split_side = split_side
        self.readonly = readonly
        self.visible_sides = tuple(visible_sides)
        self.structured = self.root / "structured"
        if not readonly:
            (self.root / "raw").mkdir(parents=True, exist_ok=True)
            self.structured.mkdir(parents=True, exist_ok=True)
            (self.root / "notes").mkdir(parents=True, exist_ok=True)

    def card_corpus_roots(self, *, require: bool = False) -> list[Path]:
        """Return the complete reconstructed-card roots visible to this session.

        The LLM arm receives these directories directly.  Unlike ``precedents``,
        this is not a ranked or truncated view: every YAML card for every visible
        side is present below the returned roots.  ``require`` makes a missing or
        empty side a hard error so an experiment cannot silently become a
        no-memory control.
        """
        roots: list[Path] = []
        for side in self.visible_sides:
            if side not in ("train", "test"):
                raise WMError(f"invalid memory side {side!r}; choose train or test")
            root = self.root / "corpus" / side
            has_cards = root.is_dir() and next(root.glob("r-*/exp-*.yaml"), None) is not None
            if require and not root.is_dir():
                raise WMError(
                    f"llm memory side {side!r} has no full experiment-card corpus under {root}; "
                    "re-seed it with `awm wm memory seed <results-dir> --side " + side + "`"
                )
            if require:
                # Re-attest on every autonomous call.  Production mounts this
                # corpus read-only as well, but audit validity must not depend
                # on a stale in-process cache.
                _validate_card_corpus_root(root, side)
                roots.append(root.resolve())
            elif has_cards:
                roots.append(root.resolve())
        if require and not roots:
            raise WMError(f"no full experiment-card corpus is visible under {self.root}")
        return roots

    # ---- provenance

    def _prov(self) -> dict[str, str]:
        return {"session": self.session, "arm": self.arm, "split_side": self.split_side, "at": now()}

    def _append(self, table: str, row: dict[str, Any]) -> None:
        if self.readonly:
            return
        row = {**row, "provenance": self._prov()}
        with (self.structured / f"{table}.jsonl").open("a") as fh:
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    def _rows(self, table: str) -> list[dict[str, Any]]:
        path = self.structured / f"{table}.jsonl"
        return read_jsonl(path) if path.is_file() else []

    # ---- writes

    def record_observation(self, card: dict[str, Any], observation: dict[str, Any]) -> None:
        self._append("observations", {
            "card_id": card["card_id"],
            "base_model": _base_model(card),
            "method_family": card["setup"]["method"].get("family"),
            "obs_id": observation["obs_id"],
            "step": observation["checkpoint"].get("step"),
            "fraction": observation.get("fraction"),
            "evaluators": {k: {kk: v.get(kk) for kk in ("value", "n", "stderr", "delta_vs_parent", "delta_vs_prev")}
                           for k, v in observation["evaluators"].items()},
            "cause": observation.get("cause"),
        })

    def record_interaction(self, card_id: str, ping: dict[str, Any], reply: dict[str, Any] | None) -> None:
        self._append("interactions", {
            "card_id": card_id, "ping_id": ping["ping_id"], "kind": ping["kind"],
            "raised_by": ping.get("raised_by"), "prediction": ping.get("prediction"),
            "options": [o["id"] for o in ping.get("options", [])],
            "choice": reply["choice"] if reply else None, "by": reply.get("by") if reply else None,
        })

    def record_card(self, card_dir: Path, card: dict[str, Any], contract: dict[str, Any] | None,
                    result: dict[str, Any] | None, state: dict[str, Any]) -> None:
        if self.readonly:
            return
        raw = self.root / "raw" / self.session / card["card_id"]
        raw.mkdir(parents=True, exist_ok=True)
        for name in ("card.yaml", "contract.yaml", "seal.json", "state.json", "manifest.json"):
            src = card_dir / name
            if src.is_file():
                shutil.copy2(src, raw / name)
        for sub in ("observations", "pings", "replies"):
            if (card_dir / sub).is_dir():
                shutil.copytree(card_dir / sub, raw / sub, dirs_exist_ok=True)
        best = None
        for obs in state.get("observations", []):
            sel = (contract or {}).get("selection", {}).get("evaluator")
            v = obs.get("evaluators", {}).get(sel, {}).get("value") if sel else None
            if v is not None and (best is None or v > best):
                best = v
        self._append("cards", {
            "card_id": card["card_id"],
            "base_model": _base_model(card),
            "parent_origin": card["setup"]["parent_checkpoint"].get("origin"),
            "method_family": card["setup"]["method"].get("family"),
            "data_sources": [d.get("source") for d in card["setup"].get("data", [])],
            "problem": card["problem"].get("statement"),
            "claim": card["hypothesis"].get("claim"),
            "hyperparams": card["setup"]["method"].get("hyperparams"),
            "n_observations": len(state.get("observations", [])),
            "best_selection_value": best,
            "parent_value": (state.get("parent") or {}).get(
                (contract or {}).get("selection", {}).get("evaluator", ""), {}).get("value"),
            "final_status": state.get("status"),
            "execution": (result or {}).get("result", {}).get("execution"),
            "verdict": (result or {}).get("conclusion", {}).get("verdict"),
            "decision": (result or {}).get("conclusion", {}).get("decision"),
            "sealed_obs": (state.get("seal") or {}).get("obs_id"),
            "raw_dir": str(raw),
        })

    def record_outcome(self, card_id: str, session: str, official: dict[str, Any]) -> None:
        self._append("outcomes", {"card_id": card_id, "for_session": session, **official})

    def note(self, card_id: str, text: str) -> Path | None:
        if self.readonly:
            return None
        path = self.root / "notes" / f"{self.session}-{card_id}.md"
        path.write_text(text)
        return path

    # ---- reads

    def precedents(self, card: dict[str, Any], k: int = 5) -> list[dict[str, Any]]:
        """Nearest closed cards by token overlap on base model, method, data, problem, claim."""
        query = tokens(" ".join([
            _base_model(card) or "", card["setup"]["method"].get("family", ""),
            " ".join(str(d.get("source", "")) for d in card["setup"].get("data", [])),
            str(card["problem"].get("statement", "")), str(card["hypothesis"].get("claim", "")),
        ]))
        scored = []
        for row in self._rows("cards"):
            if row.get("provenance", {}).get("split_side") not in self.visible_sides:
                continue
            doc = tokens(" ".join([
                str(row.get("base_model", "")), str(row.get("method_family", "")),
                " ".join(map(str, row.get("data_sources") or [])),
                str(row.get("problem", "")), str(row.get("claim", "")),
            ]))
            if not doc:
                continue
            overlap = len(query & doc) / max(len(query | doc), 1)
            same_model = 0.25 if row.get("base_model") == _base_model(card) else 0.0
            same_family = 0.15 if row.get("method_family") == card["setup"]["method"].get("family") else 0.0
            scored.append((overlap + same_model + same_family, row))
        scored.sort(key=lambda t: -t[0])
        out = []
        for score, row in scored[:k]:
            delta = None
            if row.get("best_selection_value") is not None and row.get("parent_value") is not None:
                delta = round(row["best_selection_value"] - row["parent_value"], 4)
            out.append({"similarity": round(score, 3), "card_id": row["card_id"],
                        "session": row["provenance"]["session"], "base_model": row.get("base_model"),
                        "method_family": row.get("method_family"), "data_sources": row.get("data_sources"),
                        "delta_best_vs_parent": delta, "decision": row.get("decision"),
                        "verdict": row.get("verdict"), "raw_dir": row.get("raw_dir")})
        return out

    def curves(self, card_ids: list[tuple[str, str]]) -> dict[str, list[dict[str, Any]]]:
        """Observation rows for (session, card_id) pairs, in step order."""
        want = set(card_ids)
        out: dict[str, list[dict[str, Any]]] = {}
        for row in self._rows("observations"):
            key = (row["provenance"]["session"], row["card_id"])
            if key in want:
                out.setdefault(f"{key[0]}/{key[1]}", []).append(row)
        for rows in out.values():
            rows.sort(key=lambda r: (r.get("step") or 0))
        return out

    def stats(self) -> dict[str, int]:
        return {t: len(self._rows(t)) for t in ("cards", "observations", "interactions", "outcomes")}

    # ---- seeding from reconstructed cards

    def seed_from_exp_cards(self, results_dir: Path, *, side: str = "train") -> int:
        """Materialise full cards and structured rows from an experiment-card split.

        The exact YAML files are copied to ``corpus/<side>`` for the autonomous
        LLM arm.  The structured rows remain available to the deterministic
        retrieval arm.  Re-seeding a side replaces that side atomically instead
        of appending duplicate rows.
        """
        import yaml

        if self.readonly:
            raise WMError("cannot seed a read-only memory")
        if side not in ("train", "test"):
            raise WMError(f"invalid seed side {side!r}; choose train or test")
        source_root = Path(results_dir) / side
        paths = sorted(source_root.glob("r-*/exp-*.yaml"))
        if not paths:
            raise WMError(f"no experiment cards found under {source_root}")

        corpus_parent = self.root / "corpus"
        corpus_parent.mkdir(parents=True, exist_ok=True)
        stage = corpus_parent / f".{side}.tmp-{os.getpid()}"
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)

        seeded: list[dict[str, Any]] = []
        manifest: list[dict[str, Any]] = []
        for path in paths:
            try:
                card = yaml.safe_load(path.read_text())
            except yaml.YAMLError as exc:
                shutil.rmtree(stage, ignore_errors=True)
                raise WMError(f"invalid experiment card {path}: {exc}") from exc
            if not isinstance(card, dict) or "setup" not in card:
                shutil.rmtree(stage, ignore_errors=True)
                raise WMError(f"invalid experiment card {path}: expected a mapping with setup")

            rel = path.relative_to(source_root)
            dest = stage / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            setup = card.get("setup") or {}
            res = card.get("result") or {}
            con = card.get("conclusion") or {}
            meas = [m.get("value") for m in (res.get("measurements") or []) if isinstance(m.get("value"), (int, float))]
            comp = ((card.get("evaluation") or {}).get("comparator") or {}).get("value")
            row = {
                "card_id": f"{path.parent.name}/{path.stem}",
                "base_model": (setup.get("parent_checkpoint") or {}).get("path"),
                "parent_origin": (setup.get("parent_checkpoint") or {}).get("origin"),
                "method_family": (setup.get("method") or {}).get("family"),
                "data_sources": [d.get("source") for d in (setup.get("data") or []) if isinstance(d, dict)],
                "problem": (card.get("problem") or {}).get("statement"),
                "claim": (card.get("hypothesis") or {}).get("claim"),
                "hyperparams": (setup.get("method") or {}).get("hyperparams"),
                "n_observations": len(meas),
                "best_selection_value": max(meas) if meas else None,
                "parent_value": comp if isinstance(comp, (int, float)) else None,
                "final_status": "reconstructed",
                "execution": res.get("execution"),
                "verdict": con.get("verdict"),
                "decision": con.get("decision"),
                "reconstructed": True,
                "corpus_path": str(Path("corpus") / side / rel),
                "source_path": str(Path(side) / rel),
                "content_sha256": sha256_file(dest),
                "raw_dir": str(self.root / "corpus" / side / rel.parent),
            }
            row["provenance"] = {
                "session": self.session,
                "arm": self.arm,
                "split_side": side,
                "at": now(),
            }
            seeded.append(row)
            manifest.append({
                "card_id": row["card_id"],
                "path": str(rel),
                "sha256": row["content_sha256"],
                "run_ref": path.parent.name,
                "side": side,
            })

        _write_jsonl(stage / "manifest.jsonl", manifest)
        coverage_path = Path(results_dir) / "coverage.json"
        source_coverage: dict[str, Any] = {}
        if coverage_path.is_file():
            try:
                source_coverage = json.loads(coverage_path.read_text())
            except json.JSONDecodeError as exc:
                shutil.rmtree(stage, ignore_errors=True)
                raise WMError(f"invalid corpus coverage metadata {coverage_path}: {exc}") from exc
        card_runs = sorted({row["run_ref"] for row in manifest})
        missing = (
            (source_coverage.get("runs_without_cards") or {}).get("by_side", {}).get(side)
            or []
        )
        missing = sorted(str(run_ref) for run_ref in missing)
        expected_runs = (source_coverage.get("expected_runs_by_side") or {}).get(side)
        if expected_runs is not None and int(expected_runs) != len(card_runs) + len(missing):
            shutil.rmtree(stage, ignore_errors=True)
            raise WMError(
                f"coverage for {side} says {expected_runs} expected runs, but the corpus has "
                f"{len(card_runs)} card-bearing plus {len(missing)} recorded missing runs"
            )
        corpus_manifest = {
            "schema_version": "awm-exp-card-corpus-v1",
            "side": side,
            "card_count": len(seeded),
            "card_bearing_run_count": len(card_runs),
            "card_bearing_run_refs": card_runs,
            "expected_run_count": int(expected_runs) if expected_runs is not None else None,
            "expected_run_refs": sorted(set(card_runs) | set(missing)) if expected_runs is not None else None,
            "missing_run_count": len(missing) if expected_runs is not None else None,
            "missing_run_refs": missing if expected_runs is not None else None,
            "missing_cause": (
                (source_coverage.get("runs_without_cards") or {}).get("cause")
                if expected_runs is not None
                else "coverage metadata unavailable"
            ),
            "coverage_evidence": (
                (source_coverage.get("runs_without_cards") or {}).get("evidence")
                if expected_runs is not None
                else None
            ),
            "source_coverage_sha256": sha256_file(coverage_path) if coverage_path.is_file() else None,
            "cards_manifest": "manifest.jsonl",
        }
        dump_json(stage / "manifest.json", corpus_manifest)
        final = corpus_parent / side
        backup = corpus_parent / f".{side}.old-{os.getpid()}"
        try:
            if final.exists():
                os.replace(final, backup)
            os.replace(stage, final)
        except Exception:
            if backup.exists() and not final.exists():
                os.replace(backup, final)
            shutil.rmtree(stage, ignore_errors=True)
            raise
        else:
            shutil.rmtree(backup, ignore_errors=True)

        existing = [
            row for row in self._rows("cards")
            if not (
                row.get("reconstructed") is True
                and row.get("provenance", {}).get("split_side") == side
            )
        ]
        _write_jsonl(self.structured / "cards.jsonl", existing + seeded)
        return len(seeded)


def _base_model(card: dict[str, Any]) -> str | None:
    setup = card.get("setup") or {}
    return setup.get("base_model") or (setup.get("parent_checkpoint") or {}).get("origin_model") \
        or (setup.get("parent_checkpoint") or {}).get("path")


def dump_debug(path: Path, value: Any) -> None:
    dump_json(path, value)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically write deterministic JSONL without exposing a partial corpus index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def _validate_card_corpus_root(root: Path, side: str) -> None:
    """Verify the seeded manifest, inventory, side coverage and every card hash."""
    if root.is_symlink() or not root.is_dir():
        raise WMError(f"full card corpus root is missing or symlinked: {root}")
    manifest_path = root / "manifest.json"
    cards_path = root / "manifest.jsonl"
    if (
        manifest_path.is_symlink()
        or cards_path.is_symlink()
        or not manifest_path.is_file()
        or not cards_path.is_file()
    ):
        raise WMError(f"full card corpus {root} is missing manifest.json or manifest.jsonl")
    try:
        manifest = json.loads(manifest_path.read_text())
        rows = read_jsonl(cards_path)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise WMError(f"invalid card-corpus manifest under {root}: {exc}") from exc
    if manifest.get("schema_version") != "awm-exp-card-corpus-v1":
        raise WMError(f"invalid card-corpus schema under {root}")
    if manifest.get("side") != side:
        raise WMError(f"card-corpus manifest side is {manifest.get('side')!r}, expected {side!r}")
    actual = {path.relative_to(root).as_posix(): path for path in root.glob("r-*/exp-*.yaml")}
    expected_dirs = {root.resolve()}
    expected_dirs.update(path.parent.resolve() for path in actual.values())
    for path in root.rglob("*"):
        if path.is_symlink():
            raise WMError(f"card corpus contains a symlink: {path}")
        if path.is_dir() and path.resolve() not in expected_dirs:
            raise WMError(f"card corpus contains an unexpected directory: {path}")
        if path.is_file():
            rel = path.relative_to(root)
            allowed = (
                (len(rel.parts) == 1 and rel.name in ("manifest.json", "manifest.jsonl"))
                or (
                    len(rel.parts) == 2
                    and rel.parts[0].startswith("r-")
                    and rel.name.startswith("exp-")
                    and rel.suffix == ".yaml"
                )
            )
            if not allowed:
                raise WMError(f"card corpus contains an unexpected file: {path}")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        rel = row.get("path")
        digest = row.get("sha256")
        if not isinstance(rel, str) or not isinstance(digest, str):
            raise WMError(f"invalid card entry in {cards_path}")
        candidate = (root / rel).resolve()
        if not inside(candidate, root) or candidate.suffix != ".yaml":
            raise WMError(f"unsafe card path {rel!r} in {cards_path}")
        if rel in indexed:
            raise WMError(f"duplicate card path {rel!r} in {cards_path}")
        indexed[rel] = row
    if set(indexed) != set(actual):
        missing = sorted(set(indexed) - set(actual))
        extra = sorted(set(actual) - set(indexed))
        raise WMError(f"card corpus inventory mismatch under {root}: missing={missing}, extra={extra}")
    if manifest.get("card_count") != len(rows):
        raise WMError(f"card corpus {root} says {manifest.get('card_count')} cards, found {len(rows)}")
    for rel, path in actual.items():
        if path.is_symlink() or not path.is_file():
            raise WMError(f"card corpus entry is not a regular copied file: {path}")
        if sha256_file(path) != indexed[rel]["sha256"]:
            raise WMError(f"card corpus hash mismatch: {path}")

    card_runs = sorted({Path(rel).parent.name for rel in actual})
    if manifest.get("card_bearing_run_refs") != card_runs:
        raise WMError(f"card-bearing run refs do not match card inventory under {root}")
    if manifest.get("card_bearing_run_count") != len(card_runs):
        raise WMError(f"card-bearing run count does not match card inventory under {root}")
    expected = manifest.get("expected_run_count")
    expected_refs = manifest.get("expected_run_refs")
    missing_refs = manifest.get("missing_run_refs")
    if not isinstance(expected, int) or not isinstance(expected_refs, list) or not isinstance(missing_refs, list):
        raise WMError(f"card corpus {root} lacks expected/missing-run coverage; re-seed from published metadata")
    if len(expected_refs) != expected or len(set(expected_refs)) != expected:
        raise WMError(f"expected run refs/count are inconsistent under {root}")
    if set(expected_refs) != set(card_runs) | set(missing_refs):
        raise WMError(f"expected run refs are not card-bearing plus missing refs under {root}")
    if set(card_runs) & set(missing_refs):
        raise WMError(f"a run is both card-bearing and missing under {root}")
    if manifest.get("missing_run_count") != len(missing_refs):
        raise WMError(f"missing run refs/count are inconsistent under {root}")
    if not manifest.get("source_coverage_sha256"):
        raise WMError(f"card corpus {root} has no source coverage hash")
