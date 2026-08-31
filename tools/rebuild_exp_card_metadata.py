#!/usr/bin/env python3
"""Rebuild the published experiment-card corpus metadata.

Only cards under ``train/`` and ``test/`` are published cards.  Hidden probe
directories are deliberately outside the input set.  The committed run split
is the authority for expected run membership; a run reference is the opaque
``r-`` prefix plus the first eight hex digits of SHA-256 over its split path.

The per-run ``index.md`` files contain extractor notes and are intentionally
not generated here.  This script owns only the corpus-root ``coverage.json``
and ``index.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = REPO / "results" / "exp-cards" / "gsm8k-gemma-holdout-v1"
DEFAULT_SPLIT = REPO / "splits" / "posttrainbench" / "gsm8k-gemma-holdout-v1.yaml"
SIDES = ("train", "test")
RUN_REF_RE = re.compile(r"^r-[0-9a-f]{8}$")
CARD_FILE_RE = re.compile(r"^exp-(\d+)\.yaml$")

FIELD_PATHS = (
    "conclusion.decision",
    "conclusion.mechanism_verdict",
    "conclusion.next_step",
    "conclusion.summary",
    "conclusion.verdict",
    "evaluation.comparator",
    "evaluation.diagnostic",
    "evaluation.protocol",
    "hypothesis.claim",
    "hypothesis.expected_effect",
    "hypothesis.falsified_if",
    "hypothesis.mechanism",
    "problem.affected_share",
    "problem.evidence",
    "problem.failure_examples",
    "problem.statement",
    "result.diagnostic_result",
    "result.duration_h",
    "result.execution",
    "result.failure",
    "result.measurements",
    "result.output_checkpoint",
    "result.steps",
    "result.training_summary",
    "result.wall_h",
    "setup.budget",
    "setup.command",
    "setup.data",
    "setup.method",
    "setup.parent_checkpoint",
)

INDEX_COLUMNS = (
    "side",
    "run_ref",
    "card",
    "base model",
    "launch_i",
    "family",
    "parent",
    "data sources",
    "exec",
    "best own eval",
    "verdict",
    "decision",
    "hyp stated",
    "official",
)


class MetadataError(ValueError):
    """The published card tree and its split contract disagree."""


@dataclass(frozen=True)
class PublishedCard:
    side: str
    run_ref: str
    number: int
    path: Path
    value: dict[str, Any]


@dataclass(frozen=True)
class RenderedMetadata:
    coverage: str
    index: str


def run_ref(run_path: str) -> str:
    """Return the opaque reference used by the reconstructed-card corpus."""

    return "r-" + hashlib.sha256(run_path.encode("utf-8")).hexdigest()[:8]


def expected_run_refs(split_path: Path) -> dict[str, tuple[str, ...]]:
    split = _load_split(split_path)
    parts = split["splits"]

    out: dict[str, tuple[str, ...]] = {}
    for side in SIDES:
        paths = parts.get(side)
        if not isinstance(paths, list) or not all(isinstance(p, str) and p for p in paths):
            raise MetadataError(f"{split_path}: splits.{side} must be a list of run paths")
        refs = tuple(run_ref(path) for path in paths)
        if len(set(refs)) != len(refs):
            raise MetadataError(f"{split_path}: splits.{side} has colliding run references")
        out[side] = refs
    overlap = set(out["train"]) & set(out["test"])
    if overlap:
        raise MetadataError(f"{split_path}: run references occur on both sides: {sorted(overlap)}")
    return out


def _load_split(split_path: Path) -> dict[str, Any]:
    try:
        split = yaml.safe_load(split_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise MetadataError(f"cannot read split {split_path}: {exc}") from exc
    parts = split.get("splits") if isinstance(split, dict) else None
    if not isinstance(parts, dict) or set(parts) != set(SIDES):
        raise MetadataError(f"{split_path}: splits must contain exactly train and test")
    return split


def _split_base_models(split_path: Path) -> dict[str, str]:
    """Recover the trained HF id encoded in each PostTrainBench run name."""

    split = _load_split(split_path)
    benchmark = split.get("benchmark")
    if not isinstance(benchmark, str) or not benchmark:
        return {}
    prefix = benchmark + "_"
    out: dict[str, str] = {}
    for side in SIDES:
        for run_path in split["splits"][side]:
            name = run_path.rsplit("/", 1)[-1]
            if not name.startswith(prefix) or "_" not in name[len(prefix) :]:
                continue
            encoded = name[len(prefix) :].rsplit("_", 1)[0]
            if "_" not in encoded:
                continue
            namespace, repository = encoded.split("_", 1)
            out[run_ref(run_path)] = f"{namespace}/{repository}"
    return out


def load_published_cards(corpus: Path, expected: dict[str, tuple[str, ...]]) -> list[PublishedCard]:
    """Load strict ``SIDE/r-*/exp-NN.yaml`` inputs, excluding all probe trees."""

    cards: list[PublishedCard] = []
    seen: set[tuple[str, str, int]] = set()
    for side in SIDES:
        side_dir = corpus / side
        if not side_dir.is_dir():
            raise MetadataError(f"missing published-card directory {side_dir}")
        expected_side = set(expected[side])
        for run_dir in sorted(p for p in side_dir.iterdir() if p.is_dir()):
            if not RUN_REF_RE.fullmatch(run_dir.name):
                raise MetadataError(f"unexpected run directory {run_dir}")
            if run_dir.name not in expected_side:
                raise MetadataError(f"{run_dir}: run_ref is not in splits.{side}")
            for path in sorted(run_dir.glob("exp-*.yaml")):
                match = CARD_FILE_RE.fullmatch(path.name)
                if match is None:
                    raise MetadataError(f"unexpected card filename {path}")
                number = int(match.group(1))
                key = (side, run_dir.name, number)
                if key in seen:
                    raise MetadataError(
                        f"duplicate published card {side}/{run_dir.name}/exp-{number}"
                    )
                seen.add(key)
                try:
                    value = yaml.safe_load(path.read_text(encoding="utf-8"))
                except (OSError, yaml.YAMLError) as exc:
                    raise MetadataError(f"cannot read card {path}: {exc}") from exc
                if not isinstance(value, dict):
                    raise MetadataError(f"{path}: top level must be a mapping")
                want_id = path.stem
                if value.get("card_id") != want_id:
                    raise MetadataError(
                        f"{path}: card_id is {value.get('card_id')!r}, expected {want_id!r}"
                    )
                provenance = value.get("provenance")
                got_ref = provenance.get("run_ref") if isinstance(provenance, dict) else None
                if got_ref != run_dir.name:
                    raise MetadataError(
                        f"{path}: provenance.run_ref is {got_ref!r}, expected {run_dir.name!r}"
                    )
                cards.append(PublishedCard(side, run_dir.name, number, path, value))
    return sorted(cards, key=lambda card: (SIDES.index(card.side), card.run_ref, card.number))


def _field(value: dict[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _present(value: Any) -> bool:
    return value not in (None, "", [], {})


def _preserved_problems(corpus: Path) -> list[str]:
    """Load source-audit flags that cannot be reconstructed from cards alone.

    The original checker compared each card with a private source digest.  Those
    digests and ``sources.json`` are intentionally not published, so rebuilding
    counts from the public tree must carry the checker's existing findings
    forward rather than replacing them with an empty list.
    """

    path = corpus / "coverage.json"
    if not path.is_file():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MetadataError(f"cannot preserve source-audit problems from {path}: {exc}") from exc
    problems = value.get("problems") if isinstance(value, dict) else None
    if problems is None:
        return []
    if not isinstance(problems, list) or not all(isinstance(problem, str) for problem in problems):
        raise MetadataError(f"{path}: problems must be a list of strings")
    return list(problems)


def _coverage(
    cards: list[PublishedCard],
    expected: dict[str, tuple[str, ...]],
    *,
    problems: list[str],
) -> dict[str, Any]:
    card_counts = {side: sum(card.side == side for card in cards) for side in SIDES}
    seen_refs = {
        side: {card.run_ref for card in cards if card.side == side}
        for side in SIDES
    }
    missing = {
        side: sorted(set(expected[side]) - seen_refs[side])
        for side in SIDES
    }
    return {
        "cards": len(cards),
        "cards_by_side": card_counts,
        "expected_runs": sum(len(expected[side]) for side in SIDES),
        "expected_runs_by_side": {side: len(expected[side]) for side in SIDES},
        "expected_run_refs_by_side": {
            side: list(expected[side]) for side in SIDES
        },
        "field_coverage": {
            dotted: sum(_present(_field(card.value, dotted)) for card in cards)
            for dotted in FIELD_PATHS
        },
        "problems": problems,
        "runs": sum(len(seen_refs[side]) for side in SIDES),
        "runs_by_side": {side: len(seen_refs[side]) for side in SIDES},
        "runs_without_cards": {
            "count": sum(len(missing[side]) for side in SIDES),
            "by_side": missing,
            "cause": "unknown",
            "evidence": (
                "The published corpus contains no zero-card manifests, source digests, "
                "or exclusion report from which to distinguish no qualifying launch from "
                "an extraction omission or failure."
            ),
        },
        "source_audit": {
            "status": "not_recomputed",
            "preserved_problem_count": len(problems),
            "reason": (
                "sources.json and the source digests are not part of the published corpus; "
                "the original checker's problems are preserved from coverage.json"
            ),
        },
    }


def _scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


def _markdown(value: Any) -> str:
    return " ".join(_scalar(value).split()).replace("|", r"\|")


def _best_measurement(card: dict[str, Any]) -> int | float | None:
    measurements = _field(card, "result.measurements")
    if not isinstance(measurements, list):
        return None
    values = [
        measurement.get("value")
        for measurement in measurements
        if isinstance(measurement, dict)
        and isinstance(measurement.get("value"), (int, float))
        and not isinstance(measurement.get("value"), bool)
    ]
    return max(values) if values else None


def _data_sources(card: dict[str, Any]) -> str:
    data = _field(card, "setup.data")
    if not isinstance(data, list):
        return ""
    return ",".join(
        str(row.get("source"))
        for row in data
        if isinstance(row, dict) and row.get("source") not in (None, "")
    )


def _index_row(card: PublishedCard, split_base_model: str | None = None) -> list[Any]:
    value = card.value
    setup = value.get("setup") if isinstance(value.get("setup"), dict) else {}
    parent = setup.get("parent_checkpoint")
    parent = parent if isinstance(parent, dict) else {}
    method = setup.get("method")
    method = method if isinstance(method, dict) else {}
    provenance = value.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    stated = provenance.get("stated_by_agent")
    stated = stated if isinstance(stated, dict) else {}
    base_model = split_base_model or setup.get("base_model") or parent.get("path")
    return [
        card.side,
        card.run_ref,
        f"exp-{card.number:02d}",
        base_model,
        provenance.get("launch_i"),
        method.get("family"),
        parent.get("origin"),
        _data_sources(value),
        _field(value, "result.execution"),
        _best_measurement(value),
        _field(value, "conclusion.verdict"),
        _field(value, "conclusion.decision"),
        stated.get("hypothesis"),
        _field(value, "outcome.official_accuracy"),
    ]


def _render_index(
    cards: list[PublishedCard], coverage: dict[str, Any], split_base_models: dict[str, str]
) -> str:
    train_runs = coverage["runs_by_side"]["train"]
    test_runs = coverage["runs_by_side"]["test"]
    lines = [
        "<!-- Generated by tools/rebuild_exp_card_metadata.py; do not edit by hand. -->",
        "# Index of reconstructed cards",
        "",
        (
            f"Published corpus: {coverage['cards']:,} cards across {coverage['runs']} runs "
            f"({train_runs} train, {test_runs} test). Hidden `.probe-*` trees are excluded."
        ),
        "",
        "| " + " | ".join(INDEX_COLUMNS) + " |",
        "|" + "|".join("---" for _ in INDEX_COLUMNS) + "|",
    ]
    lines.extend(
        "| "
        + " | ".join(
            _markdown(cell) for cell in _index_row(card, split_base_models.get(card.run_ref))
        )
        + " |"
        for card in cards
    )
    return "\n".join(lines) + "\n"


def render_metadata(
    corpus: Path,
    split_path: Path,
    *,
    problems: list[str] | None = None,
) -> RenderedMetadata:
    expected = expected_run_refs(split_path)
    cards = load_published_cards(corpus, expected)
    coverage = _coverage(
        cards,
        expected,
        problems=_preserved_problems(corpus) if problems is None else list(problems),
    )
    split_base_models = _split_base_models(split_path)
    return RenderedMetadata(
        coverage=json.dumps(coverage, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        index=_render_index(cards, coverage, split_base_models),
    )


def _outputs(corpus: Path, rendered: RenderedMetadata) -> tuple[tuple[Path, str], ...]:
    return ((corpus / "coverage.json", rendered.coverage), (corpus / "index.md", rendered.index))


def stale_outputs(corpus: Path, rendered: RenderedMetadata) -> list[Path]:
    stale = []
    for path, expected in _outputs(corpus, rendered):
        actual = path.read_text(encoding="utf-8") if path.is_file() else None
        if actual != expected:
            stale.append(path)
    return stale


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_outputs(corpus: Path, rendered: RenderedMetadata) -> None:
    for path, text in _outputs(corpus, rendered):
        _atomic_write(path, text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="make no writes; exit 1 if coverage.json or the root index.md is stale",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        rendered = render_metadata(args.corpus.resolve(), args.split.resolve())
    except MetadataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    stale = stale_outputs(args.corpus, rendered)
    if args.check:
        if stale:
            print(
                "stale generated metadata: " + ", ".join(str(path) for path in stale),
                file=sys.stderr,
            )
            return 1
        print("experiment-card metadata is current")
        return 0
    write_outputs(args.corpus, rendered)
    print("updated " + ", ".join(str(path) for path, _ in _outputs(args.corpus, rendered)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
