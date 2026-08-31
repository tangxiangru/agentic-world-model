#!/usr/bin/env python3
"""Emit or validate the complete scientist x information study matrix."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass


SCIENTIST_MODELS = (
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
)
CONDITIONS = ("c1", "c2", "c3")
SCOPES = ("train", "train,test")
REPETITIONS = (1, 2)
BENCHMARK = "gsm8k"
BASE_MODEL = "google/gemma-3-4b-pt"
NUM_HOURS = 10


@dataclass(frozen=True)
class Cell:
    condition: str
    scientist_model: str
    scope: str
    repetition: int

    @property
    def spec(self) -> str:
        fields = [self.condition, self.scientist_model]
        if self.condition != "c1":
            fields.append("llm")
        fields.extend((self.scope, str(self.repetition)))
        return ":".join(fields)

    def record(self) -> dict[str, str | int]:
        return {
            **asdict(self),
            "benchmark": BENCHMARK,
            "base_model": BASE_MODEL,
            "num_hours": NUM_HOURS,
            "study_mode": "production",
            "spec": self.spec,
        }


def study_matrix() -> tuple[Cell, ...]:
    cells = tuple(
        Cell(condition, model, scope, repetition)
        for repetition in REPETITIONS
        for scope in SCOPES
        for condition in CONDITIONS
        for model in SCIENTIST_MODELS
    )
    if len(cells) != 36 or len({cell.spec for cell in cells}) != len(cells):
        raise RuntimeError("internal error: study matrix is not 36 unique cells")
    return cells


def validate_specs(supplied: list[str], expected: tuple[Cell, ...]) -> None:
    counts = Counter(supplied)
    duplicates = sorted(spec for spec, count in counts.items() if count > 1)
    expected_specs = {cell.spec for cell in expected}
    actual_specs = set(supplied)
    missing = sorted(expected_specs - actual_specs)
    unexpected = sorted(actual_specs - expected_specs)
    if duplicates or missing or unexpected or len(supplied) != len(expected):
        details = {
            "duplicates": duplicates,
            "expected_count": len(expected),
            "missing": missing,
            "supplied_count": len(supplied),
            "unexpected": unexpected,
        }
        raise ValueError(json.dumps(details, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=("json", "specs"),
        default="json",
        help="JSON records (default), or one wm_pack.sbatch spec per line",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="validate that the positional specs are exactly the 36-cell matrix",
    )
    parser.add_argument("spec", nargs="*", help="cell specs used with --validate")
    args = parser.parse_args(argv)

    matrix = study_matrix()
    if args.validate:
        try:
            validate_specs(args.spec, matrix)
        except ValueError as exc:
            print(f"invalid study matrix: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({"cell_count": len(matrix), "valid": True}, sort_keys=True))
        return 0
    if args.spec:
        parser.error("positional specs require --validate")

    if args.format == "specs":
        print(*(cell.spec for cell in matrix), sep="\n")
    else:
        print(json.dumps([cell.record() for cell in matrix], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
