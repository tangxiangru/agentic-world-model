#!/usr/bin/env python3
"""Emit or validate the complete scientist x information study matrix."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass


SCIENTIST_MODELS = (
    "claude-opus-4-8",
    "claude-opus-5",
)
CONDITIONS = ("c1", "c2", "c3")
ARMS = {"c0": None, "c1": None, "c2": "traj", "c3": "retrieval"}
# C0 is the no-prior-information baseline. It has no scope factor (there is no
# corpus to scope), so four repetitions per scientist model give the same four
# runs per model that every other condition gets from 2 scopes x 2 repetitions.
NO_PRIOR_CONDITION = "c0"
NO_PRIOR_SCOPE = "none"
NO_PRIOR_REPETITIONS = (1, 2, 3, 4)
SCOPES = ("train", "train,test")
REPETITIONS = (1, 2)
EXPECTED_CELL_COUNT = (
    len(SCIENTIST_MODELS) * len(CONDITIONS) * len(SCOPES) * len(REPETITIONS)
)
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
        if self.condition == NO_PRIOR_CONDITION:
            fields.append(str(self.repetition))
            return ":".join(fields)
        if ARMS[self.condition] is not None:
            fields.append(ARMS[self.condition])
        fields.extend((self.scope, str(self.repetition)))
        return ":".join(fields)

    def record(self) -> dict[str, str | int]:
        return {
            **asdict(self),
            "benchmark": BENCHMARK,
            "base_model": BASE_MODEL,
            "num_hours": NUM_HOURS,
            "wma_arm": ARMS[self.condition],
            "prior_rollout_count": (
                0 if self.condition == NO_PRIOR_CONDITION
                else 193 if self.scope == "train,test" else 143
            ),
            "includes_gemma_trajectories": self.scope == "train,test",
            "setting": {
                "c0": "no_prior_information",
                "c1": "raw_trajectories_no_wma",
                "c2": "raw_trajectories_with_traj_wma",
                "c3": "experiment_cards_with_retrieval_wma",
            }[self.condition],
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
    if (
        len(cells) != EXPECTED_CELL_COUNT
        or len({cell.spec for cell in cells}) != len(cells)
    ):
        raise RuntimeError(
            f"internal error: study matrix is not {EXPECTED_CELL_COUNT} unique cells"
        )
    return cells


def c0_matrix() -> tuple[Cell, ...]:
    """The no-prior baseline: every scientist model x four repetitions, no scope."""
    cells = tuple(
        Cell(NO_PRIOR_CONDITION, model, NO_PRIOR_SCOPE, repetition)
        for repetition in NO_PRIOR_REPETITIONS
        for model in SCIENTIST_MODELS
    )
    if len({cell.spec for cell in cells}) != len(cells):
        raise RuntimeError("internal error: C0 matrix has duplicate cells")
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
        help=(
            "validate that the positional specs are exactly the "
            f"{EXPECTED_CELL_COUNT}-cell matrix"
        ),
    )
    parser.add_argument(
        "--c0",
        action="store_true",
        help="emit or validate the C0 no-prior baseline cells instead of the 24-cell matrix",
    )
    parser.add_argument("spec", nargs="*", help="cell specs used with --validate")
    args = parser.parse_args(argv)

    matrix = c0_matrix() if args.c0 else study_matrix()
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
