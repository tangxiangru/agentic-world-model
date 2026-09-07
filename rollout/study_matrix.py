#!/usr/bin/env python3
"""Emit or validate the recorder study matrix.

The collection run wants coverage of recipe space, not convergence on known-good
recipes, so every cell is a *recorder* cell: the scientist gets no prior
information of any kind and registers each experiment by command
(``awm wm submit``). The matrix is balanced by construction — half the cells
run each scientist model — over two PostTrainBench tasks, two base models, and
N repetitions (HealthBench parked until a grader key exists):

    2 scientists x 1 task x 2 base models x N repetitions    (N = 2 -> 8 cells)

Every cell is one H100 for ten hours. ``--format specs`` emits one
``wm_pack.sbatch`` spec per line; ``--reps`` changes N.
"""

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
# qwen3.6-27b is the RPM paper's frozen backbone, served locally by vLLM and
# driven through PTB's opencode scaffold; select it with --scientists.
KNOWN_SCIENTISTS = SCIENTIST_MODELS + ("qwen3.6-27b",)
# healthbench is LLM-graded (gpt-5-mini via OPENAI_API_KEY) and is parked until a
# grader key exists; the launcher still accepts it for individual cells.
TASKS = ("gpqamain",)
# spec alias -> Hugging Face id, as PostTrainBench names its base models.
BASE_MODELS = {
    "gemma3-4b": "google/gemma-3-4b-pt",
    "qwen3-4b": "Qwen/Qwen3-4B-Base",
}
RECORDER_CONDITION = "r"
DEFAULT_REPETITIONS = 2
MAX_REPETITIONS = 8          # wm_pack.sbatch accepts r:...:[1-8]
NUM_HOURS = 10

# C0 is the no-registration no-prior baseline kept for reference: gsm8k on
# gemma, four repetitions per scientist, no scope factor.
NO_PRIOR_CONDITION = "c0"
NO_PRIOR_TASK = "gsm8k"
NO_PRIOR_BASE = "gemma3-4b"
NO_PRIOR_REPETITIONS = (1, 2, 3, 4)


@dataclass(frozen=True)
class Cell:
    condition: str
    scientist_model: str
    task: str
    base_alias: str
    repetition: int

    @property
    def spec(self) -> str:
        if self.condition == NO_PRIOR_CONDITION:
            return ":".join((self.condition, self.scientist_model, str(self.repetition)))
        return ":".join(
            (self.condition, self.scientist_model, self.task, self.base_alias, str(self.repetition))
        )

    def record(self) -> dict[str, str | int | None]:
        return {
            **asdict(self),
            "benchmark": self.task,
            "base_model": BASE_MODELS[self.base_alias],
            "num_hours": NUM_HOURS,
            "wma_arm": None,
            "prior_rollout_count": 0,
            "includes_gemma_trajectories": False,
            "setting": {
                RECORDER_CONDITION: "no_prior_information_recorder",
                NO_PRIOR_CONDITION: "no_prior_information",
            }[self.condition],
            "study_mode": "production",
            "spec": self.spec,
        }


def recorder_matrix(repetitions: int = DEFAULT_REPETITIONS,
                    scientists: tuple[str, ...] = SCIENTIST_MODELS) -> tuple[Cell, ...]:
    """Scientists x tasks x base models x repetitions; equal cells per scientist."""
    if not 1 <= repetitions <= MAX_REPETITIONS:
        raise ValueError(f"repetitions must be 1..{MAX_REPETITIONS}, got {repetitions}")
    unknown = sorted(set(scientists) - set(KNOWN_SCIENTISTS))
    if unknown or not scientists:
        raise ValueError(f"unknown scientists {unknown}; known: {list(KNOWN_SCIENTISTS)}")
    cells = tuple(
        Cell(RECORDER_CONDITION, model, task, base, repetition)
        for repetition in range(1, repetitions + 1)
        for task in TASKS
        for base in BASE_MODELS
        for model in scientists
    )
    expected = len(scientists) * len(TASKS) * len(BASE_MODELS) * repetitions
    per_model = Counter(cell.scientist_model for cell in cells)
    if (
        len(cells) != expected
        or len({cell.spec for cell in cells}) != len(cells)
        or len(set(per_model.values())) != 1
    ):
        raise RuntimeError(f"internal error: recorder matrix is not {expected} balanced unique cells")
    return cells


def c0_matrix() -> tuple[Cell, ...]:
    """The no-registration baseline: every scientist model x four repetitions."""
    cells = tuple(
        Cell(NO_PRIOR_CONDITION, model, NO_PRIOR_TASK, NO_PRIOR_BASE, repetition)
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--format",
        choices=("json", "specs"),
        default="json",
        help="JSON records (default), or one wm_pack.sbatch spec per line",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=DEFAULT_REPETITIONS,
        help=f"repetitions per (scientist, task, base) cell, 1..{MAX_REPETITIONS} (default {DEFAULT_REPETITIONS})",
    )
    parser.add_argument(
        "--scientists",
        default=",".join(SCIENTIST_MODELS),
        help=f"comma-separated scientists (default: the two Claude models; known: {', '.join(KNOWN_SCIENTISTS)})",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="validate that the positional specs are exactly the matrix",
    )
    parser.add_argument(
        "--c0",
        action="store_true",
        help="emit or validate the C0 no-registration baseline cells instead",
    )
    parser.add_argument("spec", nargs="*", help="cell specs used with --validate")
    args = parser.parse_args(argv)

    try:
        scientists = tuple(x for x in args.scientists.split(",") if x)
        matrix = c0_matrix() if args.c0 else recorder_matrix(args.reps, scientists)
    except ValueError as exc:
        parser.error(str(exc))
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
