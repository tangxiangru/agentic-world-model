"""Turn a batch of per-run annotations into tables, and say how much to trust them.

Two jobs. The first is mechanical: read the annotation files, drop anything whose
evidence pointer does not resolve, and stack what survives into the three tables
the reference document reports from.

The second decides whether those tables may be quoted at all. Two agents
annotate the same runs without seeing each other's work, and the agreement
between them is what places each judged field on the evidence ladder. Cohen's
kappa rather than raw agreement, because the categories are lopsided — most
trainings change both the recipe and the hyperparameters, so two annotators who
answered ``both`` every time would agree about 60% of the time while knowing
nothing.

The thresholds are the spec's, and they bind: below 0.6 a field is discarded
along with every conclusion resting on it. That is the point of measuring. A
number produced by an agent and never checked against a second agent is not
observational evidence, it is one opinion with a table around it.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from awm.traj import verify_annotations as va

#: Spec §5.2. Where a field lands decides whether it may enter the main table.
KAPPA_QUOTABLE = 0.8
KAPPA_FLOOR = 0.6

_TABLES = ("changes", "trainings", "verifications")
_PROPOSALS = ("proposed_category", "definition_defect", "boundary_case")


@dataclass(frozen=True)
class Agreement:
    field: str
    n: int
    observed: float
    expected: float
    kappa: float

    @property
    def verdict(self) -> str:
        if self.kappa >= KAPPA_QUOTABLE:
            return "quotable"
        if self.kappa >= KAPPA_FLOOR:
            return "findings_only"
        return "discard"


def cohens_kappa(pairs: list[tuple[Any, Any]]) -> Agreement | None:
    """Chance-corrected agreement over paired labels, or ``None`` if degenerate."""
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    left, right = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(left[k] * right[k] for k in set(left) | set(right)) / (n * n)
    if expected >= 1.0:
        # One label used throughout by both: agreement is total but uninformative.
        return Agreement("", n, observed, expected, float("nan"))
    return Agreement("", n, observed, expected, (observed - expected) / (1 - expected))


def jaccard(pairs: list[tuple[set[Any], set[Any]]]) -> float | None:
    """Mean overlap of two link sets; two empty sets count as full agreement."""
    if not pairs:
        return None
    scores = []
    for a, b in pairs:
        union = a | b
        scores.append(1.0 if not union else len(a & b) / len(union))
    return sum(scores) / len(scores)


def load_batch(
    directory: Path, events_root: Path | None = None
) -> tuple[dict[str, dict[str, Any]], dict[str, va.Report]]:
    """Every annotation in a directory, with its pointer-check report."""
    annotations: dict[str, dict[str, Any]] = {}
    reports: dict[str, va.Report] = {}
    for path in sorted(Path(directory).glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        run_id = data.get("run_id") or path.stem
        data["run_id"] = run_id
        annotations[run_id] = data
        try:
            reports[run_id] = va.check_annotation(data, events_root)
        except FileNotFoundError:
            reports[run_id] = va.Report(
                problems=[va.Problem(run_id, "-", 0, "no_stream", "event file missing")]
            )
    return annotations, reports


def _rejected_rows(report: va.Report) -> set[tuple[str, int]]:
    return {(p.table, p.row) for p in report.problems}


def tables(
    annotations: dict[str, dict[str, Any]], reports: dict[str, va.Report]
) -> dict[str, pd.DataFrame]:
    """The stacked tables, with rows whose evidence did not resolve removed."""
    out: dict[str, list[dict[str, Any]]] = {t: [] for t in _TABLES + _PROPOSALS}
    for run_id, data in annotations.items():
        rejected = _rejected_rows(reports.get(run_id, va.Report()))
        for table in _TABLES + _PROPOSALS:
            for row, entry in enumerate(data.get(table) or []):
                if not isinstance(entry, dict) or (table, row) in rejected:
                    continue
                out[table].append({"run_id": run_id, **entry})
    return {t: pd.DataFrame(rows) for t, rows in out.items()}


def resolve_unclear(trainings: pd.DataFrame, spans: pd.DataFrame) -> pd.DataFrame:
    """Split ``unclear`` into the schema's shortfall and genuine insufficiency.

    Six annotators, independently, reported the same thing: a smoke test has no
    tested variable and the first real training has nothing to be compared
    against, so both were forced into ``unclear`` — the value the spec reserves
    for *insufficient evidence*. Mixing them makes the unclear share, which §9
    requires reported, mean two different things at once.

    The separation needs no judgement: the mechanical layer already knows which
    spans are smoke tests and which real training came first. Applying it here
    rather than asking for re-annotation keeps the runs already annotated and
    the ones still to come on the same footing.

    What remains ``unclear`` after this is the honest number: restart chains
    where every change is a memory compensation, and cases where the trace
    genuinely does not say.
    """
    if not len(trainings) or "tested_variable" not in trainings:
        return trainings
    out = trainings.copy()
    kind = {(r.run_id, r.i): r.kind for r in spans.itertuples()}
    first_real = (
        spans[spans["kind"] == "real"].groupby("run_id")["i"].min().to_dict()
        if len(spans) else {}
    )
    for pos, row in out.iterrows():
        if row.get("tested_variable") != "unclear":
            continue
        key = (row.get("run_id"), row.get("i"))
        if kind.get(key) == "smoke":
            out.at[pos, "tested_variable"] = "smoke"
        elif first_real.get(row.get("run_id")) == row.get("i"):
            out.at[pos, "tested_variable"] = "baseline"
    return out


def _by_key(rows: Iterable[dict[str, Any]], key: str) -> dict[Any, dict[str, Any]]:
    return {r[key]: r for r in rows if key in r}


def agreement(
    first: dict[str, dict[str, Any]], second: dict[str, dict[str, Any]]
) -> dict[str, Agreement | float | None]:
    """Field-level agreement over the runs both annotators covered."""
    shared = sorted(set(first) & set(second))
    tested: list[tuple[Any, Any]] = []
    category: list[tuple[Any, Any]] = []
    links: list[tuple[set[Any], set[Any]]] = []

    for run_id in shared:
        a, b = first[run_id], second[run_id]
        ta, tb = _by_key(a.get("trainings") or [], "i"), _by_key(b.get("trainings") or [], "i")
        for i in set(ta) & set(tb):
            tested.append((ta[i].get("tested_variable"), tb[i].get("tested_variable")))
        # Changes have no shared id, so compare the category assigned at each
        # anchoring event; a change one annotator saw and the other did not is a
        # coverage difference, not a disagreement, and is counted separately.
        ca, cb = _by_key(a.get("changes") or [], "i"), _by_key(b.get("changes") or [], "i")
        for i in set(ca) & set(cb):
            category.append((ca[i].get("category"), cb[i].get("category")))
        # Link sets are compared by the *anchoring event* of each change, never
        # by the change_id: the two annotators name the same change ``c1`` and
        # ``x1``, so comparing ids measures nothing but their private spelling.
        anchor_a = {c.get("change_id"): c.get("i") for c in a.get("changes") or []}
        anchor_b = {c.get("change_id"): c.get("i") for c in b.get("changes") or []}
        va_, vb = (
            _by_key(a.get("verifications") or [], "i"),
            _by_key(b.get("verifications") or [], "i"),
        )
        for i in set(va_) & set(vb):
            links.append((
                {anchor_a[c] for c in va_[i].get("judges_changes") or [] if c in anchor_a},
                {anchor_b[c] for c in vb[i].get("judges_changes") or [] if c in anchor_b},
            ))

    out: dict[str, Agreement | float | None] = {}
    for name, pairs in (("tested_variable", tested), ("category", category)):
        k = cohens_kappa(pairs)
        if k is not None:
            out[name] = Agreement(name, k.n, k.observed, k.expected, k.kappa)
        else:
            out[name] = None
    out["judges_changes_jaccard"] = jaccard(links)
    out["runs_compared"] = float(len(shared))
    return out


def summarise(
    annotations: dict[str, dict[str, Any]], reports: dict[str, va.Report]
) -> dict[str, Any]:
    """Batch-level health: coverage, rejection rate, and the unclear share."""
    t = tables(annotations, reports)
    trainings = t["trainings"]
    checked = sum(r.checked for r in reports.values())
    rejected = sum(len(r.problems) for r in reports.values())
    unclear = (
        float((trainings["tested_variable"] == "unclear").mean())
        if len(trainings) and "tested_variable" in trainings
        else float("nan")
    )
    return {
        "runs": len(annotations),
        "judgements_checked": checked,
        "judgements_rejected": rejected,
        "rejection_rate": rejected / checked if checked else float("nan"),
        "changes": len(t["changes"]),
        "trainings": len(trainings),
        "verifications": len(t["verifications"]),
        "unclear_share": unclear,
        "proposals": {p: len(t[p]) for p in _PROPOSALS},
    }


__all__ = [
    "Agreement",
    "resolve_unclear",
    "KAPPA_FLOOR",
    "KAPPA_QUOTABLE",
    "agreement",
    "cohens_kappa",
    "jaccard",
    "load_batch",
    "summarise",
    "tables",
]
