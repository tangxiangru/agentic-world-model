"""Shared readers for the WMA root-cause kit.

Everything here reads a harvested PTB cell directory
(``results/ptb/<batch>/<cell>/``: ``metrics.json``, ``task/memory/cards/*.yaml`` with their
``.lock.json`` / ``.verdict.json``, ``task/.wma/{processed,responses}``, ``wma_private/*.transcript.jsonl.gz``,
``solve_parsed.txt.gz``, ``time_taken.txt``) or an in-flight snapshot (``<cell>.inflight/``, transcripts
only). Nothing is written by this module; the scripts beside it write under ``--out``.

Scientist identity is never read or used: a cell is its id and its files.
"""

from __future__ import annotations

import gzip
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml

TRAIN_FAMILIES = {"sft", "rft", "dpo", "grpo", "rl", "distill", "continued-pretrain", "lora", "full-ft"}
TURN = re.compile(r"^(Assistant|User) — turn \d+ \| (\S+)")
TOOL_CALL = re.compile(r"^  Tool call — (\w+) \(")
TOOL_RESULT = re.compile(r"^  Tool result — (\w+) \(")
LAUNCH = re.compile(r"nohup|evaluate\.py|train_sft|train\.py|accelerate launch|torchrun|trl ")


# ---------------------------------------------------------------- small helpers

def ts(value: str | None) -> datetime | None:
    if not value:
        return None
    value = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def hhmm(value: datetime | None) -> str | None:
    return value.isoformat()[11:16] if value else None


def g(d: Any, path: str, default: Any = None) -> Any:
    for key in path.split("."):
        if not isinstance(d, dict) or key not in d:
            return default
        d = d[key]
    return d


def num(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def noise_floor(n: float | None) -> float | None:
    """The manual's §5 rule of thumb: what an accuracy read at n items cannot resolve."""
    if not n or n <= 0:
        return None
    floor = 0.01 if n >= 1000 else 0.02 if n >= 500 else 0.03 if n >= 150 else 0.07 if n >= 50 else 0.12
    return max(floor, 1.0 / n)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    def rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for k in range(i, j + 1):
                ranks[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return ranks

    if len(xs) < 3 or len(xs) != len(ys):
        return None
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx and vy else None


def median(values: list[float]) -> float | None:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


# ---------------------------------------------------------------- cell discovery

def batch_cells(batch_dirs: list[str | Path], *, inflight: bool = False) -> Iterator[tuple[str, str, Path, bool]]:
    """Yield (batch, cell, dir, is_inflight) for every harvested cell (and, with inflight=True,
    every ``<cell>.inflight`` snapshot) under the given batch directories."""
    for batch_dir in batch_dirs:
        batch_dir = Path(batch_dir)
        batch = batch_dir.name
        for child in sorted(batch_dir.iterdir()):
            if not child.is_dir():
                continue
            if child.name.endswith(".inflight"):
                if inflight:
                    yield batch, child.name.split(".")[0], child, True
            elif (child / "task").is_dir() or (child / "metrics.json").is_file():
                yield batch, child.name, child, False


def cell_arm(cell_dir: Path) -> str:
    """wma when a sidecar left its status file or any verdict beside the cards; ctl otherwise."""
    if (cell_dir / "task" / ".wma" / "sidecar_status.json").is_file():
        return "wma"
    if any((cell_dir / "task" / "memory" / "cards").glob("exp-*.verdict.json*")):
        return "wma"
    if (cell_dir / "wma_private").is_dir():
        return "wma"
    return "ctl"


# ---------------------------------------------------------------- readers

def final_accuracy(cell_dir: Path) -> float | None:
    path = cell_dir / "metrics.json"
    if not path.is_file():
        return None
    try:
        return num(json.loads(path.read_text()).get("accuracy"))
    except (OSError, ValueError):
        return None


def time_taken_h(cell_dir: Path) -> float | None:
    path = cell_dir / "time_taken.txt"
    if not path.is_file():
        return None
    try:
        h, m, s = path.read_text().strip().split(":")
        return int(h) + int(m) / 60 + int(s) / 3600
    except ValueError:
        return None


def load_cards(cell_dir: Path) -> dict[str, dict]:
    cards: dict[str, dict] = {}
    for path in sorted((cell_dir / "task" / "memory" / "cards").glob("exp-*.yaml")):
        try:
            loaded = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError):
            continue
        if isinstance(loaded, dict):
            cards[path.stem] = loaded
    return cards


def load_locks(cell_dir: Path) -> dict[str, dict]:
    """card id -> lock file (``locked_at``, ``relocked_from``, and since 2026-09-03 ``wma``)."""
    locks: dict[str, dict] = {}
    for path in (cell_dir / "task" / "memory" / "cards").glob("exp-*.lock.json"):
        try:
            locks[path.name.split(".")[0]] = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
    return locks


def load_verdicts(cell_dir: Path) -> dict[str, dict]:
    """card id -> verdict; a ``.rejected`` file (validator refusal) is kept with ``_rejected=True``
    only when no accepted verdict exists for that card."""
    verdicts: dict[str, dict] = {}
    for path in sorted((cell_dir / "task" / "memory" / "cards").glob("exp-*.verdict.json*")):
        card_id = path.name.split(".")[0]
        try:
            verdict = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        verdict["_rejected"] = path.name.endswith(".rejected")
        if card_id not in verdicts or (verdicts[card_id]["_rejected"] and not verdict["_rejected"]):
            verdicts[card_id] = verdict
    return verdicts


def load_queue(cell_dir: Path) -> tuple[dict[str, str], dict[str, dict]]:
    """Requests (card id -> created_at) and responses (card id -> {completed_at, state, error})."""
    control = cell_dir / "task" / ".wma"
    requests: dict[str, str] = {}
    responses: dict[str, dict] = {}
    for sub in ("processed", "processing", "requests"):
        for path in (control / sub).glob("*.json") if (control / sub).is_dir() else []:
            try:
                request = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            for card_id in request.get("card_ids") or []:
                requests.setdefault(str(card_id), request.get("created_at"))
    for path in (control / "responses").glob("*.json") if (control / "responses").is_dir() else []:
        try:
            response = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        for card_id in response.get("card_ids") or []:
            responses.setdefault(str(card_id), {
                "completed_at": response.get("completed_at"), "state": response.get("state"),
                "error": (response.get("errors") or {}).get(card_id) or (response.get("errors") or {}).get("request"),
            })
    return requests, responses


def parse_transcript(cell_dir: Path) -> tuple[list[tuple[datetime | None, str, str]], list[tuple[datetime | None, str]]]:
    """Tool calls ``(time, tool, text)`` and tool results ``(time, text)`` from ``solve_parsed.txt.gz``.
    The text is the call's arguments (or the result's first lines) — enough to find commands, never
    printed by the scripts beyond one line."""
    path = cell_dir / "solve_parsed.txt.gz"
    calls: list[tuple[datetime | None, str, str]] = []
    results: list[tuple[datetime | None, str]] = []
    if not path.is_file():
        return calls, results
    lines = gzip.open(path, "rt", errors="replace").read().splitlines()
    current: datetime | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        turn = TURN.match(line)
        if turn:
            current = ts(turn.group(2))
            i += 1
            continue
        call = TOOL_CALL.match(line)
        if call:
            buffer: list[str] = []
            j = i + 1
            while j < len(lines) and (lines[j].startswith("    ") or lines[j] == "") and len(buffer) <= 60:
                buffer.append(lines[j].strip())
                j += 1
            calls.append((current, call.group(1), "\n".join(buffer)))
            i = j
            continue
        result = TOOL_RESULT.match(line)
        if result:
            buffer = []
            j = i + 1
            while j < len(lines) and (lines[j].startswith("    ") or lines[j] == "") and len(buffer) <= 400:
                buffer.append(lines[j].strip())
                j += 1
            results.append((current, "\n".join(buffer)))
            i = j
            continue
        i += 1
    return calls, results


def bash_command(text: str) -> str:
    """The command line of a Bash tool call, on one line."""
    one = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    if one.startswith("{"):
        try:
            one = json.loads(text).get("command", one)
        except (ValueError, AttributeError):
            pass
    return re.sub(r"\s+", " ", str(one))


def launch_time(card_id: str, card: dict, locked_at: datetime | None,
                calls: list[tuple[datetime | None, str, str]]) -> datetime | None:
    """First Bash call after (lock − 3 min) that launches a training/evaluation naming this card,
    its output directory, or — for eval-only cards — any launch at all."""
    family = g(card, "setup.method.family")
    out_dir = str(g(card, "setup.output_dir") or "").rstrip("/").split("/")[-1]
    for when, tool, text in calls:
        if tool != "Bash" or when is None or locked_at is None or when < locked_at - timedelta(minutes=3):
            continue
        if LAUNCH.search(text) and (card_id in text or (out_dir and out_dir in text) or family == "other"):
            return when
    return None


def read_time(card_id: str, issued: datetime | None, calls, results) -> datetime | None:
    """First tool call (or a result that shows the verdict body) touching ``exp-NN.verdict`` after
    the verdict was issued."""
    if issued is None:
        return None
    for when, _tool, text in calls:
        if when and when >= issued and f"{card_id}.verdict" in text:
            return when
    for when, text in results:
        if when and when >= issued and f"{card_id}.verdict" in text and "L2" in text:
            return when
    return None


def suggestion_text(verdict: dict) -> str:
    parts: list[str] = []
    for key in ("preconditions", "cheaper_variants"):
        for item in (g(verdict, f"suggestions.{key}") or []):
            parts.append(item if isinstance(item, str) else json.dumps(item))
    return "\n".join(parts)


def measurement(card: dict) -> dict:
    measurements = g(card, "result.measurements") or []
    first = measurements[0] if measurements and isinstance(measurements[0], dict) else {}
    return {
        "n": first.get("n"), "value": num(first.get("value")), "delta": num(first.get("delta_vs_comparator")),
        "max_n": max([num(m.get("n")) or 0 for m in measurements if isinstance(m, dict)] + [0]),
    }


def write_outputs(out_dir: Path, name: str, payload: Any, markdown: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(payload, indent=1, default=str) + "\n")
    (out_dir / f"{name}.md").write_text(markdown)
