"""The card chain and the index built from it.

``chain`` follows ``setup.parent_checkpoint.origin`` back to the base model:
that path is the recipe that shipped. ``starting_points`` says which adopted
checkpoints still exist on disk — those can be resumed — and which survive
only as a recipe. ``memory/index.md`` is one line per card; a resumed
scientist reads it first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .lock import read_lock
from .schema import CardError, get, load_card

BASE = "base_model"
UNKNOWN = "unknown"


def cards_dir(session_dir: Path) -> Path:
    return Path(session_dir) / "memory" / "cards"


def load_cards(cards_directory: Path,
               problems: list[tuple[Path, str]] | None = None) -> dict[str, dict[str, Any]]:
    """Every readable ``exp-*.yaml``; unreadable ones are skipped and, if ``problems`` is given, reported there."""
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(Path(cards_directory).glob("exp-*.yaml")):
        try:
            card = load_card(p)
        except CardError as exc:
            if problems is not None:
                problems.append((p, str(exc)))
            continue
        card["_path"] = str(p)
        out[str(card.get("card_id") or p.stem)] = card
    return out


def chain(cards: dict[str, dict[str, Any]], card_id: str) -> list[str]:
    """``card_id`` back to ``base_model``; ``unknown`` when a card does not say, never a guessed parent."""
    out = [card_id]
    seen = {card_id}
    cur = card_id
    while True:
        card = cards.get(cur)
        if card is None:
            out.append(f"missing:{cur}")
            return out
        origin = get(card, "setup.parent_checkpoint.origin")
        if origin is None or origin == "":
            out.append(UNKNOWN)
            return out
        if not isinstance(origin, str):
            out.append(f"invalid:{origin!r}")
            return out
        if origin == BASE:
            out.append(BASE)
            return out
        if origin in seen:
            out.append(f"cycle:{origin}")
            return out
        seen.add(origin)
        out.append(origin)
        cur = origin


def _best(card: dict[str, Any]) -> str:
    ms = get(card, "result.measurements") or []
    ms = [m for m in ms if isinstance(m, dict)
          and isinstance(m.get("value"), (int, float)) and not isinstance(m.get("value"), bool)]
    if not ms:
        return ""
    m = max(ms, key=lambda x: x["value"])
    return f"{m.get('metric', '?')}={m['value']}"


def _status(card: dict[str, Any]) -> str:
    if isinstance(card.get("conclusion"), dict) and get(card, "conclusion.decision"):
        return "closed"
    return "open"


def index_rows(cards: dict[str, dict[str, Any]], cards_directory: Path) -> list[dict[str, Any]]:
    rows = []
    for card_id in sorted(cards):
        card = cards[card_id]
        path = Path(card.get("_path") or Path(cards_directory) / f"{card_id}.yaml")
        info = read_lock(path)
        rows.append({
            "card_id": card_id,
            "elapsed_h": get(card, "situation.elapsed_h"),
            "family": get(card, "setup.method.family") or "",
            "parent": get(card, "setup.parent_checkpoint.origin") or "",
            "status": _status(card),
            "locked": info is not None,
            "relocks": len((info or {}).get("relocked_from") or []),
            "verdict": get(card, "conclusion.verdict") or "",
            "decision": get(card, "conclusion.decision") or "",
            "best": _best(card),
            "checkpoint": get(card, "result.output_checkpoint") or "",
        })
    return rows


def render_index(rows: list[dict[str, Any]], unreadable: list[tuple[Path, str]] | None = None) -> str:
    head = ["| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |",
            "|---|---:|---|---|---|---|---|---|---|---|"]
    body = []
    for r in rows:
        locked = "yes" if r["locked"] else "no"
        if r.get("relocks"):
            locked += f" (re-locked {r['relocks']}x)"
        cells = {**r, "elapsed_h": "" if r["elapsed_h"] is None else r["elapsed_h"], "locked": locked}
        body.append("| {card_id} | {elapsed_h} | {family} | {parent} | {status} | {locked} | {verdict} "
                    "| {decision} | {best} | {checkpoint} |".format(**cells))
    text = ("# Experiment index\n\nOne line per card; newest last. Read this before opening a new card.\n\n"
            + "\n".join(head + body) + "\n")
    if unreadable:
        text += "\n## Unreadable card files\n\n" + "".join(f"- `{p.name}`: {why}\n" for p, why in unreadable)
    return text


def write_index(session_dir: Path) -> Path:
    cdir = cards_dir(session_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    problems: list[tuple[Path, str]] = []
    rows = index_rows(load_cards(cdir, problems=problems), cdir)
    out = Path(session_dir) / "memory" / "index.md"
    out.write_text(render_index(rows, problems))
    return out


def starting_points(cards: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Adopted cards: resume from the checkpoint if it still exists, else rerun the chain."""
    points = []
    for card_id in sorted(cards):
        card = cards[card_id]
        if get(card, "conclusion.decision") != "adopt":
            continue
        ckpt = get(card, "result.output_checkpoint")
        exists = bool(ckpt) and Path(str(ckpt)).is_dir()
        points.append({"card_id": card_id, "checkpoint": ckpt if exists else None,
                       "level": "checkpoint" if exists else "recipe",
                       "measurement": _best(card), "chain": chain(cards, card_id)})
    return points
