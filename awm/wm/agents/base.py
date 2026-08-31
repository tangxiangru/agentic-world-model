"""The agent interface. The runtime calls these; arms override what they add."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..schema import CONTRACT_SCHEMA, RULE_SCHEMA


@dataclass
class Brief:
    """What the agent returns for a proposal. The runtime turns it into a brief ping."""

    contract: dict[str, Any]
    grounding: list[dict[str, Any]]
    precedents: list[dict[str, Any]] = field(default_factory=list)
    prediction: dict[str, Any] | None = None
    objections: list[dict[str, Any]] = field(default_factory=list)  # {field, severity: blocking|advisory, fix}
    evidence: list[dict[str, Any]] = field(default_factory=list)
    audit: dict[str, Any] | None = None
    summary: str = ""
    produced_by: str = "deterministic"
    degraded: str | None = None


@dataclass
class Advice:
    """What the agent returns for an observation.

    ``kind`` is ``notice`` (default), ``yield_request`` (asks for evaluators
    from ``on_request``), or ``decision`` (agent-raised, with a recommendation).
    The runtime may upgrade a notice to a decision when a contract rule fires.
    """

    kind: str = "notice"
    summary: str = ""
    evidence: list[dict[str, Any]] = field(default_factory=list)
    prediction: dict[str, Any] | None = None
    request_evaluators: list[str] = field(default_factory=list)
    recommendation: str | None = None  # continue | abort | select:<obs-id>
    precedents: list[dict[str, Any]] = field(default_factory=list)
    audit: dict[str, Any] | None = None
    produced_by: str = "deterministic"
    degraded: str | None = None


class WorldModelAgent:
    """Base policy: deterministic grounding and a default contract, nothing else."""

    arm = "base"

    def __init__(self, **kwargs: Any):
        self.config = kwargs

    # ---- hooks the runtime calls

    def on_proposal(self, card: dict[str, Any], grounding: list[dict[str, Any]],
                    memory: Any, config: dict[str, Any]) -> Brief:
        return Brief(contract=default_contract(card, config), grounding=grounding,
                     summary=default_brief_summary(card, grounding))

    def on_observation(self, observation: dict[str, Any], history: list[dict[str, Any]],
                       contract: dict[str, Any], card: dict[str, Any], memory: Any,
                       config: dict[str, Any]) -> Advice:
        return Advice(kind="notice", summary=observation_summary(observation, contract),
                      evidence=observation_evidence(observation))

    def on_reply(self, ping: dict[str, Any], reply: dict[str, Any], memory: Any) -> None:
        return None

    def on_close(self, card: dict[str, Any], state: dict[str, Any], result: dict[str, Any] | None,
                 memory: Any) -> None:
        return None

    def predict(self, card: dict[str, Any], history: list[dict[str, Any]], memory: Any) -> dict[str, Any] | None:
        return None


# ---------------------------------------------------------------- defaults shared by arms

def default_contract(card: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """A contract derived from the card's own evaluation section and watch set."""
    ev = card["evaluation"]
    n = int(ev["protocol"]["n"])
    metric = card["hypothesis"]["expected_effect"].get("metric", "accuracy")
    direction = card["hypothesis"]["expected_effect"].get("direction", "higher")
    dev_name = f"dev{n}"
    evaluators = [{
        "name": dev_name, "kind": "official", "adapter": config.get("official_adapter", "posttrainbench"),
        "metric": metric, "direction": direction, "n": n, "seed": ev["protocol"].get("seed", 0),
        "stderr": {"method": "bernoulli"},
    }]
    watch = card["problem"]["watch_set"]
    evaluators.append({
        "name": "watch", "kind": "custom", "adapter": "items_exact_match", "metric": "accuracy",
        "direction": "higher", "items": watch["path"], "n": int(watch["n"]), "seed": 0,
        "stderr": {"method": "bernoulli"},
    })
    on_request: list[str] = []
    diag = ev.get("diagnostic") or {}
    if diag.get("items"):
        evaluators.append({
            "name": "diag", "kind": "custom", "adapter": "items_exact_match",
            "metric": diag.get("metric", "accuracy"), "direction": diag.get("direction", "higher"),
            "items": diag["items"], "n": int(diag.get("n", 100)), "seed": 0,
            "stderr": {"method": "bernoulli"},
        })
        on_request.append("diag")
    fractions = config.get("standing_fractions", [0.25, 0.5, 0.75, 1.0])
    regress = float(config.get("regress_threshold", 0.03))
    return {
        "schema_version": CONTRACT_SCHEMA,
        "evaluators": evaluators,
        "standing_yields": {
            "progress": dict(card["setup"]["progress"]),
            "cadence": {"kind": "fractions", "values": list(fractions)},
            "evaluators": [dev_name, "watch"],
        },
        "on_request": on_request,
        "rule_schema": RULE_SCHEMA,
        "rules": [
            {"id": "regress", "op": "consecutive_threshold", "evaluator": dev_name,
             "field": "delta_vs_parent", "comparator": "lt" if direction == "higher" else "gt",
             "threshold": -regress if direction == "higher" else regress, "count": 2, "action": "abort"},
        ],
        "selection": {"evaluator": dev_name, "direction": direction, "completion_policy": "best_observation"},
    }


def default_brief_summary(card: dict[str, Any], grounding: list[dict[str, Any]]) -> str:
    failed = [g["check"] for g in grounding if not g["passed"]]
    n_ok = sum(1 for g in grounding if g["passed"])
    head = f"{card['card_id']}: {n_ok}/{len(grounding)} grounding checks passed"
    if failed:
        head += f" (advisory: {', '.join(failed)})"
    return head + ". Contract proposed from the card's evaluation section."


def observation_summary(obs: dict[str, Any], contract: dict[str, Any]) -> str:
    sel = contract["selection"]["evaluator"]
    parts = []
    for name, m in obs["evaluators"].items():
        d = m.get("delta_vs_parent")
        se = m.get("stderr")
        s = f"{name}={m['value']:.3f}"
        if d is not None:
            s += f" ({d:+.3f} vs parent"
            if se:
                s += f", ±{se:.3f}"
            s += ")"
        parts.append(s)
    w = obs.get("watch")
    if w:
        parts.append(f"watch fixed {w['fixed']} / still {w['still_failing']} / regressed {w['regressions']}")
    lead = f"{obs['obs_id']} at step {obs['checkpoint'].get('step')}"
    frac = obs.get("fraction")
    if frac is not None:
        lead += f" ({frac:.0%})"
    return f"{lead}: " + "; ".join(parts) + f". Selection metric: {sel}."


def observation_evidence(obs: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for name, m in obs["evaluators"].items():
        out.append({"path": m.get("raw"), "locator": name,
                    "observation": f"{m['metric']}={m['value']:.4f} n={m['n']}"})
    return out
