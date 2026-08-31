"""The retrieval arm: null + precedents from memory, at the brief and after each observation.

It adds facts about what happened before — nearest closed cards, their
best-vs-parent deltas, and how their curves looked at the same fraction. It
does not write prose beyond that, does not predict, and does not request
evaluations.
"""

from __future__ import annotations

from .base import Advice, Brief, WorldModelAgent, observation_evidence, observation_summary


class RetrievalAgent(WorldModelAgent):
    arm = "retrieval"

    def on_proposal(self, card, grounding, memory, config) -> Brief:
        brief = super().on_proposal(card, grounding, memory, config)
        brief.precedents = memory.precedents(card, k=int(config.get("retrieval_k", 5)))  # recorded in config.yaml
        if brief.precedents:
            deltas = [p["delta_best_vs_parent"] for p in brief.precedents if p.get("delta_best_vs_parent") is not None]
            tail = ""
            if deltas:
                tail = (f"; best-vs-parent deltas {', '.join(f'{d:+.3f}' for d in deltas)}"
                        f" (median {sorted(deltas)[len(deltas) // 2]:+.3f})")
            brief.summary += f" {len(brief.precedents)} precedents in memory{tail}."
        else:
            brief.summary += " No precedents in memory."
        return brief

    def on_observation(self, observation, history, contract, card, memory, config) -> Advice:
        advice = Advice(kind="notice", summary=observation_summary(observation, contract),
                        evidence=observation_evidence(observation))
        precedents = memory.precedents(card, k=int(config.get("retrieval_k", 5)))
        if not precedents:
            return advice
        keys = [(p["session"], p["card_id"]) for p in precedents]
        curves = memory.curves(keys)
        sel = contract["selection"]["evaluator"]
        frac = observation.get("fraction")
        comparable: list[str] = []
        for key, rows in curves.items():
            if frac is None:
                continue
            near = [r for r in rows if r.get("fraction") is not None and abs(r["fraction"] - frac) <= 0.13]
            for r in near[:1]:
                d = (r.get("evaluators") or {}).get(sel, {}).get("delta_vs_parent")
                if d is not None:
                    comparable.append(f"{key}@{r['fraction']:.0%} {d:+.3f}")
        advice.precedents = precedents
        if comparable:
            advice.summary += " Precedent curves at this point: " + ", ".join(comparable[:5]) + "."
        return advice
