"""Card builders shared by the exp_protocol tests. Not a test module (no ``test_`` prefix)."""

from __future__ import annotations

from awm.exp_protocol import schema


def plan_card() -> dict:
    """A card with every required pre-launch field and nothing else."""
    return {
        "schema_version": schema.CARD_SCHEMA,
        "card_id": "exp-01",
        "created_at": "2026-09-01T00:00:00Z",
        "situation": {"elapsed_h": 1.0, "trigger": "base model gets 0.33 on dev-150"},
        "problem": {"statement": "arithmetic slips inside otherwise correct chains"},
        "hypothesis": {"claim": "SFT on filtered self-samples cuts arithmetic slips"},
        "setup": {
            "parent_checkpoint": {"path": "google/gemma-3-4b-pt", "origin": "base_model"},
            "data": [{"path": "/t/data/train.jsonl", "source": "synthetic:self", "n_examples": 120}],
            "method": {"family": "sft"},
            "command": {"argv": ["python", "train.py"], "cwd": "/t"},
            "output_dir": "/t/ckpts/exp-01",
            "checkpoints": {"keep": "last"},
        },
        "evaluation": {"protocol": {"n": 150}},
    }


def closed_card() -> dict:
    card = plan_card()
    card["result"] = {
        "execution": "completed",
        "output_checkpoint": "/t/ckpts/exp-01/final",
        "measurements": [{"metric": "accuracy", "value": 0.41, "n": 150, "path": "/t/eval/exp-01.json"}],
    }
    card["conclusion"] = {
        "verdict": "supported", "mechanism_verdict": "not_tested",
        "summary": "up 8 points on dev-150", "decision": "adopt",
    }
    return card
