"""The part that varies: one policy per arm behind one interface."""

from __future__ import annotations

from .base import Brief, WorldModelAgent
from .null import NullAgent
from .retrieval import RetrievalAgent

ARMS = {
    "null": NullAgent,
    "retrieval": RetrievalAgent,
}


def make_agent(arm: str, **kwargs) -> WorldModelAgent:
    if arm in ARMS:
        return ARMS[arm](**kwargs)
    if arm in ("llm", "traj"):
        from .llm import LLMAgent, TrajAgent

        return (TrajAgent if arm == "traj" else LLMAgent)(**kwargs)
    if arm == "predictor":
        from .predictor import PredictorAgent

        return PredictorAgent(**kwargs)
    raise ValueError(f"unknown arm {arm!r}; choose from null, retrieval, llm, traj, predictor")


__all__ = ["ARMS", "Brief", "NullAgent", "RetrievalAgent", "WorldModelAgent", "make_agent"]
