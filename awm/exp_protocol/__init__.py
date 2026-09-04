"""The experiment protocol: how a scientist does an experiment correctly and leaves a record.

Deterministic discipline only — schema, checks, hashes, an index, and an optional
foreground runner for the scientist's already-declared locked command. Nothing
here chooses a research direction, predicts an outcome, or waits for an LLM. See
doc/reference/exp_protocol_and_wma_policy.md for the boundary.
"""
