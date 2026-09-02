"""The world-model agent, as an estimator.

Given a card's pre-launch sections it returns a four-level verdict — runs /
valid candidate / effect vs the incumbent / worth it now — with evidence,
the probes it ran, and suggestions that hang off the verdict. It estimates;
the scientist decides. Every verdict is reconciled against the outcome; the
set of them is the ledger the agent learns from. See
doc/spec/2026-09-01-wma-v1-design.md.
"""
