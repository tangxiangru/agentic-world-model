"""The world-model agent's toolbelt.

Two Claude Code sessions share a sandbox: a research scientist and a world-model
agent (WMA). The WMA lives in ``wma/`` (its CLAUDE.md and skills); this package
is what it calls from the shell: draft a card from the scientist's words, search
past experiments, produce the default evaluation plan, parse the scientist's
eval outputs, and log every consult. See doc/spec/2026-08-30-world-model-agent.md.
"""
