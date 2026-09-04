#!/usr/bin/env bash
# No reference solution is registered for R2AbsMolecularPropertyPredictionQm9MeanAbsoluteError yet.
# `harbor run -a oracle` on this task will fail on purpose rather than score a
# fabricated submission: see REFERENCE_SOLUTIONS in awm/adapters/airs.py.
set -euo pipefail
echo "no reference solution for R2AbsMolecularPropertyPredictionQm9MeanAbsoluteError" >&2
exit 1
