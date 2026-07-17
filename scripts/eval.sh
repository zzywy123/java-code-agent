#!/usr/bin/env sh
set -eu

FIXTURE="${1:-./demo-repo}"
OUTPUT="${2:-./reports}"
RUNS="${EVAL_RUNS:-1}"

exec python -m agent.eval.runner \
  --fixture "$FIXTURE" \
  --output "$OUTPUT" \
  --runs "$RUNS"
