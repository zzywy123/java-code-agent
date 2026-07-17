#!/usr/bin/env sh
set -eu

REPO_ROOT="${1:-./demo-repo}"
export AGENT_REPO_ROOT="$REPO_ROOT"

echo "Starting Java Coding Agent for: $AGENT_REPO_ROOT"
echo "Try: explain OrderService.calculateTotal and cite the source lines."
exec coding-agent
