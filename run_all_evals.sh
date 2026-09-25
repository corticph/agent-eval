#!/usr/bin/env bash
# Thin wrapper around `agent-evals sweep` for muscle memory.
# All flags are forwarded directly.
set -euo pipefail
exec uv run agent-evals sweep "$@"
