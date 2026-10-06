#!/usr/bin/env bash
# Thin wrapper around `agent-evals run` for muscle memory.
# Usage: run_eval.sh <suite_path> --env <environment> [extra args...]
set -euo pipefail
exec uv run agent-evals run "$@"
