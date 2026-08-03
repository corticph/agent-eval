#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Must match the environments offered by src/agent_evals/environment.py;
# the CLI re-validates, so drift fails loudly rather than silently.
SUPPORTED_ENVIRONMENTS=(local eu us)
SUPPORTED_ENVS="${SUPPORTED_ENVIRONMENTS[*]}"

usage() {
  echo "Usage: $0 <eval_path> --env <environment>"
  echo ""
  echo "Arguments:"
  echo "  eval_path    Relative path under evals/ (e.g., smoke/hello)"
  echo "  --env        Environment to run against. Supported: ${SUPPORTED_ENVS}"
  echo ""
  echo "Example:"
  echo "  $0 smoke/hello --env eu"
  exit 1
}

# Check minimum arguments (eval_path plus an --env form)
if [[ $# -lt 2 ]]; then
  usage
fi

EVAL_NAME="$1"
shift

ENV=""

# Parse named arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
  --env)
    if [[ $# -lt 2 ]]; then
      echo "Error: --env requires a value"
      usage
    fi
    ENV="$2"
    shift 2
    ;;
  --env=*)
    ENV="${1#*=}"
    shift
    ;;
  *)
    echo "Error: Unknown argument: $1"
    usage
    ;;
  esac
done

# Validate env argument
if [[ -z "$ENV" ]]; then
  echo "Error: --env is required"
  usage
fi

env_is_supported=0
for supported in "${SUPPORTED_ENVIRONMENTS[@]}"; do
  if [[ "$ENV" == "$supported" ]]; then
    env_is_supported=1
    break
  fi
done
if [[ "$env_is_supported" -eq 0 ]]; then
  echo "Error: Invalid environment '$ENV'. Supported: ${SUPPORTED_ENVS}"
  exit 1
fi

# Construct paths
EVAL_PATH="${ROOT_DIR}/evals/${EVAL_NAME}.yaml"

# Validate eval file exists
if [[ ! -f "$EVAL_PATH" ]]; then
  echo "Error: Eval file not found: $EVAL_PATH"
  exit 1
fi

echo "[agent-evals] Running ${EVAL_NAME}"
echo "[agent-evals] Environment: ${ENV}"

uv run agent-evals run "${EVAL_PATH}" --env "${ENV}"
