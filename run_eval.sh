#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Must match the environments offered by src/agent_evals/environment.py.
# The CLI validates --env dynamically (based on which env vars resolve),
# so we don't hardcode a list here — any drift would make the sweep reject
# a target the CLI accepts, or vice versa.

usage() {
  echo "Usage: $0 <eval_path> --env <environment> [--evals-dir <path>]"
  echo ""
  echo "Arguments:"
  echo "  eval_path    Relative path under evals/ (e.g., smoke/hello)"
  echo "  --env        Environment to run against."
  echo "               The CLI validates which environments are reachable"
  echo "               based on your .env file."
  echo "  --evals-dir  Directory containing suite YAML files (default: ./evals)."
  echo "               If not found, falls back to ../agent-eval-cases/evals."
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
EVALS_DIR_ARG=""

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
    --evals-dir)
      if [[ $# -lt 2 ]]; then
        echo "Error: --evals-dir requires a value"
        usage
      fi
      EVALS_DIR_ARG="$2"
      shift 2
      ;;
    --evals-dir=*)
      EVALS_DIR_ARG="${1#*=}"
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

# Load environment variables from .env file
if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  source "${ROOT_DIR}/.env"
  set +a
else
  echo "Warning: .env file not found at ${ROOT_DIR}/.env"
fi

# Validate environment via the CLI's own dynamic discovery
SUPPORTED_ENVS="$(uv run python -c 'from agent_evals.environment import supported_environments; print(" ".join(supported_environments()))')"
if ! echo " ${SUPPORTED_ENVS} " | grep -q -- " ${ENV} "; then
  echo "Error: Invalid environment '${ENV}'. Supported: ${SUPPORTED_ENVS// /, }"
  exit 1
fi

# Locate the evals directory: explicit --evals-dir wins, then a local evals/,
# then a sibling agent-eval-cases/evals/ checkout.
if [[ -n "${EVALS_DIR_ARG}" ]]; then
  if [[ ! -d "${EVALS_DIR_ARG}" ]]; then
    echo "Error: --evals-dir '${EVALS_DIR_ARG}' does not exist or is not a directory."
    exit 1
  fi
  EVALS_DIR="${EVALS_DIR_ARG}"
elif [[ -d "${ROOT_DIR}/evals" ]]; then
  EVALS_DIR="${ROOT_DIR}/evals"
elif [[ -d "${ROOT_DIR}/../agent-eval-cases/evals" ]]; then
  EVALS_DIR="${ROOT_DIR}/../agent-eval-cases/evals"
else
  echo "Error: No evals/ directory found."
  echo ""
  echo "This repo is the eval harness only — the suite YAML, expectations, and"
  echo "fixtures live in a separate cases repo.  To get started:"
  echo ""
  echo "  1. Clone the cases repo as a sibling:"
  echo "       git clone <cases-repo-url> ../agent-eval-cases"
  echo ""
  echo "  2. Or symlink its evals/ into this repo:"
  echo "       ln -s /path/to/agent-eval-cases/evals evals"
  echo ""
  echo "  3. Or point --evals-dir at any directory of suite YAML files:"
  echo "       $0 smoke/hello --env local --evals-dir /path/to/evals"
  echo ""
  exit 1
fi

# Construct paths
EVAL_PATH="${EVALS_DIR}/${EVAL_NAME}.yaml"

# Validate eval file exists
if [[ ! -f "$EVAL_PATH" ]]; then
  echo "Error: Eval file not found: $EVAL_PATH"
  exit 1
fi

echo "[agent-evals] Running ${EVAL_NAME}"
echo "[agent-evals] Environment: ${ENV}"

uv run agent-evals run "${EVAL_PATH}" --env "${ENV}"
