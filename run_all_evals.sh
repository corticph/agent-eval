#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Must match the environments offered by src/agent_evals/environment.py;
# the CLI re-validates, so drift fails loudly rather than silently.
SUPPORTED_ENVIRONMENTS=(local eu us)
SUPPORTED_ENVS="${SUPPORTED_ENVIRONMENTS[*]}"

usage() {
  echo "Usage: $0 --env <environment> [extra agent-evals args...]"
  echo ""
  echo "Runs every suite under evals/ against the given environment."
  echo ""
  echo "Arguments:"
  echo "  --env        Environment to run against (required, no default)."
  echo "               Supported: ${SUPPORTED_ENVS}"
  echo ""
  echo "Example:"
  echo "  $0 --env eu -v"
  exit 1
}

ENVIRONMENT=""

# Parse environment parameter
while [[ $# -gt 0 ]]; do
  case $1 in
    --env)
      if [[ $# -lt 2 ]]; then
        echo "Error: --env requires a value"
        usage
      fi
      ENVIRONMENT="$2"
      shift 2
      ;;
    --env=*)
      ENVIRONMENT="${1#*=}"
      shift
      ;;
    *)
      break
      ;;
  esac
done

# Deliberately no default environment: a sweep must name its target,
# matching the CLI's required --env rule.
if [[ -z "${ENVIRONMENT}" ]]; then
  echo "Error: --env is required"
  usage
fi

# Validate environment
env_is_supported=0
for supported in "${SUPPORTED_ENVIRONMENTS[@]}"; do
  if [[ "${ENVIRONMENT}" == "${supported}" ]]; then
    env_is_supported=1
    break
  fi
done
if [[ "${env_is_supported}" -eq 0 ]]; then
  echo "Error: Invalid environment '${ENVIRONMENT}'. Supported: ${SUPPORTED_ENVS}"
  exit 1
fi

# Load environment variables from .env file
if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  source "${ROOT_DIR}/.env"
  set +a
else
  echo "Warning: .env file not found at ${ROOT_DIR}/.env"
fi

echo "Running evals for environment: ${ENVIRONMENT}"

# Discover all suite YAML files recursively, excluding *_local.yaml variants.
# (Portable read loop instead of `mapfile`, which macOS's bash 3.2 lacks.)
all_suites=()
while IFS= read -r suite_line; do
  all_suites+=("${suite_line}")
done < <(find "${ROOT_DIR}/evals" -name '*.yaml' ! -name '_*.yaml' ! -name '*_local.yaml' -size +0c | sort)
extra_args=()
if (($#)); then
  extra_args=("$@")
fi
failed=0

for suite_path in "${all_suites[@]}"; do
  rel_path="${suite_path#"${ROOT_DIR}"/evals/}"
  echo "[agent-evals] Running ${rel_path}"
  # The sweep is the recorded path: --opik is baked in so every real run lands
  # in Opik without anyone remembering the flag. The single-eval dev loop
  # (run_eval.sh) stays bare for offline iteration.
  cmd=(uv run agent-evals run "${suite_path}" --env "${ENVIRONMENT}" --opik)
  if ((${#extra_args[@]})); then
    cmd+=("${extra_args[@]}")
  fi
  if ! "${cmd[@]}"; then
    failed=1
  fi
done

exit "${failed}"
