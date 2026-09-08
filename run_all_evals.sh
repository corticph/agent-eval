#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  echo "Usage: $0 --env <environment> [--evals-dir <path>] [--suite <pattern>] [--tag <tag>] [-j <jobs>] [--resume | --resume-file <path>] [extra agent-evals args...]"
  echo ""
  echo "Runs every suite under evals/ against the given environment."
  echo ""
  echo "Arguments:"
  echo "  --env        Environment to run against (required, no default)."
  echo "               The CLI validates which environments are reachable"
  echo "               based on your .env file."
  echo "  --evals-dir  Directory containing suite YAML files (default: ./evals)."
  echo "               If not found, falls back to ../agent-eval-cases/evals."
  echo "  --suite      Substring filter for suite paths (repeatable)."
  echo "               e.g. --suite smoke runs only smoke/* suites."
  echo "  --tag        Tag for all Opik experiments in this sweep (repeatable)."
  echo "               Useful for grouping runs in compare_experiments."
  echo "               Forwarded to agent-evals as --tag."
  echo "  -j, --jobs   Max concurrent suites (default: 10). Use 1 for sequential."
  echo "  --retries    Retry failed suites N times (default: 2). The kubectl"
  echo "               tunnel drops connections under load but recovers fast."
  echo "               Set to 0 to disable retries."
  echo "  --resume     Re-run only suites missing from Opik for the first --tag."
  echo "               Queries Opik for experiments with that tag, compares"
  echo "               against the full suite list, and runs the ones that"
  echo "               don't have an experiment yet (i.e. failed before the"
  echo "               experiment was created). Requires --tag."
  echo "  --resume-file <path>"
  echo "               Resume from an explicit failed-suites file instead of"
  echo "               querying Opik. One suite path per line."
  echo "               Lines starting with # are ignored."
  echo ""
  echo "Example:"
  echo "  $0 --env eu -v"
  echo "  $0 --env dev-weu --suite smoke"
  echo "  $0 --env local --suite smoke --tag release-v1.2"
  echo "  $0 --env eu --suite smoke -j 8"
  echo "  $0 --env eu --tag eu-20260907-120000 --resume"
  exit 1
}

ENVIRONMENT=""
EVALS_DIR_ARG=""
SUITE_FILTERS=()
TAGS=()
JOBS=10
RETRIES=2
RESUME=false
RESUME_FILE=""

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
    --suite)
      if [[ $# -lt 2 ]]; then
        echo "Error: --suite requires a value"
        usage
      fi
      SUITE_FILTERS+=("$2")
      shift 2
      ;;
    --suite=*)
      SUITE_FILTERS+=("${1#*=}")
      shift
      ;;
    --tag)
      if [[ $# -lt 2 ]]; then
        echo "Error: --tag requires a value"
        usage
      fi
      TAGS+=("$2")
      shift 2
      ;;
    --tag=*)
      TAGS+=("${1#*=}")
      shift
      ;;
    -j|--jobs)
      if [[ $# -lt 2 ]]; then
        echo "Error: --jobs requires a value"
        usage
      fi
      JOBS="$2"
      shift 2
      ;;
    --jobs=*)
      JOBS="${1#*=}"
      shift
      ;;
    --retries)
      if [[ $# -lt 2 ]]; then
        echo "Error: --retries requires a value"
        usage
      fi
      RETRIES="$2"
      shift 2
      ;;
    --retries=*)
      RETRIES="${1#*=}"
      shift
      ;;
    --resume)
      RESUME=true
      shift
      ;;
    --resume-file)
      if [[ $# -lt 2 ]]; then
        echo "Error: --resume-file requires a value"
        usage
      fi
      RESUME=true
      RESUME_FILE="$2"
      shift 2
      ;;
    --resume-file=*)
      RESUME=true
      RESUME_FILE="${1#*=}"
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

# Load environment variables from .env file
if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  source "${ROOT_DIR}/.env"
  set +a
else
  echo "Warning: .env file not found at ${ROOT_DIR}/.env"
fi

# Validate environment via the CLI's own dynamic discovery — it knows
# which environments are reachable from the current .env, so there's no
# hardcoded list to drift.
SUPPORTED_ENVS="$(uv run python -c 'from agent_evals.environment import supported_environments; print(" ".join(supported_environments()))')"
if ! echo " ${SUPPORTED_ENVS} " | grep -q -- " ${ENVIRONMENT} "; then
  echo "Error: Invalid environment '${ENVIRONMENT}'. Supported: ${SUPPORTED_ENVS// /, }"
  exit 1
fi

echo "Running evals for environment: ${ENVIRONMENT}"

# --- locate the evals directory (always needed) --------------------------------
# Explicit --evals-dir wins, then a local evals/, then a sibling checkout.
# The evals directory is needed both for discovery (normal mode) and for
# the Opik resume query (which maps suite names back to file paths).

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
  echo "       $0 --env local --evals-dir /path/to/evals"
  echo ""
  exit 1
fi

# --- build the suite list (resume or discover) --------------------------------

if [[ "${RESUME}" == true ]]; then
  if [[ -n "${RESUME_FILE}" ]]; then
    # Explicit file-based resume: read suite paths from a file.
    if [[ ! -f "${RESUME_FILE}" ]]; then
      echo "Nothing to resume: ${RESUME_FILE} does not exist."
      exit 0
    fi
    all_suites=()
    while IFS= read -r line; do
      line="${line#"${line%%[![:space:]]*}"}"
      line="${line%"${line##*[![:space:]]}"}"
      [[ -z "${line}" ]] && continue
      [[ "${line}" == \#* ]] && continue
      if [[ ! -f "${line}" ]]; then
        echo "Warning: suite file '${line}' from ${RESUME_FILE} no longer exists; skipping."
        continue
      fi
      all_suites+=("${line}")
    done < "${RESUME_FILE}"
    if [[ ${#all_suites[@]} -eq 0 ]]; then
      echo "Nothing to resume: ${RESUME_FILE} contains no suite paths."
      exit 0
    fi
    echo "Resuming ${#all_suites[@]} suite(s) from ${RESUME_FILE}"
  else
    # Opik-based resume: query Opik for experiments with the first --tag,
    # compare against the full suite list, and run the ones missing from Opik.
    if ((${#TAGS[@]} == 0)); then
      echo "Error: --resume requires --tag (to find the Opik experiments for this sweep)."
      exit 1
    fi
    RESUME_TAG="${TAGS[0]}"
    echo "Querying Opik for suites missing from tag '${RESUME_TAG}'..."
    suite_filter_args=()
    for pattern in "${SUITE_FILTERS[@]+"${SUITE_FILTERS[@]}"}"; do
      suite_filter_args+=(--suite "$pattern")
    done
    missing_output="$(uv run python -m agent_evals.scripts.resume_missing \
      --tag "${RESUME_TAG}" \
      --env "${ENVIRONMENT}" \
      --evals-dir "${EVALS_DIR}" \
      "${suite_filter_args[@]+"${suite_filter_args[@]}"}" 2>&1)" || true
    # The script prints missing suite paths to stdout and status messages to stderr.
    # Lines that are valid file paths are suites to run; everything else is a message.
    all_suites=()
    while IFS= read -r line; do
      [[ -z "${line}" ]] && continue
      if [[ -f "${line}" ]]; then
        all_suites+=("${line}")
      else
        echo "${line}"
      fi
    done <<< "${missing_output}"
    if [[ ${#all_suites[@]} -eq 0 ]]; then
      echo "Nothing to resume: all suites have experiments in Opik for tag '${RESUME_TAG}'."
      exit 0
    fi
    echo "Resuming ${#all_suites[@]} missing suite(s) from Opik tag '${RESUME_TAG}'"
  fi
else
  # Normal discovery: find all suite YAML files under the evals directory.
  SEARCH_DIRS=("${EVALS_DIR}")
  all_suites=()
  while IFS= read -r suite_line; do
    all_suites+=("${suite_line}")
  done < <(find "${SEARCH_DIRS[@]}" -name '*.yaml' ! -name '_*.yaml' ! -name '*_local.yaml' -size +0c | sort)

  # Apply --suite substring filters if any were given.
  if ((${#SUITE_FILTERS[@]})); then
    filtered_suites=()
    for suite_path in "${all_suites[@]}"; do
      for pattern in "${SUITE_FILTERS[@]}"; do
        if [[ "${suite_path}" == *"${pattern}"* ]]; then
          filtered_suites+=("${suite_path}")
          break
        fi
      done
    done
    all_suites=(${filtered_suites[@]+"${filtered_suites[@]}"})
  fi

  if [[ ${#all_suites[@]} -eq 0 ]]; then
    echo "Error: No suite files matched the --suite filter(s): ${SUITE_FILTERS[*]}"
    exit 1
  fi
fi

extra_args=()
for tag in "${TAGS[@]+"${TAGS[@]}"}"; do
  extra_args+=(--tag "$tag")
done
if (($#)); then
  extra_args+=("$@")
fi
failed=0

# --- pre-warm the Opik tunnel -------------------------------------------------
# When --opik is baked in (as this sweep does), every suite calls
# resolve_opik_url() which may start a kubectl port-forward.  In parallel mode
# several suites race to bind the same port and fail.  Start the tunnel once
# here so the per-suite ping sees it already up.

if [[ -z "${OPIK_URL_OVERRIDE:-}" ]]; then
  TUNNEL_PORT="${OPIK_TUNNEL_PORT:-18080}"
  if ! curl -sf "http://localhost:${TUNNEL_PORT}/is-alive/ping" >/dev/null 2>&1; then
    TUNNEL_CONTEXT="${OPIK_TUNNEL_CONTEXT:-dev-weu}"
    echo "Starting Opik tunnel (kubectl port-forward to ${TUNNEL_CONTEXT})..."
    kubectl --context "${TUNNEL_CONTEXT}" port-forward -n mlservices svc/opik-backend \
      "${TUNNEL_PORT}:8080" >/dev/null 2>&1 &
    TUNNEL_PID=$!
    # Wait for readiness (up to 15s).
    for _ in $(seq 1 50); do
      curl -sf "http://localhost:${TUNNEL_PORT}/is-alive/ping" >/dev/null 2>&1 && break
      sleep 0.3
    done
    if curl -sf "http://localhost:${TUNNEL_PORT}/is-alive/ping" >/dev/null 2>&1; then
      echo "Opik tunnel ready on localhost:${TUNNEL_PORT}"
    else
      echo "Warning: Opik tunnel did not become ready; suites will retry individually."
      kill "${TUNNEL_PID}" 2>/dev/null || true
      TUNNEL_PID=""
    fi
  fi
fi

# --- run suites (parallel when JOBS > 1) -------------------------------------
#
# Each suite is independent (its own agent + dataset), so they can run
# concurrently. Output is buffered per-suite to a temp file and printed
# with a header when the job finishes, so parallel output never interleaves.
# Failed suites are retried (the kubectl tunnel drops connections under load
# but recovers fast).

run_one() {
  local suite_path="$1" outfile="$2"
  local rel_path="${suite_path##*/}"
  local cmd=(uv run agent-evals run "${suite_path}" --env "${ENVIRONMENT}" --opik)
  if ((${#extra_args[@]})); then
    cmd+=("${extra_args[@]}")
  fi
  local attempt=0
  local max=$((RETRIES + 1))
  local rc=1
  while ((attempt < max)); do
    attempt=$((attempt + 1))
    {
      if ((max > 1)); then
        echo "[agent-evals] Running ${rel_path} attempt ${attempt}/${max}"
      else
        echo "[agent-evals] Running ${rel_path}"
      fi
      "${cmd[@]}"
    } > "$outfile" 2>&1
    rc=$?
    ((rc == 0)) && break
    ((attempt < max)) && sleep 2
  done
  return "$rc"
}

if ((JOBS <= 1)); then
  # Sequential: output in real time.
  for suite_path in "${all_suites[@]}"; do
    rel_path="${suite_path#"${EVALS_DIR}"/}"
    tmpfile="$(mktemp)"
    if run_one "$suite_path" "$tmpfile"; then rc=0; else rc=$?; fi
    cat "$tmpfile"
    rm -f "$tmpfile"
    ((rc != 0)) && failed=1
  done
else
  # Parallel: launch up to JOBS suites, buffer output per-suite.
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "$tmpdir"' EXIT

  # running entries: "pid|rel_path|outfile"
  running=()
  total=${#all_suites[@]}
  done_count=0
  job_seq=0

  for suite_path in "${all_suites[@]}"; do
    rel_path="${suite_path#"${EVALS_DIR}"/}"

    # Wait for a free slot.
    while ((${#running[@]} >= JOBS)); do
      sleep 0.3
      still_running=()
      for entry in "${running[@]}"; do
        pid="${entry%%|*}"
        rest="${entry#*|}"
        rpath="${rest%%|*}"
        outfile="${rest#*|}"
        if kill -0 "$pid" 2>/dev/null; then
          still_running+=("$entry")
        else
          wait "$pid" && rc=0 || rc=$?
          done_count=$((done_count + 1))
          echo "-- ${rpath} ${done_count}/${total} exit ${rc} --"
          cat "$outfile"
          rm -f "$outfile"
          if ((rc != 0)); then failed=1; fi
        fi
      done
      if ((${#still_running[@]})); then
        running=("${still_running[@]}")
      else
        running=()
      fi
    done

    job_seq=$((job_seq + 1))
    outfile="${tmpdir}/job_${job_seq}.out"
    ( run_one "$suite_path" "$outfile" ) &
    pid=$!
    running+=("${pid}|${rel_path}|${outfile}")
  done

  # Drain remaining jobs.
  for entry in "${running[@]+"${running[@]}"}"; do
    pid="${entry%%|*}"
    rest="${entry#*|}"
    rpath="${rest%%|*}"
    outfile="${rest#*|}"
    wait "$pid" && rc=0 || rc=$?
    done_count=$((done_count + 1))
    echo "-- ${rpath} ${done_count}/${total} exit ${rc} --"
    cat "$outfile"
    rm -f "$outfile"
    if ((rc != 0)); then failed=1; fi
  done
fi

echo ""
echo "============================================================"
echo "SUMMARY  env=${ENVIRONMENT} jobs=${JOBS} suites=${#all_suites[@]}"
if ((failed)); then
  echo "  some suites FAILED"
  echo "  resume with: $0 --env ${ENVIRONMENT} --tag <tag> --resume"
else
  echo "  all suites passed"
fi
echo "============================================================"

# Clean up the tunnel we started (if any).
# Only kill it if no other run_all_evals.sh processes are still running —
# when two sweeps run in parallel they share the tunnel, and the first to
# finish must not kill it out from under the second.
if [[ -n "${TUNNEL_PID:-}" ]]; then
  other_count=$(pgrep -f 'run_all_evals\.sh' | grep -cv "$$" || true)
  if [[ "${other_count}" -eq 0 ]]; then
    kill "${TUNNEL_PID}" 2>/dev/null || true
  else
    echo "Leaving Opik tunnel running for other sweep, pid ${TUNNEL_PID}."
  fi
fi

exit "${failed}"
