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
  echo "  -j, --jobs   Max concurrent suites (default: 4). Use 1 for sequential."
  echo "  --retries    Retry failed suites N times (default: 2). The kubectl"
  echo "               tunnel drops connections under load but recovers fast."
  echo "               Set to 0 to disable retries."
  echo "  --resume     Re-run only suites that failed in the last sweep for this"
  echo "               environment. Reads runs/failed_<env>.txt (written"
  echo "               automatically after every sweep). When the file is empty"
  echo "               or missing, prints 'nothing to resume' and exits 0."
  echo "  --resume-file <path>"
  echo "               Resume from an explicit failed-suites file instead of"
  echo "               the default runs/failed_<env>.txt. One suite path per line."
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
JOBS=3
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

# --- state directory for resume -------------------------------------------------
# runs/ holds failed-suite lists so the next invocation can --resume only the
# suites that failed.  It's gitignored; one file per environment.
RUNS_DIR="${ROOT_DIR}/runs"
FAILED_FILE="${RUNS_DIR}/failed_${ENVIRONMENT}.txt"
if [[ -n "${RESUME_FILE}" ]]; then
  FAILED_FILE="${RESUME_FILE}"
fi
mkdir -p "${RUNS_DIR}"

# --- build the suite list (resume or discover) --------------------------------
#
# With --resume, the list comes from a previously-written failed-suites file
# (one path per line, # comments ignored).  Without --resume, suites are
# discovered recursively under the evals directory and optionally filtered
# by --suite.

if [[ "${RESUME}" == true ]]; then
  if [[ ! -f "${FAILED_FILE}" ]]; then
    echo "Nothing to resume: ${FAILED_FILE} does not exist (all suites passed in the last sweep, or no sweep has been run yet)."
    exit 0
  fi

  # Read non-comment, non-empty lines into all_suites.
  all_suites=()
  while IFS= read -r line; do
    # Strip leading/trailing whitespace.
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "${line}" ]] && continue
    [[ "${line}" == \#* ]] && continue
    if [[ ! -f "${line}" ]]; then
      echo "Warning: suite file '${line}' from ${FAILED_FILE} no longer exists; skipping."
      continue
    fi
    all_suites+=("${line}")
  done < "${FAILED_FILE}"

  if [[ ${#all_suites[@]} -eq 0 ]]; then
    echo "Nothing to resume: ${FAILED_FILE} contains no suite paths (all suites passed in the last sweep)."
    # Clean up the empty file so a future --resume without a new run also exits cleanly.
    rm -f "${FAILED_FILE}"
    exit 0
  fi

  echo "Resuming ${#all_suites[@]} failed suite(s) from ${FAILED_FILE}"
else
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
    echo "       $0 --env local --evals-dir /path/to/evals"
    echo ""
    exit 1
  fi

  SEARCH_DIRS=("${EVALS_DIR}")

  # Discover all suite YAML files recursively, excluding *_local.yaml variants.
  # (Portable read loop instead of `mapfile`, which macOS's bash 3.2 lacks.)
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
FAILED_SUITES=()

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
    rel_path="${suite_path#"${EVALS_DIR:-}"/}"
    tmpfile="$(mktemp)"
    if run_one "$suite_path" "$tmpfile"; then rc=0; else rc=$?; fi
    cat "$tmpfile"
    rm -f "$tmpfile"
    if ((rc != 0)); then
      failed=1
      FAILED_SUITES+=("$suite_path")
    fi
  done
else
  # Parallel: launch up to JOBS suites, buffer output per-suite.
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "$tmpdir"' EXIT

  # running entries: "pid|suite_path|rel_path|outfile"
  running=()
  total=${#all_suites[@]}
  done_count=0
  job_seq=0

  for suite_path in "${all_suites[@]}"; do
    rel_path="${suite_path#"${EVALS_DIR:-}"/}"

    # Wait for a free slot.
    while ((${#running[@]} >= JOBS)); do
      sleep 0.3
      still_running=()
      for entry in "${running[@]}"; do
        pid="${entry%%|*}"
        rest="${entry#*|}"
        full_path="${rest%%|*}"
        rest="${rest#*|}"
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
          if ((rc != 0)); then
            failed=1
            FAILED_SUITES+=("$full_path")
          fi
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
    running+=("${pid}|${suite_path}|${rel_path}|${outfile}")
  done

  # Drain remaining jobs.
  for entry in "${running[@]+"${running[@]}"}"; do
    pid="${entry%%|*}"
    rest="${entry#*|}"
    full_path="${rest%%|*}"
    rest="${rest#*|}"
    rpath="${rest%%|*}"
    outfile="${rest#*|}"
    wait "$pid" && rc=0 || rc=$?
    done_count=$((done_count + 1))
    echo "-- ${rpath} ${done_count}/${total} exit ${rc} --"
    cat "$outfile"
    rm -f "$outfile"
    if ((rc != 0)); then
      failed=1
      FAILED_SUITES+=("$full_path")
    fi
  done
fi

echo ""
echo "============================================================"
echo "SUMMARY  env=${ENVIRONMENT} jobs=${JOBS} suites=${#all_suites[@]}"
if ((failed)); then
  echo "  some suites FAILED"
else
  echo "  all suites passed"
fi
echo "============================================================"

# --- write failed-suites file for resume ---------------------------------------
# Always (re)write the file: a non-empty list lets the next --resume pick up
# the remaining failures; an empty file signals "all passed" so a subsequent
# --resume exits cleanly.
if (( ${#FAILED_SUITES[@]} )); then
  printf '%s\n' "${FAILED_SUITES[@]}" > "${FAILED_FILE}"
  echo "Failed suites written to ${FAILED_FILE}"
  echo "Resume with: $0 --env ${ENVIRONMENT} --resume"
else
  : > "${FAILED_FILE}"
fi

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
