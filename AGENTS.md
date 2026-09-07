# Agent Guide: Running and Comparing Evals

This repo (`agent-eval`) is an evaluation harness for AI agent workflows. Evals
run against named environments (local, staging-eu, eu, us, etc.), record results
to Opik, and can be compared across runs to find regressions.

## 1. Running eval sweeps

Use `run_all_evals.sh` to run every suite against an environment. Each suite
becomes a separate Opik experiment. Tag sweeps to group them for comparison
later.

The eval case definitions live in a separate cases repo. By default the
scripts look for a local `evals/` directory first, then fall back to
`../agent-eval-cases/evals/`. You can also pass `--evals-dir <path>` to
point at any directory of suite YAML files. This repo (`agent-eval`) is the
harness only; the cases themselves (suite YAML, expectations, fixtures) are
authored and versioned separately.

> **No `evals/` directory?** See [Linking the evals
> directory](#linking-the-evals-directory) below for setup instructions.

### Tagging convention

Use **timestamp-based tags** so every sweep gets a unique, sortable tag:  
`<env>-<YYYYMMDD>-<HHMMSS>` (e.g. `staging-eu-20260904-140612`, `dev-weu-20260904-140612`).

Generate the tag at sweep start:
```bash
TAG="$(date +%Y%m%d-%H%M%S)"
bash run_all_evals.sh --env staging-eu --tag "staging-eu-${TAG}"
```

This makes it trivial to compare runs across environments and avoids tag
collisions when running sweeps in parallel.

```bash
# Run all suites against local, tagged for later comparison
bash run_all_evals.sh --env local --tag "local-$(date +%Y%m%d-%H%M%S)" --jobs 1

# Run against staging-eu with a timestamped tag
bash run_all_evals.sh --env staging-eu --tag "staging-eu-$(date +%Y%m%d-%H%M%S)"

# Run two environments in parallel with unique tags
bash run_all_evals.sh --env staging-eu --tag "staging-eu-$(date +%Y%m%d-%H%M%S)" -j 3 &
bash run_all_evals.sh --env dev-weu  --tag "dev-weu-$(date +%Y%m%d-%H%M%S)"  -j 3 &
wait
```

Key flags:
- `--env` (required): environment to run against (local, staging-eu, eu, us, dev-weu)
- `--evals-dir`: directory containing suite YAML files (default: `./evals`, then `../agent-eval-cases/evals`)
- `--suite`: substring filter for suite paths (repeatable)
- `--tag`: tag all Opik experiments in the sweep (repeatable, forwarded to `--opik`)
- `-j / --jobs`: max concurrent suites (default 3, use 1 for sequential)
- `--retries`: retry failed suites N times (default 2; the kubectl tunnel drops connections under load but recovers fast)
- `--resume`: re-run only suites missing from Opik for the first `--tag` (queries Opik, compares against the full suite list, runs the ones without an experiment); requires `--tag`
- `--resume-file <path>`: resume from an explicit failed-suites file instead of querying Opik
- Extra args after the flags are forwarded to `agent-evals run` (e.g. `-v`, `--runs 3`)

> **Parallel sweeps share the Opik tunnel.** The first sweep to finish will
> leave the tunnel running if other `run_all_evals.sh` processes are still
> active, so the second sweep's Opik uploads are not interrupted.

The script pre-warms the kubectl tunnel to Opik before launching suites. All
runs include `--opik` so results land in Opik automatically.

### Resuming failed suites

`--resume` uses Opik as the source of truth: it queries Opik for all
experiments with the first `--tag`, compares their names against the full
suite list, and re-runs only the suites that don't have an experiment yet
(i.e. suites that failed before the experiment was created — tunnel drops,
rate limiting, crashes).  Suites whose evals ran but failed still have an
experiment in Opik, so they are **not** resumed (the eval failures are real,
not infrastructure failures).

```bash
# First sweep: some suites fail (rate-limiting, tunnel drops, etc.)
bash run_all_evals.sh --env eu --tag "eu-$(date +%Y%m%d-%H%M%S)"

# Resume only the missing suites (same env + tag so Opik groups them together)
bash run_all_evals.sh --env eu --tag "eu-20260907-120000" --resume

# Repeat until "Nothing to resume"
bash run_all_evals.sh --env eu --tag "eu-20260907-120000" --resume
```

When all suites have experiments in Opik, `--resume` prints "Nothing to
resume" and exits 0.  Use `--resume-file <path>` to resume from an explicit
file (one suite path per line, `#` comments supported) instead of querying
Opik.

## 2. Comparing experiments

Use `compare_experiments` to find evals with the biggest score differences
between two runs. It discovers all experiments matching a name prefix, pairs
them by experiment name across the two selectors, and shows the per-case
deltas.

```bash
# Compare by tag (most common after tagging sweeps)
uv run python -m agent_evals.scripts.compare_experiments \
    --name <suite-prefix> --tag1 staging-eu-20260904-140612 --tag2 dev-weu-20260904-140612 \
    --show-reason --sort regression

# Compare by environment
uv run python -m agent_evals.scripts.compare_experiments \
    --name <suite-prefix> --env1 staging-eu --env2 local

# Compare by experiment IDs directly (from Opik UI)
uv run python -m agent_evals.scripts.compare_experiments \
    --exp1 01a0672b-... --exp2 01a06729-...

# List available experiments to find IDs
uv run python -m agent_evals.scripts.compare_experiments --list --name <suite-prefix>
```

Key flags:
- `--name`: experiment name substring (matches all suites with that prefix)
- `--tag1 / --tag2`: filter each side by Opik tag
- `--env1 / --env2`: filter each side by environment (from experiment metadata)
- `--exp1 / --exp2`: direct experiment IDs (bypasses discovery)
- `--score`: feedback score to compare (default: `overall`; also `must_include`, `expected_state`, `judge`, etc.)
- `--sort`: `delta` (biggest absolute change, default), `regression` (worst first), `improvement` (best first)
- `--n`: number of items to show (default 20)
- `--show-reason`: include the failure reason from each side's score
- `--no-trace`: hide trace URLs (shown by default)

Output format shows `experiment_name / case_name` with scores and delta, plus
trace URLs for both sides.

## 3. Inspecting a specific eval

Once you've found a case with a score delta, use `inspect_eval` to see the full
input (what was sent), output (what the agent responded), and per-expectation
verdicts (why each check passed or failed).

```bash
# List all cases in an experiment
uv run python -m agent_evals.scripts.inspect_eval \
    --exp 01a0672b-4234-7b58-bb27-0d4b048c4a2b --list

# Inspect a specific case (full input + output + verdicts + scores)
uv run python -m agent_evals.scripts.inspect_eval \
    --exp 01a0672b-4234-7b58-bb27-0d4b048c4a2b \
    --case allergies_then_labs_two_round_trips

# Quick view — just expectation verdicts, skip full input/output
uv run python -m agent_evals.scripts.inspect_eval \
    --exp 01a0672b-4234-7b58-bb27-0d4b048c4a2b \
    --case allergies_then_labs_two_round_trips --expectations-only

# Include the full agent config (very large, hidden by default)
uv run python -m agent_evals.scripts.inspect_eval \
    --exp 01a0672b-4234-7b58-bb27-0d4b048c4a2b \
    --case allergies_then_labs_two_round_trips --show-agent
```

The output is structured as:
- **INPUT**: case name, agent (name only by default), per-step message + declared expectations
- **OUTPUT**: per-step response state, response text, and expectation verdicts (`[PASS]`/`[FAIL]` with per-term detail)
- **FEEDBACK SCORES**: all scores (overall + per-expectation-type) with reasons

The `--exp` ID comes from the Opik UI or the `--list` output of `compare_experiments`.

## Typical workflow

1. Tag and run a sweep: `bash run_all_evals.sh --env local --tag "local-$(date +%Y%m%d-%H%M%S)"`
2. Compare against a baseline: `uv run python -m agent_evals.scripts.compare_experiments --name <suite-prefix> --tag1 local-20260904-120000 --tag2 local-20260904-140612 --show-reason --sort regression`
3. Inspect failures: `uv run python -m agent_evals.scripts.inspect_eval --exp <id> --case <case-name>`

## 4. Generating eval reports

When the user asks for a report of eval results, follow the style guide in
[`eval-report-style.md`](eval-report-style.md). It captures layout, theme, and
UX preferences for self-contained HTML reports.

## 5. Linking the evals directory

This repo is the eval harness only. The suite YAML files, expectations, and
fixtures live in a separate cases repo. The scripts look for them in this
order:

1. `--evals-dir <path>` on the command line (highest priority)
2. `./evals/` inside this repo (e.g. a symlink)
3. `../agent-eval-cases/evals/` (a sibling checkout)

### Setup options

**Clone as a sibling** (recommended for multi-repo workflows):
```bash
git clone <cases-repo-url> ../agent-eval-cases
```

**Symlink into this repo** (keeps everything reachable from `./evals`):
```bash
ln -s /path/to/agent-eval-cases/evals evals
```
The symlink is gitignored (see `.gitignore`), so it won't be committed.

**Point at an arbitrary path** (ad-hoc / CI):
```bash
bash run_all_evals.sh --env local --evals-dir /path/to/my-evals
bash run_eval.sh smoke/hello --env local --evals-dir /path/to/my-evals
```
