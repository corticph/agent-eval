# Agent Guide: Running and Comparing Evals

This repo (`agent-eval`) is an evaluation harness for AI agent workflows. Evals
run against named environments (local, staging-eu, eu, us, etc.), record results
to Opik, and can be compared across runs to find regressions.

### Environment ordering

Environments progress from newest to oldest, newest on the left:

1. **local** — developer's machine; bleeding edge
2. **dev** (dev-weu) — shared dev cluster; next-to-merge changes
3. **staging** (staging-eu) — pre-prod; stable, close to production
4. **prod** (eu, us) — production

When comparing two environments, the **newer** environment (left) is the one
under test; the **older** environment (right) is the baseline. Regressions
are score drops from newer vs older; improvements are score gains.

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
bash run_all_evals.sh --env staging-eu --tag "staging-eu-$(date +%Y%m%d-%H%M%S)" &
bash run_all_evals.sh --env dev-weu  --tag "dev-weu-$(date +%Y%m%d-%H%M%S)"  &
wait
```

Key flags:
- `--env` (required): environment to run against (local, staging-eu, eu, us, dev-weu)
- `--evals-dir`: directory containing suite YAML files (default: `./evals`, then `../agent-eval-cases/evals`)
- `--suite`: substring filter for suite paths (repeatable)
- `--tag`: tag all Opik experiments in the sweep (repeatable, forwarded to `--opik`)
- `-j / --jobs`: max concurrent suites (default 10, use 1 for sequential; failed suites can be resumed with `--resume`)
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

#### Re-running suites that hit HTTP 502 (or other mid-eval failures)

`--resume` **will not** re-run suites where 502 errors occurred mid-eval —
the experiment already exists in Opik (it just has 0.0-scored cases).  To
re-run only the affected suites, use `--resume-file` with **absolute
paths**:

```bash
# 1. Create a file listing the affected suite files (absolute paths):
cat > /tmp/rerun.txt <<EOF
/path/to/agent-eval-cases/agent/evals/suite_name/case_a.yaml
/path/to/agent-eval-cases/agent/evals/suite_name/case_b.yaml
EOF

# 2. Re-run with a new tag (so the new results are the ones picked up):
bash run_all_evals.sh --env dev-weu --evals-dir /path/to/evals \
    --tag "dev-weu-$(date +%Y%m%d-%H%M%S)" --resume-file /tmp/rerun.txt
```

**Pitfalls to avoid:**
- **Use absolute paths** in `--resume-file`.  Relative paths are resolved
  from the current working directory, not the `--evals-dir`, and silently
  skip with a "no longer exists" warning.
- **Don't use `--suite` to target specific suites** for re-runs — it's a
  substring filter that matches all suites containing that substring (e.g.
  `--suite foo` re-runs all suites containing "foo", not just the ones that 502'd).
- **Use a new tag** for the re-run so the report generator's "newest per
  suite" merge picks up the fresh results instead of the 502'd ones.
- Alternatively, run individual suite files directly with
  `uv run agent-evals run <path> --env dev-weu --opik --tag <tag>`.

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

## 4. Fetching OpenInference traces

`inspect_eval` shows what went in and what came out; to understand *why* the
agent behaved the way it did — which tools it called, how many tokens each
LLM call consumed, what the reasoning chain looked like — fetch the
OpenInference trace from the agent API's trace endpoint.

```bash
# List cases and their context IDs (no trace fetch):
uv run python -m agent_evals.scripts.fetch_traces --exp <experiment-id> --list

# Fetch a single case's trace as compact text (default):
uv run python -m agent_evals.scripts.fetch_traces --exp <experiment-id> --case my_case

# Fetch all cases' traces:
uv run python -m agent_evals.scripts.fetch_traces --exp <experiment-id>

# Write foldable HTML (collapsible span tree) to a file:
uv run python -m agent_evals.scripts.fetch_traces --exp <experiment-id> --html -o traces.html

# Raw JSON for programmatic access:
uv run python -m agent_evals.scripts.fetch_traces --exp <experiment-id> --json -o traces.json

# Verbose text (timestamps, full tool def descriptions):
uv run python -m agent_evals.scripts.fetch_traces --exp <experiment-id> --case my_case --verbose

# Override the agent API environment (defaults to experiment metadata):
uv run python -m agent_evals.scripts.fetch_traces --exp <experiment-id> --env eu
```

The default compact-text output shows only the conversation messages
(user/assistant/tool), tool definitions (names only), tool calls with
arguments, and tool results — deduplicated across LLM calls so repeated
context doesn't fill the output. Use `--verbose` for timestamps and full
tool descriptions, `--html` for a foldable report, or `--json` for the raw
OpenInference span data.

Each trace contains spans in a tree: root `CHAIN` spans wrap `LLM` calls
and `TOOL` invocations. The compact output walks this tree, showing:
- **[LLM]**: model name, token count, new messages since last call, tool
  list, and the assistant's response (tool calls or text).
- **[TOOL]**: tool name, arguments, and result.
- **[CHAIN]**: orchestration spans (MCP registration, tool listing, etc.).

### Using traces for root-cause analysis

When a case fails or regresses, fetch the trace to understand the agent's
reasoning chain:

1. **Did the agent call the right tools?** Check `[TOOL]` spans — were the
   expected connectors invoked? Were arguments correct?
2. **Did the LLM choose the right tool?** Look at the `[LLM]` response — did
   it pick the right tool call, or hallucinate arguments?
3. **Was the context too large?** Check token counts in `[LLM]` headers — a
   case that regressed may have hit a context window limit.
4. **Did a tool return unexpected data?** Compare `[TOOL]` results between
   the two environments' traces for the same case.
5. **Did the tool list change?** The `tools:` line shows what tools the LLM
   was offered — a missing or renamed tool could explain a regression.

Compare traces across environments by fetching the same case from two
experiments:

```bash
uv run python -m agent_evals.scripts.fetch_traces --exp <exp1> --case my_case -o trace1.txt
uv run python -m agent_evals.scripts.fetch_traces --exp <exp2> --case my_case -o trace2.txt
diff trace1.txt trace2.txt
```

## Typical workflow

1. Tag and run a sweep: `bash run_all_evals.sh --env local --tag "local-$(date +%Y%m%d-%H%M%S)"`
2. Compare against a baseline: `uv run python -m agent_evals.scripts.compare_experiments --name <suite-prefix> --tag1 local-20260904-120000 --tag2 local-20260904-140612 --show-reason --sort regression`
3. Inspect failures: `uv run python -m agent_evals.scripts.inspect_eval --exp <id> --case <case-name>`
4. Fetch the trace to understand the agent's reasoning: `uv run python -m agent_evals.scripts.fetch_traces --exp <id> --case <case-name>`
5. (Optional) Generate an HTML report following [`eval-report-style.md`](eval-report-style.md).

### Deep-dive on regressions

After comparing, don't just report the deltas — **do root-cause dives** on the
worst regressions. For each case with a significant score drop:

1. Fetch the trace from both environments (see [Fetching OpenInference
   traces](#4-fetching-openinference-traces)).
2. Compare the traces side-by-side (`diff trace1.txt trace2.txt`) to find
   where the agent's behaviour diverged.
3. Check whether the failure is an infrastructure issue (tunnel drops,
   rate limiting) or a real regression (wrong tool call, hallucinated
   arguments, context window hit).

**Use sub-agents in parallel.** When several cases regress, dispatch one
sub-agent per case to fetch traces, inspect evals, and summarise the root
cause independently. This is much faster than serial investigation and
each sub-agent's context stays focused on a single failure.

```bash
# Example: parallel root-cause dives on top regressions
# Each sub-agent gets one case and does: fetch traces (both envs), diff,
# inspect_eval, and report back with a short root-cause summary.
```

## 5. Generating eval reports

When the user asks for a report of eval results, follow the style guide in
[`eval-report-style.md`](eval-report-style.md). It captures layout, theme, and
UX preferences for self-contained HTML reports.

The report generator (`agent_evals.scripts.generate_report`) is an **example
script** — adapt the categorization, styling, and layout to your own workflow.
It produces the report in two steps:

```bash
# 1. Generate the report from Opik data (supports multiple tags per side,
#    newest experiment per suite name wins):
uv run python -m agent_evals.scripts.generate_report generate \
    --tag1 <baseline-tag> --tag2 <beta-tag> \
    --label1 "baseline" --label2 "beta" \
    -o report.html

# 2. Inspect the report, find element IDs you want to link to
#    (e.g. #rca-timeout, #case-no-data-parts-my_case, #imp-my_case)

# 3. Write author insights as HTML (insights.html) with <a href="#..."> links
#    to specific sections/cases, then inject:
uv run python -m agent_evals.scripts.generate_report add-insights \
    -o report.html --insights insights.html
```

### Report structure

The report includes these sections (all linkable via `id`):

- **Summary line** — mean scores, delta, regressed/improved/unchanged counts,
  credit totals.
- **Insights** (`#insights`) — author-written HTML with links to specific
  cases and sections. Open by default. Injected via `add-insights`.
- **Beta Changes Context** (`#beta-context`) — summary of what changed
  between the two environments. Open by default.
- **Root Cause Analysis** (`#root-cause-analysis`) — regressions grouped by
  pattern. Each category is a collapsible (`#rca-{key}`) with a callout box
  and nested case cards (`#case-{cat_key}-{case_name}`).
- **Improvements** (`#improvements`) — all improved cases, collapsible.
  Each case is linkable (`#imp-{case_name}`).
- **Credit Usage by Suite** (`#credit-usage`) — per-suite credit table with
  deltas and percentage change.
- **Suite Breakdown** (`#suite-breakdown`) — every suite as a collapsible
  card (`#suite-{name}`), with nested case cards
  (`#suitecase-{suite_name}-{case_name}`).

### Deep-dive root cause analysis

After comparing, don't just report the deltas — **do root-cause dives** on the
worst regressions. For each case with a significant score drop:

1. Fetch the trace from both environments (see [Fetching OpenInference
   traces](#4-fetching-openinference-traces)).
2. Compare the traces side-by-side (`diff trace1.txt trace2.txt`) to find
   where the agent's behaviour diverged.
3. Check whether the failure is an infrastructure issue (tunnel drops,
   rate limiting) or a real regression (wrong tool call, hallucinated
   arguments, context window hit).

**Use sub-agents in parallel.** When several cases regress, dispatch one
sub-agent per pattern (not per case — one pattern spans multiple cases) to
fetch traces, inspect evals, and summarise the root cause independently.
This is much faster than serial investigation and each sub-agent's context
stays focused on a single failure pattern.

### Verifying insights before publishing

Before injecting insights into the report, **verify every quantitative claim**
against the data:

1. **Run the comparison data through a script** (or `full_comparison.py`) to
   get exact counts: total cases, suites, regressed/improved/unchanged,
   credits, mean scores. Do not round or approximate from memory.
2. **Categorize all regressions** programmatically — don't leave a large
   "Other" bucket. If >10% of regressions are "other", the categorization
   function in the report generator needs new patterns.
3. **Verify cost claims**: count infrastructure-failure cases and their
   credits separately. Do not estimate "roughly X% of the reduction" —
   compute it: `infra_credits / abs(credit_delta) * 100`.
4. **Check that insight links resolve**: every `#rca-*`, `#case-*`,
   `#imp-*`, `#suite-*` href in the insights HTML must match an `id` in
   the generated report. After `add-insights`, grep for `href="#` and
   verify each target exists.
5. **Re-run insights after re-runs**: if you re-run failed suites with new
   tags, regenerate the report and re-inject insights. Old insights will
   have stale numbers (e.g. "17 cases hit 502" when they've been re-run).

### Cost analysis

Each Opik experiment item carries a `usage.credits` field in its task
output. The report extracts this per case and aggregates per suite. The
credit usage table shows both sides' totals and the percentage change,
making it easy to see whether the beta is more or less expensive than
the baseline.

**When reporting credit reductions**, distinguish between:
- **Efficiency gains**: cases that ran successfully on both sides but
  consumed fewer credits on the beta (leaner history, fewer LLM calls).
- **Infrastructure failures**: cases that failed on the beta (502,
  timeout, connection error) and consumed ~0 credits. These inflate the
  reduction but are not real savings.
- Compute `infra_credits / abs(total_credit_delta) * 100` to quantify
  the infrastructure contribution before claiming the reduction is from
  efficiency.

## 6. Linking the evals directory

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
