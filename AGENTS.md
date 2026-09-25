# Agent Guide: Running and Comparing Evals

This repo (`agent-eval`) is an evaluation harness for AI agent workflows. Evals
run against named environments (local, staging-eu, eu, us, etc.), record results
to Opik, and can be compared across runs to find regressions.

### Guides

This file covers the core workflows. For deeper topics, see:

- [`docs/eval-report-style.md`](docs/eval-report-style.md) — layout, theme, and
  UX preferences for self-contained HTML eval reports (2-sided `generate_report`)

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

Use `agent-evals sweep` to run every suite against an environment. Each suite
becomes a separate experiment (Opik and/or local JSON). Tag sweeps to group
them for comparison later.

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
uv run agent-evals sweep --env staging-eu --tag "staging-eu-${TAG}"
```

This makes it trivial to compare runs across environments and avoids tag
collisions when running sweeps in parallel.

```bash
# Run all suites against local, tagged for later comparison
uv run agent-evals sweep --env local --tag "local-$(date +%Y%m%d-%H%M%S)" --jobs 1

# Run against staging-eu with a timestamped tag
uv run agent-evals sweep --env staging-eu --tag "staging-eu-$(date +%Y%m%d-%H%M%S)"

# Run two environments in parallel with unique tags
uv run agent-evals sweep --env staging-eu --tag "staging-eu-$(date +%Y%m%d-%H%M%S)" &
uv run agent-evals sweep --env dev-weu  --tag "dev-weu-$(date +%Y%m%d-%H%M%S)"  &
wait
```

> **Sweeps are long-running.** Run it as a background process and check in
> every 2–4 minutes:
>
> ```bash
> # Start in the background, capture output to a log
> uv run agent-evals sweep --env dev-weu --tag "dev-weu-$(date +%Y%m%d-%H%M%S)" \
>     > /tmp/sweep.log 2>&1 &
> echo $!  # note the PID
>
> # Check progress
> tail -20 /tmp/sweep.log
> ```

Key flags:
- `--env` (required): environment to run against (local, staging-eu, eu, us, dev-weu)
- `--evals-dir`: directory containing suite YAML files (default: `./evals`, then `../agent-eval-cases/evals`)
- `--suite`: substring filter for suite paths (repeatable)
- `--tag`: tag experiments in the sweep (repeatable, stored in Opik tags and/or local JSON metadata)
- `--model`: override the LLM model for all agents in the sweep (forwarded to each `run` subprocess)
- `-j / --jobs`: max concurrent suites (default 10, use 1 for sequential; failed suites can be resumed with `--resume`)
- `--retries`: retry failed suites N times (default 2; the kubectl tunnel drops connections under load but recovers fast)
- `--no-opik`: skip Opik recording and tunnel (write local JSON only; tags are still stored in JSON metadata)
- `--resume`: re-run only suites missing from the first `--tag` (checks local results first, then Opik); requires `--tag`
- `--resume-file <path>`: resume from an explicit failed-suites file instead of querying Opik
- Extra args after `--` are forwarded to `agent-evals run` (e.g. `-- --runs 3`, `-- -v --stop-on-failure`)

> **Parallel sweeps share the Opik tunnel.** The first sweep to finish will
> leave the tunnel running if other sweep processes are still active, so
> the second sweep's Opik uploads are not interrupted.

The command pre-warms the kubectl tunnel to Opik before launching suites
(unless `--no-opik` is passed). Without `--no-opik`, all runs include `--opik`
so results land in Opik automatically.

> **Thin shell wrappers** `run_all_evals.sh` and `run_eval.sh` are kept as
> one-line wrappers for muscle memory — they just forward to `agent-evals
> sweep` and `agent-evals run` respectively.

### Running locally (without Opik)

Add `--no-opik` to any sweep to skip Opik recording and write results to
`results/*.json` only. Tags are persisted in the JSON metadata, so you can
compare local runs the same way you compare Opik runs.

**Results directory**: results are always written to `<repo>/results/`,
regardless of where the evals directory lives (symlink, sibling checkout, or
`--evals-dir`).  The local cache reads from this location by default — no
`--results-dir` needed for the common case.  Use `--results-dir` to scan
additional or legacy directories.

```bash
# Run sweep locally with a tag
uv run agent-evals sweep --env local --tag "local-$(date +%Y%m%d-%H%M%S)" --no-opik

# Or via the shell wrapper
bash run_all_evals.sh --env staging-eu --tag "staging-eu-$(date +%Y%m%d-%H%M%S)" --no-opik --jobs 2

# Compare two local sweeps by tag (no --name needed — matches all suites)
uv run python -m agent_evals.scripts.compare_experiments \
    --source local --tag1 local-20260909-120000 --tag2 local-20260909-140000 \
    --sort regression

# Or with results in a sibling cases repo
uv run python -m agent_evals.scripts.compare_experiments \
    --source local --results-dir ../agent-eval-cases/results \
    --tag1 tag-a --tag2 tag-b --sort regression

# Or scan multiple result directories (newest per suite wins)
uv run python -m agent_evals.scripts.compare_experiments \
    --source local \
    --results-dir results --results-dir ../agent-eval-cases/results \
    --tag1 tag-a --tag2 tag-b --sort regression
```

The `--source local` flag is supported on all three comparison scripts:
`compare_experiments`, `inspect_eval`, and `generate_report`.  Without it,
they default to Opik (`--source opik`).

- **compare_experiments**: discover experiments by tag or environment, compare scores
- **inspect_eval**: with `--exp` set to a local file path (e.g. `results/smoke/hello.json`)
- **generate_report**: generate HTML reports from local JSON using `--source local`

Each result JSON file starts with a `_metadata` entry carrying tags and
environment.  Results without one (written before this feature) still work
via substring matching on file paths.

#### Local-cache-first (default Opik mode)

Even with `--source opik` (the default), the comparison scripts check
`results/*.json` **before** hitting the Opik API.  If a local result file
with a matching tag exists, it's used directly — avoiding expensive Opik
fetches for recent runs whose JSON is still on disk.  Falls back to Opik
only when local data is missing.

This means you don't need `--source local` for the common case of comparing
recent sweeps: just run the sweep (which writes local JSON automatically),
then compare by tag as usual.

```bash
# Run sweep (writes local JSON + uploads to Opik)
uv run agent-evals sweep --env staging-eu --tag "staging-eu-$(date +%Y%m%d-%H%M%S)"

# Compare — uses local cache first, falls back to Opik API
uv run python -m agent_evals.scripts.compare_experiments \
    --tag1 staging-eu-20260909-120000 --tag2 dev-weu-20260909-120000 \
    --sort regression
```

Use `--results-dir <path>` to point the local cache at a non-default
location (e.g. `../agent-eval-cases/results`).

When comparing by tag, you can omit `--name` to match all suites (tag-only
mode).  If one side's results are on disk and the other's aren't, the
comparison automatically uses local JSON for one side and the Opik API for
the other — no extra flags needed.

> **Note:** the default discovery limit is 500 experiments.  If your Opik
> project has more, pass `--limit 5000` so the Opik fallback can find older
> tagged experiments.

### Resuming failed suites

`--resume` checks local results first, then falls back to Opik: it queries for all
experiments with the first `--tag`, compares their names against the full
suite list, and re-runs only the suites that don't have an experiment yet
(i.e. suites that failed before the experiment was created — tunnel drops,
rate limiting, crashes).  Suites whose evals ran but failed still have an
experiment, so they are **not** resumed (the eval failures are real,
not infrastructure failures).

```bash
# First sweep: some suites fail (rate-limiting, tunnel drops, etc.)
uv run agent-evals sweep --env eu --tag "eu-$(date +%Y%m%d-%H%M%S)"

# Resume only the missing suites (same env + tag so Opik groups them together)
uv run agent-evals sweep --env eu --tag "eu-20260907-120000" --resume

# Repeat until "Nothing to resume"
uv run agent-evals sweep --env eu --tag "eu-20260907-120000" --resume
```

When all suites have experiments, `--resume` prints "Nothing to
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
uv run agent-evals sweep --env dev-weu --evals-dir /path/to/evals \
    --tag "dev-weu-$(date +%Y%m%d-%H%M%S)" --resume-file /tmp/rerun.txt
```

**Pitfalls to avoid:**
- **Use absolute paths** in `--resume-file`.  Relative paths are resolved
  from the current working directory, not the `--evals-dir`, and silently
  skip with a "no longer exists" warning.
- **`--suite` is a substring filter**, not an exact match — `--suite foo`
  matches all suites containing "foo". For re-running specific suites, use
  `--resume-file` or run individual suite files directly.
- **Use `--` before extra args** (e.g. `-- --runs 3`) — `argparse.REMAINDER`
  captures everything after the first unknown token, so flags like
  `--stop-on-failure` must come after `--`.
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

# Compare local results by tag (no --name needed — matches all suites)
uv run python -m agent_evals.scripts.compare_experiments \
    --source local --tag1 local-20260909-120000 --tag2 local-20260909-140000 \
    --sort regression

# Or with results in a sibling cases repo
uv run python -m agent_evals.scripts.compare_experiments \
    --source local --results-dir ../agent-eval-cases/results \
    --tag1 tag-a --tag2 tag-b --sort regression

# Or scan multiple result directories (newest per suite wins)
uv run python -m agent_evals.scripts.compare_experiments \
    --source local \
    --results-dir results --results-dir ../agent-eval-cases/results \
    --tag1 tag-a --tag2 tag-b --sort regression

# List available experiments to find IDs
uv run python -m agent_evals.scripts.compare_experiments --list --name <suite-prefix>
```

Key flags:
- `--name`: experiment name substring (omit to match all suites in tag-only mode)
- `--tag1 / --tag2`: filter each side by tag (accepts multiple values — newest per suite wins)
- `--env1 / --env2`: filter each side by environment (from experiment metadata)
- `--exp1 / --exp2`: direct experiment IDs (bypasses discovery)
- `--source`: `opik` (default) or `local` (uses `results/*.json`)
- `--results-dir`: path to results directory for local source (repeatable, default: `results/`). Use this when results live outside this repo (e.g. `../agent-eval-cases/results/`).
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

# With local source, --exp is a file path:
uv run python -m agent_evals.scripts.inspect_eval \
    --source local --exp results/smoke/hello.json --case my_case
```

The output is structured as:
- **INPUT**: case name, agent (name only by default), per-step message + declared expectations
- **OUTPUT**: per-step response state, response text, and expectation verdicts (`[PASS]`/`[FAIL]` with per-term detail)
- **FEEDBACK SCORES**: all scores (overall + per-expectation-type) with reasons

The `--exp` ID comes from the Opik UI or the `--list` output of `compare_experiments`.
For local source, `--exp` is a file path (e.g. `results/smoke/hello.json`).

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

1. Tag and run a sweep: `uv run agent-evals sweep --env local --tag "local-$(date +%Y%m%d-%H%M%S)"`
2. Compare against a baseline: `uv run python -m agent_evals.scripts.compare_experiments --name <suite-prefix> --tag1 local-20260904-120000 --tag2 local-20260904-140612 --show-reason --sort regression`
3. Inspect failures: `uv run python -m agent_evals.scripts.inspect_eval --exp <id> --case <case-name>`
4. Fetch the trace to understand the agent's reasoning: `uv run python -m agent_evals.scripts.fetch_traces --exp <id> --case <case-name>`
5. (Optional) Generate an HTML report following [`docs/eval-report-style.md`](docs/eval-report-style.md).

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
[`docs/eval-report-style.md`](docs/eval-report-style.md). It captures layout, theme, and
UX preferences for self-contained HTML reports.

The report generator (`agent_evals.scripts.generate_report`) is an **example
script** — adapt the categorization, styling, and layout to your own workflow.
It produces the report in two steps:

```bash
# 1. Generate the report from Opik or local data (supports multiple tags per side,
#    newest experiment per suite name wins):
uv run python -m agent_evals.scripts.generate_report generate \
    --tag1 <baseline-tag> --tag2 <beta-tag> \
    --label1 "baseline" --label2 "beta" \
    -o report.html

# 2. Inspect the report, find element IDs you want to link to
#    (e.g. #rca-timeout, #case-no-data-parts-my_case, #imp-my_case)

# 3. Write author insights as HTML with <a href="#..."> links to specific
#    sections/cases, then inject:
uv run python -m agent_evals.scripts.generate_report add-insights \
    -o report.html --insights /tmp/insights.html
```

**Before injecting insights, verify every quantitative claim** against the
data — run the comparison through a script to get exact counts, categorize
all regressions programmatically, and grep the generated HTML to confirm
every `href="#..."` link resolves to an existing `id`. See
[`docs/eval-report-style.md`](docs/eval-report-style.md) for the full checklist and
report structure.

## 6. Multi-model comparison

For comparing 3+ model runs side by side, use `compare_multi`. Run one sweep
per model with `--model` and a unique tag, then generate the report:

```bash
# Run one sweep per model (parallel, each with its own tag):
TS=$(date +%Y%m%d-%H%M%S)
uv run agent-evals sweep --env dev-weu --no-opik --jobs 5 \
    --tag "dev-default-${TS}" > /tmp/sweep-default.log 2>&1 &
uv run agent-evals sweep --env dev-weu --no-opik --jobs 5 \
    --model corti-s1-instant --tag "dev-corti-s1-instant-${TS}" > /tmp/sweep-instant.log 2>&1 &
uv run agent-evals sweep --env dev-weu --no-opik --jobs 5 \
    --model corti-s1 --tag "dev-corti-s1-${TS}" > /tmp/sweep-s1.log 2>&1 &
wait

# Generate the multi-way comparison report:
uv run python -m agent_evals.scripts.compare_multi \
    --source local \
    --tag "dev-default-${TS}" --label "Default (dev)" \
    --tag "dev-corti-s1-instant-${TS}" --label "corti-s1-instant" \
    --tag "dev-corti-s1-${TS}" --label "corti-s1" \
    --baseline 0 \
    --insights /tmp/insights.html \
    -o results/multi-model-comparison.html
```

Key flags:
- `--tag` — repeatable; multiple tags per run are merged (newest result per
  suite wins). Each `--tag` group starts a new run; pair with `--label`.
- `--label` — one per `--tag` group; must match count.
- `--baseline N` — 0-based index of the baseline run (default: 0).
- `--source` — `opik` (default, local-cache-first) or `local`.
- `--insights` — path to an HTML file with author-written analysis.

The report has tabbed suite breakdown (Suite Deltas / Credits / Case-by-Case)
with sortable column headers, and anchor links that auto-expand parent
details and switch tabs. Use `add-insights` to inject or update insights
after generation:

```bash
uv run python -m agent_evals.scripts.compare_multi add-insights \
    -o results/multi-model-comparison.html --insights /tmp/insights.html
```

### Filling coverage gaps

Parallel sweeps on dev-weu can hit rate limiting or SSL errors on later
suites. Re-run just the missing suites with `agent-evals run` and the same
tag:

```bash
uv run agent-evals run <evals-dir>/pubmed/reference.yaml --env dev-weu \
    --model corti-s1 --tag "dev-corti-s1-${TS}"
```

Verify all runs have the same suites before generating the report —
`compare_multi` shows "Only in" warnings, but the report is cleaner when
all sides match.

## 7. Linking the evals directory

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
uv run agent-evals sweep --env local --evals-dir /path/to/my-evals
uv run agent-evals run smoke/hello --env local --evals-dir /path/to/my-evals
```
