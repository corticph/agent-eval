# Multi-Model Eval Comparison

How to run multi-model eval comparisons and generate HTML reports using
`compare_multi.py`. Written after a full 3-model comparison on dev-weu
(Default vs corti-s1-instant vs corti-s1, 113 cases across 19 suites).

## 1. Running eval sweeps with model overrides

Use `--model` to override the LLM model for all agents in a sweep. Tag
each sweep with `<env>-<model>-<YYYYMMDD>-<HHMMSS>` for unique, sortable tags:

```bash
uv run agent-evals sweep --env dev-weu --no-opik --jobs 5 \
    --model corti-s1-instant --tag "dev-corti-s1-instant-$(date +%Y%m%d-%H%M%S)"
```

- `--no-opik` skips the Opik tunnel — results go to `results/*.json` only.
  Use this for speed unless you need Opik traces.
- `--model` overrides the LLM model for all agents in the sweep. Available
  models: `corti-s1`, `corti-s1-instant`, `corti-s1-mini`, `corti-s1-mini-instant`,
  `corti-s1-tiny`, `corti-s1-tiny-instant` (check `GET https://ai.eu.corti.app/v1/models`).
- `--jobs` controls concurrency. 5 is a good default for dev-weu; higher values
  risk rate limiting and SSL errors under load.
- Results always go to `<repo>/results/` regardless of where the evals
  directory lives.

### Running 3 sweeps in parallel

Launch all 3 as background processes, each with its own log file and tag:

```bash
TS=$(date +%Y%m%d-%H%M%S)
uv run agent-evals sweep --env dev-weu --no-opik --jobs 5 \
    --tag "dev-default-${TS}" > /tmp/sweep-default.log 2>&1 &
uv run agent-evals sweep --env dev-weu --no-opik --jobs 5 \
    --model corti-s1-instant --tag "dev-corti-s1-instant-${TS}" > /tmp/sweep-instant.log 2>&1 &
uv run agent-evals sweep --env dev-weu --no-opik --jobs 5 \
    --model corti-s1 --tag "dev-corti-s1-${TS}" > /tmp/sweep-s1.log 2>&1 &
```

3 sweeps × 5 jobs = 15 concurrent `agent-evals run` processes hitting the same
dev-weu API. This works but the later suites (alphabetically) may hit token
fetch failures or SSL errors if the API is under load. If a run is missing
suites, re-run just those suites individually (see below).

## 2. Running a subset of suites

The `--suite` flag on `sweep` is a substring filter (repeatable) that works
when used alone. However, extra args forwarded via `REMAINDER` (e.g.
`--runs 3`, `--stop-on-failure`) must come after `--` to avoid argparse
confusion:

```bash
# --suite works fine on its own:
uv run agent-evals sweep --env dev-weu --suite pubmed --tag "dev-$(date +%Y%m%d-%H%M%S)"

# Extra args need -- separator:
uv run agent-evals sweep --env dev-weu --suite pubmed --tag "dev-$(date +%Y%m%d-%H%M%S)" -- --runs 3
```

For running individual suites directly (more reliable than `--suite` for
re-runs), use `agent-evals run`:

```bash
uv run agent-evals run <evals-dir>/pubmed/reference.yaml --env dev-weu \
    --model corti-s1 --tag "dev-corti-s1-$(date +%Y%m%d-%H%M%S)"
```

## 3. Filling coverage gaps

After a sweep, some suites may have failed (token fetch errors, HTTP 500,
timeouts). Re-run just the missing suites with the same tag:

```bash
for suite in pubmed/reference.yaml pubmed/retrieval.yaml; do
    uv run agent-evals run "<evals-dir>/$suite" --env dev-weu \
        --model corti-s1 --tag "dev-corti-s1-${TS}" &
done
```

### Verifying all runs have the same suites

```bash
python3 -c "
import json
from pathlib import Path
tag_groups = {
    'Default': ['dev-default-...', 'dev-default-...'],
    'corti-s1-instant': ['dev-instant-...', 'dev-instant-...'],
    'corti-s1': ['dev-s1-...', 'dev-s1-...'],
}
for label, tags in tag_groups.items():
    suites = set()
    for f in Path('results').rglob('*.json'):
        d = json.load(open(f))
        meta = d[0] if isinstance(d, list) and d and '_metadata' in d[0] else {}
        if any(t in meta.get('tags', []) for t in tags):
            items = [e for e in d if isinstance(e, dict) and not e.get('_metadata') and not e.get('is_step_result')]
            if items:
                suites.add(meta.get('suite_name', '?'))
    print(f'{label}: {sorted(suites)}')
"
```

All runs should list the same suites before generating the report.

## 4. Generating the multi-way comparison report

`compare_multi.py` supports N runs (3+) with a single HTML report containing
summary scores, rankings, tabbed suite breakdown (Suite Deltas / Credits /
Case-by-Case), and an author-written insights section.

```bash
uv run python -m agent_evals.scripts.compare_multi \
    --source local \
    --tag "dev-default-20260909-233541" "dev-default-20260910-001528" --label "Default (dev)" \
    --tag "dev-corti-s1-instant-20260909-233644" "dev-corti-s1-instant-20260910-001528" --label "corti-s1-instant" \
    --tag "dev-corti-s1-20260909-233541" "dev-corti-s1-20260910-001528" --label "corti-s1" \
    --baseline 0 \
    --insights /tmp/insights.html \
    -o results/multi-model-comparison.html
```

Key flags:
- `--source local` — reads from `results/*.json` (no Opik needed).
- `--tag` — repeatable; multiple tags per run are merged (newest result per
  suite wins). Pass both the old and new sweep tags to fill gaps.
- `--label` — one per `--tag` group; must match count.
- `--baseline N` — 0-based index of the baseline run (default: 0 = first).
- `--insights` — path to an HTML file with author-written analysis.

### Adding insights after generating

```bash
uv run python -m agent_evals.scripts.compare_multi add-insights \
    -o results/multi-model-comparison.html --insights /tmp/insights.html
```

### Writing insights

Insights are raw HTML injected into a `<details open id="insights">` block.
Keep them concise — a TL;DR paragraph plus 3-5 short sections with links to
specific cases. The report generates anchor IDs from suite + case names:

```python
# Anchor format (sanitized: spaces/slashes/dots -> dashes, lowercased):
# case-{suite_name}-{case_name}
```

Example insight link:
```html
<a href="#case-pubmed_direct_attach-l0_plain_lookup">l0_plain_lookup</a>
```

To find the exact anchor ID, grep the generated HTML:
```bash
grep -oE 'id="case-[^"]*"' results/multi-model-comparison.html | sort -u
```

## 5. Digging into results for root-cause analysis

### Per-case scores and reasons

```bash
uv run python -c "
import json
from pathlib import Path
from agent_evals.scripts.local_store import _compute_feedback_scores

tag = 'dev-corti-s1-20260910-001528'
for f in sorted(Path('results').rglob('*.json'), key=lambda x: -x.stat().st_mtime):
    d = json.load(open(f))
    meta = d[0] if isinstance(d, list) and d and '_metadata' in d[0] else {}
    if tag not in meta.get('tags', []): continue
    for entry in d:
        if entry.get('_metadata') or entry.get('is_step_result'): continue
        sr = entry.get('step_results', [])
        scores = _compute_feedback_scores(sr)
        overall = next((s['value'] for s in scores if s['name'] == 'overall'), None)
        if overall is not None and overall < 1.0:
            reason = next((s['reason'] for s in scores if s['name'] == 'overall'), '')
            print(f'{entry[\"name\"]}: {overall:.3f} — {reason[:200]}')
"
```

### Response text (for understanding why a case failed)

Response text lives in `step_results[].response.task.artifacts[].parts[].text`:

```python
for sr in entry.get('step_results', []):
    resp = sr.get('response', {})
    task = resp.get('task', {})
    for a in task.get('artifacts', []):
        for p in a.get('parts', []):
            if 'text' in p:
                print(p['text'][:400])
```

### Durations

`entry['duration_seconds']` gives the wall-clock time. Compare across runs
to identify timeout-driven failures.

## 6. Scoring: how infra failures are handled

### The problem

`_compute_feedback_scores` in `local_store.py` computes scores from
`step_results[].expectation_results`. When a step fails with no expectation
results (e.g. timeout, network error), there are no checks to score. Without
a fix, such steps are invisible — `overall` defaults to 1.0.

### The fix

Two cases are handled:

1. **All steps have no expectations and at least one failed**: `overall = 0.0`
   with the harness error message as the reason.

2. **Some steps have expectations, others failed with none**: A synthetic
   `_failed` column with value 0.0 is injected for each failed step. This
   drags `overall` down proportionally. For example, a 2-step case where
   step 1 passes (3/3) and step 2 times out scores 0.5 overall (was 1.0
   before the fix).

This only affects the local JSON path (`local_store.py`). The Opik path
(`_sequential_columns` in `reporting/opik.py`) uses declared expectations
from the suite YAML to generate all-failed placeholders for unexecuted
steps, so it already scored these cases correctly.

### Detecting affected cases

```bash
uv run python -c "
import json
from pathlib import Path
for f in Path('results').rglob('*.json'):
    d = json.load(open(f))
    for entry in d:
        if entry.get('_metadata') or entry.get('is_step_result'): continue
        for sr in entry.get('step_results', []) or []:
            er = sr.get('expectation_results') or []
            if not er and not sr.get('success', True):
                he = sr.get('harness_error') or {}
                print(f'{f.name} | {entry[\"name\"]} | {he.get(\"code\")}: {str(he.get(\"message\",\"\"))[:60]}')
"
```

## 7. HTML report UX

### Sortable tables

Suite Deltas and Credits tables have clickable column headers. Click a
header to sort by that column; click again to toggle direction. The sort
state is shared across all three tabs, so switching from Suite Deltas to
Case-by-Case preserves the same suite ordering.

### Anchor links that work across tabs

Case anchor links (`#case-...`) in the insights section point to `<details>`
elements inside collapsed parent `<details>` (suite cards) on the
Case-by-Case tab. The `openAnchorTarget(id)` JS function opens the target,
expands all parent `<details>`, switches to the correct tab, and scrolls
into view. This also works on page load (e.g. `report.html#case-foo`).

## 8. Pitfalls to avoid

- **Don't expect credit data from `--model` overrides** — the agent API
  returns `null` for `credits` when a non-default model is selected. Credit
  comparison is only meaningful between runs without model overrides.
- **Do re-run missing suites** — parallel sweeps on dev-weu can hit token
  fetch failures (SSL errors, rate limiting) on later suites. Re-run just
  the affected suites with the same tag.
- **Do verify all runs have the same suites** before generating the report.
  `compare_multi` shows "Only in" warnings but the report is cleaner when
  all sides match.
- **Use `--` before extra args on sweep** — `argparse.REMAINDER` captures
  everything after the first unknown token. Put extra `run` flags (e.g.
  `--runs 3`) after `--` so they're forwarded correctly.
