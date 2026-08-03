# Agent Evaluation Harness

A lightweight Python toolkit for running repeatable evaluations against the AI agent service.

## Features

- Declarative YAML suites describing each evaluation case
- First-class support for two-step agent interactions (create agent, then send a message)
- Variable injection from environment variables for secrets and tokens
- JSON output for downstream analysis (plus a Markdown transcript)
- Simple CLI (`uv run agent-evals run`) with verbosity controls

## Quick Start

1. **Install dependencies (creates `.venv/` automatically)**

   ```bash
   uv sync
   ```

2. **Provide credentials for the environment you want to target**

   Create a `.env` file in the project root (see [Configuration](#configuration) for the full inventory):

   ```bash
   AGENT_API_CLIENT_ID_EU=your-eu-client-id
   AGENT_API_CLIENT_SECRET_EU=your-eu-client-secret
   ```

3. **Run a suite**

   Every run names its target environment — there is no default:

   ```bash
   uv run agent-evals run evals/hello.yaml --env eu -v
   # Run with up to 4 cases in flight at a time
   uv run agent-evals run evals/hello.yaml --env eu --concurrency 4 -v
   ```

   Each command will create the agent, send the evaluation prompt, and record the response and expectation status into a JSON file under `results/` (override the path with `--output`). Every result entry also includes the elapsed runtime in seconds (`duration_seconds`). A sibling Markdown file (same name, `.md` extension) captures the same duration, pretty-printed input/output, and a Trace link into the chosen environment's Opik project (the thread view for the response's `contextId`). Local runs have no Opik, so their results simply omit the Trace line. Setting `OPIK_URL_OVERRIDE` redirects trace links (and Opik sink writes when `--opik` is on) to that Opik instead.

   To make concurrency permanent for a suite, add `concurrency: <n>` under the `globals` section of the YAML file.

   To quickly inspect the cases in a suite without running them, use the show shortcut:

   ```bash
   uv run agent-evals --show evals/hello.yaml
   ```

## Configuration

The whole configuration surface is: **pick an environment, provide that environment's credentials, done.** The inventory below is also printed by `uv run agent-evals run --help` — both derive from the same declarations in the code, so they cannot drift apart.

### 1. Pick an environment

Select the target with `--env <name>`. It is required and has no default; a run without it fails immediately, before any network activity. Supported environments:

| Environment  | Target                                            |
|--------------|----------------------------------------------------|
| `local`      | A locally running agent service (`localhost:8080`; override with `AGENT_API_URL_LOCAL`) |
| `eu`         | Production — Europe                                |
| `us`         | Production — US                                    |

The environment resolves everything else: base URL, token, tenant header, and where trace links point. There are no per-knob overrides (base URL, token, tenant, scope, auth header) — a stale flag like `--base-url` is rejected by the CLI rather than silently ignored.

### 2. Provide its credentials

Remote environments authenticate via the OAuth client-credentials flow; each has its own suffixed variable pair, so one `.env` file can hold every target at once and switching environments is only a `--env` change:

```bash
# eu (Production - Europe)
AGENT_API_CLIENT_ID_EU=...
AGENT_API_CLIENT_SECRET_EU=...

# us (Production - US)
AGENT_API_CLIENT_ID_US=...
AGENT_API_CLIENT_SECRET_US=...

# local (static token — the local stack has no OAuth)
AGENT_API_TOKEN_LOCAL=...
```

**Note:** The `.env` file is gitignored to prevent committing sensitive credentials.

**External clients:** you don't need the suffixed pairs above — run `make setup` and paste the Corti Console's "Copy all as .env variables" block (Console → your project → Developer quickstart). The block names your tenant, serves as the OAuth client for any environment without its own `AGENT_API_*` pair, and credentials the LLM judge (missing credentials surface at the run-start preflight; invalid credentials fail on the first judge call). When both are set, an environment's own `AGENT_API_*` pair wins over the block.

A missing or unresolvable token is an immediate error naming the variable(s) to set — no request is ever sent doomed to 401. For manual token inspection, `uv run python get_token.py <environment>` prints the same access token the harness would use.

Fixed by design, not configurable: the OAuth scope is always `openid`. The tenant (Keycloak realm and `Tenant-Name` header are one concept) comes from `CORTI_TENANT_NAME` and defaults to `base`.

### 3. (Optional) Redirect Opik

`OPIK_URL_OVERRIDE` points both Opik sink writes (when `--opik` is on) and result-file trace links at another Opik instance (a local one, or CI machines without cluster credentials). Unset, both derive from the environment. See [Opik Evals](#opik-evals-where-results-go).

## Helper Scripts

Run a single suite by its path under `evals/` (environment names are passed verbatim — no aliases):

```bash
./run_eval.sh hello --env eu
```

Run every suite under `evals/` (excluding `*_local.yaml` variants) against one environment:

```bash
./run_all_evals.sh --env eu -v
```

A `.env` file in the project root is picked up automatically, so both scripts work from a clean shell.

## Writing Your Own Evals

Each suite file contains:

- Optional `globals` for shared defaults.
- A list of `evals`, each with its own agent payload, message, and expectations.
- `variables` entries map placeholder names (e.g., `api_token`) to sources (`env:API_TOKEN`).

Placeholders use double braces (`{{placeholder}}`) anywhere inside the payloads. Variable sources can be `env:VAR_NAME` to pull from the environment or raw strings for literals.

### Minimal Example

```yaml
evals:
  - name: hello_world
    agent:
      name: Example Agent
      systemPrompt: |
        You respond with a polite acknowledgement.
    message:
      message:
        role: user
        kind: message
        parts:
          - kind: text
            text: Hello there!
    expectations:
      must_include:
        - hello
```

Save the file (e.g., `evals/hello.yaml`) and run `uv run agent-evals run evals/hello.yaml --env eu`.

### Expectations

Every eval case (and every step in a sequential case) accepts an `expectations` block. All fields are optional.

| Field | Type | Description |
|---|---|---|
| `must_include` | `list[str]` | Substrings that must appear in the full JSON response. |
| `must_not_include` | `list[str]` | Substrings that must not appear. |
| `max_duration_seconds` | `float` | Fail if the round-trip takes longer than this. |
| `expected_output_text` | `str` | Reference answer — an LLM judge on Corti Models decides semantic equivalence. Authenticated with the Corti Console's `CORTI_*` credentials (or a pre-built `JUDGE_API_KEY`); see `example.env`. |
| `jsonpath` | `list[Assertion]` | JSONPath assertions against the response's `kind:"data"` parts (see below). |
| `must_include_json` | `list[Any]` | Partial JSON fragments that must match at least one `kind:"data"` part (see below). |
| `expected_state` | `str` | Exact task state the response must end in (e.g. `rejected`, `input-required`). When unset, a task ending in `failed` or `rejected` fails the eval even if all content checks pass. In a `sequential` step, `expected_state: input-required` also carries the response's `taskId` forward so the next step continues the same task. |

Expectation values must not reference internal task-metadata keys (`expertId`, `_agent`, `_meter`, `opik_*`, ...): substring checks run over the full serialized task JSON, and those fields are internal to the agent service and change between versions.

#### `jsonpath` assertions

Each entry is a mapping with a `path` (JSONPath expression) and one or more comparators. All comparators in one entry are AND-ed. An assertion passes if it holds for *at least one* data part.

| Comparator | Description |
|---|---|
| `equals` | All matched values equal this (scalars) or the list equals this exactly. |
| `contains` | Every item in this list must appear among the matched values. |
| `length` | Collection-aware: if the match is a single list/string, checks its length; otherwise checks the number of matched nodes. |
| `count` | Raw count of matched nodes. |
| `min` / `max` | All matched numeric values must be ≥ min / ≤ max. |
| `regex` | All matched strings must satisfy this regular expression. |
| `exists` | `true` / `false` — asserts presence or absence of any match. |

```yaml
expectations:
  jsonpath:
    - { path: "$.items", length: 4 }
    - { path: "$.items[*].type", contains: ["TEMP", "SYS"] }
    - { path: "$.items[*].value", min: 0, max: 300 }
    - { path: "$.items[*].timestamp", regex: "^\\d{4}-\\d{2}-\\d{2}T" }
```

#### `must_include_json`

Declares partial JSON fragments that must appear in at least one `kind:"data"` response part. Useful for asserting the shape and values of structured data without enumerating every field.

Matching rules:
- **Dict** — subset match: every declared key must exist in the actual object and match recursively. Extra keys in the actual object are ignored.
- **List** — order-independent: every declared element must have at least one matching counterpart in the actual list (partial match applied recursively).
- **Scalar** — type-sensitive equality (`38.5 ≠ "38.5"`, `true ≠ 1`).

```yaml
expectations:
  must_include_json:
    - items:
        - { type: "TEMP",  value: 38.5, unitName: "°C"  }
        - { type: "SYS",   value: 120,  unitName: "mmHg" }
        - { type: "DIAS",  value: 80,   unitName: "mmHg" }
        - { type: "PULSE", value: 90,   unitName: "Bpm"  }
```

### Timeouts

`timeout_seconds` imposes a wall-clock bound on an entire eval — agent creation, every step, and any `delay_before_seconds` sleeps. When the bound is exceeded the eval is recorded as failed with a typed Harness Failure (`harness_error` code `eval_timeout`) and the suite moves on to the next eval instead of blocking. Unset means no timeout.

The value must be greater than the HTTP request timeout (125s: 5s connect + 120s read). A smaller bound could never interrupt a single in-flight request, so suites declaring one are rejected at load time.

Set it per eval, or under `globals` as a suite-wide default (a per-eval value overrides the global one):

```yaml
globals:
  timeout_seconds: 300   # default for every eval in the suite
evals:
  - name: slow_case
    timeout_seconds: 600 # this eval gets a larger budget
    ...
```

Notes:
- This differs from the `max_duration_seconds` expectation, which only checks the measured duration *after* the eval completes — it never unblocks a stuck run.
- Python threads cannot be killed, so the abandoned worker may keep running in the background after a timeout; its in-flight HTTP request is still bounded by the client's per-request read timeout.

### Reusing an Existing Agent

If you already have an agent provisioned, add an `agent_id` field to the case. When present, the harness skips agent creation and calls the send-message endpoint directly. Placeholders continue to resolve via suite variables, so the identifier can come from the environment or globals.

```yaml
evals:
  - name: reuse_existing
    agent_id: {{ agent_identifier }}
    message:
      message:
        role: user
        kind: message
        parts:
          - kind: text
            text: Hello there!
```

With this configuration you can omit the `agent` section entirely. If both are present, the explicit `agent_id` takes precedence for the message step.

### Sequential Evaluations

Set `type: sequential` on a case to chain multiple sub-evals that share the same agent. The harness will create (or reuse) the agent once, run each step in order, and inject the `contextId` from the first response into every subsequent message. If a step is expected to return `status: input-required`, assert `expected_state: input-required` in its `expectations` so its `taskId` is carried forward to the next message.

```yaml
evals:
  - name: intake_follow_up
    type: sequential
    agent:
      name: IntakeAgent
      systemPrompt: Gather intake details.
    steps:
      - name: initial_prompt
        message:
          message:
            parts:
              - kind: text
                text: "I need to schedule a check-up."
        expectations:
          expected_state: input-required
          must_include:
            - '"state": "input-required"'
      - name: provide_details
        message:
          message:
            parts:
              - kind: text
                text: "Here are the details you asked for."
        expectations:
          must_not_include:
            - error
```

Sequential results include a step-by-step breakdown in both the JSON and Markdown outputs, including the actual request payload sent (with propagated identifiers) and the response for each step.
Each step is also emitted as its own entry in the JSON array (`is_step_result: true`) so downstream tooling can chart or aggregate them individually.

## CLI Reference

- `uv run agent-evals run <suite>` – Execute the evaluations in a suite file (or every suite in a directory).
  - `--env <name>` – Environment to run against. Required.
  - `--output <path>` – JSON file to store results (a sibling `.md` file is emitted automatically).
  - `--stop-on-failure` – Stop after the first failing evaluation.
  - `--concurrency <n>` – Maximum number of evals to run in parallel.
  - `--name <eval>` – Run only evals matching this name (repeatable).
  - `--opik` – Also record the run in Opik (needs a reachable Opik; see [Opik Evals](#opik-evals-where-results-go)).
  - `--dataset-name <name>` / `--experiment-name <name>` – Override the Opik dataset/experiment names (only with `--opik`).
  - `-v/--verbose` – Increase logging (repeat for debug).
  - `-q/--quiet` – Reduce logging (repeat to silence).

- `uv run agent-evals show <suite>` – List the evaluation case names in a suite.
- `uv run agent-evals --show <suite>` – Alternate shorthand for the same listing.

`uv run agent-evals run --help` prints the full configuration inventory (every environment variable, its default, and its effect).

## Result Format

The JSON output file contains an array of result objects. Each case entry carries its Step Trail under `step_results`; every step records its per-term expectation verdicts and, separately, at most one typed Harness Failure:

```json
[
  {
    "name": "painkiller_lookup",
    "success": true,
    "agent_id": "...",
    "duration_seconds": 3.142,
    "step_results": [
      {
        "name": null,
        "success": true,
        "request": { "...": "..." },
        "response": { "...": "..." },
        "harness_error": null,
        "duration_seconds": 3.141,
        "context_id": "...",
        "usage": { "input_tokens": 26514, "output_tokens": 3319, "credits": 0.15916 },
        "expectation_results": [
          {
            "key": "must_include",
            "checks": [
              { "label": "ibuprofen", "passed": true, "detail": "passed" }
            ],
            "show_on_pass": false
          }
        ]
      }
    ]
  }
]
```

`expectation_results` holds one entry per expectation (keyed by its YAML field, e.g. `must_include`), each with its per-term checks (`{label, passed, detail}`). A failed check is the one and only record of a miss — the markdown `Errors:` line and the CLI failure log are derived from these, so there is no separate error list to reconcile.

`harness_error` is `null` unless the harness itself gave out before the step could be evaluated — a timeout, network error, HTTP error, or unexpected exception. Then it holds exactly one typed Harness Failure (`{code, message, context}`) on an aborted step entry with no request/response and an empty `expectation_results`, so infrastructure flakiness stays machine-distinguishable from a failed check.

`usage` reports the request's LLM usage/credits accounting, extracted from the task's `_usage` and `credits` metadata (documented public extensions of the agent service's wire contract). For sequential (multi-step) evals, each step entry carries its own `usage` and the case entry carries the field-wise sum across steps. Fields the deployment doesn't report are `null`; the key is `null` when no accounting was returned at all.

`duration_seconds` records the total case runtime, enabling assertions such as:

```yaml
expectations:
  max_duration_seconds: 5
```

If provided, the run fails when the case takes longer than the configured threshold.

## Opik Evals: Where Results Go

Pass `--opik` to record a run in Opik alongside the local results files. Both outputs describe exactly one execution — the same runner, the same cases, the same results. The local `results/*.json|md` files are always written; `--opik` is opt-in.

`--opik` writes go to the Opik instance named by `OPIK_URL_OVERRIDE` (set via `make setup`), into the **`Agents` project** (override with the `OPIK_PROJECT_NAME` env var) — the same project agent-api logs its traces to. `OPIK_URL_OVERRIDE` is the primary `--opik` workflow: point it at the Opik you want writes to go to — a local instance, a team-hosted one, or a CI box — and the run health-checks it first, failing fast if nothing answers there. Drop `--opik` for offline development; the local result files are written either way.

The eval run logs **no agent traces or spans of its own** — agent-api already traces every interaction it serves, and each result row's `trace_url` links to those (an unconfigured checkout omits trace links). Token usage (`input_tokens` / `output_tokens` / `credits`) rides the task output's `usage` column instead, summed across steps for sequential cases.

The official sweep script (`run_all_evals.sh`) bakes `--opik` in; the single-suite dev script (`run_eval.sh`) stays bare so local iteration works on any machine.

### Gotchas

- **Shared instance, shared datasets.** The eval clears a suite's dataset before inserting, so two people running the same suite concurrently against the same Opik overwrite each other's rows. Coordinate for now; per-user dataset suffixing or content-hash dedup is a possible follow-up.

## Suite Layout

Suites live under `evals/` (create the directory in your checkout), organized by topic. Files ending in `_local.yaml` are local-stack variants excluded from directory runs and `run_all_evals.sh` sweeps; files starting with `_` (e.g. `_defaults.yaml`) are shared fragments, not runnable suites.

## Development

Run `uv run agent-evals run --help` to explore CLI options during development. Contributions can add richer expectations, multi-step conversations, or additional output sinks.
