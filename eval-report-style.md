# Eval Report Style Guide

Preferences learned for generating self-contained HTML eval reports.

## Theme

- **White theme by default**, with a dark theme toggle (moon/sun icon, fixed top-right).
- Persist preference via `localStorage` so it survives reload.
- Use CSS custom properties (`--bg`, `--card`, `--text`, etc.) with a
  `[data-theme="dark"]` override block — no hardcoded colors in element styles.

## Layout

- **Max width ~920px**, centered, generous padding (`3rem` top/bottom).
- **Compact summary line** — all key numbers inline on one line, not big card
  grids. Example: `0.938 local · 0.963 staging · -2.5pp delta · 20 regressed · 7 improved · 122 unchanged`.
  Numbers bold and colored (red/green/neutral), labels small and faint.
- **Slim trend bar** (8px tall) above the sections to visualize the
  regressed/improved/unchanged proportions.

## Progressive disclosure (avoid text overwhelm)

- Each root cause is a **collapsible section** (white card, badge, title,
  meta). Starts collapsed; click to expand.
- Within each section, **individual cases are nested collapsibles** — click a
  case name to see its detail (response text, failure reason, trace link,
  inspect command).
- **Expand all / Collapse all** buttons at the top for power users.
- The Recommendations/Summary section is open by default; everything else
  starts collapsed.

### Use native `<details>`/`<summary>` — never CSS `display:none` + JS toggle

- **Never** hide content with `display: none` and reveal it via a JS class
  toggle. If JavaScript is disabled or stripped (Slack unfurl, email
  previews, no-JS browsers), the content is permanently invisible.
- **Always** use native `<details>`/`<summary>` elements for collapsibles.
  They work without JavaScript, degrade gracefully, and are fully styleable
  with CSS.
- Hide the default disclosure triangle with
  `summary::-webkit-details-marker { display: none }` and
  `summary { list-style: none }` (or use a custom marker).
- Expand all / Collapse all buttons set `details.open = true/false` via JS —
  a progressive enhancement, not a requirement for content visibility.
- Nested `<details>` (cases inside root-cause sections) work natively and
  remain independently toggleable.

## Content per case

- **Case name** and **score** visible in the collapsed header.
- Expanded detail includes:
  - Short description of what failed and why.
  - The agent's actual response text in a monospace block.
  - Root cause callout (agent bug vs eval brittleness, color-coded).
  - Eval quality note (amber box) when the eval itself is too demanding or
    checking for the wrong things.
  - Opik trace link (clickable).
  - `inspect_eval` command (copy-pasteable).
  - `fetch_traces` command (copy-pasteable) so the reader can pull the
    full OpenInference trace — the reasoning chain, tool calls, and token
    counts — for deeper root-cause analysis.

## Root cause analysis

- Group failures by **root cause pattern**, not by suite. One pattern can span
  multiple suites.
- Label each as **Agent bug** (red), **Mixed** (amber), or **Improved** (green).
- For each pattern, state the root cause explicitly in a callout box.
- **Always inspect both sides** (local + staging) to understand what changed.
- **Fetch traces for both sides** when the root cause isn't obvious from the
  response text alone — the trace shows which tools were called, what
  arguments were used, and whether the tool list or token counts changed
  between environments. See [Fetching OpenInference
  traces](#fetching-openinference-traces) in `AGENTS.md` for commands.
- Flag when an eval is **too demanding** or **too brittle** — the user wants to
  know if the eval itself needs fixing, not just the agent.

## What the user wants to see

- Whether regressions are real agent bugs or eval brittleness.
- Trace links for every failing case (to Opik).
- Inspect commands so they can drill deeper without looking up experiment IDs.
- Improvements, not just regressions — the user wants the full picture.
- A recommendations section at the end: what to fix in the agent, what to fix
  in the evals.

## Trace output (fetch_traces --html)

When embedding or linking trace output in a report, the `fetch_traces --html`
mode follows the same progressive-disclosure rules:

- Each span (`[LLM]`, `[TOOL]`, `[CHAIN]`) is a `<details>` collapsible,
  starting collapsed.
- Within an `[LLM]` span, messages, tool definitions, and the response are
  nested `<details>` sections.
- Messages are deduplicated across LLM calls — repeated context doesn't
  appear. When the tool set hasn't changed, show `(same as above)` instead
  of repeating definitions.
- Default view shows only tool names (not full descriptions); full
  descriptions appear inside the expanded `tools` `<details>`.
- Tool calls and results are truncated to 500 characters in the summary
  view; the full data is available in `--json` mode.

## Footer

- Include a **generated-by footer** at the bottom of the report with the
  command that produced the comparison, so anyone reading the report can
  reproduce it. Use a monospace block, small and faint. Example:

  ```
  Generated by agent-eval · compare_experiments --name <suite-prefix> --tag1 staging --tag2 local-new-base-prompt
  ```
