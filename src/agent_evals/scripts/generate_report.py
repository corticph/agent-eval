"""Generate a self-contained HTML eval report comparing N eval runs.

A pairwise comparison is just a special case of multi-run comparison (N=2).
This script handles 2+ runs in a single pass, showing every run's score per
case/suite with deltas relative to a chosen baseline.

Usage (2 runs)::

    uv run python -m agent_evals.scripts.generate_report generate \
        --tag staging-eu-20260910-134632 --label "staging-eu" \
        --tag dev-weu-20260910-134632 --label "dev-weu" \
        -o report.html

Usage (4 runs)::

    uv run python -m agent_evals.scripts.generate_report generate \
        --tag staging-eu-20260910-134632 --label "staging-eu" \
        --tag dev-weu-20260910-134632 --label "dev (default)" \
        --tag dev-weu-corti-s1-20260910-142440 --label "dev (corti-s1)" \
        --tag dev-weu-corti-s1-instant-20260910-142440 --label "dev (instant)" \
        --baseline 0 \
        -o report.html

The first ``--tag/--label`` pair is the baseline by default; all deltas are
computed relative to it. Use ``--baseline N`` to pick a different baseline by
index (0-based).

The two-step workflow (generate + add-insights) lets you inject author-written
analysis with clickable links to specific sections/cases::

    uv run python -m agent_evals.scripts.generate_report add-insights \
        -o report.html --insights insights.html
"""

from __future__ import annotations

import argparse
import html
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import dotenv

from .data_source import DataSource, make_source
from .regression_categories import categorize as _categorize

_REPO_ROOT = Path(__file__).resolve().parents[3]
dotenv.load_dotenv(_REPO_ROOT / ".env")


# --- experiment discovery ----------------------------------------------------

def _find_by_tag(source: DataSource, tag: str, label: str, limit: int = 5000) -> dict[str, SimpleNamespace]:
    exps = source.list_experiments(name=None, tag=tag, limit=limit)
    if not exps:
        raise SystemExit(f"{label}: no experiments found for tag={tag!r}")
    return dict(exps)


def _find_by_tags(source: DataSource, tags: list[str], label: str, limit: int = 5000) -> dict[str, SimpleNamespace]:
    """Merge experiments from multiple tags, taking the newest per suite name."""
    merged: dict[str, SimpleNamespace] = {}
    for tag in tags:
        try:
            found = _find_by_tag(source, tag, label, limit=limit)
        except SystemExit:
            continue
        for name, exp in found.items():
            if name not in merged or (exp.created_at or "") > (merged[name].created_at or ""):
                merged[name] = exp
    if not merged:
        raise SystemExit(f"{label}: no experiments found for tags={tags!r}")
    return merged


# --- per-item helpers --------------------------------------------------------

def _usage(item):
    out = item.evaluation_task_output or {}
    return out.get("usage") or {}


def _credits(item):
    u = _usage(item)
    try:
        return float(u.get("credits") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _duration(item):
    out = item.evaluation_task_output or {}
    try:
        return float(out.get("duration_seconds") or 0.0)
    except (TypeError, ValueError):
        return 0.0


# --- infrastructure-failure detection -----------------------------------------

_INFRA_FAILURE_PATTERNS = (
    "HTTP 401",
    "HTTP 503",
    "timed out",
    "engine is currently busy",
    "Connection refused",
    "Connection failed",
    "could not reach",
)


def _is_infra_failure(reason: str) -> bool:
    """True when the failure reason indicates an infrastructure issue, not a quality issue."""
    r = reason or ""
    return any(pat in r for pat in _INFRA_FAILURE_PATTERNS)


# --- build comparison data ---------------------------------------------------

def build_comparison(
    source: DataSource,
    sides: list[dict[str, SimpleNamespace]],
    score: str = "overall",
    baseline_idx: int = 0,
) -> tuple[list[SimpleNamespace], list[SimpleNamespace], list[set[str]]]:
    """Build comparison rows across N sides.

    Returns (rows, suite_summaries, only_sets) where:
    - rows: per-case rows with scores from every side
    - suite_summaries: per-suite aggregate stats
    - only_sets: list of sets of suite names only present in that side
    """
    n = len(sides)
    matched_names = sorted(set.intersection(*(set(s) for s in sides)))

    rows: list[SimpleNamespace] = []
    suite_summaries: list[SimpleNamespace] = []

    for exp_name in matched_names:
        exps = [s[exp_name] for s in sides]
        items = [source.get_items(e) for e in exps]
        by_names = [{source.case_name(it): it for it in its} for its in items]

        all_case_names = sorted(set.union(*(set(bn) for bn in by_names)))

        suite_cases: list[SimpleNamespace] = []
        credits_totals = [0.0] * n
        duration_totals = [0.0] * n

        for case_name in all_case_names:
            its = [bn.get(case_name) for bn in by_names]
            sms = [source.score_map(it) if it else {} for it in its]

            if not any(score in sm for sm in sms):
                continue

            vals = []
            reasons = []
            traces = []
            all_scores_list = []

            for i, it in enumerate(its):
                sm = sms[i]
                v, r = sm.get(score, (None, ""))
                is_infra = v is not None and v == 0.0 and _is_infra_failure(r)
                if is_infra:
                    v = None
                vals.append(v)
                reasons.append(r)
                traces.append(source.trace_url(it) if it else None)
                all_scores_list.append(
                    {k: (None if is_infra else v2[0]) for k, v2 in sm.items()}
                )
                credits_totals[i] += _credits(it) if it else 0.0
                duration_totals[i] += _duration(it) if it else 0.0

            if not any(v is not None for v in vals):
                continue

            row = SimpleNamespace(
                exp_name=exp_name,
                case_name=case_name,
                vals=vals,
                reasons=reasons,
                traces=traces,
                all_scores_list=all_scores_list,
                credits=[_credits(it) if it else 0.0 for it in its],
                durations=[_duration(it) if it else 0.0 for it in its],
                exp_ids=[e.id for e in exps],
            )
            rows.append(row)
            suite_cases.append(row)

        scored = [r for r in suite_cases if all(v is not None for v in r.vals)]
        if scored:
            means = [sum(r.vals[i] for r in scored) / len(scored) for i in range(n)]
            regressed = 0
            improved = 0
            unchanged = 0
            for r in scored:
                bl_val = r.vals[baseline_idx]
                any_worse = any(
                    r.vals[i] < bl_val - 0.001
                    for i in range(n) if i != baseline_idx and r.vals[i] is not None
                )
                any_better = any(
                    r.vals[i] > bl_val + 0.001
                    for i in range(n) if i != baseline_idx and r.vals[i] is not None
                )
                if any_worse:
                    regressed += 1
                elif any_better:
                    improved += 1
                else:
                    unchanged += 1
            suite_summaries.append(SimpleNamespace(
                name=exp_name,
                means=means,
                n=len(scored),
                credits=credits_totals,
                durations=duration_totals,
                exp_ids=[e.id for e in exps],
                regressed=regressed,
                improved=improved,
                unchanged=unchanged,
            ))

    if n == 1:
        only_sets = [set()]
    else:
        only_sets = [
            set(sides[i]) - set.union(*(set(sides[j]) for j in range(n) if j != i))
            for i in range(n)
        ]
    return rows, suite_summaries, only_sets


# --- HTML helpers ------------------------------------------------------------

def _esc(text):
    return html.escape(str(text)) if text is not None else ""


def _fmt_score(v):
    return f"{v:.3f}" if v is not None else "\u2014"


def _fmt_delta(d):
    if d is None:
        return "\u2014"
    return f"{d:+.3f}"


def _delta_color(d):
    if d is None:
        return "neutral"
    if d < -0.001:
        return "red"
    if d > 0.001:
        return "green"
    return "neutral"


def _fmt_pp(d):
    if d is None:
        return "\u2014"
    return f"{d*100:+.1f}pp"


def _fmt_duration(seconds):
    if seconds is None or seconds == 0:
        return "0s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m{s:.0f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m)}m"


def _slug(text):
    return text.replace(" ", "-").replace("/", "-").replace(".", "-").lower()


def _worst_delta(r, baseline_idx):
    """Most negative non-baseline delta (or None)."""
    bl = r.vals[baseline_idx]
    deltas = [
        r.vals[i] - bl
        for i in range(len(r.vals))
        if i != baseline_idx and r.vals[i] is not None and bl is not None
    ]
    return min(deltas) if deltas else None


def _best_delta(r, baseline_idx):
    """Most positive non-baseline delta (or None)."""
    bl = r.vals[baseline_idx]
    deltas = [
        r.vals[i] - bl
        for i in range(len(r.vals))
        if i != baseline_idx and r.vals[i] is not None and bl is not None
    ]
    return max(deltas) if deltas else None


def _worst_run_reason(r, baseline_idx):
    """Reason from the worst-scoring non-baseline run."""
    bl = r.vals[baseline_idx]
    worst_i = None
    worst_val = None
    for i in range(len(r.vals)):
        if i != baseline_idx and r.vals[i] is not None and bl is not None:
            if worst_val is None or r.vals[i] < worst_val:
                worst_val = r.vals[i]
                worst_i = i
    return r.reasons[worst_i] if worst_i is not None else ""


def _is_regression(r, baseline_idx):
    """True if any non-baseline run is worse than baseline."""
    bl = r.vals[baseline_idx]
    if bl is None:
        return False
    return any(
        r.vals[i] is not None and r.vals[i] < bl - 0.001
        for i in range(len(r.vals)) if i != baseline_idx
    )


def _is_improvement(r, baseline_idx):
    """True if any non-baseline run is better than baseline (and none worse)."""
    bl = r.vals[baseline_idx]
    if bl is None:
        return False
    any_worse = any(
        r.vals[i] is not None and r.vals[i] < bl - 0.001
        for i in range(len(r.vals)) if i != baseline_idx
    )
    if any_worse:
        return False
    return any(
        r.vals[i] is not None and r.vals[i] > bl + 0.001
        for i in range(len(r.vals)) if i != baseline_idx
    )


# --- HTML head (CSS + JS) ----------------------------------------------------

_HTML_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eval Report</title>
<style>
:root {
  --bg: #fff; --card: #fff; --text: #1a1a1a; --faint: #888; --border: #e0e0e0;
  --red: #c62828; --green: #2e7d32; --neutral: #666; --amber: #f57f17;
  --trend-r: #ef5350; --trend-i: #66bb6a; --trend-u: #e0e0e0;
  --badge-r: #ffcdd2; --badge-g: #c8e6c9; --badge-n: #f5f5f5; --badge-a: #fff8e1;
  --code-bg: #f8f8f8; --link: #1976d2;
  --callout-r: #ffebee; --callout-a: #fff8e1; --callout-g: #e8f5e9;
}
[data-theme="dark"] {
  --bg: #1a1a2e; --card: #16213e; --text: #e0e0e0; --faint: #888; --border: #333;
  --red: #ef5350; --green: #66bb6a; --neutral: #aaa; --amber: #ffb74d;
  --trend-r: #ef5350; --trend-i: #66bb6a; --trend-u: #333;
  --badge-r: #4a1a1a; --badge-g: #1a3a1a; --badge-n: #2a2a2a; --badge-a: #3a3520;
  --code-bg: #0d1117; --link: #64b5f6;
  --callout-r: #3a1a1a; --callout-a: #3a3520; --callout-g: #1a3a1a;
}
body { font-family: -apple-system, system-ui, sans-serif; max-width: 1100px;
       margin: 3rem auto; padding: 0 1rem; line-height: 1.5;
       background: var(--bg); color: var(--text); }
h1 { font-size: 1.4rem; }
h2 { font-size: 1.1rem; margin-top: 2rem; }
h3 { font-size: 0.95rem; margin: 0.5rem 0; }
.meta { color: var(--faint); font-size: 0.85rem; }
.label { color: var(--faint); font-size: 0.8rem; }
.num { font-weight: 600; font-size: 0.95rem; }
.num.red { color: var(--red); } .num.green { color: var(--green); } .num.neutral { color: var(--neutral); }
.summary-line { display: flex; flex-wrap: wrap; gap: 0.3rem; align-items: baseline; margin-top: 0.5rem; }
.trend-bar-group { display: flex; align-items: center; gap: 0.5rem; margin: 0.3rem 0; }
.trend-bar-label { font-size: 0.8rem; color: var(--faint); white-space: nowrap; min-width: 100px; }
.trend-bar { flex: 1; display: flex; height: 8px; border-radius: 4px; overflow: hidden; }
.trend-regressed { background: var(--trend-r); }
.trend-improved { background: var(--trend-i); }
.trend-unchanged { background: var(--trend-u); }
.controls { margin: 0.5rem 0; }
.btn { cursor: pointer; background: var(--card); border: 1px solid var(--border);
       padding: 0.3rem 0.8rem; border-radius: 4px; font-size: 0.85rem; color: var(--text); }
details { margin: 0.3rem 0; padding: 0.3rem 0.5rem; border-radius: 4px;
          border: 1px solid var(--border); background: var(--card); }
details[open] { border-color: #aaa; }
summary { cursor: pointer; font-weight: 500; list-style: none; }
summary::-webkit-details-marker { display: none; }
summary::before { content: "\\25b8 "; color: var(--faint); }
details[open] > summary::before { content: "\\25be "; }
.badge { display: inline-block; padding: 0.1rem 0.4rem; border-radius: 3px;
          font-size: 0.75rem; font-weight: 600; font-family: monospace; }
.badge.red { background: var(--badge-r); color: var(--red); }
.badge.green { background: var(--badge-g); color: var(--green); }
.badge.neutral { background: var(--badge-n); color: var(--neutral); }
.badge.amber { background: var(--badge-a); color: var(--amber); }
.badge.best { background: var(--badge-g); color: var(--green); border: 1px solid var(--green); }
.case-card { margin-left: 1rem; border: 1px solid var(--border); }
.case-detail { padding: 0.5rem 0; }
.scores-table, .cost-table, .multi-table { border-collapse: collapse; width: 100%; font-size: 0.85rem; margin: 0.3rem 0; }
.scores-table th, .cost-table th, .multi-table th { text-align: left; padding: 0.2rem 0.4rem; border-bottom: 1px solid var(--border); color: var(--faint); font-weight: 500; }
.scores-table td, .cost-table td, .multi-table td { padding: 0.2rem 0.4rem; }
.num-cell { text-align: right; font-family: monospace; }
.totals { font-weight: 600; border-top: 2px solid var(--border); }
.reasons { margin: 0.3rem 0; }
.reasons code { display: block; font-size: 0.8rem; background: var(--code-bg); padding: 0.3rem;
                border-radius: 3px; white-space: pre-wrap; word-break: break-word; color: var(--text); }
.trace-links { margin: 0.3rem 0; }
.trace-link { display: inline-block; margin-right: 0.5rem; padding: 0.2rem 0.5rem;
              background: var(--code-bg); border: 1px solid var(--border); border-radius: 3px;
              text-decoration: none; font-size: 0.8rem; color: var(--link); }
.cmds { margin: 0.3rem 0; }
.cmds code { display: block; font-size: 0.75rem; background: var(--code-bg); padding: 0.2rem;
             border-radius: 3px; word-break: break-all; color: var(--text); }
.callout { padding: 0.5rem 0.8rem; border-radius: 4px; margin: 0.3rem 0; font-size: 0.85rem; }
.callout.red { background: var(--callout-r); border-left: 3px solid var(--red); }
.callout.amber { background: var(--callout-a); border-left: 3px solid var(--amber); }
.callout.green { background: var(--callout-g); border-left: 3px solid var(--green); }
.root-cause { margin: 0.5rem 0; }
.footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border); }
.footer code { font-size: 0.75rem; color: var(--faint); }
.theme-toggle { position: fixed; top: 1rem; right: 1rem; cursor: pointer;
                 background: var(--card); border: 1px solid var(--border);
                 padding: 0.3rem 0.6rem; border-radius: 4px; font-size: 1rem; z-index: 100; }
.run-diff { font-size: 0.85rem; }
.run-diff ul { margin: 0.3rem 0; padding-left: 1.2rem; }
.run-diff li { margin: 0.1rem 0; }
.insights { padding: 0.8rem 0; font-size: 0.92rem; }
.insights p { margin: 0.6rem 0; line-height: 1.65; }
.insights ul { margin: 0.4rem 0 0.6rem 1.2rem; padding-left: 0; }
.insights li { margin: 0.3rem 0; line-height: 1.55; }
.insights h3 { font-size: 0.95rem; margin: 1.2rem 0 0.4rem; padding-top: 0.6rem; border-top: 1px solid var(--border); }
.insights h3:first-child { border-top: none; padding-top: 0; }
.insights strong { font-weight: 600; }
.insights code { background: var(--code-bg); padding: 0.1rem 0.25rem; border-radius: 3px; font-size: 0.82rem; }
.insights .red { color: var(--red); } .insights .amber { color: var(--amber); } .insights .green { color: var(--green); }
.suite-regressed { border-left: 3px solid var(--red); }
.suite-improved { border-left: 3px solid var(--green); }
.suite-unchanged { border-left: 3px solid var(--neutral); }
details:target { scroll-margin-top: 1rem; }
details:target > summary { font-weight: 700; }
a[href^="#"] { color: var(--link); text-decoration: none; }
a[href^="#"]:hover { text-decoration: underline; }
.rank-table { border-collapse: collapse; width: 100%; font-size: 0.9rem; margin: 0.5rem 0; }
.rank-table th, .rank-table td { padding: 0.3rem 0.5rem; border-bottom: 1px solid var(--border); }
.rank-table .rank-1 { background: var(--badge-g); }
.rank-table .rank-2 { background: var(--badge-n); }
.rank-table .rank-3 { background: var(--badge-r); }
.tab-bar { display: flex; gap: 0; border-bottom: 2px solid var(--border); margin: 1rem 0 0.5rem; }
.tab-bar input[type="radio"] { display: none; }
.tab-bar label { padding: 0.4rem 1rem; cursor: pointer; border: 1px solid var(--border);
  border-bottom: none; border-radius: 4px 4px 0 0; font-size: 0.85rem; font-weight: 500;
  color: var(--faint); background: var(--card); margin-bottom: -2px; }
.tab-bar input[type="radio"]:checked + label { color: var(--text); border-color: var(--border);
  border-bottom: 2px solid var(--bg); font-weight: 700; }
.tab-bar label:hover { color: var(--text); }
.tab-panel { display: none; }
.tab-panel.active { display: block; }
.rca-panel { display: none; }
.rca-panel.active { display: block; }
.imp-panel { display: none; }
.imp-panel.active { display: block; }
.sortable { cursor: pointer; user-select: none; }
.sortable:hover { color: var(--text); }
.sortable::after { content: ""; font-size: 0.75rem; margin-left: 0.2rem; opacity: 0.4; }
.sortable.sort-asc::after { content: " \\25b2"; opacity: 1; }
.sortable.sort-desc::after { content: " \\25bc"; opacity: 1; }
</style>
<noscript><style>
.tab-panel { display: block; }
.tab-bar { display: none; }
.controls { display: none; }
#filter-toggle { display: none; }
</style></noscript>
</head>
<body>
<button class="theme-toggle" onclick="toggleTheme()"></button>
<script>
function toggleTheme() {
  var cur = document.documentElement.getAttribute("data-theme");
  var next = cur === "dark" ? "" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("theme", next);
}
(function() {
  var saved = localStorage.getItem("theme");
  if (saved) document.documentElement.setAttribute("data-theme", saved);
})();
function toggleAll(open) {
  document.querySelectorAll("details").forEach(function(d) { d.open = open; });
}
function switchTab(name) {
  document.querySelectorAll(".tab-panel").forEach(function(p) { p.classList.remove("active"); });
  document.querySelectorAll(".tab-bar label").forEach(function(l) { l.style.color = ""; l.style.fontWeight = ""; });
  document.getElementById("tab-" + name).classList.add("active");
  var label = document.querySelector('label[for="rt-' + name + '"]');
  if (label) { label.style.color = "var(--text)"; label.style.fontWeight = "700"; }
  updateSortIndicators(name);
  applySort();
}
var currentSort = {col: 0, asc: false, tab: "deltas"};
function sortSuites(col) {
  var tab = currentTab();
  if (currentSort.tab === tab && currentSort.col === col) {
    currentSort.asc = !currentSort.asc;
  } else {
    currentSort.col = col;
    currentSort.asc = false;
    currentSort.tab = tab;
  }
  updateSortIndicators(tab);
  applySort();
}
function currentTab() {
  var a = document.querySelector(".tab-panel.active");
  return a ? a.id.replace("tab-", "") : "deltas";
}
function updateSortIndicators(tab) {
  document.querySelectorAll(".sortable").forEach(function(th) {
    th.classList.remove("sort-asc", "sort-desc");
  });
  document.querySelectorAll('#tab-' + tab + ' th[data-col="' + currentSort.col + '"]').forEach(function(th) {
    th.classList.add(currentSort.asc ? "sort-asc" : "sort-desc");
  });
}
function applySort() {
  var col = currentSort.col, asc = currentSort.asc, sortTab = currentSort.tab;
  var attr = "data-sort-" + sortTab + "-" + col;
  ["deltas", "credits", "time", "cases"].forEach(function(tabName) {
    var panel = document.getElementById("tab-" + tabName);
    if (!panel) return;
    var containers = panel.querySelectorAll("table.multi-table tbody, table.cost-table tbody, .cases-container");
    containers.forEach(function(container) {
      var rows = Array.from(container.children).filter(function(r) {
        return r.tagName === "TR" || r.tagName === "DETAILS";
      });
      var totals = rows.filter(function(r) { return r.classList.contains("totals"); });
      var dataRows = rows.filter(function(r) { return !r.classList.contains("totals"); });
      dataRows.sort(function(a, b) {
        var av = a.getAttribute(attr);
        var bv = b.getAttribute(attr);
        if (av === null) return 1;
        if (bv === null) return -1;
        var an = parseFloat(av), bn = parseFloat(bv);
        if (isNaN(an) && isNaN(bn)) return asc ? a.textContent.localeCompare(b.textContent) : b.textContent.localeCompare(a.textContent);
        if (isNaN(an)) return 1;
        if (isNaN(bn)) return -1;
        return asc ? an - bn : bn - an;
      });
      dataRows.forEach(function(r) { container.appendChild(r); });
      totals.forEach(function(r) { container.appendChild(r); });
    });
  });
}
(function() { sortSuites(0); })();

function switchRcaTab(idx) {
  document.querySelectorAll(".rca-panel").forEach(function(p) { p.classList.remove("active"); });
  document.querySelectorAll('input[name="rca-tabs"]').forEach(function(l) { l.checked = false; });
  document.querySelectorAll('.rca-tab-label').forEach(function(l) { l.style.color = ""; l.style.fontWeight = ""; });
  var panel = document.getElementById("rca-panel-" + idx);
  if (panel) panel.classList.add("active");
  var radio = document.getElementById("rca-rt-" + idx);
  if (radio) radio.checked = true;
  var label = document.querySelector('label[for="rca-rt-' + idx + '"]');
  if (label) { label.style.color = "var(--text)"; label.style.fontWeight = "700"; }
}
function switchImpTab(idx) {
  document.querySelectorAll(".imp-panel").forEach(function(p) { p.classList.remove("active"); });
  document.querySelectorAll('input[name="imp-tabs"]').forEach(function(l) { l.checked = false; });
  document.querySelectorAll('.imp-tab-label').forEach(function(l) { l.style.color = ""; l.style.fontWeight = ""; });
  var panel = document.getElementById("imp-panel-" + idx);
  if (panel) panel.classList.add("active");
  var radio = document.getElementById("imp-rt-" + idx);
  if (radio) radio.checked = true;
  var label = document.querySelector('label[for="imp-rt-' + idx + '"]');
  if (label) { label.style.color = "var(--text)"; label.style.fontWeight = "700"; }
}

function openAnchorTarget(id) {
  var el = document.getElementById(id);
  if (!el) return false;
  if (el.tagName === "DETAILS") el.open = true;
  var parent = el.parentElement;
  while (parent) {
    if (parent.tagName === "DETAILS") parent.open = true;
    parent = parent.parentElement;
  }
  var panel = el.closest(".tab-panel");
  if (panel) {
    var tabName = panel.id.replace("tab-", "");
    switchTab(tabName);
  }
  var rcaPanel = el.closest(".rca-panel");
  if (rcaPanel) {
    var rcaIdx = rcaPanel.id.replace("rca-panel-", "");
    switchRcaTab(rcaIdx);
  }
  var impPanel = el.closest(".imp-panel");
  if (impPanel) {
    var impIdx = impPanel.id.replace("imp-panel-", "");
    switchImpTab(impIdx);
  }
  el.scrollIntoView({behavior: "smooth", block: "start"});
  return true;
}

document.addEventListener("click", function(e) {
  var a = e.target.closest("a[href^='#']");
  if (!a) return;
  var id = a.getAttribute("href").slice(1);
  if (!id) return;
  if (openAnchorTarget(id)) {
    e.preventDefault();
    history.replaceState(null, "", "#" + id);
  }
});

window.addEventListener("hashchange", function() {
  var id = location.hash.slice(1);
  if (id) openAnchorTarget(id);
});

if (location.hash) {
  var id = location.hash.slice(1);
  if (id) setTimeout(function() { openAnchorTarget(id); }, 100);
}

function toggleFullScoreFilter() {
  var cb = document.getElementById("filter-fullscore");
  var filtered = cb.checked;
  document.querySelectorAll(".view-all").forEach(function(el) { el.style.display = filtered ? "none" : ""; });
  document.querySelectorAll(".view-filtered").forEach(function(el) { el.style.display = filtered ? "" : "none"; });
  var cases = document.querySelectorAll("#tab-cases .case-card");
  cases.forEach(function(c) {
    c.style.display = (filtered && !c.classList.contains("fullscore")) ? "none" : "";
  });
  var suites = document.querySelectorAll("#tab-cases .cases-container > details");
  suites.forEach(function(s) {
    var visible = Array.from(s.querySelectorAll(".case-card")).filter(function(c) { return c.style.display !== "none"; });
    s.style.display = visible.length > 0 ? "" : "none";
  });
  applySort();
}
</script>
"""

_HTML_TAIL = """\
</body>
</html>
"""


# --- case card renderer ------------------------------------------------------

def _render_case_card(parts, r, labels, baseline_idx, case_id, badge_delta=None, suite_name=None):
    """Render a case card showing all N runs."""
    n = len(labels)

    if badge_delta is not None:
        badge_cls = _delta_color(badge_delta)
        badge_html = f'<span class="badge {badge_cls}">{_fmt_delta(badge_delta)}</span>'
    else:
        badge_html = '<span class="badge neutral">\u2014</span>'

    summary_text = badge_html + f' {_esc(r.case_name)}'
    if suite_name:
        summary_text += f' <span class="meta">({_esc(suite_name)})</span>'

    parts.append(f'<details class="case-card" id="{case_id}">')
    parts.append(f'<summary>{summary_text}</summary>')
    parts.append('<div class="case-detail">')

    all_score_names = sorted(set().union(*(set(sl) for sl in r.all_scores_list)))
    if all_score_names:
        parts.append('<table class="scores-table"><thead><tr><th>Score</th>')
        for i in range(n):
            parts.append(f'<th>{_esc(labels[i])}</th>')
        parts.append('</tr></thead><tbody>')
        for sn in all_score_names:
            parts.append(f'<tr><td>{_esc(sn)}</td>')
            for i in range(n):
                sv = r.all_scores_list[i].get(sn) if i < len(r.all_scores_list) else None
                parts.append(f'<td class="num-cell">{_fmt_score(sv)}</td>')
            parts.append('</tr>')
        parts.append('</tbody></table>')

    credits_str = " / ".join(f"{r.credits[i]:.4f}" for i in range(n))
    time_str = " / ".join(_fmt_duration(r.durations[i]) for i in range(n))
    parts.append(f'<p class="meta">Credits: {credits_str} \u00b7 Time: {time_str}</p>')

    has_reasons = any(r.reasons[i] for i in range(n))
    if has_reasons:
        parts.append('<div class="reasons">')
        for i in range(n):
            if r.reasons[i]:
                parts.append(f'<p><strong>{_esc(labels[i])}:</strong> <code>{_esc(r.reasons[i][:500])}</code></p>')
        parts.append('</div>')

    has_traces = any(r.traces[i] for i in range(n))
    if has_traces:
        parts.append('<div class="trace-links">')
        for i in range(n):
            if r.traces[i]:
                parts.append(f'<a href="{_esc(r.traces[i])}" target="_blank" class="trace-link">Trace ({_esc(labels[i])}) \u2197</a>')
        parts.append('</div>')

    parts.append('<div class="cmds">')
    for i in range(n):
        if r.exp_ids[i]:
            parts.append(f'<code>uv run python -m agent_evals.scripts.inspect_eval --exp {r.exp_ids[i]} --case "{_esc(r.case_name)}"</code><br>')
            parts.append(f'<code>uv run python -m agent_evals.scripts.fetch_traces --exp {r.exp_ids[i]} --case "{_esc(r.case_name)}"</code>')
            if i < n - 1:
                parts.append('<br>')
    parts.append('</div>')

    parts.append('</div></details>')


# --- main HTML renderer ------------------------------------------------------

def generate_report_html(
    rows: list[SimpleNamespace],
    suite_summaries: list[SimpleNamespace],
    only_sets: list[set[str]],
    labels: list[str],
    tags_list: list[list[str]],
    score: str,
    baseline_idx: int,
    insights_html: str = "",
    run_differences_list: list[str] | None = None,
) -> str:
    n = len(labels)
    bl = baseline_idx
    if run_differences_list is None:
        run_differences_list = [""] * n

    # --- Compute aggregates for ALL rows (per-model means over their full support) ---
    support_counts = [sum(1 for r in rows if r.vals[i] is not None) for i in range(n)]
    all_means = []
    for i in range(n):
        vals = [r.vals[i] for r in rows if r.vals[i] is not None]
        all_means.append(sum(vals) / len(vals) if vals else 0)

    # --- Compute aggregates for FULLY-SCORED rows only (fair comparison) ---
    scored = [r for r in rows if all(v is not None for v in r.vals)]
    total = len(scored)
    total_cases = len(rows)
    overall_means = []
    for i in range(n):
        vals = [r.vals[i] for r in scored]
        overall_means.append(sum(vals) / len(vals) if vals else 0)

    # Regression/improvement/unchanged counts (fully-scored)
    regressed_count = 0
    improved_count = 0
    unchanged_count = 0
    for r in scored:
        if _is_regression(r, bl):
            regressed_count += 1
        elif _is_improvement(r, bl):
            improved_count += 1
        else:
            unchanged_count += 1

    total_credits = [sum(r.credits[i] for r in rows) for i in range(n)]
    total_durations = [sum(r.durations[i] for r in rows) for i in range(n)]

    # Win counts (fully-scored)
    win_counts = [0] * n
    tie_count = 0
    for r in scored:
        max_val = max(r.vals)
        winners = [i for i, v in enumerate(r.vals) if v == max_val and v is not None]
        if len(winners) == 1:
            win_counts[winners[0]] += 1
        elif len(winners) > 1:
            tie_count += 1

    # --- Per-suite filtered summaries (fully-scored only) ---
    by_suite_filtered: dict[str, list[SimpleNamespace]] = defaultdict(list)
    for r in scored:
        by_suite_filtered[r.exp_name].append(r)
    filtered_suite_summaries: list[SimpleNamespace] = []
    for exp_name, suite_rows in by_suite_filtered.items():
        f_means = [sum(r.vals[i] for r in suite_rows) / len(suite_rows) for i in range(n)]
        f_credits = [sum(r.credits[i] for r in suite_rows) for i in range(n)]
        f_durations = [sum(r.durations[i] for r in suite_rows) for i in range(n)]
        f_reg = sum(1 for r in suite_rows if _is_regression(r, bl))
        f_imp = sum(1 for r in suite_rows if _is_improvement(r, bl))
        f_unch = len(suite_rows) - f_reg - f_imp
        filtered_suite_summaries.append(SimpleNamespace(
            name=exp_name, means=f_means, n=len(suite_rows),
            credits=f_credits, durations=f_durations,
            exp_ids=suite_rows[0].exp_ids,
            regressed=f_reg, improved=f_imp, unchanged=f_unch,
        ))
    filtered_suite_summaries.sort(key=lambda s: s.name)

    # --- Per-suite all summaries (each model over own support) ---
    by_suite_all: dict[str, list[SimpleNamespace]] = defaultdict(list)
    for r in rows:
        by_suite_all[r.exp_name].append(r)
    all_suite_summaries: list[SimpleNamespace] = []
    for exp_name, suite_rows in by_suite_all.items():
        a_means = []
        a_credits = [0.0] * n
        a_durations = [0.0] * n
        for i in range(n):
            vals_i = [r.vals[i] for r in suite_rows if r.vals[i] is not None]
            a_means.append(sum(vals_i) / len(vals_i) if vals_i else 0)
            a_credits[i] = sum(r.credits[i] for r in suite_rows)
            a_durations[i] = sum(r.durations[i] for r in suite_rows)
        f_scored = [r for r in suite_rows if all(v is not None for v in r.vals)]
        a_reg = sum(1 for r in f_scored if _is_regression(r, bl))
        a_imp = sum(1 for r in f_scored if _is_improvement(r, bl))
        a_unch = len(f_scored) - a_reg - a_imp
        all_suite_summaries.append(SimpleNamespace(
            name=exp_name, means=a_means, n=len(suite_rows),
            credits=a_credits, durations=a_durations,
            exp_ids=suite_rows[0].exp_ids,
            regressed=a_reg, improved=a_imp, unchanged=a_unch,
        ))
    all_suite_summaries.sort(key=lambda s: s.name)

    f_total_credits = [sum(s.credits[i] for s in filtered_suite_summaries) for i in range(n)]
    f_total_durations = [sum(s.durations[i] for s in filtered_suite_summaries) for i in range(n)]

    # --- Build HTML ---
    parts: list[str] = []
    parts.append(_HTML_HEAD)

    # Title
    if n == 2:
        parts.append(f"<h1>Eval Report: {_esc(labels[1 - bl])} vs {_esc(labels[bl])}</h1>")
    else:
        parts.append(f"<h1>Eval Report: {n} runs compared</h1>")
    parts.append(f'<p class="meta">Score: <code>{_esc(score)}</code> \u00b7 '
                 f"Compared {total_cases} cases across {len(suite_summaries)} suites "
                 f"({total} fully scored, {total_cases - total} with infra-failure gaps) \u00b7 "
                 f"Baseline: <code>{_esc(labels[bl])}</code></p>")

    # --- Trend bars (one per non-baseline run) ---
    if total > 0:
        for i in range(n):
            if i == bl:
                continue
            i_reg = sum(1 for r in scored if r.vals[i] is not None and r.vals[bl] is not None and r.vals[i] < r.vals[bl] - 0.001)
            i_imp = sum(1 for r in scored if r.vals[i] is not None and r.vals[bl] is not None and r.vals[i] > r.vals[bl] + 0.001)
            i_unch = total - i_reg - i_imp
            r_pct = i_reg / total * 100
            i_pct = i_imp / total * 100
            u_pct = i_unch / total * 100
            parts.append('<div class="trend-bar-group">')
            parts.append(f'<span class="trend-bar-label">{_esc(labels[i])}</span>')
            parts.append(f'<div class="trend-bar"><div class="trend-regressed" style="width:{r_pct:.1f}%"></div>'
                         f'<div class="trend-improved" style="width:{i_pct:.1f}%"></div>'
                         f'<div class="trend-unchanged" style="width:{u_pct:.1f}%"></div></div>')
            parts.append('</div>')

    # --- Filter toggle ---
    parts.append('<div id="filter-toggle" style="margin:0.5rem 0 1rem 0"><label class="meta" style="cursor:pointer">'
                 '<input type="checkbox" id="filter-fullscore" onchange="toggleFullScoreFilter()" style="margin-right:0.3rem"/>'
                 '<strong>Show only fully-scored cases</strong> (all runs have quality data \u2014 '
                 f'{total} of {total_cases} cases)'
                 '</label></div>')

    # --- Controls ---
    parts.append('<div class="controls">')
    parts.append('<button class="btn" onclick="toggleAll(true)">Expand all</button> ')
    parts.append('<button class="btn" onclick="toggleAll(false)">Collapse all</button>')
    parts.append('</div>')

    # --- Only-in sets ---
    has_only = any(only_sets[i] for i in range(n))
    if has_only:
        parts.append('<details id="only-in"><summary class="meta">Run info</summary>')
        for i in range(n):
            if only_sets[i]:
                parts.append(f'<p class="meta"><strong>Only in {_esc(labels[i])}:</strong> {", ".join(sorted(only_sets[i]))}</p>')
        parts.append('</details>')

    # --- Insights ---
    if insights_html:
        parts.append('<details open id="insights">')
        parts.append('<summary><strong>Summary &amp; Insights</strong></summary>')
        parts.append(f'<div class="insights">{insights_html}</div>')
        parts.append('</details>')

    # --- Run info (collapsible, one section with per-run sub-headings) ---
    has_run_info = any(i < len(run_differences_list) and run_differences_list[i] for i in range(n))
    if has_run_info:
        parts.append('<details open id="run-info">')
        parts.append('<summary><strong>Run Info</strong></summary>')
        for i in range(n):
            if i < len(run_differences_list) and run_differences_list[i]:
                parts.append(f'<h3>{_esc(labels[i])}</h3>')
                parts.append(f'<div class="run-diff">{run_differences_list[i]}</div>')
        parts.append('</details>')

    # --- Rankings table ---
    def _rankings_table(means, credits, durations, cases_counts, case_total):
        ranked = sorted(range(n), key=lambda i: -means[i])
        p = ['<table class="rank-table"><thead><tr><th>Rank</th><th>Run</th><th>Mean Score</th><th>Cases</th><th>Credits</th><th>Time</th>'
             f'<th title="Wins among {case_total} fully-scored cases (fair comparison only)">Wins</th>']
        for i in range(n):
            if i != bl:
                p.append(f'<th>\u0394 vs {_esc(labels[i])}</th>')
        p.append('</tr></thead><tbody>')
        for rank, idx in enumerate(ranked, 1):
            p.append(f'<tr class="rank-{rank}"><td>{rank}</td><td>{_esc(labels[idx])}</td>'
                     f'<td class="num-cell">{means[idx]:.3f}</td>'
                     f'<td class="num-cell">{cases_counts[idx]}/{case_total}</td>'
                     f'<td class="num-cell">{credits[idx]:.4f}</td>'
                     f'<td class="num-cell">{_fmt_duration(durations[idx])}</td>'
                     f'<td class="num-cell" title="{win_counts[idx]} wins among {case_total} fully-scored cases">{win_counts[idx]}</td>')
            for i in range(n):
                if i != bl:
                    d = means[idx] - means[i]
                    p.append(f'<td class="num-cell {_delta_color(d)}">{_fmt_pp(d)}</td>')
            p.append('</tr>')
        p.append('</tbody></table>')
        return "".join(p)

    parts.append('<h2 id="rankings">Overall Rankings</h2>')
    parts.append('<div class="view-all">')
    parts.append(_rankings_table(all_means, total_credits, total_durations, support_counts, total_cases))
    parts.append('</div>')
    parts.append('<div class="view-filtered" style="display:none">')
    f_support = [total] * n
    parts.append(_rankings_table(overall_means, f_total_credits, f_total_durations, f_support, total))
    parts.append('</div>')

    # --- Root Cause Analysis (per non-baseline run) ---
    non_bl_indices = [i for i in range(n) if i != bl]

    parts.append('<h2 id="root-cause-analysis">Root Cause Analysis</h2>')

    # Build per-run regression data
    rca_data: list[tuple[int, list[tuple]]] = []  # (run_idx, sorted_categories)
    total_regressions = 0
    for i in non_bl_indices:
        run_reg_rows = [
            r for r in rows
            if r.vals[i] is not None and r.vals[bl] is not None
            and r.vals[i] < r.vals[bl] - 0.001
        ]
        total_regressions += len(run_reg_rows)
        cats: dict[str, tuple] = {}
        for r in run_reg_rows:
            reason = r.reasons[i]
            cat_key, cat_title, cat_color, cat_desc = _categorize(reason)
            if cat_key not in cats:
                cats[cat_key] = (cat_title, cat_color, cat_desc, [])
            cats[cat_key][3].append(r)
        sorted_cats_i = sorted(cats.items(), key=lambda x: -len(x[1][3]))
        rca_data.append((i, sorted_cats_i))

    # Tab bar (only if 2+ non-baseline runs)
    if len(non_bl_indices) > 1:
        parts.append('<div class="tab-bar">')
        for idx, i in enumerate(non_bl_indices):
            reg_count = sum(len(c[1][3]) for c in rca_data[idx][1])
            is_first = idx == 0
            parts.append(f'<input type="radio" name="rca-tabs" id="rca-rt-{i}"'
                         f'{" checked" if is_first else ""} onchange="switchRcaTab({i})"/>')
            parts.append(f'<label for="rca-rt-{i}" class="rca-tab-label">'
                         f'{_esc(labels[i])} ({reg_count} regressed)</label>')
        parts.append('</div>')

    # Per-run panels
    for idx, (i, sorted_cats_i) in enumerate(rca_data):
        reg_count = sum(len(c[1][3]) for c in sorted_cats_i)
        active = " active" if (len(non_bl_indices) == 1 or idx == 0) else ""
        parts.append(f'<div class="rca-panel{active}" id="rca-panel-{i}">')
        parts.append(f'<p class="meta">{reg_count} regressed cases for {_esc(labels[i])} vs {_esc(labels[bl])}, grouped by pattern:</p>')

        for cat_key, (cat_title, cat_color, cat_desc, cat_rows) in sorted_cats_i:
            if not cat_rows:
                continue
            cat_id = f"rca-{i}-{cat_key}"
            parts.append(f'<details class="root-cause" id="{cat_id}">')
            parts.append(f'<summary><span class="badge {cat_color}">{len(cat_rows)}</span> {cat_title}</summary>')
            parts.append(f'<div class="callout {cat_color}">{cat_desc}</div>')

            for r in sorted(cat_rows, key=lambda r: (r.vals[i] - r.vals[bl]) if r.vals[i] is not None and r.vals[bl] is not None else 0):
                case_id = f"case-{i}-{cat_key}-{_slug(r.case_name)}"
                delta = r.vals[i] - r.vals[bl] if r.vals[i] is not None and r.vals[bl] is not None else None
                _render_case_card(parts, r, labels, bl, case_id, badge_delta=delta, suite_name=r.exp_name)

            parts.append('</details>')
        parts.append('</div>')

    # --- Improvements (per non-baseline run) ---
    parts.append('<h2 id="improvements">Improvements</h2>')

    # Build per-run improvement data
    imp_data: list[tuple[int, list[SimpleNamespace]]] = []
    for i in non_bl_indices:
        run_imp_rows = [
            r for r in rows
            if r.vals[i] is not None and r.vals[bl] is not None
            and r.vals[i] > r.vals[bl] + 0.001
        ]
        run_imp_rows.sort(key=lambda r: r.vals[i] - r.vals[bl], reverse=True)
        imp_data.append((i, run_imp_rows))

    has_improvements = any(len(rows_i) > 0 for _, rows_i in imp_data)
    if has_improvements:
        # Tab bar (only if 2+ non-baseline runs)
        if len(non_bl_indices) > 1:
            parts.append('<div class="tab-bar">')
            for idx, i in enumerate(non_bl_indices):
                imp_count = len(imp_data[idx][1])
                is_first = idx == 0
                parts.append(f'<input type="radio" name="imp-tabs" id="imp-rt-{i}"'
                             f'{" checked" if is_first else ""} onchange="switchImpTab({i})"/>')
                parts.append(f'<label for="imp-rt-{i}" class="imp-tab-label">'
                             f'{_esc(labels[i])} ({imp_count} improved)</label>')
            parts.append('</div>')

        for idx, (i, run_imp_rows) in enumerate(imp_data):
            active = " active" if (len(non_bl_indices) == 1 or idx == 0) else ""
            parts.append(f'<div class="imp-panel{active}" id="imp-panel-{i}">')
            parts.append(f'<p class="meta">{len(run_imp_rows)} cases improved for {_esc(labels[i])} vs {_esc(labels[bl])}:</p>')
            parts.append('<details class="root-cause" id="all-improvements"><summary><span class="badge green">'
                         f'{len(run_imp_rows)}</span> All improvements</summary>')
            for r in run_imp_rows:
                imp_id = f"imp-{i}-{_slug(r.case_name)}"
                delta = r.vals[i] - r.vals[bl]
                _render_case_card(parts, r, labels, bl, imp_id, badge_delta=delta, suite_name=r.exp_name)
            parts.append('</details>')
            parts.append('</div>')

    # --- Tabbed section: Suite Deltas | Credits | Time | Case-by-Case ---
    parts.append('<h2 id="suite-breakdown">Suite Breakdown</h2>')
    parts.append('<div class="tab-bar">')
    parts.append('<input type="radio" name="suite-tabs" id="rt-deltas" checked onchange="switchTab(\'deltas\')"/>')
    parts.append('<label for="rt-deltas">Suite Deltas</label>')
    parts.append('<input type="radio" name="suite-tabs" id="rt-credits" onchange="switchTab(\'credits\')"/>')
    parts.append('<label for="rt-credits">Credits</label>')
    parts.append('<input type="radio" name="suite-tabs" id="rt-time" onchange="switchTab(\'time\')"/>')
    parts.append('<label for="rt-time">Time</label>')
    parts.append('<input type="radio" name="suite-tabs" id="rt-cases" onchange="switchTab(\'cases\')"/>')
    parts.append('<label for="rt-cases">Case-by-Case</label>')
    parts.append('</div>')

    # --- Tab 1: Suite Deltas (dual) ---
    def _suite_deltas_table(summaries, o_means):
        p = ['<table class="multi-table"><thead><tr><th class="sortable" data-col="0" onclick="sortSuites(0)">Suite</th>']
        for i in range(n):
            p.append(f'<th class="sortable" data-col="{i+1}" onclick="sortSuites({i+1})">{_esc(labels[i])}</th>')
        for i in range(n):
            if i != bl:
                p.append(f'<th class="sortable" data-col="{n+i}" onclick="sortSuites({n+i})">\u0394 ({_esc(labels[i])})</th>')
        p.append('</tr></thead><tbody>')
        for s in sorted(summaries, key=lambda s: -s.means[bl]):
            d_sort = [s.name] + [f"{s.means[i]:.6f}" for i in range(n)] + [f"{s.means[i] - s.means[bl]:.6f}" for i in range(n) if i != bl]
            c_sort = [s.name] + [f"{s.credits[i]:.8f}" for i in range(n)]
            t_sort = [s.name] + [f"{s.durations[i]:.8f}" for i in range(n)]
            sort_attrs = ""
            for i, v in enumerate(d_sort):
                sort_attrs += f' data-sort-deltas-{i}="{v}"'
            for i, v in enumerate(c_sort):
                sort_attrs += f' data-sort-credits-{i}="{v}"'
            for i, v in enumerate(t_sort):
                sort_attrs += f' data-sort-time-{i}="{v}"'
            p.append(f'<tr {sort_attrs}><td>{_esc(s.name)}</td>')
            for i in range(n):
                p.append(f'<td class="num-cell">{_fmt_score(s.means[i])}</td>')
            for i in range(n):
                if i != bl:
                    d = s.means[i] - s.means[bl]
                    p.append(f'<td class="num-cell {_delta_color(d)}">{_fmt_pp(d)}</td>')
            p.append('</tr>')
        p.append('<tr class="totals"><td>Overall</td>')
        for i in range(n):
            p.append(f'<td class="num-cell">{o_means[i]:.3f}</td>')
        for i in range(n):
            if i != bl:
                d = o_means[i] - o_means[bl]
                p.append(f'<td class="num-cell {_delta_color(d)}">{_fmt_pp(d)}</td>')
        p.append('</tr></tbody></table>')
        return "".join(p)

    parts.append('<div class="tab-panel active" id="tab-deltas">')
    parts.append('<div class="view-all">')
    parts.append(_suite_deltas_table(all_suite_summaries, all_means))
    parts.append('</div>')
    parts.append('<div class="view-filtered" style="display:none">')
    parts.append(_suite_deltas_table(filtered_suite_summaries, overall_means))
    parts.append('</div>')
    parts.append('</div>')

    # --- Tab 2: Credits (dual) ---
    def _credits_table(summaries, t_credits):
        p = ['<table class="cost-table"><thead><tr><th class="sortable" data-col="0" onclick="sortSuites(0)">Suite</th>']
        for i in range(n):
            p.append(f'<th class="sortable" data-col="{i+1}" onclick="sortSuites({i+1})">{_esc(labels[i])}</th>')
        p.append('</tr></thead><tbody>')
        for s in sorted(summaries, key=lambda s: -sum(s.credits)):
            d_sort = [s.name] + [f"{s.means[i]:.6f}" for i in range(n)] + [f"{s.means[i] - s.means[bl]:.6f}" for i in range(n) if i != bl]
            c_sort = [s.name] + [f"{s.credits[i]:.8f}" for i in range(n)]
            t_sort = [s.name] + [f"{s.durations[i]:.8f}" for i in range(n)]
            sort_attrs = ""
            for i, v in enumerate(d_sort):
                sort_attrs += f' data-sort-deltas-{i}="{v}"'
            for i, v in enumerate(c_sort):
                sort_attrs += f' data-sort-credits-{i}="{v}"'
            for i, v in enumerate(t_sort):
                sort_attrs += f' data-sort-time-{i}="{v}"'
            p.append(f'<tr {sort_attrs}><td>{_esc(s.name)}</td>')
            for i in range(n):
                p.append(f'<td class="num-cell">{s.credits[i]:.4f}</td>')
            p.append('</tr>')
        p.append('<tr class="totals"><td>Total</td>')
        for i in range(n):
            p.append(f'<td class="num-cell">{t_credits[i]:.4f}</td>')
        p.append('</tr></tbody></table>')
        return "".join(p)

    parts.append('<div class="tab-panel" id="tab-credits">')
    parts.append('<div class="view-all">')
    parts.append(_credits_table(all_suite_summaries, total_credits))
    parts.append('</div>')
    parts.append('<div class="view-filtered" style="display:none">')
    parts.append(_credits_table(filtered_suite_summaries, f_total_credits))
    parts.append('</div>')
    parts.append('</div>')

    # --- Tab 3: Time (dual) ---
    def _time_table(summaries, t_durations):
        p = ['<table class="cost-table"><thead><tr><th class="sortable" data-col="0" onclick="sortSuites(0)">Suite</th>']
        for i in range(n):
            p.append(f'<th class="sortable" data-col="{i+1}" onclick="sortSuites({i+1})">{_esc(labels[i])}</th>')
        p.append('</tr></thead><tbody>')
        for s in sorted(summaries, key=lambda s: -sum(s.durations)):
            d_sort = [s.name] + [f"{s.means[i]:.6f}" for i in range(n)] + [f"{s.means[i] - s.means[bl]:.6f}" for i in range(n) if i != bl]
            c_sort = [s.name] + [f"{s.credits[i]:.8f}" for i in range(n)]
            t_sort = [s.name] + [f"{s.durations[i]:.8f}" for i in range(n)]
            sort_attrs = ""
            for i, v in enumerate(d_sort):
                sort_attrs += f' data-sort-deltas-{i}="{v}"'
            for i, v in enumerate(c_sort):
                sort_attrs += f' data-sort-credits-{i}="{v}"'
            for i, v in enumerate(t_sort):
                sort_attrs += f' data-sort-time-{i}="{v}"'
            p.append(f'<tr {sort_attrs}><td>{_esc(s.name)}</td>')
            for i in range(n):
                p.append(f'<td class="num-cell">{_fmt_duration(s.durations[i])}</td>')
            p.append('</tr>')
        p.append('<tr class="totals"><td>Total</td>')
        for i in range(n):
            p.append(f'<td class="num-cell">{_fmt_duration(t_durations[i])}</td>')
        p.append('</tr></tbody></table>')
        return "".join(p)

    parts.append('<div class="tab-panel" id="tab-time">')
    parts.append('<div class="view-all">')
    parts.append(_time_table(all_suite_summaries, total_durations))
    parts.append('</div>')
    parts.append('<div class="view-filtered" style="display:none">')
    parts.append(_time_table(filtered_suite_summaries, f_total_durations))
    parts.append('</div>')
    parts.append('</div>')

    # --- Tab 4: Case-by-Case ---
    by_suite: dict[str, list[SimpleNamespace]] = defaultdict(list)
    for r in rows:
        by_suite[r.exp_name].append(r)

    filtered_by_name = {s.name: s for s in filtered_suite_summaries}
    all_by_name = {s.name: s for s in all_suite_summaries}

    parts.append('<div class="tab-panel" id="tab-cases">')
    parts.append('<div class="cases-container">')
    for s in sorted(all_suite_summaries, key=lambda s: -s.means[bl]):
        suite_rows = by_suite.get(s.name, [])
        suite_id = f"suite-{_slug(s.name)}"
        bl_mean = s.means[bl]
        best_mean = max(s.means)
        suite_class = "suite-improved" if best_mean > bl_mean + 0.001 else "suite-regressed" if best_mean < bl_mean - 0.001 else "suite-unchanged"

        d_sort = [s.name] + [f"{s.means[i]:.6f}" for i in range(n)] + [f"{s.means[i] - s.means[bl]:.6f}" for i in range(n) if i != bl]
        c_sort = [s.name] + [f"{s.credits[i]:.8f}" for i in range(n)]
        t_sort = [s.name] + [f"{s.durations[i]:.8f}" for i in range(n)]
        sort_attrs = ""
        for i, v in enumerate(d_sort):
            sort_attrs += f' data-sort-deltas-{i}="{v}"'
        for i, v in enumerate(c_sort):
            sort_attrs += f' data-sort-credits-{i}="{v}"'
        for i, v in enumerate(t_sort):
            sort_attrs += f' data-sort-time-{i}="{v}"'

        parts.append(f'<details class="{suite_class}" id="{suite_id}" {sort_attrs}>')

        fs = filtered_by_name.get(s.name)
        aS = all_by_name.get(s.name, s)
        def _case_summary(summ, case_count):
            return (f'{_esc(summ.name)} \u00b7 '
                    + " / ".join(f"{_fmt_score(m)}" for m in summ.means) +
                    f' \u00b7 {case_count} cases \u00b7 '
                    + " / ".join(_fmt_duration(d) for d in summ.durations))

        parts.append(f'<summary>'
                     f'<span class="view-all">{_case_summary(aS, len(suite_rows))}</span>')
        if fs:
            fs_rows = sum(1 for r in suite_rows if all(v is not None for v in r.vals))
            parts.append(f'<span class="view-filtered" style="display:none">{_case_summary(fs, fs_rows)}</span>')
        parts.append('</summary>')

        for r in sorted(suite_rows, key=lambda r: -(max(r.vals) - min(r.vals) if all(v is not None for v in r.vals) else 0)):
            case_id = f"case-{_slug(s.name)}-{_slug(r.case_name)}"
            all_scored = all(v is not None for v in r.vals)
            parts.append(f'<details class="case-card{" fullscore" if all_scored else ""}" id="{case_id}">')

            max_val = max((v for v in r.vals if v is not None), default=None)
            score_strs = []
            for i in range(n):
                v = r.vals[i]
                badge_class = "best" if v == max_val and v is not None else _delta_color(
                    v - r.vals[bl] if i != bl and v is not None and r.vals[bl] is not None else None
                )
                score_strs.append(f'<span class="badge {badge_class}">{_fmt_score(v)}</span> {_esc(labels[i])}')

            score_summary = " \u00b7 ".join(score_strs)
            parts.append(f'<summary>{score_summary} \u00b7 {_esc(r.case_name)}</summary>')
            parts.append('<div class="case-detail">')

            all_score_names = sorted(set().union(*(set(sl) for sl in r.all_scores_list)))
            if all_score_names:
                parts.append('<table class="scores-table"><thead><tr><th>Score</th>')
                for i in range(n):
                    parts.append(f'<th>{_esc(labels[i])}</th>')
                parts.append('</tr></thead><tbody>')
                for sn in all_score_names:
                    parts.append(f'<tr><td>{_esc(sn)}</td>')
                    for i in range(n):
                        sv = r.all_scores_list[i].get(sn) if i < len(r.all_scores_list) else None
                        parts.append(f'<td class="num-cell">{_fmt_score(sv)}</td>')
                    parts.append('</tr>')
                parts.append('</tbody></table>')

            credits_str = " / ".join(f"{r.credits[i]:.4f}" for i in range(n))
            time_str = " / ".join(_fmt_duration(r.durations[i]) for i in range(n))
            parts.append(f'<p class="meta">Credits: {credits_str} \u00b7 Time: {time_str}</p>')

            has_reasons = any(r.reasons[i] for i in range(n))
            if has_reasons:
                parts.append('<div class="reasons">')
                for i in range(n):
                    if r.reasons[i]:
                        parts.append(f'<p><strong>{_esc(labels[i])}:</strong> <code>{_esc(r.reasons[i][:400])}</code></p>')
                parts.append('</div>')

            has_traces = any(r.traces[i] for i in range(n))
            if has_traces:
                parts.append('<div class="trace-links">')
                for i in range(n):
                    if r.traces[i]:
                        parts.append(f'<a href="{_esc(r.traces[i])}" target="_blank" class="trace-link">Trace ({_esc(labels[i])}) \u2197</a>')
                parts.append('</div>')

            parts.append('<div class="cmds">')
            for i in range(n):
                if r.exp_ids[i]:
                    parts.append(f'<code>uv run python -m agent_evals.scripts.inspect_eval --exp {r.exp_ids[i]} --case "{_esc(r.case_name)}"</code><br>')
                    parts.append(f'<code>uv run python -m agent_evals.scripts.fetch_traces --exp {r.exp_ids[i]} --case "{_esc(r.case_name)}"</code>')
                    if i < n - 1:
                        parts.append('<br>')
            parts.append('</div>')

            parts.append('</div></details>')

        parts.append('</details>')

    parts.append('</div>')
    parts.append('</div>')

    # --- Footer ---
    cmd_parts = []
    for i in range(n):
        cmd_parts.append(f"--tag {' '.join(tags_list[i])} --label {labels[i]}")
    parts.append('<div class="footer"><code>Generated by agent-eval \u00b7 generate_report '
                 + ' '.join(cmd_parts) +
                 f' --score {_esc(score)}</code></div>')

    parts.append(_HTML_TAIL)
    return "\n".join(parts)


# --- insights injection ------------------------------------------------------

def _add_insights_to_report(report_path: Path, insights_html: str) -> None:
    """Inject or replace insights HTML in an existing report file."""
    import re
    content = report_path.read_text(encoding="utf-8")

    insights_block = (
        '<details open id="insights">\n'
        '<summary><strong>Summary &amp; Insights</strong></summary>\n'
        f'<div class="insights">\n{insights_html}\n</div>\n'
        '</details>'
    )

    pattern = r'<details open id="insights">.*?</details>'
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, insights_block, content, count=1, flags=re.DOTALL)
    elif '<details open id="run-info">' in content:
        content = content.replace('<details open id="run-info">', insights_block + '\n\n    <details open id="run-info">', 1)
    elif '<h2 id="rankings">' in content:
        content = content.replace('<h2 id="rankings">', insights_block + '\n\n    <h2 id="rankings">', 1)
    else:
        print("Warning: could not find insertion point for insights.", file=sys.stderr)
        return

    report_path.write_text(content, encoding="utf-8")
    print(f"Insights injected into {report_path} ({report_path.stat().st_size} bytes)")


# --- main --------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a self-contained HTML eval report comparing N eval runs."
    )
    subparsers = parser.add_subparsers(dest="mode")

    gen = subparsers.add_parser("generate", help="Generate a new HTML report")
    gen.add_argument(
        "--tag", action="append", nargs="+", required=True,
        help="Tag(s) for a run (repeatable, multiple tags per run are merged). "
             "Each --tag group starts a new run; use --label to name it.",
    )
    gen.add_argument(
        "--label", action="append", required=True,
        help="Label for each run (must match number of --tag groups).",
    )
    gen.add_argument(
        "--score", default="overall",
        help="Feedback score to compare (default: overall).",
    )
    gen.add_argument(
        "-o", "--output", default="report.html",
        help="Output HTML file.",
    )
    gen.add_argument(
        "--baseline", type=int, default=0,
        help="Baseline run index (0-based, default: 0 = first --tag/--label pair).",
    )
    gen.add_argument(
        "--source", default="opik", choices=("opik", "local"),
        help="Data source (default: opik).",
    )
    gen.add_argument(
        "--results-dir", action="append", default=[],
        help="Path to results directory for local source (repeatable).",
    )
    gen.add_argument(
        "--insights", default=None,
        help="Path to an HTML file with author-written insights (optional).",
    )
    gen.add_argument(
        "--run-info", action="append", default=[],
        help="HTML file with run-specific context (one per run, same order as --tag/--label). "
             "Example: --tag A --label \"run A\" --run-info a.html",
    )

    add = subparsers.add_parser("add-insights", help="Inject author-written insights into an existing report")
    add.add_argument("-o", "--output", required=True, help="Path to existing report HTML")
    add.add_argument("--insights", required=True, help="Path to an HTML file with author-written insights")

    # Backwards compat: if no subcommand, treat as generate
    if len(sys.argv) > 1 and sys.argv[1] not in ("generate", "add-insights", "--help", "-h"):
        sys.argv.insert(1, "generate")

    args = parser.parse_args()

    if args.mode == "add-insights":
        report_path = Path(args.output)
        if not report_path.exists():
            raise SystemExit(f"Report file not found: {report_path}")
        insights_path = Path(args.insights)
        if not insights_path.exists():
            raise SystemExit(f"Insights file not found: {insights_path}")
        insights_html = insights_path.read_text(encoding="utf-8")
        _add_insights_to_report(report_path, insights_html)
        return

    if not hasattr(args, "tag"):
        parser.error("Use 'generate' or 'add-insights' subcommand")

    if len(args.tag) != len(args.label):
        parser.error(f"Got {len(args.tag)} --tag groups but {len(args.label)} --label values; they must match.")

    n = len(args.tag)
    if args.baseline < 0 or args.baseline >= n:
        parser.error(f"--baseline {args.baseline} out of range (0..{n-1}).")

    source = make_source(args.source, results_dir=args.results_dir or None)

    labels = args.label
    tags_list = args.tag

    # Load run-info files (pair by position with --tag/--label groups)
    run_differences_list: list[str] = []
    rd_files = args.run_info or []
    if len(rd_files) > n:
        parser.error(
            f"Got {len(rd_files)} --run-info files but only {n} runs. "
            f"Files pair by position with --tag/--label groups:\n"
            + "\n".join(f"  {i}: {labels[i]}" for i in range(n))
        )
    for f in rd_files[:n]:
        p = Path(f)
        if p.exists():
            run_differences_list.append(p.read_text(encoding="utf-8"))
        else:
            print(f"Warning: run-info file not found: {f}", file=sys.stderr)
            run_differences_list.append("")
    while len(run_differences_list) < n:
        run_differences_list.append("")

    sides: list[dict[str, SimpleNamespace]] = []
    for i in range(n):
        print(f"Discovering experiments for {labels[i]!r} (tags={tags_list[i]!r})...")
        exps = _find_by_tags(source, tags_list[i], labels[i])
        print(f"  found {len(exps)} experiments")
        sides.append(exps)

    matched = sorted(set.intersection(*(set(s) for s in sides)))
    print(f"Matched: {len(matched)} suites")

    print(f"Building comparison (score={args.score}, baseline={labels[args.baseline]!r})...")
    rows, suite_summaries, only_sets = build_comparison(source, sides, args.score, args.baseline)

    scored = [r for r in rows if all(v is not None for v in r.vals)]
    if scored:
        for i in range(n):
            m = sum(r.vals[i] for r in scored) / len(scored)
            print(f"  {labels[i]}: mean={m:.3f}")
        r_count = sum(1 for r in scored if _is_regression(r, args.baseline))
        i_count = sum(1 for r in scored if _is_improvement(r, args.baseline))
        u_count = len(scored) - r_count - i_count
        print(f"  {len(scored)} matched cases: {i_count} improved, {r_count} regressed, {u_count} unchanged")

    for i in range(n):
        if only_sets[i]:
            print(f"Only in {labels[i]}: {', '.join(sorted(only_sets[i]))}")

    print(f"\nGenerating HTML report...")
    insights_html = ""
    if getattr(args, "insights", None):
        ins_path = Path(args.insights)
        if ins_path.exists():
            insights_html = ins_path.read_text(encoding="utf-8")

    html_content = generate_report_html(
        rows, suite_summaries, only_sets, labels, tags_list, args.score, args.baseline,
        insights_html=insights_html,
        run_differences_list=run_differences_list,
    )

    out_path = Path(args.output)
    out_path.write_text(html_content, encoding="utf-8")
    print(f"Report written to {out_path} ({out_path.stat().st_size} bytes)")
    print(f"\nTo add insights: uv run python -m agent_evals.scripts.generate_report add-insights -o {out_path} --insights insights.html")


if __name__ == "__main__":
    main()
