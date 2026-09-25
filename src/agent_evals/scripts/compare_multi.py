"""Compare N eval runs side by side and generate an HTML report.

Unlike ``compare_experiments`` (2-sided) and ``generate_report`` (2-sided HTML),
this script handles 3+ runs in a single pass, showing every run's score
per case/suite with deltas relative to a chosen baseline.

Usage::

    uv run python -m agent_evals.scripts.compare_multi \\
        --tag dev-default-20260909-120000 --label "Default (dev)" \\
        --tag dev-cort-s1-instant-20260909-120000 --label "cort-s1-instant" \\
        --tag dev-corti-s1-20260909-120000 --label "corti-s1" \\
        -o report.html

The first ``--tag/--label`` pair is the baseline; all deltas are computed
relative to it. Use ``--baseline N`` to pick a different baseline by index
(0-based).
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import dotenv

from .data_source import DataSource, make_source

_REPO_ROOT = Path(__file__).resolve().parents[3]
dotenv.load_dotenv(_REPO_ROOT / ".env")


# --- experiment discovery ----------------------------------------------------

def _find_by_tag(source: DataSource, tag: str, label: str, limit: int = 5000) -> dict[str, SimpleNamespace]:
    exps = source.list_experiments(name=None, tag=tag, limit=limit)
    if not exps:
        raise SystemExit(f"{label}: no experiments found for tag={tag!r}")
    return dict(exps)


def _find_by_tags(source: DataSource, tags: list[str], label: str, limit: int = 5000) -> dict[str, SimpleNamespace]:
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


# --- multi-way comparison ---------------------------------------------------

def build_multi_comparison(
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
    all_suite_names = sorted(set().union(*(set(s) for s in sides)))
    matched_names = sorted(set.intersection(*(set(s) for s in sides)))

    rows: list[SimpleNamespace] = []
    suite_summaries: list[SimpleNamespace] = []

    for exp_name in matched_names:
        exps = [s[exp_name] for s in sides]
        items = [source.get_items(e) for e in exps]
        by_names = [{source.case_name(it): it for it in its} for its in items]

        all_case_names = sorted(set.intersection(*(set(bn) for bn in by_names)))

        suite_cases: list[SimpleNamespace] = []
        credits_totals = [0.0] * n

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
                vals.append(v)
                reasons.append(r)
                traces.append(source.trace_url(it) if it else None)
                all_scores_list.append({k: v2[0] for k, v2 in sm.items()})
                credits_totals[i] += _credits(it) if it else 0.0

            row = SimpleNamespace(
                exp_name=exp_name,
                case_name=case_name,
                vals=vals,
                reasons=reasons,
                traces=traces,
                all_scores_list=all_scores_list,
                credits=[_credits(it) if it else 0.0 for it in its],
                exp_ids=[e.id for e in exps],
            )
            rows.append(row)
            suite_cases.append(row)

        scored = [r for r in suite_cases if all(v is not None for v in r.vals)]
        if scored:
            means = [sum(r.vals[i] for r in scored) / len(scored) for i in range(n)]
            suite_summaries.append(SimpleNamespace(
                name=exp_name,
                means=means,
                n=len(scored),
                credits=credits_totals,
                exp_ids=[e.id for e in exps],
            ))

    only_sets = [set(sides[i]) - set.intersection(*(set(sides[j]) for j in range(n) if j != i)) for i in range(n)]
    return rows, suite_summaries, only_sets


# --- HTML generation ---------------------------------------------------------

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


_HTML_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Multi-Run Eval Report</title>
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
.callout { padding: 0.5rem 0.8rem; border-radius: 4px; margin: 0.3rem 0; font-size: 0.85rem; }
.callout.red { background: var(--callout-r); border-left: 3px solid var(--red); }
.callout.amber { background: var(--callout-a); border-left: 3px solid var(--amber); }
.callout.green { background: var(--callout-g); border-left: 3px solid var(--green); }
.footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border); }
.footer code { font-size: 0.75rem; color: var(--faint); }
.theme-toggle { position: fixed; top: 1rem; right: 1rem; cursor: pointer;
                 background: var(--card); border: 1px solid var(--border);
                 padding: 0.3rem 0.6rem; border-radius: 4px; font-size: 1rem; z-index: 100; }
.winner-cell { font-weight: 700; color: var(--green); }
.rank-table { border-collapse: collapse; width: 100%; font-size: 0.9rem; margin: 0.5rem 0; }
.rank-table th, .rank-table td { padding: 0.3rem 0.5rem; border-bottom: 1px solid var(--border); }
.rank-table .rank-1 { background: var(--badge-g); }
.rank-table .rank-2 { background: var(--badge-n); }
.rank-table .rank-3 { background: var(--badge-r); }
details:target { scroll-margin-top: 1rem; }
details:target > summary { font-weight: 700; }
a[href^="#"] { color: var(--link); text-decoration: none; }
a[href^="#"]:hover { text-decoration: underline; }
.suite-regressed { border-left: 3px solid var(--red); }
.suite-improved { border-left: 3px solid var(--green); }
.suite-unchanged { border-left: 3px solid var(--neutral); }
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
.sortable { cursor: pointer; user-select: none; }
.sortable:hover { color: var(--text); }
.sortable::after { content: ""; font-size: 0.75rem; margin-left: 0.2rem; opacity: 0.4; }
.sortable.sort-asc::after { content: " \\25b2"; opacity: 1; }
.sortable.sort-desc::after { content: " \\25bc"; opacity: 1; }
</style>
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
  document.querySelectorAll(".tab-panel.active details").forEach(function(d) { d.open = open; });
}
function switchTab(name) {
  document.querySelectorAll(".tab-panel").forEach(function(p) { p.classList.remove("active"); });
  document.querySelectorAll(".tab-bar label").forEach(function(l) { l.style.color = ""; l.style.fontWeight = ""; });
  document.getElementById("tab-" + name).classList.add("active");
  var label = document.querySelector('label[for="rt-' + name + '"]');
  if (label) { label.style.color = "var(--text)"; label.style.fontWeight = "700"; }
  applySort();
}
var currentSort = {col: 0, asc: false};
function sortSuites(col) {
  if (currentSort.col === col) { currentSort.asc = !currentSort.asc; }
  else { currentSort.col = col; currentSort.asc = false; }
  document.querySelectorAll(".sortable").forEach(function(th) {
    th.classList.remove("sort-asc", "sort-desc");
  });
  document.querySelectorAll('th[data-col="' + col + '"]').forEach(function(th) {
    th.classList.add(currentSort.asc ? "sort-asc" : "sort-desc");
  });
  applySort();
}
function applySort() {
  var col = currentSort.col, asc = currentSort.asc;
  ["deltas", "credits", "cases"].forEach(function(tabName) {
    var panel = document.getElementById("tab-" + tabName);
    if (!panel) return;
    var containers = panel.querySelectorAll("table tbody, .cases-container");
    containers.forEach(function(container) {
      var rows = Array.from(container.children).filter(function(r) {
        return r.tagName === "TR" || r.tagName === "DETAILS";
      });
      var totals = rows.filter(function(r) { return r.classList.contains("totals"); });
      var dataRows = rows.filter(function(r) { return !r.classList.contains("totals"); });
      dataRows.sort(function(a, b) {
        var av = a.getAttribute("data-sort-" + col);
        var bv = b.getAttribute("data-sort-" + col);
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
</script>
"""

_HTML_TAIL = """\
</body>
</html>
"""


def generate_multi_html(
    rows: list[SimpleNamespace],
    suite_summaries: list[SimpleNamespace],
    only_sets: list[set[str]],
    labels: list[str],
    tags_list: list[list[str]],
    score: str,
    baseline_idx: int,
    insights_html: str = "",
) -> str:
    n = len(labels)
    bl = labels[baseline_idx]

    scored = [r for r in rows if all(v is not None for v in r.vals)]
    total = len(scored)

    overall_means = []
    for i in range(n):
        vals = [r.vals[i] for r in scored]
        overall_means.append(sum(vals) / len(vals) if vals else 0)

    total_credits = [sum(s.credits[i] for s in suite_summaries) for i in range(n)]

    # Per-run win counts (cases where this run has the highest score)
    win_counts = [0] * n
    tie_count = 0
    for r in scored:
        max_val = max(r.vals)
        winners = [i for i, v in enumerate(r.vals) if v == max_val and v is not None]
        if len(winners) == 1:
            win_counts[winners[0]] += 1
        elif len(winners) > 1:
            tie_count += 1

    parts: list[str] = []
    parts.append(_HTML_HEAD)

    # Title
    parts.append(f"<h1>Multi-Run Eval Report: {n} runs compared</h1>")
    parts.append(f'<p class="meta">Score: <code>{_esc(score)}</code> \u00b7 '
                 f"Compared {total} cases across {len(suite_summaries)} suites \u00b7 "
                 f"Baseline: <code>{_esc(bl)}</code></p>")

    # Summary line: mean score for each run
    parts.append('<div class="summary-line">')
    for i in range(n):
        delta = overall_means[i] - overall_means[baseline_idx] if i != baseline_idx else None
        color = _delta_color(delta) if delta is not None else "neutral"
        parts.append(f'<span class="num {color}">{overall_means[i]:.3f}</span> <span class="label">{_esc(labels[i])}'
                     + (f' ({_fmt_pp(delta)})' if delta is not None else '') + '</span>')
        if i < n - 1:
            parts.append(' \u00b7 ')
    parts.append('</div>')

    # Credits summary
    parts.append('<div class="summary-line" style="margin-top:0.5rem">')
    parts.append('<span class="label">Credits:</span>')
    for i in range(n):
        delta_c = total_credits[i] - total_credits[baseline_idx] if i != baseline_idx else None
        color = _delta_color(-delta_c) if delta_c is not None else "neutral"
        parts.append(f'<span class="num {color}">{total_credits[i]:.4f}</span> <span class="label">{_esc(labels[i])}'
                     + (f' ({delta_c:+.4f})' if delta_c is not None else '') + '</span>')
        if i < n - 1:
            parts.append(' \u00b7 ')
    parts.append('</div>')

    # Win counts
    parts.append('<div class="summary-line" style="margin-top:0.5rem">')
    parts.append('<span class="label">Win counts (highest score per case):</span>')
    for i in range(n):
        parts.append(f'<span class="num green">{win_counts[i]}</span> <span class="label">{_esc(labels[i])}</span>')
        if i < n - 1:
            parts.append(' \u00b7 ')
    if tie_count:
        parts.append(f' \u00b7 <span class="num neutral">{tie_count}</span> <span class="label">ties</span>')
    parts.append(f' \u00b7 <span class="label">{total} total scored cases</span>')
    parts.append('</div>')

    # Controls
    parts.append('<div class="controls">')
    parts.append('<button class="btn" onclick="toggleAll(true)">Expand all</button> ')
    parts.append('<button class="btn" onclick="toggleAll(false)">Collapse all</button>')
    parts.append('</div>')

    # Only-in sets
    for i in range(n):
        if only_sets[i]:
            parts.append(f'<p class="meta"><strong>Only in {_esc(labels[i])}:</strong> {", ".join(sorted(only_sets[i]))}</p>')

    # Insights section (open by default, author-written)
    if insights_html:
        parts.append('<details open id="insights">')
        parts.append('<summary><strong>Summary &amp; Insights</strong></summary>')
        parts.append(f'<div class="insights">{insights_html}</div>')
        parts.append('</details>')

    # --- Ranking table ---
    parts.append('<h2 id="rankings">Overall Rankings</h2>')
    ranked = sorted(range(n), key=lambda i: -overall_means[i])
    parts.append('<table class="rank-table"><thead><tr><th>Rank</th><th>Run</th><th>Mean Score</th><th>Credits</th><th>Wins</th>')
    for i in range(n):
        if i != baseline_idx:
            parts.append(f'<th>\u0394 vs {_esc(labels[i])}</th>')
    parts.append('</tr></thead><tbody>')
    for rank, idx in enumerate(ranked, 1):
        parts.append(f'<tr class="rank-{rank}"><td>{rank}</td><td>{_esc(labels[idx])}</td>'
                     f'<td class="num-cell">{overall_means[idx]:.3f}</td>'
                     f'<td class="num-cell">{total_credits[idx]:.4f}</td>'
                     f'<td class="num-cell">{win_counts[idx]}</td>')
        for i in range(n):
            if i != baseline_idx:
                d = overall_means[idx] - overall_means[i]
                parts.append(f'<td class="num-cell {_delta_color(d)}">{_fmt_pp(d)}</td>')
        parts.append('</tr>')
    parts.append('</tbody></table>')

    # --- Tabbed section: Suite Deltas | Credits | Cases ---
    parts.append('<h2 id="suite-breakdown">Suite Breakdown</h2>')
    parts.append('<div class="tab-bar">')
    parts.append('<input type="radio" name="suite-tabs" id="rt-deltas" checked onchange="switchTab(\'deltas\')"/>')
    parts.append('<label for="rt-deltas">Suite Deltas</label>')
    parts.append('<input type="radio" name="suite-tabs" id="rt-credits" onchange="switchTab(\'credits\')"/>')
    parts.append('<label for="rt-credits">Credits</label>')
    parts.append('<input type="radio" name="suite-tabs" id="rt-cases" onchange="switchTab(\'cases\')"/>')
    parts.append('<label for="rt-cases">Case-by-Case</label>')
    parts.append('</div>')

    # --- Tab 1: Suite Deltas ---
    parts.append('<div class="tab-panel active" id="tab-deltas">')
    parts.append('<table class="multi-table"><thead><tr><th class="sortable" data-col="0" onclick="sortSuites(0)">Suite</th>')
    for i in range(n):
        parts.append(f'<th class="sortable" data-col="{i+1}" onclick="sortSuites({i+1})">{_esc(labels[i])}</th>')
    for i in range(n):
        if i != baseline_idx:
            parts.append(f'<th class="sortable" data-col="{n+i}" onclick="sortSuites({n+i})">\u0394 ({_esc(labels[i])})</th>')
    parts.append('</tr></thead><tbody>')
    for s in sorted(suite_summaries, key=lambda s: -s.means[baseline_idx]):
        sort_vals = [s.name]
        sort_vals += [f"{s.means[i]:.6f}" for i in range(n)]
        sort_vals += [f"{s.means[i] - s.means[baseline_idx]:.6f}" for i in range(n) if i != baseline_idx]
        sort_attrs = " ".join(f'data-sort-{i}="{v}"' for i, v in enumerate(sort_vals))
        parts.append(f'<tr {sort_attrs}><td>{_esc(s.name)}</td>')
        for i in range(n):
            parts.append(f'<td class="num-cell">{_fmt_score(s.means[i])}</td>')
        for i in range(n):
            if i != baseline_idx:
                d = s.means[i] - s.means[baseline_idx]
                parts.append(f'<td class="num-cell {_delta_color(d)}">{_fmt_pp(d)}</td>')
        parts.append('</tr>')
    parts.append(f'<tr class="totals"><td>Overall</td>')
    for i in range(n):
        parts.append(f'<td class="num-cell">{overall_means[i]:.3f}</td>')
    for i in range(n):
        if i != baseline_idx:
            d = overall_means[i] - overall_means[baseline_idx]
            parts.append(f'<td class="num-cell {_delta_color(d)}">{_fmt_pp(d)}</td>')
    parts.append('</tr></tbody></table>')
    parts.append('</div>')

    # --- Tab 2: Credits ---
    parts.append('<div class="tab-panel" id="tab-credits">')
    parts.append('<table class="cost-table"><thead><tr><th class="sortable" data-col="0" onclick="sortSuites(0)">Suite</th>')
    for i in range(n):
        parts.append(f'<th class="sortable" data-col="{i+1}" onclick="sortSuites({i+1})">{_esc(labels[i])}</th>')
    parts.append('</tr></thead><tbody>')
    for s in sorted(suite_summaries, key=lambda s: -sum(s.credits)):
        sort_vals = [s.name] + [f"{s.credits[i]:.8f}" for i in range(n)]
        sort_attrs = " ".join(f'data-sort-{i}="{v}"' for i, v in enumerate(sort_vals))
        parts.append(f'<tr {sort_attrs}><td>{_esc(s.name)}</td>')
        for i in range(n):
            parts.append(f'<td class="num-cell">{s.credits[i]:.4f}</td>')
        parts.append('</tr>')
    parts.append(f'<tr class="totals"><td>Total</td>')
    for i in range(n):
        parts.append(f'<td class="num-cell">{total_credits[i]:.4f}</td>')
    parts.append('</tr></tbody></table>')
    parts.append('</div>')

    # --- Tab 3: Case-by-Case ---
    by_suite: dict[str, list[SimpleNamespace]] = defaultdict(list)
    for r in rows:
        by_suite[r.exp_name].append(r)

    parts.append('<div class="tab-panel" id="tab-cases">')
    parts.append('<div class="cases-container">')
    for s in sorted(suite_summaries, key=lambda s: -s.means[baseline_idx]):
        suite_rows = by_suite.get(s.name, [])
        suite_id = f"suite-{s.name}".replace(" ", "-").replace("/", "-").replace(".", "-").lower()
        bl_mean = s.means[baseline_idx]
        best_mean = max(s.means)
        suite_class = "suite-improved" if best_mean > bl_mean + 0.001 else "suite-regressed" if best_mean < bl_mean - 0.001 else "suite-unchanged"

        sort_vals = [s.name] + [f"{s.means[i]:.6f}" for i in range(n)]
        sort_vals += [f"{s.means[i] - s.means[baseline_idx]:.6f}" for i in range(n) if i != baseline_idx]
        sort_attrs = " ".join(f'data-sort-{i}="{v}"' for i, v in enumerate(sort_vals))

        parts.append(f'<details class="{suite_class}" id="{suite_id}" {sort_attrs}>')
        parts.append(f'<summary>{_esc(s.name)} \u00b7 '
                     + " / ".join(f"{_fmt_score(m)}" for m in s.means) +
                     f' \u00b7 {len(suite_rows)} cases</summary>')

        for r in sorted(suite_rows, key=lambda r: -(max(r.vals) - min(r.vals) if all(v is not None for v in r.vals) else 0)):
            case_id = f"case-{s.name}-{r.case_name}".replace(" ", "-").replace("/", "-").replace(".", "-").lower()
            max_val = max(v for v in r.vals if v is not None) if any(v is not None for v in r.vals) else None
            parts.append(f'<details class="case-card" id="{case_id}">')

            # Summary line with badges
            score_strs = []
            for i in range(n):
                v = r.vals[i]
                badge_class = "best" if v == max_val and v is not None else _delta_color(v - r.vals[baseline_idx] if i != baseline_idx and v is not None and r.vals[baseline_idx] is not None else None)
                score_strs.append(f'<span class="badge {badge_class}">{_fmt_score(v)}</span> {_esc(labels[i])}')

            score_summary = " \u00b7 ".join(score_strs)
            parts.append(f'<summary>{score_summary} \u00b7 {_esc(r.case_name)}</summary>')
            parts.append('<div class="case-detail">')

            # Scores table
            all_score_names = sorted(set().union(*(set(sl) for sl in r.all_scores_list)))
            if all_score_names:
                parts.append('<table class="scores-table"><thead><tr><th>Score</th>')
                for i in range(n):
                    parts.append(f'<th>{_esc(labels[i])}</th>')
                parts.append('</tr></thead><tbody>')
                for sn in all_score_names:
                    parts.append(f'<tr><td>{_esc(sn)}</td>')
                    for i in range(n):
                        sv = r.all_scores_list[i].get(sn)
                        parts.append(f'<td class="num-cell">{_fmt_score(sv)}</td>')
                    parts.append('</tr>')
                parts.append('</tbody></table>')

            # Credits
            credits_str = " / ".join(f"{r.credits[i]:.4f}" for i in range(n))
            parts.append(f'<p class="meta">Credits: {credits_str}</p>')

            # Reasons
            has_reasons = any(r.reasons[i] for i in range(n))
            if has_reasons:
                parts.append('<div class="reasons">')
                for i in range(n):
                    if r.reasons[i]:
                        parts.append(f'<p><strong>{_esc(labels[i])}:</strong> <code>{_esc(r.reasons[i][:400])}</code></p>')
                parts.append('</div>')

            # Trace links
            has_traces = any(r.traces[i] for i in range(n))
            if has_traces:
                parts.append('<div class="trace-links">')
                for i in range(n):
                    if r.traces[i]:
                        parts.append(f'<a href="{_esc(r.traces[i])}" target="_blank" class="trace-link">Trace ({_esc(labels[i])}) \u2197</a>')
                parts.append('</div>')

            parts.append('</div></details>')

        parts.append('</details>')

    parts.append('</div>')  # close cases-container
    parts.append('</div>')  # close tab-cases
    parts.append('<div class="footer"><code>Generated by compare_multi \u00b7 '
                 + " vs ".join(_esc(l) for l in labels) +
                 f' \u00b7 score={_esc(score)}</code></div>')

    parts.append(_HTML_TAIL)
    return "\n".join(parts)


# --- main --------------------------------------------------------------------

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
    elif '<h2 id="rankings">' in content:
        content = content.replace('<h2 id="rankings">', insights_block + '\n\n    <h2 id="rankings">', 1)
    else:
        print("Warning: could not find insertion point for insights.", file=sys.stderr)
        return

    report_path.write_text(content, encoding="utf-8")
    print(f"Insights injected into {report_path} ({report_path.stat().st_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare N eval runs side by side and generate an HTML report."
    )
    subparsers = parser.add_subparsers(dest="mode")

    gen = subparsers.add_parser("generate", help="Generate a new multi-run HTML report")
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
        "-o", "--output", default="multi-report.html",
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

    sides: list[dict[str, SimpleNamespace]] = []
    for i in range(n):
        print(f"Discovering experiments for {labels[i]!r} (tags={tags_list[i]!r})...")
        exps = _find_by_tags(source, tags_list[i], labels[i])
        print(f"  found {len(exps)} experiments")
        sides.append(exps)

    matched = sorted(set.intersection(*(set(s) for s in sides)))
    print(f"Matched: {len(matched)} suites")

    print(f"Building multi-way comparison (score={args.score}, baseline={labels[args.baseline]!r})...")
    rows, suite_summaries, only_sets = build_multi_comparison(source, sides, args.score, args.baseline)

    scored = [r for r in rows if all(v is not None for v in r.vals)]
    if scored:
        for i in range(n):
            m = sum(r.vals[i] for r in scored) / len(scored)
            print(f"  {labels[i]}: mean={m:.3f}")
        print(f"  {len(scored)} matched cases across {len(suite_summaries)} suites")

    for i in range(n):
        if only_sets[i]:
            print(f"  Only in {labels[i]}: {', '.join(sorted(only_sets[i]))}")

    print(f"\nGenerating HTML report...")
    insights_html = ""
    if getattr(args, "insights", None):
        ins_path = Path(args.insights)
        if ins_path.exists():
            insights_html = ins_path.read_text(encoding="utf-8")

    html_content = generate_multi_html(rows, suite_summaries, only_sets, labels, tags_list, args.score, args.baseline,
                                        insights_html=insights_html)

    out_path = Path(args.output)
    out_path.write_text(html_content, encoding="utf-8")
    print(f"Report written to {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
