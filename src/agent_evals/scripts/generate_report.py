"""Generate a self-contained HTML eval report comparing two environments.

This is an example script — adapt the categorization, styling, and layout
to your own eval workflow. The two-step workflow (generate + add-insights)
lets you inject author-written analysis with clickable links to specific
sections/cases.

Usage:
    uv run python -m agent_evals.scripts.generate_report generate \
        --tag1 staging-eu-20260908-181220 --tag2 dev-weu-20260908-181220 \
        --label1 "staging-eu" --label2 "dev-weu (beta)" \
        -o report.html

    uv run python -m agent_evals.scripts.generate_report add-insights \
        -o report.html --insights insights.html
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


# --- categorize regressions --------------------------------------------------

def _categorize(r):
    """Categorize a regression row by root cause pattern.

    Matches on failure reason text from the beta (side2) response. Categories
    are generic infrastructure / agent-behaviour patterns, not tied to specific
    eval suites or product names.
    """
    r2_lower = (r.reason2 or "").lower()
    if "502" in r2_lower or "http 502" in r2_lower:
        return ("infra-502", "Infrastructure: HTTP 502", "amber",
                "The beta agent API returned 502 Bad Gateway during the eval run. "
                "These are transient infrastructure failures, not agent bugs. The cases "
                "scored 0.0 because the agent never responded.")
    if "timed out" in r2_lower and "request" in r2_lower:
        return ("infra-timeout", "Infrastructure: Request timeout (connection failure)", "amber",
                "The request to the beta agent API timed out or the connection failed. "
                "These are transient infrastructure failures, not agent bugs.")
    if "connection failed" in r2_lower or "could not reach" in r2_lower:
        return ("infra-timeout", "Infrastructure: Request timeout (connection failure)", "amber",
                "The request to the beta agent API timed out or the connection failed. "
                "These are transient infrastructure failures, not agent bugs.")
    if "duration" in r2_lower and "exceeded" in r2_lower:
        return ("timeout", "Performance: Eval timeout (max_duration_seconds exceeded)", "amber",
                "The beta agent is slower, exceeding max_duration_seconds. "
                "The answers are often correct \u2014 only the timeout check fails.")
    if "input-required" in r2_lower and "completed" in r2_lower:
        return ("premature-completion", "Agent bug: Premature task completion", "red",
                "The agent completed the task instead of staying in input-required state. "
                "This may indicate a change in the orchestrator's completion signalling.")
    if "forbidden phrase" in r2_lower:
        return ("citation-leak", "Agent bug: Citation markers leaking as forbidden phrases", "red",
                "Internal citation markers appear in the agent's output and match the "
                "eval's forbidden-phrase checks.")
    if "no data parts" in r2_lower:
        return ("no-data-parts", "Agent bug: Response missing data parts", "red",
                "The agent's response has no data parts to evaluate. The response may "
                "be text-only where structured data parts were expected.")
    if "429" in r2_lower:
        return ("rate-limited", "External: API rate-limiting (429)", "amber",
                "An external API returned HTTP 429 (rate limited). The agent gave up "
                "instead of using the results it already had.")
    if "fabricated" in r2_lower or "not present in retrieved sources" in r2_lower:
        return ("fabrication", "Agent bug: Fabricated citations", "red",
                "The agent cites sources not present in the retrieved results.")
    if "judge returned empty" in r2_lower:
        return ("judge-empty", "Eval infra: Judge returned empty content", "amber",
                "The LLM judge returned empty content after 3 attempts. This is a judge "
                "infrastructure issue, not an agent bug or eval correctness issue.")
    if "missing required phrase" in r2_lower or "no match for required pattern" in r2_lower:
        return ("clinical-content", "Agent bug: Missing expected content", "red",
                "The agent's response is missing key terms that the eval expects. "
                "This could be a real content regression or an eval brittleness issue.")
    return ("other", "Other regressions", "amber",
            "Regressions that don't fit the main patterns. Inspect individually.")


# --- build comparison data ---------------------------------------------------

def build_comparison(source: DataSource, exps1: dict[str, SimpleNamespace], exps2: dict[str, SimpleNamespace], score: str = "overall"):
    matched_names = sorted(set(exps1) & set(exps2))
    rows = []
    suite_summaries = []

    for exp_name in matched_names:
        e1 = exps1[exp_name]
        e2 = exps2[exp_name]
        items1 = source.get_items(e1)
        items2 = source.get_items(e2)
        by_name1 = {source.case_name(it): it for it in items1}
        by_name2 = {source.case_name(it): it for it in items2}

        suite_cases = []
        credits1_total = 0.0
        credits2_total = 0.0

        for case_name in sorted(set(by_name1) | set(by_name2)):
            it1 = by_name1.get(case_name)
            it2 = by_name2.get(case_name)
            sm1 = source.score_map(it1) if it1 else {}
            sm2 = source.score_map(it2) if it2 else {}

            if score not in sm1 and score not in sm2:
                continue

            v1, r1 = sm1.get(score, (None, ""))
            v2, r2 = sm2.get(score, (None, ""))
            delta = (v2 - v1) if v1 is not None and v2 is not None else None

            c1 = _credits(it1) if it1 else 0.0
            c2 = _credits(it2) if it2 else 0.0
            credits1_total += c1
            credits2_total += c2

            all_scores1 = {k: v[0] for k, v in sm1.items()}
            all_scores2 = {k: v[0] for k, v in sm2.items()}

            row = SimpleNamespace(
                exp_name=exp_name, exp_id1=e1.id, exp_id2=e2.id,
                case_name=case_name, v1=v1, v2=v2, delta=delta,
                reason1=r1, reason2=r2,
                trace1=source.trace_url(it1) if it1 else None,
                trace2=source.trace_url(it2) if it2 else None,
                credits1=c1, credits2=c2,
                all_scores1=all_scores1, all_scores2=all_scores2,
            )
            rows.append(row)
            suite_cases.append(row)

        scored = [r for r in suite_cases if r.delta is not None]
        if scored:
            mean1 = sum(r.v1 for r in scored) / len(scored)
            mean2 = sum(r.v2 for r in scored) / len(scored)
            regressed = sum(1 for r in scored if r.delta < 0)
            improved = sum(1 for r in scored if r.delta > 0)
            unchanged = sum(1 for r in scored if r.delta == 0)
            suite_summaries.append(SimpleNamespace(
                name=exp_name, mean1=mean1, mean2=mean2,
                delta=mean2 - mean1, n=len(scored),
                regressed=regressed, improved=improved, unchanged=unchanged,
                credits1=credits1_total, credits2=credits2_total,
                exp_id1=e1.id, exp_id2=e2.id,
            ))

    return rows, suite_summaries, set(exps1) - set(exps2), set(exps2) - set(exps1)


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
body { font-family: -apple-system, system-ui, sans-serif; max-width: 920px;
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
.trend-bar { display: flex; height: 8px; margin: 0.5rem 0; border-radius: 4px; overflow: hidden; }
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
.suite-regressed { border-left: 3px solid var(--red); }
.suite-improved { border-left: 3px solid var(--green); }
.suite-unchanged { border-left: 3px solid var(--neutral); }
.case-card { margin-left: 1rem; border: 1px solid var(--border); }
.case-detail { padding: 0.5rem 0; }
.scores-table, .cost-table { border-collapse: collapse; width: 100%; font-size: 0.85rem; margin: 0.3rem 0; }
.scores-table th, .cost-table th { text-align: left; padding: 0.2rem 0.4rem; border-bottom: 1px solid var(--border); color: var(--faint); font-weight: 500; }
.scores-table td, .cost-table td { padding: 0.2rem 0.4rem; }
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
.beta-context { font-size: 0.85rem; }
.beta-context ul { margin: 0.3rem 0; padding-left: 1.2rem; }
.beta-context li { margin: 0.1rem 0; }
.insights { padding: 0.8rem 0; font-size: 0.92rem; }
.insights p { margin: 0.6rem 0; line-height: 1.65; }
.insights ul { margin: 0.4rem 0 0.6rem 1.2rem; padding-left: 0; }
.insights li { margin: 0.3rem 0; line-height: 1.55; }
.insights h3 { font-size: 0.95rem; margin: 1.2rem 0 0.4rem; padding-top: 0.6rem; border-top: 1px solid var(--border); }
.insights h3:first-child { border-top: none; padding-top: 0; }
.insights strong { font-weight: 600; }
.insights code { background: var(--code-bg); padding: 0.1rem 0.25rem; border-radius: 3px; font-size: 0.82rem; }
.insights .red { color: var(--red); } .insights .amber { color: var(--amber); } .insights .green { color: var(--green); }
details:target { scroll-margin-top: 1rem; }
details:target > summary { font-weight: 700; }
a[href^="#"] { color: var(--link); text-decoration: none; }
a[href^="#"]:hover { text-decoration: underline; }
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
  document.querySelectorAll("details").forEach(function(d) { d.open = open; });
}
</script>
"""

_HTML_TAIL = """\
</body>
</html>
"""


def generate_html(rows, suite_summaries, only1, only2, tags1, tags2, label1, label2, score, insights_html="", beta_context_html=""):
    scored = [r for r in rows if r.delta is not None]
    mean1 = sum(r.v1 for r in scored) / len(scored) if scored else 0
    mean2 = sum(r.v2 for r in scored) / len(scored) if scored else 0
    regressed = sum(1 for r in scored if r.delta < 0)
    improved = sum(1 for r in scored if r.delta > 0)
    unchanged = sum(1 for r in scored if r.delta == 0)
    total = len(scored)

    total_credits1 = sum(s.credits1 for s in suite_summaries)
    total_credits2 = sum(s.credits2 for s in suite_summaries)

    # Separate 502 failures for cost analysis
    infra_502_rows = [r for r in rows if r.delta is not None and r.delta < -0.001
                       and "502" in (r.reason2 or "").lower()]
    infra_502_credits = sum(r.credits2 for r in infra_502_rows)
    valid_credits1 = total_credits1
    valid_credits2 = total_credits2 - infra_502_credits  # 502 failures cost ~0

    # Group rows by suite
    by_suite = defaultdict(list)
    for r in rows:
        by_suite[r.exp_name].append(r)

    # Categorize regressions
    regression_rows = [r for r in rows if r.delta is not None and r.delta < -0.001]
    categories = defaultdict(list)
    for r in regression_rows:
        cat_key, cat_title, cat_color, cat_desc = _categorize(r)
        categories[cat_key] = (cat_title, cat_color, cat_desc, [])

    for r in regression_rows:
        cat_key, _, _, _ = _categorize(r)
        categories[cat_key][3].append(r)

    # Sort categories by count
    sorted_cats = sorted(categories.items(), key=lambda x: -len(x[1][3]))

    parts = []
    parts.append(_HTML_HEAD)

    # Title
    parts.append(f"<h1>Eval Report: {_esc(label2)} vs {_esc(label1)}</h1>")
    parts.append(f'<p class="meta">Score: <code>{_esc(score)}</code> \u00b7 '
                 f"Compared {total} cases across {len(suite_summaries)} suites \u00b7 "
                 f"Tags: <code>{_esc(' '.join(tags1))}</code> \u2192 <code>{_esc(' '.join(tags2))}</code></p>")

    # Summary line
    parts.append('<div class="summary-line">')
    parts.append(f'<span class="num {"red" if mean2 < mean1 else "green"}">{mean1:.3f}</span> <span class="label">{_esc(label1)}</span>')
    parts.append(f' \u00b7 <span class="num {"green" if mean2 > mean1 else "red"}">{mean2:.3f}</span> <span class="label">{_esc(label2)}</span>')
    delta_mean = mean2 - mean1
    parts.append(f' \u00b7 <span class="num {_delta_color(delta_mean)}">{_fmt_pp(delta_mean)}</span> <span class="label">delta</span>')
    parts.append(f' \u00b7 <span class="num red">{regressed}</span> <span class="label">regressed</span>')
    parts.append(f' \u00b7 <span class="num green">{improved}</span> <span class="label">improved</span>')
    parts.append(f' \u00b7 <span class="num neutral">{unchanged}</span> <span class="label">unchanged</span>')
    parts.append('</div>')

    # Cost summary
    parts.append('<div class="summary-line" style="margin-top:0.5rem">')
    parts.append(f'<span class="label">Credits:</span> <span class="num">{total_credits1:.4f}</span> <span class="label">{_esc(label1)}</span>')
    parts.append(f' \u00b7 <span class="num">{total_credits2:.4f}</span> <span class="label">{_esc(label2)}</span>')
    credit_delta = total_credits2 - total_credits1
    credit_pct = (credit_delta / total_credits1 * 100) if total_credits1 > 0 else 0
    parts.append(f' \u00b7 <span class="num {_delta_color(-credit_delta)}">{credit_delta:+.4f} ({credit_pct:+.1f}%)</span>')
    if infra_502_credits < 0.001 and len(infra_502_rows) > 0:
        parts.append(f' <span class="label">({len(infra_502_rows)} cases failed with 502, ~0 credits)</span>')
    parts.append('</div>')

    # Trend bar
    if total > 0:
        r_pct = regressed / total * 100
        i_pct = improved / total * 100
        u_pct = unchanged / total * 100
        parts.append(f'<div class="trend-bar"><div class="trend-regressed" style="width:{r_pct:.1f}%"></div>'
                     f'<div class="trend-improved" style="width:{i_pct:.1f}%"></div>'
                     f'<div class="trend-unchanged" style="width:{u_pct:.1f}%"></div></div>')

    # Controls
    parts.append('<div class="controls">')
    parts.append('<button class="btn" onclick="toggleAll(true)">Expand all</button> ')
    parts.append('<button class="btn" onclick="toggleAll(false)">Collapse all</button>')
    parts.append('</div>')

    # Only-in sets
    if only1:
        parts.append(f'<p class="meta"><strong>Only in {_esc(label1)}:</strong> {", ".join(sorted(only1))}</p>')
    if only2:
        parts.append(f'<p class="meta"><strong>Only in {_esc(label2)}:</strong> {", ".join(sorted(only2))}</p>')

    # Insights section (open by default, author-written)
    if insights_html:
        parts.append('<details open id="insights">')
        parts.append('<summary><strong>Summary &amp; Insights</strong></summary>')
        parts.append(f'<div class="insights">{insights_html}</div>')
        parts.append('</details>')

    # Beta context (optional, from --beta-context file)
    if beta_context_html:
        parts.append('<details open id="beta-context">')
        parts.append(f'<summary><strong>Beta Changes Context</strong> <span class="meta">({_esc(label2)} = beta, {_esc(label1)} = baseline)</span></summary>')
        parts.append(f'<div class="beta-context">{beta_context_html}</div>')
        parts.append('</details>')

    # Root cause analysis
    parts.append('<h2 id="root-cause-analysis">Root Cause Analysis</h2>')
    parts.append(f'<p class="meta">{regressed} regressed cases grouped by pattern:</p>')

    for cat_key, (cat_title, cat_color, cat_desc, cat_rows) in sorted_cats:
        if not cat_rows:
            continue
        cat_id = f"rca-{cat_key}"
        parts.append(f'<details class="root-cause" id="{cat_id}">')
        parts.append(f'<summary><span class="badge {cat_color}">{len(cat_rows)}</span> {cat_title}</summary>')
        parts.append(f'<div class="callout {cat_color}">{cat_desc}</div>')

        for r in cat_rows:
            case_id = f"case-{cat_key}-{r.case_name}".replace(" ", "-").replace("/", "-").replace(".", "-").lower()
            parts.append(f'<details class="case-card" id="{case_id}">')
            parts.append(f'<summary><span class="badge {_delta_color(r.delta)}">{_fmt_delta(r.delta)}</span> '
                         f'{_esc(r.case_name)} <span class="meta">({_esc(r.exp_name)})</span></summary>')
            parts.append('<div class="case-detail">')

            # Scores table
            all_score_names = sorted(set(r.all_scores1) | set(r.all_scores2))
            if all_score_names:
                parts.append('<table class="scores-table"><thead><tr><th>Score</th>'
                             f'<th>{_esc(label1)}</th><th>{_esc(label2)}</th><th>Delta</th></tr></thead><tbody>')
                for sn in all_score_names:
                    sv1 = r.all_scores1.get(sn)
                    sv2 = r.all_scores2.get(sn)
                    sd = (sv2 - sv1) if sv1 is not None and sv2 is not None else None
                    parts.append(f'<tr><td>{_esc(sn)}</td>'
                                 f'<td class="num-cell">{_fmt_score(sv1)}</td>'
                                 f'<td class="num-cell">{_fmt_score(sv2)}</td>'
                                 f'<td class="num-cell {_delta_color(sd)}">{_fmt_delta(sd)}</td></tr>')
                parts.append('</tbody></table>')

            # Credits
            parts.append(f'<p class="meta">Credits: {r.credits1:.4f} \u2192 {r.credits2:.4f}</p>')

            # Reasons
            if r.reason1 or r.reason2:
                parts.append('<div class="reasons">')
                if r.reason1:
                    parts.append(f'<p><strong>{_esc(label1)}:</strong> <code>{_esc(r.reason1[:500])}</code></p>')
                if r.reason2:
                    parts.append(f'<p><strong>{_esc(label2)}:</strong> <code>{_esc(r.reason2[:500])}</code></p>')
                parts.append('</div>')

            # Trace links
            if r.trace1 or r.trace2:
                parts.append('<div class="trace-links">')
                if r.trace1:
                    parts.append(f'<a href="{_esc(r.trace1)}" target="_blank" class="trace-link">Trace ({_esc(label1)}) \u2197</a>')
                if r.trace2:
                    parts.append(f'<a href="{_esc(r.trace2)}" target="_blank" class="trace-link">Trace ({_esc(label2)}) \u2197</a>')
                parts.append('</div>')

            # Inspect/fetch commands
            parts.append('<div class="cmds">')
            parts.append(f'<code>uv run python -m agent_evals.scripts.inspect_eval --exp {r.exp_id1} --case "{_esc(r.case_name)}"</code><br>')
            parts.append(f'<code>uv run python -m agent_evals.scripts.inspect_eval --exp {r.exp_id2} --case "{_esc(r.case_name)}"</code><br>')
            parts.append(f'<code>uv run python -m agent_evals.scripts.fetch_traces --exp {r.exp_id1} --case "{_esc(r.case_name)}"</code><br>')
            parts.append(f'<code>uv run python -m agent_evals.scripts.fetch_traces --exp {r.exp_id2} --case "{_esc(r.case_name)}"</code>')
            parts.append('</div>')

            parts.append('</div>')  # case-detail
            parts.append('</details>')  # case-card

        parts.append('</details>')  # root-cause

    # Improvements
    improvement_rows = sorted([r for r in rows if r.delta is not None and r.delta > 0.001],
                               key=lambda r: -r.delta)
    if improvement_rows:
        parts.append('<h2 id="improvements">Improvements</h2>')
        parts.append(f'<p class="meta">{len(improvement_rows)} cases improved:</p>')
        parts.append('<details class="root-cause" id="all-improvements"><summary><span class="badge green">'
                       f'{len(improvement_rows)}</span> All improvements</summary>')
        for r in improvement_rows:
            imp_id = f"imp-{r.case_name}".replace(" ", "-").replace("/", "-").replace(".", "-").lower()
            parts.append(f'<details class="case-card" id="{imp_id}">')
            parts.append(f'<summary><span class="badge green">{_fmt_delta(r.delta)}</span> '
                         f'{_esc(r.case_name)} <span class="meta">({_esc(r.exp_name)})</span></summary>')
            parts.append('<div class="case-detail">')
            if r.reason1:
                parts.append(f'<p><strong>{_esc(label1)}:</strong> <code>{_esc(r.reason1[:400])}</code></p>')
            if r.reason2:
                parts.append(f'<p><strong>{_esc(label2)}:</strong> <code>{_esc(r.reason2[:400])}</code></p>')
            if r.trace1:
                parts.append(f'<a href="{_esc(r.trace1)}" target="_blank" class="trace-link">Trace ({_esc(label1)}) \u2197</a>')
            if r.trace2:
                parts.append(f'<a href="{_esc(r.trace2)}" target="_blank" class="trace-link">Trace ({_esc(label2)}) \u2197</a>')
            parts.append(f'<p class="meta">Credits: {r.credits1:.4f} \u2192 {r.credits2:.4f}</p>')
            parts.append('</div></details>')
        parts.append('</details>')

    # Cost analysis section
    parts.append('<h2 id="credit-usage">Credit Usage by Suite</h2>')
    parts.append('<table class="cost-table"><thead><tr>'
                 f'<th>Suite</th><th>{_esc(label1)} credits</th><th>{_esc(label2)} credits</th><th>Delta</th><th>% change</th>'
                 '</tr></thead><tbody>')
    for s in sorted(suite_summaries, key=lambda s: s.credits2 - s.credits1, reverse=True):
        c_delta = s.credits2 - s.credits1
        c_pct = (c_delta / s.credits1 * 100) if s.credits1 > 0 else 0
        parts.append(f'<tr><td>{_esc(s.name)}</td>'
                     f'<td class="num-cell">{s.credits1:.4f}</td>'
                     f'<td class="num-cell">{s.credits2:.4f}</td>'
                     f'<td class="num-cell {_delta_color(c_delta)}">{c_delta:+.4f}</td>'
                     f'<td class="num-cell {_delta_color(c_delta)}">{c_pct:+.1f}%</td></tr>')
    parts.append(f'<tr class="totals"><td>Total</td>'
                 f'<td class="num-cell">{total_credits1:.4f}</td>'
                 f'<td class="num-cell">{total_credits2:.4f}</td>'
                 f'<td class="num-cell {_delta_color(total_credits2-total_credits1)}">{total_credits2-total_credits1:+.4f}</td>'
                 f'<td class="num-cell {_delta_color(total_credits2-total_credits1)}">{(total_credits2-total_credits1)/total_credits1*100 if total_credits1>0 else 0:+.1f}%</td></tr>')
    parts.append('</tbody></table>')

    if len(infra_502_rows) > 0:
        parts.append(f'<p class="meta">Note: {len(infra_502_rows)} cases failed with HTTP 502 '
                       f'(infrastructure), consuming ~0 credits on {label2}. Excluding these, '
                       f'the effective credit usage is {valid_credits1:.4f} \u2192 {valid_credits2:.4f}.</p>')

    # Suite-by-suite breakdown
    parts.append('<h2 id="suite-breakdown">Suite Breakdown</h2>')
    suite_summaries.sort(key=lambda s: s.delta if s.delta is not None else 0)
    for s in suite_summaries:
        suite_rows = by_suite.get(s.name, [])
        suite_class = "suite-regressed" if s.delta < -0.001 else "suite-improved" if s.delta > 0.001 else "suite-unchanged"
        suite_id = f"suite-{s.name}".replace(" ", "-").replace("/", "-").replace(".", "-").lower()
        parts.append(f'<details class="suite-card {suite_class}" id="{suite_id}">')
        parts.append(f'<summary><span class="badge {_delta_color(s.delta)}">{_fmt_delta(s.delta)}</span> '
                     f'{_esc(s.name)} <span class="meta">{_fmt_score(s.mean1)} \u2192 {_fmt_score(s.mean2)} \u00b7 '
                     f'{s.regressed} regressed, {s.improved} improved, {s.unchanged} unchanged \u00b7 '
                     f'credits: {s.credits1:.4f} \u2192 {s.credits2:.4f}</span></summary>')

        for r in sorted(suite_rows, key=lambda r: r.delta if r.delta is not None else 0):
            case_class = _delta_color(r.delta)
            case_id = f"suitecase-{s.name}-{r.case_name}".replace(" ", "-").replace("/", "-").replace(".", "-").lower()
            parts.append(f'<details class="case-card" id="{case_id}">')
            parts.append(f'<summary><span class="badge {case_class}">{_fmt_delta(r.delta)}</span> '
                         f'{_esc(r.case_name)} <span class="meta">{_fmt_score(r.v1)} \u2192 {_fmt_score(r.v2)}</span></summary>')
            parts.append('<div class="case-detail">')

            all_score_names = sorted(set(r.all_scores1) | set(r.all_scores2))
            if all_score_names:
                parts.append('<table class="scores-table"><thead><tr><th>Score</th>'
                             f'<th>{_esc(label1)}</th><th>{_esc(label2)}</th><th>Delta</th></tr></thead><tbody>')
                for sn in all_score_names:
                    sv1 = r.all_scores1.get(sn)
                    sv2 = r.all_scores2.get(sn)
                    sd = (sv2 - sv1) if sv1 is not None and sv2 is not None else None
                    parts.append(f'<tr><td>{_esc(sn)}</td>'
                                 f'<td class="num-cell">{_fmt_score(sv1)}</td>'
                                 f'<td class="num-cell">{_fmt_score(sv2)}</td>'
                                 f'<td class="num-cell {_delta_color(sd)}">{_fmt_delta(sd)}</td></tr>')
                parts.append('</tbody></table>')

            parts.append(f'<p class="meta">Credits: {r.credits1:.4f} \u2192 {r.credits2:.4f}</p>')

            if r.reason1 or r.reason2:
                parts.append('<div class="reasons">')
                if r.reason1:
                    parts.append(f'<p><strong>{_esc(label1)}:</strong> <code>{_esc(r.reason1[:500])}</code></p>')
                if r.reason2:
                    parts.append(f'<p><strong>{_esc(label2)}:</strong> <code>{_esc(r.reason2[:500])}</code></p>')
                parts.append('</div>')

            if r.trace1 or r.trace2:
                parts.append('<div class="trace-links">')
                if r.trace1:
                    parts.append(f'<a href="{_esc(r.trace1)}" target="_blank" class="trace-link">Trace ({_esc(label1)}) \u2197</a>')
                if r.trace2:
                    parts.append(f'<a href="{_esc(r.trace2)}" target="_blank" class="trace-link">Trace ({_esc(label2)}) \u2197</a>')
                parts.append('</div>')

            parts.append('<div class="cmds">')
            parts.append(f'<code>uv run python -m agent_evals.scripts.inspect_eval --exp {r.exp_id1} --case "{_esc(r.case_name)}"</code><br>')
            parts.append(f'<code>uv run python -m agent_evals.scripts.inspect_eval --exp {r.exp_id2} --case "{_esc(r.case_name)}"</code><br>')
            parts.append(f'<code>uv run python -m agent_evals.scripts.fetch_traces --exp {r.exp_id1} --case "{_esc(r.case_name)}"</code><br>')
            parts.append(f'<code>uv run python -m agent_evals.scripts.fetch_traces --exp {r.exp_id2} --case "{_esc(r.case_name)}"</code>')
            parts.append('</div>')

            parts.append('</div></details>')

        parts.append('</details>')

    # Footer
    parts.append('<div class="footer"><code>Generated by agent-eval \u00b7 '
                 f'compare_experiments --tag1 {" ".join(tags1)} --tag2 {" ".join(tags2)} --score {_esc(score)}</code><br>'
                 f'<code>Full beta-vs-main.md: agent-api/docs/beta-vs-main.md</code></div>')

    parts.append(_HTML_TAIL)
    return "\n".join(parts)


def _add_insights_to_report(report_path: Path, insights_html: str) -> None:
    """Inject or replace insights HTML in an existing report file.

    Looks for ``<details open id="insights">`` and replaces its inner content.
    If the insights section doesn't exist yet, inserts it right before the
    ``<details open id="beta-context">`` element. If that anchor is missing,
    inserts after the controls div.
    """
    import re
    content = report_path.read_text(encoding="utf-8")

    insights_block = (
        '<details open id="insights">\n'
        '<summary><strong>Summary &amp; Insights</strong></summary>\n'
        f'<div class="insights">\n{insights_html}\n</div>\n'
        '</details>'
    )

    # Check if insights section already exists
    pattern = r'<details open id="insights">.*?</details>'
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, insights_block, content, count=1, flags=re.DOTALL)
    elif '<details open id="beta-context">' in content:
        content = content.replace(
            '<details open id="beta-context">',
            insights_block + '\n\n    <details open id="beta-context">',
            1,
        )
    else:
        # Fallback: insert after the controls div
        anchor = '</div>\n\n    # Beta context'
        if anchor in content:
            content = content.replace(anchor, '</div>\n\n    ' + insights_block + '\n\n    # Beta context', 1)
        else:
            print("Warning: could not find insertion point for insights.", file=sys.stderr)

    report_path.write_text(content, encoding="utf-8")
    print(f"Insights injected into {report_path} ({report_path.stat().st_size} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Generate HTML eval report")
    subparsers = parser.add_subparsers(dest="mode")

    # Default: generate report
    gen = subparsers.add_parser("generate", help="Generate a new HTML report from Opik or local data")
    gen.add_argument("--tag1", required=True, nargs="+", help="Tag(s) for side 1 (baseline). Multiple tags are merged, newest per suite wins.")
    gen.add_argument("--tag2", required=True, nargs="+", help="Tag(s) for side 2 (under test). Multiple tags are merged, newest per suite wins.")
    gen.add_argument("--score", default="overall", help="Feedback score to compare")
    gen.add_argument("-o", "--output", default="report.html", help="Output file")
    gen.add_argument("--label1", default=None, help="Label for side 1")
    gen.add_argument("--label2", default=None, help="Label for side 2")
    gen.add_argument("--beta-context", default=None, help="Path to an HTML file with beta-changes context (optional)")
    gen.add_argument("--insights", default=None, help="Path to an HTML file with author-written insights (optional, can also be added later via add-insights)")
    gen.add_argument("--source", default="opik", choices=("opik", "local"), help="Data source (default: opik).")
    gen.add_argument("--results-dir", action="append", default=[], help="Path to results directory for local source (repeatable, default: results/).")

    # Add insights to existing report
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

    # Generate mode
    if not hasattr(args, "tag1"):
        parser.error("Use 'generate' or 'add-insights' subcommand")

    tags1 = args.tag1 if isinstance(args.tag1, list) else [args.tag1]
    tags2 = args.tag2 if isinstance(args.tag2, list) else [args.tag2]
    label1 = args.label1 or tags1[0]
    label2 = args.label2 or tags2[0]

    source = make_source(getattr(args, "source", "opik"), results_dir=getattr(args, "results_dir", []) or None)
    print(f"Discovering experiments for tags1={tags1!r}...")
    exps1 = _find_by_tags(source, tags1, "side1")
    print(f"  found {len(exps1)} experiments")
    print(f"Discovering experiments for tags2={tags2!r}...")
    exps2 = _find_by_tags(source, tags2, "side2")
    print(f"  found {len(exps2)} experiments")
    matched = sorted(set(exps1) & set(exps2))
    print(f"Matched: {len(matched)} suites")

    print(f"Building comparison (score={args.score})...")
    rows, suite_summaries, only1, only2 = build_comparison(source, exps1, exps2, args.score)

    scored = [r for r in rows if r.delta is not None]
    if scored:
        m1 = sum(r.v1 for r in scored) / len(scored)
        m2 = sum(r.v2 for r in scored) / len(scored)
        r_count = sum(1 for r in scored if r.delta < 0)
        i_count = sum(1 for r in scored if r.delta > 0)
        u_count = sum(1 for r in scored if r.delta == 0)
        print(f"\nMean {args.score}: {m1:.3f} -> {m2:.3f} ({len(scored)} matched: {i_count} improved, {r_count} regressed, {u_count} unchanged)")
        tc1 = sum(s.credits1 for s in suite_summaries)
        tc2 = sum(s.credits2 for s in suite_summaries)
        print(f"Credits: {tc1:.4f} -> {tc2:.4f} ({tc2-tc1:+.4f}, {(tc2-tc1)/tc1*100 if tc1>0 else 0:+.1f}%)")

    if only1:
        print(f"Only in {label1}: {', '.join(sorted(only1))}")
    if only2:
        print(f"Only in {label2}: {', '.join(sorted(only2))}")

    insights_html = ""
    if getattr(args, "insights", None):
        ins_path = Path(args.insights)
        if ins_path.exists():
            insights_html = ins_path.read_text(encoding="utf-8")

    # Optional beta context
    beta_context_html = ""
    if getattr(args, "beta_context", None):
        bc_path = Path(args.beta_context)
        if bc_path.exists():
            beta_context_html = bc_path.read_text(encoding="utf-8")

    print(f"\nGenerating HTML report...")
    html_content = generate_html(rows, suite_summaries, only1, only2, tags1, tags2, label1, label2, args.score,
                                  insights_html=insights_html,
                                  beta_context_html=beta_context_html)

    out_path = Path(args.output)
    out_path.write_text(html_content, encoding="utf-8")
    print(f"Report written to {out_path} ({out_path.stat().st_size} bytes)")
    print(f"\nTo add insights: uv run python -m agent_evals.scripts.generate_report add-insights -o {out_path} --insights insights.html")


if __name__ == "__main__":
    main()
