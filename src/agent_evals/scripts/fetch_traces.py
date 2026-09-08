"""Fetch OpenInference traces for every case in an Opik experiment.

Given an Opik experiment ID, resolves each case's ``context_id`` from the
experiment's task output (the ``trace_url`` carries it as the ``thread``
query parameter), then fetches the full OpenInference trace — all spans,
all pages — from the agent API's trace endpoint
(``GET /v2/agentic/contexts/{contextId}/trace``).

The environment for the agent API is read from the experiment's config
metadata (the ``environment`` field the Opik sink writes); pass ``--env`` to
override.

Output modes::

    # Default: compact text — only messages, tool calls, and tool defs.
    # Deduplicates repeated messages and tool definitions across LLM calls.
    uv run python -m agent_evals.scripts.fetch_traces --exp <id> --case my_case

    # Raw JSON to a file:
    uv run python -m agent_evals.scripts.fetch_traces --exp <id> -o traces.json

    # Foldable HTML with <details>/<summary> collapsibles:
    uv run python -m agent_evals.scripts.fetch_traces --exp <id> --html -o traces.html

    # List cases with their context IDs (no trace fetch):
    uv run python -m agent_evals.scripts.fetch_traces --exp <id> --list

    # Override the environment (defaults to the experiment's metadata):
    uv run python -m agent_evals.scripts.fetch_traces --exp <id> --env eu

    # Full verbose text (timestamps, metadata, all tool def details):
    uv run python -m agent_evals.scripts.fetch_traces --exp <id> --case my_case --verbose
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import dotenv

from ..__main__ import resolve_environment
from ..client import AgentClient
from ..tracing import fetch_all_trace_pages

_REPO_ROOT = Path(__file__).resolve().parents[3]
dotenv.load_dotenv(_REPO_ROOT / ".env")

_MAX_TEXT_LEN = 500
_MAX_TOOL_DEF_LEN = 200


# --- Opik client (reuse the pattern from inspect_eval / compare_experiments) ---


def _make_opik_client() -> Any:
    """Create an Opik client using the same URL resolution as the eval harness."""
    import opik

    from ..environment import OPIK_URL_OVERRIDE_VAR
    from ..reporting.opik_target import resolve_opik_url

    url = resolve_opik_url()
    os.environ.setdefault(OPIK_URL_OVERRIDE_VAR, url)
    project = os.environ.get("OPIK_PROJECT_NAME") or "Agents"
    os.environ["OPIK_PROJECT_NAME"] = project
    return opik.Opik(
        host=url,
        workspace="default",
        api_key=os.environ.get("OPIK_API_KEY"),
    )


def _get_experiment_data(client: Any, experiment_id: str) -> Any:
    """Fetch the ``ExperimentPublic`` (metadata, dataset_id, dataset_name)."""
    return client.get_experiment_by_id(experiment_id).get_experiment_data()


def _experiment_env(data: Any) -> str | None:
    """Read the ``environment`` from the experiment's config metadata."""
    meta = data.metadata
    if isinstance(meta, dict):
        return meta.get("environment")
    return None


def _get_items(client: Any, experiment_id: str, dataset_id: str) -> list:
    """Fetch all experiment items by experiment id.

    Uses ``dataset_id`` directly (from ``get_experiment_data``) rather than
    the SDK's ``Experiment.get_items()``, which resolves ``dataset_id`` from
    ``dataset_name`` — a field that can be blank on some experiments.
    """
    from opik.api_objects.experiment import rest_operations

    return rest_operations.find_experiment_items_for_dataset(
        rest_client=client._rest_client,
        dataset_id=dataset_id,
        experiment_ids=[experiment_id],
        max_results=10000,
        truncate=False,
    )


def _case_name(item: Any) -> str:
    data = item.dataset_item_data or {}
    return data.get("name") or "(unnamed)"


def _context_id_from_item(item: Any) -> str | None:
    """Extract the ``context_id`` from an item's ``trace_url``.

    The trace URL is ``{opik_url}/default/projects/{id}/logs?thread={context_id}``;
    the ``thread`` query parameter is the agent context ID the trace endpoint
    expects.
    """
    out = item.evaluation_task_output or {}
    trace_url = out.get("trace_url")
    if not trace_url:
        return None
    parsed = urlparse(trace_url)
    qs = parse_qs(parsed.query)
    thread = qs.get("thread", [None])[0]
    return thread if thread and thread.strip() else None


# --- retry helper (tunnel is flaky under load) -------------------------------


def _retry(fn: Any, *, what: str, max_retries: int = 3, backoff: float = 2.0) -> Any:
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == max_retries:
                raise
            wait = backoff * attempt
            print(
                f"  retry {attempt}/{max_retries} ({what}): {exc} "
                f"— waiting {wait:.0f}s",
                file=sys.stderr,
            )
            time.sleep(wait)


# --- span extraction helpers --------------------------------------------------


def _span_kind(span: dict[str, Any]) -> str:
    return span.get("attributes", {}).get("openinference.span.kind", "")


def _sort_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort spans by start_time (ascending)."""
    return sorted(spans, key=lambda s: s.get("start_time", ""))


def _build_tree(spans: list[dict[str, Any]]) -> dict[str | None, list[dict[str, Any]]]:
    """Build parent→children map from spans."""
    children: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    for span in spans:
        pid = span.get("parent_span_id")
        children[pid].append(span)
    for kids in children.values():
        kids.sort(key=lambda s: s.get("start_time", ""))
    return children


def _parse_json_attr(value: Any) -> Any:
    """Parse a JSON string attribute, returning None on failure."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def _extract_llm_messages(span: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract conversation messages from an LLM span.

    Prefers the parsed ``llm.input_messages`` array; falls back to parsing
    the ``input`` JSON string's ``messages`` field.
    """
    attrs = span.get("attributes", {})
    msgs = attrs.get("llm.input_messages")
    if isinstance(msgs, list):
        return msgs
    parsed = _parse_json_attr(attrs.get("input"))
    if isinstance(parsed, dict):
        return parsed.get("messages", [])
    return []


def _extract_tool_defs(span: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract tool definitions from an LLM span's ``input`` attribute."""
    attrs = span.get("attributes", {})
    parsed = _parse_json_attr(attrs.get("input"))
    if isinstance(parsed, dict):
        return parsed.get("tools", [])
    return []


def _extract_llm_response(span: dict[str, Any]) -> dict[str, Any]:
    """Extract the assistant's response from an LLM span's ``output``."""
    attrs = span.get("attributes", {})
    parsed = _parse_json_attr(attrs.get("output"))
    if isinstance(parsed, dict):
        choices = parsed.get("choices", [])
        if choices:
            return choices[0].get("message", {})
    return {}


def _extract_tool_args(span: dict[str, Any]) -> dict[str, Any]:
    """Extract tool call arguments from a TOOL span."""
    attrs = span.get("attributes", {})
    raw = attrs.get("gen_ai.tool.call.arguments")
    parsed = _parse_json_attr(raw)
    if isinstance(parsed, dict):
        return parsed
    return {}


def _extract_tool_result(span: dict[str, Any]) -> Any:
    """Extract tool result from a TOOL span."""
    attrs = span.get("attributes", {})
    raw = attrs.get("gen_ai.tool.call.result")
    return _parse_json_attr(raw)


def _message_key(msg: dict[str, Any]) -> str:
    """A dedup key for a message: role + content hash + tool_calls hash."""
    role = msg.get("role", "")
    content = msg.get("content", "") or ""
    tc = msg.get("tool_calls")
    tc_str = json.dumps(tc, sort_keys=True) if tc else ""
    return f"{role}:{content[:100]}:{tc_str[:100]}"


def _tool_def_names(tool_defs: list[dict[str, Any]]) -> list[str]:
    """Extract tool names from tool definitions."""
    names: list[str] = []
    for td in tool_defs:
        fn = td.get("function", {}) if isinstance(td, dict) else {}
        name = fn.get("name", "")
        if name:
            names.append(name)
    return names


def _truncate(text: str, max_len: int = _MAX_TEXT_LEN) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _format_tool_call(tc: dict[str, Any]) -> str:
    """Format a single tool call compactly."""
    fn = tc.get("function", {})
    name = fn.get("name", "?")
    args = fn.get("arguments", "")
    if isinstance(args, str):
        parsed = _parse_json_attr(args)
        if parsed is not None:
            args = json.dumps(parsed, ensure_ascii=False)
    return f"{name}({args})"


# --- compact text rendering ---------------------------------------------------


def _render_compact(
    traces: dict[str, Any] | None,
    context_id: str,
    case_name: str,
    *,
    verbose: bool = False,
) -> str:
    """Render traces as compact text: only messages, tool calls, tool defs."""
    lines: list[str] = []
    lines.append(f"{'=' * 80}")
    lines.append(f"  {case_name}  (context: {context_id})")
    lines.append(f"{'=' * 80}")

    if not traces:
        lines.append("  no traces")
        return "\n".join(lines)

    seen_msgs: set[str] = set()
    seen_tool_names: list[str] | None = None

    for trace in traces.get("traces", []):
        trace_info = trace.get("trace", {})
        tname = trace_info.get("name", "?")
        spans = _sort_spans(trace.get("spans", []))
        lines.append(f"\n  trace: {tname}  ({len(spans)} spans)")

        children = _build_tree(spans)

        def _walk(span: dict[str, Any], depth: int) -> None:
            nonlocal seen_tool_names
            indent = "    " + "  " * depth
            kind = _span_kind(span)
            name = span.get("name", "?")
            attrs = span.get("attributes", {})

            if kind == "LLM":
                model = attrs.get("llm.model_name", "?")
                tokens = attrs.get("llm.token_count.total")
                header = f"{indent}[LLM] {model}"
                if tokens:
                    header += f" ({tokens} tokens)"
                if verbose:
                    st = span.get("start_time", "")
                    header += f"  {st}"
                lines.append(header)

                # Messages (deduplicated)
                msgs = _extract_llm_messages(span)
                new_msgs = [m for m in msgs if _message_key(m) not in seen_msgs]
                for m in new_msgs:
                    seen_msgs.add(_message_key(m))
                if new_msgs:
                    lines.append(f"{indent}  messages:")
                    for m in new_msgs:
                        role = m.get("role", "?")
                        content = m.get("content", "") or ""
                        tc = m.get("tool_calls")
                        if tc:
                            tc_str = ", ".join(_format_tool_call(t) for t in tc)
                            lines.append(f"{indent}    {role}: → {tc_str}")
                        else:
                            lines.append(f"{indent}    {role}: {_truncate(content)}")

                # Tool definitions (deduplicated by name set)
                tool_defs = _extract_tool_defs(span)
                current_names = _tool_def_names(tool_defs)
                if tool_defs:
                    if seen_tool_names == current_names:
                        lines.append(f"{indent}  tools: (same {len(current_names)})")
                    else:
                        if verbose:
                            lines.append(f"{indent}  tools ({len(current_names)}):")
                            for td in tool_defs:
                                fn = td.get("function", {})
                                tname_ = fn.get("name", "?")
                                desc = fn.get("description", "")
                                lines.append(
                                    f"{indent}    {tname_}: "
                                    f"{_truncate(desc, _MAX_TOOL_DEF_LEN)}"
                                )
                        else:
                            lines.append(f"{indent}  tools: {', '.join(current_names)}")
                        seen_tool_names = current_names

                # Response
                resp = _extract_llm_response(span)
                if resp:
                    lines.append(f"{indent}  response:")
                    content = resp.get("content", "") or ""
                    tcs = resp.get("tool_calls")
                    if tcs:
                        for tc in tcs:
                            lines.append(f"{indent}    → {_format_tool_call(tc)}")
                    elif content:
                        lines.append(f"{indent}    {_truncate(content)}")
                    else:
                        lines.append(f"{indent}    (empty)")

            elif kind == "TOOL":
                args = _extract_tool_args(span)
                result = _extract_tool_result(span)
                args_str = json.dumps(args, ensure_ascii=False) if args else "{}"
                lines.append(f"{indent}[TOOL] {name}")
                lines.append(f"{indent}  args: {_truncate(args_str)}")
                if result is not None:
                    result_str = (
                        json.dumps(result, ensure_ascii=False)
                        if not isinstance(result, str)
                        else result
                    )
                    lines.append(f"{indent}  result: {_truncate(result_str)}")

            else:
                # CHAIN and other spans: name + brief output
                label = f"{indent}[{kind or 'SPAN'}] {name}"
                if verbose:
                    st = span.get("start_time", "")
                    label += f"  {st}"
                lines.append(label)
                out_raw = attrs.get("output")
                if out_raw:
                    parsed = _parse_json_attr(out_raw)
                    if isinstance(parsed, dict):
                        out_brief = parsed.get("output") or parsed.get("tool_count", "")
                        if out_brief:
                            lines.append(
                                f"{indent}  → {_truncate(str(out_brief), 120)}"
                            )
                    elif isinstance(out_raw, str):
                        lines.append(f"{indent}  → {_truncate(out_raw, 120)}")

            for child in children.get(span.get("span_id"), []):
                _walk(child, depth + 1)

        roots = children.get(None, spans)
        for root in roots:
            _walk(root, 0)

    return "\n".join(lines)


# --- HTML rendering with <details>/<summary> ---------------------------------


_HTML_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 920px;
         margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }
  h1 { font-size: 1.4rem; }
  h2 { font-size: 1.1rem; margin-top: 1.5rem; }
  details { margin: 0.3rem 0; padding: 0.3rem 0.5rem; border-radius: 4px;
            border: 1px solid #e0e0e0; }
  details[open] { border-color: #bbb; }
  summary { cursor: pointer; font-weight: 500; list-style: none; }
  summary::-webkit-details-marker { display: none; }
  summary::before { content: "▸ "; color: #888; }
  details[open] > summary::before { content: "▾ "; }
  .msg { margin: 0.2rem 0; padding: 0.2rem 0.4rem; }
  .msg-role { font-weight: 600; color: #555; }
  .msg-content { font-family: monospace; white-space: pre-wrap;
                 margin: 0.2rem 0 0.2rem 1.2rem; font-size: 0.85rem; }
  .tool-def { font-family: monospace; font-size: 0.85rem; margin: 0.2rem 0 0.2rem 1.2rem; }
  .tool-name { font-weight: 600; }
  .tool-desc { color: #666; }
  .tool-call { font-family: monospace; margin: 0.2rem 0 0.2rem 1.2rem;
               font-size: 0.85rem; }
  .meta { color: #888; font-size: 0.8rem; }
  .toggle-all { cursor: pointer; background: #f0f0f0; border: 1px solid #ccc;
                padding: 0.3rem 0.8rem; border-radius: 4px; font-size: 0.85rem; }
</style>
</head>
<body>
"""

_HTML_TAIL = """\
<script>
function toggleAll(open) {
  document.querySelectorAll('details').forEach(d => d.open = open);
}
</script>
</body>
</html>
"""


def _render_html(
    traces: dict[str, Any] | None,
    context_id: str,
    case_name: str,
) -> str:
    """Render traces as foldable HTML using <details>/<summary>."""
    parts: list[str] = [_HTML_HEAD]
    parts.append(f"<h1>{html.escape(case_name)}</h1>")
    parts.append(f'<p class="meta">context: <code>{html.escape(context_id)}</code></p>')
    parts.append(
        '<button class="toggle-all" onclick="toggleAll(true)">Expand all</button> '
        '<button class="toggle-all" onclick="toggleAll(false)">Collapse all</button>'
    )

    if not traces:
        parts.append("<p>no traces</p>")
        parts.append(_HTML_TAIL)
        return "\n".join(parts)

    seen_msgs: set[str] = set()
    seen_tool_names: list[str] | None = None

    for trace in traces.get("traces", []):
        trace_info = trace.get("trace", {})
        tname = trace_info.get("name", "?")
        spans = _sort_spans(trace.get("spans", []))

        parts.append(f"<h2>{html.escape(tname)} ({len(spans)} spans)</h2>")

        children = _build_tree(spans)

        def _walk(span: dict[str, Any], depth: int) -> str:
            nonlocal seen_tool_names
            kind = _span_kind(span)
            name = html.escape(span.get("name", "?"))
            attrs = span.get("attributes", {})
            parts_list: list[str] = []

            if kind == "LLM":
                model = attrs.get("llm.model_name", "?")
                tokens = attrs.get("llm.token_count.total", "")
                header = f"[LLM] {html.escape(model)}"
                if tokens:
                    header += f" ({tokens} tokens)"
                parts_list.append(f"<details><summary>{header}</summary>")

                # Messages
                msgs = _extract_llm_messages(span)
                new_msgs = [m for m in msgs if _message_key(m) not in seen_msgs]
                for m in new_msgs:
                    seen_msgs.add(_message_key(m))
                if new_msgs:
                    parts_list.append("<details><summary>messages</summary>")
                    for m in new_msgs:
                        role = html.escape(m.get("role", "?"))
                        content = m.get("content", "") or ""
                        tc = m.get("tool_calls")
                        if tc:
                            tc_str = html.escape(
                                ", ".join(_format_tool_call(t) for t in tc)
                            )
                            parts_list.append(
                                f'<div class="msg"><span class="msg-role">{role}</span>: '
                                f'→ <span class="tool-call">{tc_str}</span></div>'
                            )
                        else:
                            parts_list.append(
                                f'<div class="msg"><span class="msg-role">{role}</span>:'
                                f'<div class="msg-content">{html.escape(_truncate(content))}</div></div>'
                            )
                    parts_list.append("</details>")

                # Tool defs
                tool_defs = _extract_tool_defs(span)
                current_names = _tool_def_names(tool_defs)
                if tool_defs:
                    if seen_tool_names == current_names:
                        parts_list.append(
                            f"<details><summary>tools ({len(current_names)}, same as above)</summary></details>"
                        )
                    else:
                        parts_list.append(
                            f"<details><summary>tools ({len(current_names)})</summary>"
                        )
                        for td in tool_defs:
                            fn = td.get("function", {})
                            tn = html.escape(fn.get("name", "?"))
                            desc = html.escape(
                                _truncate(
                                    fn.get("description", ""),
                                    _MAX_TOOL_DEF_LEN,
                                )
                            )
                            parts_list.append(
                                f'<div class="tool-def"><span class="tool-name">{tn}</span>: '
                                f'<span class="tool-desc">{desc}</span></div>'
                            )
                        parts_list.append("</details>")
                        seen_tool_names = current_names

                # Response
                resp = _extract_llm_response(span)
                if resp:
                    parts_list.append("<details><summary>response</summary>")
                    content = resp.get("content", "") or ""
                    tcs = resp.get("tool_calls")
                    if tcs:
                        for tc in tcs:
                            parts_list.append(
                                f'<div class="tool-call">→ {html.escape(_format_tool_call(tc))}</div>'
                            )
                    elif content:
                        parts_list.append(
                            f'<div class="msg-content">{html.escape(_truncate(content))}</div>'
                        )
                    parts_list.append("</details>")

                # Children
                for child in children.get(span.get("span_id"), []):
                    parts_list.append(_walk(child, depth + 1))

                parts_list.append("</details>")

            elif kind == "TOOL":
                args = _extract_tool_args(span)
                result = _extract_tool_result(span)
                args_str = html.escape(
                    json.dumps(args, ensure_ascii=False) if args else "{}"
                )
                parts_list.append(f"<details><summary>[TOOL] {name}</summary>")
                parts_list.append(f'<div class="msg-content">args: {args_str}</div>')
                if result is not None:
                    result_str = (
                        json.dumps(result, ensure_ascii=False)
                        if not isinstance(result, str)
                        else result
                    )
                    parts_list.append(
                        f'<div class="msg-content">result: {html.escape(_truncate(result_str))}</div>'
                    )
                for child in children.get(span.get("span_id"), []):
                    parts_list.append(_walk(child, depth + 1))
                parts_list.append("</details>")

            else:
                label = f"[{kind or 'SPAN'}] {name}"
                out_raw = attrs.get("output")
                out_brief = ""
                if out_raw:
                    parsed = _parse_json_attr(out_raw)
                    if isinstance(parsed, dict):
                        out_brief = str(
                            parsed.get("output") or parsed.get("tool_count", "")
                        )
                    elif isinstance(out_raw, str):
                        out_brief = _truncate(out_raw, 120)
                if out_brief:
                    label += f" → {html.escape(out_brief)}"
                parts_list.append(f"<details><summary>{label}</summary>")
                for child in children.get(span.get("span_id"), []):
                    parts_list.append(_walk(child, depth + 1))
                parts_list.append("</details>")

            return "\n".join(parts_list)

        roots = children.get(None, spans)
        for root in roots:
            parts.append(_walk(root, 0))

    parts.append(_HTML_TAIL)
    return "\n".join(parts)


# --- main --------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch OpenInference traces for an Opik experiment."
    )
    parser.add_argument("--exp", required=True, help="Opik experiment ID.")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--case", default=None, help="Fetch trace for a single case by name."
    )
    grp.add_argument(
        "--list",
        action="store_true",
        help="List cases with their context IDs and exit.",
    )
    parser.add_argument(
        "--env",
        default=None,
        help="Override the agent API environment (defaults to the experiment's metadata).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write output to this file (JSON with -o, HTML with --html -o).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON (full trace data).",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Output foldable HTML with <details>/<summary> collapsibles.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include timestamps, full tool def descriptions, and metadata.",
    )
    args = parser.parse_args()

    opik_client = _make_opik_client()

    # One call to get the experiment's metadata (environment) and dataset_id.
    exp_data = _retry(
        lambda: _get_experiment_data(opik_client, args.exp),
        what=f"get experiment {args.exp[:8]}",
    )

    # Determine the agent API environment.
    env_name = args.env
    if env_name is None:
        env_name = _experiment_env(exp_data)
    if env_name is None:
        raise SystemExit(
            "Could not determine the agent API environment from the experiment's "
            "metadata. Pass --env <name> explicitly."
        )

    environment = resolve_environment(env_name)
    agent_client = AgentClient(environment)

    dataset_id = exp_data.dataset_id
    if not dataset_id:
        raise SystemExit(
            f"Experiment {args.exp} has no dataset_id; cannot fetch items."
        )

    items = _retry(
        lambda: _get_items(opik_client, args.exp, dataset_id),
        what=f"get items {args.exp[:8]}",
    )

    if not items:
        print("No items found in this experiment.")
        return

    # Build (case_name, context_id) pairs.
    cases: list[tuple[str, str | None]] = []
    for item in items:
        name = _case_name(item)
        ctx = _context_id_from_item(item)
        cases.append((name, ctx))

    if args.list:
        print(f"{'case name':<50} {'context_id':<40}")
        print("-" * 92)
        for name, ctx in cases:
            print(f"{name[:50]:<50} {ctx or '(none)':<40}")
        return

    if args.case:
        matches = [(n, c) for n, c in cases if n == args.case]
        if not matches:
            print(f"Case {args.case!r} not found. Available:")
            for n, _ in cases:
                print(f"  {n}")
            return
        cases = matches

    # Fetch traces for each case.
    raw_traces: dict[str, Any] = {}
    for name, ctx in cases:
        if ctx is None:
            raw_traces[name] = {"error": "no context_id (no trace_url in task output)"}
            continue
        trace = _retry(
            lambda: fetch_all_trace_pages(agent_client, ctx),
            what=f"trace {ctx[:12]}",
        )
        raw_traces[name] = trace

    # Output.
    if args.json:
        output_text = json.dumps(raw_traces, ensure_ascii=False, indent=2)
        if args.output:
            args.output.write_text(output_text + "\n", encoding="utf-8")
            print(f"Wrote JSON traces for {len(raw_traces)} case(s) to {args.output}")
        else:
            print(output_text)
    elif args.html:
        html_parts: list[str] = []
        for name, ctx in cases:
            trace = raw_traces.get(name)
            if trace and "error" not in trace:
                html_parts.append(_render_html(trace, ctx or "?", name))
            else:
                html_parts.append(
                    f"<h1>{html.escape(name)}</h1><p>no trace available</p>"
                )
        full_html = "\n<hr>\n".join(html_parts)
        if args.output:
            args.output.write_text(full_html, encoding="utf-8")
            print(f"Wrote HTML traces for {len(cases)} case(s) to {args.output}")
        else:
            print(full_html)
    else:
        # Default: compact text.
        for name, ctx in cases:
            trace = raw_traces.get(name)
            if trace and "error" not in trace:
                print(
                    _render_compact(
                        trace,
                        ctx or "?",
                        name,
                        verbose=args.verbose,
                    )
                )
            else:
                error = (
                    trace.get("error", "no trace")
                    if isinstance(trace, dict)
                    else "no trace"
                )
                print(f"\n{'=' * 80}")
                print(f"  {name}  (context: {ctx or '?'}")
                print(f"  {error}")
                print(f"{'=' * 80}")

    agent_client.close()


if __name__ == "__main__":
    main()
