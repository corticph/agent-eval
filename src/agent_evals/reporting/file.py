"""The file Sink: local JSON plus Markdown results files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..expectations import ExpectationResult
from ..loader import EvaluationCase, EvaluationSuite
from ..results import EvaluationResult, UsageMetrics
from ..schemas.response import Response
from .trace import build_trace_url


def _response_to_dict(response: Response | None) -> dict[str, Any] | None:
    """Return the full response dict (or ``None``) for serialisation/trace lookup."""
    return response.to_dict() if response is not None else None


class FileSink:
    """The local file Sink: collects results and writes JSON plus Markdown summaries."""

    def __init__(
        self,
        path: Path,
        *,
        trace_base_url: str | None = None,
    ) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._results: list[dict[str, Any]] = []
        self._entries: list[tuple[EvaluationCase, EvaluationResult]] = []
        # Already fully resolved by the Environment; None means the
        # environment has no trace destination (local) and links are omitted.
        self._trace_base_url = trace_base_url

    def on_start(self, suite: "EvaluationSuite") -> None:
        """Nothing to prepare: the constructor already claimed the output directory."""

    def write(self, case: EvaluationCase, result: EvaluationResult) -> None:
        main_entry = result.as_dict()
        trace_url = build_trace_url(self._trace_base_url, result.trace_context_id())
        if trace_url:
            main_entry["trace_url"] = trace_url
        self._results.append(main_entry)
        # A single synthesized Step (no name) is already fully represented by the
        # case row; only named Steps earn their own flat Step Trail rows.
        for step in result.step_results:
            if step.name is None:
                continue
            step_entry = {
                "name": f"{result.name}:{step.name}",
                "parent_name": result.name,
                "step_name": step.name,
                "success": step.success,
                "agent_id": result.agent_id,
                "request": step.request.to_dict() if step.request is not None else None,
                "response": _response_to_dict(step.response),
                "harness_error": step.harness_error.to_dict()
                if step.harness_error
                else None,
                "duration_seconds": step.duration_seconds,
                "is_step_result": True,
                "context_id": step.context_id,
                "usage": step.usage.as_dict() if step.usage else None,
                "expectation_results": [result.to_dict() for result in step.results],
            }
            step_trace_url = build_trace_url(self._trace_base_url, step.context_id)
            if step_trace_url:
                step_entry["trace_url"] = step_trace_url
            self._results.append(step_entry)
        self._entries.append((case, result))

    def close(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(self._results, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        self._write_markdown()

    def aggregate_runs(self) -> None:
        """No-op — cross-run aggregation is not needed for per-run file output."""

    def _write_markdown(self) -> None:
        if self.path.suffix:
            md_path = self.path.with_suffix(".md")
        else:
            md_path = self.path.with_name(f"{self.path.name}.md")
        lines: list[str] = ["# Evaluation Results", ""]
        for case, result in self._entries:
            lines.append(f"## {case.name}")
            status = "✅ Passed" if result.success else "❌ Failed"
            lines.append(f"- **Status:** {status}")
            if result.agent_id:
                lines.append(f"- **Agent ID:** `{result.agent_id}`")
            if result.duration_seconds is not None:
                lines.append(f"- **Duration:** {result.duration_seconds:.3f}s")
            usage = result.aggregate_usage()
            if usage is not None:
                lines.append(f"- **Usage:** {self._format_usage(usage)}")
            context_id = result.trace_context_id()
            trace_url = build_trace_url(self._trace_base_url, context_id)
            if trace_url:
                lines.append(f"- **Trace:** [{context_id}]({trace_url})")
            # A single synthesized Step (no name) surfaces its per-term verdicts
            # at the case level so a single-message case reads without a Step Trail.
            for step in result.step_results:
                if step.name is None:
                    lines.extend(self._format_checks_markdown(step.results))
            errors = result.error_summary()
            if errors:
                lines.append(f"- **Errors:** {errors}")
            lines.append("")
            named_steps = [s for s in result.step_results if s.name is not None]
            if named_steps:
                lines.append("<details>")
                lines.append(f"<summary>Step Results ({len(named_steps)})</summary>")
                lines.append("")
                for step_result in named_steps:
                    lines.append(f"#### {step_result.name}")
                    step_status = "✅ Passed" if step_result.success else "❌ Failed"
                    lines.append(f"- **Status:** {step_status}")
                    if step_result.duration_seconds is not None:
                        lines.append(
                            f"- **Duration:** {step_result.duration_seconds:.3f}s"
                        )
                    if step_result.usage is not None:
                        lines.append(
                            f"- **Usage:** {self._format_usage(step_result.usage)}"
                        )
                    lines.extend(self._format_checks_markdown(step_result.results))
                    failures = step_result.failure_messages()
                    if failures:
                        lines.append(f"- **Errors:** {'; '.join(failures)}")
                    step_trace_url = build_trace_url(
                        self._trace_base_url, step_result.context_id
                    )
                    if step_trace_url:
                        lines.append(
                            f"- **Trace:** [{step_result.context_id}]({step_trace_url})"
                        )
                    lines.append("")
                    request = step_result.request
                    lines.extend(
                        self._collapsible(
                            "Request", request.to_dict() if request else None
                        )
                    )
                    lines.extend(
                        self._collapsible(
                            "Response", _response_to_dict(step_result.response)
                        )
                    )
                lines.append("</details>")
                lines.append("")
            else:
                single = result.step_results[0] if result.step_results else None
                request = single.request if single is not None else None
                if request is not None:
                    lines.extend(self._collapsible("Input Message", request.to_dict()))
                response = single.response if single is not None else None
                lines.extend(
                    self._collapsible("Output Response", _response_to_dict(response))
                )
        md_path.write_text("\n".join(lines), encoding="utf-8")

    def _collapsible(self, summary: str, payload: dict[str, Any] | None) -> list[str]:
        """A ``<details open>`` block rendering *payload* as a JSON code block."""
        return [
            "<details open>",
            f"<summary>{summary}</summary>",
            "",
            self._format_json_block(payload),
            "",
            "</details>",
            "",
        ]

    @classmethod
    def _format_checks_markdown(cls, results: list[ExpectationResult]) -> list[str]:
        """Render a step's per-term verdicts, in the canonical column order.

        Each expectation reports its own terms; the judge shows its verdict and
        explanation on a pass as well as a fail. Passing scalar checks (the
        always-on state guard, an unbreached duration) stay quiet so the block
        surfaces what an author cares about — declared terms and failures.
        """
        lines: list[str] = []
        for result in results:
            if result.show_on_pass:
                lines.extend(cls._format_judge_markdown(result))
            else:
                lines.extend(cls._format_expectation_markdown(result))
        return lines

    @staticmethod
    def _format_expectation_markdown(result: ExpectationResult) -> list[str]:
        """One bullet of per-term verdicts for a non-judge expectation.

        A labelled term always renders as passed or failed, so a reviewer sees
        exactly which part of a multi-term expectation broke. An unlabelled check
        — a scalar guard (state, duration) or the folded missing-task failure —
        has no term of its own, so it renders only when it fails, as its message.
        The whole bullet is dropped when nothing is worth showing (every scalar
        passed), keeping a clean case quiet.
        """
        terms: list[str] = []
        for check in result.checks:
            icon = "✅" if check.passed else "❌"
            if check.label:
                term = f"{icon} `{check.label}`"
                if not check.passed:
                    term += f" — {check.detail}"
                terms.append(term)
            elif not check.passed:
                terms.append(f"{icon} {check.detail}")
        if not terms:
            return []
        return [f"- **{result.key}:** {', '.join(terms)}"]

    @staticmethod
    def _format_judge_markdown(result: ExpectationResult) -> list[str]:
        """Render the judge result as readable markdown lines.

        The verdict is a labelled bullet with an icon; the LLM's explanation is a
        blockquote so it's visually distinct and easy to scan — shown on a pass
        as well as a fail, so an author sees *why* an output was accepted.
        """
        icon, verdict = ("✅", "PASS") if result.passed else ("❌", "FAIL")
        lines = [f"- **Judge Verdict:** {icon} {verdict}"]
        explanation = result.checks[0].detail if result.checks else ""
        if explanation:
            lines.append(f"  > {explanation}")
        return lines

    @staticmethod
    def _format_usage(usage: UsageMetrics) -> str:
        parts = []
        for key, value in usage.as_dict().items():
            if value is None:
                continue
            if isinstance(value, float):
                value = round(value, 6)
            parts.append(f"{key}={value}")
        return ", ".join(parts)

    @staticmethod
    def _format_json_block(payload: Any) -> str:
        pretty = (
            json.dumps(payload, ensure_ascii=False, indent=2, default=str)
            if payload is not None
            else "null"
        )
        return f"```json\n{pretty}\n```"
