"""The reporter renders the Step Trail uniformly, regardless of Step count.

A single-Step Case reads as trivial single-message output — no per-Step counter,
labelled by the Case name — while a multi-step Case renders each named Step's own
section. Asserted through the ``FileSink`` markdown seam (prior art:
``test_build_trace_url``).
"""

from __future__ import annotations

import json

from agent_evals.errors import ErrorCode, EvalError
from agent_evals.expectations import CheckResult, ExpectationResult
from agent_evals.loader import EvaluationCase, Step
from agent_evals.reporting import FileSink
from agent_evals.results import EvaluationResult, StepResult
from agent_evals.schemas.agent import Agent
from agent_evals.schemas.message import MessagePayload


def _write(tmp_path, case: EvaluationCase, result: EvaluationResult) -> None:
    writer = FileSink(tmp_path / "out.json")
    writer.write(case, result)
    writer.close()


def _markdown(tmp_path, case: EvaluationCase, result: EvaluationResult) -> str:
    _write(tmp_path, case, result)
    return (tmp_path / "out.md").read_text(encoding="utf-8")


def _json_entries(
    tmp_path, case: EvaluationCase, result: EvaluationResult
) -> list[dict]:
    _write(tmp_path, case, result)
    return json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))


def test_single_step_result_omits_step_counter_and_labels_by_case_name(
    tmp_path,
) -> None:
    # A single-message Case is one synthesized (unnamed) Step.
    case = EvaluationCase(
        name="my_case",
        agent=Agent(name="a"),
        steps=[Step(message=MessagePayload())],
    )
    result = EvaluationResult(
        name="my_case",
        success=True,
        agent_id="agent-1",
        step_results=[
            StepResult(name=None, success=True, request=MessagePayload(), response=None)
        ],
    )

    markdown = _markdown(tmp_path, case, result)

    # The case name is the only heading; the lone Step renders inline (its
    # Output Response), never as its own counted "Step Results" section.
    assert "## my_case" in markdown
    assert "Step Results" not in markdown
    assert "#### " not in markdown
    assert "Output Response" in markdown


def test_multi_step_result_renders_each_named_section(tmp_path) -> None:
    case = EvaluationCase(
        name="flow",
        agent=Agent(name="a"),
        steps=[
            Step(message=MessagePayload(), name="intake"),
            Step(message=MessagePayload(), name="wrap"),
        ],
    )
    result = EvaluationResult(
        name="flow",
        success=True,
        agent_id="agent-1",
        step_results=[
            StepResult(
                name="intake", success=True, request=MessagePayload(), response=None
            ),
            StepResult(
                name="wrap", success=True, request=MessagePayload(), response=None
            ),
        ],
    )

    markdown = _markdown(tmp_path, case, result)

    assert "## flow" in markdown
    assert "#### intake" in markdown
    assert "#### wrap" in markdown


def test_unlabelled_checks_render_without_empty_term_spans(tmp_path) -> None:
    # A folded missing-task failure leaves ``expected_state`` with two unlabelled
    # checks (the passing state assertion + the failed guard). Unlabelled checks
    # carry no term, so they must render as their message, never as empty ``.
    folded = ExpectationResult(
        "expected_state",
        [
            CheckResult("", True, "passed"),
            CheckResult("", False, "expected task id in response to continue the flow"),
        ],
    )
    case = EvaluationCase(
        name="thread",
        agent=Agent(name="a"),
        steps=[Step(message=MessagePayload(), name="ask")],
    )
    result = EvaluationResult(
        name="thread",
        success=False,
        agent_id="agent-1",
        step_results=[
            StepResult(
                name="ask",
                success=False,
                request=MessagePayload(),
                response=None,
                results=[folded],
            )
        ],
    )

    markdown = _markdown(tmp_path, case, result)

    state_line = next(
        line
        for line in markdown.splitlines()
        if line.startswith("- **expected_state:**")
    )
    # The unlabelled checks render as their message, with no inline-code spans.
    assert (
        state_line
        == "- **expected_state:** ❌ expected task id in response to continue the flow"
    )
    assert "`" not in state_line


def _failing_check_result() -> tuple[EvaluationCase, EvaluationResult]:
    missed = ExpectationResult(
        "must_include",
        [
            CheckResult("refund", True, "passed"),
            CheckResult("apology", False, "missing required phrase: 'apology'"),
        ],
    )
    case = EvaluationCase(
        name="flow",
        agent=Agent(name="a"),
        steps=[Step(message=MessagePayload(), name="ask")],
    )
    result = EvaluationResult(
        name="flow",
        success=False,
        agent_id="agent-1",
        step_results=[
            StepResult(
                name="ask",
                success=False,
                request=MessagePayload(),
                response=None,
                results=[missed],
            )
        ],
    )
    return case, result


def test_json_step_entries_for_a_check_failure_carry_no_error_objects(tmp_path) -> None:
    case, result = _failing_check_result()

    entries = _json_entries(tmp_path, case, result)

    # A failed check lives once, inside the step's expectation results — the
    # harness_error slot stays null in both the case row's embedded trail and
    # the flat step row (only Harness Failures populate it), and the old
    # list-valued errors key is gone entirely.
    step_row = next(e for e in entries if e.get("is_step_result"))
    assert step_row["harness_error"] is None
    assert "errors" not in step_row
    embedded = next(e for e in entries if not e.get("is_step_result"))["step_results"][
        0
    ]
    assert embedded["harness_error"] is None
    assert "errors" not in embedded
    failed = [
        check
        for term in embedded["expectation_results"]
        for check in term["checks"]
        if not check["passed"]
    ]
    assert [check["label"] for check in failed] == ["apology"]


def test_json_harness_failed_step_carries_typed_failure(tmp_path) -> None:
    case = EvaluationCase(
        name="dead",
        agent=Agent(name="a"),
        steps=[Step(message=MessagePayload())],
    )
    result = EvaluationResult(
        name="dead",
        success=False,
        agent_id=None,
        step_results=[
            StepResult(
                name=None,
                success=False,
                request=None,
                response=None,
                harness_error=EvalError(
                    ErrorCode.EVAL_TIMEOUT, "eval exceeded wall-clock timeout of 5s"
                ),
            )
        ],
    )

    entries = _json_entries(tmp_path, case, result)

    embedded = entries[0]["step_results"][0]
    assert embedded["harness_error"]["code"] == "eval_timeout"
    assert "timeout" in embedded["harness_error"]["message"]


def test_markdown_errors_lines_derive_from_failed_checks(tmp_path) -> None:
    case, result = _failing_check_result()

    markdown = _markdown(tmp_path, case, result)

    # The case-level summary names the step, the expectation key, and the
    # failed check's detail; the step section repeats it without the step name.
    assert (
        "- **Errors:** ask: must_include: missing required phrase: 'apology'"
        in markdown
    )
    assert "- **Errors:** must_include: missing required phrase: 'apology'" in markdown


def test_markdown_renders_a_step_harness_failure(tmp_path) -> None:
    case = EvaluationCase(
        name="flaky",
        agent=Agent(name="a"),
        steps=[Step(message=MessagePayload(), name="ask")],
    )
    result = EvaluationResult(
        name="flaky",
        success=False,
        agent_id="agent-1",
        step_results=[
            StepResult(
                name="ask",
                success=False,
                request=MessagePayload(),
                response=None,
                harness_error=EvalError(
                    ErrorCode.NETWORK_ERROR, "connection reset by peer"
                ),
            )
        ],
    )

    markdown = _markdown(tmp_path, case, result)

    assert "- **Errors:** ask: connection reset by peer" in markdown
    assert "- **Errors:** connection reset by peer" in markdown


def test_markdown_harness_failure_follows_failed_checks(tmp_path) -> None:
    # A step can carry both a miss and a Harness Failure (e.g. the failure hit
    # after evaluation); the summary lists failed checks first, then the typed
    # failure — beside each other, never merged into one entry.
    missed = ExpectationResult(
        "must_include",
        [CheckResult("apology", False, "missing required phrase: 'apology'")],
    )
    case = EvaluationCase(
        name="flow",
        agent=Agent(name="a"),
        steps=[Step(message=MessagePayload(), name="ask")],
    )
    result = EvaluationResult(
        name="flow",
        success=False,
        agent_id="agent-1",
        step_results=[
            StepResult(
                name="ask",
                success=False,
                request=MessagePayload(),
                response=None,
                results=[missed],
                harness_error=EvalError(
                    ErrorCode.NETWORK_ERROR, "connection reset by peer"
                ),
            )
        ],
    )

    markdown = _markdown(tmp_path, case, result)

    assert (
        "- **Errors:** ask: must_include: missing required phrase: 'apology'; "
        "ask: connection reset by peer" in markdown
    )


def test_markdown_passing_case_renders_no_errors_line(tmp_path) -> None:
    passing = ExpectationResult("must_include", [CheckResult("refund", True, "passed")])
    case = EvaluationCase(
        name="clean",
        agent=Agent(name="a"),
        steps=[Step(message=MessagePayload(), name="ask")],
    )
    result = EvaluationResult(
        name="clean",
        success=True,
        agent_id="agent-1",
        step_results=[
            StepResult(
                name="ask",
                success=True,
                request=MessagePayload(),
                response=None,
                results=[passing],
            )
        ],
    )

    markdown = _markdown(tmp_path, case, result)

    assert "**Errors:**" not in markdown
