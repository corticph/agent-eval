"""Parse seam for the unified Case model: what a Suite normalizes into.

Both authoring shapes — a flat ``message:``/``expectations:`` case and a
``steps:`` list — load through ``load_suite`` into one ``EvaluationCase`` holding
an ordered ``list[Step]`` (length ≥ 1). The shape is inferred from the file, so
no ``type:`` discriminator is read. Asserted from the outside, through the same
loader the suite already uses (prior art: ``test_folder_defaults``,
``test_eval_names``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_evals.loader import Step, load_suite


def _write_suite(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "suite.yaml"
    path.write_text(body, encoding="utf-8")
    return path


FLAT_CASE = """\
name: flat_suite
evals:
  - name: greet
    agent:
      name: Agent
    message:
      message:
        parts:
          - kind: text
            text: hello
    expectations:
      must_include: [hi]
"""

STEPS_CASE = """\
name: steps_suite
evals:
  - name: conversation
    agent:
      name: Agent
    steps:
      - name: intake
        message:
          message:
            parts:
              - kind: text
                text: start
      - name: follow_up
        message:
          message:
            parts:
              - kind: text
                text: more
      - name: wrap_up
        message:
          message:
            parts:
              - kind: text
                text: done
"""


def test_flat_case_normalizes_to_one_step(tmp_path: Path) -> None:
    (case,) = load_suite(_write_suite(tmp_path, FLAT_CASE)).cases
    assert len(case.steps) == 1
    assert isinstance(case.steps[0], Step)


def test_synthesized_single_step_has_no_name(tmp_path: Path) -> None:
    (case,) = load_suite(_write_suite(tmp_path, FLAT_CASE)).cases
    # A single-message case stays terse: its lone Step carries the case's
    # expectations but needs no name of its own.
    assert case.steps[0].name is None
    # Expectations parse straight into the canonical list; the declared
    # must_include rides its own registered expectation.
    must_include = next(
        e for e in case.steps[0].expectations if e.key == "must_include"
    )
    assert must_include.phrases == ["hi"]


def test_steps_case_yields_one_step_per_entry(tmp_path: Path) -> None:
    (case,) = load_suite(_write_suite(tmp_path, STEPS_CASE)).cases
    assert len(case.steps) == 3
    assert all(isinstance(step, Step) for step in case.steps)
    assert [step.name for step in case.steps] == ["intake", "follow_up", "wrap_up"]


def test_case_with_neither_message_nor_steps_raises(tmp_path: Path) -> None:
    body = """\
name: empty_suite
evals:
  - name: nothing
    agent:
      name: Agent
"""
    with pytest.raises(ValueError):
        load_suite(_write_suite(tmp_path, body))


def test_empty_steps_list_raises(tmp_path: Path) -> None:
    body = """\
name: empty_steps_suite
evals:
  - name: nothing
    agent:
      name: Agent
    steps: []
"""
    with pytest.raises(ValueError):
        load_suite(_write_suite(tmp_path, body))


def test_multi_step_entry_requires_a_name(tmp_path: Path) -> None:
    body = """\
name: unnamed_step_suite
evals:
  - name: conversation
    agent:
      name: Agent
    steps:
      - message:
          message:
            parts:
              - kind: text
                text: start
"""
    with pytest.raises(ValueError):
        load_suite(_write_suite(tmp_path, body))


def test_retired_turns_shape_fails_strict_parse(tmp_path: Path) -> None:
    # The retired `turns:`-based authoring shape gets no translation: its
    # cases carry neither `message` nor `steps`, so loading stops at the parse
    # seam with the strict-parse error — even though such files also lack the
    # canonical `name`/`agent` envelope keys, the shape diagnosis wins.
    body = """\
name: old_shape_suite
agent_config:
  name: Agent
evals:
  - id: greet
    turns:
      - input:
          text: hello
        expected_output:
          must_include:
            text: [hi]
"""
    with pytest.raises(
        ValueError, match="must declare either a 'message' or a 'steps' list"
    ):
        load_suite(_write_suite(tmp_path, body))


def test_case_without_name_fails_with_clear_error(tmp_path: Path) -> None:
    body = FLAT_CASE.replace("  - name: greet\n", "  - description: nameless\n")
    with pytest.raises(ValueError, match="every eval must declare a 'name'"):
        load_suite(_write_suite(tmp_path, body))


def test_stray_type_key_is_ignored(tmp_path: Path) -> None:
    # An older Suite that still carries the retired ``type:`` discriminator
    # keeps loading — the key is a harmless no-op, and the shape is inferred.
    body = FLAT_CASE.replace("  - name: greet\n", "  - name: greet\n    type: single\n")
    (case,) = load_suite(_write_suite(tmp_path, body)).cases
    assert len(case.steps) == 1
