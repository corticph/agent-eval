"""Tests for _defaults.yaml folder-level globals merging."""

from __future__ import annotations

from pathlib import Path


from agent_evals.loader import load_suite


def _write(tmp_path: Path, rel: str, content: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


MINIMAL_SUITE = """\
name: test_suite
evals:
  - name: case1
    message:
      message:
        parts:
          - kind: text
            text: hello
"""

DEFAULTS = """\
globals:
  agent:
    name: DefaultAgent
    description: from defaults
  message:
    message:
      messageId: ""
      role: user
      kind: message
"""


def test_defaults_applied_when_no_suite_globals(tmp_path: Path) -> None:
    _write(tmp_path, "_defaults.yaml", DEFAULTS)
    suite_path = _write(tmp_path, "suite.yaml", MINIMAL_SUITE)
    suite = load_suite(suite_path)
    agent = suite.cases[0].agent
    assert agent.name == "DefaultAgent"
    assert agent.description == "from defaults"


def test_suite_globals_override_defaults(tmp_path: Path) -> None:
    _write(tmp_path, "_defaults.yaml", DEFAULTS)
    suite_path = _write(
        tmp_path,
        "suite.yaml",
        MINIMAL_SUITE + "globals:\n  agent:\n    name: OverrideAgent\n",
    )
    suite = load_suite(suite_path)
    agent = suite.cases[0].agent
    assert agent.name == "OverrideAgent"
    # Keys from defaults not overridden are still present.
    assert agent.description == "from defaults"


def test_parent_defaults_used_as_fallback(tmp_path: Path) -> None:
    # No _defaults.yaml in the suite folder -> fall back to the parent's.
    _write(tmp_path, "pyproject.toml", "")  # mark tmp_path as the project root
    _write(tmp_path, "_defaults.yaml", DEFAULTS)
    suite_path = _write(tmp_path, "problem/suite.yaml", MINIMAL_SUITE)
    suite = load_suite(suite_path)
    assert suite.cases[0].agent.name == "DefaultAgent"


def test_suite_folder_defaults_win_over_parent(tmp_path: Path) -> None:
    # Ancestors are deep-merged, nearest wins per key.
    _write(tmp_path, "pyproject.toml", "")  # mark tmp_path as the project root
    _write(tmp_path, "_defaults.yaml", DEFAULTS)
    _write(
        tmp_path,
        "problem/_defaults.yaml",
        "globals:\n  agent:\n    name: ProblemAgent\n",
    )
    suite_path = _write(tmp_path, "problem/suite.yaml", MINIMAL_SUITE)
    suite = load_suite(suite_path)
    agent = suite.cases[0].agent
    # Nearest file overrides the key it sets...
    assert agent.name == "ProblemAgent"
    # ...but keys only the parent sets are still deep-merged in.
    assert agent.description == "from defaults"


def test_no_defaults_file_still_works(tmp_path: Path) -> None:
    suite_content = MINIMAL_SUITE + "globals:\n  agent:\n    name: StandaloneAgent\n"
    suite_path = _write(tmp_path, "suite.yaml", suite_content)
    suite = load_suite(suite_path)
    assert suite.cases[0].agent.name == "StandaloneAgent"
