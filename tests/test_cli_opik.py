"""The ``run --opik`` CLI surface.

There is one way to run a Suite: ``run``. The separate Opik subcommand is
gone; ``--opik`` opts a run into the Opik sink (with dataset/experiment name
overrides), the ``opik`` package is imported only when the flag is passed, and
the process exit code derives from the runner's results alone.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest

import agent_evals.__main__ as cli
from agent_evals.__main__ import _build_parser
from agent_evals.results import EvaluationResult

# --- flag parsing ------------------------------------------------------------


def test_opik_defaults_off() -> None:
    args = _build_parser().parse_args(["run", "suite.yaml", "--env", "dev-weu"])
    assert args.opik is False


def test_opik_flag_parses() -> None:
    args = _build_parser().parse_args(
        ["run", "suite.yaml", "--env", "dev-weu", "--opik"]
    )
    assert args.opik is True


def test_dataset_and_experiment_name_overrides_parse() -> None:
    args = _build_parser().parse_args(
        [
            "run",
            "suite.yaml",
            "--env",
            "dev-weu",
            "--opik",
            "--dataset-name",
            "custom-ds",
            "--experiment-name",
            "custom-exp",
        ]
    )
    assert args.dataset_name == "custom-ds"
    assert args.experiment_name == "custom-exp"


def test_name_overrides_default_to_none() -> None:
    args = _build_parser().parse_args(["run", "suite.yaml", "--env", "dev-weu"])
    assert args.dataset_name is None
    assert args.experiment_name is None
    assert args.tag is None


def test_tag_flag_parses_repeatable() -> None:
    args = _build_parser().parse_args(
        [
            "run",
            "suite.yaml",
            "--env",
            "dev-weu",
            "--opik",
            "--tag",
            "baseline",
            "--tag",
            "gpt-4o",
        ]
    )
    assert args.tag == ["baseline", "gpt-4o"]


@pytest.mark.parametrize("flag", ["--dataset-name", "--experiment-name", "--tag"])
def test_name_override_without_opik_is_rejected(flag: str) -> None:
    args = _build_parser().parse_args(
        ["run", "suite.yaml", "--env", "dev-weu", flag, "x"]
    )
    with pytest.raises(SystemExit) as excinfo:
        cli._handle_run(args)
    assert "--opik" in str(excinfo.value)


# --- the removed subcommand --------------------------------------------------


def test_run_opik_subcommand_no_longer_parses() -> None:
    with pytest.raises(SystemExit) as excinfo:
        _build_parser().parse_args(["run-opik", "suite.yaml", "--env", "dev-weu"])
    assert excinfo.value.code != 0


# --- the lazy import ---------------------------------------------------------


def test_cli_module_does_not_eagerly_import_opik() -> None:
    # A bare ``run`` loads ``__main__`` but must not pull in the optional
    # ``opik`` package. A fresh interpreter, so this suite's imports don't taint
    # the check.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import agent_evals.__main__, sys; "
            "assert 'opik' not in sys.modules, 'opik imported eagerly'",
        ],
        check=True,
    )


# --- the exit code source ----------------------------------------------------


@pytest.fixture
def _stub_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the suite pipeline so ``_handle_run`` exercises only the exit path.

    The exit code must come from the runner's results, so everything up to
    ``run_suite`` is inert and each test supplies its own results.
    """
    from pathlib import Path

    class _Suite:
        name = "stub"
        cases: list[Any] = [object()]

    monkeypatch.setattr(cli, "load_suite", lambda _p: _Suite())
    monkeypatch.setattr(cli, "_apply_runtime_overrides", lambda *_: None)
    monkeypatch.setattr(cli, "_resolve_suite_paths", lambda _a: [Path("suite.yaml")])
    monkeypatch.setattr(cli, "FileSink", lambda *a, **k: object())

    class _Client:
        def __init__(self, *_a: Any, **_k: Any) -> None: ...
        def close(self) -> None: ...

    monkeypatch.setattr(cli, "AgentClient", _Client)


def _result(name: str, *, success: bool) -> EvaluationResult:
    return EvaluationResult(name=name, success=success, agent_id="a", step_results=[])


def test_exit_code_comes_from_runner_results_failure(
    monkeypatch: pytest.MonkeyPatch, _stub_pipeline: None
) -> None:
    monkeypatch.setattr(cli, "run_suite", lambda *a, **k: [_result("c", success=False)])
    args = _build_parser().parse_args(["run", "suite.yaml", "--env", "dev-weu"])
    assert cli._handle_run(args) == 2


def test_exit_code_comes_from_runner_results_success(
    monkeypatch: pytest.MonkeyPatch, _stub_pipeline: None
) -> None:
    monkeypatch.setattr(cli, "run_suite", lambda *a, **k: [_result("c", success=True)])
    args = _build_parser().parse_args(["run", "suite.yaml", "--env", "dev-weu"])
    assert cli._handle_run(args) == 0
