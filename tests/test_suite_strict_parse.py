"""Every suite under ``evals/`` must parse cleanly through the strict registry.

Locks in the migration: an unregistered top-level ``expectations:`` key (silently
ignored by the old field-based config) is now a load error, so this guard fails
the moment a future edit introduces one instead of letting it pass vacuously.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pytest
import yaml

from agent_evals.expectations import parse_expectations

EVALS_DIR = Path(__file__).resolve().parent.parent / "evals"

YAML_FILES = sorted(p for p in EVALS_DIR.rglob("*.yaml") if not p.name.startswith("_"))


def _expectation_blocks(raw: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield every ``expectations:`` block a suite declares, across both shapes."""
    for case in raw.get("evals", []):
        if not isinstance(case, dict):
            continue
        for step in case.get("steps") or []:
            if isinstance(step, dict) and isinstance(step.get("expectations"), dict):
                yield step["expectations"]
        if isinstance(case.get("expectations"), dict):
            yield case["expectations"]


@pytest.mark.parametrize(
    "yaml_path",
    YAML_FILES,
    ids=[str(p.relative_to(EVALS_DIR)) for p in YAML_FILES],
)
def test_suite_parses_through_registry(yaml_path: Path) -> None:
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    for block in _expectation_blocks(raw):
        # Raises on an unregistered key — the strict-parse guarantee under test.
        # Placeholder tokens (``{{var}}``) are stored verbatim, so unresolved
        # variables don't affect key validation.
        parse_expectations(block)
