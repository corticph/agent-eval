"""Tests for the jsonpath data-part assertions."""

from __future__ import annotations

from agent_evals.expectations import evaluate_response, parse_expectations


def _run(block: dict) -> tuple[bool, list[str]]:
    """Resolve *block* against RESPONSE into ``(passed, failure details)``.

    The seam now returns per-term ``ExpectationResult``s; a caller wanting the
    old ``(ok, errors)`` shape derives it from the failed checks.
    """
    results = evaluate_response(parse_expectations(block), RESPONSE)
    ok = all(r.passed for r in results)
    details = [c.detail for r in results for c in r.checks if not c.passed]
    return ok, details


# Mirrors a live observations output: the data part is duplicated.
_DATA_PART = {
    "items": [
        {
            "timestamp": "2026-06-03T10:20:00Z",
            "type": "TEMP",
            "unitName": "°C",
            "value": 38.5,
        },
        {
            "timestamp": "2026-06-03T10:20:00Z",
            "type": "SYS",
            "unitName": "mmHg",
            "value": 120,
        },
        {
            "timestamp": "2026-06-03T10:20:00Z",
            "type": "DIAS",
            "unitName": "mmHg",
            "value": 80,
        },
        {
            "timestamp": "2026-06-03T10:20:00Z",
            "type": "PULSE",
            "unitName": "Bpm",
            "value": 90,
        },
    ],
    "locale": "en",
}
RESPONSE = {
    "task": {
        "artifacts": [
            {
                "parts": [
                    {"kind": "text", "text": "Clinical Observations Extracted: ..."},
                    {"kind": "data", "data": _DATA_PART},
                    {"kind": "data", "data": _DATA_PART},
                ]
            }
        ]
    }
}


def _eval(assertions: list[dict]) -> tuple[bool, list[str]]:
    return _run({"jsonpath": assertions})


def test_length_and_equals_and_contains():
    ok, errors = _eval(
        [
            {"path": "$.items", "length": 4},  # collection-aware: array has 4 elements
            {"path": "$.items[*]", "count": 4},  # raw match count of elements
            {"path": "$.locale", "equals": "en"},
            {"path": "$.items[*].type", "contains": ["TEMP", "SYS", "DIAS", "PULSE"]},
        ]
    )
    assert ok, errors


def test_wrong_length_fails():
    ok, errors = _eval([{"path": "$.items", "length": 5}])
    assert not ok
    assert any("expected 5" in e for e in errors)


def test_contains_missing_value_fails():
    ok, errors = _eval([{"path": "$.items[*].type", "contains": ["GLUCOSE"]}])
    assert not ok
    assert any("missing 'GLUCOSE'" in e for e in errors)


def test_min_max_numeric_bounds():
    ok, errors = _eval([{"path": "$.items[*].value", "min": 0, "max": 300}])
    assert ok, errors
    bad, _ = _eval([{"path": "$.items[*].value", "max": 100}])
    assert not bad  # 120 exceeds 100


def test_regex_on_timestamp():
    ok, errors = _eval(
        [
            {
                "path": "$.items[*].timestamp",
                "regex": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$",
            }
        ]
    )
    assert ok, errors


def test_exists():
    ok, _ = _eval([{"path": "$.locale", "exists": True}])
    assert ok
    ok2, _ = _eval([{"path": "$.nonexistent", "exists": False}])
    assert ok2
    ok3, _ = _eval([{"path": "$.nonexistent", "exists": True}])
    assert not ok3


def test_filter_expression():
    # Extended jsonpath: select the TEMP item's value via a filter.
    ok, errors = _eval([{"path": "$.items[?(@.type=='TEMP')].value", "equals": 38.5}])
    assert ok, errors


def test_no_data_parts_fails():
    results = evaluate_response(
        parse_expectations({"jsonpath": [{"path": "$.items", "length": 1}]}),
        {"task": {"artifacts": []}},
    )
    ok = all(r.passed for r in results)
    details = [c.detail for r in results for c in r.checks if not c.passed]
    assert not ok
    assert any("no data parts" in e for e in details)
