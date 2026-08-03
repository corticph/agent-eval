"""Tests for the must_include_json partial-JSON validator."""

from __future__ import annotations

from agent_evals.expectations import evaluate_response, parse_expectations

_DATA_PART = {
    "locale": "en",
    "items": [
        {
            "type": "TEMP",
            "value": 38.5,
            "unitName": "°C",
            "timestamp": "2024-01-01T00:00:00Z",
        },
        {
            "type": "SYS",
            "value": 120,
            "unitName": "mmHg",
            "timestamp": "2024-01-01T00:00:00Z",
        },
        {
            "type": "DIAS",
            "value": 80,
            "unitName": "mmHg",
            "timestamp": "2024-01-01T00:00:00Z",
        },
        {
            "type": "PULSE",
            "value": 90,
            "unitName": "Bpm",
            "timestamp": "2024-01-01T00:00:00Z",
        },
    ],
}

_RESPONSE = {"task": {"artifacts": [{"parts": [{"kind": "data", "data": _DATA_PART}]}]}}


def _eval(pattern, response=_RESPONSE):
    results = evaluate_response(
        parse_expectations({"must_include_json": [pattern]}), response
    )
    ok = all(r.passed for r in results)
    details = [c.detail for r in results for c in r.checks if not c.passed]
    return ok, details


def test_all_four_items_match():
    ok, errors = _eval(
        {
            "items": [
                {"type": "TEMP", "value": 38.5, "unitName": "°C"},
                {"type": "SYS", "value": 120, "unitName": "mmHg"},
                {"type": "DIAS", "value": 80, "unitName": "mmHg"},
                {"type": "PULSE", "value": 90, "unitName": "Bpm"},
            ]
        }
    )
    assert ok, errors


def test_partial_dict_match_ignores_extra_fields():
    # Only checking type+value — unitName and timestamp are not declared.
    ok, errors = _eval({"items": [{"type": "TEMP", "value": 38.5}]})
    assert ok, errors


def test_wrong_value_fails():
    ok, errors = _eval({"items": [{"type": "TEMP", "value": 99.0}]})
    assert not ok
    assert any("no data part matched" in e for e in errors)


def test_type_mismatch_float_vs_string_fails():
    ok, errors = _eval({"items": [{"type": "TEMP", "value": "38.5"}]})
    assert not ok


def test_missing_item_type_fails():
    ok, errors = _eval({"items": [{"type": "WEIGHT", "value": 70}]})
    assert not ok


def test_top_level_scalar_field():
    ok, errors = _eval({"locale": "en"})
    assert ok, errors


def test_no_data_parts_fails():
    ok, errors = _eval({"items": []}, response={"task": {"artifacts": []}})
    assert not ok
    assert any("no data parts" in e for e in errors)
