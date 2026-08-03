"""Tests for the deterministic cited_pubmed_ids expectation.

Given a synthetic response (an agent summary + a retrieved-source data part),
does the expectation pass when every cited PMID was retrieved and fail (naming
the offender) when a citation is fabricated? No LLM, no mocking.
"""

from __future__ import annotations

import json

from agent_evals.expectations import parse_expectations
from agent_evals.expectations.base import EvaluationContext


def _response(summary: str, retrieved_pmids: list[str]) -> dict:
    """Build a response: one text part (the summary) + one $.response carrier.

    The carrier mirrors the live shape — a data part ``{"response": "<JSON
    string of web_result objects>"}`` — with a pubmed URL per retrieved PMID.
    """
    web_results = [
        {"url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}", "title": f"Study {pmid}"}
        for pmid in retrieved_pmids
    ]
    return {
        "task": {
            "artifacts": [
                {
                    "parts": [
                        {"text": summary},
                        {"data": {"response": json.dumps(web_results)}},
                    ]
                }
            ]
        }
    }


def _eval(summary: str, retrieved_pmids: list[str]):
    expectations = parse_expectations({"cited_pubmed_ids": True})
    ctx = EvaluationContext.from_response(_response(summary, retrieved_pmids))
    return [e.resolve(ctx) for e in expectations]


def test_all_cited_pmids_present_passes():
    results = _eval(
        "The evidence (PMID 111, PMID: 222) supports this. Source: PubMed.",
        ["111", "222", "333"],
    )
    assert all(r.passed for r in results), results


def test_fabricated_pmid_fails_and_names_it():
    results = _eval(
        "See PMID 999 for the definitive result. Source: PubMed.",
        ["111", "222"],
    )
    cited_result = [r for r in results if r.key == "cited_pubmed_ids"]
    assert cited_result
    assert not cited_result[0].passed
    assert any("999" in c.detail for c in cited_result[0].checks)


def test_cited_via_url_is_matched():
    results = _eval(
        "See https://pubmed.ncbi.nlm.nih.gov/111 for details.",
        ["111"],
    )
    assert all(r.passed for r in results), results


def test_no_citations_passes():
    results = _eval("No relevant results were found. Source: PubMed.", ["111"])
    assert all(r.passed for r in results), results


def test_citation_with_no_sources_fails():
    results = _eval("The key paper is PMID 123.", [])
    cited_result = [r for r in results if r.key == "cited_pubmed_ids"]
    assert cited_result
    assert not cited_result[0].passed
    assert any("123" in c.detail for c in cited_result[0].checks)
