"""Shared infrastructure used across expectation types.

This package holds helpers that are not expectations themselves — each
expectation type lives in its own module at the parent level — but are shared
by several of them.  Import the submodules directly:

    from .shared.utils import extract_source_at
"""
