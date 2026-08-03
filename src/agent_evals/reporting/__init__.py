"""The reporting layer: the Sinks the runner drives, and the trace-link helper.

The :class:`Sink` protocol itself lives in :mod:`agent_evals.results` — the
neutral execute→report contract the runner produces against. This package holds
the concrete consumers of that contract.

The Opik sink is exposed lazily (via module ``__getattr__``) so a bare ``run``
never imports the (heavy, but always-installed) ``opik`` package: importing this
package pulls in the file sink and the trace helper only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .file import FileSink
from .multi_run import (
    format_stats_markdown,
    write_combined_json,
    write_stats_csv,
)
from .stats_sink import StatsSink
from .trace import build_trace_url

if TYPE_CHECKING:
    from .opik import ExpectationMetric, OpikSink, suite_to_dataset_items

# Names served on demand from the Opik sink module, which imports the heavy
# ``opik`` package; touching one of these is the opt-in that pulls it in.
_LAZY = frozenset({"OpikSink", "ExpectationMetric", "suite_to_dataset_items"})

__all__ = [
    "FileSink",
    "StatsSink",
    "build_trace_url",
    "format_stats_markdown",
    "write_combined_json",
    "write_stats_csv",
    "OpikSink",
    "ExpectationMetric",
    "suite_to_dataset_items",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from . import opik

        return getattr(opik, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
