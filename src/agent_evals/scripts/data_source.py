"""Unified data source for eval scripts: Opik or local JSON files.

Wraps the data-access functions the scripts need (list experiments, get items,
extract scores/case-name/trace-url) behind a single interface so scripts can
work with either source via ``--source opik`` (default) or ``--source local``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from . import local_store


class DataSource:
    """Abstract data source for experiment data."""

    def list_experiments(
        self, *, name: str | None = None, env: str | None = None,
        tag: str | None = None, limit: int = 500,
    ) -> dict[str, SimpleNamespace]:
        raise NotImplementedError

    def get_items(self, exp: SimpleNamespace) -> list:
        raise NotImplementedError

    @staticmethod
    def score_map(item: Any) -> dict[str, tuple[float, str]]:
        out: dict[str, tuple[float, str]] = {}
        for fs in item.feedback_scores or []:
            out[fs["name"]] = (fs["value"], fs.get("reason") or "")
        return out

    @staticmethod
    def case_name(item: Any) -> str:
        data = item.dataset_item_data or {}
        return data.get("name") or "(unnamed)"

    @staticmethod
    def trace_url(item: Any) -> str | None:
        out = item.evaluation_task_output or {}
        return out.get("trace_url")

    @staticmethod
    def available_scores(item: Any) -> list[str]:
        return sorted(fs["name"] for fs in item.feedback_scores or [])

    def has_tag(self, exp: SimpleNamespace, tag: str) -> bool:
        return tag in (exp.tags or [])


class OpikSource(DataSource):
    """Data source backed by Opik (the default)."""

    def __init__(self) -> None:
        from .compare_experiments import _make_client
        self._client = _make_client()

    def list_experiments(
        self, *, name: str | None = None, env: str | None = None,
        tag: str | None = None, limit: int = 500,
    ) -> dict[str, SimpleNamespace]:
        from .compare_experiments import _list_experiments, _env_of
        exps = _list_experiments(self._client, name=name, limit=limit)
        if env is not None:
            exps = [e for e in exps if _env_of(e) == env]
        if tag is not None:
            exps = [e for e in exps if self.has_tag(e, tag)]
        by_name: dict[str, SimpleNamespace] = {}
        for e in exps:
            by_name.setdefault(e.name, []).append(e)
        chosen: dict[str, SimpleNamespace] = {}
        for exp_name, lst in by_name.items():
            lst.sort(key=lambda e: e.created_at or "", reverse=True)
            if len(lst) > 1:
                print(f"Warning: {len(lst)} experiments named {exp_name!r}; using newest.")
            chosen[exp_name] = lst[0]
        return chosen

    def get_items(self, exp: SimpleNamespace) -> list:
        from .compare_experiments import _get_items
        return _get_items(self._client, exp.id)


class LocalSource(DataSource):
    """Data source backed by local ``results/*.json`` files."""

    def __init__(self, results_dir: Path | list[Path]) -> None:
        if isinstance(results_dir, (list, tuple)):
            self._results_dirs = list(results_dir)
        else:
            self._results_dirs = [results_dir]

    def list_experiments(
        self, *, name: str | None = None, env: str | None = None,
        tag: str | None = None, limit: int = 500,
    ) -> dict[str, SimpleNamespace]:
        # Scan all results directories and merge; per-directory dedup happens
        # inside local_store.list_experiments, then we merge across dirs with
        # the newest file winning (same merge logic as within a single dir).
        all_exps: dict[str, list[SimpleNamespace]] = {}
        for rd in self._results_dirs:
            if not rd.is_dir():
                continue
            exps = local_store.list_experiments(rd, env=env, name=name, tag=tag)
            for suite_name, exp in exps.items():
                all_exps.setdefault(suite_name, []).append(exp)

        # Merge across dirs: newest wins
        chosen: dict[str, SimpleNamespace] = {}
        for suite_name, lst in all_exps.items():
            lst.sort(key=lambda e: e.created_at or "", reverse=True)
            if len(lst) > 1:
                print(f"Warning: {len(lst)} result files for suite {suite_name!r}; using newest.")
            chosen[suite_name] = lst[0]
        return dict(list(chosen.items())[:limit])

    def get_items(self, exp: SimpleNamespace) -> list:
        return local_store.get_items(exp)

    def has_tag(self, exp: SimpleNamespace, tag: str) -> bool:
        return local_store.has_tag(exp, tag)


def make_source(source: str = "opik", *, results_dir: str | list[str] | None = None) -> DataSource:
    """Create a data source by name.

    ``opik`` (default) connects to Opik.  ``local`` reads from
    ``results_dir`` (default: ``results/``).  ``results_dir`` may be a
    single path or a list of paths to scan multiple directories.
    """
    if source == "local":
        if isinstance(results_dir, list):
            dirs = [Path(d) for d in results_dir]
        elif results_dir:
            dirs = [Path(results_dir)]
        else:
            dirs = [Path("results")]
        found = [d for d in dirs if d.is_dir()]
        if not found:
            dir_list = ", ".join(str(d) for d in dirs)
            raise SystemExit(
                f"Local source requires a results directory. "
                f"None of [{dir_list}] exist. Pass --results-dir <path>."
            )
        return LocalSource(found)
    return OpikSource()
