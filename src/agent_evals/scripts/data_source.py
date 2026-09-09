"""Unified data source for eval scripts: Opik or local JSON files.

Wraps the data-access functions the scripts need (list experiments, get items,
extract scores/case-name/trace-url) behind a single interface so scripts can
work with either source via ``--source opik`` (default) or ``--source local``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import dotenv

from . import local_store

_REPO_ROOT = Path(__file__).resolve().parents[3]
dotenv.load_dotenv(_REPO_ROOT / ".env")

_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0


def _make_opik_client():
    """Create an Opik client (lazy import so local-only runs never need it)."""
    import opik
    from ..environment import OPIK_URL_OVERRIDE_VAR
    from ..reporting.opik_target import resolve_opik_url

    url = resolve_opik_url()
    os.environ.setdefault(OPIK_URL_OVERRIDE_VAR, url)
    project = os.environ.get("OPIK_PROJECT_NAME") or "Agents"
    os.environ["OPIK_PROJECT_NAME"] = project
    return opik.Opik(
        host=url, workspace="default",
        api_key=os.environ.get("OPIK_API_KEY"),
    )


def _retry(fn, *, what: str):
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == _MAX_RETRIES:
                raise
            wait = _RETRY_BACKOFF * attempt
            print(f"  retry {attempt}/{_MAX_RETRIES} ({what}): {exc} — waiting {wait:.0f}s")
            time.sleep(wait)


def _opik_list_experiments(client, name=None, limit=500):
    """List experiments from Opik."""
    http = client._rest_client._client_wrapper.httpx_client
    rows: list[SimpleNamespace] = []
    page, size = 1, 100
    while len(rows) < limit:
        resp = _retry(
            lambda: http.request(
                "v1/private/experiments", method="GET",
                params={"page": page, "size": size, "name": name, "dataset_deleted": False},
            ),
            what=f"list experiments page {page}",
        )
        resp.raise_for_status()
        content = (resp.json() or {}).get("content") or []
        if not content:
            break
        for d in content:
            rows.append(SimpleNamespace(
                id=d.get("id"), name=d.get("name"),
                tags=d.get("tags") or [], created_at=d.get("created_at"),
                metadata=d.get("metadata") or {},
            ))
            if len(rows) >= limit:
                break
        if len(content) < size:
            break
        page += 1
    rows.sort(key=lambda e: e.created_at or "", reverse=True)
    return rows


def _opik_get_items(client, experiment_id):
    """Fetch all experiment items by experiment id."""
    return _retry(
        lambda: client.get_experiment_by_id(experiment_id).get_items(),
        what=f"get items {experiment_id[:8]}",
    )


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
    """Data source backed by Opik, with local-cache-first.

    When a ``results_dir`` is provided, ``list_experiments`` and ``get_items``
    check local JSON files first.  If local results exist for the requested
    tag (or experiment), they are used instead of hitting the Opik API.
    This avoids expensive Opik fetches for recent runs whose JSON is still
    on disk.
    """

    def __init__(self, results_dir: Path | list[Path] | None = None) -> None:
        self._client = _make_opik_client()
        if results_dir is None:
            results_dir = [Path("results")]
        elif isinstance(results_dir, (list, tuple)):
            results_dir = [Path(d) for d in results_dir]
        else:
            results_dir = [Path(results_dir)]
        self._local = LocalSource([d for d in results_dir if d.is_dir()]) if any(d.is_dir() for d in results_dir) else None

    def list_experiments(
        self, *, name: str | None = None, env: str | None = None,
        tag: str | None = None, limit: int = 500,
    ) -> dict[str, SimpleNamespace]:
        # Try local cache first when searching by tag.
        if self._local is not None and tag is not None:
            local_exps = self._local.list_experiments(name=name, env=env, tag=tag, limit=limit)
            if local_exps:
                return local_exps
        # Fall back to Opik API.
        exps = _opik_list_experiments(self._client, name=name, limit=limit)
        if env is not None:
            exps = [e for e in exps if (e.metadata or {}).get("environment") == env]
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
        # If the experiment ID is a local file path, read from local cache.
        if self._local is not None and str(exp.id).endswith(".json"):
            return self._local.get_items(exp)
        return _opik_get_items(self._client, exp.id)


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

    ``opik`` (default) connects to Opik but checks local results first when
    a ``results_dir`` is available (local-cache-first).  ``local`` reads
    exclusively from ``results_dir`` (default: ``results/``).  ``results_dir``
    may be a single path or a list of paths to scan multiple directories.
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
    # Opik with local-cache-first: pass results_dir so OpikSource can check
    # local JSON files before hitting the API.
    if results_dir:
        if isinstance(results_dir, list):
            dirs = [Path(d) for d in results_dir]
        else:
            dirs = [Path(results_dir)]
    else:
        dirs = [Path("results")]
    return OpikSource(dirs)
