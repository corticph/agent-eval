"""All Suite loading, end-to-end: YAML file → the input model.

The one place a suite file becomes objects — file I/O, folder-defaults
merging, the variables machinery, case/step construction, and parse-time
validation — plus the input model those steps produce (:class:`Step`,
:class:`EvaluationCase`, :class:`EvaluationSuite`, :class:`SuiteOptions`),
since the loader's output classes belong with their construction.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from .client import MAX_HTTP_TIMEOUT_SECONDS
from .expectations import Expectation, parse_expectations
from .schemas.agent import Agent
from .schemas.message import MessagePayload

_LOGGER = logging.getLogger(__name__)

PLACEHOLDER_PATTERN = re.compile(r"{{\s*([a-zA-Z0-9_:\-\.]+)\s*}}")


def validate_timeout_seconds(value: float, *, context: str) -> float:
    """Require *value* to exceed the HTTP request timeout.

    A wall-clock timeout at or below the HTTP timeout could never pre-empt a
    single in-flight request, so it would only ever abandon worker threads.
    """
    if value <= MAX_HTTP_TIMEOUT_SECONDS:
        raise ValueError(
            f"{context}: timeout_seconds ({value:g}s) must be greater than "
            f"the HTTP request timeout ({MAX_HTTP_TIMEOUT_SECONDS:g}s)"
        )
    return value


@dataclass(slots=True)
class Step:
    """One message-and-expectations unit within a case.

    A single-message case is one Step whose ``name`` is ``None``; a multi-step
    case names each Step so its Step Trail stays readable.
    """

    message: MessagePayload
    expectations: list[Expectation] = field(default_factory=list)
    delay_before_seconds: float | None = None
    name: str | None = None

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        base_variables: dict[str, str],
        message_defaults: dict[str, Any] | None = None,
    ) -> "Step":
        # A step in a ``steps:`` list must name itself, so the Step Trail is legible.
        if "name" not in data:
            raise ValueError("each step in a multi-step case must define a 'name'")
        step_variables = _extend_variables(base_variables, data)
        message, expectations = _resolve_message_and_expectations(
            data, variables=step_variables, message_defaults=message_defaults
        )
        delay_before_seconds = (
            float(data["delay_before_seconds"])
            if "delay_before_seconds" in data
            else None
        )
        return cls(
            message=message,
            expectations=expectations,
            delay_before_seconds=delay_before_seconds,
            name=data["name"],
        )


@dataclass(slots=True)
class EvaluationCase:
    """An evaluation: an envelope plus an ordered list of one or more Steps."""

    name: str
    agent: Agent
    steps: list[Step]
    agent_id_override: str | None = None
    use_connector_name: str | None = None
    description: str | None = None
    timeout_seconds: float | None = None

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        variables: dict[str, str] | None = None,
        agent_defaults: dict[str, Any] | None = None,
        message_defaults: dict[str, Any] | None = None,
    ) -> "EvaluationCase":
        base_variables = _extend_variables(variables or {}, data)
        # Shape before envelope: a case from a retired authoring format has
        # neither `message` nor `steps`, and that diagnosis is worth more to
        # the author than a complaint about whichever envelope key is read
        # first.
        name = data.get("name")
        steps = _normalize_steps(
            data,
            base_variables=base_variables,
            message_defaults=message_defaults,
            case_name=name or "<unnamed>",
        )
        if not name:
            raise ValueError("every eval must declare a 'name'")
        agent = _merge_agent(
            agent_defaults or {},
            data.get("agent", {}),
            variables=base_variables,
            case_name=name,
        )
        agent_id_override = _extract_agent_id_override(data, variables=base_variables)
        if "use_expert_name" in data:  # v1-fail-fast-guard
            raise ValueError(
                f"case {name!r} uses the retired 'use_expert_name' key; "  # v1-fail-fast-guard
                "target a connector with 'use_connector_name'"
            )
        use_connector_name = data.get("use_connector_name") or None
        description = _clean_description(data.get("description"))
        timeout_seconds = (
            validate_timeout_seconds(
                float(data["timeout_seconds"]), context=f"eval {name!r}"
            )
            if "timeout_seconds" in data
            else None
        )
        return cls(
            name=name,
            agent=agent,
            steps=steps,
            agent_id_override=agent_id_override,
            use_connector_name=use_connector_name,
            description=description,
            timeout_seconds=timeout_seconds,
        )


@dataclass(slots=True)
class SuiteOptions:
    """Runtime options for the evaluation suite."""

    stop_on_failure: bool = False
    output_path: Path | None = None
    concurrency: int = 5
    timeout_seconds: float | None = None


@dataclass(slots=True)
class EvaluationSuite:
    """In-memory representation of a suite file."""

    name: str
    cases: Sequence[EvaluationCase]
    options: SuiteOptions = field(default_factory=SuiteOptions)
    variables: dict[str, str] = field(default_factory=dict)


def _extend_variables(base: dict[str, str], data: dict[str, Any]) -> dict[str, str]:
    """Layer a Case's or Step's own ``variables:`` block over the inherited ones.

    Shared so a Step's variables and a Case's variables can't resolve by
    different rules.
    """
    variables = dict(base)
    if "variables" in data:
        variables.update(build_variables(data.get("variables")))
    return variables


def _resolve_message_and_expectations(
    data: dict[str, Any],
    *,
    variables: dict[str, str],
    message_defaults: dict[str, Any] | None,
) -> tuple[MessagePayload, list[Expectation]]:
    """Build the message/expectations pair both authoring shapes share.

    Both a Step and a synthesized single-message Case resolve here, so the two
    shapes can never drift apart on how a message and its expectations are built.
    Expectations parse straight into the canonical ``list[Expectation]`` the
    registry produces — a mistyped key fails loudly here, at load.
    """
    message = MessagePayload.from_dict(
        _merge_and_resolve(
            message_defaults or {}, data.get("message", {}), variables=variables
        )
    )
    expectations = parse_expectations(
        resolve_variables(data.get("expectations"), variables)
    )
    return message, expectations


def _normalize_steps(
    data: dict[str, Any],
    *,
    base_variables: dict[str, str],
    message_defaults: dict[str, Any] | None,
    case_name: str,
) -> list[Step]:
    """Collapse both authoring shapes into an ordered ``list[Step]`` (length ≥ 1).

    The shape is inferred from the keys present, so an author needs no ``type:``
    discriminator. A Case that declares neither ``message`` nor ``steps`` raises
    here, so a malformed Case is caught at load time rather than mid-run.
    """
    if "steps" in data:
        raw_steps = data.get("steps") or []
        if not raw_steps:
            raise ValueError(f"eval {case_name!r} must define at least one step")
        return [
            Step.from_dict(
                step_data,
                base_variables=base_variables,
                message_defaults=message_defaults,
            )
            for step_data in raw_steps
        ]
    if "message" in data or "expectations" in data:
        message, expectations = _resolve_message_and_expectations(
            data, variables=base_variables, message_defaults=message_defaults
        )
        return [Step(message=message, expectations=expectations)]
    raise ValueError(
        f"eval {case_name!r} must declare either a 'message' or a 'steps' list"
    )


def load_variable_value(raw_value: str) -> str:
    """Resolve a variable source such as "env:NAME"."""
    if raw_value.startswith("env:"):
        env_var = raw_value.split(":", 1)[1]
        value = os.environ.get(env_var)
        if value is None:
            raise RuntimeError(
                f"Environment variable {env_var} is required but missing"
            )
        return value
    return raw_value


def resolve_variables(definition: Any, variables: dict[str, str]) -> Any:
    """Replace placeholder tokens within an arbitrary data structure."""
    if isinstance(definition, str):

        def _replacement(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in variables:
                raise RuntimeError(f"No variable defined for placeholder {key!r}")
            return variables[key]

        return PLACEHOLDER_PATTERN.sub(_replacement, definition)
    if isinstance(definition, dict):
        return {
            key: resolve_variables(value, variables)
            for key, value in definition.items()
        }
    if isinstance(definition, list):
        return [resolve_variables(value, variables) for value in definition]
    return definition


def _builtin_variables() -> dict[str, str]:
    """Date helpers available in every eval without explicit declaration."""
    today = datetime.datetime.now(datetime.timezone.utc).date()
    yesterday = today - datetime.timedelta(days=1)
    return {
        "today": today.isoformat(),
        "yesterday": yesterday.isoformat(),
    }


def build_variables(raw_variables: dict[str, Any] | None) -> dict[str, str]:
    """Construct the variable mapping, resolving env sources."""
    resolved: dict[str, str] = _builtin_variables()
    if not raw_variables:
        return resolved
    for key, raw_value in raw_variables.items():
        if isinstance(raw_value, str):
            resolved[key] = load_variable_value(raw_value)
        elif isinstance(raw_value, dict):
            resolved[key] = _resolve_variable_spec(raw_value)
        else:
            raise TypeError(f"Unsupported variable type for {key!r}: {type(raw_value)}")
    return resolved


def _resolve_variable_spec(spec: dict[str, Any]) -> str:
    """Resolve structured variable specs such as random number generators."""
    spec_type = spec.get("type")
    if spec_type == "random_int":
        if "min" not in spec or "max" not in spec:
            raise ValueError("random_int spec requires 'min' and 'max'")
        min_value = int(spec["min"])
        max_value = int(spec["max"])
        if min_value > max_value:
            raise ValueError(
                f"random_int min ({min_value}) cannot exceed max ({max_value})"
            )
        return str(random.randint(min_value, max_value))
    raise TypeError(f"Unsupported variable spec type {spec_type!r}")


def _extract_agent_id_override(
    data: dict[str, Any], *, variables: dict[str, str]
) -> str | None:
    for key in ("agent_id", "agentId"):
        if key not in data:
            continue
        raw_value = data[key]
        if raw_value is None:
            return None
        if not isinstance(raw_value, str):
            raise TypeError(f"Expected string for {key!r}, got {type(raw_value)}")
        resolved = resolve_variables(raw_value, variables)
        if not isinstance(resolved, str):
            raise TypeError(f"Resolved agent id for {key!r} must be a string")
        resolved_id = resolved.strip()
        return resolved_id or None
    return None


def _clean_description(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _merge_and_resolve(
    *payloads: dict[str, Any], variables: dict[str, str]
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for payload in payloads:
        merged = _deep_merge(merged, payload)
    return resolve_variables(merged, variables)


def _merge_agent(
    *payloads: dict[str, Any], variables: dict[str, str], case_name: str
) -> Agent:
    """Deep-merge agent dicts, resolve variables, validate, and build an :class:`Agent`.

    Wraps :func:`_merge_and_resolve` (still dict-based, since message payloads
    reuse it) and returns the typed agent object used by evaluation cases.
    """
    merged = _merge_and_resolve(*payloads, variables=variables)
    _validate_agent_payload(merged, case_name=case_name)
    return Agent.from_dict(merged)


def _validate_agent_payload(agent_payload: dict[str, Any], *, case_name: str) -> None:
    """Reject the retired v1 agent shape and disallowed whitespace in names."""

    for legacy_key in ("experts", "mcpServers"):  # v1-fail-fast-guard
        if legacy_key in agent_payload:
            raise ValueError(
                f"Agent for case {case_name!r} uses the retired {legacy_key!r} key; "  # v1-fail-fast-guard
                "author v2 'connectors' instead"
            )

    agent_name = agent_payload.get("name")
    if isinstance(agent_name, str) and _contains_whitespace(agent_name):
        raise ValueError(
            f"Agent name for eval {case_name!r} cannot contain whitespace: {agent_name!r}"
        )

    connectors = agent_payload.get("connectors")
    if not isinstance(connectors, list):
        return

    for index, connector in enumerate(connectors):
        if not isinstance(connector, dict):
            continue
        connector_name = connector.get("name")
        if isinstance(connector_name, str) and _contains_whitespace(connector_name):
            raise ValueError(
                f"Connector name at index {index} for case {case_name!r} cannot contain whitespace: {connector_name!r}"
            )
        # An inline agent connector embeds a full create payload; hold it to
        # the same rules as the agent that carries it.
        if connector.get("type") == "agent" and not connector.get("agentId"):
            _validate_agent_payload(
                {k: v for k, v in connector.items() if k != "type"}, case_name=case_name
            )


def _contains_whitespace(value: str) -> bool:
    return any(char.isspace() for char in value)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_data_files(config: Any, base_dir: Path) -> None:
    """Walk the raw config and replace ``data_file`` references with loaded JSON content."""
    if isinstance(config, dict):
        # ``data_file`` marks a data part whose payload is loaded from disk. It
        # only ever appears on data parts, so its presence is enough — we no
        # longer require the v1 ``kind: data`` discriminator alongside it.
        if "data_file" in config:
            file_path = base_dir / config.pop("data_file")
            with file_path.open("r", encoding="utf-8") as fh:
                config["data"] = json.load(fh)
        for value in config.values():
            _resolve_data_files(value, base_dir)
    elif isinstance(config, list):
        for item in config:
            _resolve_data_files(item, base_dir)


def _resolve_schema_ref_entry(entry: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Resolve a single ``schema_ref`` connector entry into a full v2 connector.

    Reads the referenced schema file. Supports two formats:

    * **Flat format** (preferred): ``{name, description, schema, type}`` — the
      ``schema`` key holds the JSON schema directly.
    * **Legacy tools format**: ``{tools: [{name, description, inputSchema}]}`` —
      the schema is extracted from ``tools[0].inputSchema``.

    In both cases, ``name``, ``description``, ``type`` and ``transition`` are
    lifted from the file unless the entry overrides them. ``transition`` is only
    read from the top level of the schema document, never from ``tools[0]``.
    """
    schema_ref = entry.pop("schema_ref")
    schema_path = base_dir / schema_ref
    if not schema_path.exists():
        raise FileNotFoundError(f"schema_ref file not found: {schema_path}")
    with schema_path.open("r", encoding="utf-8") as fh:
        schema_doc = json.load(fh)

    if "schema" in schema_doc:
        file_name = schema_doc.get("name")
        file_description = schema_doc.get("description")
        resolved_schema = schema_doc.get("schema")
        if resolved_schema is None:
            raise ValueError(
                f"schema_ref file {schema_path}: 'schema' key is null or missing"
            )
    else:
        tools = schema_doc.get("tools")
        if not isinstance(tools, list) or len(tools) == 0:
            raise ValueError(f"schema_ref file {schema_path} has 0 tools (malformed)")
        if len(tools) > 1:
            raise ValueError(f"schema_ref file {schema_path} has more than 1 tool")
        tool = tools[0]
        if not isinstance(tool, dict):
            raise ValueError(
                f"schema_ref file {schema_path}: tools[0] is not an object (malformed)"
            )
        file_name = tool.get("name")
        file_description = tool.get("description")
        resolved_schema = tool.get("inputSchema")
        if resolved_schema is None:
            raise ValueError(
                f"schema_ref file {schema_path}: tools[0] missing 'inputSchema'"
            )

    entry_name = entry.get("name")
    if entry_name and file_name and entry_name != file_name:
        _LOGGER.warning(
            "Name mismatch in schema_ref: entry name %r != file name %r (using entry name)",
            entry_name,
            file_name,
        )
    if not entry_name and not file_name:
        raise ValueError(
            f"No name available: entry has no name and {schema_path} has no name"
        )
    connector: dict[str, Any] = {**entry}
    connector["name"] = entry_name or file_name
    if "description" not in connector:
        connector["description"] = file_description
    if "type" not in connector and "type" in schema_doc:
        connector["type"] = schema_doc["type"]
    # Whether a tool ends the turn belongs to the tool, not to one suite.
    if "transition" not in connector and "transition" in schema_doc:
        connector["transition"] = schema_doc["transition"]
    connector["schema"] = resolved_schema
    return connector


def _resolve_connector_entry(entry: Any, base_dir: Path) -> dict[str, Any]:
    """Resolve a single connector entry, passing through non-``schema_ref`` entries as-is."""
    if isinstance(entry, dict) and "schema_ref" in entry:
        return _resolve_schema_ref_entry(entry, base_dir)
    return entry


def resolve_connectors(config: Any, base_dir: Path) -> None:
    """Walk the raw config and resolve ``connectors_file`` / ``schema_ref`` references.

    For each dict carrying a ``connectors_file`` key, the referenced JSON array
    is loaded and its ``schema_ref`` entries resolved into full v2
    ``SchemaConnector`` objects; the result replaces ``connectors_file`` as an
    inline ``connectors`` array. Inline ``connectors`` arrays (without a
    ``connectors_file``) have their ``schema_ref`` entries resolved in place.
    Entries without ``schema_ref`` (e.g. ``type: registry``, ``type: mcp``)
    pass through untouched.
    """
    if isinstance(config, dict):
        if "connectors_file" in config:
            file_path = base_dir / config.pop("connectors_file")
            if not file_path.exists():
                raise FileNotFoundError(f"connectors_file not found: {file_path}")
            with file_path.open("r", encoding="utf-8") as fh:
                entries = json.load(fh)
            if not isinstance(entries, list):
                raise ValueError(
                    f"connectors_file {file_path} must contain a JSON array"
                )
            config["connectors"] = [
                _resolve_connector_entry(e, base_dir) for e in entries
            ]
        elif "connectors" in config and isinstance(config["connectors"], list):
            config["connectors"] = [
                _resolve_connector_entry(e, base_dir) for e in config["connectors"]
            ]
        # Walk sibling values, but never the resolved ``connectors`` list —
        # its entries carry inline ``schema`` JSON that must not be re-walked
        # (wasteful O(schema) traversal, and a schema property named
        # ``connectors_file`` would raise ``TypeError``).
        for key, value in config.items():
            if key == "connectors":
                continue
            resolve_connectors(value, base_dir)
    elif isinstance(config, list):
        for item in config:
            resolve_connectors(item, base_dir)


def _find_project_root(start: Path) -> Path | None:
    """Return the nearest ancestor (inclusive) marking a project boundary."""
    for directory in [start, *start.parents]:
        if (directory / ".git").exists() or (directory / "pyproject.toml").exists():
            return directory
    return None


def _load_folder_defaults(suite_path: Path) -> dict[str, Any]:
    """Load and merge ``_defaults.yaml`` files from ancestor directories.

    Walks from the project root down to the suite's parent directory,
    deep-merging each ``_defaults.yaml`` found along the way so that closer
    (more-specific) files override farther (more-general) ones. The walk is
    bounded by the enclosing project root (nearest ancestor with ``.git`` or
    ``pyproject.toml``) so it never reads ``_defaults.yaml`` from directories
    outside the repo. If no project root is found, only the suite's own
    directory is considered.
    """
    suite_dir = suite_path.parent
    project_root = _find_project_root(suite_dir)
    # Collect ancestor directories from project root → suite parent (inclusive)
    if project_root is None:
        ancestors = [suite_dir]
    else:
        ancestors = [
            d
            for d in [suite_dir, *suite_dir.parents]
            if d == project_root or project_root in d.parents
        ]
        ancestors.reverse()  # general (project root) → specific (suite dir)
    merged: dict[str, Any] = {}
    for directory in ancestors:
        defaults_path = directory / "_defaults.yaml"
        if not defaults_path.exists():
            continue
        with defaults_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise ValueError(
                f"{defaults_path} must be a YAML mapping, got {type(raw).__name__}"
            )
        _resolve_data_files(raw, defaults_path.parent)
        resolve_connectors(raw, defaults_path.parent)
        merged = _deep_merge(merged, raw)
    return merged


def load_suite(path: str | Path) -> EvaluationSuite:
    """Load an evaluation suite from a YAML file.

    If a ``_defaults.yaml`` file exists in the same directory, its ``globals``
    block is used as a base layer that the suite's own ``globals`` overrides
    (deep-merge, suite wins on every key).
    """
    suite_path = Path(path)
    with suite_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
    _resolve_data_files(raw_config, suite_path.parent)
    resolve_connectors(raw_config, suite_path.parent)
    name = raw_config.get("name", suite_path.stem)
    folder_defaults = _load_folder_defaults(suite_path)
    folder_globals: dict[str, Any] = folder_defaults.get("globals") or {}
    suite_globals: dict[str, Any] = raw_config.get("globals") or {}
    globals_cfg: dict[str, Any] = _deep_merge(folder_globals, suite_globals)
    variables = build_variables(globals_cfg.get("variables"))
    agent_defaults = globals_cfg.get("agent", {})
    message_defaults = globals_cfg.get("message", {})
    options = SuiteOptions(
        stop_on_failure=bool(globals_cfg.get("stop_on_failure", False)),
        output_path=Path(globals_cfg["output_path"])
        if globals_cfg.get("output_path")
        else None,
        concurrency=int(globals_cfg["concurrency"])
        if globals_cfg.get("concurrency")
        else SuiteOptions().concurrency,
        timeout_seconds=(
            validate_timeout_seconds(
                float(globals_cfg["timeout_seconds"]), context=f"suite {name!r} globals"
            )
            if globals_cfg.get("timeout_seconds")
            else None
        ),
    )
    raw_cases: Iterable[dict[str, Any]] = raw_config.get("evals", [])
    if not raw_cases:
        raise ValueError(f"No evals defined in suite {suite_path}")
    cases = [
        EvaluationCase.from_dict(
            item,
            variables=variables,
            agent_defaults=agent_defaults,
            message_defaults=message_defaults,
        )
        for item in raw_cases
    ]
    return EvaluationSuite(name=name, cases=cases, options=options, variables=variables)
