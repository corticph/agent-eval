"""Utility script: print an access token for an environment."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

# Reuse the CLI's selection rule (explicit argument, else a fail-fast
# message) so this script and the harness cannot drift apart.
from agent_evals.__main__ import resolve_environment  # noqa: E402
from agent_evals.environment import supported_environments  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "Usage: python get_token.py <environment>\n"
            f"Supported environments: {', '.join(supported_environments())}",
            file=sys.stderr,
        )
        return 1
    name = sys.argv[1]
    try:
        print(resolve_environment(name).resolve_token())
    except (ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
