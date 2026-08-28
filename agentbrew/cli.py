"""Command line entry point for AgentBrew runs."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from agentbrew.callbacks.vprint import VerbosePrintCallback
from agentbrew.core.runner import Runner
from agentbrew.core.run_config import RunConfig


def _import_builtin_environments() -> None:
    """Import built-in environment packages so they register themselves."""
    import agentbrew.environments.github  # noqa: F401
    import agentbrew.environments.notion  # noqa: F401
    import agentbrew.environments.postgres  # noqa: F401


async def _run(args: argparse.Namespace) -> int:
    _import_builtin_environments()
    config = RunConfig.from_file(args.config)
    if args.env_file:
        config.env_file = args.env_file
    if args.output_root:
        config.output.root = args.output_root
    callbacks = [] if args.quiet else [VerbosePrintCallback()]
    runner = Runner(config, callbacks=callbacks)
    result = await runner.run()
    payload = {
        "records": [record.model_dump(mode="json") for record in result.records],
        "output_root": str(Path(config.output.root).resolve()),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 1 if any(record.status == "failed" for record in result.records) else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AgentBrew configs.")
    parser.add_argument("config", help="Path to a run YAML config.")
    parser.add_argument("--env-file", default=None, help="Override env file path.")
    parser.add_argument("--output-root", default=None, help="Override output root.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-iteration verbose output.")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
