"""Subprocess entry point for AgentBrew PostgreSQL evaluator functions."""

from __future__ import annotations

import argparse
import asyncio
import json
import traceback

RESULT_PREFIX = "AGENTBREW_POSTGRES_VERIFIER_RESULT="


async def _run(verifier: str) -> dict[str, object]:
    from agentbrew.environments.postgres.benchmark.evaluator.registry import (  # pylint: disable=import-outside-toplevel
        COMPARISON_FUNCTIONS,
    )

    function = COMPARISON_FUNCTIONS.get(verifier)
    if function is None:
        raise KeyError(f"Unknown PostgreSQL verifier: {verifier}")
    passed, reason = await function(None, None, None)
    return {"passed": bool(passed), "reason": str(reason or ""), "error": ""}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verifier", required=True)
    args = parser.parse_args()
    try:
        payload = asyncio.run(_run(args.verifier))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        payload = {
            "passed": False,
            "reason": "",
            "error": f"{exc}\n{traceback.format_exc()}",
        }
    print(RESULT_PREFIX + json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
