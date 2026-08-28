"""GitHub account loading and validation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import yaml


DEFAULT_SERVER_CONFIG = Path(__file__).parent / "servers.yaml"
API_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "AgentBrew",
}


@dataclass(frozen=True)
class GitHubAccount:
    """One configured GitHub account."""

    account: str
    token: str
    source: str
    worker_id: int | None = None


@dataclass(frozen=True)
class GitHubAccountStatus:
    """Read-only validation result for one GitHub token."""

    configured_account: str
    authenticated_account: str
    source: str
    worker_id: int | None
    valid: bool
    account_matches: bool
    limit: int | None = None
    remaining: int | None = None
    used: int | None = None
    reset: int | None = None
    error: str = ""


def load_accounts(config_path: str | Path = DEFAULT_SERVER_CONFIG) -> list[GitHubAccount]:
    """Load benchmark and sampling accounts, deduplicated by token."""
    path = Path(config_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    private = data.get("_github", {}) or {}
    records: list[GitHubAccount] = []

    workers = (private.get("sampling", {}) or {}).get("workers", {}) or {}
    for raw_worker_id, worker in sorted(workers.items(), key=lambda item: int(item[0])):
        worker = worker or {}
        if not worker.get("token"):
            continue
        records.append(
            GitHubAccount(
                account=str(worker.get("account") or ""),
                token=str(worker["token"]).strip(),
                source=f"sampling.workers.{raw_worker_id}",
                worker_id=int(raw_worker_id),
            )
        )

    benchmark = private.get("benchmark", {}) or {}
    if benchmark.get("token"):
        records.append(
            GitHubAccount(
                account=str(benchmark.get("account") or ""),
                token=str(benchmark["token"]).strip(),
                source="benchmark",
            )
        )

    unique: list[GitHubAccount] = []
    seen_tokens: set[str] = set()
    for record in records:
        if record.token in seen_tokens:
            continue
        seen_tokens.add(record.token)
        unique.append(record)
    return unique


def validate_account(
    account: GitHubAccount,
    *,
    timeout: float = 15,
) -> GitHubAccountStatus:
    """Validate token identity and core API rate limit without modifying GitHub."""
    headers = {**API_HEADERS, "Authorization": f"Bearer {account.token}"}
    try:
        user_response = requests.get(
            "https://api.github.com/user",
            headers=headers,
            timeout=timeout,
        )
        if user_response.status_code != 200:
            return GitHubAccountStatus(
                configured_account=account.account,
                authenticated_account="",
                source=account.source,
                worker_id=account.worker_id,
                valid=False,
                account_matches=False,
                error=f"/user returned HTTP {user_response.status_code}",
            )

        authenticated_account = str(user_response.json().get("login") or "")
        account_matches = authenticated_account.casefold() == account.account.casefold()

        rate_response = requests.get(
            "https://api.github.com/rate_limit",
            headers=headers,
            timeout=timeout,
        )
        if rate_response.status_code != 200:
            return GitHubAccountStatus(
                configured_account=account.account,
                authenticated_account=authenticated_account,
                source=account.source,
                worker_id=account.worker_id,
                valid=False,
                account_matches=account_matches,
                error=f"/rate_limit returned HTTP {rate_response.status_code}",
            )

        core: dict[str, Any] = (
            (rate_response.json().get("resources", {}) or {}).get("core", {}) or {}
        )
        remaining = core.get("remaining")
        limit = core.get("limit")
        valid = (
            account_matches
            and limit == 5000
            and isinstance(remaining, int)
            and remaining > 0
        )
        error = ""
        if not account_matches:
            error = "configured account does not match token owner"
        elif limit != 5000:
            error = f"core API limit is {limit!r}; sampling requires 5000"
        elif not isinstance(remaining, int) or remaining <= 0:
            error = "core API rate limit is exhausted or unavailable"
        return GitHubAccountStatus(
            configured_account=account.account,
            authenticated_account=authenticated_account,
            source=account.source,
            worker_id=account.worker_id,
            valid=valid,
            account_matches=account_matches,
            limit=limit,
            remaining=remaining,
            used=core.get("used"),
            reset=core.get("reset"),
            error=error,
        )
    except requests.RequestException as exc:
        return GitHubAccountStatus(
            configured_account=account.account,
            authenticated_account="",
            source=account.source,
            worker_id=account.worker_id,
            valid=False,
            account_matches=False,
            error=str(exc),
        )


def validate_accounts(
    accounts: list[GitHubAccount],
    *,
    workers: int = 8,
    timeout: float = 15,
) -> list[GitHubAccountStatus]:
    """Validate configured tokens concurrently while preserving config order."""
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        return list(
            executor.map(
                lambda account: validate_account(account, timeout=timeout),
                accounts,
            )
        )
