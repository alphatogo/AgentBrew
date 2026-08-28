"""Worker and job partition helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .task import TaskSpec


@dataclass(frozen=True)
class TaskJob:
    """One executable task job."""

    task: TaskSpec
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.task.id or self.task.source_path or self.task.question[:80]


def partition_jobs(jobs: list[TaskJob], worker_count: int, *, partition_by: str = "category") -> list[list[TaskJob]]:
    """Partition jobs while keeping same partition key together when possible."""
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")

    groups: dict[str, list[TaskJob]] = defaultdict(list)
    for job in jobs:
        group_key = getattr(job.task, partition_by, None) or "default"
        groups[str(group_key)].append(job)

    buckets: list[list[TaskJob]] = [[] for _ in range(worker_count)]
    loads = [0 for _ in range(worker_count)]
    for _, group_jobs in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        idx = min(range(worker_count), key=lambda i: (loads[i], i))
        buckets[idx].extend(group_jobs)
        loads[idx] += len(group_jobs)
    return buckets

