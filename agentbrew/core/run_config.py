"""Run configuration models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from .errors import ConfigError

RunMode = Literal["task_sample", "trajectory_sample", "benchmark"]
TaskSourceType = Literal["file", "directory", "manifest", "inline", "none"]


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    type: str = "local_llm"
    config: dict[str, Any] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    """Agent configuration."""

    type: str = "react_summary"
    config: dict[str, Any] = Field(default_factory=dict)


class TaskSourceConfig(BaseModel):
    """Where tasks come from for trajectory and benchmark modes."""

    type: TaskSourceType = "none"
    path: str | None = None
    tasks: list[dict[str, Any]] = Field(default_factory=list)
    glob: str = "*.json"


class TaskSamplingConfig(BaseModel):
    """Task sampling controls."""

    engine: Literal["llm", "agent"] = "llm"
    prompt: str | None = None
    few_shot_path: str | None = None
    output_task_dir: str | None = None
    output_jsonl: str | None = None
    seeds: dict[str, Any] = Field(default_factory=dict)
    num_tasks: int | None = None


class EvaluationConfig(BaseModel):
    """Evaluation controls."""

    enabled: bool = True
    fail_fast: bool = False


class ExecutionConfig(BaseModel):
    """Execution controls."""

    workers: int = 1
    task_timeout_seconds: int | None = None
    resume: bool = True
    partition_by: str = "category"
    overwrite: bool = True


class OutputConfig(BaseModel):
    """Artifact output controls."""

    root: str = "outputs/run"
    traces: bool = True
    include_prompts: bool = False
    reports: bool = True
    task_results: bool = True


class RunConfig(BaseModel):
    """Top-level AgentBrew run config."""

    mode: RunMode
    domain: str
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    task_source: TaskSourceConfig = Field(default_factory=TaskSourceConfig)
    task_sampling: TaskSamplingConfig = Field(default_factory=TaskSamplingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    env_file: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> "RunConfig":
        """Load a YAML run config."""
        config_path = Path(path)
        if not config_path.exists():
            raise ConfigError(f"Run config not found: {config_path}")
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    def validate_for_mode(self) -> None:
        """Validate mode-specific requirements."""
        if self.mode in {"trajectory_sample", "benchmark"} and self.task_source.type == "none":
            raise ConfigError(f"{self.mode} requires a task_source")
        if self.mode == "task_sample" and not self.task_sampling.output_task_dir:
            raise ConfigError("task_sample requires task_sampling.output_task_dir")
