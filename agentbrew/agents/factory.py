"""Agent factory used by the core runner."""

from __future__ import annotations

import copy
from pathlib import Path

from agentbrew.agents.react import ReAct
from agentbrew.agents.react_summary import ReActSummary
from agentbrew.core.context import Context
from agentbrew.core.environment import Environment
from agentbrew.core.run_config import RunConfig
from agentbrew.llms import ModelManager
from agentbrew.mcp.manager import MCPManager


AGENT_CLASSES = {
    "react": ReAct,
    "react_summary": ReActSummary,
}


def default_agent_factory(config: RunConfig, environment: Environment, context: Context):
    """Build the configured agent with the domain's MCP server config."""
    agent_cls = AGENT_CLASSES.get(config.agent.type)
    if agent_cls is None:
        raise ValueError(f"Unknown agent type: {config.agent.type}")

    llm = ModelManager().build_model(config.llm.type, config.llm.config)
    llm.set_context(context)

    package_root = Path(__file__).resolve().parents[1]
    server_config = package_root / "environments" / environment.name / "servers.yaml"
    mcp_manager = MCPManager(str(server_config), context=context)
    agent_config = _environment_agent_config(config, environment, package_root)

    return agent_cls(
        mcp_manager=mcp_manager,
        llm=llm,
        config=agent_config,
    )


def _read_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _environment_agent_config(config: RunConfig, environment: Environment, package_root: Path) -> dict:
    """Apply environment-owned prompt defaults without overriding run YAML values."""
    agent_config = copy.deepcopy(config.agent.config)
    if config.mode in {"benchmark", "trajectory_sample"}:
        prompt_dir = (
            package_root
            / "environments"
            / environment.name
            / "benchmark"
            / "prompts"
        )
        instruction_path = prompt_dir / "instruction.j2"
        task_prompt_path = prompt_dir / "task_prompt.j2"
        if "instruction" not in agent_config and instruction_path.exists():
            agent_config["instruction"] = _read_prompt(instruction_path)
        if "task_prompt" not in agent_config and task_prompt_path.exists():
            agent_config["task_prompt"] = _read_prompt(task_prompt_path)
    elif config.mode == "task_sample":
        sampling_prompt_dir = (
            package_root
            / "environments"
            / environment.name
            / "task_sampling"
            / "prompts"
        )
        system_prompt_path = sampling_prompt_dir / "prompt.j2"
        instruction_path = sampling_prompt_dir / "instruction.j2"
        summary_prompt_path = sampling_prompt_dir / "summary_prompt.j2"
        if "system_prompt" not in agent_config and system_prompt_path.exists():
            agent_config["system_prompt"] = str(system_prompt_path)
        if "instruction" not in agent_config and instruction_path.exists():
            agent_config["instruction"] = instruction_path.read_text(encoding="utf-8")
        if "summary_prompt" not in agent_config and summary_prompt_path.exists():
            agent_config["summary_prompt"] = str(summary_prompt_path)
        agent_config.setdefault("instruction", "")
        agent_config.setdefault("task_prompt", "")
        agent_config.setdefault("max_iterations", 5)
        agent_config.setdefault("max_tool_output_length", 5000)
        agent_config.setdefault("summarize_tool_response", False)

    return agent_config
