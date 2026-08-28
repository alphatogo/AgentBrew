"""GitHub-specific trajectory rendering and retrospective-task prompt."""

from __future__ import annotations

import json
from typing import Any


__all__ = [
    "DEFAULT_CREDIT_METHOD",
    "build_credit_prompt_prefix",
    "build_task_inference_prompt",
    "format_trajectory",
    "is_eligible_task",
    "is_error",
    "normalize_task_inference_output",
    "summarize_changes",
]


DEFAULT_CREDIT_METHOD = "adjacent"


def is_error(step: dict[str, Any]) -> bool:
    return bool(step.get("isError", False))


def is_eligible_task(task: dict[str, Any]) -> bool:
    return bool(task["trajectory"])


def _truncate_middle(value: Any, limit: int = 1000) -> str:
    text = str(value).replace("\n", " ")
    if len(text) <= limit:
        return text
    keep = limit // 2
    return text[:keep] + "\n...[MIDDLE TRUNCATED]...\n" + text[-keep:]


def _truncate_arguments(value: Any, limit: int = 200) -> Any:
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "...[TRUNCATED]"
    if isinstance(value, dict):
        return {key: _truncate_arguments(item, limit) for key, item in value.items()}
    if isinstance(value, list):
        return [_truncate_arguments(item, limit) for item in value]
    return value


def format_trajectory(trajectory: list[dict[str, Any]]) -> str:
    lines = []
    for index, step in enumerate(trajectory):
        lines.extend(
            [
                f"[Step {index}] Tool: {step.get('tool_name')}",
                f"Args: {json.dumps(_truncate_arguments(step.get('arguments', {})), ensure_ascii=False)}",
                f"Feedback (Error: {is_error(step)}): "
                f"{_truncate_middle(step.get('content', ''), 1000)}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def summarize_changes(trajectory: list[dict[str, Any]]) -> str:
    lines = []
    for step in trajectory:
        if is_error(step):
            continue
        tool = str(step.get("tool_name") or "")
        args = step.get("arguments", {}) or {}
        lowered = tool.lower()
        if any(
            marker in lowered
            for marker in (
                "create",
                "update",
                "delete",
                "merge",
                "comment",
                "push",
                "fork",
            )
        ):
            lines.append(
                f"- {tool}: {json.dumps(_truncate_arguments(args), ensure_ascii=False)}"
            )
    return "\n".join(lines) if lines else "- No confirmed successful GitHub changes detected."


def build_task_inference_prompt(
    original_task: str,
    trajectory_str: str,
    change_summary: str,
) -> str:
    return f"""You are an expert prompt engineer. Your goal is to minimally revise the ORIGINAL TASK so it matches the actions that the AI agent actually completed in the trajectory below.

Keep the original request's tone, persona, format, and high-level intent. Remove or weaken unsupported requirements instead of reconstructing a new request from scratch.

Here are two examples of well-written requests for reference on TONE and FORMAT only:
---
EXAMPLE 1:
"I am a first-year PhD student in Computer Science. My supervisor has assigned me a project to build a code large language model fine-tuning framework. He wants the project to be called 'BigCodeLLM-FT-Proj'. To finish this project, I also want to invite my friend to join me as a collaborator, so I need three branches: main, dev-me, and dev-friend. I need to create a README.md file in the main branch with the content "# BigCodeLLM-FT-Proj\\n\\nA comprehensive framework for fine-tuning large language models.". I also need to create a .gitignore file in the main branch with the exact content: "# Python cache and virtual environments\\n__pycache__/\\n*.pyc\\n*.py.class\\nvenv/\\n*.env". In my dev branch, I want to copy the entire content of example_instructions.py from meta-llama's official codellama repository and give it the same name. I also want in my friend's branch to help me copy the entire content of generation.py from meta-llama's official codellama repository and give it the same name. Finally, create a pull request to merge my branch into main with the title "Add example instructions" and description "This PR adds the example instructions for the fine-tuning framework."

EXAMPLE 2:
"Hi! I'm a student working on learning GitHub automation and I really need your help. Could you please help me create a new project repository named auto-comment-bot-x? I need to initialize it with just the main branch and include an initial README.md file with the content "# Automated Comment Bot\\n\\nA repository to test GitHub automation for adding comments to issues." I'm struggling with GitHub automation workflows and would really appreciate your help developing a script that automatically adds a comment 'Thank you for your contribution!' to any new issue created. After we set up the automation script, I need to test it by creating three sample issues with different titles ("Bug report", "Feature request", "Documentation update"). I'm really grateful for any assistance you can provide!"
---

ORIGINAL TASK:
{original_task}

OBSERVED CHANGES:
{change_summary}

TRAJECTORY:
{trajectory_str}

Follow these steps:

=== STEP 1: TRAJECTORY ANALYSIS ===
Analyze the trajectory by separating USER-SPECIFIED INPUTS from API-RETURNED OUTPUTS.

USER INPUTS (these are hard constraints that may remain in the revised request when supported):
- Search queries: copy exact query strings (these define WHAT to search)
- Labels/filters specified by the agent: e.g., label="bug", state="OPEN"
- Repository/branch/file names that the agent CREATED (not found): e.g., created repo "psf-requests-bug-stats"
- File structures the agent DESIGNED: e.g., CSV columns
- PR titles, commit messages the agent WROTE
- Source paths for file copies the agent CHOSE

API OUTPUTS (these are results the user would NOT know in advance — EXCLUDE from request):
- Which repositories a search query returned (even if the agent later called tools on them)
- The specific repos that list_issues was called on (these were DERIVED from search results, not user-specified)
- How many issues were found
- Issue titles, numbers, content
- Repository metadata (stars, forks, descriptions)

KEY DISTINCTION: If the agent searched for repos and THEN called list_issues on the results, the specific repo names come from the API, NOT the user. The user only specified the search query. Write "For each matching repository" in the request, NOT the specific repo names.

Use this analysis internally. Do not include it in the final output.

=== STEP 2: REVISE THE REQUEST ===
Minimally edit the ORIGINAL TASK based only on actions supported by the trajectory.

Rules:
- Preserve the original wording, persona, motivation, and paragraph structure wherever possible
- Write as future instructions ("Can you...", "Please..."), NOT past narrative
- Remove unsupported subtasks and claims
- Do not add actions not found in the trajectory
- Do not include API-returned result data
- Keep exact hard constraints only when they are supported by successful actions
- When describing search scope, use the EXACT query semantics
- Prefer deleting unsupported details over replacing them with speculative details
- Output only the revised request, without analysis, labels, or explanation

REVISED TASK:
"""


def normalize_task_inference_output(text: str) -> str:
    text = text.strip()
    if "REVISED TASK:" in text:
        text = text.split("REVISED TASK:", 1)[1].strip()
    elif "REQUEST:" in text:
        text = text.split("REQUEST:", 1)[1].strip()
    for prefix in ("User Request:", "Request:", "Instruction:", "Simulated User Request:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    return text


def build_credit_prompt_prefix(trajectory_str: str) -> str:
    if trajectory_str:
        return f"""An AI agent is executing GitHub tool calls. Based on the actions observed so far, what was the user's original request?

Actions observed:
{trajectory_str}

The user's original request was:
"""
    return """An AI agent is about to execute GitHub tool calls. Without seeing any actions, what was the user's original request?

The user's original request was:
"""
