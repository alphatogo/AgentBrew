#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example usage:

    python -m agentbrew.environments.postgres.task_sampling.generation analyze

    python -m agentbrew.environments.postgres.task_sampling.generation count
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import textwrap
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from openai import AsyncOpenAI, OpenAI


POSTGRES_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
META_ROOT = POSTGRES_ROOT / "assets/metadata"
FEW_SHOT_ROOT = POSTGRES_ROOT / "task_sampling/few_shot"

DEFAULT_MODEL_NAME = "./Qwen3-32B"
DEFAULT_BASE_URL = "http://localhost:2024/v1"
DEFAULT_API_KEY = ""

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_COMPLETION_TOKENS = 2200
DEFAULT_MAX_META_CHARS = 14000
DEFAULT_EXPORT_ROOT = REPO_ROOT / "outputs/postgres_task_sample"
DEFAULT_TRAJECTORY_SQL_ROOT = REPO_ROOT / "outputs/postgres_trajectory_sample/tasks"
PROMPT_VARIANTS = ("prompt_v1", "prompt_v2")
ALLOWED_TOOLS = {
    "list_schemas",
    "list_objects",
    "get_object_details",
    "explain_query",
    "analyze_workload_indexes",
    "analyze_query_indexes",
    "analyze_db_health",
    "get_top_queries",
    "execute_sql",
}
MAX_GENERATION_ATTEMPTS = 2
DEFAULT_MAX_CONCURRENCY = 200

TOOL_SUMMARY = textwrap.dedent(
    """
    Available postgres-pro tools:
    - list_schemas
    - list_objects
    - get_object_details
    - explain_query
    - analyze_workload_indexes
    - analyze_query_indexes
    - analyze_db_health
    - get_top_queries
    - execute_sql
    """
).strip()

GLOBAL_GENERATION_RULES = textwrap.dedent(
    """
    You are generating benchmark-style MCP tasks for PostgreSQL.

    Hard requirements:
    1. The task must be executable using ONLY the provided postgres-pro tools.
    2. The task must be grounded in the given database meta_data. Use real table names, columns, and relationships from the schema.
    3. The task must be self-contained. Do not reference hidden files, CSV attachments, external services, or manual UI steps.
    4. The task should feel like MCPMark: realistic business framing, clear mission, explicit deliverables, and verifiable actions.
    5. Prefer tasks that are challenging but still feasible with schema inspection plus SQL execution.
    6. Avoid requiring impossible context, such as "knowing the company's intended policy" unless it can be derived from the database itself.
    7. Preserve the selected fusion family's ordered capability composition while adapting it to the new database.
    8. Do not merely rename the few-shot task. Create a new task with the same capability profile that is specific to the target database.
    9. Use concrete object names that plausibly exist in the target schema. When the database uses weak column names like A2/A3, phrase tasks around business-safe summaries rather than unsupported semantics.
    10. The final task should be answerable by an agent, not by a human evaluator directly.
    11. Low-hallucination rule: do not assume schemas, roles, extensions, policies, or privileged users already exist unless the meta_data explicitly supports that. If such objects are needed, ask the agent to create them as part of the task.
    12. Do not treat the database name as a schema name unless the meta_data explicitly shows that schema.
    13. Every existing table or view named in schema_entities_used must already exist in the meta_data exactly as written.
    14. Prefer conservative, evidence-based tasks over clever but weakly grounded tasks.
    15. When useful for benchmark alignment, you may include exactly one compact inline payload in the task: either a broken SQL query to repair/optimize, or a small table/JSON-like patch batch to ingest and process.
    16. Keep any inline payload compact and actionable. Prefer 5-12 rows for embedded data payloads and a single focused SQL statement or CTE block for SQL-fix payloads.

    Output JSON only with this schema:
    {
      "title": "short task title",
      "capability_tag": "fusion task type id",
      "archetype": "high-level archetype label",
      "question": "full benchmark-style task prompt",
      "why_executable": "short explanation grounded in the schema and tools",
      "required_tools": ["tool1", "tool2"],
      "schema_entities_used": ["table_or_view_1", "table_or_view_2"],
      "risk_notes": "short note about constraints or assumptions"
    }
    """
).strip()


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True)
class FusionPromptSpec:
    """One runtime task type derived from the new few-shot examples."""

    task_type_id: str
    pool: str
    capabilities: Tuple[str, ...]
    example_ids: Tuple[str, ...]
    notes: Tuple[str, ...]

    @property
    def archetype(self) -> str:
        return self.pool.lower()

    @property
    def focus(self) -> str:
        return ", ".join(item.replace("_", " ") for item in self.capabilities)

    @property
    def task_characteristics(self) -> List[str]:
        return [item.replace("_", " ") for item in self.capabilities]

    @property
    def generation_rules(self) -> List[str]:
        return [
            "Preserve the ordered capability composition shown by this fusion family.",
            "Adapt every operation to real entities and relationships in the target metadata.",
            "Weave the capabilities into one coherent, SQL-verifiable business workflow.",
        ]


@dataclass(frozen=True)
class FewShotExample:
    shot_id: str
    task_type_id: str
    archetype: str
    capability_tags: List[str]
    title: str
    question: str


def normalize_task_type_id(family: str) -> str:
    """Remove the ordering prefix copied from a new few-shot family label."""
    return re.sub(r"^[A-Za-z]+\d+_", "", family).strip().lower()


@lru_cache(maxsize=1)
def load_fusion_prompt_specs() -> Dict[str, FusionPromptSpec]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for path in sorted(FEW_SHOT_ROOT.glob("*.json")):
        row = load_json(path)
        task_type_id = normalize_task_type_id(str(row["family"]))
        row["_few_shot_id"] = path.stem
        grouped.setdefault(task_type_id, []).append(row)

    specs: Dict[str, FusionPromptSpec] = {}
    for task_type_id, rows in grouped.items():
        capabilities: List[str] = []
        for row in rows:
            for capability in row.get("capabilities", []):
                if capability not in capabilities:
                    capabilities.append(str(capability))
        specs[task_type_id] = FusionPromptSpec(
            task_type_id=task_type_id,
            pool=str(rows[0].get("pool", "MIXED")),
            capabilities=tuple(capabilities),
            example_ids=tuple(str(row["_few_shot_id"]) for row in rows),
            notes=tuple(str(row.get("note", "")) for row in rows),
        )
    return dict(sorted(specs.items()))


def prompt_variants_for_task_type(task_type_id: str) -> Tuple[str, ...]:
    example_count = len(load_fusion_prompt_specs()[task_type_id].example_ids)
    return PROMPT_VARIANTS[: min(example_count, len(PROMPT_VARIANTS))]


def list_meta_paths(root: Path) -> List[Path]:
    return sorted(root.glob("*/meta.json"))


def shorten(text: str, max_chars: int) -> str:
    clean = text.strip()
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 1].rstrip() + "…"


def parse_existing_tables(meta: Dict[str, Any]) -> Set[str]:
    schema_text = meta.get("meta_data", {}).get("stateContent") or meta.get("schema", {}).get("content") or ""
    table_names = set(re.findall(r'^Table\s+"([^"]+)"', schema_text, flags=re.MULTILINE))
    if not table_names:
        table_names.update(re.findall(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s+\d+\s+rows', schema_text, flags=re.MULTILINE))
    return table_names


def parse_existing_refs(meta: Dict[str, Any]) -> Set[str]:
    schema_text = meta.get("meta_data", {}).get("stateContent") or meta.get("schema", {}).get("content") or ""
    refs = set(re.findall(r'^Ref:\s+"([^"]+)"\."[^"]+"\s+>\s+"([^"]+)"\."[^"]+"', schema_text, flags=re.MULTILINE))
    flattened = set()
    for left, right in refs:
        flattened.add(left)
        flattened.add(right)
    return flattened


def parse_table_stats(meta: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    structured = meta.get("meta_data", {}).get("tableStats")
    if isinstance(structured, dict) and structured:
        normalized = {}
        for table_name, info in structured.items():
            normalized[table_name] = {
                "rows": int(info.get("rows", 0)),
                "sample_values": {
                    column: list(values)
                    for column, values in (info.get("sample_values", {}) or {}).items()
                },
                "value_ranges": {
                    column: {
                        "min": range_info.get("min"),
                        "max": range_info.get("max"),
                    }
                    for column, range_info in (info.get("value_ranges", {}) or {}).items()
                },
            }
        return normalized

    schema_text = meta.get("meta_data", {}).get("stateContent") or meta.get("schema", {}).get("content") or ""
    stats: Dict[str, Dict[str, Any]] = {}
    pattern = re.compile(r'^(?P<table>[A-Za-z_][A-Za-z0-9_]*):\s+(?P<rows>\d+)\s+rows\n(?P<body>(?:\s{2}.+\n)+)', re.MULTILINE)
    for match in pattern.finditer(schema_text):
        table = match.group("table")
        body = match.group("body")
        columns: Dict[str, List[str]] = {}
        for line in body.splitlines():
            column_match = re.match(r'^\s{2}([A-Za-z_][A-Za-z0-9_]*):\s+(.+)$', line)
            if not column_match:
                continue
            column = column_match.group(1)
            raw_values = column_match.group(2)
            values = [v.strip().strip("'") for v in raw_values.split(",") if v.strip() and v.strip() != "..."]
            columns[column] = values
        stats[table] = {
            "rows": int(match.group("rows")),
            "sample_values": columns,
            "value_ranges": {},
        }
    return stats


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def try_parse_json_object(text: str) -> Dict[str, Any]:
    raw = strip_code_fences(text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def infer_task_traits(question: str) -> Dict[str, Any]:
    q = question.lower()
    traits = {
        "has_sql_block": "```sql" in q,
        "mentions_create": any(token in q for token in ("create ", "build ", "implement ")),
        "mentions_insert": any(token in q for token in ("insert", "migrate", "load ", "populate ")),
        "mentions_update_delete": any(token in q for token in ("update", "delete", "transfer", "reassign")),
        "mentions_reporting": any(token in q for token in ("dashboard", "report", "summary", "analysis", "charts")),
        "mentions_optimization": any(token in q for token in ("optimize", "slow", "performance", "explain", "index")),
        "mentions_security": any(token in q for token in ("security", "permission", "grant", "role", "rls", "audit")),
        "mentions_function": any(token in q for token in ("function", "trigger", "procedure")),
        "mentions_view": any(token in q for token in ("view", "materialized view")),
    }
    return traits


def infer_tool_focus(question: str) -> List[str]:
    q = question.lower()
    tools = {"execute_sql"}
    if any(token in q for token in ("schema", "structure", "discover", "objects", "tables", "views")):
        tools.update({"list_schemas", "list_objects", "get_object_details"})
    if any(token in q for token in ("optimize", "performance", "slow", "execution plan", "index", "query")):
        tools.update({"explain_query", "analyze_query_indexes"})
    if any(token in q for token in ("top queries", "resource-intensive", "workload")):
        tools.update({"get_top_queries", "analyze_workload_indexes"})
    if any(token in q for token in ("health", "buffer", "vacuum", "replication", "bloated")):
        tools.add("analyze_db_health")
    if any(token in q for token in ("permission", "security", "audit", "roles", "rls")):
        tools.update({"list_schemas", "list_objects", "get_object_details"})
    return sorted(tools)


@lru_cache(maxsize=1)
def _load_fusion_few_shot_library() -> Tuple[FewShotExample, ...]:
    specs = load_fusion_prompt_specs()
    task_type_by_example = {
        example_id: task_type_id
        for task_type_id, spec in specs.items()
        for example_id in spec.example_ids
    }
    examples: List[FewShotExample] = []
    for path in sorted(FEW_SHOT_ROOT.glob("*.json")):
        payload = load_json(path)
        shot_id = path.stem
        task_type_id = task_type_by_example.get(shot_id)
        question = str(payload.get("question", "")).strip()
        if not task_type_id or not question:
            continue
        spec = specs[task_type_id]
        first_line = question.splitlines()[0].strip().lstrip("# ")
        examples.append(
            FewShotExample(
                shot_id=shot_id,
                task_type_id=task_type_id,
                archetype=spec.archetype,
                capability_tags=list(spec.capabilities),
                title=first_line or task_type_id.replace("_", " ").title(),
                question=question,
            )
        )
    expected = sum(len(spec.example_ids) for spec in specs.values())
    if len(examples) != expected:
        raise ValueError(
            f"Fusion few-shot mismatch: expected {expected}, found {len(examples)} "
            f"in {FEW_SHOT_ROOT}"
        )
    return tuple(examples)


def build_few_shot_library() -> List[FewShotExample]:
    return list(_load_fusion_few_shot_library())


def build_fusion_analysis() -> Dict[str, Any]:
    specs = load_fusion_prompt_specs()
    pool_counter = Counter(spec.pool for spec in specs.values())
    capability_counter = Counter(
        capability for spec in specs.values() for capability in spec.capabilities
    )
    return {
        "few_shot_root": str(FEW_SHOT_ROOT),
        "task_type_count": len(specs),
        "few_shot_count": sum(len(spec.example_ids) for spec in specs.values()),
        "pools": dict(sorted(pool_counter.items())),
        "capability_coverage": dict(sorted(capability_counter.items())),
        "task_types": [
            {
                "task_type_id": task_type_id,
                "pool": spec.pool,
                "capabilities": list(spec.capabilities),
                "example_ids": list(spec.example_ids),
            }
            for task_type_id, spec in specs.items()
        ],
    }


def render_analysis_text(analysis: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"Few-shot root: {analysis['few_shot_root']}")
    lines.append(f"Task type count: {analysis['task_type_count']}")
    lines.append(f"Few-shot count: {analysis['few_shot_count']}")
    lines.append("")
    lines.append("Pools:")
    for key, value in analysis["pools"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("Task types:")
    for task in analysis["task_types"]:
        lines.append(f"- {task['task_type_id']} [{task['pool']}]")
        lines.append(f"  examples: {', '.join(task['example_ids'])}")
        lines.append(f"  capabilities: {', '.join(task['capabilities'])}")
    return "\n".join(lines)


def load_meta_by_db(root: Path) -> Dict[str, Dict[str, Any]]:
    metas = {}
    for path in list_meta_paths(root):
        payload = load_json(path)
        db_id = payload.get("db_id") or path.parent.name
        payload["_path"] = str(path)
        payload["_table_names"] = sorted(parse_existing_tables(payload))
        payload["_ref_entities"] = sorted(parse_existing_refs(payload))
        payload["_table_stats"] = parse_table_stats(payload)
        metas[db_id] = payload
    return metas


def select_db_ids(all_db_ids: Iterable[str], args: argparse.Namespace) -> List[str]:
    db_ids = sorted(all_db_ids)
    if getattr(args, "db_id", None):
        missing = [db_id for db_id in args.db_id if db_id not in db_ids]
        if missing:
            raise ValueError(f"Unknown db_id(s): {missing}")
        db_ids = list(args.db_id)
    if getattr(args, "limit_dbs", None) is not None:
        db_ids = db_ids[: args.limit_dbs]
    return db_ids


def select_task_ids(all_task_ids: Iterable[str], args: argparse.Namespace) -> List[str]:
    task_ids = sorted(all_task_ids)
    if getattr(args, "task_ids", None):
        missing = [task_id for task_id in args.task_ids if task_id not in task_ids]
        if missing:
            raise ValueError(f"Unknown task id(s): {missing}")
        task_ids = list(args.task_ids)
    if getattr(args, "limit_tasks", None) is not None:
        task_ids = task_ids[: args.limit_tasks]
    return task_ids


def select_relevant_few_shot(
    task_type_id: str,
    prompt_variant: str,
) -> FewShotExample:
    spec = load_fusion_prompt_specs()[task_type_id]
    variants = prompt_variants_for_task_type(task_type_id)
    if prompt_variant not in variants:
        raise ValueError(
            f"Task type {task_type_id!r} does not provide {prompt_variant!r}; "
            f"available variants: {list(variants)}"
        )
    example_id = spec.example_ids[variants.index(prompt_variant)]
    examples = {example.shot_id: example for example in build_few_shot_library()}
    return examples[example_id]


def format_few_shot_example(
    task_type_id: str,
    prompt_variant: str,
) -> str:
    example = select_relevant_few_shot(task_type_id, prompt_variant)
    return textwrap.dedent(
        f"""
        [Few-Shot Example: {example.shot_id}]
        Task type: {example.task_type_id}
        Capability pool: {example.archetype}
        Atomic capabilities: {", ".join(example.capability_tags)}
        Title: {example.title}
        Task:
        {example.question.strip()}
        """
    ).strip()


def get_prompt_variant_guidance(prompt_variant: str) -> str:
    if prompt_variant == "prompt_v1":
        return textwrap.dedent(
            """
            Prompt variant objective:
            - Stay close to the selected fusion example's writing rhythm and deliverable style.
            - Prefer conservative adaptation: similar sectioning, comparable task scope, and familiar task framing.
            - Match benchmark complexity: 4-8 major numbered requirements, concrete objects, explicit verification, and enough detail to induce a long rollout.
            """
        ).strip()
    if prompt_variant == "prompt_v2":
        return textwrap.dedent(
            """
            Prompt variant objective:
            - Generate a clearly different task angle within the same fusion task type.
            - Change the business framing, deliverable mix, or focal entities while preserving the same capability composition.
            - Keep benchmark-close complexity: multi-stage requirements, concrete object specs, explicit verification, and enough operational detail to induce a long rollout.
            """
        ).strip()
    raise ValueError(f"Unknown prompt variant: {prompt_variant}")


def format_meta_context(meta: Dict[str, Any], max_meta_chars: int) -> str:
    schema_text = meta.get("meta_data", {}).get("stateContent") or meta.get("schema", {}).get("content") or ""
    schema_text = shorten(schema_text, max_meta_chars)
    known_tables = ", ".join(meta.get("_table_names", []))
    stats_lines = []
    for table, info in sorted(meta.get("_table_stats", {}).items()):
        column_bits = []
        for column, values in sorted(info.get("sample_values", {}).items()):
            if values:
                column_bits.append(f"{column}=[{', '.join(values[:3])}]")
        for column, range_info in sorted(info.get("value_ranges", {}).items()):
            min_value = range_info.get("min")
            max_value = range_info.get("max")
            if min_value is not None or max_value is not None:
                column_bits.append(f"{column}_range=[{min_value}..{max_value}]")
        summary = "; ".join(column_bits[:4])
        stats_lines.append(f"- {table}: rows={info.get('rows')}" + (f"; {summary}" if summary else ""))
    stats_text = "\n".join(stats_lines[:20])
    return textwrap.dedent(
        f"""
        db_id: {meta.get('db_id')}
        source: {meta.get('source')}
        tables: {meta.get('tables')}
        total_rows: {meta.get('total_rows')}
        meta_path: {meta.get('_path')}
        source_url: {meta.get('db_url', meta.get('meta_data', {}).get('stateUrl', ''))}
        known_existing_tables: {known_tables}
        compact_table_stats:
        {stats_text}

        meta_data:
        {schema_text}
        """
    ).strip()


def build_prompt(
    meta: Dict[str, Any],
    task_type_id: str,
    prompt_variant: str,
    max_meta_chars: int,
) -> str:
    specs = load_fusion_prompt_specs()
    analysis = build_fusion_analysis()
    spec = specs[task_type_id]
    traits = "\n".join(f"- {item}" for item in spec.task_characteristics)
    rules = "\n".join(f"- {item}" for item in spec.generation_rules)
    likely_tools = ", ".join(infer_tool_focus(" ".join(spec.capabilities + spec.notes)))
    pool_count = analysis["pools"].get(spec.pool, 0)
    variant_guidance = get_prompt_variant_guidance(prompt_variant)

    return textwrap.dedent(
        f"""
        {GLOBAL_GENERATION_RULES}

        {TOOL_SUMMARY}

        Target fusion task type: {task_type_id}
        Prompt variant: {prompt_variant}
        Capability pool: {spec.pool}
        Target capability composition: {spec.focus}
        Fusion task types in this pool: {pool_count}
        Few-shot examples available for this type: {len(spec.example_ids)}
        Likely tool usage pattern: {likely_tools}

        Ordered atomic capabilities to preserve:
        {traits}

        Generation rules for this fusion task type:
        {rules}

        Variant-specific guidance:
        {variant_guidance}

        Few-shot fusion reference:
        {format_few_shot_example(task_type_id, prompt_variant)}

        Target database context:
        {format_meta_context(meta, max_meta_chars)}

        Now generate one NEW task for this database that matches the target fusion task type.
        The few-shot reference is a style-and-capability anchor. Follow its tone, structure, and execution style closely.
        Keep the writing style close to MCPMark: realistic business setup, clear sections, explicit deliverables.
        Match benchmark-like complexity rather than writing a short summary task.
        Prefer 4-8 major numbered requirements, concrete object names, explicit schemas/fields/rules when justified by the database, and a closing verification phase.
        The task should be detailed enough to plausibly induce a 10-30 step agent rollout rather than a one-query answer.
        When it fits the capability composition, you may embed one compact task payload:
        - a broken SQL query that the agent must fix or optimize, or
        - a small table / JSON-like patch batch that the agent must stage, insert, migrate, deduplicate, or repair.
        If you embed a payload, keep it compact and make the downstream tasks explicitly consume it.
        The task must be distinct from the fusion references and must use the target database's actual schema entities.
        When this task type has another prompt variant, the task generated for {prompt_variant} must be meaningfully different from it.
        Existing entities in schema_entities_used must be selected only from known_existing_tables.
        You may ask the agent to create new tables/views/functions/audit logs, but do not list those new objects in schema_entities_used.
        Do not require a pre-existing custom role, schema, extension, or policy unless the meta_data explicitly shows it exists.
        If you mention concrete categorical values, statuses, or enum-like labels, they should come from the sample values shown in meta_data.
        Do not invent specific numeric ID ranges like "100-200" unless the meta_data explicitly supports that range. Prefer phrases like "existing accounts from a selected subset" or instruct the agent to discover the exact IDs first.
        Row counts can justify task scale, but they do not justify invented primary-key ranges.
        Return JSON only.
        """
    ).strip()


def call_model(
    client: OpenAI,
    prompt: str,
    model_name: str,
    temperature: float,
    max_completion_tokens: int,
) -> str:
    response = client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        messages=[
            {
                "role": "system",
                "content": "You are a precise PostgreSQL task writer. Return JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""


async def call_model_async(
    client: AsyncOpenAI,
    prompt: str,
    model_name: str,
    temperature: float,
    max_completion_tokens: int,
) -> str:
    response = await client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        messages=[
            {
                "role": "system",
                "content": "You are a precise PostgreSQL task writer. Return JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""


def validate_generation_payload(payload: Dict[str, Any], task_id: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    required_keys = [
        "title",
        "capability_tag",
        "archetype",
        "question",
        "why_executable",
        "required_tools",
        "schema_entities_used",
        "risk_notes",
    ]
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(f"Generated payload missing keys: {missing}")
    payload["capability_tag"] = payload.get("capability_tag") or task_id
    if not isinstance(payload["required_tools"], list):
        raise ValueError("required_tools must be a list")
    if not isinstance(payload["schema_entities_used"], list):
        raise ValueError("schema_entities_used must be a list")
    if not payload["schema_entities_used"]:
        raise ValueError("schema_entities_used must not be empty")
    unknown_tools = [tool for tool in payload["required_tools"] if tool not in ALLOWED_TOOLS]
    if unknown_tools:
        raise ValueError(f"required_tools contains unsupported tools: {unknown_tools}")

    existing_tables = set(meta.get("_table_names", []))
    unknown_entities = [entity for entity in payload["schema_entities_used"] if entity not in existing_tables]
    if unknown_entities:
        raise ValueError(f"schema_entities_used contains unknown existing entities: {unknown_entities}")

    question = payload["question"].strip()
    if not question:
        raise ValueError("question must not be empty")

    db_id = str(meta.get("db_id", "")).strip()
    if db_id and re.search(rf"\b{re.escape(db_id)}\s+schema\b", question, flags=re.IGNORECASE):
        raise ValueError(f"question incorrectly treats db_id '{db_id}' as a schema name")

    # Reject tasks that require pre-existing custom roles instead of asking the agent to create them.
    if re.search(r"only users with [`'\"]?[A-Za-z_][A-Za-z0-9_]*[`'\"]?\s+role", question, flags=re.IGNORECASE):
        if not re.search(r"\bcreate\s+role\b|\bcreate\s+the\s+role\b|\bgrant\b", question, flags=re.IGNORECASE):
            raise ValueError("question depends on an unexplained pre-existing custom role")

    if re.search(r"\bsecurity\s+definer'?s?\s+rights\b", question, flags=re.IGNORECASE):
        if "function" not in question.lower():
            raise ValueError("security definer requirement appears without a function context")

    # Reject invented hard-coded ID ranges unless meta explicitly gives such evidence.
    id_range_patterns = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*_id)\b[^.\n]{0,40}\b(\d+)\s*-\s*(\d+)\b", question, flags=re.IGNORECASE)
    if id_range_patterns:
        sample_values = {}
        for table_info in meta.get("_table_stats", {}).values():
            sample_values.update(table_info.get("sample_values", {}))
        for column_name, start, end in id_range_patterns:
            observed = sample_values.get(column_name, [])
            observed_numeric = {v for v in observed if re.fullmatch(r"\d+", v)}
            if not observed_numeric:
                raise ValueError(f"question invents numeric range for {column_name} without sample-value support")
            start_i = int(start)
            end_i = int(end)
            observed_ints = [int(v) for v in observed_numeric]
            if start_i < min(observed_ints) or end_i > max(observed_ints):
                raise ValueError(f"question invents unsupported numeric range for {column_name}: {start}-{end}")

    # Reject invented categorical values for existing columns when meta provides sample values.
    sample_values_by_column: Dict[str, Set[str]] = {}
    for table_info in meta.get("_table_stats", {}).values():
        for column_name, values in table_info.get("sample_values", {}).items():
            sample_values_by_column.setdefault(column_name.lower(), set()).update(v for v in values if v and v != "...")

    for column_name, quoted_value in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b\s*(?:=|to)\s*'([^']+)'", question, flags=re.IGNORECASE):
        observed_values = sample_values_by_column.get(column_name.lower())
        if observed_values and quoted_value not in observed_values:
            raise ValueError(f"question invents unsupported categorical value for {column_name}: {quoted_value!r}")

    # Existing entities should appear in the natural-language task, to reduce disconnected outputs.
    missing_mentions = []
    lowered_question = question.lower()
    for entity in payload["schema_entities_used"]:
        if entity.lower() not in lowered_question:
            missing_mentions.append(entity)
    if missing_mentions:
        raise ValueError(f"question does not mention some schema_entities_used: {missing_mentions}")

    return payload


def generate_with_retries(
    client: OpenAI,
    prompt: str,
    model_name: str,
    temperature: float,
    max_completion_tokens: int,
    task_id: str,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    active_prompt = prompt

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        raw_text = call_model(
            client=client,
            prompt=active_prompt,
            model_name=model_name,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
        try:
            parsed = try_parse_json_object(raw_text)
            return validate_generation_payload(parsed, task_id, meta)
        except Exception as exc:
            last_error = exc
            active_prompt = (
                prompt
                + "\n\n"
                + "Your previous output was rejected.\n"
                + f"Reason: {exc}\n"
                + "Return corrected JSON only. Be conservative and low-hallucination."
            )

    assert last_error is not None
    raise last_error


async def generate_with_retries_async(
    client: AsyncOpenAI,
    prompt: str,
    model_name: str,
    temperature: float,
    max_completion_tokens: int,
    task_id: str,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    active_prompt = prompt

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        raw_text = await call_model_async(
            client=client,
            prompt=active_prompt,
            model_name=model_name,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
        try:
            parsed = try_parse_json_object(raw_text)
            return validate_generation_payload(parsed, task_id, meta)
        except Exception as exc:
            last_error = exc
            active_prompt = (
                prompt
                + "\n\n"
                + "Your previous output was rejected.\n"
                + f"Reason: {exc}\n"
                + "Return corrected JSON only. Be conservative and low-hallucination."
            )

    assert last_error is not None
    raise last_error


def build_task_config(db_id: str, question: str) -> Dict[str, Any]:
    return {
        "category": db_id,
        "question": question,
        "use_specified_server": True,
        "mcp_servers": [{"name": "postgres-pro"}],
        "output_format": {"status": "PostgreSQL task completion report"},
        "evaluators": [],
        "prepares": [
            {"prepare_func": "mcpmark_postgres_setup", "prepare_args": {}}
        ],
        "cleanups": [
            {
                "server": "mcpmark",
                "tool": "",
                "cleanup_func": "postgres_cleanup",
                "cleanup_args": {},
            }
        ],
    }


def build_export_output_path(export_root: Path, task_id: str, prompt_variant: str, db_id: str) -> Path:
    variant_dir = export_root / task_id / prompt_variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    return variant_dir / f"{db_id}.json"


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_valid_existing_output(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    required = {"category", "question", "use_specified_server", "mcp_servers", "output_format", "evaluators", "prepares", "cleanups"}
    return required.issubset(payload.keys())


def count_scope(_: argparse.Namespace) -> int:
    specs = load_fusion_prompt_specs()
    database_count = len(load_meta_by_db(META_ROOT))
    prompt_slots = sum(len(prompt_variants_for_task_type(item)) for item in specs)
    summary = {
        "database_count": database_count,
        "fusion_task_type_count": len(specs),
        "few_shot_prompt_slot_count": prompt_slots,
        "total_generation_jobs": database_count * prompt_slots,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def analyze_fusion(args: argparse.Namespace) -> int:
    analysis = build_fusion_analysis()
    rendered = (
        json.dumps(analysis, ensure_ascii=False, indent=2)
        if args.format == "json"
        else render_analysis_text(analysis)
    )
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect fusion-derived PostgreSQL task-sampling types."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--format", choices=("text", "json"), default="text")
    analyze_parser.add_argument("--output", default="")
    analyze_parser.set_defaults(func=analyze_fusion)
    count_parser = subparsers.add_parser("count")
    count_parser.set_defaults(func=count_scope)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
