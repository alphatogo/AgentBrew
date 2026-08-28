#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filter generated trajectory SQL tasks using fusion-reference and schema-grounding heuristics.

This script scans an existing trajectory_sql tree, evaluates each JSON task against:
1. Basic JSON/task structure validity
2. Fusion-reference length expectations per task type
3. Payload alignment for fix/optimization tasks
4. Meta-schema grounding checks using the corresponding meta.json
5. A few explicit invalid-pattern heuristics discovered during manual review

It writes:
- kept/<mirrored task tree>
- rejected/<mirrored task tree>
- filter_report.jsonl
- filter_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


POSTGRES_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT_ROOT = REPO_ROOT / "outputs/postgres_task_sample"
DEFAULT_META_ROOT = POSTGRES_ROOT / "assets/metadata"
DEFAULT_REFERENCE_ROOT = POSTGRES_ROOT / "task_sampling/few_shot"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/postgres_task_sample_filtered"

PAYLOAD_SQL_TASK_IDS = {
    "debug_fix_analytics_query",
    "optimize_slow_analytics_query",
    "query_opt_with_report",
    "debug_plus_optimize_plus_index",
}

TASK_IDS_WHERE_PAYLOAD_IS_HELPFUL = {
    "bulk_data_migration",
    "consistency_trigger_enforcement",
    "inventory_crud_report",
    "scored_status_summary",
    "migration_plus_hierarchy_rollup",
}

GENERIC_TOKENS = {
    "select",
    "insert",
    "update",
    "delete",
    "create",
    "table",
    "view",
    "materialized",
    "function",
    "trigger",
    "policy",
    "role",
    "index",
    "report",
    "summary",
    "status",
    "reason",
    "timestamp",
    "current_user",
}


@dataclass
class Issue:
    severity: str
    code: str
    message: str


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_task_files(root: Path) -> List[Path]:
    return sorted(root.glob("*/*/*.json"))


def list_reference_files(root: Path) -> List[Path]:
    return sorted(root.glob("**/*.json"))


def parse_schema_tables(meta_payload: Dict[str, Any]) -> Dict[str, Set[str]]:
    text = meta_payload.get("meta_data", {}).get("stateContent") or meta_payload.get("schema", {}).get("content") or ""
    tables: Dict[str, Set[str]] = {}
    table_matches = list(re.finditer(r'Table\s+"([^"]+)"\s+\{', text))
    for index, match in enumerate(table_matches):
        table_name = match.group(1)
        start = match.end()
        end = table_matches[index + 1].start() if index + 1 < len(table_matches) else len(text)
        block = text[start:end]
        columns = set(re.findall(r'^\s+"([^"]+)"', block, flags=re.MULTILINE))
        tables[table_name.lower()] = {column.lower() for column in columns}
    return tables


def load_meta_schemas(meta_root: Path) -> Dict[str, Dict[str, Set[str]]]:
    metas: Dict[str, Dict[str, Set[str]]] = {}
    for path in sorted(meta_root.glob("*/meta.json")):
        payload = load_json(path)
        db_id = payload.get("db_id") or path.parent.name
        metas[db_id] = parse_schema_tables(payload)
    return metas


def load_reference_lengths(reference_root: Path) -> Dict[str, int]:
    grouped: Dict[str, List[int]] = {}
    for path in list_reference_files(reference_root):
        payload = load_json(path)
        family = str(payload.get("family", ""))
        task_type_id = re.sub(r"^[A-Za-z]+\d+_", "", family).strip().lower()
        if task_type_id:
            grouped.setdefault(task_type_id, []).append(
                len(payload.get("question", ""))
            )
    return {
        task_type_id: sum(values) // len(values)
        for task_type_id, values in grouped.items()
    }


def extract_task_info(task_path: Path, input_root: Path) -> Tuple[str, str, str]:
    rel = task_path.relative_to(input_root)
    return rel.parts[0], rel.parts[1], rel.stem


def has_sql_payload(question: str) -> bool:
    return "```sql" in question.lower()


def has_data_payload(question: str) -> bool:
    return "```json" in question.lower() or "| --- |" in question


def find_explicit_table_column_refs(text: str) -> List[Tuple[str, str]]:
    refs: List[Tuple[str, str]] = []

    for table, column in re.findall(r'`([A-Za-z_][A-Za-z0-9_]*)`\.`([A-Za-z_][A-Za-z0-9_]*)`', text):
        refs.append((table.lower(), column.lower()))
    for table, column in re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"\."([^"]+)"', text):
        refs.append((table.lower(), column.lower()))
    for table, column in re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b', text):
        refs.append((table.lower(), column.lower()))

    deduped: List[Tuple[str, str]] = []
    seen = set()
    for ref in refs:
        if ref not in seen:
            seen.add(ref)
            deduped.append(ref)
    return deduped


def find_line_scoped_column_claims(question: str, schema: Dict[str, Set[str]]) -> List[Tuple[str, str, str]]:
    findings: List[Tuple[str, str, str]] = []
    for raw_line in question.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        tables_in_line = [table for table in schema if re.search(rf'(`{re.escape(table)}`|\b{re.escape(table)}\b)', lowered)]
        if not tables_in_line:
            continue
        backticked_tokens = [token.lower() for token in re.findall(r'`([^`]+)`', line)]
        candidate_columns = []
        for token in backticked_tokens:
            if token in schema:
                continue
            if token in GENERIC_TOKENS:
                continue
            if re.fullmatch(r"[a-z_][a-z0-9_]*", token):
                candidate_columns.append(token)
        for table in tables_in_line:
            existing_columns = schema.get(table, set())
            for column in candidate_columns:
                if column not in existing_columns and column.endswith(("_id", "_date", "_status", "_balance")):
                    findings.append((table, column, line))
    return findings


def analyze_task(
    task_payload: Dict[str, Any],
    task_id: str,
    db_id: str,
    schema: Dict[str, Set[str]],
    reference_length: int,
    min_length_ratio: float,
) -> List[Issue]:
    issues: List[Issue] = []
    question = task_payload.get("question")

    required_fields = {
        "category",
        "question",
        "use_specified_server",
        "mcp_servers",
        "output_format",
        "evaluators",
        "prepares",
        "cleanups",
    }
    missing_fields = sorted(field for field in required_fields if field not in task_payload)
    if missing_fields:
        issues.append(Issue("reject", "missing_fields", f"Missing required fields: {missing_fields}"))
        return issues

    if not isinstance(question, str) or not question.strip():
        issues.append(Issue("reject", "empty_question", "Question is empty or invalid."))
        return issues

    if task_payload.get("category") != db_id:
        issues.append(Issue("reject", "category_db_mismatch", f"category={task_payload.get('category')} but db_id={db_id}"))

    min_chars = min(max(1000, int(reference_length * min_length_ratio)), 3200)
    if len(question) < min_chars:
        issues.append(Issue("reject", "too_short", f"Question length {len(question)} is below threshold {min_chars} for {task_id}."))

    sql_payload = has_sql_payload(question)
    data_payload = has_data_payload(question)

    if task_id in PAYLOAD_SQL_TASK_IDS and not sql_payload:
        issues.append(Issue("reject", "missing_sql_payload", f"{task_id} should usually include an inline SQL payload."))

    explicit_refs = find_explicit_table_column_refs(question)
    for table, column in explicit_refs:
        if table in schema and column not in schema[table]:
            issues.append(Issue("reject", "unknown_table_column", f"Explicit reference {table}.{column} not found in meta schema."))

    for table, column, line in find_line_scoped_column_claims(question, schema):
        issues.append(Issue("reject", "line_scoped_unknown_column", f"Line suggests {table}.{column}, but that column is not in schema: {line}"))

    if "materialized view" in question.lower() and "on commit" in question.lower():
        issues.append(Issue("reject", "invalid_materialized_view_refresh", "Question mentions materialized view refresh ON COMMIT, which is not valid PostgreSQL behavior."))

    if sql_payload:
        payload = extract_first_sql_block(question)
        for alias, column in detect_subquery_alias_projection_issues(payload):
            issues.append(
                Issue(
                    "reject",
                    "broken_sql_payload_shape",
                    f'SQL payload references {alias}."{column}" outside the subquery, but that column is not clearly projected by alias {alias}.',
                )
            )

    if task_id in {
        "rls_role_based_security",
        "multi_entity_rls_matrix",
        "rls_matrix_plus_permission_audit",
    }:
        if "current user's assigned" in question.lower() and "mapping table" not in question.lower() and "create" not in question.lower():
            issues.append(Issue("warn", "implicit_scope_mapping", "Security task relies on current-user scope mapping without clearly creating or grounding that mapping."))

    if task_id in TASK_IDS_WHERE_PAYLOAD_IS_HELPFUL and not sql_payload and not data_payload:
        issues.append(Issue("warn", "no_inline_payload", f"{task_id} could benefit from an inline payload but none was included."))

    return issues


def extract_first_sql_block(question: str) -> str:
    match = re.search(r"```sql\s*(.*?)```", question, re.DOTALL | re.IGNORECASE)
    return match.group(1) if match else ""


def detect_subquery_alias_projection_issues(sql: str) -> List[Tuple[str, str]]:
    issues: List[Tuple[str, str]] = []
    if not sql:
        return issues

    alias_projection_map: Dict[str, Set[str]] = {}
    pattern = re.compile(r"FROM\s*\(\s*SELECT(.*?)\)\s*([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE | re.DOTALL)
    for select_body, alias in pattern.findall(sql):
        projected = set()
        for quoted_column in re.findall(r'"([^"]+)"', select_body):
            projected.add(quoted_column.lower())
        for bare_column in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", select_body):
            projected.add(bare_column.lower())
        alias_projection_map[alias] = projected

    outer_refs = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\."([^"]+)"', sql)
    seen = set()
    for alias, column in outer_refs:
        if alias not in alias_projection_map:
            continue
        key = (alias, column.lower())
        if key in seen:
            continue
        seen.add(key)
        if column.lower() not in alias_projection_map[alias]:
            issues.append((alias, column))
    return issues


def classify_issues(issues: List[Issue]) -> str:
    if any(issue.severity == "reject" for issue in issues):
        return "reject"
    return "keep"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def mirror_copy(src: Path, dest_root: Path, relative_path: Path) -> Path:
    dest = dest_root / relative_path
    ensure_parent(dest)
    shutil.copy2(src, dest)
    return dest


def filter_task_tree(
    input_root: Path,
    meta_root: Path = DEFAULT_META_ROOT,
    reference_root: Path = DEFAULT_REFERENCE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    min_length_ratio: float = 0.25,
) -> Dict[str, Any]:
    """Apply the migrated post-generation filter and write its mirrored outputs."""
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    kept_root = output_root / "kept"
    rejected_root = output_root / "rejected"
    report_path = output_root / "filter_report.jsonl"
    summary_path = output_root / "filter_summary.json"
    reference_lengths = load_reference_lengths(reference_root.resolve())
    meta_schemas = load_meta_schemas(meta_root.resolve())
    task_files = list_task_files(input_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_counter = Counter()
    by_task_id = Counter()
    report_rows = []
    for task_file in task_files:
        task_id, prompt_variant, db_id = extract_task_info(task_file, input_root)
        by_task_id[task_id] += 1
        task_payload = load_json(task_file)
        issues = analyze_task(
            task_payload=task_payload,
            task_id=task_id,
            db_id=db_id,
            schema=meta_schemas.get(db_id, {}),
            reference_length=reference_lengths.get(task_id, 2000),
            min_length_ratio=min_length_ratio,
        )
        decision = classify_issues(issues)
        summary_counter[decision] += 1
        relative_path = task_file.relative_to(input_root)
        mirror_copy(
            task_file,
            kept_root if decision == "keep" else rejected_root,
            relative_path,
        )
        report_rows.append(
            {
                "task_path": str(task_file),
                "relative_path": str(relative_path),
                "task_id": task_id,
                "prompt_variant": prompt_variant,
                "db_id": db_id,
                "decision": decision,
                "issue_count": len(issues),
                "issues": [
                    {
                        "severity": issue.severity,
                        "code": issue.code,
                        "message": issue.message,
                    }
                    for issue in issues
                ],
                "question_length_chars": len(task_payload.get("question", "")),
                "has_sql_payload": has_sql_payload(task_payload.get("question", "")),
                "has_data_payload": has_data_payload(task_payload.get("question", "")),
            }
        )
    with report_path.open("w", encoding="utf-8") as handle:
        for row in report_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "task_file_count": len(task_files),
        "kept": summary_counter["keep"],
        "rejected": summary_counter["reject"],
        "task_counts": dict(sorted(by_task_id.items())),
        "report_path": str(report_path),
        "kept_root": str(kept_root),
        "rejected_root": str(rejected_root),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter generated trajectory SQL tasks.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--meta-root", type=Path, default=DEFAULT_META_ROOT)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--min-length-ratio", type=float, default=0.25)
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    kept_root = output_root / "kept"
    rejected_root = output_root / "rejected"
    report_path = output_root / "filter_report.jsonl"
    summary_path = output_root / "filter_summary.json"

    reference_lengths = load_reference_lengths(args.reference_root.resolve())
    meta_schemas = load_meta_schemas(args.meta_root.resolve())
    task_files = list_task_files(input_root)

    output_root.mkdir(parents=True, exist_ok=True)

    summary_counter = Counter()
    by_task_id = Counter()
    report_rows = []

    for task_file in task_files:
        task_id, prompt_variant, db_id = extract_task_info(task_file, input_root)
        by_task_id[task_id] += 1
        task_payload = load_json(task_file)
        schema = meta_schemas.get(db_id, {})
        reference_length = reference_lengths.get(task_id, 2000)
        issues = analyze_task(
            task_payload=task_payload,
            task_id=task_id,
            db_id=db_id,
            schema=schema,
            reference_length=reference_length,
            min_length_ratio=args.min_length_ratio,
        )
        decision = classify_issues(issues)
        summary_counter[decision] += 1

        relative_path = task_file.relative_to(input_root)
        if decision == "keep":
            mirror_copy(task_file, kept_root, relative_path)
        else:
            mirror_copy(task_file, rejected_root, relative_path)

        report_rows.append(
            {
                "task_path": str(task_file),
                "relative_path": str(relative_path),
                "task_id": task_id,
                "prompt_variant": prompt_variant,
                "db_id": db_id,
                "decision": decision,
                "issue_count": len(issues),
                "issues": [{"severity": issue.severity, "code": issue.code, "message": issue.message} for issue in issues],
                "question_length_chars": len(task_payload.get("question", "")),
                "has_sql_payload": has_sql_payload(task_payload.get("question", "")),
                "has_data_payload": has_data_payload(task_payload.get("question", "")),
            }
        )

    with report_path.open("w", encoding="utf-8") as f:
        for row in report_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "task_file_count": len(task_files),
        "kept": summary_counter["keep"],
        "rejected": summary_counter["reject"],
        "task_counts": dict(sorted(by_task_id.items())),
        "report_path": str(report_path),
        "kept_root": str(kept_root),
        "rejected_root": str(rejected_root),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
