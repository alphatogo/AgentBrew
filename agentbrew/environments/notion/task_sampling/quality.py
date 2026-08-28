"""Quality checks for generated Notion sampling tasks."""

from __future__ import annotations

import json
import re
from typing import Any


_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_GENERIC_PLACEHOLDER_RE = re.compile(
    r"^(?:[A-Za-z\d]|[A-Z]{2,}|(?:your\s+)?(?:employee|team|task|user|person|people|"
    r"member|manager|lead|project|item|entry|record|row|page|database|name|date|time|"
    r"value|title|text|status|count|number|description|type|category|label|field|column|"
    r"property|tag|priority|note|comment|result|output|answer|original|new|existing|"
    r"current|growth|area)(?:\s+\w+){0,2})$",
    re.IGNORECASE,
)


def extract_json_payload(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model response."""
    stripped = text.strip()
    think_end = stripped.rfind("</think>")
    if think_end != -1:
        stripped = stripped[think_end + len("</think>") :].strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError("Model response is not a JSON object")
        return payload
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model response")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                payload = json.loads(stripped[start : index + 1])
                if not isinstance(payload, dict):
                    raise ValueError("Model response is not a JSON object")
                return payload
    raise ValueError("Unterminated JSON object in model response")


def extract_generated_question(response: str) -> str:
    """Read the generated task from the current or legacy response field."""
    payload = extract_json_payload(response)
    question = payload.get("answer") or payload.get("generated_question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Model response must contain a non-empty answer")
    return question.strip()


def _context_objects(template_item: dict[str, Any]) -> dict[str, set[str]]:
    context = template_item.get("taskgen_context", {}) or {}
    pages = context.get("pages", []) or []
    data_sources = context.get("explicit_data_sources", []) or []
    child_databases = context.get("child_databases", []) or []
    page_titles = {str(page["title"]) for page in pages if page.get("title")}
    data_source_names = {
        str(data_source.get("display_name") or data_source.get("name"))
        for data_source in data_sources
        if data_source.get("display_name") or data_source.get("name")
    }
    child_database_titles = {
        str(database["title"])
        for database in child_databases
        if database.get("title")
    }
    field_names = {
        str(field["name"])
        for data_source in data_sources
        for field in (data_source.get("field_summaries") or [])
        if field.get("name")
    }
    return {
        "page_titles": page_titles,
        "data_source_names": data_source_names,
        "existing_database_titles": data_source_names | child_database_titles,
        "field_names": field_names,
    }


def _contains_placeholder(text: str) -> bool:
    for match in re.finditer(r"\[([^\]]+)\]", text):
        inner = match.group(1).strip()
        if re.search(r"\b(unnamed|omitted)\b", inner, re.IGNORECASE):
            continue
        if len(inner.split()) > 4:
            continue
        if _GENERIC_PLACEHOLDER_RE.match(inner):
            return True
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in ("fill in", "existing team member", "existing employee", "to be filled")
    )


def _common_issues(question: str, template_item: dict[str, Any]) -> list[str]:
    """Reject unsafe or nonexistent object references using old sampler rules."""
    reasons: list[str] = []
    lowered = question.lower()
    objects = _context_objects(template_item)
    if _contains_placeholder(question):
        reasons.append("contains placeholder text")
    if _UUID_RE.search(question):
        reasons.append("contains hardcoded UUID")

    for field in objects["field_names"]:
        field_lower = field.lower()
        if any(
            phrase in lowered
            for phrase in (
                f"@-mention {field_lower}",
                f"link to the {field_lower}",
                f"mention of the {field_lower}",
            )
        ):
            reasons.append("references a field/property as an object")
            break

    for page_title in objects["page_titles"]:
        page_lower = page_title.lower()
        if (
            f"database from the {page_lower} page" in lowered
            or f"embedded database from the {page_lower} page" in lowered
        ):
            reasons.append("treats a page as a concrete database source")
            break

    existing_databases = {name.lower() for name in objects["existing_database_titles"]}
    existing_pages = {name.lower() for name in objects["page_titles"]}
    quoted_database_patterns = [
        r"relation to the [\"']([^\"']+)[\"'] database",
        r"existing entry in the [\"']([^\"']+)[\"'] database",
        r"entries in the [\"']([^\"']+)[\"'] database",
        r"rows in the [\"']([^\"']+)[\"'] database",
        r"from the [\"']([^\"']+)[\"'] database",
        r"references? the [\"']([^\"']+)[\"'] database",
    ]
    for pattern in quoted_database_patterns:
        for name in re.findall(pattern, question, flags=re.IGNORECASE):
            if name.lower() not in existing_databases:
                reasons.append(f"references nonexistent database title: {name}")

    for pattern in (
        r"under the existing [\"']([^\"']+)[\"'] page",
        r"within the existing [\"']([^\"']+)[\"'] page",
        r"under the [\"']([^\"']+)[\"'] page",
    ):
        for name in re.findall(pattern, question, flags=re.IGNORECASE):
            if name.lower() not in existing_pages:
                reasons.append(f"references nonexistent page title: {name}")

    if "preserve its original schema" in lowered or "preserve their original schema" in lowered:
        reasons.append("requires preserving schema of an existing database")
    if "preserve all rows" in lowered or "all 7 rows" in lowered:
        reasons.append("requires preserving or copying all rows of an existing database")
    return list(dict.fromkeys(reasons))


def _contains(text: str, phrases: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


def _capability_predicates(question: str, template_item: dict[str, Any]) -> dict[str, bool]:
    """Map coarse capability tags to the same lexical checks used by the old sampler."""
    query = _contains(
        question,
        (
            "query",
            "filter",
            "where",
            "sorted by",
            "within the next",
            "before",
            "after",
            "only rows",
            "retrieve",
            "find all",
            "look up",
            "fetch",
            "whose",
            "that have",
            "with status",
            "with a",
            "matching",
        ),
    )
    aggregate = _contains(
        question,
        (
            "average",
            "most common",
            "summary",
            "summarize",
            "top ",
            "frequent",
            "count",
            "total",
            "sum",
            "percentage",
            "how many",
            "number of",
            "per category",
            "aggregate",
            "statistics",
            "breakdown",
            "across all",
            "tally",
        ),
    )
    preserve = _contains(
        question,
        (
            "preserve",
            "without losing",
            "maintain",
            "in order",
            "keep the order",
            "original order",
            "existing order",
            "do not change",
        ),
    )
    relations = _contains(question, ("relation", "relate", "link", "connect", "bidirectional", "related to"))
    return {
        "create_database": _contains(
            question,
            ("create a new database", "create a database", "create a new inline database", "add a database"),
        ),
        "define_schema": _contains(
            question,
            (
                "with the following properties",
                "properties *exactly*",
                "schema",
                "with properties",
                "with fields",
                "with the properties",
                "following columns",
                "the following fields",
                "property types",
            ),
        ),
        "populate_rows": (
            "populate" in question.lower()
            and _contains(question, ("rows", "entries", "records", "pages"))
        ) or _contains(
            question,
            (
                "add the following rows",
                "add rows",
                "create the following entries",
                "add entries",
                "add records",
                "create rows",
                "insert rows",
                "fill in the rows",
            ),
        ),
        "create_child_page": _contains(
            question,
            (
                "create a child page",
                "create a new child page",
                "create a new page",
                "create a sub-page",
                "add a child page",
                "create a page titled",
            ),
        ),
        "insert_ordered_sections": _contains(
            question,
            (
                "in order",
                "ordered sections",
                "containing, in order",
                "in the following order",
                "sections:",
                "with the following sections",
                "the following blocks",
                "in sequence",
            ),
        ),
        "reference_existing_object": _references_existing_object(question, template_item),
        "query_and_filter_database": query,
        "query_existing_data": query,
        "query_multiple_sources": _contains(
            question,
            ("query", "filter", "from", "retrieve", "find", "search", "look up"),
        ),
        "summarize_from_records": aggregate,
        "compute_aggregates": aggregate,
        "reorganize_layout": _contains(
            question,
            (
                "restructure",
                "reorganize",
                "convert",
                "toggle",
                "move content",
                "swap",
                "column",
                "rearrange",
                "reorder",
                "reformat",
                "relocate",
                "nest",
            ),
        ),
        "add_exact_blocks": _contains(
            question,
            (
                "callout",
                "heading_",
                "heading ",
                "bulleted list",
                "numbered list",
                "to-do",
                "toggle",
                "table block",
                "divider",
                "code block",
                "quote block",
                "paragraph block",
                "exactly",
                "the following blocks",
            ),
        ),
        "convert_block_type": _contains(
            question,
            ("convert", "turn into", "change to a toggle", "change to a heading", "as a toggle"),
        ),
        "move_existing_blocks": _contains(
            question,
            ("move", "relocate", "place under", "nest under", "drag", "under the"),
        ),
        "preserve_content_order": preserve,
        "preserve_relative_order": preserve,
        "preserve_order": preserve,
        "preserve_format": preserve,
        "update_schema": _contains(
            question,
            (
                "add a",
                "add the",
                "new property",
                "new field",
                "new column",
                "update the schema",
                "add relation",
                "add a relation",
                "add a rollup",
                "add a formula",
                "add a property",
            ),
        ),
        "build_relations": relations,
        "cross_database_linking": relations,
        "insert_block_at_position": _contains(
            question,
            (
                "insert",
                "add a block",
                "place a",
                "add below",
                "add above",
                "at the top of",
                "at the bottom of",
                "after the",
                "before the",
                "just below",
                "directly below",
                "immediately after",
            ),
        ),
        "insert_standard_block": _contains(
            question,
            (
                "insert",
                "add a block",
                "place a",
                "add below",
                "add above",
                "at the top of",
                "at the bottom of",
                "after the",
                "before the",
                "just below",
                "directly below",
                "immediately after",
            ),
        ),
        "match_existing_format": _contains(
            question,
            (
                "match",
                "consistent with",
                "same style",
                "following the",
                "mimic",
                "same format",
                "same structure",
                "same layout",
                "same pattern",
            ),
        ),
        "migrate_rows": _contains(
            question,
            ("migrate", "transfer", "copy", "move the row", "move entries", "import"),
        ),
        "archive_or_delete_items": _contains(
            question,
            ("archive", "delete", "remove", "retire", "trash", "mark as archived"),
        ),
        "batch_update_pages": _contains(
            question,
            ("for each", "for all", "all matching", "every", "batch", "across all", "each page", "each row", "each entry", "update all"),
        ),
        "batch_update_items": _contains(
            question,
            ("for each", "all", "every", "batch", "across all", "each item", "update all", "for all"),
        ),
        "update_existing_rows": _contains(
            question,
            ("update", "change", "set", "modify", "edit", "mark as", "assign", "fill in", "overwrite", "revise"),
        ),
        "coordinated_multirow_edit": _contains(
            question,
            ("across", "multiple rows", "both", "consistently", "swap", "reassign", "exchange", "coordinated", "simultaneously"),
        ),
        "precise_selection": _contains(
            question,
            ("only", "specific", "matching", "whose", "where", "criteria", "that are", "with status", "only the", "exact", "precisely"),
        ),
        "cross_database_sync": _contains(question, ("sync", "corresponding", "mirror", "replicate", "duplicate into")),
        "field_mapping": _contains(
            question,
            ("map", "corresponding field", "field mapping", "map to", "into the", "as the", "using the value", "using the", "from the field"),
        ),
        "transform_database_to_blocks": _contains(
            question,
            ("convert", "replace", "remove the database", "as a block", "as plain text"),
        ),
        "update_media_or_style": _contains(
            question,
            ("color", "style", "icon", "image", "cover", "appearance", "background", "font", "emoji"),
        ),
        "update_style_or_color": _contains(
            question,
            ("color", "style", "icon", "image", "cover", "appearance", "background", "font", "emoji"),
        ),
        "rollup_formula_schema": _contains(
            question,
            ("rollup", "formula", "computed", "calculate", "derives", "based on the"),
        ),
        "dependency_graph_editing": _contains(
            question,
            ("dependency", "parent", "sub-item", "prerequisite", "depends on", "blocked by"),
        ),
        "rewrite_existing_blocks": _contains(
            question,
            ("rewrite", "replace", "update the text", "edit the", "modify the", "change the text", "revise", "overwrite"),
        ),
        "fill_template": _contains(question, ("fill", "template", "complete the", "placeholder", "fill out")),
        "edit_nested_pages": _contains(question, ("nested", "child page", "subpage", "sub-page", "inside the page")),
        "create_columns": "column" in question.lower(),
        "render_table_from_records": _contains(question, ("table", "render a table", "in a table format")),
        "preserve_sort_order": _contains(
            question,
            ("sort", "sorted by", "order by", "ascending", "descending", "in order of"),
        ),
        "lightweight_page_edit": _contains(question, ("edit", "update", "add", "change", "modify", "rewrite", "insert")),
    }


def _references_existing_object(question: str, template_item: dict[str, Any]) -> bool:
    lowered = question.lower()
    if _contains(
        question,
        ("reference", "linked mention", "link-to-page", "link to", "mention", "the existing"),
    ):
        return True
    objects = _context_objects(template_item)
    return any(title.lower() in lowered for title in objects["page_titles"]) or any(
        name.lower() in lowered for name in objects["data_source_names"]
    )


def judge_generated_task(
    question: str,
    capabilities: list[str],
    template_item: dict[str, Any],
) -> dict[str, Any]:
    """Apply object-validity and minimum capability-coverage filtering."""
    reasons = _common_issues(question, template_item)
    predicates = _capability_predicates(question, template_item)
    covered = [capability for capability in capabilities if predicates.get(capability, True)]
    minimum = max(1, len(capabilities) // 3)
    if len(covered) < minimum:
        reasons.append(f"capability coverage too low: {len(covered)}/{len(capabilities)}")
    return {
        "accepted": not reasons,
        "covered_capabilities": covered,
        "missing_capabilities": [capability for capability in capabilities if capability not in covered],
        "hard_reject_reasons": reasons,
    }


def process_generation(
    response: str,
    capabilities: list[str],
    template_item: dict[str, Any],
) -> dict[str, Any]:
    """Parse a model response and return the normalized task plus judgment."""
    try:
        question = extract_generated_question(response)
    except (ValueError, json.JSONDecodeError) as exc:
        return {
            "accepted": False,
            "question": "",
            "raw_response": response,
            "judgment": {
                "accepted": False,
                "covered_capabilities": [],
                "missing_capabilities": capabilities,
                "hard_reject_reasons": [str(exc)],
            },
        }
    judgment = judge_generated_task(question, capabilities, template_item)
    return {
        "accepted": judgment["accepted"],
        "question": question,
        "raw_response": response,
        "judgment": judgment,
    }
