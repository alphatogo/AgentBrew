"""
Notion tool output post-processing.

Compresses Notion MCP tool responses before passing them to the LLM,
reducing token consumption while preserving task-critical structure.

Supported tools
---------------
- API-get-block-children : removes obviously low-value metadata while keeping
  near-lossless block structure
- API-post-search : keeps identity/title/schema/property summaries while
  removing verbose metadata
- API-post-database-query : keeps row identity/title/property summaries while
  removing verbose metadata
"""

from __future__ import annotations

import json
from typing import Any


_DROP_TOP_LEVEL_KEYS = {
    "object",
    "created_time",
    "last_edited_time",
    "created_by",
    "last_edited_by",
    "archived",
}


def _join_plain_text(rich_items: Any) -> str:
    """Extract concatenated plain_text from a rich text array."""
    if not isinstance(rich_items, list):
        return ""
    return "".join(
        item.get("plain_text", "")
        for item in rich_items
        if isinstance(item, dict)
    )


def _compact_parent(parent: Any) -> Any:
    """Keep only the minimal parent identity needed for structure reasoning."""
    if not isinstance(parent, dict):
        return parent
    parent_type = parent.get("type")
    compact: dict[str, Any] = {}
    if parent_type:
        compact["type"] = parent_type
    for key in (
        "page_id",
        "block_id",
        "database_id",
        "workspace",
    ):
        if key in parent:
            compact[key] = parent[key]
    return compact or parent


def _compact_value(value: Any) -> Any:
    """
    Recursively drop verbose metadata but keep semantically important content.

    Important design choice:
    - We do not truncate text.
    - We do not collapse block-type payloads into plain text.
    - Unknown structures are preserved unless they contain obviously low-value
      timestamps/editor metadata.
    """
    if isinstance(value, list):
        return [_compact_value(item) for item in value]
    if not isinstance(value, dict):
        return value

    compact: dict[str, Any] = {}
    for key, item in value.items():
        if key in _DROP_TOP_LEVEL_KEYS:
            continue
        if key == "parent":
            compact[key] = _compact_parent(item)
            continue
        compact[key] = _compact_value(item)
    return compact


def _compact_block(block: dict[str, Any]) -> dict[str, Any]:
    """Return a compact but high-fidelity representation of a Notion block."""
    compact = _compact_value(block)
    btype = compact.get("type")

    # Ensure the type-specific payload is always surfaced explicitly when present.
    if isinstance(btype, str) and btype in block and btype not in compact:
        compact[btype] = _compact_value(block.get(btype))

    return compact


def _compact_icon(icon: Any) -> Any:
    """Keep only the concise icon identity."""
    if not isinstance(icon, dict):
        return icon
    icon_type = icon.get("type")
    compact: dict[str, Any] = {}
    if icon_type:
        compact["type"] = icon_type
    if icon_type == "emoji":
        compact["emoji"] = icon.get("emoji")
    elif icon_type == "icon":
        icon_data = icon.get("icon", {})
        if isinstance(icon_data, dict):
            compact["icon"] = {
                key: icon_data.get(key)
                for key in ("name", "color")
                if key in icon_data
            }
    elif icon_type in {"external", "file"}:
        icon_data = icon.get(icon_type, {})
        if isinstance(icon_data, dict) and "url" in icon_data:
            compact["url"] = icon_data.get("url")
    return compact or icon


def _compact_option(option: Any) -> Any:
    """Keep only the human-meaningful parts of a select-like option."""
    if not isinstance(option, dict):
        return option
    return {
        key: option.get(key)
        for key in ("name", "color")
        if key in option
    }


def _compact_formula(formula: Any) -> Any:
    """Preserve only the computed formula value, not internal metadata."""
    if not isinstance(formula, dict):
        return formula
    formula_type = formula.get("type")
    if formula_type and formula_type in formula:
        return formula.get(formula_type)
    return _compact_value(formula)


def _compact_property_value(prop: Any) -> Any:
    """Compact a page property value while preserving its semantics."""
    if not isinstance(prop, dict):
        return prop

    prop_type = prop.get("type")
    compact: dict[str, Any] = {}
    if prop_type:
        compact["type"] = prop_type

    if prop_type == "title":
        compact["value"] = _join_plain_text(prop.get("title", []))
    elif prop_type == "rich_text":
        compact["value"] = _join_plain_text(prop.get("rich_text", []))
    elif prop_type in {"number", "checkbox", "url", "email", "phone_number"}:
        compact["value"] = prop.get(prop_type)
    elif prop_type == "select":
        compact["value"] = _compact_option(prop.get("select"))
    elif prop_type == "status":
        compact["value"] = _compact_option(prop.get("status"))
    elif prop_type == "multi_select":
        compact["value"] = [_compact_option(option) for option in prop.get("multi_select", [])]
    elif prop_type == "date":
        compact["value"] = prop.get("date")
    elif prop_type == "people":
        compact["count"] = len(prop.get("people", []))
    elif prop_type == "relation":
        compact["ids"] = [item.get("id") for item in prop.get("relation", []) if isinstance(item, dict)]
        compact["has_more"] = prop.get("has_more", False)
    elif prop_type == "formula":
        compact["value"] = _compact_formula(prop.get("formula"))
    elif prop_type in {"created_time", "last_edited_time"}:
        compact["value"] = prop.get(prop_type)
    elif prop_type in {"created_by", "last_edited_by"}:
        user = prop.get(prop_type, {})
        if isinstance(user, dict):
            compact["value"] = {key: user.get(key) for key in ("id", "name") if key in user}
    elif prop_type == "files":
        compact["files"] = [
            {
                key: item.get(key)
                for key in ("name", "type")
                if key in item
            }
            for item in prop.get("files", [])
            if isinstance(item, dict)
        ]
    else:
        # Preserve unsupported property types generically so we don't silently
        # discard task-relevant information.
        if prop_type and prop_type in prop:
            compact["value"] = _compact_value(prop.get(prop_type))
        else:
            compact = _compact_value(prop)

    return compact


def _compact_page_properties(properties: Any) -> Any:
    """Compact page properties into a concise, task-usable summary."""
    if not isinstance(properties, dict):
        return properties
    return {
        name: _compact_property_value(prop)
        for name, prop in properties.items()
    }


def _compact_schema_property(prop: Any) -> Any:
    """Compact a database/data source schema property definition."""
    if not isinstance(prop, dict):
        return prop

    prop_type = prop.get("type")
    compact: dict[str, Any] = {}
    if prop_type:
        compact["type"] = prop_type

    if prop_type in {"select", "multi_select", "status"}:
        config = prop.get(prop_type, {})
        if isinstance(config, dict):
            compact["options"] = [
                _compact_option(option)
                for option in config.get("options", [])
            ]
    elif prop_type == "formula":
        formula = prop.get("formula", {})
        if isinstance(formula, dict):
            compact["expression"] = formula.get("expression")
    elif prop_type == "relation":
        relation = prop.get("relation", {})
        if isinstance(relation, dict):
            compact["data_source_id"] = relation.get("data_source_id") or relation.get("database_id")
    elif prop_type == "rollup":
        rollup = prop.get("rollup", {})
        if isinstance(rollup, dict):
            compact["rollup_property_name"] = rollup.get("rollup_property_name")
            compact["relation_property_name"] = rollup.get("relation_property_name")
            compact["function"] = rollup.get("function")

    return compact or _compact_value(prop)


def _compact_schema(properties: Any) -> Any:
    """Compact database/data source schema while preserving names and types."""
    if not isinstance(properties, dict):
        return properties
    return {
        name: _compact_schema_property(prop)
        for name, prop in properties.items()
    }


def _extract_search_title(result: dict[str, Any]) -> str:
    """Extract the most useful human-readable title from a search result."""
    obj_type = result.get("object")

    if obj_type == "page":
        properties = result.get("properties", {})
        if isinstance(properties, dict):
            title_prop = properties.get("title")
            if isinstance(title_prop, dict) and title_prop.get("type") == "title":
                title = _join_plain_text(title_prop.get("title", []))
                if title:
                    return title

            for prop in properties.values():
                if isinstance(prop, dict) and prop.get("type") == "title":
                    title = _join_plain_text(prop.get("title", []))
                    if title:
                        return title

    title = _join_plain_text(result.get("title", []))
    if title:
        return title

    child_page = result.get("child_page", {})
    if isinstance(child_page, dict):
        return child_page.get("title", "")

    child_database = result.get("child_database", {})
    if isinstance(child_database, dict):
        return child_database.get("title", "")

    return ""


def _compact_search_result(result: dict[str, Any]) -> dict[str, Any]:
    """Return a compact but broadly useful representation of a search result."""
    compact: dict[str, Any] = {
        "object": result.get("object"),
        "id": result.get("id"),
        "title": _extract_search_title(result),
        "parent": _compact_parent(result.get("parent")),
        "in_trash": result.get("in_trash", False),
        "archived": result.get("archived", result.get("is_archived", False)),
    }

    if "url" in result:
        compact["url"] = result.get("url")
    if "icon" in result and result.get("icon") is not None:
        compact["icon"] = _compact_icon(result.get("icon"))

    obj_type = result.get("object")
    if obj_type == "page":
        properties = result.get("properties")
        if isinstance(properties, dict):
            compact["properties"] = _compact_page_properties(properties)
    elif obj_type in {"database", "data_source"}:
        title = _join_plain_text(result.get("title", []))
        if title:
            compact["title"] = title
        properties = result.get("properties")
        if isinstance(properties, dict):
            compact["schema"] = _compact_schema(properties)

    return compact


def _compress_search_results(raw: str) -> str:
    """
    Compress API-post-search results into a compact object/title/schema summary.

    Design goal:
    - Preserve enough information to identify pages, database entries, and
      databases/data sources across the full Notion benchmark.
    - Strip high-volume metadata that rarely matters for planning.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw

    if not isinstance(data, dict) or "results" not in data:
        return raw

    compact_results = []
    for i, result in enumerate(data.get("results", []), start=1):
        if isinstance(result, dict):
            compact_result = _compact_search_result(result)
            compact_result["_index"] = i
            compact_results.append(compact_result)
        else:
            compact_results.append(result)

    compact_response = {
        "object": "search_results_compact",
        "has_more": data.get("has_more", False),
        "next_cursor": data.get("next_cursor"),
        "results_count": len(compact_results),
        "results": compact_results,
    }
    return json.dumps(compact_response, ensure_ascii=False, separators=(",", ":"))


def _compact_database_row(result: dict[str, Any]) -> dict[str, Any]:
    """Return a compact but broadly useful representation of a database row."""
    compact: dict[str, Any] = {
        "object": result.get("object"),
        "id": result.get("id"),
        "title": _extract_search_title(result),
        "parent": _compact_parent(result.get("parent")),
        "in_trash": result.get("in_trash", False),
        "archived": result.get("archived", result.get("is_archived", False)),
    }

    if "icon" in result and result.get("icon") is not None:
        compact["icon"] = _compact_icon(result.get("icon"))

    properties = result.get("properties")
    if isinstance(properties, dict):
        compact["properties"] = _compact_page_properties(properties)

    return compact


def _compress_database_query_results(raw: str) -> str:
    """
    Compress API-post-database-query results into compact row/property summaries.

    Design goal:
    - Preserve row-level values needed for filtering, grouping, aggregation,
      copying, and verification across the Notion benchmark.
    - Remove repetitive page metadata that adds token cost but rarely affects
      task decisions.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw

    if not isinstance(data, dict) or "results" not in data:
        return raw

    compact_results = []
    for i, result in enumerate(data.get("results", []), start=1):
        if isinstance(result, dict):
            compact_result = _compact_database_row(result)
            compact_result["_index"] = i
            compact_results.append(compact_result)
        else:
            compact_results.append(result)

    compact_response = {
        "object": "database_query_compact",
        "has_more": data.get("has_more", False),
        "next_cursor": data.get("next_cursor"),
        "results_count": len(compact_results),
        "results": compact_results,
    }
    return json.dumps(compact_response, ensure_ascii=False, separators=(",", ":"))


def _compress_block_children(raw: str) -> str:
    """
    Compress the JSON response of API-get-block-children into a compact,
    near-lossless, LLM-readable format.

    Input (raw JSON string from Notion API):
        {"object":"list","results":[{...full block object...}, ...],
         "next_cursor":"...", "has_more":true, ...}

    Output:
        A minified JSON object with pagination metadata and a compacted result set.

    Fields intentionally dropped:
        object, created_time, last_edited_time, created_by, last_edited_by,
        archived

    Fields intentionally preserved:
        id, type, parent, has_children, in_trash, all block-type-specific data
        (including rich_text, annotations, href, checked state, colors, captions,
        code language, table cells, child_page titles, URLs, etc.)
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw  # not JSON — return as-is

    if not isinstance(data, dict) or "results" not in data:
        return raw

    compact_results = []
    for i, block in enumerate(data.get("results", []), start=1):
        if isinstance(block, dict):
            compact_block = _compact_block(block)
            compact_block["_index"] = i
            compact_results.append(compact_block)
        else:
            compact_results.append(block)

    compact_response = {
        "object": "block_children_compact",
        "has_more": data.get("has_more", False),
        "next_cursor": data.get("next_cursor"),
        "results_count": len(compact_results),
        "results": compact_results,
    }
    return json.dumps(compact_response, ensure_ascii=False, separators=(",", ":"))


# Map of tool-name → compressor function
_NOTION_COMPRESSORS = {
    "API-get-block-children": _compress_block_children,
    "API-post-database-query": _compress_database_query_results,
    "API-post-search": _compress_search_results,
}


def postprocess_notion_tool_output(tool_name: str, raw: str) -> str:
    """
    Entry point: compress a Notion tool response if a compressor exists.
    Falls through unchanged for unsupported tools.
    """
    compressor = _NOTION_COMPRESSORS.get(tool_name)
    if compressor is None:
        return raw
    return compressor(raw)
