"""Notion-specific trajectory rendering and retrospective-task prompt."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
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


MUTATING_TOOLS = {
    "API-patch-block-children",
    "API-post-page",
    "API-create-a-database",
    "API-patch-page",
    "API-update-a-block",
    "API-update-a-database",
    "API-delete-a-block",
    "API-create-a-comment",
}

DEFAULT_CREDIT_METHOD = "best_prefix"


_TEXT_ERROR_PREFIX = re.compile(r"^\s*(?:tool execution error|error)\b", re.IGNORECASE)


def is_error(step: dict[str, Any]) -> bool:
    """Detect Notion failures even when the sampler mislabeled ``isError``.

    Notion API failures are normally serialized into the tool ``content`` as
    an object such as ``{"status": 404, "object": "error", ...}``.  Some
    sampled trajectories incorrectly carry ``isError: false`` for those
    responses, so the environment-specific response is authoritative here.
    """
    if bool(step.get("isError", False)):
        return True

    content = step.get("content")
    parsed: Any = content
    if isinstance(content, str):
        if _TEXT_ERROR_PREFIX.match(content):
            return True
        try:
            parsed = json.loads(content)
        except (TypeError, ValueError):
            return False

    if not isinstance(parsed, dict):
        return False
    if str(parsed.get("object", "")).lower() == "error":
        return True

    status = parsed.get("status")
    try:
        return 400 <= int(status) < 600
    except (TypeError, ValueError):
        return False


def is_eligible_task(task: dict[str, Any]) -> bool:
    trajectory = task["trajectory"]
    successful_mutations = [
        step
        for step in trajectory
        if step.get("tool_name") in MUTATING_TOOLS and not is_error(step)
    ]
    return (
        task.get("final_answer") is not None
        and bool(successful_mutations)
        and task["error_ratio"] < 0.5
    )


def eligibility_failure_reason(task: dict[str, Any]) -> str:
    """Explain why a normalized Notion trajectory is excluded from LLM work."""
    if task.get("final_answer") is None:
        return "missing_final_answer"
    if task["error_ratio"] >= 0.5:
        return "error_ratio_at_least_0.5"
    successful_mutations = [
        step
        for step in task["trajectory"]
        if step.get("tool_name") in MUTATING_TOOLS and not is_error(step)
    ]
    if not successful_mutations:
        return "no_successful_mutation"
    return "environment_filter"


def _clip(value: Any, limit: int = 120) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "...[TRUNCATED]"


def _rich_text(items: Any) -> str:
    if not isinstance(items, list):
        return ""
    parts = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("plain_text"):
            parts.append(str(item["plain_text"]))
        elif isinstance(item.get("text"), dict) and item["text"].get("content"):
            parts.append(str(item["text"]["content"]))
    return _clip("".join(parts), 100)


def _title(properties: Any) -> str:
    if not isinstance(properties, dict):
        return ""
    for value in properties.values():
        if isinstance(value, dict):
            title = _rich_text(value.get("title"))
            if title:
                return title
    return ""


def _block_summary(child: dict[str, Any]) -> dict[str, Any]:
    block_type = child.get("type")
    payload = child.get(block_type, {}) if isinstance(block_type, str) else {}
    summary: dict[str, Any] = {"type": block_type}
    if isinstance(payload, dict):
        text = (
            _rich_text(payload.get("rich_text"))
            or _rich_text(payload.get("title"))
            or _rich_text(payload.get("caption"))
        )
        if text:
            summary["text"] = text
        if "checked" in payload:
            summary["checked"] = payload["checked"]
        if isinstance(payload.get("children"), list):
            summary["nested_children_count"] = len(payload["children"])
    return summary


def _sanitize(value: Any, limit: int = 120) -> Any:
    if isinstance(value, str):
        return _clip(value, limit)
    if isinstance(value, list):
        return [_sanitize(item, limit) for item in value[:8]]
    if isinstance(value, dict):
        return {key: _sanitize(item, limit) for key, item in list(value.items())[:20]}
    return value


def _simplified_summarize_arguments(step: dict[str, Any]) -> str:
    tool = step.get("tool_name", "")
    args = step.get("arguments", {}) or {}
    if tool == "API-post-search":
        summary = {
            "query": args.get("query", ""),
            "filter": args.get("filter"),
            "page_size": args.get("page_size"),
        }
    elif tool == "API-post-page":
        parent = args.get("parent", {}) or {}
        summary = {
            "parent_type": parent.get("type"),
            "parent_page_id": parent.get("page_id"),
            "parent_database_id": parent.get("database_id"),
            "title": _title(args.get("properties")),
        }
    elif tool == "API-create-a-database":
        summary = {
            "parent": _sanitize(args.get("parent", {})),
            "title": _rich_text(args.get("title")),
            "property_names": list((args.get("properties") or {}).keys())[:20],
        }
    elif tool == "API-patch-block-children":
        children = args.get("children", []) or []
        summary = {
            "block_id": args.get("block_id"),
            "after": args.get("after"),
            "child_count": len(children),
            "child_previews": [
                _block_summary(child) for child in children[:20] if isinstance(child, dict)
            ],
        }
    elif tool == "API-patch-page":
        summary = {
            "page_id": args.get("page_id"),
            "property_names": list((args.get("properties") or {}).keys())[:20],
            "icon": bool(args.get("icon")),
            "cover": bool(args.get("cover")),
            "archived": args.get("archived"),
        }
    elif tool == "API-update-a-database":
        summary = {
            "database_id": args.get("database_id"),
            "title_present": bool(args.get("title")),
            "description_present": bool(args.get("description")),
            "property_names": list((args.get("properties") or {}).keys())[:20],
        }
    else:
        summary = _sanitize(args)
    return json.dumps(summary, ensure_ascii=False)


def _simplified_summarize_feedback(step: dict[str, Any]) -> str:
    content = str(step.get("content") or "")
    try:
        parsed = json.loads(content)
    except Exception:
        return _clip(content, 500)
    if not isinstance(parsed, dict):
        return _clip(parsed, 500)
    if parsed.get("object") == "list":
        results = parsed.get("results") or []
        previews = []
        for result in results[:8]:
            if not isinstance(result, dict):
                continue
            previews.append(
                {
                    "object": result.get("object"),
                    "id": result.get("id"),
                    "name": _title(result.get("properties")) or _rich_text(result.get("title")),
                    "parent": result.get("parent"),
                }
            )
        return json.dumps(
            {
                "object": "list",
                "result_count": len(results),
                "has_more": parsed.get("has_more"),
                "type": parsed.get("type"),
                "results_preview": previews,
            },
            ensure_ascii=False,
        )
    return json.dumps(
        {
            "object": parsed.get("object"),
            "id": parsed.get("id"),
            "name": _title(parsed.get("properties")) or _rich_text(parsed.get("title")),
            "type": parsed.get("type"),
            "status": parsed.get("status"),
            "code": parsed.get("code"),
            "message": _clip(parsed.get("message", ""), 180),
            "parent": parsed.get("parent"),
        },
        ensure_ascii=False,
    )


def _simplified_format_trajectory(trajectory: list[dict[str, Any]]) -> str:
    lines = []
    for index, step in enumerate(trajectory):
        lines.extend(
            [
                f"[Step {index}] Tool: {step.get('tool_name')}",
                f"Args Summary: {summarize_arguments(step)}",
                f"Feedback Summary (Error: {is_error(step)}): {summarize_feedback(step)}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def _simplified_summarize_changes(trajectory: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    database_entries: dict[str, list[str]] = defaultdict(list)
    for step in trajectory:
        if is_error(step) or step.get("tool_name") not in MUTATING_TOOLS:
            continue
        tool = step.get("tool_name")
        args = step.get("arguments", {}) or {}
        if tool == "API-post-page":
            title = _title(args.get("properties")) or "[unknown]"
            parent = args.get("parent", {}) or {}
            if parent.get("database_id"):
                database_entries[str(parent["database_id"])].append(title)
            else:
                lines.append(f"- created page: {title}")
        elif tool == "API-create-a-database":
            lines.append(
                f"- created database: {_rich_text(args.get('title')) or '[unknown]'} "
                f"properties={list((args.get('properties') or {}).keys())[:10]}"
            )
        elif tool == "API-patch-block-children":
            children = [
                _block_summary(child)
                for child in (args.get("children") or [])
                if isinstance(child, dict)
            ]
            counts = Counter(item.get("type") for item in children)
            headings = [item.get("text") for item in children if str(item.get("type", "")).startswith("heading")]
            lines.append(
                f"- added page structure: block_id={args.get('block_id')}; "
                f"block_types={dict(counts)}; headings={headings[:6]}"
            )
        elif tool == "API-patch-page":
            lines.append(
                f"- updated page properties: {list((args.get('properties') or {}).keys())[:10]}"
            )
        elif tool == "API-update-a-database":
            lines.append(
                f"- updated database schema/properties: "
                f"{list((args.get('properties') or {}).keys())[:10]}"
            )
        elif tool == "API-delete-a-block":
            lines.append("- removed or updated existing block")
        elif tool == "API-create-a-comment":
            lines.append("- created comment")
    for database_id, titles in database_entries.items():
        lines.append(f"- created database entries in {database_id}: {titles[:12]}")
    return "\n".join(lines) if lines else "- No successful workspace changes detected."


# The functions below intentionally preserve the legacy Notion hindsight
# renderer byte-for-byte. Credit is computed from token NLL, so seemingly
# harmless changes to truncation, selected fields, or step numbering alter the
# prompt and therefore the resulting credit values.


def _legacy_truncate_middle(text: str, max_len: int = 800) -> str:
    text = text.replace("\n", " ")
    if len(text) <= max_len:
        return text
    keep_each = max_len // 2
    return text[:keep_each] + "\n...[MIDDLE TRUNCATED]...\n" + text[-keep_each:]


def _legacy_clip_text(text: str, max_len: int = 80) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "...[TRUNCATED]"


def _legacy_sanitize(value: Any, max_str_len: int = 120) -> Any:
    if isinstance(value, str):
        compact = value.replace("\n", " ")
        return _legacy_clip_text(compact, max_len=max_str_len)
    if isinstance(value, list):
        return [_legacy_sanitize(item, max_str_len=max_str_len) for item in value[:8]]
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            if key in {"plain_text", "content", "expression", "url"} and isinstance(item, str):
                sanitized[key] = _legacy_clip_text(item.replace("\n", " "), max_len=max_str_len)
            else:
                sanitized[key] = _legacy_sanitize(item, max_str_len=max_str_len)
        return sanitized
    return value


def _legacy_title(properties: dict[str, Any]) -> str:
    for value in properties.values():
        if not isinstance(value, dict):
            continue
        title_items = value.get("title")
        if not isinstance(title_items, list):
            continue
        pieces = []
        for item in title_items:
            if isinstance(item, dict):
                plain_text = item.get("plain_text")
                if plain_text:
                    pieces.append(str(plain_text))
                    continue
                text = item.get("text")
                if isinstance(text, dict) and text.get("content"):
                    pieces.append(str(text["content"]))
        title = "".join(pieces).strip()
        if title:
            return _legacy_clip_text(title, max_len=100)
    return ""


def _legacy_rich_text(items: Any) -> str:
    if not isinstance(items, list):
        return ""
    pieces: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        plain_text = item.get("plain_text")
        if plain_text:
            pieces.append(str(plain_text))
            continue
        text = item.get("text")
        if isinstance(text, dict) and text.get("content"):
            pieces.append(str(text["content"]))
    return _legacy_clip_text("".join(pieces).strip(), max_len=100)


def _legacy_block_payload(child: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    if not isinstance(child, dict):
        return None, None
    block_type = child.get("type")
    if isinstance(block_type, str) and block_type in child and isinstance(child.get(block_type), dict):
        return block_type, child.get(block_type)
    for key, value in child.items():
        if key in {"object", "id", "type", "has_children", "archived", "in_trash"}:
            continue
        if isinstance(value, dict):
            return str(block_type or key), value
    return (str(block_type) if block_type else None), None


def _legacy_block_summary(child: dict[str, Any]) -> dict[str, Any]:
    block_type, payload = _legacy_block_payload(child)
    summary: dict[str, Any] = {"type": block_type}
    if not isinstance(payload, dict):
        return summary
    text = ""
    for key in ("rich_text", "title", "caption"):
        text = _legacy_rich_text(payload.get(key))
        if text:
            break
    if not text and isinstance(payload.get("children"), list) and payload["children"]:
        first_child = payload["children"][0]
        if isinstance(first_child, dict):
            child_type, child_payload = _legacy_block_payload(first_child)
            if isinstance(child_payload, dict):
                text = _legacy_rich_text(child_payload.get("rich_text"))
            if child_type and not summary.get("nested_type"):
                summary["nested_type"] = child_type
    if text:
        summary["text"] = text
    if "checked" in payload:
        summary["checked"] = payload.get("checked")
    if "color" in payload and payload.get("color") not in {None, "default"}:
        summary["color"] = payload.get("color")
    if "icon" in payload and isinstance(payload.get("icon"), dict):
        summary["icon_type"] = payload["icon"].get("type")
    if "children" in payload and isinstance(payload.get("children"), list):
        summary["nested_children_count"] = len(payload["children"])
    return summary


def _legacy_block_structure(children: list[dict[str, Any]]) -> dict[str, Any]:
    headings = [
        item.get("text") for item in children
        if isinstance(item.get("type"), str)
        and item.get("type", "").startswith("heading") and item.get("text")
    ][:4]
    paragraphs = [
        item.get("text") for item in children
        if item.get("type") == "paragraph" and item.get("text")
    ][:4]
    block_types = [item.get("type") for item in children if item.get("type")]
    structure: dict[str, Any] = {}
    if headings:
        structure["headings"] = headings
    if paragraphs:
        structure["paragraph_examples"] = paragraphs
    if block_types:
        structure["block_type_examples"] = block_types[:8]
    counts: dict[str, int] = defaultdict(int)
    for item in children:
        block_type = item.get("type")
        if isinstance(block_type, str) and block_type:
            counts[block_type] += 1
    interesting_counts = {
        key: counts[key]
        for key in (
            "paragraph", "to_do", "bulleted_list_item", "numbered_list_item",
            "divider", "link_to_page", "child_database", "table", "callout", "toggle",
        )
        if counts.get(key)
    }
    if interesting_counts:
        structure["counts"] = interesting_counts
    return structure


def _legacy_display_name(obj: dict[str, Any]) -> str:
    direct_title = _legacy_rich_text(obj.get("title"))
    if direct_title:
        return direct_title
    properties = obj.get("properties")
    if isinstance(properties, dict):
        title = _legacy_title(properties)
        if title:
            return title
    for key in ("name", "plain_text"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return _legacy_clip_text(value.strip(), max_len=100)
    return ""


def _legacy_parent(parent: Any) -> Any:
    if not isinstance(parent, dict):
        return parent
    summary: dict[str, Any] = {"type": parent.get("type")}
    for key in ("page_id", "database_id", "block_id", "workspace"):
        if key in parent:
            summary[key] = parent.get(key)
    return summary


def _legacy_result_object(obj: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"object": obj.get("object"), "id": obj.get("id")}
    display_name = _legacy_display_name(obj)
    if display_name:
        summary["name"] = display_name
    if "parent" in obj:
        summary["parent"] = _legacy_parent(obj.get("parent"))
    if obj.get("object") == "block":
        block_summary = _legacy_block_summary(obj)
        if block_summary.get("type"):
            summary["type"] = block_summary.get("type")
        if block_summary.get("text"):
            summary["text"] = block_summary.get("text")
        if "checked" in block_summary:
            summary["checked"] = block_summary.get("checked")
        if block_summary.get("nested_children_count"):
            summary["nested_children_count"] = block_summary.get("nested_children_count")
    if obj.get("object") == "database":
        properties = obj.get("properties")
        if isinstance(properties, dict):
            summary["property_names"] = list(properties.keys())[:12]
    elif obj.get("object") == "page":
        properties = obj.get("properties")
        if isinstance(properties, dict):
            property_names = list(properties.keys())[:8]
            if property_names:
                summary["property_names"] = property_names
    if obj.get("url"):
        summary["url"] = obj.get("url")
    return summary


def _legacy_result_from_content(content: str) -> dict[str, Any] | None:
    if not content:
        return None
    try:
        parsed = json.loads(content)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _legacy_result_name(content: str) -> str:
    parsed = _legacy_result_from_content(content)
    return _legacy_display_name(parsed) if parsed else ""


def summarize_arguments(step: dict[str, Any]) -> str:
    tool_name = step.get("tool_name", "")
    args = step.get("arguments", {}) or {}
    if tool_name == "API-post-search":
        return json.dumps({"query": args.get("query", ""), "filter": args.get("filter"), "page_size": args.get("page_size")}, ensure_ascii=False)
    if tool_name == "API-post-page":
        parent = args.get("parent", {})
        return json.dumps({"parent_type": parent.get("type"), "parent_page_id": parent.get("page_id"), "parent_database_id": parent.get("database_id"), "title": _legacy_title(args.get("properties", {}) or {})}, ensure_ascii=False)
    if tool_name == "API-create-a-database":
        title_parts = []
        for item in args.get("title", []) or []:
            if isinstance(item, dict) and isinstance(item.get("text"), dict) and item["text"].get("content"):
                title_parts.append(str(item["text"]["content"]))
        return json.dumps({"parent": _legacy_sanitize(args.get("parent", {})), "title": _legacy_clip_text("".join(title_parts).strip(), max_len=100), "property_names": list((args.get("properties") or {}).keys())[:20]}, ensure_ascii=False)
    if tool_name == "API-patch-page":
        return json.dumps({"page_id": args.get("page_id"), "property_names": list((args.get("properties") or {}).keys())[:20], "icon": bool(args.get("icon")), "cover": bool(args.get("cover")), "archived": args.get("archived")}, ensure_ascii=False)
    if tool_name == "API-patch-block-children":
        children = args.get("children", []) or []
        previews = [_legacy_block_summary(child) for child in children[:20] if isinstance(child, dict)]
        return json.dumps({"block_id": args.get("block_id"), "after": args.get("after"), "child_count": len(children), "child_types": [item.get("type") for item in previews], "child_previews": previews}, ensure_ascii=False)
    if tool_name == "API-update-a-block":
        block = args.get("block", {})
        return json.dumps({"block_id": args.get("block_id"), "block_type": block.get("type") if isinstance(block, dict) else None, "archived": args.get("archived"), "in_trash": args.get("in_trash")}, ensure_ascii=False)
    if tool_name == "API-update-a-database":
        return json.dumps({"database_id": args.get("database_id"), "title_present": bool(args.get("title")), "description_present": bool(args.get("description")), "property_names": list((args.get("properties") or {}).keys())[:20]}, ensure_ascii=False)
    if tool_name == "API-delete-a-block":
        return json.dumps({"block_id": args.get("block_id")}, ensure_ascii=False)
    if tool_name == "API-create-a-comment":
        return json.dumps({"parent": _legacy_sanitize(args.get("parent", {})), "discussion_id": args.get("discussion_id")}, ensure_ascii=False)
    return json.dumps(_legacy_sanitize(args), ensure_ascii=False)


def summarize_feedback(step: dict[str, Any], max_len: int = 500) -> str:
    content = str(step.get("content") or "")
    if not content:
        return ""
    try:
        parsed = json.loads(content)
    except Exception:
        return _legacy_truncate_middle(content, max_len=max_len)
    if isinstance(parsed, dict):
        if parsed.get("object") == "list":
            previews = [_legacy_result_object(obj) for obj in (parsed.get("results") or [])[:8] if isinstance(obj, dict)]
            return json.dumps({"object": "list", "result_count": len(parsed.get("results") or []), "has_more": parsed.get("has_more"), "type": parsed.get("type"), "results_preview": previews}, ensure_ascii=False)
        if parsed.get("object") == "error" or "status" in parsed:
            return json.dumps({"status": parsed.get("status"), "object": parsed.get("object"), "code": parsed.get("code"), "message": _legacy_clip_text(str(parsed.get("message", "")), max_len=180)}, ensure_ascii=False)
        summary = _legacy_result_object(parsed)
        if "type" in parsed:
            summary["type"] = parsed.get("type")
        return json.dumps(summary, ensure_ascii=False)
    return _legacy_truncate_middle(content, max_len=max_len)


def format_trajectory(trajectory: list[dict[str, Any]]) -> str:
    formatted: list[str] = []
    for idx, step in enumerate(trajectory):
        formatted.append(f"[Step {idx}] Tool: {step.get('tool_name')}")
        formatted.append(f"Args Summary: {summarize_arguments(step)}")
        formatted.append(f"Feedback Summary (Error: {bool(step.get('isError', False))}): {summarize_feedback(step)}")
        formatted.append("")
    return "\n".join(formatted).strip()


def summarize_changes(trajectory: list[dict[str, Any]]) -> str:
    created_databases: list[str] = []
    created_pages: list[str] = []
    deleted_blocks: list[str] = []
    updated_pages: list[str] = []
    updated_databases: list[str] = []
    comments_created = 0
    database_entry_creations: dict[str, list[str]] = defaultdict(list)
    page_block_changes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    retrieved_block_children: dict[str, list[dict[str, Any]]] = {}

    for step in trajectory:
        if step.get("tool_name") != "API-get-block-children" or step.get("isError", False):
            continue
        block_id = str((step.get("arguments", {}) or {}).get("block_id") or "")
        if not block_id:
            continue
        try:
            parsed = json.loads(str(step.get("content") or ""))
        except Exception:
            continue
        summaries = [_legacy_block_summary(item) for item in (parsed.get("results") or [])[:40] if isinstance(item, dict)]
        if summaries:
            retrieved_block_children[block_id] = summaries

    for step in trajectory:
        if step.get("tool_name") not in MUTATING_TOOLS or step.get("isError", False):
            continue
        args = step.get("arguments", {}) or {}
        tool_name = step.get("tool_name")
        if tool_name == "API-post-page":
            title = _legacy_title(args.get("properties", {}) or {}) or _legacy_result_name(str(step.get("content") or ""))
            parent = args.get("parent", {}) or {}
            if isinstance(parent, dict) and parent.get("database_id"):
                database_entry_creations[str(parent.get("database_id"))].append(title or "[unknown]")
            else:
                created_pages.append(title or "[unknown]")
        elif tool_name == "API-create-a-database":
            title_parts = [str(item["text"]["content"]) for item in args.get("title", []) or [] if isinstance(item, dict) and isinstance(item.get("text"), dict) and item["text"].get("content")]
            created_databases.append(f"{''.join(title_parts).strip() or '[unknown]'} properties={list((args.get('properties') or {}).keys())[:10]}")
        elif tool_name == "API-patch-page":
            updated_pages.append(str(list((args.get("properties") or {}).keys())[:10]))
        elif tool_name == "API-patch-block-children":
            children = args.get("children", []) or []
            summaries = [_legacy_block_summary(child) for child in children[:20] if isinstance(child, dict)]
            block_id = str(args.get("block_id") or "[unknown]")
            if summaries:
                metadata: dict[str, Any] = {"type": "__write_metadata__", "write_mode": "insert_after" if args.get("after") else "append_or_prepend_unspecified"}
                if args.get("after"):
                    metadata["after"] = args.get("after")
                page_block_changes[block_id].append(metadata)
                page_block_changes[block_id].extend(summaries)
        elif tool_name == "API-update-a-block":
            block = args.get("block", {}) if isinstance(args.get("block"), dict) else {}
            deleted_blocks.append(f"update block type={block.get('type')}")
        elif tool_name == "API-update-a-database":
            updated_databases.append(str(list((args.get("properties") or {}).keys())[:10]))
        elif tool_name == "API-delete-a-block":
            deleted_obj = _legacy_result_from_content(str(step.get("content") or ""))
            deleted_blocks.append(deleted_obj.get("type") if deleted_obj else "block")
        elif tool_name == "API-create-a-comment":
            comments_created += 1

    lines: list[str] = []
    lines.extend(f"- created database: {item}" for item in created_databases)
    lines.extend(f"- created page: {title}" for title in created_pages)
    for _, entry_titles in database_entry_creations.items():
        preview = [title for title in entry_titles if title][:3]
        lines.append(f"- populated a created database with {len(entry_titles)} entry pages" + (f" (examples: {preview})" if preview else ""))
    for block_id, block_summaries in page_block_changes.items():
        metadata = [item for item in block_summaries if item.get("type") == "__write_metadata__"]
        content_blocks = [item for item in block_summaries if item.get("type") != "__write_metadata__"]
        source_context = _legacy_block_structure(retrieved_block_children.get(block_id, []))
        headings = [item.get("text") for item in block_summaries if item.get("type", "").startswith("heading")]
        bullet_count = sum(item.get("type") == "bulleted_list_item" for item in block_summaries)
        numbered_count = sum(item.get("type") == "numbered_list_item" for item in block_summaries)
        todo_count = sum(item.get("type") == "to_do" for item in block_summaries)
        embed_count = sum(item.get("type") == "embed" for item in block_summaries)
        paragraphs = [item.get("text") for item in block_summaries if item.get("type") == "paragraph" and item.get("text")][:3]
        parts: list[str] = []
        if metadata and metadata[0].get("write_mode"):
            parts.append(f"write_mode={metadata[0].get('write_mode')}")
        if headings:
            parts.append(f"headings={headings[:6]}")
        if bullet_count:
            parts.append(f"bulleted_items={bullet_count}")
        if numbered_count:
            parts.append(f"numbered_items={numbered_count}")
        if todo_count:
            parts.append(f"to_dos={todo_count}")
        if embed_count:
            parts.append(f"embeds={embed_count}")
        if paragraphs:
            parts.append(f"paragraph_examples={paragraphs}")
        if source_context:
            parts.append(f"existing_container_context={source_context}")
        if not parts:
            parts.append(f"block_count={len(content_blocks)}")
        lines.append(f"- added page structure: {'; '.join(parts)}")
    lines.extend(f"- removed or updated existing block: {item}" for item in deleted_blocks)
    if updated_pages:
        lines.append(f"- updated page properties: {updated_pages[:3]}")
    if updated_databases:
        lines.append(f"- updated database schema/properties: {updated_databases[:3]}")
    if comments_created:
        lines.append(f"- created comments: {comments_created}")
    return "\n".join(lines) if lines else "- No successful workspace changes detected."


def build_task_inference_prompt(
    original_task: str,
    trajectory_str: str,
    change_summary: str,
) -> str:
    return f"""You revise benchmark-style Notion tasks from agent trajectories.

Your job is to minimally edit the ORIGINAL TASK so it matches what the agent actually completed.

Main goal:
- Keep the original task wording and structure as much as possible
- Only change the parts describing completed page modifications
- Remove or weaken unsupported subtasks
- Do not rewrite the whole task
- Do not turn query/retrieve steps into detailed task instructions

Use the trajectory as evidence, with priority on:
1. successful workspace changes
2. query/retrieve steps only when needed to identify the page, database, or content used in a successful write

Rules:
- Write the task in English.
- Begin exactly with: `Please use Notion tools to finish the following task:`
- Output only the revised task text.
- Preserve original wording wherever possible.
- If a part of the original task is unsupported, delete it or weaken it.
- Focus on user-visible end state: page created, section added, headings added, list added, to-dos added, content removed, content reformatted.
- Do not add process details, tool behavior, or explanations.
- Do not add checkmarks, comments, or status markers.
- Do not preserve strong claims like `all`, `each`, `exact count`, `verbatim`, or exact field reuse unless directly supported by successful writes.
- Do not change the main task type if the trajectory supports the original one.

Example 1

ORIGINAL TASK:
Inside the page "Reading Hub", create a new child page titled "Weekly Reading Summary". The task must:
1. Query the Books database to retrieve all rows
2. Add a heading_1 block titled "📚 Weekly Reading Summary"
3. For each book row, add a bullet showing [Title] — [Status]
4. Add a heading_2 block titled "Next Reads"
5. Add 3 to-do items for unread books
6. Include a heading_3 block showing the exact total number of books

OBSERVED CHANGES:
- created page: Weekly Reading Summary
- added heading_1: 📚 Weekly Reading Summary
- added multiple bullet list items
- added heading_2: Next Reads
- added 3 to-do items

REVISED TASK:
Inside the page "Reading Hub", create a new child page titled "Weekly Reading Summary". The task must:
1. Add a heading_1 block titled "📚 Weekly Reading Summary"
2. Add bullet list items showing book titles and statuses
3. Add a heading_2 block titled "Next Reads"
4. Add 3 to-do items for unread books

Example 2

ORIGINAL TASK:
Go to the page "Team Onboarding" and update it with the following:
1. Remove the old checklist database
2. Add a new section called "First Week Plan"
3. Insert a bullet list with onboarding steps
4. Reformat the existing resources section into a two-column layout
5. Replace all icons with blue circle icons

OBSERVED CHANGES:
- added section: First Week Plan
- added bullet list with onboarding steps

REVISED TASK:
Go to the page "Team Onboarding" and update it with the following:
1. Add a new section called "First Week Plan"
2. Insert a bullet list with onboarding steps

Now revise the following task.

ORIGINAL TASK:
{original_task}

OBSERVED CHANGES:
{change_summary}

TRAJECTORY:
{trajectory_str}

REVISED TASK:
"""


def normalize_task_inference_output(text: str) -> str:
    text = text.strip()
    if "REQUEST:" in text:
        text = text.split("REQUEST:", 1)[1].strip()
    for prefix in ("User Request:", "Request:", "Instruction:", "Simulated User Request:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    return text


def build_credit_prompt_prefix(trajectory_str: str) -> str:
    if trajectory_str:
        return f"""An AI agent is solving a Notion workspace task with tool calls.
Based on the tool actions observed so far, what was the user's original Notion request?

Actions observed:
{trajectory_str}

The user's original request was:
"""
    return """An AI agent is about to solve a Notion workspace task with tool calls.
Without seeing any actions, what was the user's original Notion request?

The user's original request was:
"""
